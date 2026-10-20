# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Interactive 3D Point Cloud Viewer using Viser.

This module provides the PointCloudViewer class for visualizing 3D reconstruction results,
including point clouds, camera poses, and animated playback.
"""

import os
import time
import threading
import subprocess
import tempfile
import shutil
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import cv2
import matplotlib.cm as cm
from tqdm.auto import tqdm


from lingbot_map.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from lingbot_map.vis.utils import CameraState
from lingbot_map.vis.sky_segmentation import apply_sky_segmentation


class PointCloudViewer:
    """
    Interactive 3D point cloud viewer with camera visualization.

    Features:
    - Point cloud visualization with confidence-based filtering
    - Camera frustum visualization with gradient colors
    - Frame-by-frame playback animation (3D/4D modes)
    - Range-based and recent-N-frames visualization modes
    - Video export with FFmpeg

    Args:
        model: Optional model for interactive inference
        state_args: Optional state arguments
        pc_list: List of point clouds per frame
        color_list: List of colors per frame
        conf_list: List of confidence scores per frame
        cam_dict: Camera dictionary with focal, pp, R, t
        image_mask: Optional image mask
        edge_color_list: Optional edge colors
        device: Device for computation
        port: Viser server port
        show_camera: Whether to show camera frustums
        vis_threshold: Visibility threshold for filtering
        size: Image size
        downsample_factor: Point cloud downsample factor
        point_size: Initial point size
        pred_dict: Prediction dictionary (alternative to pc_list/color_list/conf_list)
        init_conf_threshold: Initial confidence threshold percentage
        use_point_map: Use point map instead of depth-based points
        mask_sky: Apply sky segmentation
        image_folder: Path to image folder (for sky segmentation)
    """

    def __init__(
        self,
        model=None,
        state_args=None,
        pc_list=None,
        color_list=None,
        conf_list=None,
        cam_dict=None,
        image_mask=None,
        edge_color_list=None,
        device: str = "cpu",
        port: int = 8080,
        show_camera: bool = True,
        vis_threshold: float = 1.0,
        size: int = 512,
        downsample_factor: int = 10,
        point_size: float = 0.00001,
        pred_dict: Optional[Dict] = None,
        init_conf_threshold: float = 50.0,
        use_point_map: bool = False,
        mask_sky: bool = False,
        image_folder: Optional[str] = None,
        sky_mask_dir: Optional[str] = None,
        sky_mask_visualization_dir: Optional[str] = None,
        depth_stride: int = 1,
    ):
        self.model = model
        self.size = size
        self.state_args = state_args
        self.device = device
        self.conf_list = conf_list
        self.vis_threshold = vis_threshold
        self.point_size = point_size
        self.tt = lambda x: torch.from_numpy(x).float().to(device)

        # Process the prediction dictionary to create pc_list, color_list, conf_list
        if pred_dict is not None:
            pc_list, color_list, conf_list, cam_dict = self._process_pred_dict(
                pred_dict, use_point_map, mask_sky, image_folder,
                sky_mask_dir=sky_mask_dir,
                sky_mask_visualization_dir=sky_mask_visualization_dir,
                depth_stride=depth_stride,
            )
        else:
            self.original_images = []
            if color_list is not None:
                for img in color_list:
                    if len(img) > 0:
                        # Convert float color (0-1) or uint8 color to uint8 image
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8)
                        self.original_images.append(img)

        self.pcs, self.all_steps = self.read_data(
            pc_list, color_list, conf_list, edge_color_list
        )
        self.cam_dict = cam_dict
        self.num_frames = len(self.all_steps)
        
        # Initialize 2D Bird's-Eye View map limits
        if self.cam_dict is not None and "t" in self.cam_dict and len(self.all_steps) > 0:
            cam_positions = np.array([self.cam_dict["t"][s] for s in self.all_steps])
            x_coords = cam_positions[:, 0]
            z_coords = cam_positions[:, 2]
            x_center = (np.min(x_coords) + np.max(x_coords)) / 2.0
            z_center = (np.min(z_coords) + np.max(z_coords)) / 2.0
            max_range = max(np.ptp(x_coords), np.ptp(z_coords)) or 1.0
            padding = max_range * 0.75
            self.bev_x_min = x_center - padding
            self.bev_x_max = x_center + padding
            self.bev_z_min = z_center - padding
            self.bev_z_max = z_center + padding
        else:
            self.bev_x_min, self.bev_x_max = -5.0, 5.0
            self.bev_z_min, self.bev_z_max = -5.0, 5.0

        self.image_mask = image_mask
        self.show_camera = show_camera
        self.on_replay = False
        self.vis_pts_list = []
        self.traj_list = []
        self.orig_img_list = [x[0] for x in color_list if len(x) > 0] if color_list else []
        self.via_points = []


    def _process_pred_dict(
        self,
        pred_dict: Dict,
        use_point_map: bool,
        mask_sky: bool,
        image_folder: Optional[str],
        sky_mask_dir: Optional[str] = None,
        sky_mask_visualization_dir: Optional[str] = None,
        depth_stride: int = 1,
    ) -> Tuple[List, List, List, Dict]:
        """Process prediction dictionary to extract visualization data.

        Args:
            pred_dict: Model prediction dictionary.
            use_point_map: Use point map instead of depth-based projection.
            mask_sky: Apply sky segmentation to filter sky points.
            image_folder: Path to images for sky segmentation.
            sky_mask_dir: Directory for cached sky masks.
            sky_mask_visualization_dir: Directory for sky mask visualization images.
            depth_stride: Only project depth to point cloud every N frames.
                Frames not projected will have empty point clouds but still
                show camera frustums and images. 1 = every frame (default).
        """
        images = pred_dict["images"]  # (S, 3, H, W)

        depth_map = pred_dict.get("depth")  # (S, H, W, 1)
        depth_conf = pred_dict.get("depth_conf")  # (S, H, W)

        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

        # Compute world points from depth if not using the precomputed point map
        if not use_point_map:
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
            conf = depth_conf
        else:
            world_points = pred_dict["world_points"]  # (S, H, W, 3)
            conf = pred_dict.get("world_points_conf", depth_conf)  # (S, H, W)

        # Apply sky segmentation if enabled
        if mask_sky:
            conf = apply_sky_segmentation(
                conf, image_folder=image_folder, images=images,
                sky_mask_dir=sky_mask_dir,
                sky_mask_visualization_dir=sky_mask_visualization_dir,
            )

        # Convert images from (S, 3, H, W) to (S, H, W, 3)
        colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
        S = world_points.shape[0]

        # Store original images for camera frustum display
        self.original_images = []
        for i in range(S):
            img = images[i]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            self.original_images.append(img)

        # Create lists - apply depth_stride to skip frames for point projection
        H, W = world_points.shape[1], world_points.shape[2]
        pc_list = []
        color_list = []
        conf_list = []
        skipped = 0
        for i in range(S):
            if depth_stride > 1 and i % depth_stride != 0:
                # Empty point cloud for skipped frames
                pc_list.append(np.zeros((0, 0, 3), dtype=np.float32))
                color_list.append(np.zeros((0, 0, 3), dtype=np.float32))
                conf_list.append(np.zeros((0, 0), dtype=np.float32))
                skipped += 1
            else:
                pc_list.append(world_points[i])
                color_list.append(colors[i])
                if conf is not None:
                    conf_list.append(conf[i])
                else:
                    conf_list.append(np.ones(world_points[i].shape[:2], dtype=np.float32))

        if depth_stride > 1:
            print(f'  depth_stride={depth_stride}: projecting {S - skipped}/{S} frames, skipping {skipped}')

        # Create camera dictionary (all frames keep cameras)
        cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
        cam_dict = {
            "focal": [intrinsics_cam[i, 0, 0] for i in range(S)],
            "pp": [(intrinsics_cam[i, 0, 2], intrinsics_cam[i, 1, 2]) for i in range(S)],
            "R": [cam_to_world_mat[i, :3, :3] for i in range(S)],
            "t": [cam_to_world_mat[i, :3, 3] for i in range(S)],
        }

        return pc_list, color_list, conf_list, cam_dict

    def _compute_scene_center_and_scale(self) -> Tuple[np.ndarray, float]:
        """Compute scene center and scale from camera positions and point clouds.

        Returns:
            Tuple of (center as 3D array, scale as float distance).
        """
        # Use camera positions as primary reference (more reliable than noisy points)
        if self.cam_dict is not None and "t" in self.cam_dict:
            cam_positions = np.array([self.cam_dict["t"][s] for s in self.all_steps])
            center = np.mean(cam_positions, axis=0)
            if len(cam_positions) > 1:
                extent = np.ptp(cam_positions, axis=0)  # range per axis
                scale = np.linalg.norm(extent)
            else:
                scale = 1.0
        else:
            # Fallback: use point cloud data
            all_pts = []
            for step in self.all_steps:
                pc = self.pcs[step]["pc"].reshape(-1, 3)
                # subsample for speed
                if len(pc) > 1000:
                    pc = pc[::len(pc) // 1000]
                all_pts.append(pc)
            all_pts = np.concatenate(all_pts, axis=0)
            center = np.median(all_pts, axis=0)
            extent = np.percentile(all_pts, 95, axis=0) - np.percentile(all_pts, 5, axis=0)
            scale = np.linalg.norm(extent)

        return center, max(scale, 0.1)












    @staticmethod
    def generate_pseudo_intrinsics(h: int, w: int) -> np.ndarray:
        """Generate pseudo intrinsics from image size."""
        focal = (h**2 + w**2) ** 0.5
        return np.array([[focal, 0, w // 2], [0, focal, h // 2], [0, 0, 1]]).astype(np.float32)


    @staticmethod
    def set_color_border(image, border_width=5, color=[1, 0, 0]):
        """Add colored border to image."""
        image[:border_width, :, 0] = color[0]
        image[:border_width, :, 1] = color[1]
        image[:border_width, :, 2] = color[2]
        image[-border_width:, :, 0] = color[0]
        image[-border_width:, :, 1] = color[1]
        image[-border_width:, :, 2] = color[2]
        image[:, :border_width, 0] = color[0]
        image[:, :border_width, 1] = color[1]
        image[:, :border_width, 2] = color[2]
        image[:, -border_width:, 0] = color[0]
        image[:, -border_width:, 1] = color[1]
        image[:, -border_width:, 2] = color[2]
        return image

    def read_data(self, pc_list, color_list, conf_list, edge_color_list=None):
        """Read and organize point cloud data."""
        pcs = {}
        step_list = []
        if pc_list is None:
            return pcs, step_list
        for i, pc in enumerate(pc_list):
            step = i
            pcs.update({
                step: {
                    "pc": pc,
                    "color": color_list[i],
                    "conf": conf_list[i],
                    "edge_color": (
                        None if edge_color_list is None or edge_color_list[i] is None
                        else edge_color_list[i]
                    ),
                }
            })
            step_list.append(step)

        # Generate camera gradient colors
        num_cameras = len(pc_list)
        if num_cameras > 1:
            normalized_indices = np.array(list(range(num_cameras))) / (num_cameras - 1)
        else:
            normalized_indices = np.array([0.0])
        cmap = cm.get_cmap('viridis')
        self.camera_colors = cmap(normalized_indices)
        return pcs, step_list

    def parse_pc_data(
        self,
        pc,
        color,
        conf=None,
        edge_color=[0.251, 0.702, 0.902],
        set_border_color=False,
        downsample_factor=1,
    ):
        """Parse and filter point cloud data."""
        pred_pts = pc.reshape(-1, 3)

        if set_border_color and edge_color is not None:
            color = self.set_color_border(color[0], color=edge_color)
        if np.isnan(color).any():
            color = np.zeros((pred_pts.shape[0], 3))
            color[:, 2] = 1
        else:
            color = color.reshape(-1, 3)

        # Remove NaN / Inf points
        valid = np.isfinite(pred_pts).all(axis=1)
        if not valid.all():
            pred_pts = pred_pts[valid]
            color = color[valid]
            if conf is not None:
                conf = conf.reshape(-1)[valid]

        # Confidence threshold filter
        if conf is not None:
            conf_flat = conf.reshape(-1) if conf.ndim > 1 else conf
            mask = conf_flat > self.vis_threshold
            pred_pts = pred_pts[mask]
            color = color[mask]

        if len(pred_pts) == 0:
            return pred_pts, color

        # Downsample
        if downsample_factor > 1 and len(pred_pts) > 0:
            indices = np.arange(0, len(pred_pts), downsample_factor)
            pred_pts = pred_pts[indices]
            color = color[indices]

        return pred_pts, color

    def add_pc(self, step):
        """Add point cloud for a frame."""
        pc = self.pcs[step]["pc"]
        color = self.pcs[step]["color"]
        conf = self.pcs[step]["conf"]
        edge_color = self.pcs[step].get("edge_color", None)

        pred_pts, color = self.parse_pc_data(
            pc, color, conf, edge_color, set_border_color=True,
            downsample_factor=self.downsample_slider.value
        )

        self.vis_pts_list.append(pred_pts)
        self.pc_handles.append(
            self.server.scene.add_point_cloud(
                name=f"/frames/{step}/pred_pts",
                points=pred_pts,
                colors=color,
                point_size=self.psize_slider.value,
            )
        )


    def animate(self):
        """Setup and run animation controls."""
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Train Step", min=0, max=self.num_frames - 1, step=1, initial_value=0, disabled=False
            )
            gui_next_frame = self.server.gui.add_button("Next Step", disabled=False)
            gui_prev_frame = self.server.gui.add_button("Prev Step", disabled=False)
            gui_playing = self.server.gui.add_checkbox("Playing", True)
            gui_framerate = self.server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=20)
            gui_framerate_options = self.server.gui.add_button_group("FPS options", ("10", "20", "30", "60"))

        @gui_next_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % self.num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % self.num_frames

        @gui_playing.on_update
        def _(_) -> None:
            self.gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)

        prev_timestep = self.gui_timestep.value

        @self.gui_timestep.on_update
        def _(_) -> None:
            nonlocal prev_timestep
            current_timestep = self.gui_timestep.value

            if self.current_frame_image is not None and hasattr(self, 'original_images'):
                if current_timestep < len(self.original_images):
                    self.current_frame_image.image = self.original_images[current_timestep]

            with self.server.atomic():
                self.frame_nodes[current_timestep].visible = True
                self.frame_nodes[prev_timestep].visible = False
            self.server.flush()

            prev_timestep = current_timestep

        self.server.scene.add_frame("/frames", show_axes=False)
        self.frame_nodes = []
        for i in range(self.num_frames):
            step = self.all_steps[i]
            self.frame_nodes.append(
                self.server.scene.add_frame(f"/frames/{step}", show_axes=False)
            )
            self.add_pc(step)
            if self.show_camera:
                downsample_factor = int(self.camera_downsample_slider.value)
                if i % downsample_factor == 0:
                    self.add_camera(step)

        prev_timestep = self.gui_timestep.value
        while True:
            if self.on_replay:
                pass
            else:
                if gui_playing.value:
                    self.gui_timestep.value = (self.gui_timestep.value + 1) % self.num_frames
                self.update_frame_visibility()

            time.sleep(1.0 / gui_framerate.value)



    def _save_original_video(self, pointcloud_video_path: str, fps: int, width: int, height: int):
        """Save original images as video."""
        base_path = os.path.splitext(pointcloud_video_path)[0]
        original_video_path = f"{base_path}_original.mp4"

        print(f"Saving original images video to {original_video_path}...")

        try:
            temp_dir = tempfile.mkdtemp(prefix="original_video_")

            for i, img in enumerate(tqdm(self.original_images, desc="Saving original frames")):
                frame = cv2.resize(img, (width, height))
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, frame)

            print("Encoding original video with ffmpeg...")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                original_video_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"Original video saved successfully to {original_video_path}")
            else:
                print(f"FFmpeg error for original video: {result.stderr}")

            shutil.rmtree(temp_dir)

        except Exception as e:
            print(f"Error saving original video: {e}")
            import traceback
            traceback.print_exc()



    def process_uploaded_video(self, file_name: str, file_content: bytes):
        """Processes uploaded video with in-memory model inference and updates viewer state."""
        import tempfile
        import shutil
        import torch
        from lingbot_map.vis.sky_segmentation import apply_sky_segmentation

        print(f"Processing uploaded video for active viewer: {file_name}")
        
        if self.model is None:
            # Laptop / Mock mode fallback
            print("Mock mode: Simulating model inference on uploaded video...")
            # Generate new mock data to simulate successful upload and mapping
            num_frames = 45
            t_list = []
            R_list = []
            
            # Generate a different layout (e.g. S-curve) to show it updated
            for i in range(num_frames):
                tz = float(i) * 0.4
                tx = 2.0 * np.sin(float(i) * 0.2)
                # Yaw rotation matching the tangent
                yaw = 0.4 * np.cos(float(i) * 0.2)
                R = np.array([
                    [np.cos(yaw), 0.0, np.sin(yaw)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(yaw), 0.0, np.cos(yaw)]
                ])
                t_list.append(np.array([tx, 0.0, tz]))
                R_list.append(R)

            self.cam_dict = {
                "t": t_list,
                "R": R_list,
                "focal": [300.0] * num_frames,
                "pp": [(250.0, 250.0)] * num_frames
            }
            
            pc_list = []
            color_list = []
            conf_list = []
            for i in range(num_frames):
                t = t_list[i]
                left_pts = np.stack([np.full(30, t[0]-2.0), np.zeros(30), np.linspace(t[2]-1.0, t[2]+3.0, 30)], axis=1)
                right_pts = np.stack([np.full(30, t[0]+2.0), np.zeros(30), np.linspace(t[2]-1.0, t[2]+3.0, 30)], axis=1)
                pts = np.concatenate([left_pts, right_pts], axis=0)
                pts += np.random.normal(0, 0.05, pts.shape)
                
                pc_list.append(pts)
                color_list.append(np.ones((len(pts), 3)) * 0.9)
                conf_list.append(np.ones(len(pts)) * 1.5)
                
            self.pcs, self.all_steps = self.read_data(pc_list, color_list, conf_list)
            self.num_frames = len(self.all_steps)
            
            cam_positions = np.array([self.cam_dict["t"][s] for s in self.all_steps])
            x_coords = cam_positions[:, 0]
            z_coords = cam_positions[:, 2]
            x_center = (np.min(x_coords) + np.max(x_coords)) / 2.0
            z_center = (np.min(z_coords) + np.max(z_coords)) / 2.0
            max_range = max(np.ptp(x_coords), np.ptp(z_coords)) or 1.0
            padding = max_range * 0.75
            self.bev_x_min = x_center - padding
            self.bev_x_max = x_center + padding
            self.bev_z_min = z_center - padding
            self.bev_z_max = z_center + padding
            
            # Populate dummy original images
            self.original_images = [np.zeros((300, 400, 3), dtype=np.uint8) for _ in range(num_frames)]
            print("Mock video upload successfully processed!")
            return

        # Real model inference mode
        temp_dir = tempfile.mkdtemp(prefix="viser_web_upload_")
        video_path = os.path.join(temp_dir, file_name)
        with open(video_path, "wb") as f:
            f.write(file_content)

        fps_rate = getattr(self.args, "fps", 10)
        stride = getattr(self.args, "stride", 3)
        image_size = getattr(self.args, "image_size", 518)

        from demo import load_images
        images, paths, resolved_folder = load_images(
            video_path=video_path,
            fps=fps_rate,
            first_k=None,
            stride=stride,
            image_size=image_size,
            patch_size=getattr(self.args, "patch_size", 14),
            rotate_clockwise_90=getattr(self.args, "rotate_clockwise_90", False),
        )

        images_gpu = images.to(self.device)
        num_frames = images.shape[0]

        # Resolve keyframe_interval dynamically for the uploaded video length
        keyframe_interval = getattr(self.args, "keyframe_interval", None)
        if keyframe_interval is None:
            if getattr(self.args, "mode", "streaming") == "streaming" and num_frames > 320:
                keyframe_interval = (num_frames + 319) // 320
            else:
                keyframe_interval = 1

        with torch.no_grad(), torch.amp.autocast("cuda" if "cuda" in str(self.device) else "cpu", dtype=self.dtype if "cuda" in str(self.device) else torch.bfloat16):
            if getattr(self.args, "mode", "streaming") == "streaming":
                predictions = self.model.inference_streaming(
                    images_gpu,
                    num_scale_frames=self.args.num_scale_frames,
                    keyframe_interval=keyframe_interval,
                    output_device=None,
                )
            else:
                predictions = self.model.inference_windowed(
                    images_gpu,
                    window_size=self.args.window_size,
                    overlap_size=self.args.overlap_size,
                    overlap_keyframes=self.args.overlap_keyframes,
                    num_scale_frames=self.args.num_scale_frames,
                    keyframe_interval=keyframe_interval,
                    output_device=None
                )

        from demo import postprocess, prepare_for_visualization
        predictions, images_cpu = postprocess(predictions, images)
        preds_processed = prepare_for_visualization(predictions, images_cpu)

        pc_list, color_list, conf_list, cam_dict = self._process_pred_dict(
            preds_processed, use_point_map=False, mask_sky=False, image_folder=resolved_folder
        )

        self.pcs, self.all_steps = self.read_data(
            pc_list, color_list, conf_list
        )
        self.cam_dict = cam_dict
        self.num_frames = len(self.all_steps)

        if self.cam_dict is not None and "t" in self.cam_dict and len(self.all_steps) > 0:
            cam_positions = np.array([self.cam_dict["t"][s] for s in self.all_steps])
            x_coords = cam_positions[:, 0]
            z_coords = cam_positions[:, 2]
            x_center = (np.min(x_coords) + np.max(x_coords)) / 2.0
            z_center = (np.min(z_coords) + np.max(z_coords)) / 2.0
            max_range = max(np.ptp(x_coords), np.ptp(z_coords)) or 1.0
            padding = max_range * 0.75
            self.bev_x_min = x_center - padding
            self.bev_x_max = x_center + padding
            self.bev_z_min = z_center - padding
            self.bev_z_max = z_center + padding

        shutil.rmtree(temp_dir)
        print("Successfully processed video upload!")

    def run(self, background_mode: bool = False):
        """Run the viewer."""
        self.animate()
        if background_mode:
            def server_loop():
                while True:
                    time.sleep(0.001)

            thread = threading.Thread(target=server_loop, daemon=True)
            thread.start()
        else:
            while True:
                time.sleep(10.0)
    def _get_2d_map_render(self, idx: int) -> np.ndarray:
        W_p, H_p = 500, 500
        img = np.full((H_p, W_p, 3), fill_value=[45, 30, 26], dtype=np.uint8)

        if self.cam_dict is None or "t" not in self.cam_dict or len(self.all_steps) == 0:
            return img

        x_min = self.bev_x_min
        x_max = self.bev_x_max
        z_min = self.bev_z_min
        z_max = self.bev_z_max

        # Helper to convert world coordinates to pixel coordinates
        def to_pixels(x, z):
            px = int((x - x_min) / (x_max - x_min) * W_p) if x_max != x_min else W_p // 2
            py = int((z_max - z) / (z_max - z_min) * H_p) if z_max != z_min else H_p // 2
            return px, py

        # 1. Render Free Space wedges
        for i in range(idx + 1):
            step = self.all_steps[i]
            t = self.cam_dict["t"][step]
            R = self.cam_dict["R"][step] if "R" in self.cam_dict else np.eye(3)
            focal = self.cam_dict["focal"][step] if "focal" in self.cam_dict else 350.0
            pp = self.cam_dict["pp"][step] if "pp" in self.cam_dict else (W_p / 2, H_p / 2)

            # Determine distance of frustum wedge based on point cloud depths
            d_frustum = 5.0
            if step in self.pcs:
                pc = self.pcs[step]["pc"]
                conf = self.pcs[step]["conf"]
                pts = pc.reshape(-1, 3)
                conf_flat = conf.reshape(-1)
                valid = np.isfinite(pts).all(axis=1)
                if np.any(valid):
                    pts = pts[valid]
                    conf_flat = conf_flat[valid]
                    mask = conf_flat > self.vis_threshold
                    pts_valid = pts[mask]
                    if len(pts_valid) > 0:
                        dists = np.linalg.norm(pts_valid[:, [0, 2]] - t[[0, 2]], axis=1)
                        d_frustum = np.percentile(dists, 90)
                        d_frustum = np.clip(d_frustum, 1.0, 10.0)

            # Camera directions in horizontal X-Z plane
            dir_fwd = np.array([R[0, 2], R[2, 2]])
            norm_fwd = np.linalg.norm(dir_fwd)
            if norm_fwd > 1e-5:
                dir_fwd = dir_fwd / norm_fwd

            dir_right = np.array([R[0, 0], R[2, 0]])
            norm_right = np.linalg.norm(dir_right)
            if norm_right > 1e-5:
                dir_right = dir_right / norm_right

            # Frustum base half-width
            fov_factor = pp[0] / focal
            half_w = d_frustum * fov_factor

            # Frustum corners in world coordinates
            P_left = t[[0, 2]] + d_frustum * dir_fwd - half_w * dir_right
            P_right = t[[0, 2]] + d_frustum * dir_fwd + half_w * dir_right

            # Convert to pixel space
            c_px = to_pixels(t[0], t[2])
            l_px = to_pixels(P_left[0], P_left[1])
            r_px = to_pixels(P_right[0], P_right[1])

            pts_poly = np.array([c_px, l_px, r_px], dtype=np.int32)
            cv2.fillPoly(img, [pts_poly], color=[216, 200, 178])

        # 2. Render Obstacle points
        all_pts_x = []
        all_pts_z = []
        for i in range(idx + 1):
            step = self.all_steps[i]
            if step in self.pcs:
                pc = self.pcs[step]["pc"]
                conf = self.pcs[step]["conf"]
                pts = pc.reshape(-1, 3)
                conf_flat = conf.reshape(-1)
                valid = np.isfinite(pts).all(axis=1)
                pts = pts[valid]
                conf_flat = conf_flat[valid]
                mask = conf_flat > self.vis_threshold
                pts_valid = pts[mask]
                if len(pts_valid) > 0:
                    ds = max(1, len(pts_valid) // 150)
                    all_pts_x.append(pts_valid[::ds, 0])
                    all_pts_z.append(pts_valid[::ds, 2])

        if len(all_pts_x) > 0:
            pts_x = np.concatenate(all_pts_x)
            pts_z = np.concatenate(all_pts_z)

            # Map to pixel space
            pxs = ((pts_x - x_min) / (x_max - x_min) * W_p).astype(np.int32)
            pys = ((z_max - pts_z) / (z_max - z_min) * H_p).astype(np.int32)

            mask_in = (pxs >= 0) & (pxs < W_p) & (pys >= 0) & (pys < H_p)
            pxs = pxs[mask_in]
            pys = pys[mask_in]

            # Set obstacle pixels
            img[pys, pxs] = [56, 200, 251]
            img[np.clip(pys + 1, 0, H_p - 1), pxs] = [56, 200, 251]
            img[pys, np.clip(pxs + 1, 0, W_p - 1)] = [56, 200, 251]

        # 3. Draw Grid Lines and metric ticks
        grid_min_x = int(np.floor(x_min / 5.0) * 5.0)
        grid_max_x = int(np.ceil(x_max / 5.0) * 5.0)
        grid_min_z = int(np.floor(z_min / 5.0) * 5.0)
        grid_max_z = int(np.ceil(z_max / 5.0) * 5.0)

        for gx in range(grid_min_x, grid_max_x + 5, 5):
            px, _ = to_pixels(gx, z_min)
            if 0 <= px < W_p:
                cv2.line(img, (px, 0), (px, H_p), color=[60, 50, 40], thickness=1)
                cv2.putText(img, str(gx), (px + 4, H_p - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, [150, 140, 130], 1)

        for gz in range(grid_min_z, grid_max_z + 5, 5):
            _, py = to_pixels(x_min, gz)
            if 0 <= py < H_p:
                cv2.line(img, (0, py), (W_p, py), color=[60, 50, 40], thickness=1)
                cv2.putText(img, str(gz), (10, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, [150, 140, 130], 1)

        # 4. Draw Camera Trajectory
        traj_pts = []
        for i in range(idx + 1):
            step = self.all_steps[i]
            t = self.cam_dict["t"][step]
            px, py = to_pixels(t[0], t[2])
            traj_pts.append([px, py])

        if len(traj_pts) > 1:
            cv2.polylines(img, [np.array(traj_pts, dtype=np.int32)], isClosed=False, color=[104, 206, 90], thickness=2)

        # 5. Draw Active/Current Camera Position
        if len(traj_pts) > 0:
            cv2.circle(img, tuple(traj_pts[-1]), radius=6, color=[11, 221, 251], thickness=-1)

        return img


    def run(self, background_mode: bool = False):
        import uvicorn
        from backend.server import app
        import backend.server
        backend.server.viewer_instance = self
        uvicorn.run(app, host="0.0.0.0", port=8000)
