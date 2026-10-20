"""
Quick visualization wrapper for 2D Sparse Map with Pose Estimation using Viser.

Required Dependencies:
    - opencv-python (cv2): Used to extract/save video frames, and render the grid floor map (drawing wedges with cv2.fillPoly, trajectories with cv2.polylines, and camera points with cv2.circle).
    - viser: The main WebGL visualization server.
    - numpy: For array transformations and coordinate projections.
    - matplotlib: Generates the color gradient map for camera pose visualizations.
    - torch (PyTorch): Handles model loading and inference on GPU/CPU.
"""

import time
import threading
import os
import tempfile
import shutil
from typing import List, Optional
import numpy as np
import cv2
import torch
import viser
import viser.transforms as tf
from tqdm.auto import tqdm

from lingbot_map.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from lingbot_map.vis.sky_segmentation import apply_sky_segmentation


def viser_2d_wrapper(
    model,
    pred_dict: Optional[dict] = None,
    port: int = 8080,
    init_conf_threshold: float = 1.5,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: Optional[str] = None,
    device = "cpu",
    dtype = torch.float32,
    args = None,
):
    """
    Visualize predicted 2D Sparse Map, camera trajectory, and frustums with Viser.

    Required Dependencies:
        - opencv-python (cv2): Used to extract/save video frames, and render the grid floor map (drawing wedges with cv2.fillPoly, trajectories with cv2.polylines, and camera points with cv2.circle).
        - viser: The main WebGL visualization server.
        - numpy: For array transformations and coordinate projections.
        - matplotlib: Generates the color gradient map for camera pose visualizations.
        - torch (PyTorch): Handles model loading and inference on GPU/CPU.

    Args:
        model: Trained model instance for running dynamic video inference.
        pred_dict: Optional dictionary containing initial predictions.
        port: Port number for the viser server.
        init_conf_threshold: Initial confidence threshold for obstacle filtering.
        use_point_map: Whether to visualize world_points or use depth-based points.
        background_mode: Whether to run the server in a background thread.
        mask_sky: Whether to apply sky segmentation to filter out sky points.
        image_folder: Path to the folder containing input images.
        device: Torch device (cpu or cuda).
        dtype: Model precision datatype.
        args: Parsed command-line arguments.
    """
    print(f"Starting 2D Sparse Map viser server on port {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Viewer state variables
    state = {
        "pcs": {},
        "all_steps": [],
        "cam_dict": None,
        "num_frames": 0,
        "original_images": [],
        "bev_x_min": -10.0,
        "bev_x_max": 10.0,
        "bev_z_min": -10.0,
        "bev_z_max": 10.0,
        "camera_colors": np.array([[1.0, 1.0, 1.0, 1.0]]),
    }

    # Lock for thread-safety during dynamic updates
    update_lock = threading.Lock()

    # GUI handles
    map_status = server.gui.add_text("Status", initial_value="Ready")
    upload_video_button = server.gui.add_upload_button("Upload Video", mime_type="video/*")
    upload_stride_slider = server.gui.add_slider("Upload Stride", min=1, max=10, step=1, initial_value=3)
    upload_res_dropdown = server.gui.add_dropdown("Upload Resolution", options=["256", "384", "518"], initial_value="384")

    gui_timestep = server.gui.add_slider(
        "Timeline Step", min=0, max=0, step=1, initial_value=0, disabled=True
    )
    gui_playing = server.gui.add_checkbox("Play", initial_value=False)
    gui_fps = server.gui.add_slider("Playback FPS", min=1, max=30, step=1, initial_value=10)

    # 2D Map visualizer options
    show_2d_map_checkbox = server.gui.add_checkbox("Show 2D Map Floor", initial_value=True)
    show_camera_checkbox = server.gui.add_checkbox("Show Cameras", initial_value=False)
    camera_downsample_slider = server.gui.add_slider("Camera Interval", min=1, max=10, step=1, initial_value=1)
    
    # Map Alignment controls
    with server.gui.add_folder("Map Alignment"):
        map_offset_x = server.gui.add_slider("Offset X", min=-20.0, max=20.0, step=0.1, initial_value=0.0)
        map_offset_z = server.gui.add_slider("Offset Z", min=-20.0, max=20.0, step=0.1, initial_value=0.0)
        map_height = server.gui.add_slider("Map Height Y", min=-5.0, max=5.0, step=0.05, initial_value=0.0)
        map_scale = server.gui.add_slider("Map Scale", min=0.1, max=5.0, step=0.05, initial_value=1.0)
        map_rotation = server.gui.add_slider("Map Rotation", min=0, max=360, step=1, initial_value=180)

    # 2D Map sidebar image and scene handles
    bev_image_ui = server.gui.add_image(np.zeros((100, 100, 3), dtype=np.uint8), label="2D Floor Map Preview")
    bev_scene_handle = None

    # Camera handles lists
    cam_handles = []
    frame_nodes = []

    def get_ground_y_level() -> float:
        """Find lowest camera position coordinate as reference level."""
        if state["cam_dict"] is not None and "t" in state["cam_dict"] and len(state["all_steps"]) > 0:
            ts = np.array([state["cam_dict"]["t"][s] for s in state["all_steps"]])
            return float(np.min(ts[:, 1])) - 0.2
        return -1.0

    def compute_scene_center_and_scale():
        """Compute spatial center and scale of camera positions."""
        if state["cam_dict"] is not None and "t" in state["cam_dict"] and len(state["all_steps"]) > 0:
            cam_positions = np.array([state["cam_dict"]["t"][s] for s in state["all_steps"]])
            center = np.mean(cam_positions, axis=0)
            if len(cam_positions) > 1:
                extent = np.ptp(cam_positions, axis=0)
                scale = np.linalg.norm(extent)
            else:
                scale = 1.0
        else:
            center = np.array([0.0, 0.0, 0.0])
            scale = 10.0
        return center, max(scale, 0.1)

    def get_2d_map_render(idx: int) -> np.ndarray:
        """Render the 2D grid floor map up to frame index `idx`."""
        # High-Definition 4K Canvas
        W_p, H_p = 2000, 2000
        # Deep slate-navy background for dark mode theme
        img = np.full((H_p, W_p, 3), fill_value=[34, 29, 24], dtype=np.uint8)

        if state["cam_dict"] is None or "t" not in state["cam_dict"] or len(state["all_steps"]) == 0:
            return img

        x_min = state["bev_x_min"]
        x_max = state["bev_x_max"]
        z_min = state["bev_z_min"]
        z_max = state["bev_z_max"]

        def to_pixels(x, z):
            px = int((x - x_min) / (x_max - x_min) * W_p) if x_max != x_min else W_p // 2
            py = int((z_max - z) / (z_max - z_min) * H_p) if z_max != z_min else H_p // 2
            return px, py

        # Initialize cache if missing or scrubbing backwards
        if "_map_cache_idx" not in state or idx < state.get("_map_cache_idx", -1):
            state["_map_cache_idx"] = -1
            state["_cached_free_space"] = np.zeros((H_p, W_p), dtype=np.uint8)
            state["_cached_obstacle"] = np.zeros((H_p, W_p), dtype=np.uint8)

        sampled_indices = list(range(idx + 1))
        new_indices = [i for i in sampled_indices if i > state["_map_cache_idx"]]

        # Process delta (O(1) incremental caching)
        for i in new_indices:
            step = state["all_steps"][i]
            t = state["cam_dict"]["t"][step]
            R = state["cam_dict"]["R"][step] if "R" in state["cam_dict"] else np.eye(3)
            focal = state["cam_dict"]["focal"][step] if "focal" in state["cam_dict"] else 350.0
            pp = state["cam_dict"]["pp"][step] if "pp" in state["cam_dict"] else (W_p / 2, H_p / 2)

            d_frustum = 5.0
            c_px, c_py = to_pixels(t[0], t[2])
            if step in state["pcs"]:
                pc = state["pcs"][step]["pc"]
                conf = state["pcs"][step]["conf"]
                pts = pc.reshape(-1, 3)
                conf_flat = conf.reshape(-1)
                valid = np.isfinite(pts).all(axis=1)
                if np.any(valid):
                    pts = pts[valid]
                    conf_flat = conf_flat[valid]
                    mask = conf_flat > init_conf_threshold
                    pts_valid = pts[mask]
                    if len(pts_valid) > 0:
                        # 1. Frustum dist & 8-Meter Distance Clipping Filter
                        dists = np.linalg.norm(pts_valid[:, [0, 2]] - t[[0, 2]], axis=1)
                        dist_mask = dists < 8.0
                        pts_valid = pts_valid[dist_mask]
                        dists = dists[dist_mask]

                        if len(pts_valid) > 0:
                            d_frustum = np.percentile(dists, 90)
                            d_frustum = np.clip(d_frustum, 1.0, 10.0)

                        # Obstacle extraction with Z-Axis Height Filtering
                        map_h_val = map_height.value if 'map_height' in locals() or 'map_height' in globals() else 0.0
                        slice_y = t[1] + map_h_val
                        height_band = 0.15
                        height_mask = (pts_valid[:, 1] > slice_y - height_band) & (pts_valid[:, 1] < slice_y + height_band)
                        pts_valid = pts_valid[height_mask]

                        if len(pts_valid) > 0:
                            ds = max(1, len(pts_valid) // 150)
                            pxs = ((pts_valid[::ds, 0] - x_min) / (x_max - x_min) * W_p).astype(np.int32)
                            pys = ((z_max - pts_valid[::ds, 2]) / (z_max - z_min) * H_p).astype(np.int32)
                            mask_in = (pxs >= 0) & (pxs < W_p) & (pys >= 0) & (pys < H_p)
                            pxs = pxs[mask_in]
                            pys = pys[mask_in]
                            if len(pxs) > 0:
                                state["_cached_obstacle"][pys, pxs] = 255

            dir_fwd = np.array([R[0, 2], R[2, 2]])
            norm_fwd = np.linalg.norm(dir_fwd)
            if norm_fwd > 1e-5: dir_fwd = dir_fwd / norm_fwd

            dir_right = np.array([R[0, 0], R[2, 0]])
            norm_right = np.linalg.norm(dir_right)
            if norm_right > 1e-5: dir_right = dir_right / norm_right

            fov_factor = pp[0] / focal
            half_w = d_frustum * fov_factor

            P_left = t[[0, 2]] + d_frustum * dir_fwd - half_w * dir_right
            P_right = t[[0, 2]] + d_frustum * dir_fwd + half_w * dir_right

            c_px, c_py = to_pixels(t[0], t[2])
            l_px, l_py = to_pixels(P_left[0], P_left[1])
            r_px, r_py = to_pixels(P_right[0], P_right[1])

            pts_poly = np.array([[c_px, c_py], [l_px, l_py], [r_px, r_py]], dtype=np.int32)
            cv2.fillPoly(state["_cached_free_space"], [pts_poly], 255)

        state["_map_cache_idx"] = idx

        # Clone cached masks to safely apply morphology
        free_space_mask = state["_cached_free_space"].copy()
        obstacle_mask = state["_cached_obstacle"].copy()

        # Apply morphology smoothing (Vacuum Cleaner floor plan style)
        kernel_free = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        free_space_mask = cv2.morphologyEx(free_space_mask, cv2.MORPH_CLOSE, kernel_free)

        # Re-adding obstacle dilation to thicken the 1-pixel sparse dots in 4K
        kernel_obs = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        obstacle_mask = cv2.dilate(obstacle_mask, kernel_obs, iterations=1)

        # Style colors:
        # Explored Free Space: soft light gray-blue
        img[free_space_mask > 0] = [178, 200, 216]
        # Obstacle Walls: Crisp White
        img[obstacle_mask > 0] = [255, 255, 255]

        # 3. Draw Grid Lines
        grid_min_x = int(np.floor(x_min / 5.0) * 5.0)
        grid_max_x = int(np.ceil(x_max / 5.0) * 5.0)
        grid_min_z = int(np.floor(z_min / 5.0) * 5.0)
        grid_max_z = int(np.ceil(z_max / 5.0) * 5.0)

        for gx in range(grid_min_x, grid_max_x + 5, 5):
            px, _ = to_pixels(gx, z_min)
            if 0 <= px < W_p:
                cv2.line(img, (px, 0), (px, H_p), color=[40, 50, 60], thickness=2)
                cv2.putText(img, str(gx), (px + 8, H_p - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [130, 140, 150], 2)

        for gz in range(grid_min_z, grid_max_z + 5, 5):
            _, py = to_pixels(x_min, gz)
            if 0 <= py < H_p:
                cv2.line(img, (0, py), (W_p, py), color=[40, 50, 60], thickness=2)
                cv2.putText(img, str(gz), (20, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [130, 140, 150], 2)

        # 4. Draw Camera Trajectory
        traj_pts = []
        for i in range(idx + 1):
            step = state["all_steps"][i]
            t = state["cam_dict"]["t"][step]
            px, py = to_pixels(t[0], t[2])
            traj_pts.append([px, py])

        if len(traj_pts) > 1:
            # Green trajectory path
            cv2.polylines(img, [np.array(traj_pts, dtype=np.int32)], isClosed=False, color=[90, 206, 104], thickness=6)

        # 5. Draw Active/Current Camera Position
        if len(traj_pts) > 0:
            # Yellow pointer dot
            cv2.circle(img, tuple(traj_pts[-1]), radius=20, color=[251, 221, 11], thickness=-1)

        return img

    def update_bev_scene_image(idx: int):
        """Update the BEV image in the 3D scene."""
        nonlocal bev_scene_handle
        if not show_2d_map_checkbox.value:
            if bev_scene_handle is not None:
                bev_scene_handle.remove()
                bev_scene_handle = None
            return

        rgb_img = get_2d_map_render(idx)
        
        offset_x = map_offset_x.value if 'map_offset_x' in locals() or 'map_offset_x' in globals() else 0.0
        offset_z = map_offset_z.value if 'map_offset_z' in locals() or 'map_offset_z' in globals() else 0.0
        offset_y = map_height.value if 'map_height' in locals() or 'map_height' in globals() else 0.0
        scale_val = map_scale.value if 'map_scale' in locals() or 'map_scale' in globals() else 1.0
        rot_deg = map_rotation.value if 'map_rotation' in locals() or 'map_rotation' in globals() else 180.0

        x_center = (state["bev_x_min"] + state["bev_x_max"]) / 2.0 + offset_x
        z_center = (state["bev_z_min"] + state["bev_z_max"]) / 2.0 + offset_z
        y_level = get_ground_y_level() + offset_y

        q_rot = tf.SO3(np.array([0.7071, 0.7071, 0.0, 0.0]))
        theta = rot_deg * np.pi / 180.0
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0]
        ])
        R_new = R_z @ q_rot.as_matrix()
        q_new = tf.SO3.from_matrix(R_new)

        bev_scene_handle = server.scene.add_image(
            name="/2d_map",
            image=rgb_img,
            render_width=(state["bev_x_max"] - state["bev_x_min"]) * scale_val,
            render_height=(state["bev_z_max"] - state["bev_z_min"]) * scale_val,
            format="png",
            position=(x_center, y_level, z_center),
            # lies flat on the horizontal X-Z plane, rotated 90 deg around X and custom yaw around Z
            wxyz=q_new.wxyz,
        )

    def add_camera(step):
        """Draw camera coordinate axis and frustum."""
        cam = state["cam_dict"]
        focal = cam["focal"][step] if cam and "focal" in cam else 1.0
        pp = cam["pp"][step] if cam and "pp" in cam else (1.0, 1.0)
        R = cam["R"][step] if cam and "R" in cam else np.eye(3)
        t = cam["t"][step] if cam and "t" in cam else np.zeros(3)

        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(pp[0] / focal)
        aspect = pp[0] / pp[1]

        step_index = state["all_steps"].index(step) if step in state["all_steps"] else 0
        camera_color = state["camera_colors"][step_index % len(state["camera_colors"])]
        camera_color_rgb = tuple((camera_color[:3] * 255).astype(int))

        server.scene.add_frame(
            f"/frames/{step}/camera_frame",
            wxyz=q,
            position=t,
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
        )

        frustum_handle = server.scene.add_camera_frustum(
            name=f"/frames/{step}/camera",
            fov=fov,
            aspect=aspect,
            wxyz=q,
            position=t,
            scale=0.03,
            color=camera_color_rgb,
        )

        if frustum_handle is not None:
            @frustum_handle.on_click
            def _(event) -> None:
                look_at_pt = t + R[:, 2] * 0.5
                up_dir = -R[:, 1]
                for client in server.get_clients().values():
                    client.camera.up_direction = tuple(up_dir)
                    client.camera.position = tuple(t)
                    client.camera.look_at = tuple(look_at_pt)

            cam_handles.append(frustum_handle)

    def regenerate_cameras():
        """Redraw all camera frustums according to downsampling."""
        for handle in cam_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        cam_handles.clear()

        if show_camera_checkbox.value:
            downsample = int(camera_downsample_slider.value)
            for i, step in enumerate(state["all_steps"]):
                if i % downsample == 0:
                    add_camera(step)

    def process_pred_dict(preds: dict):
        """Unpack prediction output dictionary to viewer data."""
        with update_lock:
            # 1. Unpack point maps and camera poses
            images_t = preds["images"]
            world_pts = preds["world_points"]
            conf_pts = preds["world_points_conf"]
            depth_map = preds["depth"]
            depth_conf = preds["depth_conf"]
            extrinsics = preds["extrinsic"]
            intrinsics = preds["intrinsic"]

            if not use_point_map:
                world_points = unproject_depth_map_to_point_map(depth_map, extrinsics, intrinsics)
                conf = depth_conf
            else:
                world_points = world_pts
                conf = conf_pts

            if mask_sky and image_folder is not None:
                conf = apply_sky_segmentation(conf, image_folder)

            # Colors and metadata setup
            colors = images_t.transpose(0, 2, 3, 1)
            S_len = world_points.shape[0]

            state["pcs"].clear()
            state["all_steps"] = list(range(S_len))
            state["num_frames"] = S_len
            state["original_images"] = [colors[i] for i in range(S_len)]

            cam_to_world_mat = closed_form_inverse_se3(extrinsics)
            cam_dict = {
                "R": cam_to_world_mat[:, :3, :3],
                "t": cam_to_world_mat[:, :3, 3],
                "focal": intrinsics[:, 0, 0],
                "pp": intrinsics[:, :2, 2],
            }
            state["cam_dict"] = cam_dict

            # Store per-frame cloud data
            for i in range(S_len):
                state["pcs"][i] = {
                    "pc": world_points[i],
                    "color": colors[i],
                    "conf": conf[i]
                }

            # Recalculate aspect-ratio limits for BEV Grid
            if cam_dict is not None and "t" in cam_dict and S_len > 0:
                cam_positions = cam_dict["t"]
                x_coords = cam_positions[:, 0]
                z_coords = cam_positions[:, 2]
                x_center = (np.min(x_coords) + np.max(x_coords)) / 2.0
                z_center = (np.min(z_coords) + np.max(z_coords)) / 2.0
                max_range = max(np.ptp(x_coords), np.ptp(z_coords)) or 1.0
                padding = max_range * 0.75
                state["bev_x_min"] = x_center - padding
                state["bev_x_max"] = x_center + padding
                state["bev_z_min"] = z_center - padding
                state["bev_z_max"] = z_center + padding

            # Color spectrum for cameras
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("viridis")
            state["camera_colors"] = cmap(np.linspace(0, 1, max(1, S_len)))

            # Recreate frame nodes
            try:
                server.scene.add_frame("/frames", show_axes=False).remove()
            except Exception:
                pass
            server.scene.add_frame("/frames", show_axes=False)

            frame_nodes.clear()
            for i in range(S_len):
                frame_nodes.append(server.scene.add_frame(f"/frames/{i}", show_axes=False))

            # Update timeline sliders limits
            gui_timestep.max = max(0, S_len - 1)
            gui_timestep.value = 0
            gui_timestep.disabled = (S_len <= 1)

            # Draw initial 2D maps and cameras
            bev_image_ui.image = get_2d_map_render(0)
            update_bev_scene_image(0)
            regenerate_cameras()

            # Center client cameras to newly loaded scene
            center, scale = compute_scene_center_and_scale()
            for client in server.get_clients().values():
                client.camera.up_direction = (0.0, 0.0, 1.0)
                client.camera.look_at = tuple(center)
                client.camera.position = (center[0] - scale * 1.0, center[1] - scale * 1.5, center[2] + scale * 1.0)

    # Timestep slider callback
    @gui_timestep.on_update
    def _(_) -> None:
        current_t = gui_timestep.value
        bev_image_ui.image = get_2d_map_render(current_t)
        update_bev_scene_image(current_t)

        # Update frame visibility
        if len(frame_nodes) > 0:
            with server.atomic():
                for i, node in enumerate(frame_nodes):
                    node.visible = (i == current_t)

    # Dynamic video upload handler
    def on_video_upload(file_name: str, file_content: bytes):
        try:
            map_status.value = "Uploading & extracting..."
            print(f"Processing uploaded video: {file_name}")

            temp_dir = tempfile.mkdtemp(prefix="viser_2d_upload_")
            video_path = os.path.join(temp_dir, file_name)
            with open(video_path, "wb") as f:
                f.write(file_content)

            fps_rate = getattr(args, "fps", 10)
            stride = int(upload_stride_slider.value)
            image_size = int(upload_res_dropdown.value)

            from demo import load_images
            images, paths, resolved_folder = load_images(
                video_path=video_path,
                fps=fps_rate,
                first_k=None,
                stride=stride,
                image_size=image_size,
                patch_size=getattr(args, "patch_size", 14),
                rotate_clockwise_90=getattr(args, "rotate_clockwise_90", False),
            )

            map_status.value = f"Running model inference ({len(paths)} frames)..."
            images_gpu = images.to(device)

            with torch.no_grad(), torch.amp.autocast("cuda" if "cuda" in str(device) else "cpu", dtype=dtype if "cuda" in str(device) else torch.bfloat16):
                if getattr(args, "mode", "streaming") == "streaming":
                    predictions = model.inference_streaming(
                        images_gpu,
                        num_scale_frames=args.num_scale_frames,
                        keyframe_interval=args.keyframe_interval,
                        output_device=None,
                    )
                else:
                    predictions = model.inference_windowed(
                        images_gpu,
                        window_size=args.window_size,
                        overlap_size=args.overlap_size,
                        overlap_keyframes=args.overlap_keyframes,
                        num_scale_frames=args.num_scale_frames,
                        keyframe_interval=args.keyframe_interval,
                        output_device=None
                    )

            from demo import postprocess, prepare_for_visualization
            predictions, images_cpu = postprocess(predictions, images)
            preds_processed = prepare_for_visualization(predictions, images_cpu)

            # Repopulate scene
            map_status.value = "Updating map..."
            process_pred_dict(preds_processed)

            shutil.rmtree(temp_dir)
            map_status.value = "Ready (Processed)"
            print("Successfully processed video upload!")

        except Exception as e:
            map_status.value = f"Error: {e}"
            print(f"Error processing video upload: {e}")
            import traceback
            traceback.print_exc()

    @upload_video_button.on_upload
    def _(_) -> None:
        uploaded_file = upload_video_button.value
        if uploaded_file is not None:
            # Start background thread to keep viser responsive
            thread = threading.Thread(
                target=on_video_upload,
                args=(uploaded_file.name, uploaded_file.content),
                daemon=True
            )
            thread.start()

    # Camera toggles
    @show_camera_checkbox.on_update
    def _(_) -> None:
        regenerate_cameras()

    @camera_downsample_slider.on_update
    def _(_) -> None:
        regenerate_cameras()

    # Map Alignment slider updates
    @map_offset_x.on_update
    def _(_) -> None:
        update_bev_scene_image(gui_timestep.value)

    @map_offset_z.on_update
    def _(_) -> None:
        update_bev_scene_image(gui_timestep.value)

    @map_height.on_update
    def _(_) -> None:
        update_bev_scene_image(gui_timestep.value)

    @map_scale.on_update
    def _(_) -> None:
        update_bev_scene_image(gui_timestep.value)

    @map_rotation.on_update
    def _(_) -> None:
        update_bev_scene_image(gui_timestep.value)

    # Client connection handler
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print(f"Client connected: {client.client_id}")
        client.camera.projection_type = "perspective"

        center, scale = compute_scene_center_and_scale()
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.look_at = tuple(center)
        client.camera.position = (center[0] - scale * 1.0, center[1] - scale * 1.5, center[2] + scale * 1.0)

    # Initialize with pred_dict if provided
    if pred_dict is not None:
        process_pred_dict(pred_dict)

    # Playback update loop
    def playback_loop():
        while True:
            time.sleep(1.0 / max(1, gui_fps.value))
            if gui_playing.value and state["num_frames"] > 0:
                with update_lock:
                    next_t = (gui_timestep.value + 1) % state["num_frames"]
                    gui_timestep.value = next_t

    thread_play = threading.Thread(target=playback_loop, daemon=True)
    thread_play.start()

    print("Starting viser 2D map visualization wrapper...")
    if background_mode:
        def background_loop():
            while True:
                time.sleep(0.01)
        thread_bg = threading.Thread(target=background_loop, daemon=True)
        thread_bg.start()
    else:
        while True:
            time.sleep(0.1)

    return server
