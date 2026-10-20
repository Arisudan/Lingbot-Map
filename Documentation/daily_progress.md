# 2D Sparse Mapping - Daily Progress Log

This progress log tracks all daily updates, implementation details, package installations, and visualizer improvements made to the GCT 2D Sparse Mapping pipeline.

---

## 27-07-2026 --> Progress

Today, the main focus was to bridge the code updates from the GPU PC server to the local laptop, package all our code optimizations into a standalone wrapper, and organize our visualizer development pipeline.

Here is what was completed:

1. **Synchronized Workspace over SSH**:
   * Established the local workspace connections.
   * Transferred the updated, clean codebase files (`demo.py` and `point_cloud_viewer.py`) from the Linux PC terminal directly onto the laptop using `scp` commands.
   * Verified that the laptop files compile cleanly without any python syntax errors.

2. **Implemented Standalone 2D Viser Wrapper (`viser_2d_wrapper.py`)**:
   * Created a clean, standalone wrapper file under `lingbot_map/vis/viser_2d_wrapper.py` referencing the structural layout of Meta's original visualizer wrapper.
   * Wrapped the entire pipeline so it can run independently of the main GUI class when needed, allowing quick visualization of predicted pose coordinates and trajectory paths.

3. **External Dependencies & OpenCV Implementation**:
   * Integrated `opencv-python` (imported as `cv2`) to handle all 2D canvas drawing calculations. We used it to:
     * Compute and render free-space camera frustum wedges (`cv2.fillPoly`).
     * Plot color-coded obstacle coordinates and grids (`cv2.circle`, `cv2.line`).
     * Connect frame-by-frame trajectory coordinates using paths (`cv2.polylines`).
   * Documented dependencies (`opencv-python`, `viser`, `numpy`, `matplotlib`, `torch`) inside the code docstrings and the optimization report.

4. **Optimized Frame Extraction Pipeline**:
   * Refined how the model handles video uploads. By calculating the frame extraction interval dynamically via `effective_interval = interval * stride` inside `demo.py`, we skipped writing thousands of redundant video frames to the disk. 
   * Video uploads are now processed in just seconds instead of minutes.

5. **Timeline Sync in 3D Scene**:
   * Linked the timeline's update listener to dynamically refresh the flat BEV scene image on every timestep change, making the drone pointer and trajectory path trace in real-time.

6. **Hiding Trajectory Frustums (Task 1)**:
   * Disabled showing the 3D camera wireframe frustum squares/pyramids in the Viser scene by default (`show_camera` defaults to `False` in both the wrapper and `PointCloudViewer`). Users can still toggle them via the GUI panel if needed.

7. **Orientation Angle Correction (Task 2)**:
   * Rotated the 2D floor map plane in the 3D Viser canvas by 180 degrees (pi radians) around the vertical Z-axis. This corrects the orientation and prevents it from appearing upside down to the user.

8. **Zero-Lag Frame Processing Optimization (Task 3)**:
   * Replaced standard `cap.read()` sequential decoding with optimized `cap.grab()` calls for all skipped frames in `demo.py`. This avoids decoding unneeded frames, resulting in lag-free video sparsing and faster extraction speeds.

9. **Interactive Map Alignment Controls**:
   * Implemented a collapsible **"Map Alignment"** GUI panel in both the main visualizer and the 2D wrapper. Added translation sliders (Offset X, Offset Z, Height Y), a scale slider (Map Scale), and a yaw rotation slider (Map Rotation). This gives users complete flexibility to adjust, rotate, and align the 2D floor map plane anywhere in the 3D scene.

10. **Morphological Vacuum Cleaner Floor Plan Styling**:
    * Redesigned the 2D map renderer to generate a clean, connected floor plan resembling a robot vacuum cleaner map.
    * Used OpenCV morphological closing (`cv2.morphologyEx` with `cv2.MORPH_CLOSE`) to smooth and merge the free-space camera frustum wedges.
    * Used OpenCV dilation (`cv2.dilate`) to group sparse obstacle points into thick, solid, continuous boundary walls.
    * Implemented a premium dark mode color theme: deep slate background, soft light-blue explored area, and bold amber obstacle walls.


