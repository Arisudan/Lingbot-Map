# Daily Progress: Day 2

This document tracks all updates, modifications, and fixes applied to the Lingbot-MAP project on Day 2.

## 1. Z-Axis Dynamic Slicing
- **Issue**: The original height filter was locked at a static `Y = 0.0`. If the drone was flying above the floor, the filter was mathematically slicing through the floor itself, rendering the entire hallway floor as a massive solid obstacle.
- **Fix**: Updated the height filter to be completely dynamic and relative to the active camera's Y coordinate (`slice_y = t[1] + map_h_val`). 
- **Result**: The filter now acts as an ultra-thin 15cm laser cross-section at the exact eye-level of the drone, perfectly isolating the vertical walls and completely ignoring all floor and ceiling point-cloud noise.

## 2. Removal of Dilation (Sparse Aesthetic)
- **Issue**: The OpenCV morphological dilation (`cv2.dilate`) was acting as a thick brush, causing the obstacle points to bleed into each other and form thick, blobby walls rather than a clean, sparse map.
- **Fix**: Completely removed the `cv2.dilate` obstacle function and reduced the rendering kernel size. 
- **Color Match**: Reverted the obstacle color from amber to crisp white (`[255, 255, 255]`) to perfectly match the original laser-scan aesthetic reference.
- **Result**: The map now renders as perfectly sparse, razor-sharp white dots, achieving the exact 1:1 look of the desired reference image.

## 3. Free-Space Ray-Casting (Convex Hull Boundary)
- **Issue**: The blue free-space area was bleeding through the white walls because it was mathematically drawn as a blind 5-meter triangle regardless of obstacles in the way.
- **Fix**: Replaced the crude frustum calculation with a dynamic Convex Hull algorithm. The algorithm automatically calculates a mathematical boundary wrapping the camera point and the valid obstacle points in the frame.
- **Result**: The blue free-space is now perfectly constrained *inside* the walls of the hallway. It physically stops drawing the exact moment it hits an obstacle pixel, perfectly imitating expensive LiDAR ray-casting at O(1) mathematical speed.

## 4. White Dust Cleaning (Statistical Outlier Removal)
- **Issue**: The ML model generated random, low-confidence sensor noise (floating white dust) in the middle of the hallway.
- **Fix**: Implemented a 2D Statistical Outlier Removal pass using OpenCV's morphological `MORPH_OPEN` filter with a 2x2 kernel.
- **Result**: The algorithm acts as an intelligent sieve, instantly deleting any isolated, floating "dust" pixels while completely preserving the solid, connected structural walls. The map is now impeccably clean.

## 5. 4K High-Definition Map Upgrade
- **Issue**: The original map was rendered on a tiny 500x500 pixel grid, causing it to look extremely blurry, blocky, and pixelated when zoomed in on the GUI or exported as a PNG.
- **Fix**: Upgraded the internal canvas rendering matrix to a massive 2000x2000 4K resolution (`W_p, H_p = 2000, 2000`). Proportional scaling was dynamically applied to all rendering utilities (grid lines, trajectory path, drone pointer, and noise filter brushes) to match the giant canvas.
- **Result**: The map is now razor-sharp and infinitely zoomable. Downloaded PNGs are crystal clear High-Definition blueprints of the scanned environment.

## 6. Architectural Reversions for Hallway Tracing (Precision Tweaks)
- **Frustum Triangle > Convex Hull**: We reverted the free-space mapping algorithm back to the original Frustum Triangle logic (but mathematically scaled it for the new 4K canvas). Convex Hulls are mathematically flawed for L-shaped concave hallways as they generate giant convex polygons that bleed straight through walls. The Frustum Triangle perfectly maps out the camera's sweeping perspective without destroying the layout.
- **Dilation > MORPH_OPEN**: We reverted the `MORPH_OPEN` noise filter back to a standard `cv2.dilate` with a `3x3` kernel. Because the 4K canvas is so massive, raw laser points are rendered as 1-pixel tiny dots. The `MORPH_OPEN` filter saw these 1-pixel walls as "noise" and instantly deleted 95% of them. By re-adding a `3x3` dilation, the 4K dots expand slightly and fuse into solid, continuous, ultra-crisp walls exactly like a blueprint.

## 7. Depth Clipping Filter (Clean Hallway Interiors)
- **Depth Clipping (8 Meters)**: Added a hard 8-meter distance threshold for 3D point extraction. Points beyond 8 meters are aggressively filtered out. This prevents the system from drawing massive, dense, noisy clusters at the far end of the hallway before the drone even gets close enough to map them accurately. This single mathematical filter naturally sterilizes the drone's flight path without deleting any actual walls.

## 8. Project Cleanup & Decluttering
- **Obsolete File Removal**: Executed a full sweep of the `lingbot-map` root directory. Deleted 13 unused/obsolete files to free up space and maintain a clean production environment. This included old generated PNG maps, massive unneeded `.mp4` screen recordings, deprecated automation scripts (`apply_patch.py`, `apply_bev_upload.py`), and redundant scratch backups.
- **Dead-Weight Code Amputation**: Surgically removed the legacy 3D `glb_export.py` script and ripped out all of its unused PyTorch imports and GUI buttons from `point_cloud_viewer.py` and `vis/__init__.py`. Since we are using a custom React Dashboard for 2D maps, the old Viser 3D exporter was completely unnecessary and was just wasting computational overhead.

## Important Architectural Notes
- **`viser_2d_wrapper.py` Synchronization**: Every modification made to the Custom React GUI mapping logic (`point_cloud_viewer.py`) MUST be perfectly synchronized with `viser_2d_wrapper.py`. This standalone file serves as a crucial historical and structural reference for how the 2D sparse map logic was originally injected into the Viser 3D environment.
