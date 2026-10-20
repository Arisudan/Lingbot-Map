import os
import sys
import pickle
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="LingBot-MAP API Server")

# Allow CORS for React frontend (Vite default is port 5173 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state mirroring the visualizer (fallback when run standalone)
state = {
    "pcs": {},
    "all_steps": [],
    "num_frames": 0,
    "original_images": [],
    "cam_dict": None,
    "bev_x_min": -10.0,
    "bev_x_max": 10.0,
    "bev_z_min": -10.0,
    "bev_z_max": 10.0,
    "vis_threshold": 3.0,
    "bev_stride": 3,
}

# Dynamic GUI Alignment offsets
alignment = {
    "offset_x": 0.0,
    "offset_z": 0.0,
    "offset_y": 0.0,
    "scale": 1.0,
    "rotation": 0.0,
}

# Global viewer reference for in-memory execution inside demo.py
viewer_instance = None

# Try loading predictions.pkl if it exists in workspace root
PREDS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictions.pkl")
if os.path.exists(PREDS_PATH):
    try:
        with open(PREDS_PATH, "rb") as f:
            data = pickle.load(f)
            
        state["pcs"] = data.get("pcs", {})
        state["all_steps"] = data.get("all_steps", [])
        state["num_frames"] = len(state["all_steps"])
        state["original_images"] = data.get("original_images", [])
        state["cam_dict"] = data.get("cam_dict", None)
        state["bev_x_min"] = data.get("bev_x_min", -10.0)
        state["bev_x_max"] = data.get("bev_x_max", 10.0)
        state["bev_z_min"] = data.get("bev_z_min", -10.0)
        state["bev_z_max"] = data.get("bev_z_max", 10.0)
        print(f"Loaded {state['num_frames']} frames from {PREDS_PATH}")
    except Exception as e:
        print(f"Failed to load predictions.pkl: {e}")

def get_active_state():
    global viewer_instance
    if viewer_instance is not None:
        return {
            "pcs": viewer_instance.pcs,
            "all_steps": viewer_instance.all_steps,
            "num_frames": viewer_instance.num_frames,
            "original_images": viewer_instance.original_images,
            "cam_dict": viewer_instance.cam_dict,
            "bev_x_min": viewer_instance.bev_x_min,
            "bev_x_max": viewer_instance.bev_x_max,
            "bev_z_min": viewer_instance.bev_z_min,
            "bev_z_max": viewer_instance.bev_z_max,
            "vis_threshold": viewer_instance.vis_threshold,
        }
    return state

def get_ground_y_level() -> float:
    active = get_active_state()
    if active["cam_dict"] is not None and "t" in active["cam_dict"] and len(active["all_steps"]) > 0:
        ts = np.array([active["cam_dict"]["t"][s] for s in active["all_steps"]])
        return float(np.min(ts[:, 1])) - 0.2
    return -1.0

@app.get("/api/metadata")
def get_metadata():
    active = get_active_state()
    return {
        "num_frames": active["num_frames"],
        "all_steps": active["all_steps"],
        "bev_x_min": active["bev_x_min"],
        "bev_x_max": active["bev_x_max"],
        "bev_z_min": active["bev_z_min"],
        "bev_z_max": active["bev_z_max"],
        "ground_y": get_ground_y_level(),
    }

@app.get("/api/frame/{idx}")
def get_frame(idx: int):
    active = get_active_state()
    if idx < 0 or idx >= len(active["original_images"]):
        raise HTTPException(status_code=404, detail="Frame index out of range")
    
    img = active["original_images"][idx]
    if isinstance(img, np.ndarray):
        # Convert RGB to BGR for OpenCV encoding
        img_bgr = cv2.cvtColor((img * 255).astype(np.uint8) if img.dtype != np.uint8 else img, cv2.COLOR_RGB2BGR)
        _, encoded = cv2.imencode(".jpg", img_bgr)
        return Response(content=encoded.tobytes(), media_type="image/jpeg")
    
    raise HTTPException(status_code=500, detail="Frame data is not a valid image array")

@app.get("/api/pointcloud/{idx}")
def get_pointcloud(idx: int):
    active = get_active_state()
    if idx not in active["pcs"]:
        raise HTTPException(status_code=404, detail="Point cloud step not found")
        
    pc_data = active["pcs"][idx]
    pc = pc_data["pc"]
    conf = pc_data["conf"]
    valid = np.isfinite(pc).all(axis=1) & (conf > active["vis_threshold"])
    
    filtered_pc = pc[valid]
    filtered_color = pc_data["color"][valid] if "color" in pc_data else None
    
    return {
        "points": filtered_pc.tolist(),
        "colors": filtered_color.tolist() if filtered_color is not None else [],
        "camera_pos": active["cam_dict"]["t"][idx].tolist() if active["cam_dict"] is not None else [0.0, 0.0, 0.0],
        "camera_rot": active["cam_dict"]["R"][idx].tolist() if active["cam_dict"] is not None else np.eye(3).tolist(),
    }

@app.get("/api/2dmap/{idx}")
def get_2dmap(idx: int):
    global viewer_instance
    if viewer_instance is not None:
        # Use the highly-tuned, styling-correct render function directly from the viewer
        img = viewer_instance._get_2d_map_render(idx)
    else:
        # Fallback to blank image if viewer is not running
        img = np.full((2000, 2000, 3), fill_value=[34, 29, 24], dtype=np.uint8)

    H_p, W_p = img.shape[:2]

    # Apply alignment translation, scale, and rotation to the entire image at once
    center = (W_p // 2, H_p // 2)
    # Rotation angle needs to be inverted to match standard counter-clockwise rotation behavior
    rot_m = cv2.getRotationMatrix2D(center, -alignment["rotation"], alignment["scale"])
    rot_m[0, 2] += alignment["offset_x"] * 10
    rot_m[1, 2] += alignment["offset_z"] * 10

    # Ensure borderValue perfectly matches the dark mode background (BGR: 34, 29, 24)
    img = cv2.warpAffine(img, rot_m, (W_p, H_p), borderValue=[34, 29, 24])

    _, encoded = cv2.imencode(".png", img)
    return Response(content=encoded.tobytes(), media_type="image/png")

@app.get("/api/alignment")
def get_alignment():
    return alignment

@app.post("/api/alignment")
def post_alignment(
    offset_x: float = Form(...),
    offset_z: float = Form(...),
    offset_y: float = Form(...),
    scale: float = Form(...),
    rotation: float = Form(...),
):
    global alignment
    alignment["offset_x"] = offset_x
    alignment["offset_z"] = offset_z
    alignment["offset_y"] = offset_y
    alignment["scale"] = scale
    alignment["rotation"] = rotation
    return {"status": "success"}

@app.post("/api/settings")
def post_settings(
    vis_threshold: float = Form(...),
    bev_stride: int = Form(...),
):
    global viewer_instance
    if viewer_instance is not None:
        viewer_instance.vis_threshold = vis_threshold
    else:
        state["vis_threshold"] = vis_threshold
        state["bev_stride"] = bev_stride
    return {"status": "success"}

@app.post("/api/upload")
async def post_upload(file: UploadFile = File(...)):
    global viewer_instance
    if viewer_instance is None:
        raise HTTPException(status_code=400, detail="In-memory PointCloudViewer is not running. Please start via demo.py")
    
    try:
        content = await file.read()
        viewer_instance.process_uploaded_video(file.filename, content)
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Serve Frontend Built Static Files
dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "dist")
if os.path.exists(dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")

    @app.get("/")
    def read_index():
        return FileResponse(os.path.join(dist_dir, "index.html"))

    # Fallback to index.html for React Router / Client-side routing
    @app.get("/{fallback_path:path}")
    def read_fallback(fallback_path: str):
        # Ignore api queries so they return standard 404
        if fallback_path.startswith("api"):
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(os.path.join(dist_dir, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
