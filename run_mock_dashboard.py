import os
import sys
import subprocess
import time
import numpy as np

def generate_mock_data(num_frames=30):
    t_list = []
    R_list = []
    focal_list = []
    pp_list = []
    
    # Generate an L-shaped flight path
    for i in range(num_frames):
        if i < num_frames // 2:
            tx, tz = 0.0, float(i) * 0.5
            R = np.eye(3)
        else:
            tx, tz = float(i - num_frames // 2) * 0.5, float(num_frames // 2) * 0.5
            # 90 deg turn
            R = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0]
            ])
        t_list.append(np.array([tx, 0.0, tz]))
        R_list.append(R)
        focal_list.append(300.0)
        pp_list.append((250.0, 250.0))
        
    cam_dict = {
        "t": t_list,
        "R": R_list,
        "focal": focal_list,
        "pp": pp_list
    }
    
    pc_list = []
    color_list = []
    conf_list = []
    
    for i in range(num_frames):
        t = t_list[i]
        # Wall obstacles
        left_pts = np.stack([np.full(30, -2.0), np.zeros(30), np.linspace(t[2]-1.0, t[2]+3.0, 30)], axis=1)
        right_pts = np.stack([np.full(30, 2.0), np.zeros(30), np.linspace(t[2]-1.0, t[2]+3.0, 30)], axis=1)
        pts = np.concatenate([left_pts, right_pts], axis=0)
        pts += np.random.normal(0, 0.05, pts.shape)
        
        conf = np.ones(len(pts)) * 1.5
        cols = np.ones((len(pts), 3)) * 0.9
        
        pc_list.append(pts)
        color_list.append(cols)
        conf_list.append(conf)
        
    return pc_list, color_list, conf_list, cam_dict

def main():
    print("==========================================================")
    print("         LingBot-MAP Mock Dashboard Launcher")
    print("==========================================================")
    
    # Add project root to python path so we can import properly
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # 1. Initialize PointCloudViewer with mock data
    from lingbot_map.vis.point_cloud_viewer import PointCloudViewer
    print("Generating mock flight path and obstacles...")
    pc_list, color_list, conf_list, cam_dict = generate_mock_data()
    
    viewer = PointCloudViewer(
        pc_list=pc_list,
        color_list=color_list,
        conf_list=conf_list,
        cam_dict=cam_dict,
        vis_threshold=1.0,
        downsample_factor=1,
    )
    # Populate mock camera image frames to prevent 404 on startup
    viewer.original_images = [np.zeros((300, 400, 3), dtype=np.uint8) for _ in range(len(pc_list))]
    
    # 2. Launch Vite development server in the background if possible
    has_npm = False
    try:
        subprocess.run(["npm", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        has_npm = True
    except:
        pass
        
    frontend_process = None
    if has_npm:
        print("Launching React dev server on http://localhost:5173...")
        dashboard_dir = os.path.join(project_root, "dashboard")
        if not os.path.exists(os.path.join(dashboard_dir, "node_modules")):
            subprocess.check_call("npm install", shell=True, cwd=dashboard_dir)
        frontend_process = subprocess.Popen(
            "npm run dev",
            shell=True,
            cwd=dashboard_dir
        )
    else:
        print("NPM not detected. FastAPI will serve the static build directly on http://localhost:8000")
        
    # 3. Start mock API server (blocks)
    print("Starting mock API server on http://localhost:8000...")
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if frontend_process:
            frontend_process.terminate()

if __name__ == '__main__':
    main()
