import os
import sys
import subprocess
import time

def main():
    print("==========================================================")
    print("         LingBot-MAP Web Dashboard Automation Launcher")
    print("==========================================================")

    # 1. Install FastAPI backend dependencies if missing
    print("[1/3] Checking backend dependencies...")
    try:
        import fastapi
        import uvicorn
        import multipart
    except ImportError:
        print("Installing fastapi, uvicorn, and python-multipart...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "python-multipart"])
        print("Backend dependencies installed successfully.")

    # 2. Select operational edition
    print("\nSelect the LingBot-MAP Edition to execute:")
    print("  [1] Base Edition (CPU-Only Optimization / Low-End Hardware)")
    print("  [2] Max Edition  (GPU-Enabled Optimization / High-End Hardware)")
    print("  [3] Mock Mode    (No Model / Dynamic trajectory simulation)")
    
    choice = input("\nEnter choice [1, 2, or 3]: ").strip()
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    if choice == "1":
        backend_script = os.path.join(project_root, "run_base_edition.py")
        print("\n[2/3] Launching Base Edition (CPU) backend...")
    elif choice == "2":
        backend_script = os.path.join(project_root, "run_max_edition.py")
        print("\n[2/3] Launching Max Edition (GPU) backend...")
    else:
        backend_script = os.path.join(project_root, "run_mock_dashboard.py")
        print("\n[2/3] Launching Mock Mode backend...")

    backend_process = subprocess.Popen(
        [sys.executable, backend_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Let the backend server bind to the port (starts on port 8000)
    time.sleep(3)
    if backend_process.poll() is not None:
        print("Error: Backend failed to start. Logs:")
        print(backend_process.stdout.read())
        return

    # 3. Launch Vite development frontend server if npm is installed
    has_npm = False
    try:
        subprocess.run(["npm", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        has_npm = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    frontend_process = None
    if has_npm:
        print("[3/3] Launching React+Vite dev server on port 5173...")
        dashboard_dir = os.path.join(project_root, "dashboard")
        if not os.path.exists(os.path.join(dashboard_dir, "node_modules")):
            print("node_modules not found, running npm install in dashboard...")
            subprocess.check_call("npm install", shell=True, cwd=dashboard_dir)

        frontend_process = subprocess.Popen(
            "npm run dev",
            shell=True,
            cwd=dashboard_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print("\n----------------------------------------------------------")
        print("   Dashboard is running in Development Mode!")
        print("   - API Backend: http://localhost:8000")
        print("   - Web UI Dashboard: http://localhost:5173")
        print("----------------------------------------------------------")
    else:
        print("\n[3/3] NPM/Node not detected on this machine.")
        print("      Running in STATIC PRODUCTION MODE.")
        print("      FastAPI is serving the pre-built frontend directly.")
        print("\n----------------------------------------------------------")
        print("   Dashboard is running in Production Mode!")
        print("   - URL: http://localhost:8000")
        print("----------------------------------------------------------")

    print("Press Ctrl+C to terminate the server(s)...\n")

    # Monitor processes and stream output
    try:
        while True:
            # Stream backend output
            line = backend_process.stdout.readline()
            if line:
                print(f"[Backend] {line.strip()}")
            
            # Check if any process terminated
            if backend_process.poll() is not None:
                print("Backend server terminated unexpectedly.")
                break
            if frontend_process is not None and frontend_process.poll() is not None:
                print("Frontend server terminated unexpectedly.")
                break
                
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        try:
            backend_process.terminate()
        except:
            pass
        if frontend_process:
            try:
                frontend_process.terminate()
            except:
                pass

if __name__ == '__main__':
    main()
