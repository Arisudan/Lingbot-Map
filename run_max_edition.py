import os
import sys
import subprocess

def main():
    print("==========================================================")
    print("       LingBot-MAP Launcher: MAX EDITION (GPU)")
    print("==========================================================")
    print("Optimizing backend parameters for high-end / GPU execution...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    demo_script = os.path.join(project_root, "demo.py")
    
    # Max Edition presets:
    # - --stride 3: Denser frame intervals for higher mapping fidelity
    # - --image_size 518: Standard high-resolution processing
    # - Runs natively on GPU (CUDA) automatically
    cmd = [
        sys.executable,
        demo_script,
        "--model_path", "lingbot-map.pt",
        "--video_path", "100MEDIA/DJI_0050.MP4",
        "--stride", "3",
        "--image_size", "518"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred during execution: {e}")

if __name__ == '__main__':
    main()
