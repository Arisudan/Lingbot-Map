import os
import sys
import subprocess

def main():
    print("==========================================================")
    print("       LingBot-MAP Launcher: BASE EDITION (CPU)")
    print("==========================================================")
    print("Optimizing backend parameters for low-end / CPU execution...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    demo_script = os.path.join(project_root, "demo.py")
    
    # Base Edition presets optimized for low-end laptops (saves massive RAM):
    # - --use_sdpa: Enforces standard PyTorch scaled-dot-product attention (no FlashInfer required)
    # - --stride 30: Skips frames extensively to process only ~100 frames, reducing CPU load and RAM usage by 5x
    # - --image_size 256: Reduces spatial resolution of frames for a tiny memory footprint (saves another 2x RAM)
    # - --max_frame_num 100: Hard limit on processed frames to prevent memory paging/lag
    # - --offload_to_cpu: Bypasses GPU memory allocation entirely
    cmd = [
        sys.executable,
        demo_script,
        "--model_path", "lingbot-map.pt",
        "--video_path", "100MEDIA/DJI_0050.MP4",
        "--use_sdpa",
        "--stride", "30",
        "--image_size", "256",
        "--max_frame_num", "100",
        "--offload_to_cpu"
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
