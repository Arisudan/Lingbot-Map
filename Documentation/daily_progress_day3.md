# Daily Progress: Day 3 (30-07-2026)

This document tracks all updates, modifications, and fixes planned and applied to the Lingbot-MAP project on Day 3.

## Today's Agenda: Base & Max Edition Deployment

Today's focus is to divide the execution of the LingBot-MAP project into two distinct operational modes to optimize performance across different hardware setups:

### 1. Base Edition (CPU-Only / Low-End Hardware)
- **Goal**: Allow users with standard laptops or computers without dedicated Nvidia GPUs to run the mapping pipeline without hitting memory limits or dependency compilation crashes.
- **Key Enhancements**:
  * Force execution on CPU.
  * Enforce Scaled Dot Product Attention (`--use_sdpa`) to bypass FlashInfer.
  * Reduce spatial resolution to `384px` to minimize RAM footprint.
  * Increase frame stride to `6` to process fewer frames and reduce inference time.

### 2. Max Edition (GPU-Enabled / High-End Hardware)
- **Goal**: Unlock the full power of dedicated CUDA-enabled GPUs for high-resolution, detailed mapping.
- **Key Enhancements**:
  * Execute on CUDA/GPU.
  * Run at high resolution (`518px`) for detailed point clouds.
  * Run with standard dense striding (`3`) for detailed flight path mapping.
  * Support compiled attention or half-precision optimizations.

### 3. Launcher Automation & Synchronization
- **Agenda**:
  * Create `run_base_edition.py` and `run_max_edition.py` python scripts to act as one-click launchers.
  * Update `run_dashboard.py` to offer an interactive CLI selection menu for both editions.
  * Update `lingbot_map/vis/GUI_2d_wrapper.py` to support the new attention (`--use_sdpa`) and hardware settings to keep it fully synchronized.
