#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Output file name is required"
    echo "Usage: $0 <output_file_name>"
    exit 1
fi

OUTPUT_NAME="$1"

# RENDERER_NVTX=1 enables per-method renderer ranges (update_transforms,
# update_camera, render) and benchmark::total/train/init ranges.
# Defaults to 0 (disabled) — all NVTX ranges are no-ops.
unset NVTX_DISABLE
export RENDERER_NVTX="${RENDERER_NVTX:-0}"
export OVRTX_app_profilerBackend=nvtx
export OVRTX_app_profileFromStart=true
export OVRTX_app_profilerMask=1
# Trace options: cuda, nvtx, osrt (OptiX/ray tracing runtime), vulkan (Vulkan + Vulkan ray tracing)
export CONDA_ENV=/home/horde/miniconda/envs/my_isaaclab_env
export LD_PRELOAD=$CONDA_ENV/lib/python3.12/site-packages/ovrtx/bin/plugins/libcarb.so
export OVRTX_SKIP_USD_CHECK=1
export ISAACLAB_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$ISAACLAB_PATH/source/isaaclab_tasks:$ISAACLAB_PATH/source/isaaclab_rl:$PYTHONPATH"
# Use conda env's libstdc++ to avoid CXXABI version mismatch with the system libstdc++
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"

echo "Start: $(date)"
START=$(date +%s)

nsys profile \
    --output="$OUTPUT_NAME" \
    --trace=cuda,nvtx,osrt,vulkan \
    --sample=cpu \
    --cpuctxsw=process-tree \
    --cuda-memory-usage=true \
    --stats=true \
    --force-overwrite=true \
    $CONDA_ENV/bin/python \
        scripts/benchmarks/benchmark_rlgames.py \
        --task=Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 \
        presets=newton,ovrtx_renderer \
        --headless \
        --enable_cameras \
        --num_envs 1225 \
        --max_iterations 5

END=$(date +%s)
echo ""
echo "End: $(date)"
echo "Elapsed: $((END - START)) seconds"
