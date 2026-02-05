# Warp Renderer Integration for Isaac Lab

This module integrates Newton's Warp Renderer for GPU ray-traced rendering during training.

## Architecture Overview

### Renderer Module (`isaaclab/renderer/`)

The renderer module provides a clean architecture for different rendering backends:

- **`RendererBase`**: Abstract base class defining the renderer interface
- **`NewtonWarpRenderer`**: Implementation using Newton's TiledCameraSensor for GPU ray-tracing
- **`get_renderer_class()`**: Registry pattern for lazy-loading renderer classes

**Lifecycle**: `__init__() -> initialize() -> render() -> close()`

### Training Integration

**`WarpRenderingRunner`**: Custom RSL-RL runner that integrates rendering into the training loop
- Inherits from `OnPolicyRunner`
- Calls renderer at specified intervals during training
- Supports both NewtonWarpRenderer and legacy WarpRenderer

### Usage

#### Option 1: Using the Wrapper Script (Recommended)

**With Newton Warp Renderer (default, proper architecture):**
```bash
python scripts/warp_renderer/train_with_warp_render.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --num_envs=2048 \
    --headless \
    --save_images \
    --render_interval=100
```

**With Legacy Warp Renderer (Daniela's original warp_convert.py):**
```bash
python scripts/warp_renderer/train_with_warp_render.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend warp_renderer \
    --num_envs=2048 \
    --headless \
    --save_images
```

#### Option 2: Using train.py Directly

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend newton_warp \
    --num_envs=2048 \
    --headless \
    --image_width 400 \
    --image_height 400
```

#### Option 3: No Custom Renderer (Default Isaac Sim)

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --num_envs=2048 \
    --headless
```

## Renderer Backend Comparison

**IMPORTANT**: Both backends use the **same underlying Warp GPU ray tracer** (`warp_raytrace.RenderContext` from Newton). The difference is the abstraction layer.

### `newton_warp` (Default, Recommended)
- **Architecture**: Proper renderer module with clean interface
- **Implementation**: Uses Newton's `TiledCameraSensor` → `RenderContext`
- **Interface**: `renderer.render(positions, orientations, intrinsics)` 
- **Lifecycle**: `initialize() -> render() -> close()`
- **Features**: Full support for camera intrinsics, multiple data types (RGB, depth)
- **Use case**: Production training, proper integration with Isaac Lab
- **Path**: NewtonWarpRenderer → TiledCameraSensor → RenderContext → GPU

### `warp_renderer` (Legacy)
- **Architecture**: Daniela's original implementation in `warp_convert.py`
- **Implementation**: Direct `RenderContext` usage
- **Interface**: `renderer.update() -> renderer.render()`
- **Features**: Simpler, more direct approach
- **Use case**: Backward compatibility, testing
- **Path**: WarpRenderer → RenderContext → GPU

See `ARCHITECTURE_CLARIFICATION.md` for detailed explanation of the shared core.

## Command-Line Arguments

### Renderer-Specific Arguments

- `--renderer_backend`: Renderer backend to use
  - `newton_warp` (default): Proper renderer architecture from newton branch
  - `warp_renderer`: Legacy Daniela's implementation
  - `None`: No custom renderer (default Isaac Sim rendering)
- `--render_interval`: Render every N training steps (default: 10)
- `--save_images`: Save rendered images to disk
- `--image_width`: Width of rendered images (default: 400)
- `--image_height`: Height of rendered images (default: 400)

## Files

### Core Renderer Module
- `isaaclab/renderer/__init__.py`: Module exports and renderer registry
- `isaaclab/renderer/renderer.py`: Base renderer interface
- `isaaclab/renderer/renderer_cfg.py`: Base renderer configuration
- `isaaclab/renderer/newton_warp_renderer.py`: Newton Warp renderer implementation
- `isaaclab/renderer/newton_warp_renderer_cfg.py`: Newton Warp renderer configuration

### Training Integration
- `scripts/warp_renderer/warp_rendering_runner.py`: Custom RSL-RL runner with rendering
- `scripts/warp_renderer/train_with_warp_render.py`: Convenience wrapper script

### Legacy Files (Daniela's original implementation)
- `scripts/warp_renderer/warp_convert.py`: Original WarpRenderer (still supported for backward compatibility)
- `scripts/warp_renderer/example.py`: Standalone rendering example

## Branch History

This implementation combines code from two branches:

1. **`newton/dexsuite_warp_rendering`**: Provided the proper renderer architecture
   - Added `isaaclab/renderer/` module with `NewtonWarpRenderer`
   - Used registry pattern with `get_renderer_class()`
   - Based on Newton's `TiledCameraSensor`

2. **`feature/dexsuite_vision`**: More recent Isaac Lab updates (Isaac Sim 6.0)
   - Has vision environment configurations
   - Latest Isaac Lab upstream updates
   - Was missing renderer module (now copied from newton branch)

## Integration with Cameras

The renderer gets camera data from the environment's scene:
- Camera positions: `camera_sensor.data.pos_w`
- Camera orientations: `camera_sensor.data.quat_w_world`
- Intrinsic matrices: `camera_sensor.data.intrinsic_matrices`

This data is passed to `NewtonWarpRenderer.render()` which uses Newton's GPU ray-tracer to generate images.

## CNN Policy Support

For vision-based tasks, the training uses `ActorCriticCNN` which:
- Processes both 1D observations and 3D image observations
- Builds CNN encoders from configuration
- Handles dict-based CNN configs (from Hydra)
- Automatically moves models to correct device (CPU/CUDA)
