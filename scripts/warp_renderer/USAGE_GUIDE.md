# Warp Renderer Usage Guide

## Quick Start: Choosing a Renderer Backend

When calling `train_with_warp_render.py`, you can choose between two renderer backends:

### 1. Newton Warp Renderer (Default, Recommended) ✅

```bash
python scripts/warp_renderer/train_with_warp_render.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --num_envs=2048 \
    --headless
```

Or explicitly specify it:
```bash
python scripts/warp_renderer/train_with_warp_render.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend newton_warp \
    --num_envs=2048 \
    --headless
```

**Why use this:**
- Proper architecture from `newton/dexsuite_warp_rendering` branch
- Clean renderer interface with lifecycle management
- Full camera intrinsics support
- Better integration with Isaac Lab

### 2. Legacy Warp Renderer (Daniela's Original)

```bash
python scripts/warp_renderer/train_with_warp_render.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend warp_renderer \
    --num_envs=2048 \
    --headless
```

**Why use this:**
- Backward compatibility with existing code
- Simpler, more direct implementation
- Testing and comparison purposes

## Complete Examples

### Example 1: Training with Newton Warp + Image Saving

```bash
python scripts/warp_renderer/train_with_warp_render.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend newton_warp \
    --num_envs=2048 \
    --headless \
    --save_images \
    --render_interval 100 \
    --image_width 400 \
    --image_height 400
```

### Example 2: Training with Legacy Renderer

```bash
python scripts/warp_renderer/train_with_warp_render.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend warp_renderer \
    --num_envs=1024 \
    --headless \
    --save_images
```

### Example 3: Using train.py Directly (Advanced)

```bash
# With Newton Warp
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend newton_warp \
    --num_envs=2048 \
    --headless

# With Legacy Warp
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend warp_renderer \
    --num_envs=2048 \
    --headless

# No custom renderer (default Isaac Sim)
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --num_envs=2048 \
    --headless
```

## Renderer Backend Comparison

| Feature | `newton_warp` | `warp_renderer` |
|---------|---------------|-----------------|
| Architecture | Proper renderer module | Direct implementation |
| Source | newton/dexsuite_warp_rendering | Daniela's warp_convert.py |
| Interface | `render(pos, quat, intrinsics)` | `update() + render()` |
| Camera Intrinsics | ✅ Full support | ⚠️ Limited |
| Data Types | RGB, Depth, more | RGB mainly |
| Lifecycle | `init -> render -> close` | Simple |
| Recommended | ✅ Yes | ⚠️ Legacy only |

## Troubleshooting

### Issue: "Failed to load renderer class"
**Solution**: Make sure you're using `newton_warp` backend and that the renderer module is properly installed.

### Issue: "No module named 'warp_convert'"
**Solution**: The legacy `warp_renderer` backend needs `warp_convert.py` in the same directory.

### Issue: "Camera sensor not found in scene"
**Solution**: Make sure your task has cameras enabled with `--enable_cameras` flag (automatically set when using renderer backends).

## Performance Tips

1. **Render Interval**: Use `--render_interval 100` or higher to reduce rendering overhead
2. **Image Size**: Smaller images (200x200) render faster than large ones (800x800)
3. **Save Images Sparingly**: Only use `--save_images` for debugging, not production training

## See Also

- **README.md**: Complete architecture overview
- **INTEGRATION_SUMMARY.md**: Technical details of the integration
- **example.py**: Standalone rendering example
