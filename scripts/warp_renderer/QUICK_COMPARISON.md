# Quick Renderer Comparison

## TL;DR

**Daniela's `example.py` uses the LEGACY renderer** (`warp_renderer` backend).

**IMPORTANT**: Both use the **SAME Warp GPU ray tracer** from Newton! The difference is the abstraction layer:
- Newton's goes through `TiledCameraSensor` (proper architecture)
- Daniela's uses `RenderContext` directly (simpler)

---

## Visual Comparison

```
┌─────────────────────────────────────────────────────────────┐
│  DANIELA'S LEGACY RENDERER (warp_renderer)                  │
│  File: warp_convert.py                                      │
│  Used in: example.py                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  renderer = WarpRenderer(scene, 400, 400)                   │
│                                                             │
│  # In loop:                                                 │
│  renderer.update()   ← Sync with simulation                │
│  renderer.render()   ← Render current state                │
│                                                             │
│  ✅ Simple & intuitive                                      │
│  ✅ Auto scene tracking                                     │
│  ⚠️  Less flexible                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  NEWTON'S PROPER RENDERER (newton_warp)                     │
│  File: isaaclab/renderer/newton_warp_renderer.py           │
│  Used in: Training scripts                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  cfg = NewtonWarpRendererCfg(...)                           │
│  renderer = get_renderer_class("newton_warp")(cfg)         │
│  renderer.initialize()                                      │
│                                                             │
│  # In loop:                                                 │
│  renderer.render(positions, orientations, intrinsics)       │
│                         ↑                                   │
│              Explicit camera data required                  │
│                                                             │
│  ✅ Proper architecture                                     │
│  ✅ Full camera control                                     │
│  ⚠️  More complex setup                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Differences

| Feature | Legacy (Daniela) | Newton (Proper) |
|---------|------------------|-----------------|
| **File** | `warp_convert.py` | `isaaclab/renderer/newton_warp_renderer.py` |
| **Interface** | `update() + render()` | `render(pos, quat, intrinsics)` |
| **Camera Data** | Automatic | Manual/Explicit |
| **Architecture** | Standalone | Proper module with `RendererBase` |
| **Setup** | Simple | Registry + config |
| **Used By** | `example.py` | Training scripts |
| **Flexibility** | Low | High |

---

## Which Should You Use?

### Use `--renderer_backend warp_renderer` (Daniela's) for:
- 🔬 Testing Daniela's original implementation
- 📝 Following `example.py` patterns
- ⚡ Quick prototyping with simple interface

### Use `--renderer_backend newton_warp` (Default) for:
- 🏭 Production training
- 🏗️ Proper software architecture
- 🎯 Full camera control
- 📊 Multiple output types (RGB + depth)

---

## Code Example from example.py

```python
# This is what Daniela uses in example.py:

from warp_convert import WarpRenderer  # ← Legacy renderer

renderer = WarpRenderer(scene, 400, 400)

# Rendering loop (lines 82-83)
renderer.update()
renderer.render()
```

**This corresponds to `--renderer_backend warp_renderer`**

---

## How to Run Each

### Daniela's Way (Legacy):
```bash
python scripts/warp_renderer/train_with_warp_render.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend warp_renderer \
    --num_envs=2048 \
    --headless
```

### Newton's Way (Proper Architecture, Default):
```bash
python scripts/warp_renderer/train_with_warp_render.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --num_envs=2048 \
    --headless
```

Or explicitly:
```bash
python scripts/warp_renderer/train_with_warp_render.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend newton_warp \
    --num_envs=2048 \
    --headless
```
