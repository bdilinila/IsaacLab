# Unified Renderer Interface Guide

This guide explains how to combine **Octi's interface** (`renderer_cls(cfg)` + `renderer.initialize()`) with **Daniela's interface** (`renderer.update()` + `renderer.render()`).

## Overview

We've created an **adapter pattern** that wraps Daniela's `WarpRenderer` to conform to the `RendererBase` interface. This gives you the best of both worlds:

- **Octi's clean initialization**: Configuration-based instantiation with explicit `initialize()` step
- **Daniela's rendering logic**: Proven `update()` + `render()` pattern that works with fabric transforms

## Three Renderer Options

### 1. Newton Warp Renderer (`newton_warp`)
The new architecture using `newton.sensors.TiledCameraSensor`.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend newton_warp \
    --num_envs=32
```

**Interface:**
```python
from isaaclab.renderer import get_renderer_class, NewtonWarpRendererCfg

renderer_cfg = NewtonWarpRendererCfg(
    renderer_type="newton_warp",
    width=400,
    height=400,
    num_envs=32,
    num_cameras=1,
)

renderer_cls = get_renderer_class("newton_warp")
renderer = renderer_cls(renderer_cfg)
renderer.initialize()

# During training loop:
renderer.render(camera_positions, camera_orientations, intrinsic_matrices)
```

### 2. Legacy Warp Renderer (`warp_renderer`)
Daniela's original implementation with direct instantiation.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend warp_renderer \
    --num_envs=32
```

**Interface:**
```python
from warp_convert import WarpRenderer

renderer = WarpRenderer(scene, width=400, height=400)

# During training loop:
renderer.update()  # Update transforms from fabric
renderer.render()  # Render the scene
renderer.save_image("output.png")  # Optional: save image
```

### 3. Daniela Warp Unified (`daniela_warp_unified`) ⭐ **NEW**
Daniela's renderer wrapped with the unified `RendererBase` interface.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
    --renderer_backend daniela_warp_unified \
    --num_envs=32
```

**Interface:**
```python
from daniela_warp_renderer import DanielaWarpRenderer
from daniela_warp_renderer_cfg import DanielaWarpRendererCfg

# Octi's initialization pattern
renderer_cfg = DanielaWarpRendererCfg(
    scene=scene,
    width=400,
    height=400,
    num_envs=32,
)

renderer = DanielaWarpRenderer(renderer_cfg)
renderer.initialize()

# During training loop:
renderer.step()  # Internally calls update() + render()
renderer.save_image("output.png")  # Optional: save image

# Access underlying WarpRenderer if needed:
underlying = renderer.warp_renderer
```

## How the Adapter Works

The `DanielaWarpRenderer` adapter implements the `RendererBase` interface:

```python
class DanielaWarpRenderer(RendererBase):
    def __init__(self, cfg: DanielaWarpRendererCfg):
        super().__init__(cfg)
        self._scene = cfg.scene
        self._warp_renderer = None
    
    def initialize(self):
        # Creates the underlying WarpRenderer
        self._warp_renderer = WarpRenderer(
            scene=self._scene,
            width=self.cfg.width,
            height=self.cfg.height,
        )
    
    def step(self):
        # Combines Daniela's update() + render()
        self._warp_renderer.update()
        self._warp_renderer.render()
        self._output_data_buffers["rgb"] = self._warp_renderer.color_image
```

## Benefits of the Unified Interface

### ✅ Consistent API
All renderers follow the same initialization pattern:
```python
renderer = RendererClass(config)
renderer.initialize()
```

### ✅ Configuration-Based
Renderer settings are encapsulated in a `RendererCfg` object:
```python
@configclass
class DanielaWarpRendererCfg(RendererCfg):
    scene: InteractiveScene = MISSING
    width: int = 400
    height: int = 400
    num_envs: int = MISSING
```

### ✅ Easier to Swap
Switch between renderers by changing the config and class:
```python
# Before:
renderer = WarpRenderer(scene, 400, 400)
renderer.update()
renderer.render()

# After:
renderer = DanielaWarpRenderer(cfg)
renderer.initialize()
renderer.step()
```

### ✅ Standard Output Format
All renderers return data through `get_output()`:
```python
output = renderer.get_output()
rgb_image = output["rgb"]
```

### ✅ Backward Compatible
The adapter wraps the existing `WarpRenderer` without modifying it:
```python
# Still works!
from warp_convert import WarpRenderer
renderer = WarpRenderer(scene, 400, 400)
```

## Integration with Training

The `WarpRenderingRunner` automatically detects the renderer type:

```python
class WarpRenderingRunner(OnPolicyRunner):
    def __init__(self, ..., warp_renderer, ...):
        # Detect renderer type
        self.is_newton_renderer = ...
        self.is_unified_interface = hasattr(warp_renderer, 'step')
        self.is_legacy_renderer = hasattr(warp_renderer, 'update')
    
    def learn(self, ...):
        def step_with_rendering(actions):
            obs, rewards, dones, extras = original_env_step(actions)
            
            if self.render_step % self.render_interval == 0:
                if self.is_newton_renderer:
                    self._render_newton_warp()
                elif self.is_unified_interface:
                    self.warp_renderer.step()  # Unified interface
                else:
                    self.warp_renderer.update()  # Legacy interface
                    self.warp_renderer.render()
```

## When to Use Each Option

### Use `newton_warp` when:
- You need the latest Newton sensor architecture
- You want depth/segmentation data
- You're starting a new project

### Use `warp_renderer` when:
- You have existing code using Daniela's renderer
- You need maximum control over update/render timing
- You're debugging render issues

### Use `daniela_warp_unified` when:
- You want Daniela's proven renderer logic
- You prefer the unified interface pattern
- You want easy migration to Newton renderer later

## Migration Path

### From Legacy to Unified:

**Before:**
```python
from warp_convert import WarpRenderer

renderer = WarpRenderer(scene, 400, 400)

# Training loop
renderer.update()
renderer.render()
```

**After:**
```python
from daniela_warp_renderer import DanielaWarpRenderer
from daniela_warp_renderer_cfg import DanielaWarpRendererCfg

cfg = DanielaWarpRendererCfg(
    scene=scene,
    width=400,
    height=400,
    num_envs=32,
)
renderer = DanielaWarpRenderer(cfg)
renderer.initialize()

# Training loop
renderer.step()
```

### From Unified to Newton:

**Before:**
```python
from daniela_warp_renderer import DanielaWarpRenderer
from daniela_warp_renderer_cfg import DanielaWarpRendererCfg

cfg = DanielaWarpRendererCfg(scene=scene, width=400, height=400, num_envs=32)
renderer = DanielaWarpRenderer(cfg)
renderer.initialize()
renderer.step()
```

**After:**
```python
from isaaclab.renderer import get_renderer_class, NewtonWarpRendererCfg

cfg = NewtonWarpRendererCfg(
    renderer_type="newton_warp",
    width=400,
    height=400,
    num_envs=32,
)
renderer_cls = get_renderer_class("newton_warp")
renderer = renderer_cls(cfg)
renderer.initialize()
renderer.render(camera_positions, camera_orientations, intrinsic_matrices)
```

## Summary

**Yes, it's absolutely possible to combine Octi's interface with Daniela's!** The adapter pattern gives you:

1. **Octi's clean initialization**: `renderer_cls(cfg)` + `renderer.initialize()`
2. **Daniela's proven logic**: `update()` + `render()` wrapped in `step()`
3. **Easy migration path**: Start with unified, move to Newton when ready
4. **Backward compatibility**: Legacy code still works

Choose the option that best fits your workflow! 🚀
