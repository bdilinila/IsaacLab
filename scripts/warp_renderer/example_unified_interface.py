#!/usr/bin/env python3
"""Example demonstrating the unified renderer interface.

This example shows how to use WarpRenderer with RendererBase interface
(WarpRendererAdapter) and the clean initialization pattern.
"""

import argparse
from omni.isaac.lab.app import AppLauncher

# Create argparser
parser = argparse.ArgumentParser(description="Unified Renderer Interface Example")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to render")
parser.add_argument("--width", type=int, default=400, help="Image width")
parser.add_argument("--height", type=int, default=400, help="Image height")
parser.add_argument("--output_dir", type=str, default="./renderer_output", help="Output directory for images")

# Append AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports (after simulator is launched)
import os
import sys

# Add warp_renderer to path
sys.path.insert(0, os.path.dirname(__file__))

from warp_renderer_adapter import WarpRendererAdapter
from warp_renderer_adapter_cfg import WarpRendererAdapterCfg

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass


@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    """Simple scene with a robot for rendering."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


def main():
    """Main function."""
    
    # Create scene
    scene_cfg = SimpleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    print("\n" + "="*80)
    print("UNIFIED RENDERER INTERFACE EXAMPLE")
    print("="*80)
    
    # ==========================================================================
    # OCTI'S INITIALIZATION PATTERN
    # ==========================================================================
    print("\n[1] Creating renderer with configuration...")
    
    renderer_cfg = WarpRendererAdapterCfg(
        width=args_cli.width,
        height=args_cli.height,
        num_envs=args_cli.num_envs,
    )
    
    print(f"    ✓ Configuration created: {args_cli.width}x{args_cli.height}, {args_cli.num_envs} envs")
    
    print("\n[2] Instantiating renderer...")
    renderer = WarpRendererAdapter(renderer_cfg, scene=scene)
    print(f"    ✓ Renderer created: {type(renderer).__name__}")
    
    print("\n[3] Initializing renderer...")
    renderer.initialize()
    print("    ✓ Renderer initialized (underlying WarpRenderer created)")
    
    # ==========================================================================
    # WARP RENDERER PATTERN (wrapped in step())
    # ==========================================================================
    print("\n[4] Running simulation with rendering...")
    
    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    # Reset scene
    scene.reset()
    
    # Simulation loop
    num_steps = 100
    for i in range(num_steps):
        # Step the simulation
        scene.write_data_to_sim()
        simulation_app.update()
        scene.update(dt=0.01)
        
        # Render every 10 steps
        if i % 10 == 0:
            # This is the unified interface call
            # Internally it does: update() + render()
            renderer.step()
            
            # Save image
            output_path = os.path.join(args_cli.output_dir, f"frame_{i:04d}.png")
            renderer.save_image(output_path)
            print(f"    ✓ Rendered and saved frame {i}/{num_steps}")
    
    # ==========================================================================
    # GETTING OUTPUT DATA
    # ==========================================================================
    print("\n[5] Accessing output data...")
    output_data = renderer.get_output()
    print(f"    ✓ Available data types: {list(output_data.keys())}")
    if "rgb" in output_data and output_data["rgb"] is not None:
        print(f"    ✓ RGB data shape: {output_data['rgb'].shape}")
    
    # ==========================================================================
    # ADVANCED: ACCESS UNDERLYING RENDERER
    # ==========================================================================
    print("\n[6] Advanced: Accessing underlying WarpRenderer...")
    underlying_renderer = renderer.warp_renderer
    print(f"    ✓ Underlying renderer type: {type(underlying_renderer).__name__}")
    print(f"    ✓ Can still use: update(), render(), save_image()")
    
    print("\n" + "="*80)
    print("COMPARISON: TWO WAYS TO USE WARP RENDERER")
    print("="*80)
    
    print("\n# 1. LEGACY WAY (direct instantiation)")
    print("from warp_convert import WarpRenderer")
    print("renderer = WarpRenderer(scene, width=400, height=400)")
    print("renderer.update()")
    print("renderer.render()")
    
    print("\n# 2. UNIFIED WAY (adapter pattern) ⭐ RECOMMENDED")
    print("from warp_renderer_adapter import WarpRendererAdapter")
    print("from warp_renderer_adapter_cfg import WarpRendererAdapterCfg")
    print("cfg = WarpRendererAdapterCfg(width=400, height=400, num_envs=32)")
    print("renderer = WarpRendererAdapter(cfg, scene=scene)")
    print("renderer.initialize()")
    print("renderer.step()  # Internally: update() + render()")
    
    print("\n# 3. NEWTON WAY (new architecture)")
    print("from isaaclab.renderer import get_renderer_class, NewtonWarpRendererCfg")
    print("cfg = NewtonWarpRendererCfg(renderer_type='newton_warp', ...)")
    print("renderer_cls = get_renderer_class('newton_warp')")
    print("renderer = renderer_cls(cfg)")
    print("renderer.initialize()")
    print("renderer.render(camera_positions, camera_orientations, intrinsic_matrices)")
    
    print("\n" + "="*80)
    print(f"✅ SUCCESS! Check output images in: {args_cli.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
