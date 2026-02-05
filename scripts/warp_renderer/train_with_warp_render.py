"""Train an Isaac Lab environment with RSL-RL and Warp Renderer.

This script is a thin wrapper around the standard train.py that automatically enables
the Warp renderer backend. It provides a convenient interface for training with GPU
ray-traced rendering using WarpRenderer.

Architecture:
    - Calls scripts/reinforcement_learning/rsl_rl/train.py with --renderer_backend
    - Uses either warp_renderer_direct or warp_renderer_interface (adapter with RendererBase)
    - WarpRenderingRunner integrates rendering into the training loop
    - Both backends use the same underlying Warp ray-tracing implementation

Example usage:
    # Train with interface adapter (default, recommended)
    python scripts/warp_renderer/train_with_warp_render.py \\
        --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \\
        --num_envs=2048 \\
        --headless

    # Train with direct instantiation
    python scripts/warp_renderer/train_with_warp_render.py \\
        --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \\
        --renderer_backend warp_renderer_direct \\
        --num_envs=2048 \\
        --headless

    # Train with image saving
    python scripts/warp_renderer/train_with_warp_render.py \\
        --task=Isaac-Cartpole-v0 \\
        --num_envs=1024 \\
        --save_images \\
        --render_interval=100

Available tasks can be listed with:
    ./isaaclab.sh -p scripts/environments/list_envs.py
"""

from __future__ import annotations

import argparse
import sys
import os

# =============================================================================
# Argument Parsing
# =============================================================================

parser = argparse.ArgumentParser(
    description="Train RL agent with RSL-RL and Warp Renderer (wrapper for train.py)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="This script forwards all arguments to scripts/reinforcement_learning/rsl_rl/train.py with the specified --renderer_backend"
)

# Core training arguments
parser.add_argument("--task", type=str, required=True, help="Name of the task/environment to train")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum training iterations")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable cameras (auto-enabled with Warp)")

# Warp renderer specific arguments
parser.add_argument(
    "--renderer_backend",
    type=str,
    default="warp_renderer_interface",
    choices=["warp_renderer_direct", "warp_renderer_interface"],
    help="Renderer backend to use: 'warp_renderer_interface' (recommended, uses RendererBase) or 'warp_renderer_direct' (direct instantiation)"
)
parser.add_argument("--render_interval", type=int, default=10, help="Render every N training steps")
parser.add_argument("--save_images", action="store_true", default=False, help="Save rendered images to disk")
parser.add_argument("--image_width", type=int, default=400, help="Width of rendered images")
parser.add_argument("--image_height", type=int, default=400, help="Height of rendered images")

# RSL-RL arguments
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from")
parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment folder")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix for log directory")

# Device arguments
parser.add_argument("--device", type=str, default=None, help="Device to run training on")
parser.add_argument("--distributed", action="store_true", default=False, help="Multi-GPU training")

# Parse arguments
args, unknown_args = parser.parse_known_args()

# =============================================================================
# Forward to standard train.py with Warp renderer enabled
# =============================================================================

# Construct the path to the standard train.py
script_dir = os.path.dirname(os.path.abspath(__file__))
train_script = os.path.join(script_dir, "..", "reinforcement_learning", "rsl_rl", "train.py")

# Build the command with all arguments forwarded
train_args = [
    sys.executable,  # Python interpreter
    train_script,
    "--renderer_backend", args.renderer_backend,  # Use specified renderer backend
]

# Forward standard training arguments
if args.task:
    train_args.extend(["--task", args.task])
if args.num_envs is not None:
    train_args.extend(["--num_envs", str(args.num_envs)])
if args.seed is not None:
    train_args.extend(["--seed", str(args.seed)])
if args.max_iterations is not None:
    train_args.extend(["--max_iterations", str(args.max_iterations)])
if args.headless:
    train_args.append("--headless")
if args.enable_cameras:
    train_args.append("--enable_cameras")

# Forward Warp renderer arguments
train_args.extend(["--render_interval", str(args.render_interval)])
train_args.extend(["--image_width", str(args.image_width)])
train_args.extend(["--image_height", str(args.image_height)])
if args.save_images:
    train_args.append("--save_images")

# Forward RSL-RL arguments
if args.agent:
    train_args.extend(["--agent", args.agent])
if args.resume:
    train_args.append("--resume")
if args.load_run:
    train_args.extend(["--load_run", args.load_run])
if args.checkpoint:
    train_args.extend(["--checkpoint", args.checkpoint])
if args.experiment_name:
    train_args.extend(["--experiment_name", args.experiment_name])
if args.run_name:
    train_args.extend(["--run_name", args.run_name])

# Forward device arguments
if args.device:
    train_args.extend(["--device", args.device])
if args.distributed:
    train_args.append("--distributed")

# Forward any unknown arguments
train_args.extend(unknown_args)

# Print the command for debugging
print(f"[INFO] Forwarding to train.py with renderer backend: {args.renderer_backend}")
print(f"       {' '.join(train_args)}")
print()

# Execute the train.py script by replacing current process
# This ensures all output flows through correctly
os.execvp(sys.executable, train_args)
