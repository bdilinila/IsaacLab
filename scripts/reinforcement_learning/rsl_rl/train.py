# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# Renderer backend arguments
parser.add_argument(
    "--renderer_backend",
    type=str,
    default=None,
    choices=["warp_renderer_direct", "warp_renderer_interface"],
    help="Renderer backend: 'warp_renderer_direct' (direct instantiation), 'warp_renderer_interface' (RendererBase interface)"
)
parser.add_argument("--render_interval", type=int, default=10, help="Render every N training steps when using custom renderer.")
parser.add_argument("--save_images", action="store_true", default=False, help="Save rendered images to disk.")
parser.add_argument("--image_width", type=int, default=400, help="Width of rendered images.")
parser.add_argument("--image_height", type=int, default=400, help="Height of rendered images.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video or use custom renderer
if args_cli.video or args_cli.renderer_backend:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform
from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import time
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # Import custom renderer if needed
    warp_renderer = None
    if args_cli.renderer_backend in ["warp_renderer_direct", "warp_renderer_interface"]:
        # Import custom runner for training integration
        warp_renderer_path = os.path.join(os.path.dirname(__file__), "..", "..", "warp_renderer")
        sys.path.insert(0, warp_renderer_path)
        from warp_rendering_runner import WarpRenderingRunner
        
        # Import CNN policy for vision-based tasks
        from isaaclab_rl.rsl_rl import ActorCriticCNN
        
        if args_cli.renderer_backend == "warp_renderer_direct":
            # Direct instantiation of WarpRenderer
            from warp_convert import WarpRenderer
        elif args_cli.renderer_backend == "warp_renderer_interface":
            # WarpRenderer with RendererBase interface
            from warp_renderer_adapter import WarpRendererAdapter
            from warp_renderer_adapter_cfg import WarpRendererAdapterCfg
        
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    print(f"[VERBOSE] ============================================================")
    print(f"[VERBOSE] Creating environment at {time.strftime('%H:%M:%S')}")
    print(f"[VERBOSE] Task: {args_cli.task}")
    print(f"[VERBOSE] Num envs: {env_cfg.scene.num_envs}")
    print(f"[VERBOSE] Calling gym.make()...", flush=True)
    print(f"[VERBOSE] ============================================================")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print(f"[VERBOSE] Environment created successfully at {time.strftime('%H:%M:%S')}", flush=True)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        print(f"[VERBOSE] Converting multi-agent to single-agent...")
        env = multi_agent_to_single_agent(env)
        print(f"[VERBOSE] Conversion complete")

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    print(f"[VERBOSE] Wrapping environment with RslRlVecEnvWrapper...", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[VERBOSE] Environment wrapped successfully. Num envs: {env.unwrapped.scene.num_envs}", flush=True)

    # Initialize custom renderer if requested
    if args_cli.renderer_backend in ["warp_renderer_direct", "warp_renderer_interface"]:
        print(f"[INFO] Initializing renderer backend: {args_cli.renderer_backend} ({args_cli.image_width}x{args_cli.image_height})...")
        print(f"[VERBOSE] Starting renderer initialization at {time.strftime('%H:%M:%S')}...")
        
        if args_cli.renderer_backend == "warp_renderer_direct":
            # Direct instantiation of WarpRenderer
            warp_renderer = WarpRenderer(
                env.unwrapped.scene,
                width=args_cli.image_width,
                height=args_cli.image_height
            )
        
        elif args_cli.renderer_backend == "warp_renderer_interface":
            # WarpRenderer with RendererBase interface
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "warp_renderer"))
            from warp_renderer_adapter import WarpRendererAdapter
            from warp_renderer_adapter_cfg import WarpRendererAdapterCfg
            
            renderer_cfg = WarpRendererAdapterCfg(
                width=args_cli.image_width,
                height=args_cli.image_height,
                num_envs=env.unwrapped.scene.num_envs,
            )
            
            # Pass scene separately to avoid pickling issues
            warp_renderer = WarpRendererAdapter(renderer_cfg, scene=env.unwrapped.scene)
            warp_renderer.initialize()
        
        print(f"[INFO] {args_cli.renderer_backend} renderer initialized with {env.unwrapped.scene.num_envs} worlds")
        print(f"[VERBOSE] Renderer initialization completed at {time.strftime('%H:%M:%S')}")
        
        # Inject ActorCriticCNN into RSL-RL namespace for eval()
        print(f"[VERBOSE] Injecting ActorCriticCNN into RSL-RL namespace...")
        import rsl_rl.runners.on_policy_runner as runner_module
        if not hasattr(runner_module, 'ActorCriticCNN'):
            setattr(runner_module, 'ActorCriticCNN', ActorCriticCNN)
        
        # Set output directory for images
        output_dir = os.path.join(log_dir, "renderer_images")

    # create runner from rsl-rl
    print(f"[VERBOSE] Creating RL runner at {time.strftime('%H:%M:%S')}...")
    if args_cli.renderer_backend and agent_cfg.class_name == "OnPolicyRunner":
        # Use WarpRenderingRunner for custom renderer backend
        print(f"[VERBOSE] Using WarpRenderingRunner with renderer backend: {args_cli.renderer_backend}")
        runner = WarpRenderingRunner(
            env=env,
            train_cfg=agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
            warp_renderer=warp_renderer,
            render_interval=args_cli.render_interval,
            save_images=args_cli.save_images,
            output_dir=output_dir,
        )
        print(f"[VERBOSE] WarpRenderingRunner created successfully")
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        print(f"[VERBOSE] Using DistillationRunner")
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        print(f"[VERBOSE] Runner created successfully")
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    print(f"[VERBOSE] Adding git repo to log at {time.strftime('%H:%M:%S')}...")
    runner.add_git_repo_to_log(__file__)
    print(f"[VERBOSE] Git repo added successfully")
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    print(f"[VERBOSE] Dumping configuration to log directory at {time.strftime('%H:%M:%S')}...", flush=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"[VERBOSE] Configuration dumped successfully", flush=True)

    # run training
    print(f"[VERBOSE] ============================================================", flush=True)
    print(f"[VERBOSE] STARTING TRAINING at {time.strftime('%H:%M:%S')}", flush=True)
    print(f"[VERBOSE] Max iterations: {agent_cfg.max_iterations}", flush=True)
    print(f"[VERBOSE] ============================================================", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[VERBOSE] Training completed at {time.strftime('%H:%M:%S')}", flush=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
