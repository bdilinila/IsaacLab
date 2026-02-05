# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.scene import InteractiveScene
from isaaclab.renderer import RendererCfg
from isaaclab.utils import configclass


@configclass
class WarpRendererAdapterCfg(RendererCfg):
    """Configuration for WarpRenderer Adapter (using RendererBase interface).
    
    Note: The scene is NOT stored in the config to avoid pickling issues.
    Pass it separately to the WarpRendererAdapter constructor.
    """
    
    class_name: str = "WarpRendererAdapter"
    """The renderer class name."""
    
    width: int = 400
    """The width of the rendered image."""
    
    height: int = 400
    """The height of the rendered image."""
    
    num_envs: int = MISSING
    """The number of environments to render."""
