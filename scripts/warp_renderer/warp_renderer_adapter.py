# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter to wrap WarpRenderer with RendererBase interface."""

from __future__ import annotations

from isaaclab.renderer import RendererBase

# Use absolute imports to avoid relative import issues
from warp_renderer_adapter_cfg import WarpRendererAdapterCfg
from warp_convert import WarpRenderer


class WarpRendererAdapter(RendererBase):
    """Adapter that wraps WarpRenderer to conform to RendererBase interface.
    
    This allows using WarpRenderer with the cleaner initialization pattern:
        renderer = WarpRendererAdapter(cfg)
        renderer.initialize()
        
    While internally using WarpRenderer's update/render pattern:
        renderer.update()  # Updates transforms from fabric
        renderer.render()   # Renders the scene
    """

    def __init__(self, cfg: WarpRendererAdapterCfg, scene):
        """Initialize the adapter.
        
        Args:
            cfg: Configuration for Warp renderer adapter.
            scene: The InteractiveScene to render (passed separately to avoid pickling issues).
        """
        print(f"[VERBOSE] WarpRendererAdapter.__init__ starting...", flush=True)
        super().__init__(cfg)
        self.cfg: WarpRendererAdapterCfg = cfg
        
        # Store scene reference for initialization (not in config to avoid deepcopy issues)
        self._scene = scene
        self._initialized = False
        
        # The underlying WarpRenderer (created in initialize())
        self._warp_renderer: WarpRenderer | None = None
        print(f"[VERBOSE] WarpRendererAdapter.__init__ complete", flush=True)

    def initialize(self):
        """Initialize the underlying WarpRenderer.
        
        This creates the WarpRenderer and sets up the rendering context.
        """
        print(f"[VERBOSE] WarpRendererAdapter.initialize() called...", flush=True)
        if self._initialized:
            print(f"[VERBOSE] Already initialized, skipping", flush=True)
            return
        
        print(f"[VERBOSE] Creating underlying WarpRenderer...", flush=True)
        # Create the underlying WarpRenderer with scene and dimensions
        self._warp_renderer = WarpRenderer(
            scene=self._scene,
            width=self.cfg.width,
            height=self.cfg.height,
        )
        print(f"[VERBOSE] WarpRenderer created successfully", flush=True)
        
        self._initialized = True
        
        # Initialize output buffer (RGB only for now)
        self._data_types = ["rgb"]
        print(f"[VERBOSE] Calling _initialize_output()...", flush=True)
        self._initialize_output()
        print(f"[VERBOSE] WarpRendererAdapter.initialize() complete!", flush=True)

    def step(self):
        """Step the renderer (update transforms + render).
        
        This combines WarpRenderer's update() and render() calls:
        - update(): Updates transforms from fabric
        - render(): Renders the scene
        """
        if not self._initialized:
            raise RuntimeError("Renderer not initialized. Call initialize() first.")
        
        print(f"[VERBOSE] WarpRendererAdapter.step() - calling update()...", flush=True)
        # Update transforms from fabric
        self._warp_renderer.update()
        print(f"[VERBOSE] WarpRendererAdapter.step() - update() completed", flush=True)
        
        print(f"[VERBOSE] WarpRendererAdapter.step() - calling render()...", flush=True)
        # Render the scene
        self._warp_renderer.render()
        print(f"[VERBOSE] WarpRendererAdapter.step() - render() completed", flush=True)
        
        # Store output in the standard format
        self._output_data_buffers["rgb"] = self._warp_renderer.color_image
        print(f"[VERBOSE] WarpRendererAdapter.step() - complete!", flush=True)

    def reset(self):
        """Reset the renderer.
        
        For WarpRenderer, this is a no-op as the renderer automatically
        updates based on the scene state.
        """
        pass

    def _initialize_output(self):
        """Initialize the output buffers."""
        # RGB output will be populated in step()
        self._output_data_buffers = {"rgb": None}

    def save_image(self, path: str):
        """Save the rendered image to disk.
        
        Args:
            path: Path to save the image file.
        """
        if not self._initialized:
            raise RuntimeError("Renderer not initialized. Call initialize() first.")
        self._warp_renderer.save_image(path)

    @property
    def warp_renderer(self) -> WarpRenderer:
        """Access to the underlying WarpRenderer for advanced usage."""
        if not self._initialized:
            raise RuntimeError("Renderer not initialized. Call initialize() first.")
        return self._warp_renderer
