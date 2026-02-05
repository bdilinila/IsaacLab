"""Custom RSL-RL runner that integrates Warp Renderer."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from rsl_rl.runners import OnPolicyRunner

if TYPE_CHECKING:
    from isaaclab.renderer import RendererBase


class WarpRenderingRunner(OnPolicyRunner):
    """Custom OnPolicyRunner that integrates Warp Renderer during training.
    
    Supports both:
    1. warp_renderer_interface: WarpRendererAdapter with RendererBase interface
    2. warp_renderer_direct: Direct WarpRenderer instantiation
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str,
        device: str,
        warp_renderer: Any,  # RendererBase or WarpRenderer
        render_interval: int = 10,
        save_images: bool = False,
        output_dir: str | None = None,
    ):
        """Initialize the Warp Rendering Runner.
        
        Args:
            env: The vectorized environment
            train_cfg: Training configuration dictionary
            log_dir: Directory for logging
            device: Device to use for training
            warp_renderer: The renderer instance (WarpRendererAdapter or WarpRenderer)
            render_interval: Render every N training steps
            save_images: Whether to save rendered images to disk
            output_dir: Directory to save images (if save_images=True)
        """
        # Store Warp renderer settings
        self.warp_renderer = warp_renderer
        self.render_interval = render_interval
        self.save_images = save_images
        self.output_dir = output_dir
        self.render_step = 0
        self.env = env

        # Detect renderer type
        # warp_renderer_interface: has step() from RendererBase
        # warp_renderer_direct: has update() and render() but not step()
        self.is_interface_renderer = hasattr(warp_renderer, 'step') and hasattr(warp_renderer, 'initialize')
        self.is_direct_renderer = hasattr(warp_renderer, 'update') and hasattr(warp_renderer, 'render') and not hasattr(warp_renderer, 'step')

        if self.save_images and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # Initialize base runner
        super().__init__(env, train_cfg, log_dir, device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Override learn to add Warp rendering calls.
        
        We inject rendering calls after environment steps.
        This is done by wrapping env.step() rather than alg.step().
        """
        print(f"[VERBOSE] WarpRenderingRunner.learn() called with {num_learning_iterations} iterations", flush=True)
        # Store the original env.step function
        original_env_step = self.env.step

        step_counter = [0]  # Use list to allow modification in nested function
        
        def step_with_rendering(actions):
            step_counter[0] += 1
            if step_counter[0] % 100 == 1:  # Log every 100 steps
                print(f"[VERBOSE] Environment step {step_counter[0]}", flush=True)
            
            # Call original environment step
            if step_counter[0] <= 5:  # Log first 5 steps in detail
                print(f"[VERBOSE] Calling original_env_step for step {step_counter[0]}...", flush=True)
            obs, rewards, dones, extras = original_env_step(actions)
            if step_counter[0] <= 5:
                print(f"[VERBOSE] original_env_step returned for step {step_counter[0]}", flush=True)
            
            # Render at specified intervals
            if self.render_step % self.render_interval == 0:
                if step_counter[0] <= 5 or step_counter[0] % 100 == 0:
                    print(f"[VERBOSE] Rendering at step {step_counter[0]} (render_step={self.render_step})", flush=True)
                
                if self.is_interface_renderer:
                    # RendererBase interface: step() does update + render
                    if step_counter[0] <= 5 or step_counter[0] % 100 == 0:
                        print(f"[VERBOSE] Calling warp_renderer.step()...", flush=True)
                    self.warp_renderer.step()
                    if step_counter[0] <= 5 or step_counter[0] % 100 == 0:
                        print(f"[VERBOSE] warp_renderer.step() completed", flush=True)
                    
                    if self.save_images and self.output_dir:
                        image_path = os.path.join(self.output_dir, f"render_{self.render_step:06d}.png")
                        # Check if it has save_image method
                        if hasattr(self.warp_renderer, 'save_image'):
                            self.warp_renderer.save_image(image_path)
                elif self.is_direct_renderer:
                    # Direct WarpRenderer interface: update() + render()
                    if step_counter[0] <= 5 or step_counter[0] % 100 == 0:
                        print(f"[VERBOSE] Calling warp_renderer.update()...", flush=True)
                    self.warp_renderer.update()
                    if step_counter[0] <= 5 or step_counter[0] % 100 == 0:
                        print(f"[VERBOSE] Calling warp_renderer.render()...", flush=True)
                    self.warp_renderer.render()
                    if step_counter[0] <= 5 or step_counter[0] % 100 == 0:
                        print(f"[VERBOSE] warp_renderer.render() completed", flush=True)
                    
                    if self.save_images and self.output_dir:
                        image_path = os.path.join(self.output_dir, f"render_{self.render_step:06d}.png")
                        self.warp_renderer.save_image(image_path)
            
            self.render_step += 1
            if step_counter[0] <= 5:
                print(f"[VERBOSE] Returning from step_with_rendering for step {step_counter[0]}", flush=True)
            return obs, rewards, dones, extras

        # Replace the env.step method
        self.env.step = step_with_rendering

        print(f"[VERBOSE] Starting parent OnPolicyRunner.learn() call...", flush=True)
        # Call the base class learn method
        result = super().learn(num_learning_iterations, init_at_random_ep_len)
        print(f"[VERBOSE] Parent OnPolicyRunner.learn() completed!", flush=True)
        
        # Restore the original step method
        self.env.step = original_env_step
        
        return result
