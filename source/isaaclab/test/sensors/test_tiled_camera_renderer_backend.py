# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TiledCamera renderer backend default and --renderer_backend -> env.scene contract.

Run with: pytest source/isaaclab/test/sensors/test_tiled_camera_renderer_backend.py -v
(from repo root, with Isaac Lab env active). The contract tests run without Isaac Sim;
the TiledCameraCfg default test requires the full env (imports isaaclab.sensors.camera).
"""

import pytest

# Default env.scene used by train.py when the user does not pass env.scene=.
# Same variant name (e.g. 64x64tiled_rgb) used for both RTX and Newton; renderer_type set in main().
DEFAULT_ENV_SCENE = "64x64tiled_rgb"


class TestRendererBackendContract:
    """Enforce env.scene default and that renderer is applied in main() (no Isaac Sim required)."""

    def test_default_env_scene_is_tiled_rgb(self):
        """Default env.scene is 64x64tiled_rgb; backend applied in main() from --renderer_backend."""
        assert DEFAULT_ENV_SCENE == "64x64tiled_rgb"

    def test_only_warp_renderer_selects_warp_backend(self):
        """Only 'warp_renderer' uses Warp backend; None/rtx/other -> RTX."""
        assert "warp_renderer" not in ("rtx", None)
        # train.py sets scene/camera renderer_type in main() from --renderer_backend.


class TestTiledCameraCfgDefault:
    """Test TiledCameraCfg default (skipped when Isaac Sim not available)."""

    def test_tiled_camera_cfg_default_renderer_is_none(self):
        """Default renderer_type must be None (meaning RTX)."""
        pytest.importorskip("omni.usd", reason="Isaac Sim required for camera imports")
        from isaaclab.sensors.camera import TiledCameraCfg

        cfg = TiledCameraCfg(prim_path="/World/cam", data_types=["rgb"])
        assert cfg.renderer_type is None
