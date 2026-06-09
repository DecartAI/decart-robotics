"""Oasis demo: RL examples and notebook visualization built on the decart-oasis SDK.

This package holds everything that is *not* the core Oasis connection — the depth-based collision
reward, the Gymnasium environment, the Stable-Baselines3 policy and behavior cloning, the live
notebook preview, and download helpers. It depends on the ``decart-oasis`` SDK (``A2VClient``).
The submodules pull heavy dependencies (torch, stable-baselines3, transformers, IPython), so they
are imported lazily rather than at package import time.
"""

__all__ = ["LiveCameraPreview"]


def __getattr__(name: str):
    if name == "LiveCameraPreview":
        from oasis_demo.live_preview import LiveCameraPreview

        return LiveCameraPreview
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
