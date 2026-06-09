"""RL helpers for Decart robotics examples."""

from oasis_demo.rl.reward import DepthCollisionReward

__all__ = [
    "DepthCollisionReward",
    "ThreeCameraFeatureExtractor",
    "load_pretrained_policy",
]


def __getattr__(name: str):
    if name == "ThreeCameraFeatureExtractor":
        from oasis_demo.rl import sb3

        return getattr(sb3, name)
    if name == "load_pretrained_policy":
        from oasis_demo.rl.pretrained import load_pretrained_policy

        return load_pretrained_policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
