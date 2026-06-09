"""Reward function for the A2V-based RL demo.

The reward is deliberately bare so it is easy to read and easy to change:

1. estimate monocular depth on the latest ``front`` frame,
2. measure how much of the scene is dangerously close (the "near fraction"),
3. turn that into a collision ``risk`` in ``[0, 1]``.

The agent is rewarded for driving forward (an exponential throttle curve that favors
full throttle), and the episode terminates as soon as the collision risk saturates
(a frame is dangerously close). To tune the behavior, edit the constants below or the
body of ``DepthCollisionReward`` — there are intentionally no extra configuration knobs.

It also carries a small *straightness* penalty: the steering channel is integrated
into a running "heading" (net rotation from the start of the episode), and the
reward is docked once ``abs(heading)`` exceeds a free-play allowance. A balanced
left-then-right swerve drives the integral back toward zero (penalty cancels), and
modest turning stays within the free-play band, while continuously turning one way
pushes past it — so the cheapest way past an obstacle is to weave around it rather
than circle. Set ``STRAIGHTNESS_PENALTY_WEIGHT = 0.0`` to disable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from oasis_demo.visualization import display_depth_estimates

# transformers (the depth model) is imported lazily inside estimate_depth: importing this module,
# and constructing a DepthCollisionReward, must not drag in the heavy ML stack. visualization is
# cheap to import (numpy/PIL; it defers IPython itself), so it stays a top-level import.

# --- Reward tuning constants -------------------------------------------------
IGNORE_BOTTOM_FRACTION = 0.10  # bottom image fraction to ignore (ego vehicle hood/body)
EGO_DEPTH_PERCENTILE = 95.0  # percentile of the ignored bottom strip = ego reference depth
NEAR_DEPTH_MARGIN = 0.10  # how much closer than the ego reference counts as "near"
NEAR_FRACTION_THRESHOLD = 0.06  # frame is dangerous when this fraction of the scene is "near"
FORWARD_REWARD_SHARPNESS = 3.0  # exponential throttle-reward curvature (higher favors full)

# Straightness shaping: discourage circling/turning-off-road, encourage swerving. These are the
# defaults; DepthCollisionReward exposes them as constructor args so they can be swept per run.
STRAIGHTNESS_PENALTY_WEIGHT = 0.1  # max per-step penalty for being fully rotated; 0 disables
HEADING_FREE_PLAY = 8.0  # net steering you may accumulate before any penalty (turn freely within)
HEADING_CLAMP = 16.0  # net summed steering at which the penalty saturates (caps the heading too)


@dataclass(slots=True)
class DepthCollisionReward:
    """Reward forward motion; terminate when the front camera sees a collision risk."""

    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: int | str | None = None
    debug: bool = False
    # Straightness-penalty knobs (default to the module constants; override per run to sweep).
    straightness_weight: float = STRAIGHTNESS_PENALTY_WEIGHT
    heading_free_play: float = HEADING_FREE_PLAY
    heading_clamp: float = HEADING_CLAMP
    _depth_estimator_pipeline: Any = field(init=False, repr=False)
    _heading: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._depth_estimator_pipeline = None
        self._heading = 0.0

    def reset(self) -> None:
        """Per-episode reset hook for the env's reset/step lifecycle.

        Zeroes the integrated heading so every episode starts pointing "straight",
        keeping the straightness penalty independent across Gymnasium episodes.
        """
        self._heading = 0.0

    def __call__(
        self, front_frames: list[np.ndarray], action_chunk: np.ndarray
    ) -> tuple[float, bool, dict]:
        if not front_frames:
            raise ValueError("front_frames must contain at least one frame")
        if action_chunk.shape != (4, 2):
            raise ValueError(f"action_chunk must have shape (4, 2), got {action_chunk.shape}")

        normalized_depth = _normalize_depth(self.estimate_depth(front_frames[-1]))
        risk = _risk_from_depth(normalized_depth)
        forward = float(np.mean(np.clip(action_chunk[:, 0], 0.0, 1.0)))
        straightness_penalty = self._update_straightness(action_chunk)
        reward = _forward_reward(forward) - straightness_penalty  # exp throttle, less the turn cost
        terminated = risk >= 1.0  # risk saturates at 1.0 once a frame is dangerously close
        info: dict[str, Any] = {
            "collision_risk": risk,
            "forward": forward,
            "heading": self._heading,
            "straightness_penalty": straightness_penalty,
            "depth_ignore_bottom_fraction": IGNORE_BOTTOM_FRACTION,
        }
        if self.debug:
            info["debug"] = {"depth_maps": [normalized_depth]}
        return reward, terminated, info

    def _update_straightness(self, action_chunk: np.ndarray) -> float:
        """Integrate this chunk's steering into the heading; return the straightness penalty.

        ``heading`` is the running net rotation from the episode start (sum of signed steering,
        negative = left / positive = right), clamped so the penalty stays bounded. There is no
        penalty while ``abs(heading)`` stays within ``heading_free_play`` (turn freely, e.g. to
        swerve); beyond that it ramps linearly to ``straightness_weight`` at ``heading_clamp``. So a
        balanced swerve (net ~0) and modest turning cost nothing, while sustained one-way turning
        eventually saturates the cost.
        """
        self._heading += float(np.sum(action_chunk[:, 1]))
        self._heading = float(np.clip(self._heading, -self.heading_clamp, self.heading_clamp))
        excess = max(0.0, abs(self._heading) - self.heading_free_play)
        span = max(1e-6, self.heading_clamp - self.heading_free_play)
        return self.straightness_weight * float(np.clip(excess / span, 0.0, 1.0))

    def collision_risk(self, frame: np.ndarray) -> float:
        """Collision risk in ``[0, 1]`` for a single front frame.

        Used both by the reward and by the live preview's per-frame risk bar. ``__call__`` scores
        ``front_frames[-1]`` the same way, so the two always agree for a given frame.
        """
        return _risk_from_depth(_normalize_depth(self.estimate_depth(frame)))

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Return a single-channel depth map for an RGB frame using Depth Anything V2."""
        if self._depth_estimator_pipeline is None:
            from transformers import pipeline

            print(f"Loading depth estimation model: {self.depth_model} ...")
            self._depth_estimator_pipeline = pipeline(
                "depth-estimation", model=self.depth_model, device=self.device
            )
        result = self._depth_estimator_pipeline(Image.fromarray(frame.astype(np.uint8), mode="RGB"))
        depth = np.asarray(result["depth"], dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth estimator must return a 2D map, got shape {depth.shape}")
        return depth

    def display_depth(self, frames: list[np.ndarray]) -> None:
        """Show the frames and their depth maps, marking the ignored ego region."""
        depth_maps = [self.estimate_depth(frame) for frame in frames]
        display_depth_estimates(
            frames, depth_maps, depth_ignore_bottom_fraction=IGNORE_BOTTOM_FRACTION
        )


def _forward_reward(forward: float) -> float:
    """Exponential (convex) throttle reward mapping ``[0, 1]`` -> ``[0, 1]``.

    ``0`` at no throttle and ``1`` at full throttle, but convex in between, so each increment of
    throttle is worth more than the last and full throttle is rewarded disproportionately. Bounded
    and non-negative, so the reward stays on the same scale as the old linear term and crashing to
    end the episode early is never made attractive. ``FORWARD_REWARD_SHARPNESS`` sets the curvature.
    """
    return float(np.expm1(FORWARD_REWARD_SHARPNESS * forward) / np.expm1(FORWARD_REWARD_SHARPNESS))


def _risk_from_depth(normalized: np.ndarray) -> float:
    """Collision risk in ``[0, 1]`` from a normalized depth map.

    The fraction of the scored region (the frame minus the bottom ego strip) closer than the ego
    reference depth, scaled so it saturates at 1.0.
    """
    scored_depth, ego_depth = _split_bottom(normalized, IGNORE_BOTTOM_FRACTION)
    ego_reference = float(np.percentile(ego_depth, EGO_DEPTH_PERCENTILE))
    near_cutoff = max(0.0, ego_reference - NEAR_DEPTH_MARGIN)
    near_fraction = float(np.mean(scored_depth > near_cutoff))
    return float(np.clip(near_fraction / NEAR_FRACTION_THRESHOLD, 0.0, 1.0))


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    min_depth = float(np.min(finite))
    max_depth = float(np.max(finite))
    if max_depth <= min_depth:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - min_depth) / (max_depth - min_depth)).astype(np.float32)


def _split_bottom(depth: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if fraction <= 0.0:
        return depth, depth
    cutoff = max(1, int(round(depth.shape[0] * (1.0 - fraction))))
    cutoff = min(cutoff, depth.shape[0] - 1)
    return depth[:cutoff, :], depth[cutoff:, :]
