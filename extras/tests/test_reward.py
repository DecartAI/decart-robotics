from __future__ import annotations

import numpy as np
import pytest
from oasis_demo.rl import reward as reward_module
from oasis_demo.rl.reward import DepthCollisionReward


def _constant(value: float):
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        return np.full(frame.shape[:2], value, dtype=np.float32)

    return estimate_depth


def _gradient(self, frame: np.ndarray) -> np.ndarray:
    return np.linspace(0.0, 1.0, num=frame.shape[0] * frame.shape[1], dtype=np.float32).reshape(
        frame.shape[:2]
    )


def _bottom_near(self, frame: np.ndarray) -> np.ndarray:
    depth = np.zeros(frame.shape[:2], dtype=np.float32)
    depth[-2:, :] = 1.0
    return depth


def _top_and_bottom_near(self, frame: np.ndarray) -> np.ndarray:
    depth = np.zeros(frame.shape[:2], dtype=np.float32)
    depth[:2, :] = 0.95
    depth[-1:, :] = 1.0
    return depth


def test_reward_prefers_forward_motion_without_collision(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _constant(1.0))
    reward_fn = DepthCollisionReward()
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    action = np.array([[0.5, 0.0]] * 4, dtype=np.float32)

    reward, terminated, info = reward_fn(frames, action)

    assert reward > 0
    assert not terminated
    assert info["collision_risk"] == 0.0


def test_reward_ignores_configured_bottom_depth_region_for_collision_risk(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _bottom_near)
    monkeypatch.setattr(reward_module, "IGNORE_BOTTOM_FRACTION", 0.25)
    reward_fn = DepthCollisionReward()
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    action = np.zeros((4, 2), dtype=np.float32)

    reward, terminated, info = reward_fn(frames, action)

    assert reward == 0.0
    assert not terminated
    assert info["collision_risk"] == 0.0
    assert info["depth_ignore_bottom_fraction"] == 0.25


def test_reward_uses_bottom_depth_region_when_not_ignored(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _bottom_near)
    monkeypatch.setattr(reward_module, "IGNORE_BOTTOM_FRACTION", 0.0)
    reward_fn = DepthCollisionReward()
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    action = np.zeros((4, 2), dtype=np.float32)

    _, _, info = reward_fn(frames, action)

    assert info["collision_risk"] > 0.0


def test_reward_scores_only_latest_frame(monkeypatch):
    seen_means: list[float] = []

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        seen_means.append(float(np.mean(frame)))
        return np.full(frame.shape[:2], 1.0, dtype=np.float32)

    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", estimate_depth)
    reward_fn = DepthCollisionReward(debug=True)
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in range(4)]
    action = np.zeros((4, 2), dtype=np.float32)

    _, _, info = reward_fn(frames, action)

    assert seen_means == [3.0]
    assert len(info["debug"]["depth_maps"]) == 1


def test_reward_debug_includes_normalized_depth_maps(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _gradient)
    reward_fn = DepthCollisionReward(debug=True)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    action = np.zeros((4, 2), dtype=np.float32)

    _, _, info = reward_fn(frames, action)

    depth_maps = info["debug"]["depth_maps"]
    assert len(depth_maps) == 1
    assert depth_maps[0].shape == (8, 8)
    assert np.min(depth_maps[0]) == 0.0
    assert np.max(depth_maps[0]) == 1.0


def test_reward_terminates_on_collision_risk(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _top_and_bottom_near)
    monkeypatch.setattr(reward_module, "IGNORE_BOTTOM_FRACTION", 0.125)
    reward_fn = DepthCollisionReward()
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    action = np.zeros((4, 2), dtype=np.float32)

    _, terminated, info = reward_fn(frames, action)

    assert terminated is True
    assert info["collision_risk"] > 0


def test_collision_risk_matches_call_path(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _bottom_near)
    monkeypatch.setattr(reward_module, "IGNORE_BOTTOM_FRACTION", 0.0)
    reward_fn = DepthCollisionReward()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    action = np.zeros((4, 2), dtype=np.float32)

    _, _, info = reward_fn([frame], action)

    assert reward_fn.collision_risk(frame) == info["collision_risk"]


def test_reward_rejects_bad_action_shape():
    reward_fn = DepthCollisionReward()

    with pytest.raises(ValueError, match="shape"):
        reward_fn([np.zeros((8, 8, 3), dtype=np.uint8)], np.zeros((2, 2), dtype=np.float32))


def test_display_depth_marks_ignored_region(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _constant(0.5))
    captured: dict = {}

    def fake_display(frames, depth_maps, **kwargs):
        captured["frames"] = frames
        captured["depth_maps"] = depth_maps
        captured["kwargs"] = kwargs

    monkeypatch.setattr(reward_module, "display_depth_estimates", fake_display)
    reward_fn = DepthCollisionReward()
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

    reward_fn.display_depth(frames)

    assert captured["frames"] is frames
    assert len(captured["depth_maps"]) == len(frames)
    fraction = captured["kwargs"]["depth_ignore_bottom_fraction"]
    assert fraction == reward_module.IGNORE_BOTTOM_FRACTION


# --- Straightness penalty (discourage circling, encourage swerving) --------------------------


def _action(throttle: float, steering: float) -> np.ndarray:
    return np.array([[throttle, steering]] * 4, dtype=np.float32)


def _frames():
    return [np.zeros((8, 8, 3), dtype=np.uint8)]


def _straightness_reward(weight: float = 0.3, free_play: float = 4.0, clamp: float = 12.0):
    # Construct with explicit knobs so these tests are independent of the tunable defaults.
    return DepthCollisionReward(
        straightness_weight=weight, heading_free_play=free_play, heading_clamp=clamp
    )


def test_straight_driving_has_no_penalty(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _constant(1.0))
    reward_fn = _straightness_reward()

    reward, _, info = reward_fn(_frames(), _action(0.5, 0.0))

    assert info["heading"] == 0.0
    assert info["straightness_penalty"] == 0.0
    # throttle reward only, no straightness penalty
    assert reward == pytest.approx(reward_module._forward_reward(0.5))


def test_modest_turn_within_free_play_is_not_penalized(monkeypatch):
    # The free-play allowance: you can accumulate up to heading_free_play net steering before any
    # penalty, so a single moderate turn (to swerve) is free.
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _constant(1.0))
    reward_fn = _straightness_reward(free_play=4.0)

    _, _, info = reward_fn(_frames(), _action(0.5, -0.5))  # heading -2.0, within free play (4.0)

    assert abs(info["heading"]) <= reward_fn.heading_free_play
    assert info["straightness_penalty"] == 0.0


def test_sustained_turn_beyond_free_play_accumulates_penalty(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _constant(1.0))
    reward_fn = _straightness_reward(weight=0.3, free_play=4.0, clamp=12.0)

    # Hard one-way turns: heading -4 (still free), -8, -12 (penalty kicks in past free play, grows).
    _, _, i1 = reward_fn(_frames(), _action(0.5, -1.0))
    _, _, i2 = reward_fn(_frames(), _action(0.5, -1.0))
    _, _, i3 = reward_fn(_frames(), _action(0.5, -1.0))

    assert i1["straightness_penalty"] == 0.0  # within free play
    assert i3["straightness_penalty"] > i2["straightness_penalty"] > 0  # ramps up past it


def test_lower_weight_yields_smaller_penalty(monkeypatch):
    # The knob actually drives the penalty: same steering, half the weight -> half the penalty.
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _constant(1.0))
    chunk = _action(0.5, -1.0)

    heavy = _straightness_reward(weight=0.3, free_play=4.0, clamp=12.0)
    light = _straightness_reward(weight=0.1, free_play=4.0, clamp=12.0)
    for _ in range(3):  # drive both past the free play with identical steering
        _, _, hi = heavy(_frames(), chunk)
        _, _, li = light(_frames(), chunk)

    assert hi["heading"] == li["heading"]  # same trajectory
    assert li["straightness_penalty"] == pytest.approx(hi["straightness_penalty"] / 3.0)


def test_balanced_swerve_cancels_penalty(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _constant(1.0))
    reward_fn = _straightness_reward()

    reward_fn(_frames(), _action(0.5, -1.0))  # turn hard left (heading -8, past free play)
    reward_fn(_frames(), _action(0.5, -1.0))
    reward_fn(_frames(), _action(0.5, 1.0))  # turn hard right by the same amount
    _, _, info = reward_fn(_frames(), _action(0.5, 1.0))

    assert info["heading"] == pytest.approx(0.0)  # net rotation cancels
    assert info["straightness_penalty"] == pytest.approx(0.0)


def test_penalty_saturates_at_the_clamp(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _constant(1.0))
    reward_fn = _straightness_reward(weight=0.3, free_play=4.0, clamp=12.0)

    info = {}
    for _ in range(20):  # keep turning hard one way well past the clamp
        _, _, info = reward_fn(_frames(), _action(0.0, 1.0))

    assert info["heading"] == reward_fn.heading_clamp
    assert info["straightness_penalty"] == pytest.approx(reward_fn.straightness_weight)


def test_reset_zeroes_heading(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _constant(1.0))
    reward_fn = DepthCollisionReward()

    reward_fn(_frames(), _action(0.0, 1.0))
    reward_fn(_frames(), _action(0.0, 1.0))
    assert reward_fn._heading != 0.0

    reward_fn.reset()
    assert reward_fn._heading == 0.0


# --- Exponential throttle reward -------------------------------------------------------------


def _reward_at_throttle(reward_fn, throttle):
    # Steering 0 the whole time, so heading stays 0 and only the throttle curve drives the reward.
    action = np.array([[throttle, 0.0]] * 4, dtype=np.float32)
    reward, _, _ = reward_fn([np.zeros((8, 8, 3), dtype=np.uint8)], action)
    return reward


def test_forward_reward_is_exponential_in_throttle(monkeypatch):
    monkeypatch.setattr(DepthCollisionReward, "estimate_depth", _constant(1.0))  # no collision
    reward_fn = DepthCollisionReward()

    assert _reward_at_throttle(reward_fn, 0.0) == 0.0
    assert _reward_at_throttle(reward_fn, 1.0) == pytest.approx(1.0)
    # Convex: half throttle earns much less than half of full throttle's reward.
    assert _reward_at_throttle(reward_fn, 0.5) < 0.5
    # Marginal reward grows with throttle: the top tenth is worth more than a mid tenth.
    top = _reward_at_throttle(reward_fn, 1.0) - _reward_at_throttle(reward_fn, 0.9)
    mid = _reward_at_throttle(reward_fn, 0.6) - _reward_at_throttle(reward_fn, 0.5)
    assert top > mid
