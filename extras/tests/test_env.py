from __future__ import annotations

import numpy as np
import pytest
from decart_oasis.exceptions import DecartRoboticsError
from decart_oasis.types import A2VResult
from oasis_demo.rl.env import OasisA2VEnv
from oasis_demo.rl.reward import DepthCollisionReward


class FakeClient:
    def __init__(self) -> None:
        self.initialize_calls = 0
        self.prompt_calls = 0
        self.close_calls = 0
        self.prompt_text: str | None = None

    def initialize(self) -> None:
        self.initialize_calls += 1

    def prompt(self, prompt: str) -> None:
        self.prompt_calls += 1
        self.prompt_text = prompt

    def infer(self, actions: np.ndarray) -> A2VResult:
        frames = {
            name: [np.full((8, 8, 3), idx, dtype=np.uint8) for idx in range(4)]
            for name in ("left_forward", "front", "right_forward")
        }
        return A2VResult(sequence_num=0, frames=frames, streams=())

    def close(self) -> None:
        self.close_calls += 1


def _stub_depth(monkeypatch) -> None:
    monkeypatch.setattr(
        DepthCollisionReward,
        "estimate_depth",
        lambda self, frame: np.ones(frame.shape[:2], dtype=np.float32),
    )


def test_env_reuses_one_client_and_reprompts_each_reset(monkeypatch):
    _stub_depth(monkeypatch)
    client = FakeClient()
    env = OasisA2VEnv(
        client_factory=lambda: client,
        reward_fn=DepthCollisionReward(),
        prompt="frames prompt",
        observation_shape=(8, 8, 3),
    )

    try:
        env.reset()
        env.reset()
        # One long-lived session, re-prompted to reset rather than recreated each episode.
        assert client.initialize_calls == 1
        assert client.prompt_calls == 2
        assert client.prompt_text == "frames prompt"
        assert client.close_calls == 0
    finally:
        env.close()

    assert client.close_calls == 1  # session is only torn down at env shutdown


def test_env_step_returns_info_without_frames(monkeypatch):
    _stub_depth(monkeypatch)
    env = OasisA2VEnv(
        client_factory=FakeClient,
        reward_fn=DepthCollisionReward(),
        prompt="frames prompt",
        observation_shape=(8, 8, 3),
    )

    try:
        env.reset()
        _, _, _, _, info = env.step(np.zeros((4, 2), dtype=np.float32))
    finally:
        env.close()

    # Frames now stream straight from the client to its consumer; the env no longer
    # re-attaches them to step info.
    assert "frames" not in info


def test_reset_closes_client_if_initialize_fails(monkeypatch):
    _stub_depth(monkeypatch)

    class FailingClient(FakeClient):
        def initialize(self) -> None:
            super().initialize()
            raise RuntimeError("init failed")

    client = FailingClient()
    env = OasisA2VEnv(
        client_factory=lambda: client,
        reward_fn=DepthCollisionReward(),
        prompt="p",
        observation_shape=(8, 8, 3),
    )

    with pytest.raises(RuntimeError, match="init failed"):
        env.reset()

    # The half-open client is closed and unassigned, so a later reset re-creates it cleanly.
    assert client.close_calls == 1
    assert env.client is None


def test_reset_raises_when_camera_returns_no_frames(monkeypatch):
    _stub_depth(monkeypatch)

    class EmptyFrontClient(FakeClient):
        def infer(self, actions: np.ndarray) -> A2VResult:
            result = super().infer(actions)
            result.frames["front"] = []  # service returned an empty stream for a required camera
            return result

    env = OasisA2VEnv(
        client_factory=EmptyFrontClient,
        reward_fn=DepthCollisionReward(),
        prompt="p",
        observation_shape=(8, 8, 3),
    )

    try:
        with pytest.raises(DecartRoboticsError, match="front"):
            env.reset()
    finally:
        env.close()
