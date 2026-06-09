"""Gymnasium environment wrapper for the A2V service."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from decart_oasis.client import A2VClient
from decart_oasis.exceptions import DecartRoboticsError
from gymnasium import spaces

from oasis_demo.rl.reward import DepthCollisionReward

CAMERA_STREAMS = ("left_forward", "front", "right_forward")


class OasisA2VEnv(gym.Env):
    """Gymnasium environment for chunked A2V driving actions."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        client_factory: Callable[[], A2VClient],
        reward_fn: DepthCollisionReward,
        *,
        prompt: str,
        max_steps: int = 512,
        observation_shape: tuple[int, int, int] = (512, 768, 3),
    ) -> None:
        self.client_factory = client_factory
        self.reward_fn = reward_fn
        self.prompt_text = prompt
        self.max_steps = max_steps
        self.client: A2VClient | None = None
        self.step_count = 0
        self.last_observation: dict[str, np.ndarray] | None = None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4, 2), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                name: spaces.Box(low=0, high=255, shape=observation_shape, dtype=np.uint8)
                for name in CAMERA_STREAMS
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.client is None:
            client = self.client_factory()
            try:
                client.initialize()
            except BaseException:
                # Don't keep a half-open client: a later reset() checks `self.client is None`
                # to decide whether to re-initialize, so leaving a failed client assigned would
                # skip re-init and operate on a dead session.
                client.close()
                raise
            self.client = client
        # Re-prompting resets the server's world-model context, which resets the episode
        # without tearing down and recreating the session every time.
        self.client.prompt(self.prompt_text)
        self.reward_fn.reset()
        self.step_count = 0
        zero_action = np.zeros((4, 2), dtype=np.float32)
        result = self.client.infer(zero_action)
        self.last_observation = _last_camera_observation(result.frames)
        return self.last_observation, {}

    def step(self, action):
        if self.client is None:
            raise RuntimeError("Environment must be reset before step")
        action_chunk = np.asarray(action, dtype=np.float32)
        result = self.client.infer(action_chunk)
        observation = _last_camera_observation(result.frames)
        reward, terminated, info = self.reward_fn(result.frames["front"], action_chunk)
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        self.last_observation = observation
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.last_observation is None:
            return None
        return self.last_observation["front"]

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def _last_camera_observation(frames: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    for name in CAMERA_STREAMS:
        stream = frames.get(name)
        if not stream:
            raise DecartRoboticsError(
                f"A2V response contained no frames for required camera {name!r}"
            )
        observation[name] = stream[-1].astype(np.uint8, copy=False)
    return observation
