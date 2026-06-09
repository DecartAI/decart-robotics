"""Public value types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """Metadata for one server-advertised output stream."""

    name: str
    sensor_type: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class A2VResult:
    """Decoded result from one A2V infer call."""

    sequence_num: int
    frames: dict[str, list[np.ndarray]]
    streams: tuple[StreamInfo, ...]


@runtime_checkable
class FrameConsumer(Protocol):
    """Receives generated camera frames as the client produces them.

    A consumer lets a client (or any frame source) stream frames somewhere — e.g.
    a live notebook preview — without the source knowing what happens to them.
    ``submit`` is called for every generated chunk; ``new_clip`` marks a boundary
    (a fresh prompt / episode), after which the next frames belong to a new clip.
    Both are expected to return quickly so they never slow the producer down.
    """

    def submit(self, frames: Mapping[str, Sequence[np.ndarray]]) -> None: ...

    def new_clip(self) -> None: ...
