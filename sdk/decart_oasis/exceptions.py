"""SDK exceptions."""

from __future__ import annotations

from dataclasses import dataclass


class DecartRoboticsError(RuntimeError):
    """Base exception for Decart Robotics SDK errors."""


@dataclass(slots=True)
class A2VError(DecartRoboticsError):
    """Error returned by the A2V service."""

    code: int
    message: str
    details: dict[str, str]

    def __str__(self) -> str:
        if not self.details:
            return f"A2V error {self.code}: {self.message}"
        return f"A2V error {self.code}: {self.message} ({self.details})"
