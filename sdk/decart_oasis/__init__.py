"""Decart Robotics SDK."""

from importlib.metadata import PackageNotFoundError, version

from decart_oasis.exceptions import A2VError, DecartRoboticsError
from decart_oasis.types import A2VResult, FrameConsumer, StreamInfo

try:
    __version__ = version("decart-oasis")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "A2VClient",
    "A2VError",
    "A2VResult",
    "DEFAULT_ENDPOINT",
    "DecartRoboticsError",
    "FrameConsumer",
    "StreamInfo",
    "__version__",
]


def __getattr__(name: str):
    if name == "A2VClient":
        from decart_oasis.client import A2VClient

        return A2VClient
    if name == "DEFAULT_ENDPOINT":
        from decart_oasis.client import DEFAULT_ENDPOINT

        return DEFAULT_ENDPOINT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Make the lazily-exposed names (A2VClient, DEFAULT_ENDPOINT) show up in tab-completion and
    # dir() despite being resolved through __getattr__.
    return sorted(__all__)
