"""Synchronous client for the Decart A2V gRPC service."""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Iterable, Sequence
from typing import Protocol
from urllib.parse import urlparse

import av
import grpc
import numpy as np
from PIL import Image

from decart_oasis import __version__ as DEFAULT_SDK_VERSION
from decart_oasis._proto import a2v_pb2, a2v_pb2_grpc, common_pb2
from decart_oasis.exceptions import A2VError, DecartRoboticsError
from decart_oasis.types import A2VResult, FrameConsumer, StreamInfo

DEFAULT_ENDPOINT = "https://oasis-grpc.decart.ai"
DEFAULT_ENDPOINT_ENV_VAR = "DECART_ROBOTICS_ENDPOINT"
DEFAULT_API_KEY_ENV_VAR = "DECART_API_KEY"
DEFAULT_REQUIRED_STREAMS = ("left_forward", "front", "right_forward")
SESSION_TARGET_METADATA_KEY = "x-session-target"

logger = logging.getLogger(__name__)


def _format_name(output_format: int | None) -> str:
    """Human-readable name for a FrameFormat enum value (falls back to the raw int)."""
    try:
        return common_pb2.FrameFormat.Name(output_format)
    except (ValueError, TypeError):
        return f"<unknown format {output_format!r}>"


class A2VClient:
    """Sync client for the Decart action-to-video service."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        api_key: str | None = None,
        client_sdk_version: str = DEFAULT_SDK_VERSION,
        timeout: float | None = 120.0,
        tls: bool | None = None,
        required_streams: Sequence[str] = DEFAULT_REQUIRED_STREAMS,
        frame_consumer: FrameConsumer | None = None,
        stub: a2v_pb2_grpc.A2VServiceStub | None = None,
        channel: grpc.Channel | None = None,
    ) -> None:
        endpoint = DEFAULT_ENDPOINT if endpoint is None else endpoint
        if not endpoint and stub is None:
            raise ValueError("endpoint is required")
        self.endpoint = endpoint
        # API key for the service; falls back to the DECART_API_KEY environment variable.
        self.api_key = api_key if api_key is not None else os.getenv(DEFAULT_API_KEY_ENV_VAR)
        self.client_sdk_version = client_sdk_version
        self.timeout = timeout
        self.tls = tls
        self.required_streams = tuple(required_streams)
        self.frame_consumer = frame_consumer
        self._owned_channel = channel is None and stub is None
        self._channel = channel if channel is not None else _create_channel(endpoint, tls=tls)
        self._stub = stub if stub is not None else a2v_pb2_grpc.A2VServiceStub(self._channel)
        self._session_id: str | None = None
        self._metadata: list[tuple[str, str]] = (
            [("x-api-key", self.api_key)] if self.api_key else []
        )
        self._session_target: str | None = None
        self._sequence_num = 0
        self._streams: tuple[StreamInfo, ...] = ()
        self._output_format: int | None = None
        self._decoders: dict[str, FrameDecoder] = {}

    @classmethod
    def from_env(
        cls,
        *,
        env_var: str = DEFAULT_ENDPOINT_ENV_VAR,
        default_endpoint: str | None = DEFAULT_ENDPOINT,
        **kwargs,
    ) -> A2VClient:
        endpoint = os.getenv(env_var) or default_endpoint
        if not endpoint:
            raise DecartRoboticsError(f"{env_var} is not set")
        return cls(endpoint, **kwargs)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def streams(self) -> tuple[StreamInfo, ...]:
        return self._streams

    @property
    def session_target(self) -> str | None:
        return self._session_target

    def __enter__(self) -> A2VClient:
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def initialize(self) -> tuple[StreamInfo, ...]:
        if not self.api_key:
            raise DecartRoboticsError(
                "A Decart API key is required. Pass api_key=... to A2VClient or set the "
                f"{DEFAULT_API_KEY_ENV_VAR} environment variable."
            )
        request = a2v_pb2.InitializeRequest(
            client_sdk_version=self.client_sdk_version,
            api_key=self.api_key,
            accepted_output_formats=[common_pb2.FRAME_FORMAT_VP9],
        )
        logger.info(
            "Initialize: endpoint=%s sdk_version=%s accepted_output_formats=[%s]",
            self.endpoint,
            self.client_sdk_version,
            _format_name(common_pb2.FRAME_FORMAT_VP9),
        )
        response, initial_metadata = self._initialize_with_metadata(request)
        _raise_if_error(response)
        self._capture_session_target(initial_metadata)
        self._session_id = response.session_id
        self._output_format = response.output_format
        self._streams = tuple(
            StreamInfo(
                name=stream.name,
                sensor_type=stream.type,
                height=stream.height,
                width=stream.width,
            )
            for stream in response.streams
        )
        logger.info(
            "InitializeResponse: session_id=%s output_format=%s streams=[%s]",
            response.session_id or "<empty>",
            _format_name(self._output_format),
            ", ".join(
                f"{s.name}({s.width}x{s.height})" for s in self._streams
            )
            or "<none>",
        )
        self._validate_initialized_streams()
        if self._output_format != common_pb2.FRAME_FORMAT_VP9:
            raise DecartRoboticsError(
                "A2V server negotiated output format "
                f"{_format_name(self._output_format)}, but this client only supports "
                f"{_format_name(common_pb2.FRAME_FORMAT_VP9)}. The server ignored or could not "
                "honor the requested VP9 format (check that the server and client protobuf "
                "definitions are in sync)."
            )
        self._decoders = {
            stream.name: _make_decoder(self._output_format) for stream in self._streams
        }
        return self._streams

    def prompt(self, prompt: str) -> None:
        session_id = self._require_session()
        response = self._stub.Prompt(
            a2v_pb2.PromptRequest(session_id=session_id, prompt=prompt),
            timeout=self.timeout,
            metadata=self._metadata,
        )
        _raise_if_error(response)
        # A new prompt resets the server's world-model context and its rollout: the server
        # then expects sequence_num to restart at 0, so reset our counter and begin a new clip.
        self._sequence_num = 0
        if self.frame_consumer is not None:
            self.frame_consumer.new_clip()

    def infer(self, actions: Iterable[Iterable[float]]) -> A2VResult:
        session_id = self._require_session()
        action_array = _validate_action_chunk(actions)
        sequence_num = self._sequence_num
        request = a2v_pb2.InferRequest(
            session_id=session_id,
            sequence_num=sequence_num,
            actions=[
                a2v_pb2.Action(throttle=float(throttle), steering=float(steering))
                for throttle, steering in action_array
            ],
        )
        response = self._stub.Infer(request, timeout=self.timeout, metadata=self._metadata)
        _raise_if_error(response)
        self._sequence_num += 1
        frames = self._decode_streams(response.streams)
        # Stream the freshly generated frames to any observer (e.g. a live preview)
        # without the caller having to re-fetch or forward them.
        if self.frame_consumer is not None:
            self.frame_consumer.submit(frames)
        return A2VResult(
            sequence_num=response.sequence_num,
            frames=frames,
            streams=self._streams,
        )

    def close(self) -> None:
        try:
            if self._session_id is not None:
                # Clear the session id up front so a Finish failure (or a retried/`__exit__`
                # close) never tries to Finish the same session twice.
                session_id, self._session_id = self._session_id, None
                response = self._stub.Finish(
                    a2v_pb2.FinishRequest(session_id=session_id),
                    timeout=self.timeout,
                    metadata=self._metadata,
                )
                _raise_if_error(response)
        finally:
            # Always release the channel we own, even if Finish raised — otherwise a failing
            # Finish inside a `with` block would leak the gRPC channel.
            if self._owned_channel and self._channel is not None:
                self._channel.close()
                self._channel = None

    def _require_session(self) -> str:
        if self._session_id is None:
            raise DecartRoboticsError("A2V session is not initialized")
        return self._session_id

    def _validate_initialized_streams(self) -> None:
        advertised = {stream.name for stream in self._streams}
        missing = set(self.required_streams) - advertised
        if missing:
            raise DecartRoboticsError(
                "A2V server did not advertise required streams: " + ", ".join(sorted(missing))
            )

    def _decode_streams(
        self, streams: Sequence[a2v_pb2.OutputStream]
    ) -> dict[str, list[np.ndarray]]:
        decoded: dict[str, list[np.ndarray]] = {}
        for stream in streams:
            decoder = self._decoders.get(stream.name)
            if decoder is None:
                decoder = _make_decoder(self._output_format)
                self._decoders[stream.name] = decoder
            decoded[stream.name] = [decoder.decode(frame) for frame in stream.frames]
        missing = set(self.required_streams) - set(decoded)
        if missing:
            raise DecartRoboticsError(
                "A2V response did not include required streams: " + ", ".join(sorted(missing))
            )
        return decoded

    def _initialize_with_metadata(
        self, request: a2v_pb2.InitializeRequest
    ) -> tuple[a2v_pb2.InitializeResponse, Sequence[tuple[str, str | bytes]]]:
        initialize = self._stub.Initialize
        if hasattr(initialize, "with_call"):
            response, call = initialize.with_call(
                request,
                timeout=self.timeout,
                metadata=self._metadata,
            )
            return response, tuple(call.initial_metadata())
        response = initialize(request, timeout=self.timeout, metadata=self._metadata)
        return response, ()

    def _capture_session_target(self, initial_metadata: Sequence[tuple[str, str | bytes]]) -> None:
        for key, value in initial_metadata:
            if key != SESSION_TARGET_METADATA_KEY:
                continue
            target = value.decode() if isinstance(value, bytes) else value
            self._session_target = target
            self._metadata = [
                (metadata_key, metadata_value)
                for metadata_key, metadata_value in self._metadata
                if metadata_key != SESSION_TARGET_METADATA_KEY
            ]
            self._metadata.append((SESSION_TARGET_METADATA_KEY, target))
            break


def _create_channel(endpoint: str, *, tls: bool | None = None) -> grpc.Channel:
    parsed = urlparse(endpoint)
    if parsed.scheme in ("http", "https"):
        path = "" if parsed.path == "/" else parsed.path.rstrip("/")
        target = parsed.netloc + path
        secure = parsed.scheme == "https" if tls is None else tls
    else:
        target = endpoint
        secure = True if tls is None else tls
    if not target:
        raise ValueError(f"Invalid endpoint: {endpoint!r}")
    if secure:
        return grpc.secure_channel(target, grpc.ssl_channel_credentials())
    return grpc.insecure_channel(target)


def _raise_if_error(response) -> None:
    if response.HasField("error"):
        error = response.error
        raise A2VError(code=error.code, message=error.message, details=dict(error.details))


def _validate_action_chunk(actions: Iterable[Iterable[float]]) -> np.ndarray:
    array = np.asarray(list(actions), dtype=np.float32)
    if array.shape != (4, 2):
        raise ValueError(f"actions must have shape (4, 2), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("actions must be finite")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("actions must be in [-1, 1]")
    return array


def _decode_jpeg(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


class FrameDecoder(Protocol):
    def decode(self, data: bytes) -> np.ndarray: ...


class JpegDecoder:
    def decode(self, data: bytes) -> np.ndarray:
        return _decode_jpeg(data)


class Vp9Decoder:
    def __init__(self) -> None:
        self._av = av
        self._ctx = av.CodecContext.create("vp9", "r")
        self._ctx.open()

    def decode(self, data: bytes) -> np.ndarray:
        packet = self._av.Packet(data)
        frames = list(self._ctx.decode(packet))
        if not frames:
            raise DecartRoboticsError("VP9 packet did not produce a decodable frame")
        return frames[-1].to_ndarray(format="rgb24")


def _make_decoder(output_format: int | None) -> FrameDecoder:
    if output_format == common_pb2.FRAME_FORMAT_VP9:
        return Vp9Decoder()
    if output_format == common_pb2.FRAME_FORMAT_JPEG:
        return JpegDecoder()
    raise DecartRoboticsError(f"Unsupported output format {output_format}")
