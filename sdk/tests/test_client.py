from __future__ import annotations

import io

import decart_oasis.client as client_module
import numpy as np
import pytest
from decart_oasis._proto import a2v_pb2, common_pb2
from decart_oasis.client import (
    DEFAULT_API_KEY_ENV_VAR,
    DEFAULT_ENDPOINT,
    DEFAULT_ENDPOINT_ENV_VAR,
    A2VClient,
    _create_channel,
)
from decart_oasis.exceptions import A2VError, DecartRoboticsError
from PIL import Image


class FakeStub:
    def __init__(
        self,
        streams: tuple[str, ...] = ("left_forward", "front", "right_forward"),
        initial_metadata: tuple[tuple[str, str | bytes], ...] = (),
        output_format: int = common_pb2.FRAME_FORMAT_VP9,
    ):
        self.stream_names = streams
        self.initial_metadata = initial_metadata
        self.output_format = output_format
        self.initialize_request = None
        self.initialize_metadata = None
        self.prompt_metadata = None
        self.infer_metadata = None
        self.finish_metadata = None
        self.infer_requests = []
        self.finished = False
        self.Initialize = FakeUnaryUnary(self._initialize, initial_metadata)

    def _initialize(self, request, timeout=None, metadata=None):
        self.initialize_request = request
        self.initialize_metadata = metadata
        return a2v_pb2.InitializeResponse(
            session_id="session-1",
            output_format=self.output_format,
            streams=[
                a2v_pb2.StreamInfo(
                    name=name,
                    type=a2v_pb2.SENSOR_TYPE_RGB,
                    height=512,
                    width=768,
                )
                for name in self.stream_names
            ],
        )

    def Prompt(self, request, timeout=None, metadata=None):
        self.prompt_metadata = metadata
        return a2v_pb2.PromptResponse()

    def Infer(self, request, timeout=None, metadata=None):
        self.infer_metadata = metadata
        self.infer_requests.append(request)
        frame = _jpeg(np.zeros((8, 8, 3), dtype=np.uint8))
        return a2v_pb2.InferResponse(
            sequence_num=request.sequence_num,
            streams=[
                a2v_pb2.OutputStream(name=name, frames=[frame, frame, frame, frame])
                for name in self.stream_names
            ],
        )

    def Finish(self, request, timeout=None, metadata=None):
        self.finish_metadata = metadata
        self.finished = True
        return a2v_pb2.FinishResponse()


class FakeUnaryUnary:
    def __init__(self, fn, initial_metadata: tuple[tuple[str, str | bytes], ...]):
        self.fn = fn
        self._initial_metadata = initial_metadata

    def __call__(self, request, timeout=None, metadata=None):
        return self.fn(request, timeout=timeout, metadata=metadata)

    def with_call(self, request, timeout=None, metadata=None):
        response = self.fn(request, timeout=timeout, metadata=metadata)
        return response, FakeCall(self._initial_metadata)


class FakeCall:
    def __init__(self, initial_metadata: tuple[tuple[str, str | bytes], ...]):
        self._initial_metadata = initial_metadata

    def initial_metadata(self):
        return self._initial_metadata


class FakeDecoder:
    def decode(self, data: bytes) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)


class FakeConsumer:
    def __init__(self):
        self.submitted = []
        self.new_clips = 0

    def submit(self, frames):
        self.submitted.append(frames)

    def new_clip(self):
        self.new_clips += 1


@pytest.fixture(autouse=True)
def fake_decoder(monkeypatch):
    monkeypatch.setattr(client_module, "_make_decoder", lambda output_format: FakeDecoder())


@pytest.fixture(autouse=True)
def api_key_env(monkeypatch):
    # The service requires an API key; provide one by default so tests can initialize.
    monkeypatch.setenv(DEFAULT_API_KEY_ENV_VAR, "test-key")


def test_client_initializes_prompts_infers_and_finishes():
    stub = FakeStub()
    client = A2VClient("https://example.com", stub=stub)

    with client:
        client.prompt("urban driving")
        result = client.infer(np.zeros((4, 2), dtype=np.float32))

    assert result.sequence_num == 0
    assert set(result.frames) == {"left_forward", "front", "right_forward"}
    assert result.frames["front"][0].shape == (8, 8, 3)
    assert stub.initialize_request.accepted_output_formats == [common_pb2.FRAME_FORMAT_VP9]
    assert stub.initialize_request.api_key == "test-key"  # sent on Initialize
    assert stub.infer_requests[0].sequence_num == 0
    assert stub.finished


def test_initialize_requires_api_key(monkeypatch):
    monkeypatch.delenv(DEFAULT_API_KEY_ENV_VAR, raising=False)
    client = A2VClient("https://example.com", stub=FakeStub())
    with pytest.raises(DecartRoboticsError, match="API key is required"):
        client.initialize()


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv(DEFAULT_API_KEY_ENV_VAR, "env-key")
    stub = FakeStub()
    client = A2VClient("https://example.com", api_key="explicit-key", stub=stub)
    client.initialize()
    assert stub.initialize_request.api_key == "explicit-key"


def test_client_streams_frames_to_consumer():
    consumer = FakeConsumer()
    stub = FakeStub()
    client = A2VClient("https://example.com", stub=stub, frame_consumer=consumer)

    with client:
        client.prompt("urban driving")
        client.infer(np.zeros((4, 2), dtype=np.float32))

    # prompt starts a new clip; infer hands the decoded frames to the consumer.
    assert consumer.new_clips == 1
    assert len(consumer.submitted) == 1
    assert set(consumer.submitted[0]) == {"left_forward", "front", "right_forward"}


def test_client_resets_sequence_after_reprompt():
    stub = FakeStub()
    client = A2VClient("https://example.com", stub=stub)

    client.initialize()
    client.prompt("urban driving")
    client.infer(np.zeros((4, 2), dtype=np.float32))  # sequence_num 0
    client.infer(np.zeros((4, 2), dtype=np.float32))  # sequence_num 1
    client.prompt("a different scene")  # resets the server's rollout
    client.infer(np.zeros((4, 2), dtype=np.float32))  # sequence_num 0 again

    assert [r.sequence_num for r in stub.infer_requests] == [0, 1, 0]


def test_client_uses_hosted_default_endpoint():
    stub = FakeStub()
    client = A2VClient(stub=stub)

    assert client.endpoint == DEFAULT_ENDPOINT


def test_from_env_uses_hosted_default_and_allows_override(monkeypatch):
    monkeypatch.delenv(DEFAULT_ENDPOINT_ENV_VAR, raising=False)
    default_client = A2VClient.from_env(stub=FakeStub())
    assert default_client.endpoint == DEFAULT_ENDPOINT

    monkeypatch.setenv(DEFAULT_ENDPOINT_ENV_VAR, "https://private-grpc.example.com")
    override_client = A2VClient.from_env(stub=FakeStub())
    assert override_client.endpoint == "https://private-grpc.example.com"


def test_client_reuses_session_target_metadata_after_initialize():
    stub = FakeStub(initial_metadata=(("x-session-target", "pod-a"),))
    client = A2VClient("https://example.com", stub=stub)

    with client:
        client.prompt("urban driving")
        client.infer(np.zeros((4, 2), dtype=np.float32))

    assert client.session_target == "pod-a"
    assert stub.initialize_metadata == []
    assert stub.prompt_metadata == [("x-session-target", "pod-a")]
    assert stub.infer_metadata == [("x-session-target", "pod-a")]
    assert stub.finish_metadata == [("x-session-target", "pod-a")]


def test_client_decodes_binary_session_target_metadata():
    stub = FakeStub(initial_metadata=(("x-session-target", b"pod-b"),))
    client = A2VClient("https://example.com", stub=stub)

    client.initialize()

    assert client.session_target == "pod-b"


def test_client_rejects_missing_required_stream():
    client = A2VClient("https://example.com", stub=FakeStub(streams=("front",)))
    with pytest.raises(DecartRoboticsError, match="left_forward"):
        client.initialize()


def test_client_validates_action_shape_and_range():
    client = A2VClient("https://example.com", stub=FakeStub())
    client.initialize()

    with pytest.raises(ValueError, match=r"shape"):
        client.infer(np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        client.infer(np.full((4, 2), 2.0, dtype=np.float32))


def test_client_raises_service_error():
    class ErrorStub(FakeStub):
        def __init__(self):
            super().__init__()
            self.Initialize = FakeUnaryUnary(self._initialize, ())

        def _initialize(self, request, timeout=None, metadata=None):
            return a2v_pb2.InitializeResponse(
                error=common_pb2.Error(
                    code=common_pb2.ERROR_CODE_INVALID_REQUEST,
                    message="bad request",
                )
            )

    client = A2VClient("https://example.com", stub=ErrorStub())
    with pytest.raises(A2VError, match="bad request"):
        client.initialize()


def test_close_releases_channel_even_if_finish_fails():
    class BoomFinishStub(FakeStub):
        def Finish(self, request, timeout=None, metadata=None):
            raise RuntimeError("network down")

    class RecordingChannel:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    client = A2VClient("https://example.com", stub=BoomFinishStub())
    client.initialize()
    channel = RecordingChannel()
    client._owned_channel = True
    client._channel = channel

    with pytest.raises(RuntimeError, match="network down"):
        client.close()

    # The owned channel is released despite the Finish failure, and the session is cleared up
    # front so a retried/`__exit__` close does not Finish the same session again.
    assert channel.closed == 1
    assert client.session_id is None


def test_create_channel_allows_tls_override(monkeypatch):
    calls = []

    def fake_secure_channel(target, credentials):
        calls.append(("secure", target, credentials))
        return object()

    def fake_insecure_channel(target):
        calls.append(("insecure", target))
        return object()

    monkeypatch.setattr("grpc.secure_channel", fake_secure_channel)
    monkeypatch.setattr("grpc.insecure_channel", fake_insecure_channel)
    monkeypatch.setattr("grpc.ssl_channel_credentials", lambda: "creds")

    _create_channel("api-grpc.decart.ai:443", tls=True)
    _create_channel("https://api-grpc.decart.ai", tls=False)
    _create_channel("https://api-grpc.decart.ai/", tls=True)

    assert calls == [
        ("secure", "api-grpc.decart.ai:443", "creds"),
        ("insecure", "api-grpc.decart.ai"),
        ("secure", "api-grpc.decart.ai", "creds"),
    ]


def _jpeg(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG")
    return buffer.getvalue()
