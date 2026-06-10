# decart-oasis

Python SDK for connecting to Decart's **Oasis 3 Preview** action-to-video world model. You open a session,
set a scene with a prompt, send driving actions, and get back the generated camera frames — actions
in, frames out.

This package is intentionally lightweight — just the gRPC client and VP9/JPEG frame decoding (no
torch / ML / Jupyter stack). The RL examples, depth-collision reward, and notebook visualization
live in the separate [`oasis-demo`](../extras) package, which depends on this one.

> **Full reference:** [`docs/python-sdk.mdx`](docs/python-sdk.mdx) (rendered on the Decart docs site).

## Install

```bash
pip install decart-oasis
```

Requires Python 3.10+.

## API key

Connecting requires a Decart API key. Set `DECART_API_KEY` (read automatically) or pass it in:

```bash
export DECART_API_KEY="sk-..."
```

```python
from decart_oasis import A2VClient

client = A2VClient()                 # reads DECART_API_KEY
client = A2VClient(api_key="sk-...") # or pass it explicitly
```

## Quickstart

```python
from decart_oasis import A2VClient

with A2VClient() as client:                       # opens the session, closes it on exit
    client.prompt("driving in an urban area")     # set the scene
    result = client.infer(                         # 4 actions in -> 4 frames per stream out
        [[0.2, 0.0], [0.2, 0.0], [0.2, 0.1], [0.2, 0.1]]
    )

front_frames = result.frames["front"]              # list of 4 RGB frames (H x W x 3, uint8)
```

`A2VClient` is a context manager: entering it calls `initialize()`, exiting it calls `close()` (even
on error).

## The session lifecycle

Without the `with` form, drive the four phases yourself:

```python
client = A2VClient()
client.initialize()                       # open session, authenticate, negotiate format
client.prompt("driving on a highway")     # set/reset the scene (sequence restarts at 0)
result = client.infer(actions)            # send 4 actions, get 4 frames per stream
client.close()                            # finish the session, release the channel
```

- **`initialize()`** returns the advertised streams (`left_forward`, `front`, `right_forward`).
- **`prompt(text)`** sets the scene; calling it again resets the world-model rollout.
- **`infer(actions)`** must be called after a prompt; loop it to keep driving.
- **`close()`** ends the session.

## Actions and frames

Each `infer` call takes a **chunk of four `[throttle, steering]` actions** and returns **four frames
per stream**.

- `throttle` in `[-1, 1]` — forward (`+`) / brake or reverse (`-`)
- `steering` in `[-1, 1]` — left (`-`) / right (`+`)

```python
import numpy as np

actions = np.array(
    [[0.15, -0.10], [0.20, -0.04], [0.22, 0.04], [0.18, 0.10]],
    dtype=np.float32,
)                                      # shape (4, 2), values in [-1, 1]
result = client.infer(actions)

result.sequence_num                    # server tick (0, 1, 2, ...; resets on prompt)
result.frames["front"]                 # list of 4 numpy RGB frames, shape (512, 768, 3)
set(result.frames)                     # {"left_forward", "front", "right_forward"}
```

Actions must be shape `(4, 2)`, finite, and within `[-1, 1]`, or `infer` raises `ValueError`.

## Endpoint

The SDK uses the hosted Oasis server by default (`https://oasis-grpc.decart.ai`). Override it via the
constructor, or `DECART_ROBOTICS_ENDPOINT` with `A2VClient.from_env()`:

```python
client = A2VClient("localhost:50051", tls=False)   # local insecure
client = A2VClient.from_env()                       # reads DECART_ROBOTICS_ENDPOINT
```

`https://` endpoints use TLS; `http://` uses an insecure channel. For bare `host:port` endpoints TLS
is on by default — pass `tls=False` for local/debug. The load balancer pins a session after
`Initialize` via the `x-session-target` initial-metadata header, which the SDK captures and replays
on later `Prompt`/`Infer`/`Finish` calls. Output frames are negotiated as VP9 and decoded with PyAV.

## Errors

All SDK errors subclass `DecartRoboticsError`. Errors returned by the service are raised as
`A2VError` with a `.code`, `.message`, and `.details`.

```python
from decart_oasis.exceptions import A2VError, DecartRoboticsError

try:
    with A2VClient() as client:
        client.prompt("driving in an urban area")
        client.infer(actions)
except A2VError as exc:
    print(f"service error {exc.code}: {exc.message}")   # e.g. ERROR_CODE_INVALID_API_KEY
except DecartRoboticsError as exc:
    print(f"client error: {exc}")
```

## Development

This repo is a `uv` workspace; from the repo root:

```bash
uv sync                       # installs decart-oasis + oasis-demo editable
uv run pytest sdk/tests
uv run ruff check sdk
```

Protobuf source lives in `sdk/protos/`; generated modules are committed under
`sdk/decart_oasis/_proto/` so users do not need `protoc`.
