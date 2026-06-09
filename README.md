# decart-robotics

The Python SDK for Decart's **Oasis 3** real-time world model, published to PyPI as
[`decart-oasis`](https://pypi.org/project/decart-oasis/).

```bash
pip install decart-oasis
```

```python
from decart_oasis import A2VClient

with A2VClient() as client:          # set DECART_API_KEY in the environment
    client.prompt("driving in an urban area")
    result = client.infer([[0.2, 0.0], [0.2, 0.0], [0.2, 0.1], [0.2, 0.1]])

front = result.frames["front"]       # list of 4 RGB frames (H×W×3 uint8)
```

The SDK lives in [`sdk/`](sdk) — see [`sdk/README.md`](sdk/README.md) for full usage and
[`sdk/docs/python-sdk.mdx`](sdk/docs/python-sdk.mdx) for the reference.

> The RL examples (`oasis-demo`) and the end-to-end Colab training notebook are added in a later
> phase, once the SDK is on PyPI.

## Development

```bash
uv sync                  # installs the SDK (editable) + dev tools
uv run pytest sdk/tests
uv run ruff check .
```

## Releasing

`.github/workflows/publish.yml` builds and publishes the SDK to PyPI via Trusted Publishing (OIDC) —
triggered by a GitHub Release, or manually via **Actions → Publish to PyPI → Run workflow** (choose
`testpypi` or `pypi`).
