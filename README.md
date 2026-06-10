# decart-robotics

The Python SDK for Decart's **Oasis 3 Preview** real-time world model, published to PyPI as
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

## RL demo (`oasis-demo`)

[`extras/`](extras) holds `oasis-demo`: a depth-collision reward, a Gymnasium env wrapping the SDK, a
Stable-Baselines3 PPO policy with behavior cloning, and a live notebook preview — everything on top
of the SDK for training a driving agent in Oasis. It pulls the full ML stack (torch,
stable-baselines3, transformers), so it's a separate package. See [`extras/README.md`](extras/README.md).

```bash
git clone https://github.com/DecartAI/decart-robotics.git
pip install -e decart-robotics/extras    # also pulls decart-oasis from PyPI
```

> The end-to-end Colab training notebook is added in a later phase.

## Development

```bash
uv sync                  # installs the SDK + oasis-demo (editable) + dev tools
uv run pytest            # sdk/tests + extras/tests
uv run ruff check .
```

## Releasing

`.github/workflows/publish.yml` builds and publishes the SDK to PyPI via Trusted Publishing (OIDC) —
triggered by a GitHub Release, or manually via **Actions → Publish to PyPI → Run workflow** (choose
`testpypi` or `pypi`).
