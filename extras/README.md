# oasis-demo

RL examples and notebook visualization for the Decart **Oasis** SDK — everything that is *not* the
core connection. It builds on the [`decart-oasis`](../sdk) SDK (`A2VClient`) and adds:

- `oasis_demo.rl.reward` — a depth-based collision-avoidance reward (`DepthCollisionReward`).
- `oasis_demo.rl.env` — a Gymnasium environment (`OasisA2VEnv`) wrapping the SDK client.
- `oasis_demo.rl.sb3` / `rl.sft` / `rl.pretrained` — a Stable-Baselines3 PPO policy, behavior
  cloning from recorded episodes, and a pretrained-policy loader.
- `oasis_demo.live_preview` / `oasis_demo.visualization` — a live notebook camera preview and
  GIF/H.264 encoders.

Because it pulls the full ML stack (torch, stable-baselines3, transformers, …), it is a separate
package from the lightweight SDK.

## Install

```bash
git clone https://github.com/DecartAI/decart-robotics.git
pip install -e decart-robotics/extras       # editable, so you can tweak the reward etc.
```

`oasis-demo` declares `decart-oasis` (the SDK) as a dependency, so that single install also pulls the
published `decart-oasis` from PyPI — no separate step. Installing editable keeps the source under
`extras/oasis_demo/` visible and modifiable — edit, for example, `extras/oasis_demo/rl/reward.py`,
restart the kernel, and re-run.

## Development

This repo is a `uv` workspace; from the repo root:

```bash
uv sync                        # installs decart-oasis + oasis-demo editable
uv run pytest extras/tests
uv run ruff check extras
```
