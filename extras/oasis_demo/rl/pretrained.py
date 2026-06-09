"""Load a ready-made driving policy so the notebook can skip training.

The RL notebook trains a PPO policy, which takes time and Oasis API quota. For users who just
want to *watch* a trained agent drive (notebook §5) — or warm-start RL from a decent policy —
this module loads a pre-made policy by name/URL and returns a PPO that is a drop-in for the
notebook's RL and rollout cells::

    from oasis_demo.rl.pretrained import load_pretrained_policy
    model = load_pretrained_policy(device="auto")
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

**What is distributed is a policy `state_dict` (a ``.pt`` of tensors), not an SB3 ``.zip``.** An
SB3 ``model.save()`` zip pickles the policy object and is sensitive to the exact SB3 / torch /
Python versions it was written with; a bare ``state_dict`` loaded into a freshly-:func:`build_ppo`
model (which defines the architecture in code, in this package) is far more portable across
versions. Producing the artifact from any trained model is one call::

    from oasis_demo.rl.pretrained import export_policy_state_dict
    export_policy_state_dict(model, "oasis3_ppo_pretrained.pt")

Where the file comes from is resolved in this order:

1. the ``source`` argument (a local path or an ``http(s)`` URL),
2. the ``DECART_ROBOTICS_POLICY_URL`` environment variable,
3. the module default :data:`PRETRAINED_POLICY_URL`.

``http(s)`` sources are downloaded once into a local cache (``DECART_ROBOTICS_CACHE`` or
``~/.cache/decart-robotics/policies``) and reused. The download is credential-free (plain HTTPS),
matching the keyless Oasis service — host the artifact at a public URL (e.g. a GitHub Release
asset) and set :data:`PRETRAINED_POLICY_URL` (or the env var) to it.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import torch
from decart_oasis.exceptions import DecartRoboticsError
from stable_baselines3 import PPO

from oasis_demo.rl._fetch import fetch_to_path, gdrive_file_id, is_gdrive_url
from oasis_demo.rl.sft import build_ppo

# Public default download URL for the shipped policy (a Google Drive share link; fetched via gdown).
# Override with ``source=`` or DECART_ROBOTICS_POLICY_URL. See module docstring.
PRETRAINED_POLICY_URL = "https://drive.google.com/file/d/1VD0upvod6l75bDYTudZgu_AA6b7cCllK/view?usp=sharing"

POLICY_URL_ENV_VAR = "DECART_ROBOTICS_POLICY_URL"
CACHE_DIR_ENV_VAR = "DECART_ROBOTICS_CACHE"

__all__ = ["PRETRAINED_POLICY_URL", "export_policy_state_dict", "load_pretrained_policy"]


def export_policy_state_dict(model: PPO, path: str | os.PathLike) -> Path:
    """Save ``model``'s policy weights as a portable ``state_dict`` for distribution.

    The counterpart to :func:`load_pretrained_policy`: write this file to a public URL and point
    :data:`PRETRAINED_POLICY_URL` (or ``DECART_ROBOTICS_POLICY_URL``) at it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.policy.state_dict(), path)
    return path


def load_pretrained_policy(
    device: int | str = "auto",
    *,
    source: str | os.PathLike | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> PPO:
    """Build the PPO architecture and load a pre-made policy ``state_dict`` into it.

    ``source`` may be a local path or an ``http(s)`` URL; if omitted, falls back to the
    ``DECART_ROBOTICS_POLICY_URL`` env var and then :data:`PRETRAINED_POLICY_URL`. URLs are
    downloaded once into ``cache_dir`` (default: ``DECART_ROBOTICS_CACHE`` or
    ``~/.cache/decart-robotics/policies``) and reused on subsequent calls.

    Returns a PPO that is a drop-in for the notebook's RL/rollout cells: ``model.predict(obs)``
    to evaluate, or ``model.set_env(env); model.learn(...)`` to fine-tune further with RL.
    """
    weights_path = _resolve_policy_file(source, cache_dir)
    model = build_ppo(device=device)
    state_dict = torch.load(weights_path, map_location=model.policy.device, weights_only=True)
    model.policy.load_state_dict(state_dict)
    model.policy.eval()
    print(f"Loaded pretrained policy from {weights_path}")
    return model


def _resolve_policy_file(
    source: str | os.PathLike | None, cache_dir: str | os.PathLike | None
) -> Path:
    if source:
        resolved = str(source)
    else:
        resolved = os.environ.get(POLICY_URL_ENV_VAR) or PRETRAINED_POLICY_URL
    if not resolved:
        raise DecartRoboticsError(
            "No pretrained policy source configured. Pass source=<path-or-url>, set the "
            f"{POLICY_URL_ENV_VAR} environment variable, or set "
            "oasis_demo.rl.pretrained.PRETRAINED_POLICY_URL to a hosted weights file."
        )
    if urlparse(resolved).scheme in ("http", "https"):
        return _download_to_cache(resolved, cache_dir)
    path = Path(str(resolved))
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained policy file not found: {path}")
    return path


def _default_cache_dir() -> Path:
    base = os.environ.get(CACHE_DIR_ENV_VAR)
    if base:
        return Path(base)
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "decart-robotics" / "policies"


def _cache_name(url: str) -> str:
    """Stable cache filename for ``url``.

    A normal URL keeps its basename; a Google Drive share link has no useful basename in its path
    (``…/file/d/<id>/view``), so key the cache on the Drive file id instead.
    """
    if is_gdrive_url(url):
        return f"gdrive_{gdrive_file_id(url) or 'policy'}.pt"
    return Path(urlparse(url).path).name or "pretrained_policy.pt"


def _download_to_cache(url: str, cache_dir: str | os.PathLike | None) -> Path:
    directory = Path(cache_dir) if cache_dir else _default_cache_dir()
    dest = directory / _cache_name(url)
    if dest.is_file():
        return dest
    print(f"Downloading pretrained policy from {url} ...")
    return fetch_to_path(url, dest)  # atomic; routes Google Drive URLs through gdown
