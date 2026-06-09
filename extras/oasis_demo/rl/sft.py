"""Supervised fine-tuning (behavior cloning) of the A2V PPO policy from S3 demonstrations.

A separate data-collection platform records humans driving Oasis 3 and uploads the sessions to
an S3 bucket as episode folders (see ``SftBehaviorCloningDataset`` for the on-disk layout). This
module pulls that data and **behavior-clones** the same PPO policy the RL notebook trains, so the
agent starts from human-like driving instead of random exploration. The cloned model is a drop-in
for the notebook's RL cell: it is built with the identical spaces, ``policy_kwargs`` and PPO
hyperparameters, so ``model.set_env(real_env)`` followed by ``model.learn(...)`` fine-tunes the
cloned weights with reinforcement learning.

Typical use (one call from the notebook)::

    from oasis_demo.rl.sft import pretrain_ppo_from_s3
    model = pretrain_ppo_from_s3(bucket="my-bucket", prefix="sft_data", device=DEVICE)
    # ... then the RL cell does model.set_env(env); model.learn(...)
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from oasis_demo.rl._fetch import fetch_to_path
from oasis_demo.rl.env import CAMERA_STREAMS
from oasis_demo.rl.sb3 import ThreeCameraFeatureExtractor

# Observation/action geometry — mirrors oasis_demo.rl.env.OasisA2VEnv so the cloned policy
# is built on exactly the same spaces the RL env emits.
OBSERVATION_SHAPE = (512, 768, 3)

# PPO build — kept identical to the notebook's RL cell so the SFT model is a drop-in for it.
POLICY_KWARGS = {"features_extractor_class": ThreeCameraFeatureExtractor}
PPO_HYPERPARAMS = {
    "n_steps": 32,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "verbose": 1,
}


# --- S3 download -------------------------------------------------------------------------------
def download_sft_data(
    bucket: str,
    prefix: str,
    dest_dir: str | os.PathLike,
    *,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    region_name: str | None = None,
    max_episodes: int | None = None,
    max_workers: int = 16,
    progress: bool = True,
) -> Path:
    """Download objects under ``s3://bucket/prefix`` into ``dest_dir``, concurrently.

    Preserves the key layout (``prefix/episode_*/...``) under ``dest_dir`` and skips files that
    already exist, so re-running is cheap. Downloads run in a thread pool (``max_workers``), which
    is the bottleneck fix: thousands of small frame files fetched in parallel instead of one at a
    time.

    ``max_episodes`` limits the download to the first N episode folders (lexicographic by key) —
    handy for a quick start (each episode is ~50 steps, so ``max_episodes=10`` ≈ 500 examples).

    Credentials: pass them explicitly, or leave them ``None`` to use boto3's default chain, which
    reads ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_SESSION_TOKEN`` and
    ``AWS_DEFAULT_REGION`` from the environment. **Temporary / STS credentials require
    ``aws_session_token`` (env ``AWS_SESSION_TOKEN``)** in addition to the access key and secret.

    Returns the local directory that contains the episode folders (``dest_dir/prefix``).
    """
    import boto3  # imported lazily so the rest of the module works without boto3 installed

    dest_dir = Path(dest_dir)

    # Trim credentials before signing: a trailing newline or surrounding whitespace in the env var
    # (a very common copy-paste artifact) silently breaks SigV4 and surfaces as the opaque
    # "SignatureDoesNotMatch" error. Treat blank values as unset so boto3 falls back to its chain.
    def _clean(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    session = boto3.session.Session(
        aws_access_key_id=_clean(aws_access_key_id),
        aws_secret_access_key=_clean(aws_secret_access_key),
        aws_session_token=_clean(aws_session_token),
        region_name=_clean(region_name),
    )
    client = session.client("s3")
    paginator = client.get_paginator("list_objects_v2")

    # Bound the prefix to a folder so a prefix like "sft_data" doesn't also pull a sibling
    # "sft_data_other/..." (S3 prefixes are byte-prefixes, not folder-aware). Empty prefix means
    # the whole bucket. We list with the bounded prefix and strip it the same way in episode_of.
    listing_prefix = (prefix.rstrip("/") + "/") if prefix else ""

    # List everything once (cheap), so we know the total and can filter/limit before fetching.
    objects = [
        obj
        for page in paginator.paginate(Bucket=bucket, Prefix=listing_prefix)
        for obj in page.get("Contents", [])
        if not obj["Key"].endswith("/")  # skip directory placeholders
    ]

    def episode_of(key: str) -> str:
        return key[len(listing_prefix):].split("/", 1)[0]

    if max_episodes is not None:
        ordered_episodes: list[str] = []
        for obj in objects:
            episode = episode_of(obj["Key"])
            if episode and episode not in ordered_episodes:
                ordered_episodes.append(episode)
        keep = set(ordered_episodes[:max_episodes])
        objects = [obj for obj in objects if episode_of(obj["Key"]) in keep]
        print(
            f"Limiting to {len(keep)} of {len(ordered_episodes)} episodes ({len(objects)} files)."
        )

    # Skip files already present locally (cheap local stat); only fetch the rest, in parallel.
    to_download = [
        obj
        for obj in objects
        if not (
            (local := dest_dir / obj["Key"]).exists() and local.stat().st_size == obj["Size"]
        )
    ]
    n_skipped = len(objects) - len(to_download)

    def _fetch(obj: dict) -> None:
        local_path = dest_dir / obj["Key"]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, obj["Key"], str(local_path))  # botocore client is thread-safe

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch, obj) for obj in to_download]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading SFT data from S3",
            unit="file",
            disable=not progress,
        ):
            future.result()  # surface any download error

    print(f"S3 sync done: {len(to_download)} downloaded, {n_skipped} already present.")
    root = dest_dir / prefix
    if not root.is_dir():
        raise FileNotFoundError(
            f"Expected episodes under {root}, but it does not exist. "
            f"Check the bucket ({bucket!r}) and prefix ({prefix!r})."
        )
    return root


# --- Demo dataset (keyless HTTP download) ------------------------------------------------------
# Public archive of a curated set of driving episodes, for users who don't have the recorded S3
# bucket or AWS credentials. A Google Drive share link (fetched via gdown). Override with
# ``source=`` or DECART_ROBOTICS_DEMO_DATA_URL.
DEMO_DATASET_URL = "https://drive.google.com/file/d/1YxR3NmEEqHpxn4Dq51TPOGqFKEPhrSRx/view?usp=sharing"
DEMO_DATA_URL_ENV_VAR = "DECART_ROBOTICS_DEMO_DATA_URL"


def download_demo_dataset(
    dest_dir: str | os.PathLike = "sft_data_demo",
    *,
    source: str | None = None,
    force: bool = False,
) -> Path:
    """Download and extract a small, public demo dataset of driving episodes (keyless HTTP).

    A credential-free alternative to :func:`download_sft_data` for the notebook: pulls a small
    curated archive of episodes from a public URL and extracts it locally, returning the directory
    of ``episode_*/`` folders ready for :class:`SftBehaviorCloningDataset`. Pass ``source=`` a
    ``.tar.gz``/``.zip`` URL (a Google Drive share link works too), set
    ``DECART_ROBOTICS_DEMO_DATA_URL``, or set the module default :data:`DEMO_DATASET_URL`.
    Re-running is cheap: an already-extracted dataset is reused unless ``force=True``.
    """
    resolved = source or os.environ.get(DEMO_DATA_URL_ENV_VAR) or DEMO_DATASET_URL
    if not resolved:
        raise ValueError(
            "No demo dataset source configured. Pass source=<url>, set the "
            f"{DEMO_DATA_URL_ENV_VAR} environment variable, or set "
            "oasis_demo.rl.sft.DEMO_DATASET_URL to a public .tar.gz/.zip of episode_*/ "
            "folders."
        )
    dest = Path(dest_dir)
    existing = _find_episode_root(dest)
    if existing is not None and not force:
        print(f"Demo dataset already present at {existing}")
        return existing

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading demo dataset from {resolved} ...")
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".archive")
    tmp_path = Path(tmp_name)
    os.close(tmp_fd)
    try:
        fetch_to_path(resolved, tmp_path)  # routes Google Drive URLs through gdown
        _extract_archive(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)

    root = _find_episode_root(dest)
    if root is None:
        raise FileNotFoundError(
            f"Extracted the demo dataset to {dest} but found no episode_*/ folders containing a "
            "manifest.jsonl. Check that the archive matches the expected episode layout."
        )
    print(f"Demo dataset ready at {root}")
    return root


def _extract_archive(archive: Path, dest: Path) -> None:
    # Sniff the archive by content, not by filename: a Google Drive download has no .zip/.tar.gz
    # suffix in its URL, so we can't rely on the name to tell zip from tar.
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    if tarfile.is_tarfile(archive):
        # py3.12+: refuse unsafe (path-traversal) members; py<3.12 has no extraction filter.
        with tarfile.open(archive) as tf:
            try:
                tf.extractall(dest, filter="data")
            except TypeError:
                tf.extractall(dest)
        return
    raise ValueError(
        f"Downloaded demo dataset is neither a zip nor a tar archive ({archive}). If the source "
        "is a Google Drive link, check it is shared publicly and points directly at the archive "
        "file (not a folder)."
    )


def _find_episode_root(directory: Path) -> Path | None:
    """Return the folder that directly contains ``episode_*/manifest.jsonl``, or ``None``.

    Handles archives that put the episodes directly under ``directory`` as well as ones that nest
    them inside a single top-level folder.
    """
    if not directory.is_dir():
        return None
    for candidate in (directory, *(p for p in sorted(directory.iterdir()) if p.is_dir())):
        if any(candidate.glob("*/manifest.jsonl")):
            return candidate
    return None


# --- Dataset -----------------------------------------------------------------------------------
class SftBehaviorCloningDataset(Dataset):
    """Behavior-cloning pairs from collected Oasis driving episodes.

    Expects a directory of episode folders, each laid out as::

        <root>/
          episode_<id>/
            manifest.jsonl                 # one JSON object per step (per obs->action pair)
            frames/000000_left_forward.png  000000_front.png  000000_right_forward.png  ...

    Each ``manifest.jsonl`` line has ``action`` (a 4x2 ``[throttle, steering]`` chunk in
    ``[-1, 1]``) and ``frame_left_forward`` / ``frame_front`` / ``frame_right_forward`` paths
    relative to the episode folder. ``__getitem__`` returns ``(obs, action)`` where ``obs`` is a
    dict of three ``uint8`` HWC RGB arrays (shape ``(512, 768, 3)``) keyed by ``CAMERA_STREAMS``
    and ``action`` is a ``float32`` ``(4, 2)`` array — exactly the observation/action the RL env
    uses. PNGs are RGBA on disk and are converted to RGB on load.

    Set ``drop_idle=True`` to skip leading all-zero-action steps (the car sitting still at the
    start of an episode); they are kept by default as valid "hold still" demonstrations.
    """

    def __init__(self, root: str | os.PathLike, *, drop_idle: bool = False) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"SFT data root not found: {self.root}")
        self._samples: list[tuple[Path, dict]] = []
        for manifest_path in sorted(self.root.glob("*/manifest.jsonl")):
            episode_dir = manifest_path.parent
            with manifest_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if drop_idle and not np.any(np.asarray(record["action"], dtype=np.float32)):
                        continue
                    self._samples.append((episode_dir, record))
        if not self._samples:
            raise ValueError(f"No demonstration steps found under {self.root}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        episode_dir, record = self._samples[index]
        obs: dict[str, np.ndarray] = {}
        for name in CAMERA_STREAMS:
            with Image.open(episode_dir / record[f"frame_{name}"]) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if frame.shape != OBSERVATION_SHAPE:
                raise ValueError(
                    f"{episode_dir.name} {name}: expected frame {OBSERVATION_SHAPE}, "
                    f"got {frame.shape}"
                )
            obs[name] = frame
        action = np.asarray(record["action"], dtype=np.float32)
        if action.shape != (4, 2):
            raise ValueError(f"{episode_dir.name}: action must be (4, 2), got {action.shape}")
        return obs, action


def _collate(
    batch: list[tuple[dict[str, np.ndarray], np.ndarray]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Stack dict observations and actions into batched arrays for the policy."""
    obs = {name: np.stack([item[0][name] for item in batch]) for name in CAMERA_STREAMS}
    actions = np.stack([item[1] for item in batch]).astype(np.float32)
    return obs, actions


# --- Model + behavior cloning ------------------------------------------------------------------
class _SpacesOnlyEnv(gym.Env):
    """Minimal env that only declares the A2V spaces, so PPO can be built without calling Oasis."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4, 2), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                name: spaces.Box(low=0, high=255, shape=OBSERVATION_SHAPE, dtype=np.uint8)
                for name in CAMERA_STREAMS
            }
        )


def build_ppo(device: int | str = "auto") -> PPO:
    """Build the PPO model identically to the RL notebook cell (on a spaces-only stub env).

    The stub env never contacts Oasis, so constructing the model spends no API quota. Because the
    spaces, ``policy_kwargs`` and hyperparameters match the RL cell exactly, the returned model is
    a drop-in: after behavior cloning, ``model.set_env(real_env)`` + ``model.learn(...)`` continues
    training with RL.
    """
    return PPO(
        "MultiInputPolicy",
        _SpacesOnlyEnv(),
        policy_kwargs=POLICY_KWARGS,
        device=device,
        **PPO_HYPERPARAMS,
    )


def behavior_clone(
    model: PPO,
    dataset: Dataset,
    *,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    num_workers: int = 0,
    shrink_log_std: float | None = -1.0,
    progress: bool = True,
) -> PPO:
    """Behavior-clone ``model``'s policy to imitate the demonstrations in ``dataset``.

    Regresses the policy's action-mean head to the expert action chunk (MSE on the flattened
    8-dim action). This trains the shared CNN feature extractor and the actor head — the
    parameters PPO then continues to optimize; the value head stays untrained and PPO warms it up.

    ``num_workers`` defaults to 0 (load in the main process). On memory-constrained hosts like a
    Colab T4, ``num_workers>0`` spawns worker subprocesses that ship large image batches through
    shared memory and can get OOM-killed ("DataLoader worker ... killed by signal: Killed"); only
    raise it on a box with plenty of RAM and spare CPU cores.

    ``shrink_log_std`` (if not ``None``) sets the policy's log-std after cloning so RL starts with
    modest exploration noise around the cloned behavior rather than the default std=1.
    """
    policy = model.policy
    device = policy.device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    policy.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        bar = tqdm(
            loader, desc=f"SFT epoch {epoch + 1}/{epochs}", unit="batch", disable=not progress
        )
        for obs_batch, action_batch in bar:
            obs_tensor = obs_as_tensor(obs_batch, device)
            features = policy.extract_features(obs_tensor)
            if isinstance(features, tuple):
                # A non-shared features extractor (policy_kwargs share_features_extractor=False)
                # returns (pi_features, vf_features); behavior cloning trains only the actor, so
                # take the policy-side features. With the default shared extractor this is a tensor.
                features = features[0]
            latent_pi, _ = policy.mlp_extractor(features)
            predicted_mean = policy.action_net(latent_pi)  # (B, 8), row-major [t0,s0,t1,s1,...]
            target = torch.as_tensor(action_batch, device=device).reshape(action_batch.shape[0], -1)
            loss = F.mse_loss(predicted_mean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            bar.set_postfix(loss=total_loss / n_batches)

    if shrink_log_std is not None and hasattr(policy, "log_std"):
        with torch.no_grad():
            policy.log_std.fill_(shrink_log_std)
    return model


def pretrain_ppo_from_s3(
    bucket: str,
    prefix: str = "sft_data",
    *,
    device: int | str = "auto",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    local_dir: str | os.PathLike = "sft_data_cache",
    drop_idle: bool = False,
    num_workers: int = 0,
    max_episodes: int | None = None,
    max_workers: int = 16,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    region_name: str | None = None,
    progress: bool = True,
) -> PPO:
    """Download demonstrations from S3 and return a PPO model behavior-cloned on them.

    Single entry point for the notebook: downloads ``s3://bucket/prefix`` to ``local_dir``,
    builds the PPO model (identical to the RL cell), behavior-clones it on the human driving
    data, and returns it ready for ``model.set_env(env); model.learn(...)``.

    The download runs in a thread pool (``max_workers``). Set ``max_episodes`` to start small —
    each episode is ~50 steps, so ``max_episodes=10`` ≈ 500 examples.

    Credentials are optional: leave them ``None`` to use boto3's default chain (env vars
    ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN`` /
    ``AWS_DEFAULT_REGION``). Temporary / STS credentials need ``aws_session_token``.
    """
    root = download_sft_data(
        bucket,
        prefix,
        local_dir,
        max_episodes=max_episodes,
        max_workers=max_workers,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
        progress=progress,
    )
    return _clone_from_root(
        root,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        drop_idle=drop_idle,
        num_workers=num_workers,
        progress=progress,
    )


def pretrain_ppo_from_demo(
    *,
    source: str | None = None,
    dest_dir: str | os.PathLike = "sft_data_demo",
    device: int | str = "auto",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    drop_idle: bool = False,
    num_workers: int = 0,
    force: bool = False,
    progress: bool = True,
) -> PPO:
    """Download the public demo dataset and return a PPO model behavior-cloned on it.

    Keyless counterpart to :func:`pretrain_ppo_from_s3` for the notebook: pulls the small demo
    archive of driving episodes (see :func:`download_demo_dataset` — ``source`` may be a plain URL
    or a Google Drive share link; if omitted it falls back to the ``DECART_ROBOTICS_DEMO_DATA_URL``
    env var and then :data:`DEMO_DATASET_URL`), behavior-clones the same PPO policy the RL cell
    trains, and returns it ready for ``model.set_env(env); model.learn(...)``.
    """
    root = download_demo_dataset(dest_dir, source=source, force=force)
    return _clone_from_root(
        root,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        drop_idle=drop_idle,
        num_workers=num_workers,
        progress=progress,
    )


def _clone_from_root(
    root: str | os.PathLike,
    *,
    device: int | str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    drop_idle: bool,
    num_workers: int,
    progress: bool,
) -> PPO:
    """Build the PPO model and behavior-clone it on the episodes under ``root``."""
    dataset = SftBehaviorCloningDataset(root, drop_idle=drop_idle)
    print(f"Loaded {len(dataset)} demonstration steps from {root}.")
    model = build_ppo(device=device)
    return behavior_clone(
        model,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers,
        progress=progress,
    )
