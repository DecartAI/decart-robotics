from __future__ import annotations

import io
from pathlib import Path

import oasis_demo.rl._fetch as fetch
import oasis_demo.rl.pretrained as pretrained
import pytest
import torch
from decart_oasis.exceptions import DecartRoboticsError
from oasis_demo.rl.pretrained import export_policy_state_dict, load_pretrained_policy
from oasis_demo.rl.sft import build_ppo


class _FakeResponse(io.BytesIO):
    """A urlopen() stand-in: a BytesIO usable as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _assert_same_weights(a, b) -> None:
    sa, sb = a.policy.state_dict(), b.policy.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


def test_export_then_load_roundtrips_weights(tmp_path):
    model = build_ppo(device="cpu")
    path = export_policy_state_dict(model, tmp_path / "policy.pt")
    assert path.is_file()

    loaded = load_pretrained_policy(device="cpu", source=str(path))
    _assert_same_weights(model, loaded)


def test_load_without_source_raises(monkeypatch):
    monkeypatch.delenv(pretrained.POLICY_URL_ENV_VAR, raising=False)
    monkeypatch.setattr(pretrained, "PRETRAINED_POLICY_URL", "")
    with pytest.raises(DecartRoboticsError, match="No pretrained policy source"):
        load_pretrained_policy(device="cpu")


def test_load_missing_local_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pretrained_policy(device="cpu", source=str(tmp_path / "missing.pt"))


def test_env_var_provides_the_source(monkeypatch, tmp_path):
    model = build_ppo(device="cpu")
    path = export_policy_state_dict(model, tmp_path / "policy.pt")
    monkeypatch.setenv(pretrained.POLICY_URL_ENV_VAR, str(path))
    loaded = load_pretrained_policy(device="cpu")
    _assert_same_weights(model, loaded)


def test_url_source_downloads_once_then_caches(monkeypatch, tmp_path):
    model = build_ppo(device="cpu")
    payload = export_policy_state_dict(model, tmp_path / "src.pt").read_bytes()
    cache = tmp_path / "cache"

    calls: list[str] = []

    def fake_urlopen(url):
        calls.append(url)
        return _FakeResponse(payload)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    url = "https://example.com/oasis3_ppo.pt"

    loaded = load_pretrained_policy(device="cpu", source=url, cache_dir=str(cache))
    assert (cache / "oasis3_ppo.pt").is_file()
    assert calls == [url]
    _assert_same_weights(model, loaded)

    # Second call reuses the cached file and does not hit the network again.
    load_pretrained_policy(device="cpu", source=url, cache_dir=str(cache))
    assert calls == [url]


def test_gdrive_source_downloads_via_gdown_and_caches(monkeypatch, tmp_path):
    model = build_ppo(device="cpu")
    payload = export_policy_state_dict(model, tmp_path / "src.pt").read_bytes()
    cache = tmp_path / "cache"

    calls: list[str] = []

    def fake_gdown(url, out, quiet=False, fuzzy=False):
        calls.append(url)
        Path(out).write_bytes(payload)
        return out

    # Drive URLs must not touch urllib; route them through a fake gdown instead.
    monkeypatch.setattr(fetch, "_download_gdrive", lambda url, dest: fake_gdown(url, dest))
    url = "https://drive.google.com/file/d/ABC123/view?usp=sharing"

    loaded = load_pretrained_policy(device="cpu", source=url, cache_dir=str(cache))
    # Cache name is keyed on the Drive file id, not the (useless) URL basename.
    assert (cache / "gdrive_ABC123.pt").is_file()
    assert calls == [url]
    _assert_same_weights(model, loaded)

    load_pretrained_policy(device="cpu", source=url, cache_dir=str(cache))
    assert calls == [url]  # reused from cache
