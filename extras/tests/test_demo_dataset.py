from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import oasis_demo.rl._fetch as fetch
import oasis_demo.rl.sft as sft
import pytest
from oasis_demo.rl._fetch import gdrive_file_id, is_gdrive_url
from oasis_demo.rl.sft import download_demo_dataset


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _make_episodes(base: Path) -> None:
    for name in ("episode_0", "episode_1"):
        episode = base / name
        episode.mkdir(parents=True)
        (episode / "manifest.jsonl").write_text("{}\n")


def _targz_bytes(base: Path, root: str = "") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for episode in sorted(base.iterdir()):
            arcname = f"{root}/{episode.name}" if root else episode.name
            tf.add(episode, arcname=arcname)
    return buf.getvalue()


def _serve(monkeypatch, payload: bytes) -> list[str]:
    calls: list[str] = []

    def fake_urlopen(url):
        calls.append(url)
        return _FakeResponse(payload)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_download_demo_dataset_flat_archive_and_reuse(monkeypatch, tmp_path):
    src = tmp_path / "src"
    _make_episodes(src)
    calls = _serve(monkeypatch, _targz_bytes(src))

    dest = tmp_path / "out"
    root = download_demo_dataset(dest_dir=str(dest), source="https://x/demo.tar.gz")
    assert (root / "episode_0" / "manifest.jsonl").is_file()
    assert sorted(p.name for p in root.glob("episode_*")) == ["episode_0", "episode_1"]
    assert calls == ["https://x/demo.tar.gz"]

    # An already-extracted dataset is reused without re-downloading.
    root_again = download_demo_dataset(dest_dir=str(dest), source="https://x/demo.tar.gz")
    assert root_again == root
    assert calls == ["https://x/demo.tar.gz"]


def test_download_demo_dataset_nested_archive(monkeypatch, tmp_path):
    src = tmp_path / "src"
    _make_episodes(src)
    _serve(monkeypatch, _targz_bytes(src, root="dataset"))

    dest = tmp_path / "out"
    root = download_demo_dataset(dest_dir=str(dest), source="https://x/demo.tar.gz")
    # Episodes nested under a single top-level folder are found one level down.
    assert root.name == "dataset"
    assert (root / "episode_0" / "manifest.jsonl").is_file()


def test_download_demo_dataset_zip(monkeypatch, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("episode_0/manifest.jsonl", "{}\n")
        zf.writestr("episode_1/manifest.jsonl", "{}\n")
    _serve(monkeypatch, buf.getvalue())

    dest = tmp_path / "out"
    root = download_demo_dataset(dest_dir=str(dest), source="https://x/demo.zip")
    assert (root / "episode_0" / "manifest.jsonl").is_file()


def test_download_demo_dataset_without_source_raises(monkeypatch):
    monkeypatch.delenv(sft.DEMO_DATA_URL_ENV_VAR, raising=False)
    monkeypatch.setattr(sft, "DEMO_DATASET_URL", "")
    with pytest.raises(ValueError, match="No demo dataset source"):
        download_demo_dataset(dest_dir="unused")


def test_find_episode_root_empty(tmp_path):
    assert sft._find_episode_root(tmp_path) is None  # empty dir has no episodes


def test_download_demo_dataset_from_gdrive(monkeypatch, tmp_path):
    # Drive URLs are extension-less, so extraction must sniff the content (here: a tar.gz served
    # by a fake gdown) rather than trust the URL/temp-file suffix.
    src = tmp_path / "src"
    _make_episodes(src)
    payload = _targz_bytes(src)

    calls: list[str] = []

    def fake_gdrive(url, dest):
        calls.append(url)
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(fetch, "_download_gdrive", fake_gdrive)
    url = "https://drive.google.com/file/d/EP1S0DE/view?usp=sharing"

    root = download_demo_dataset(dest_dir=str(tmp_path / "out"), source=url)
    assert (root / "episode_0" / "manifest.jsonl").is_file()
    assert calls == [url]


def test_extract_archive_rejects_non_archive(tmp_path):
    junk = tmp_path / "not_an_archive"
    junk.write_bytes(b"<html>Google Drive virus scan interstitial</html>")
    with pytest.raises(ValueError, match="neither a zip nor a tar"):
        sft._extract_archive(junk, tmp_path / "out")


def test_gdrive_url_detection_and_id():
    share = "https://drive.google.com/file/d/ABC123/view?usp=sharing"
    assert is_gdrive_url(share)
    assert gdrive_file_id(share) == "ABC123"
    assert gdrive_file_id("https://drive.google.com/uc?export=download&id=XYZ789") == "XYZ789"
    assert not is_gdrive_url("https://example.com/demo.zip")
    assert gdrive_file_id("https://example.com/demo.zip") is None
