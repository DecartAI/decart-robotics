"""Download files over HTTP(S), with special handling for Google Drive share links.

Google Drive does not serve large files (more than ~25 MB) over a plain GET: instead of the
bytes, it returns an HTML "Google Drive can't scan this file for viruses" interstitial that must
be confirmed with a token cookie. A plain ``urllib`` download therefore silently saves that HTML
page instead of the file. Drive URLs are routed through `gdown <https://pypi.org/project/gdown/>`_,
which handles the confirm token (and resumes), so the demo dataset zip and the pretrained policy
``.pt`` can be hosted on Drive and pulled straight from a share link.

``gdown`` is a dependency of the ``oasis-demo`` package and is also preinstalled on Colab. A clear
error is raised if a Drive URL is fetched without it installed.
"""

from __future__ import annotations

import re
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from decart_oasis.exceptions import DecartRoboticsError

_GDRIVE_HOSTS = {"drive.google.com", "docs.google.com"}


def is_gdrive_url(url: str) -> bool:
    """True if ``url`` points at Google Drive (and so needs the gdown download path)."""
    return urlparse(url).netloc in _GDRIVE_HOSTS


def gdrive_file_id(url: str) -> str | None:
    """Extract the Drive file id from the common share-link shapes, or ``None``.

    Handles ``/file/d/<id>/view``, ``/d/<id>``, and ``?id=<id>`` (uc/open?export=download) URLs.
    """
    parsed = urlparse(url)
    if parsed.netloc not in _GDRIVE_HOSTS:
        return None
    match = re.search(r"/(?:file/)?d/([^/]+)", parsed.path)
    if match:
        return match.group(1)
    ids = parse_qs(parsed.query).get("id")
    return ids[0] if ids else None


def fetch_to_path(url: str, dest: str | Path) -> Path:
    """Download ``url`` to ``dest`` atomically (via a ``.part`` file), returning ``dest``.

    Google Drive URLs go through gdown; everything else is a plain HTTPS download. The temporary
    ``.part`` file is renamed onto ``dest`` only on a complete download, so an interrupted fetch
    never leaves a truncated file looking complete.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        if is_gdrive_url(url):
            _download_gdrive(url, tmp)
        else:
            # Scheme is constrained to http(s) by the callers before reaching here.
            with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
                shutil.copyfileobj(response, out)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _download_gdrive(url: str, dest: Path) -> None:
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - exercised only without gdown installed
        raise DecartRoboticsError(
            "Downloading from Google Drive requires the 'gdown' package. Install it with "
            "`pip install gdown` (it ships with the oasis-demo package)."
        ) from exc
    file_id = gdrive_file_id(url)
    if not file_id:
        raise DecartRoboticsError(
            f"Could not parse a Google Drive file id from: {url}. Expected a share link like "
            "https://drive.google.com/file/d/<id>/view"
        )
    # Use the canonical uc?id= URL so we don't depend on gdown's `fuzzy` URL parsing (added in
    # gdown 4.4); this form works across gdown versions.
    out = gdown.download(f"https://drive.google.com/uc?id={file_id}", str(dest), quiet=False)
    if out is None:
        raise DecartRoboticsError(
            f"Failed to download from Google Drive: {url}\n"
            "Check that the file is shared with 'Anyone with the link' and that the link is "
            "correct. Very large or heavily-downloaded files can also hit a temporary Drive "
            "quota — retry later if so."
        )
