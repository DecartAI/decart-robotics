from __future__ import annotations

from pathlib import Path

import nbformat

# The notebook lives at <repo-root>/notebook/, outside this extras/ project; resolve it relative to
# this test file so collection works regardless of the current working directory.
COLAB_NOTEBOOK = Path(__file__).resolve().parents[2] / "notebook" / "train_oasis3_ppo_colab.ipynb"


def _notebook(path):
    return nbformat.read(Path(path), as_version=4)


def test_training_notebook_is_valid():
    nbformat.validate(_notebook(COLAB_NOTEBOOK))


def test_colab_quick_check_shows_synced_risk():
    code = "\n".join(
        c.source for c in _notebook(COLAB_NOTEBOOK).cells if c.cell_type == "code"
    )
    # The preview gets the per-frame scorer, which is what drives the playback-synced risk bar.
    assert "risk_fn=" in code
    assert "collision_risk" in code
