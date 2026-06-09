from __future__ import annotations

from io import BytesIO

import numpy as np
from oasis_demo.visualization import (
    make_camera_strip_animation,
    make_camera_strip_video,
    make_depth_animation,
)
from PIL import Image


def test_make_camera_strip_animation_panorama_over_time():
    animation = make_camera_strip_animation(_camera_frames(), frame_width=32, gutter_width=4)

    assert animation.startswith(b"GIF")
    image = Image.open(BytesIO(animation))
    assert image.size == (32 * 3 + 4 * 2, 32 + 34)
    assert image.n_frames == 4


def test_make_camera_strip_animation_caption_adds_title_row():
    plain = Image.open(
        BytesIO(make_camera_strip_animation(_camera_frames(), frame_width=32, gutter_width=4))
    )
    captioned = Image.open(
        BytesIO(
            make_camera_strip_animation(
                _camera_frames(),
                frame_width=32,
                gutter_width=4,
                caption="episode 1  4 steps  terminated",
            )
        )
    )

    assert captioned.size[0] == plain.size[0]
    assert captioned.size[1] == plain.size[1] + 28
    assert captioned.n_frames == 4


def test_make_camera_strip_video_encodes_mp4():
    video = make_camera_strip_video(
        _camera_frames(size=64),
        fps=8,
        caption="episode 1  4 steps  terminated",
    )

    assert video[4:8] == b"ftyp"  # ISO base media (MP4) signature
    assert len(video) > 0


def test_make_depth_animation_pairs_rgb_and_depth_with_gutter():
    frames = [np.full((8, 8, 3), fill_value=idx * 50, dtype=np.uint8) for idx in range(4)]
    depth_maps = [np.full((8, 8), fill_value=idx, dtype=np.float32) for idx in range(4)]

    animation = make_depth_animation(frames, depth_maps, frame_width=32, gutter_width=4)

    assert animation.startswith(b"GIF")
    image = Image.open(BytesIO(animation))
    assert image.size == (32 * 2 + 4, 32 + 34)
    assert image.n_frames == 4


def _camera_frames(size: int = 8) -> dict[str, list[np.ndarray]]:
    return {
        name: [np.full((size, size, 3), fill_value=idx * 50, dtype=np.uint8) for idx in range(4)]
        for name in ("left_forward", "front", "right_forward")
    }
