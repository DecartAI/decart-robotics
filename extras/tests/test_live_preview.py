from __future__ import annotations

import queue

import numpy as np
import pytest
from oasis_demo.live_preview import LiveCameraPreview


def _chunk(
    n_frames: int = 4, size: int = 8, fill: int | None = None
) -> dict[str, list[np.ndarray]]:
    return {
        name: [
            np.full((size, size, 3), fill if fill is not None else i * 10, dtype=np.uint8)
            for i in range(n_frames)
        ]
        for name in ("left_forward", "front", "right_forward")
    }


def test_keeps_most_recent_window():
    preview = LiveCameraPreview(background=False, max_frames=10)

    for chunk_id in range(50):  # 50 chunks x 4 frames = 200 incoming frames
        preview._ingest_step(_chunk(fill=chunk_id))  # tag every frame of this chunk

    lengths = {name: len(frames) for name, frames in preview._frames.items()}
    assert set(lengths) == {"left_forward", "front", "right_forward"}
    assert set(lengths.values()) == {10}  # exactly the window size, all cameras aligned
    # The window holds the most recent frames (the tail), not a time-lapse of the whole run.
    kept = [int(frame[0, 0, 0]) for frame in preview._frames["front"]]
    assert kept == [47, 47, 48, 48, 48, 48, 49, 49, 49, 49]


def test_ingest_copies_frames_so_caller_mutation_is_isolated():
    # The client hands the same arrays to the preview and back to its caller; some decoders return
    # writable arrays. Ingest must copy so a later caller mutation can't corrupt a buffered frame.
    preview = LiveCameraPreview(background=False, max_frames=10)
    chunk = _chunk(n_frames=1, fill=7)
    preview._ingest_step(chunk)

    for name in ("left_forward", "front", "right_forward"):
        chunk[name][0][:] = 200  # mutate the original arrays after submitting

    kept = [int(preview._frames[name][0][0, 0, 0]) for name in preview._frames]
    assert kept == [7, 7, 7]  # buffered copies are unaffected by the post-submit mutation


def test_enqueue_drops_oldest_frame_but_never_a_clip_marker():
    preview = LiveCameraPreview(background=True, max_pending=2)
    preview._queue = queue.Queue(maxsize=2)

    preview._enqueue(("frames", "a"))
    preview._enqueue(("frames", "b"))  # queue now full with a, b
    preview._enqueue(("frames", "c"))  # full -> drop oldest (a), enqueue c

    drained = []
    try:
        while True:
            drained.append(preview._queue.get_nowait())
    except queue.Empty:
        pass

    # Oldest frame chunk dropped; the queue tracks the latest frames.
    assert drained == [("frames", "b"), ("frames", "c")]


def test_enqueue_blocks_for_clip_marker_until_room(monkeypatch):
    # A clip marker must not be dropped even when the queue is momentarily full: it blocks on a
    # real put(). Simulate "room frees up" by having the blocking put succeed after a get.
    preview = LiveCameraPreview(background=True, max_pending=1)
    preview._queue = queue.Queue(maxsize=1)
    preview._queue.put(("frames", "a"))  # full

    puts = []
    real_put = preview._queue.put

    def fake_put(item, *args, **kwargs):
        # First call: make room (mimic the worker draining), then delegate to the real blocking put.
        if not puts:
            preview._queue.get_nowait()
        puts.append(item)
        return real_put(item)

    monkeypatch.setattr(preview._queue, "put", fake_put)
    preview._enqueue(("clip", None))

    assert puts == [("clip", None)]
    assert preview._queue.get_nowait() == ("clip", None)


def test_builds_video():
    preview = LiveCameraPreview(background=False, max_frames=10)
    for _ in range(5):
        preview._ingest_step(_chunk(size=64))

    data, kind = preview._build_animation(caption="clip 1  ·  20 frames")

    assert kind == "mp4"
    assert data[4:8] == b"ftyp"  # ISO base media (MP4) signature


def test_caption_reports_clip_and_frame_count():
    preview = LiveCameraPreview(background=False)

    running = preview._caption(running=True, frame_count=12)
    final = preview._caption(running=False, frame_count=40)

    assert "clip 1" in running and "12 frames" in running and "running" in running
    assert "clip 1" in final and "40 frames" in final and "running" not in final


def test_refresh_is_timed_to_clip_length():
    preview = LiveCameraPreview(background=False, fps=30, min_refresh_seconds=1.0)
    preview._last_render_at = 100.0

    assert preview._due_for_refresh(now=200.0) is False  # nothing buffered yet

    for _ in range(3):
        preview._ingest_step(_chunk())  # 12 frames -> 0.4s clip, floored to 1.0s
    assert preview._due_for_refresh(now=100.5) is False
    assert preview._due_for_refresh(now=101.5) is True

    for _ in range(15):
        preview._ingest_step(_chunk())  # 72 frames -> 72/30 = 2.4s interval
    preview._last_render_at = 100.0
    assert preview._due_for_refresh(now=102.0) is False  # 2.0s < 2.4s
    assert preview._due_for_refresh(now=103.0) is True  # 3.0s >= 2.4s


def test_new_clip_renders_resets_and_advances(monkeypatch):
    # min_refresh_seconds huge -> no mid-clip refresh; only the new_clip render fires.
    preview = LiveCameraPreview(background=False, min_refresh_seconds=1e9)
    renders = []
    monkeypatch.setattr(
        preview, "_update_display", lambda data, kind, risks=None: renders.append(kind)
    )

    for _ in range(3):
        preview._handle_frames(_chunk(size=64))
    preview.new_clip()

    assert len(renders) == 1
    assert all(len(frames) == 0 for frames in preview._frames.values())
    assert preview._clip_index == 2


def test_new_clip_on_empty_buffer_does_not_advance():
    # The first prompt fires new_clip before any frames; it must not skip a clip number.
    preview = LiveCameraPreview(background=False)

    preview.new_clip()

    assert preview._clip_index == 1


def test_reset_clears_buffer():
    preview = LiveCameraPreview(background=False, max_frames=10)
    for _ in range(5):
        preview._ingest_step(_chunk())
    assert any(preview._frames.values())

    preview._reset_buffer()

    assert all(len(frames) == 0 for frames in preview._frames.values())


def test_risk_track_aligns_with_front_frames():
    # Score each front frame by its pixel value so we can assert exact alignment.
    preview = LiveCameraPreview(
        background=False, max_frames=10, risk_fn=lambda f: float(f[0, 0, 0]) / 255.0
    )
    for chunk_id in range(5):  # 5 chunks x 4 frames; window keeps the last 10
        preview._ingest_step(_chunk(fill=chunk_id))

    fronts = [int(f[0, 0, 0]) for f in preview._frames["front"]]
    risks = list(preview._risks)
    assert len(risks) == len(fronts)  # risk track slides in lockstep with the front window
    assert risks == [value / 255.0 for value in fronts]


def test_risk_track_omitted_without_risk_fn():
    preview = LiveCameraPreview(background=False)
    for _ in range(3):
        preview._ingest_step(_chunk())
    assert len(preview._risks) == 0


def test_reset_clears_risk_track():
    preview = LiveCameraPreview(background=False, risk_fn=lambda f: 0.5)
    for _ in range(3):
        preview._ingest_step(_chunk())
    assert len(preview._risks) > 0

    preview._reset_buffer()

    assert len(preview._risks) == 0


def test_video_js_embeds_synced_risk_track():
    preview = LiveCameraPreview(background=False, fps=20, risk_threshold=0.8, risk_fn=lambda f: 0.5)

    js = preview._video_js("data:video/mp4;base64,AAAA", [0.1, 0.95])

    assert preview._dom_id in js
    assert preview._risk_dom_id in js  # the bar element it drives
    assert "[0.1, 0.95]" in js  # the per-frame risk array
    assert "currentTime" in js  # synced to the video's own playback clock
    assert "0.8" in js  # collision threshold


def test_risk_smoothing_dampens_spikes():
    preview = LiveCameraPreview(
        background=False, fps=20, risk_smoothing_seconds=0.5, risk_fn=lambda f: 0.0
    )
    raw = [0.0] * 20
    raw[10] = 1.0  # a single-frame spike

    smoothed = preview._smoothed_risks(raw)

    assert len(smoothed) == len(raw)
    assert smoothed[10] < raw[10]  # the spike is averaged down
    assert max(smoothed) < 0.5
    # a flat signal passes through unchanged
    assert preview._smoothed_risks([0.4] * 20) == pytest.approx([0.4] * 20)


def test_risk_smoothing_disabled_returns_raw():
    preview = LiveCameraPreview(background=False, risk_smoothing_seconds=0.0, risk_fn=lambda f: 0.0)
    raw = [0.0, 1.0, 0.0, 1.0]

    assert preview._smoothed_risks(raw) == raw


def test_video_js_without_risk_uses_plain_template():
    preview = LiveCameraPreview(background=False)

    js = preview._video_js("data:video/mp4;base64,AAAA", None)

    assert preview._dom_id in js
    assert "__epNextRisks" not in js  # no risk track wired in
