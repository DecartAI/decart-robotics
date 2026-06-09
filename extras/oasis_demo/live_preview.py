"""Live notebook preview of a stream of camera frames.

``LiveCameraPreview`` is a :class:`~decart_oasis.types.FrameConsumer`: hand it to
an :class:`~decart_oasis.client.A2VClient` (``A2VClient(frame_consumer=...)``)
and it renders the frames the client generates as a rolling ``left | front | right``
video in a notebook, refreshing as new frames arrive. The frame source never needs
to know a preview exists.

All real work happens on a background daemon thread. ``submit`` only hands the most
recent generated chunk to that worker as a reference — a zero-copy queue put, so the
producing thread is not slowed down. The worker keeps those full-resolution frames
in a fixed-size sliding window of the most recent ``max_frames`` frames, so memory
stays bounded no matter how long a clip runs and the video plays the latest driving
at real speed (rather than time-lapsing the whole clip). About once per clip length
the worker encodes the window into a panoramic H.264/MP4 clip at ``fps`` and pushes
it into a persistent ``<video>`` element; a tiny JS snippet swaps the source on the
video's own ``ended`` event so the clip rolls continuously and each swap lands at a
loop boundary. H.264 keeps even full-resolution clips small and fast to encode; if
no H.264 encoder is available it falls back to a GIF (replaced in place).

``new_clip`` marks a boundary (a fresh prompt / RL episode): the worker renders the
just-finished window one last time, then clears it so the next frames start a new
clip. The worker runs at its own pace; if it falls behind (e.g. during an encode)
the oldest pending chunk is dropped rather than blocking the producer — the preview
simply loses a few frames. ``max_pending`` bounds how many raw chunks may be in
flight. ``max_frames`` is both the window length and the main memory knob:
full-resolution frames are ~1 MB each, so the default 512 (~max_frames / fps
seconds of video) hold ~1.8 GB — lower it if memory is tight.

Pass ``risk_fn`` (e.g. ``DepthCollisionReward.collision_risk``) to overlay a live
collision-risk bar under the video. The worker scores every front frame, so the risk
track aligns 1:1 with the encoded clip; the bar is driven by the video's own playback
clock (``currentTime``), so as the looping clip replays the bar shows the risk of the
frame *currently on screen* — not just the most recent chunk.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from oasis_demo.visualization import (
    DEFAULT_CAMERA_ORDER,
    make_camera_strip_animation,
    make_camera_strip_video,
)

logger = logging.getLogger(__name__)

# Pushed (via a Javascript display handle) on every new clip. It never replaces the <video>
# element — it stashes the next clip and swaps the source on the video's own ``ended`` event,
# so each swap lands exactly at the loop boundary instead of mid-playback.
_PREVIEW_VIDEO_JS = """
(function () {
  var v = document.getElementById("__DOM_ID__");
  if (!v) return;
  v.__epNext = "__SRC__";
  if (!v.__epInit) {
    v.__epInit = true;
    v.onended = function () {
      if (v.__epNext) { v.src = v.__epNext; v.__epNext = null; }
      else { v.currentTime = 0; }
      v.play();
    };
    v.src = v.__epNext; v.__epNext = null; v.play();
  }
})();
"""

# Same double-buffered swap as above, plus a per-frame risk track. ``__RISKS__`` is a JSON array
# whose i-th entry is the collision risk of the i-th frame in the clip. A requestAnimationFrame
# loop reads the video's own ``currentTime``, maps it to a frame index via ``fps``, and paints the
# risk bar for the frame on screen — so the bar stays in sync as the clip replays. The risk array
# is double-buffered alongside the src and swapped on the same ``ended`` boundary, keeping the bar
# aligned with whichever clip is actually playing.
_PREVIEW_VIDEO_RISK_JS = """
(function () {
  var v = document.getElementById("__DOM_ID__");
  if (!v) return;
  var bar = document.getElementById("__RISK_ID__");
  var fps = __FPS__, thr = __THRESHOLD__;
  v.__epNext = "__SRC__";
  v.__epNextRisks = __RISKS__;
  function colorFor(r) { return r < 0.5 ? "#16a34a" : (r < 0.85 ? "#d97706" : "#dc2626"); }
  function paint(r) {
    if (!bar) return;
    var c = Math.max(0, Math.min(1, r));
    var pct = Math.round(c * 100);
    var banner = (r >= thr)
      ? '<div style="margin-top:6px;font-weight:700;color:#dc2626;">' +
        '\\u26a0 COLLISION DETECTED</div>'
      : '';
    bar.innerHTML =
      '<div style="display:flex;align-items:center;gap:8px;">' +
        '<span style="font-size:13px;color:#4b5563;font-weight:600;">Collision risk</span>' +
        '<div style="flex:1;height:14px;background:#e5e7eb;border-radius:7px;overflow:hidden;">' +
          '<div style="width:' + pct + '%;height:100%;background:' + colorFor(c) + ';"></div>' +
        '</div>' +
        '<span style="font-variant-numeric:tabular-nums;font-weight:700;min-width:3.2em;' +
        'text-align:right;">' + c.toFixed(2) + '</span>' +
      '</div>' + banner;
  }
  if (!v.__epInit) {
    v.__epInit = true;
    v.onended = function () {
      if (v.__epNext) { v.src = v.__epNext; v.__epRisks = v.__epNextRisks; v.__epNext = null; }
      else { v.currentTime = 0; }
      v.play();
    };
    function tick() {
      var rs = v.__epRisks;
      if (rs && rs.length) {
        var i = Math.floor((v.currentTime || 0) * fps);
        if (i < 0) i = 0;
        if (i >= rs.length) i = rs.length - 1;
        paint(rs[i]);
      }
      requestAnimationFrame(tick);
    }
    v.src = v.__epNext; v.__epRisks = v.__epNextRisks; v.__epNext = null; v.play();
    requestAnimationFrame(tick);
  }
})();
"""


class LiveCameraPreview:
    """Render a stream of camera frames as a live, looping video in a notebook."""

    def __init__(
        self,
        *,
        max_frames: int = 512,
        frame_width: int | None = None,
        fps: int = 20,
        min_refresh_seconds: float = 1.0,
        camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
        background: bool = True,
        max_pending: int = 32,
        risk_fn: Callable[[np.ndarray], float] | None = None,
        risk_threshold: float = 1.0,
        risk_smoothing_seconds: float = 0.5,
    ) -> None:
        self.frame_width = frame_width
        self.max_frames = max(2, max_frames)
        self.fps = max(1, fps)
        self.min_refresh_seconds = max(0.0, min_refresh_seconds)
        self.camera_order = tuple(camera_order)
        self.background = background
        self.max_pending = max(1, max_pending)
        # Optional per-frame collision-risk scorer; when set, the worker scores every front frame
        # and the preview draws a playback-synced risk bar under the video.
        self.risk_fn = risk_fn
        self.risk_threshold = risk_threshold
        self.risk_smoothing_seconds = max(0.0, risk_smoothing_seconds)
        # ``_frames`` is only ever touched by the worker thread (or, when ``background`` is
        # False, by the caller in-line) — never concurrently. It is a fixed-size sliding
        # window: once full it keeps the most recent ``max_frames`` frames (the latest
        # driving at real speed), rather than time-lapsing the whole clip.
        self._frames: dict[str, deque[np.ndarray]] = {
            name: deque(maxlen=self.max_frames) for name in self.camera_order
        }
        # Per-front-frame risk, slid in lockstep with ``_frames["front"]`` so the risk track and
        # the encoded clip share an index space.
        self._risks: deque[float] = deque(maxlen=self.max_frames)
        self._clip_index = 1
        self._last_render_at = 0.0
        self._dom_id = f"decart-live-preview-{uuid.uuid4().hex[:8]}"
        self._risk_dom_id = f"{self._dom_id}-risk"
        self._video_handle = None  # persistent <video> element (never replaced for mp4)
        self._js_handle = None  # pushes each new clip into that element
        self._queue: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        self._started = False

    # --- lifecycle -------------------------------------------------------------

    def start(self) -> LiveCameraPreview:
        """Create the notebook ``<video>`` element and spin up the background worker."""
        if self._started:
            return self
        self._started = True
        self._ensure_video_element()
        self._last_render_at = time.perf_counter()
        if self.background:
            self._queue = queue.Queue(maxsize=self.max_pending)
            self._worker = threading.Thread(
                target=self._render_worker, name="live-camera-preview", daemon=True
            )
            self._worker.start()
        return self

    def close(self) -> None:
        """Drain pending chunks and stop the background worker."""
        if self._worker is None or self._queue is None:
            return
        try:
            # FIFO: the worker drains any pending chunks first, then exits on the sentinel.
            # Bounded put + daemon worker mean shutdown never blocks on the preview.
            self._queue.put(None, timeout=5.0)
        except queue.Full:
            pass
        self._worker.join(timeout=30.0)
        self._worker = None

    def __enter__(self) -> LiveCameraPreview:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- FrameConsumer interface (producer thread) -----------------------------

    def submit(self, frames: Mapping[str, Sequence[np.ndarray]]) -> None:
        """Hand the latest generated chunk to the preview (returns immediately)."""
        if self.background:
            self._enqueue(("frames", frames))
        else:
            self._handle_frames(frames)

    def new_clip(self) -> None:
        """Mark a clip boundary: render the finished window, then start fresh."""
        if self.background:
            self._enqueue(("clip", None))
        else:
            self._handle_clip()

    # --- worker ----------------------------------------------------------------

    def _enqueue(self, msg: tuple) -> None:
        if self._queue is None:
            return
        if msg[0] != "frames":
            # Boundary markers (e.g. "clip") must never be dropped — losing one merges two
            # episodes into a single preview clip and desyncs the clip index. Block briefly to
            # hand it off; the daemon worker drains continuously, so this returns promptly.
            self._queue.put(msg)
            return
        # Frame chunks are droppable: never block the producer. If the queue is full, drop the
        # oldest pending chunk and enqueue the new one so the preview tracks the latest driving.
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(msg)
            except queue.Full:
                logger.debug("live preview: dropped a frame chunk under queue contention")

    def _render_worker(self) -> None:
        assert self._queue is not None
        while True:
            msg = self._queue.get()
            if msg is None:
                return
            kind, payload = msg
            try:
                if kind == "frames":
                    self._handle_frames(payload)
                elif kind == "clip":
                    self._handle_clip()
            except Exception:
                # A preview failure must never take down the producer (e.g. training); log at
                # debug level so a persistently-dark preview is at least diagnosable.
                logger.debug("live preview: render step failed", exc_info=True)

    def _handle_frames(self, frames: Mapping[str, Sequence[np.ndarray]]) -> None:
        self._ingest_step(frames)
        if self._due_for_refresh(time.perf_counter()):
            self._render_preview(running=True)

    def _handle_clip(self) -> None:
        if self._render_preview(running=False):
            self._clip_index += 1
        self._reset_buffer()

    def _due_for_refresh(self, now: float) -> bool:
        # Refresh about once per clip-play-duration so each swap lands near the loop point
        # (floored by ``min_refresh_seconds`` so short early clips don't re-encode too often).
        frame_count = max((len(buf) for buf in self._frames.values()), default=0)
        if frame_count == 0:
            return False
        interval = max(self.min_refresh_seconds, frame_count / self.fps)
        return (now - self._last_render_at) >= interval

    # --- buffering (worker thread) ---------------------------------------------

    def _ingest_step(self, frames: Mapping[str, Sequence[np.ndarray]]) -> None:
        present = [n for n in self.camera_order if frames.get(n) is not None and len(frames[n])]
        if not present:
            return
        length = min(len(frames[n]) for n in present)
        score_risk = self.risk_fn is not None and "front" in present
        for t in range(length):
            for name in present:
                # Copy on ingest (worker thread, off the producer's path): the client also hands
                # these same arrays back to its caller via A2VResult, and some decoders (VP9)
                # return writable arrays — owning a copy keeps a buffered frame from being mutated
                # under the worker mid-encode. The bounded deques drop the oldest once full.
                self._frames[name].append(np.array(frames[name][t], dtype=np.uint8))
            if score_risk:
                # Score every front frame so the risk track aligns 1:1 with the video frames and
                # both bounded deques slide together. A scorer failure must never break the
                # preview, so fall back to 0.0.
                try:
                    risk = float(self.risk_fn(np.asarray(frames["front"][t])))
                except Exception:
                    risk = 0.0
                self._risks.append(risk)

    def _reset_buffer(self) -> None:
        self._frames = {name: deque(maxlen=self.max_frames) for name in self.camera_order}
        self._risks = deque(maxlen=self.max_frames)

    def _smoothed_risks(self, risks: list[float]) -> list[float]:
        # Centered moving average over a ~``risk_smoothing_seconds`` window so the bar reflects a
        # short-term trend instead of per-frame jitter. Edge frames average over the partial window
        # available. A single-frame spike is damped, so the collision banner reflects a sustained
        # near-collision rather than a momentary blip.
        window = max(1, round(self.fps * self.risk_smoothing_seconds))
        if window <= 1 or len(risks) <= 1:
            return risks
        half = window // 2
        n = len(risks)
        smoothed = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed.append(sum(risks[lo:hi]) / (hi - lo))
        return smoothed

    # --- rendering (worker thread) ---------------------------------------------

    def _render_preview(self, *, running: bool) -> bool:
        snapshot = {
            name: list(self._frames[name]) for name in self.camera_order if self._frames[name]
        }
        if not snapshot:
            return False
        frame_count = max(len(frames) for frames in snapshot.values())
        caption = self._caption(running=running, frame_count=frame_count)
        data, kind = self._encode(snapshot, caption)
        risks = self._smoothed_risks(list(self._risks)) if self.risk_fn is not None else None
        self._update_display(data, kind, risks)
        self._last_render_at = time.perf_counter()
        return True

    def _encode(self, snapshot: dict, caption: str | None) -> tuple[bytes, str]:
        try:
            video = make_camera_strip_video(
                snapshot,
                camera_order=self.camera_order,
                frame_width=self.frame_width,
                fps=self.fps,
                caption=caption,
            )
            return video, "mp4"
        except Exception:
            # No H.264 encoder available (or it failed): fall back to a GIF.
            gif = make_camera_strip_animation(
                snapshot,
                camera_order=self.camera_order,
                frame_width=self.frame_width or 240,
                duration_ms=max(1, round(1000 / self.fps)),
                caption=caption,
            )
            return gif, "gif"

    def _build_animation(self, caption: str | None = None) -> tuple[bytes, str]:
        return self._encode(
            {name: list(self._frames[name]) for name in self.camera_order if self._frames[name]},
            caption,
        )

    def _ensure_video_element(self) -> None:
        from IPython.display import HTML, display

        if self._video_handle is not None:
            return
        video_html = (
            f'<video id="{self._dom_id}" autoplay muted playsinline '
            'style="max-width: 100%; background: #111;"></video>'
        )
        if self.risk_fn is not None:
            # A risk bar sits directly under the video; the per-clip JS keeps it in sync with
            # playback. Created up front so the element exists before any JS runs against it.
            video_html = (
                "<div>"
                + video_html
                + f'<div id="{self._risk_dom_id}" '
                'style="font-family: system-ui, sans-serif; margin-top: 6px;"></div>'
                "</div>"
            )
        self._video_handle = display(HTML(video_html), display_id=True)

    def _video_js(self, src: str, risks: list[float] | None) -> str:
        if not risks:
            return _PREVIEW_VIDEO_JS.replace("__DOM_ID__", self._dom_id).replace("__SRC__", src)
        return (
            _PREVIEW_VIDEO_RISK_JS.replace("__DOM_ID__", self._dom_id)
            .replace("__RISK_ID__", self._risk_dom_id)
            .replace("__FPS__", str(self.fps))
            .replace("__THRESHOLD__", repr(float(self.risk_threshold)))
            .replace("__RISKS__", json.dumps([round(float(r), 4) for r in risks]))
            .replace("__SRC__", src)
        )

    def _update_display(self, data: bytes, kind: str, risks: list[float] | None = None) -> None:
        from IPython.display import Image as IPythonImage
        from IPython.display import Javascript, display

        if kind == "mp4":
            self._ensure_video_element()
            src = "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")
            js = self._video_js(src, risks)
            if self._js_handle is None:
                self._js_handle = display(Javascript(js), display_id=True)
            else:
                self._js_handle.update(Javascript(js))
        else:
            # GIF fallback (no H.264 encoder): no in-place swap, just replace the output. A GIF
            # has no playback clock to drive the risk bar, so the synced bar is mp4-only.
            image = IPythonImage(data=data, format="gif")
            if self._video_handle is not None:
                self._video_handle.update(image)
            else:
                self._video_handle = display(image, display_id=True)

    def _caption(self, *, running: bool, frame_count: int) -> str:
        suffix = "  (running)" if running else ""
        return f"clip {self._clip_index}  ·  {frame_count} frames{suffix}"
