"""Notebook visualization helpers for Decart Robotics model outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# IPython is imported lazily inside display_depth_estimates (the only display helper here). The
# byte-producing encoders below need only numpy/PIL/av, so importing this module — and using it
# off a notebook — does not require the [viz] extra.

DEFAULT_CAMERA_ORDER = ("left_forward", "front", "right_forward")


def display_depth_estimates(
    frames: Sequence[np.ndarray],
    depth_maps: Sequence[np.ndarray],
    *,
    title: str = "Depth estimates",
    frame_width: int = 384,
    depth_ignore_bottom_fraction: float | None = None,
) -> None:
    """Display RGB frames and their depth estimates as one labeled GIF."""

    from IPython.display import HTML, display
    from IPython.display import Image as IPythonImage

    display(
        HTML(
            f'<div style="font-family: system-ui, sans-serif; margin: 8px 0 12px;">'
            f'<div style="font-size: 20px; font-weight: 700;">{title}</div>'
            f'<div style="color: #4b5563;">front camera RGB | estimated depth</div>'
            f"</div>"
        )
    )
    display(
        IPythonImage(
            data=make_depth_animation(
                frames,
                depth_maps,
                frame_width=frame_width,
                depth_ignore_bottom_fraction=depth_ignore_bottom_fraction,
            ),
            format="gif",
        )
    )


def make_camera_strip_animation(
    frames: Mapping[str, Sequence[np.ndarray]],
    *,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
    frame_width: int | None = 240,
    metadata: dict[str, str] | None = None,
    duration_ms: int = 180,
    gutter_width: int = 12,
    caption: str | None = None,
) -> bytes:
    """Create a labeled panoramic GIF over time: each frame is left | front | right.

    ``frames`` maps each camera name to a same-length sequence of RGB frames. ``frame_width``
    is the per-camera display width (``None`` keeps the native resolution). An optional
    ``caption`` is drawn once as a slim title row above the camera labels; each frame is
    labeled with its index and the clip length (``i/N``).
    """

    canvases = _compose_strip_frames(
        frames,
        camera_order=camera_order,
        frame_width=frame_width,
        metadata=metadata,
        gutter_width=gutter_width,
        caption=caption,
    )
    return _save_gif(canvases, duration_ms)


def make_camera_strip_video(
    frames: Mapping[str, Sequence[np.ndarray]],
    *,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
    frame_width: int | None = None,
    metadata: dict[str, str] | None = None,
    fps: int = 8,
    gutter_width: int = 12,
    caption: str | None = None,
) -> bytes:
    """Create a labeled panoramic H.264/MP4 clip: each frame is left | front | right.

    Same layout as :func:`make_camera_strip_animation`, but H.264 keeps full-resolution
    frames small and fast to encode (``frame_width=None`` keeps native resolution), which
    suits a live training preview that refreshes repeatedly.
    """

    canvases = _compose_strip_frames(
        frames,
        camera_order=camera_order,
        frame_width=frame_width,
        metadata=metadata,
        gutter_width=gutter_width,
        caption=caption,
    )
    return _encode_h264(canvases, fps)


def make_depth_animation(
    frames: Sequence[np.ndarray],
    depth_maps: Sequence[np.ndarray],
    *,
    frame_width: int = 384,
    duration_ms: int = 220,
    gutter_width: int = 12,
    depth_ignore_bottom_fraction: float | None = None,
) -> bytes:
    """Create a labeled GIF pairing RGB frames with normalized depth maps."""

    if not frames:
        raise ValueError("frames must contain at least one RGB frame")
    if len(frames) != len(depth_maps):
        raise ValueError("frames and depth_maps must have the same length")

    first = frames[0]
    frame_height = max(1, round(frame_width * first.shape[0] / first.shape[1]))
    label_h = 34
    canvas_width = frame_width * 2 + gutter_width
    gif_frames = []

    for t, (frame, depth) in enumerate(zip(frames, depth_maps, strict=True)):
        canvas = Image.new("RGB", (canvas_width, frame_height + label_h), (229, 231, 235))
        draw = ImageDraw.Draw(canvas)

        _draw_label(draw, (0, 0, frame_width, label_h), f"front RGB  t+{t}")
        canvas.paste(_resize_frame(frame, frame_width, frame_height), (0, label_h))

        depth_x = frame_width + gutter_width
        _draw_label(draw, (depth_x, 0, depth_x + frame_width, label_h), f"depth  t+{t}")
        depth_image = _resize_frame(_colorize_depth(depth), frame_width, frame_height)
        canvas.paste(depth_image, (depth_x, label_h))
        _draw_depth_cutoff(
            draw,
            (depth_x, label_h, depth_x + frame_width, label_h + frame_height),
            depth_ignore_bottom_fraction,
        )

        gif_frames.append(canvas)

    return _save_gif(gif_frames, duration_ms)


def _save_gif(frames: list[Image.Image], duration_ms: int) -> bytes:
    """Encode RGB frames into a looping GIF.

    Each frame is quantized with the fast octree quantizer; PIL's default GIF quantizer is
    ~40x slower for clips of dozens of frames, which matters for the live training preview.
    """

    palette_frames = [
        frame.quantize(colors=256, method=Image.Quantize.FASTOCTREE) for frame in frames
    ]
    buffer = BytesIO()
    palette_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return buffer.getvalue()


def _compose_strip_frames(
    frames: Mapping[str, Sequence[np.ndarray]],
    *,
    camera_order: Sequence[str],
    frame_width: int | None,
    metadata: dict[str, str] | None,
    gutter_width: int,
    caption: str | None,
) -> list[Image.Image]:
    """Build the per-timestep labeled panorama canvases shared by the GIF/video encoders."""

    streams = [name for name in camera_order if name in frames]
    if not streams:
        raise ValueError("none of the requested streams are present")

    frame_count = min(len(frames[name]) for name in streams)
    if frame_count == 0:
        raise ValueError("streams contain no frames")

    first = frames[streams[0]][0]
    if frame_width is None:
        width, height = int(first.shape[1]), int(first.shape[0])
    else:
        width = frame_width
        height = max(1, round(frame_width * first.shape[0] / first.shape[1]))

    label_h = 34
    caption_h = 28 if caption else 0
    canvas_width = width * len(streams) + gutter_width * (len(streams) - 1)
    canvas_height = caption_h + height + label_h
    canvases = []

    for t in range(frame_count):
        canvas = Image.new("RGB", (canvas_width, canvas_height), (229, 231, 235))
        draw = ImageDraw.Draw(canvas)
        if caption:
            _draw_label(draw, (0, 0, canvas_width, caption_h), caption)
        suffix = f"{t + 1}/{frame_count}"  # frame index / number of frames in the clip
        for col, name in enumerate(streams):
            x = col * (width + gutter_width)
            _draw_label(
                draw,
                (x, caption_h, x + width, caption_h + label_h),
                f"{_label(name, metadata)}  {suffix}",
            )
            source = frames[name][t]
            if frame_width is None:
                frame = Image.fromarray(np.asarray(source).astype(np.uint8), mode="RGB")
            else:
                frame = _resize_frame(source, width, height)
            canvas.paste(frame, (x, caption_h + label_h))
        canvases.append(canvas)

    return canvases


def _encode_h264(canvases: list[Image.Image], fps: int) -> bytes:
    """Encode RGB canvases into an H.264/MP4 clip (small even for full-resolution frames)."""

    import av

    arrays = [np.asarray(canvas) for canvas in canvases]
    height, width = arrays[0].shape[:2]
    # H.264 with yuv420p needs even dimensions.
    pad_h, pad_w = height % 2, width % 2
    if pad_h or pad_w:
        arrays = [np.pad(a, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") for a in arrays]
        height, width = arrays[0].shape[:2]

    buffer = BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=max(1, fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "ultrafast", "crf": "28"}
    for array in arrays:
        video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def _label(name: str, metadata: dict[str, str] | None) -> str:
    if metadata and name in metadata:
        return f"{name} ({metadata[name]})"
    return name


def _resize_frame(frame: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        min_depth = float(np.min(finite))
        max_depth = float(np.max(finite))
        if max_depth <= min_depth:
            normalized = np.zeros_like(depth, dtype=np.float32)
        else:
            normalized = np.clip((depth - min_depth) / (max_depth - min_depth), 0.0, 1.0)

    near = 1.0 - normalized
    rgb = np.stack(
        [
            near,
            np.sqrt(np.clip(normalized, 0.0, 1.0)),
            normalized,
        ],
        axis=-1,
    )
    return (rgb * 255.0).astype(np.uint8)


def _draw_depth_cutoff(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    ignore_bottom_fraction: float | None,
) -> None:
    if ignore_bottom_fraction is None or ignore_bottom_fraction <= 0.0:
        return
    left, top, right, bottom = box
    fraction = min(ignore_bottom_fraction, 1.0)
    y = top + int(round((bottom - top) * (1.0 - fraction)))
    y = max(top, min(bottom - 1, y))
    draw.line((left, y - 1, right, y - 1), fill=(17, 24, 39), width=1)
    draw.line((left, y, right, y), fill=(255, 255, 255), width=2)


def _draw_label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str) -> None:
    draw.rectangle(box, fill=(17, 24, 39))
    font = ImageFont.load_default()
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + max(4, (right - left - text_w) // 2)
    y = top + max(2, (bottom - top - text_h) // 2)
    draw.text((x, y), text, fill="white", font=font)
