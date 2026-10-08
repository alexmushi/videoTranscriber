"""Grab a screenshot each time the picture meaningfully changes.

The video is sampled at a low rate as tiny grayscale thumbnails. Each sample is
compared with the last *captured* frame (so slow, gradual changes still add up
and trigger a capture). When a change is detected we wait for the picture to
settle (to avoid grabbing a slide mid-transition) and then save a full-size
screenshot at that moment.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

from .util import fmt_ts

THUMB_W, THUMB_H = 160, 90
PIXEL_DELTA = 30  # grayscale levels a pixel must move to count as "changed"


def _changed_fraction(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.abs(a.astype(np.int16) - b.astype(np.int16)) > PIXEL_DELTA).mean())


def _sample_thumbnails(video: Path, every: float):
    """Yield (timestamp, thumbnail) pairs sampled every `every` seconds."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"fps=1/{every},scale={THUMB_W}:{THUMB_H},format=gray",
         "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE,
    )
    size = THUMB_W * THUMB_H
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield i * every, np.frombuffer(buf, dtype=np.uint8).reshape(THUMB_H, THUMB_W)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def detect_changes(video: Path, every: float, threshold: float, min_gap: float, duration: float) -> list[dict]:
    """Return [{'t': seconds, 'score': change fraction}] for each moment worth capturing."""
    captures: list[dict] = []
    reference = None   # thumbnail of the last captured frame
    prev = None        # previous sample
    pending = None     # (t, score) where a change was first seen, waiting to settle

    for t, thumb in _sample_thumbnails(video, every):
        if reference is None:
            captures.append({"t": t, "score": 1.0})
            reference = prev = thumb
            continue

        score = _changed_fraction(thumb, reference)
        if pending is None and score >= threshold:
            pending = (t, score)

        if pending is not None:
            settled = _changed_fraction(thumb, prev) < threshold / 3
            waited_long = t - pending[0] >= min_gap  # e.g. an embedded video that never settles
            far_enough = not captures or t - captures[-1]["t"] >= min_gap
            if (settled or waited_long) and far_enough:
                captures.append({"t": t, "score": max(pending[1], score)})
                reference = thumb
                pending = None

        prev = thumb
        if duration:
            print(f"\r[frames] scanning {fmt_ts(t)} / {fmt_ts(duration)}  captures: {len(captures)}", end="", flush=True)
    print()
    return captures


def extract_frames(video: Path, workdir: Path, every: float, threshold: float, min_gap: float,
                   max_frames: int, duration: float) -> list[dict]:
    frames_dir = workdir / "frames"
    index_path = frames_dir / "index.json"
    if index_path.exists():
        print("[frames] Reusing existing frames/")
        return json.loads(index_path.read_text())

    frames_dir.mkdir(parents=True, exist_ok=True)
    captures = detect_changes(video, every, threshold, min_gap, duration)

    if len(captures) > max_frames:
        # Keep the first frame plus the biggest changes, in time order.
        first, rest = captures[0], captures[1:]
        rest = sorted(sorted(rest, key=lambda c: -c["score"])[: max_frames - 1], key=lambda c: c["t"])
        print(f"[frames] {len(captures)} changes found; keeping the {max_frames} most significant")
        captures = [first, *rest]

    frames = []
    for n, cap in enumerate(captures):
        path = frames_dir / f"frame_{n:04d}_{fmt_ts(cap['t']).replace(':', '-')}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{cap['t']:.2f}", "-i", str(video),
             "-frames:v", "1", "-vf", "scale='min(1280,iw)':-2", "-q:v", "4", str(path)],
            check=True,
        )
        frames.append({"t": cap["t"], "path": str(path.relative_to(workdir))})
        print(f"\r[frames] saving {n + 1}/{len(captures)}", end="", flush=True)
    print()

    index_path.write_text(json.dumps(frames, indent=1))
    return frames
