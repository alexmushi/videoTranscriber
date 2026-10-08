"""Fetch a video from a Google Drive link (or pass through a local path)."""
import os
import re
from pathlib import Path

import gdown

_ID_PATTERNS = [
    r"/file/d/([A-Za-z0-9_-]{10,})",
    r"[?&]id=([A-Za-z0-9_-]{10,})",
    r"/d/([A-Za-z0-9_-]{10,})",
]


def extract_file_id(link: str) -> str | None:
    for pattern in _ID_PATTERNS:
        m = re.search(pattern, link)
        if m:
            return m.group(1)
    return None


def fetch_video(source: str, workdir: Path) -> Path:
    """Return a local path to the video, downloading it from Drive if needed."""
    if os.path.exists(source):
        return Path(source).resolve()

    file_id = extract_file_id(source)
    if not file_id:
        raise SystemExit(f"Not a local file or a recognizable Google Drive link: {source}")

    existing = [p for p in workdir.glob("video.*") if p.suffix != ".txt"]
    if existing:
        print(f"[drive] Reusing downloaded {existing[0].name}")
        return existing[0]

    print(f"[drive] Downloading Drive file {file_id} ...")
    # gdown resolves the real filename; we download into workdir then rename.
    tmp_dir = workdir / "download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = gdown.download(id=file_id, output=str(tmp_dir) + os.sep, quiet=False)
    if not path:
        raise SystemExit(
            "Download failed. Make sure the file is shared as 'Anyone with the link can view'. "
            "For private files, export your browser cookies to ~/.cache/gdown/cookies.txt."
        )
    src = Path(path)
    dest = workdir / f"video{src.suffix or '.mp4'}"
    src.rename(dest)
    tmp_dir.rmdir()
    (workdir / "video.name.txt").write_text(src.stem, encoding="utf-8")
    return dest
