import argparse
import hashlib
from pathlib import Path

from .drive import extract_file_id, fetch_video
from .frames import extract_frames
from .summarize import summarize
from .transcribe import transcribe
from .util import fmt_ts, require_ffmpeg, video_duration


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="videotranscriber",
        description="Summarize a long video (Google Drive link or local file) with Claude, "
                    "using a transcript plus a screenshot every time the screen changes.",
    )
    p.add_argument("source", help="Google Drive share link, or a path to a local video file")
    p.add_argument("-o", "--out", type=Path, help="Output folder (default: output/<video id>)")
    p.add_argument("--title", help="Title used in the prompts and summary (default: file name)")
    p.add_argument("--focus", help="Optional extra instructions, e.g. 'pricing decisions and action items'")

    g = p.add_argument_group("transcription")
    g.add_argument("--whisper-model", default="small",
                   help="faster-whisper model: tiny, base, small, medium, large-v3 (default: small)")
    g.add_argument("--language", help="Spoken language code, e.g. en, es (default: auto-detect)")
    g.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    g = p.add_argument_group("screenshots")
    g.add_argument("--sample-every", type=float, default=2.0,
                   help="Seconds between checks for screen changes (default: 2)")
    g.add_argument("--threshold", type=float, default=0.06,
                   help="Fraction of the screen that must change to take a screenshot (default: 0.06). "
                        "Raise it if a webcam overlay or cursor causes too many captures.")
    g.add_argument("--min-gap", type=float, default=8.0,
                   help="Minimum seconds between screenshots (default: 8)")
    g.add_argument("--max-frames", type=int, default=400,
                   help="Cap on screenshots for the whole video (default: 400)")

    g = p.add_argument_group("claude")
    g.add_argument("--model", default="claude-opus-5-5")
    g.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    g.add_argument("--chunk-minutes", type=float, default=15,
                   help="Minutes of video per Claude request before the final summary (default: 15)")
    g.add_argument("--workers", type=int, default=4, help="Parallel Claude requests (default: 4)")
    g.add_argument("--no-summary", action="store_true",
                   help="Only download, transcribe and capture screenshots; skip Claude")
    args = p.parse_args(argv)

    require_ffmpeg()

    if args.out:
        workdir = args.out
    else:
        key = extract_file_id(args.source) if not Path(args.source).exists() else None
        key = key or Path(args.source).stem + "-" + hashlib.sha1(str(Path(args.source).resolve()).encode()).hexdigest()[:6]
        workdir = Path("output") / key
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Working folder: {workdir}")

    video = fetch_video(args.source, workdir)
    duration = video_duration(str(video))
    name_file = workdir / "video.name.txt"
    title = args.title or (name_file.read_text(encoding="utf-8").strip() if name_file.exists() else video.stem)
    print(f"[video] {video.name}: {fmt_ts(duration)}")

    device = args.device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    segments = transcribe(video, workdir, args.whisper_model, args.language, device)
    frames = extract_frames(video, workdir, args.sample_every, args.threshold, args.min_gap,
                            args.max_frames, duration)
    print(f"[ready] {len(segments)} transcript segments, {len(frames)} screenshots")

    if args.no_summary:
        return

    out = summarize(segments, frames, duration, workdir, title=title, model=args.model, effort=args.effort,
                    chunk_minutes=args.chunk_minutes, focus=args.focus, workers=args.workers)
    print(f"\nDone. Summary: {out}")


if __name__ == "__main__":
    main()
