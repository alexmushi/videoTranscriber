"""Speech-to-text with faster-whisper (runs locally, no API key needed)."""
import json
import subprocess
from pathlib import Path

from .util import fmt_ts


def extract_audio(video: Path, workdir: Path) -> Path:
    audio = workdir / "audio.wav"
    if not audio.exists():
        print("[audio] Extracting audio track ...")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(audio)],
            check=True,
        )
    return audio


def transcribe(video: Path, workdir: Path, model_size: str, language: str | None, device: str) -> list[dict]:
    out_json = workdir / "transcript.json"
    if out_json.exists():
        print("[whisper] Reusing existing transcript.json")
        return json.loads(out_json.read_text())

    audio = extract_audio(video, workdir)

    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[whisper] Loading model '{model_size}' on {device} ...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(str(audio), language=language, vad_filter=True, beam_size=5)
    print(f"[whisper] Detected language: {info.language}; duration {fmt_ts(info.duration)}")

    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        print(f"\r[whisper] {fmt_ts(seg.end)} / {fmt_ts(info.duration)}", end="", flush=True)
    print()

    out_json.write_text(json.dumps(segments, ensure_ascii=False, indent=1))
    (workdir / "transcript.txt").write_text(
        "\n".join(f"[{fmt_ts(s['start'])}] {s['text']}" for s in segments), encoding="utf-8"
    )
    return segments
