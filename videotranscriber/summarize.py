"""Send transcript + screenshots to Claude and build the final summary.

Long videos are split into time chunks. Each chunk (its transcript interleaved
with the screenshots taken during it) gets detailed notes from Claude ("map").
All chunk notes plus the full transcript are then combined into one summary
("reduce").
"""
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from .util import fmt_ts

MAX_IMAGES_PER_CHUNK = 40

NOTES_SYSTEM = """You analyze recordings (meetings, lectures, screen shares, presentations) from two inputs: \
a timestamped speech transcript and screenshots captured whenever the screen visibly changed. \
The screenshots are what viewers saw; connect them to what was being said at that moment. \
Read text in the screenshots (slides, code, documents, dashboards, chat) and use it: it often \
contains the names, numbers and details the speaker only refers to as "this" or "here". \
The transcript is machine-generated and can mishear names or jargon; prefer the spelling shown on screen."""

NOTES_PROMPT = """This is part {part} of {total} of the video "{title}", covering {start} to {end}.

Write detailed notes for this part in Markdown:
- Walk through it in order as short sections, each starting with a timestamp like **[00:12:30]**.
- For each section, say what was discussed AND what was on screen, and how they relate.
- Capture concrete details: names, numbers, decisions, definitions, code or commands shown, questions asked and their answers.
- End with a "Key takeaways" list for this part.
Do not add an introduction or say this is part of a larger video.{focus}"""

SUMMARY_PROMPT = """Below are detailed notes for each part of the video "{title}" (total length {duration}), \
written from the transcript and the screenshots, followed by the full transcript.

Write the final summary in Markdown with these sections:
1. **TL;DR** - 3 to 5 sentences.
2. **Chapters** - a timestamped outline of the whole video (one line per chapter, e.g. `00:12:30 - Topic`), with a 1-3 sentence description each.
3. **Detailed summary** - the main content, organized by topic rather than strictly by time, including what was shown on screen (slides, demos, code, documents) and how it supported the discussion. Include timestamps so the reader can jump to the moment.
4. **Key points, decisions and numbers**
5. **Action items / open questions** - only if there are any; otherwise omit the section.
Be complete: the reader will rely on this instead of watching the {duration} video.{focus}

<notes>
{notes}
</notes>

<transcript>
{transcript}
</transcript>"""

TASK_NOTES_HINT = """

These notes will later be used for this task: "{task}". Capture everything that task needs in full, \
copying exact wording from the speech and the screen where it matters (e.g. exercise instructions, \
questions, data, code, formulas, requirements)."""

TASK_PROMPT = """Below are detailed notes for each part of the video "{title}" (total length {duration}), \
written from the transcript and the screenshots, followed by the full transcript.

Use them to do the following task:

<task>
{task}
</task>

If the task refers to something in the video (an exercise, assignment, question, template, etc.), find it in \
the notes and transcript and follow the video's instructions exactly, citing the timestamp where it appears. \
If the video leaves something needed for the task unspecified, make a reasonable assumption and state it. \
Respond in Markdown.{focus}

<notes>
{notes}
</notes>

<transcript>
{transcript}
</transcript>"""


def _image_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def build_chunks(segments: list[dict], frames: list[dict], duration: float, chunk_minutes: float) -> list[dict]:
    span = chunk_minutes * 60
    n = max(1, int(-(-duration // span)))
    chunks = [{"start": i * span, "end": min((i + 1) * span, duration), "segments": [], "frames": []} for i in range(n)]
    for seg in segments:
        chunks[min(int(seg["start"] // span), n - 1)]["segments"].append(seg)
    for fr in frames:
        chunks[min(int(fr["t"] // span), n - 1)]["frames"].append(fr)
    for c in chunks:
        if len(c["frames"]) > MAX_IMAGES_PER_CHUNK:
            step = len(c["frames"]) / MAX_IMAGES_PER_CHUNK
            c["frames"] = [c["frames"][int(i * step)] for i in range(MAX_IMAGES_PER_CHUNK)]
    return [c for c in chunks if c["segments"] or c["frames"]]


def chunk_content(chunk: dict, workdir: Path) -> list[dict]:
    """Interleave transcript lines and screenshots in time order."""
    events = [(s["start"], 1, s) for s in chunk["segments"]] + [(f["t"], 0, f) for f in chunk["frames"]]
    events.sort(key=lambda e: (e[0], e[1]))

    blocks: list[dict] = []
    lines: list[str] = []

    def flush():
        if lines:
            blocks.append({"type": "text", "text": "\n".join(lines)})
            lines.clear()

    for t, kind, item in events:
        if kind == 0:
            flush()
            blocks.append({"type": "text", "text": f"[Screenshot at {fmt_ts(t)}]"})
            blocks.append(_image_block(workdir / item["path"]))
        else:
            lines.append(f"[{fmt_ts(t)}] {item['text']}")
    flush()
    if not chunk["segments"]:
        blocks.append({"type": "text", "text": "(No speech in this part.)"})
    return blocks


def _ask(client: anthropic.Anthropic, model: str, effort: str, system: str, content: list[dict]) -> str:
    with client.beta.messages.stream(
        model=model,
        max_tokens=64000,
        system=system,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        # If a safety classifier declines, re-run on Anthropic's recommended fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": content}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        details = message.stop_details
        raise RuntimeError(f"Claude declined this request ({getattr(details, 'category', None)}): {getattr(details, 'explanation', '')}")
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    if message.stop_reason == "max_tokens":
        text += "\n\n_(Output truncated: hit max_tokens.)_"
    return text


def summarize(segments: list[dict], frames: list[dict], duration: float, workdir: Path, *, title: str,
              model: str, effort: str, chunk_minutes: float, focus: str | None, task: str | None,
              workers: int) -> Path:
    client = anthropic.Anthropic()
    focus_text = f"\n\nThe reader is especially interested in: {focus}" if focus else ""
    notes_hint = focus_text + (TASK_NOTES_HINT.format(task=task) if task else "")
    chunks = build_chunks(segments, frames, duration, chunk_minutes)

    # Notes depend on the focus/task they were written for, so each combination gets its own folder.
    key = hashlib.sha1(f"{focus}\0{task}\0{chunk_minutes}".encode()).hexdigest()[:8] if (focus or task) else None
    notes_dir = workdir / (f"notes-{key}" if key else "notes")
    notes_dir.mkdir(exist_ok=True)

    def notes_for(i: int) -> str:
        path = notes_dir / f"part_{i + 1:02d}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        c = chunks[i]
        prompt = NOTES_PROMPT.format(part=i + 1, total=len(chunks), title=title,
                                     start=fmt_ts(c["start"]), end=fmt_ts(c["end"]), focus=notes_hint)
        content = [{"type": "text", "text": prompt}, *chunk_content(c, workdir)]
        print(f"[claude] Part {i + 1}/{len(chunks)} ({fmt_ts(c['start'])}-{fmt_ts(c['end'])}, "
              f"{len(c['segments'])} transcript lines, {len(c['frames'])} screenshots) ...")
        text = _ask(client, model, effort, NOTES_SYSTEM, content)
        path.write_text(text, encoding="utf-8")
        return text

    with ThreadPoolExecutor(max_workers=workers) as pool:
        all_notes = list(pool.map(notes_for, range(len(chunks))))

    print("[claude] Doing the task ..." if task else "[claude] Writing final summary ...")
    notes = "\n\n".join(
        f"## Part {i + 1} ({fmt_ts(c['start'])} - {fmt_ts(c['end'])})\n\n{n}"
        for i, (c, n) in enumerate(zip(chunks, all_notes))
    )
    transcript = "\n".join(f"[{fmt_ts(s['start'])}] {s['text']}" for s in segments)
    template = TASK_PROMPT if task else SUMMARY_PROMPT
    prompt = template.format(title=title, duration=fmt_ts(duration), focus=focus_text, task=task,
                             notes=notes, transcript=transcript)
    result = _ask(client, model, effort, NOTES_SYSTEM, [{"type": "text", "text": prompt}])

    if task:
        out = workdir / f"result-{key}.md"
        quoted = "\n".join(f"> {line}" for line in task.splitlines())
        out.write_text(f"# {title}\n\n**Task:**\n\n{quoted}\n\n---\n\n{result}\n", encoding="utf-8")
    else:
        out = workdir / ("summary.md" if not key else f"summary-{key}.md")
        out.write_text(f"# {title}\n\n{result}\n", encoding="utf-8")
    return out
