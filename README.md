# videoTranscriber

Summarize long videos (2+ hours is fine) from a **Google Drive link** with Claude.

Claude can't listen to audio, so the tool:

1. **Downloads** the video from Google Drive (or uses a local file).
2. **Transcribes** the speech locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with timestamps.
3. **Captures screenshots only when the screen changes.** It samples the video every 2 s as tiny thumbnails and saves a
   full screenshot when enough of the screen has changed, after waiting for the change to settle (so you don't get a
   slide halfway through a transition). Small things like a moving cursor don't count.
4. **Sends Claude the transcript and screenshots interleaved in time order**, in ~15-minute parts, so it can link what was
   said to what was on screen (and read slide/code/document text the speaker only points at).
5. **Combines the notes from each part** (plus the full transcript) into one final summary: TL;DR, timestamped
   chapters, detailed summary, key points/numbers, and action items.

## Setup

Requirements: Python 3.10+, [ffmpeg](https://ffmpeg.org/download.html), and an
[Anthropic API key](https://console.anthropic.com/).

```bash
# macOS: brew install ffmpeg      Ubuntu/Debian: sudo apt install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
videotranscriber "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
```

The Drive file must be shared as **"Anyone with the link can view"**. For private files, export your
browser's Google cookies to `~/.cache/gdown/cookies.txt` (Netscape format), or download the file and pass the local path:

```bash
videotranscriber ~/Downloads/meeting.mp4 --title "Q3 planning meeting"
```

Everything goes into `output/<file id>/`:

| File | Contents |
|---|---|
| `summary.md` | **The final summary** |
| `notes/part_XX.md` | Detailed notes for each ~15-minute part |
| `transcript.txt` / `transcript.json` | Timestamped transcript |
| `frames/` | The screenshots that were sent to Claude |

Each step saves its results, so if you run the command again it skips what's already done. The download, transcript
and screenshots are reused, so trying a different `--focus` or `--task` only re-runs the Claude part.

### Asking for something other than a summary

Use `--task` to tell Claude what to do with the video instead of summarizing it:

```bash
videotranscriber "https://drive.google.com/file/d/FILE_ID/view" \
  --task "Do the exercise the instructor assigns at the end of the video. Show your work step by step."

# Longer instructions can go in a text file:
videotranscriber "https://drive.google.com/file/d/FILE_ID/view" --task-file my_task.txt
```

Other ideas: "Write a study guide with practice questions", "List every command shown on screen, in order",
"Turn this into meeting minutes", "Write the code the presenter builds in the demo".

With a task, Claude still writes notes for each part first, but is told what the task is so it copies the details
the task needs (exact exercise wording, data, code shown on screen). Then it does the task using those notes and the
full transcript. The answer is saved as `result-<id>.md` (the id changes per task, so earlier results aren't
overwritten), starting with the task you gave.

### Useful options

| Option | Default | What it does |
|---|---|---|
| `--task "..."` / `--task-file` | | Do this instead of the default summary (see above) |
| `--focus "..."` | | Tell Claude what you care about, e.g. `--focus "technical decisions and who owns each follow-up"` |
| `--whisper-model` | `small` | `medium` or `large-v3` are more accurate but slower. Use `large-v3` with a GPU |
| `--language` | auto | Spoken language, e.g. `en`, `es` |
| `--threshold` | `0.06` | Fraction of the screen that must change to take a screenshot. Raise it (e.g. `0.12`) if a webcam overlay causes too many screenshots; lower it (e.g. `0.03`) if small changes like a new bullet point are missed |
| `--min-gap` | `8` | Minimum seconds between screenshots |
| `--max-frames` | `400` | Cap on screenshots for the whole video (keeps the biggest changes) |
| `--chunk-minutes` | `15` | Length of each part sent to Claude |
| `--model` / `--effort` | `claude-opus-5-5` / `high` | Claude model and reasoning effort |
| `--no-summary` | | Only download, transcribe and take screenshots, without calling Claude. Useful for tuning `--threshold` by checking `frames/` |

### Time and cost (rough estimates)

- **Transcription** is the slowest step. On a CPU, `small` runs at very roughly 3–10× real time (about 15–40 min for a
  2 h video). A GPU is much faster.
- **Claude**: a 2-hour video with a few hundred screenshots is on the order of $1–5 of API usage on Claude Opus 5.5,
  mostly from the images. Fewer screenshots (higher `--threshold`) cost less.

## How the screenshot detection works

`videotranscriber/frames.py` compares each 160×90 grayscale thumbnail with the **last captured** frame and counts the
pixels that changed noticeably. If more than `--threshold` of the screen changed, it waits for the picture to stop
changing, then saves a 1280px-wide JPEG at that moment. Comparing with the last *capture* rather than the previous
sample means slow, gradual changes (like scrolling through a document) still add up and trigger a new screenshot.
