# ghostdecode

Reads the text hidden in [Ghost Font](https://ghostfont.org) videos using plain
motion analysis plus Tesseract OCR. **No AI model is involved in the decoding.**

![GhostDecode reading a video](docs/screenshot.png)

Ghost Font hides text in a field of moving dots: dots inside the letters drift
one way, background dots drift the other. A single frame looks like uniform
noise, so image-based readers see nothing. The site describes it as a
perception experiment, not encryption, and its own blog documents a decode via
H.264 motion vectors. This project is an independent, no-AI take on the same
idea, built from optical flow.

## Quick start

**Windows app (nothing to install):** download `GhostDecode-windows.zip` from
the Releases page of this repo, unzip it, and run `GhostDecode.exe`. Press
"Choose video..." or drag a video file onto the exe. A decode takes one to a
few minutes.

**From Python:**

    pip install -r requirements.txt
    python ghostdecode.py path/to/video.mp4

You also need the [Tesseract OCR engine](https://github.com/tesseract-ocr/tesseract)
installed (on Windows the default install path is detected automatically).

## How it works

1. **Optical flow** (Farneback) between every pair of consecutive frames.
2. **k-means** on the flow vectors splits pixels into motion groups. The
   smaller moving group is the letter dots, giving one mask per frame pair.
3. **Short-window voting.** The text drifts slowly around the screen, so
   summing the whole video blurs it. Masks are summed over short windows
   (20 frame pairs, stride 10) to get sharp heatmaps.
4. **Tesseract OCR** reads each window heatmap with several preprocessing
   variants.
5. **Majority vote** over all reads gives the final text.

## Command line

    python ghostdecode.py video.mp4
    python ghostdecode.py video.mp4 --out ghost_out   # also saves window heatmaps
    python ghostdecode.py video.mp4 --json            # full result as JSON
    python ghostdecode.py video.mp4 --winsize 7       # finer optical-flow window

Output is a per-window readout, the top candidates with agreement percentages,
and a final `DECODED:` line. The agreement figure is the share of OCR reads that
agree; it is a consensus signal, not a probability of being right.

From Python:

    from ghostdecode import decode
    result = decode("video.mp4")
    print(result["text"], result["agreement"])

## LLM agent (bring your own API key)

`agent.py` is a command-line chat where an LLM runs the decoder as a tool. The
LLM never sees the video and never reads the text itself: it calls
`decode_ghost_font`, the pipeline above does all the reading, and the LLM
reports the result. Every tool call is printed.

    copy .env.example .env      # then edit .env: provider, model and your API key
    python agent.py

Your key stays on your machine (`.env` is gitignored) and is never logged.
Providers: `openai`, `anthropic`, `gemini`, `grok`, or any OpenAI-compatible
endpoint (`custom` + `LLM_BASE_URL`). Inside the chat: `/videos` lists videos,
`/folder <path>` switches the video folder, `/quit` exits. The agent can only
read videos inside the chosen folder.

Status: verified live with OpenAI. The Anthropic, Gemini and Grok adapters
follow those providers' documented APIs but have only been tested against a
mock server, so reports are welcome.

## Desktop app (build it yourself)

With the venv active and Tesseract installed:

    .\build_exe.ps1

The app ends up in `dist\GhostDecode\` with a shareable
`dist\GhostDecode-windows.zip`. The build bundles Tesseract; see
`THIRD_PARTY_NOTICES.md` for licences.

## Synthetic test videos

`tools/make_synthetic.py` builds Ghost-Font-style videos with a known answer,
used to validate the decoder:

    pip install pillow
    python tools/make_synthetic.py "HELLO WORLD" --out samples/hello.mp4
    python ghostdecode.py samples/hello.mp4

Use `|` in the text to force line breaks. Options: `--speed`, `--drift`,
`--frames`, `--width`, `--height`. Defaults are tuned so the decoder reads the
videos reliably (a 36-character message decoded at 100%).

## Tests

    pip install pillow pytest
    pytest -q

## Tested on

Eight videos recorded from ghostfont.org's own generator:

| Video text | Decoded | Reads agreeing |
|---|---|---|
| HELLO HUMAN | HELLO HUMAN | 84% |
| WELCOME | WELCOME | 100% |
| THE QUICK BROWN FOX | THE QUICK BROWN FOX | 99% |
| THE FIVE BOXING WIZARDS JUMP QUICKLY (36 chars) | exact | ~60% |
| Meet me at 9:30, room 4B! | MEET ME AT 9.30, ROOM 4B! | 88% |
| HELLO (invert on) | HELLO | 100% |
| HELLO (slowest speed) | HELLO | 100% |
| HELLO (fastest speed) | HELLO | 100% |

## Known limits

- Output is always uppercase; case isn't recoverable from the heatmap.
- Punctuation can be misread (a colon came out as a period).
- Small text (long messages) needs the finer optical-flow window; the default
  (`--winsize 9`) handled the site's 36-character limit in testing.
- Tested only on videos from ghostfont.org's own generator, at its default
  settings plus invert, slowest and fastest speed.

## Project layout

    ghostdecode.py        the decoder (library + command line)
    ghostdecode_app.py    desktop app
    agent.py              LLM tool-calling chat
    build_exe.ps1         builds the Windows app
    tools/                synthetic video generator
    tests/                smoke tests
    samples/              put generated sample videos here
    docs/                 screenshots

## Credits and licence

Ghost Font is the work of its original creator at ghostfont.org. This project is
independent and not affiliated with it. Please respect their work and don't
redistribute their videos. Released under the MIT licence (see `LICENSE`).
