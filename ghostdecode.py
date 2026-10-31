"""
ghostdecode - read "Ghost Font" motion-hidden text from a video.

Pipeline (no AI involved in the decoding itself):
  1. Optical flow (Farneback) between every pair of consecutive frames.
  2. k-means on the flow vectors -> split pixels into motion groups;
     the smaller moving group is the letter dots. One mask per frame pair.
  3. Sum masks over SHORT windows (the text drifts, so a whole-video sum blurs it).
  4. Tesseract OCR on each window heatmap, several preprocessing variants.
  5. Majority vote over all reads.

Command line:  python ghostdecode.py video.mp4 [--out folder] [--winsize 9] [--json]

From Python:
    from ghostdecode import decode
    result = decode("video.mp4")
    print(result["text"], result["agreement"])
"""
import argparse
import collections
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import pytesseract

WIN = 20               # frame pairs per window
STRIDE = 10            # step between windows
DEFAULT_WINSIZE = 9    # Farneback window: small enough for thin strokes


class GhostDecodeError(Exception):
    """Raised for problems the caller can handle (bad file, no Tesseract...)."""


def _bundled_tesseract():
    """Find a Tesseract shipped with the app (the .exe build bundles one)."""
    bases = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bases.append(sys._MEIPASS)
    bases.append(os.path.dirname(os.path.abspath(__file__)))
    for base in bases:
        for sub in ("tesseract", os.path.join("vendor", "tesseract")):
            for name in ("tesseract.exe", "tesseract"):
                exe = os.path.join(base, sub, name)
                if os.path.isfile(exe):
                    return exe, os.path.join(base, sub, "tessdata")
    return None


def setup_tesseract():
    bundled = _bundled_tesseract()
    if bundled:
        exe, tessdata = bundled
        pytesseract.pytesseract.tesseract_cmd = exe
        if os.path.isdir(tessdata):
            os.environ["TESSDATA_PREFIX"] = tessdata
    else:
        exe = shutil.which("tesseract")
        if exe:
            pytesseract.pytesseract.tesseract_cmd = exe
        elif os.name == "nt":
            default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default):
                pytesseract.pytesseract.tesseract_cmd = default
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise GhostDecodeError(
            "Tesseract OCR not found. Install it "
            "(https://github.com/tesseract-ocr/tesseract) or add it to PATH.") from e


def load_gray_frames(path):
    if not os.path.isfile(path):
        raise GhostDecodeError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise GhostDecodeError(f"Can't open video: {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < WIN + 2:
        raise GhostDecodeError(f"Video too short ({len(frames)} frames; need {WIN + 2}+)")
    return frames


def letter_masks(frames, winsize=DEFAULT_WINSIZE, progress=None):
    """One boolean mask per frame pair: pixels in the 'letter' motion group."""
    h, w = frames[0].shape
    poly_sigma = 1.1 if winsize <= 11 else 1.2
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    n = len(frames) - 1
    masks = np.zeros((n, h, w), dtype=np.uint8)
    for i in range(n):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None, 0.5, 3, winsize, 3, 5, poly_sigma, 0)
        vecs = flow.reshape(-1, 2).astype(np.float32)
        _, lab, cen = cv2.kmeans(vecs, 3, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        lab = lab.reshape(h, w)
        static = np.argmin(np.linalg.norm(cen, axis=1))
        moving = [c for c in range(3) if c != static]
        letter = min(moving, key=lambda c: (lab == c).sum())
        masks[i] = (lab == letter)
        if progress and (i % 10 == 0 or i == n - 1):
            progress("motion", i + 1, n)
    return masks


def window_heatmaps(masks):
    n = len(masks)
    for start in range(0, n - WIN + 1, STRIDE):
        acc = masks[start:start + WIN].sum(axis=0).astype(np.float32)
        heat = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        yield start, start + WIN, heat


def _pad(img, m=40):
    return cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=255)


def ocr_reads(heat):
    """Several preprocessing x page-segmentation combos -> list of raw strings."""
    soft = _pad(255 - cv2.GaussianBlur(heat, (0, 0), 1.5))
    otsu = cv2.threshold(cv2.GaussianBlur(heat, (0, 0), 2), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    hard = _pad(255 - otsu)
    plain = _pad(255 - cv2.threshold(heat, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    combos = [(soft, 6), (soft, 11), (hard, 6), (hard, 11), (plain, 6)]
    out = []
    for img, psm in combos:
        txt = pytesseract.image_to_string(img, config=f"--psm {psm}")
        out.append(" ".join(txt.split()))
    return out


def key(s):
    return "".join(ch for ch in s.upper() if ch.isalnum())


def _save_json(result, outdir):
    if outdir:
        with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


def decode(video, outdir=None, winsize=DEFAULT_WINSIZE, progress=None, use_cache=True):
    """
    Decode the hidden text in a Ghost Font video.

    video     path to the video file
    outdir    optional folder: saves every window heatmap, result.json and a
              motion-mask cache there (a cache is only reused for the same winsize)
    winsize   optical-flow window; smaller reads finer text but is noisier
    progress  optional callback progress(stage, done, total), where stage is
              "loading", "motion" or "reading"
    use_cache reuse cached motion masks from outdir if present

    Returns a dict:
      text          best reading (upper case) or None if nothing was found
      agreement     share (0-1) of OCR reads that agree with `text`.
                    A signal, NOT a probability of being right.
      candidates    top readings: [{"text", "reads", "share"}, ...]
      windows       per window: [{"start", "end", "text"}, ...]
      total_reads   number of usable OCR reads
      winsize       optical-flow window used
      video         {"path", "frames", "width", "height"}
      heatmap_path  image of the clearest window for the winning text
                    (saved in outdir, or a temp folder if outdir is None)

    Raises GhostDecodeError for problems like a missing file or no Tesseract.
    """
    setup_tesseract()
    if progress:
        progress("loading", 0, 1)
    frames = load_gray_frames(video)
    h, w = frames[0].shape
    info = {"path": os.fspath(video), "frames": len(frames), "width": w, "height": h}
    if progress:
        progress("loading", 1, 1)

    cache = None
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # winsize is in the filename so changing it never reuses stale masks
        cache = os.path.join(outdir, f"masks_w{winsize}.npz")
    if use_cache and cache and os.path.exists(cache):
        masks = np.load(cache)["m"]
    else:
        masks = letter_masks(frames, winsize, progress)
        if use_cache and cache:
            np.savez_compressed(cache, m=masks)

    votes = collections.Counter()
    raws = collections.defaultdict(collections.Counter)
    window_info = []   # per window: dict + heat + its own key votes
    n_reads = 0
    windows = list(window_heatmaps(masks))
    for idx, (start, end, heat) in enumerate(windows):
        if progress:
            progress("reading", idx + 1, len(windows))
        if outdir:
            cv2.imwrite(os.path.join(outdir, f"window_{start:03d}_{end:03d}.png"), heat)
        wvotes = collections.Counter()
        wraws = collections.defaultdict(collections.Counter)
        for r in ocr_reads(heat):
            k = key(r)
            if len(k) >= 2:
                votes[k] += 1
                raws[k][r.upper()] += 1
                wvotes[k] += 1
                wraws[k][r.upper()] += 1
                n_reads += 1
        wtop = wvotes.most_common(1)[0][0] if wvotes else None
        window_info.append({
            "start": start, "end": end, "heat": heat, "votes": wvotes,
            "text": wraws[wtop].most_common(1)[0][0] if wtop else None,
        })

    result = {
        "text": None, "agreement": 0.0, "candidates": [],
        "windows": [{"start": wi["start"], "end": wi["end"], "text": wi["text"]}
                    for wi in window_info],
        "total_reads": n_reads, "winsize": winsize, "video": info, "heatmap_path": None,
    }
    if not votes:
        _save_json(result, outdir)
        return result

    shown = {k: raws[k].most_common(1)[0][0] for k in votes}
    result["candidates"] = [
        {"text": shown[k], "reads": c, "share": round(c / n_reads, 3)}
        for k, c in votes.most_common(3)
    ]
    best = votes.most_common(1)[0][0]
    result["text"] = shown[best]
    result["agreement"] = round(votes[best] / n_reads, 3)

    # save the clearest heatmap: the window where the winning text got the most reads
    best_wi = max(window_info, key=lambda wi: wi["votes"].get(best, 0))
    folder = outdir or tempfile.mkdtemp(prefix="ghostdecode_")
    os.makedirs(folder, exist_ok=True)
    hp = os.path.join(folder, "best_heatmap.png")
    cv2.imwrite(hp, best_wi["heat"])
    result["heatmap_path"] = hp
    _save_json(result, outdir)
    return result


def _cli_progress():
    last = {"stage": None}

    def cb(stage, done, total):
        if stage == "motion":
            print(f"  motion analysis {done}/{total}", flush=True)
        elif stage == "reading":
            if last["stage"] != "reading":
                print("Reading windows ...", flush=True)
        last["stage"] = stage
    return cb


def main():
    ap = argparse.ArgumentParser(description="Decode Ghost Font video text")
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="folder for window heatmaps/cache")
    ap.add_argument("--winsize", type=int, default=DEFAULT_WINSIZE,
                    help="optical-flow window (smaller = finer text, noisier)")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args()

    try:
        res = decode(a.video, a.out, winsize=a.winsize,
                     progress=None if a.json else _cli_progress())
    except GhostDecodeError as e:
        sys.exit(f"Error: {e}")

    if a.json:
        print(json.dumps(res, indent=2))
        return

    for w in res["windows"]:
        print(f"  window {w['start']:3d}-{w['end']:3d}: {w['text'] or '(nothing)'}")
    if res["text"] is None:
        print("\nNo text found.")
        return
    print("\nTop candidates:")
    for c in res["candidates"]:
        print(f"  {c['text']!r:40s} {c['reads']}/{res['total_reads']} reads "
              f"({100 * c['share']:.0f}%)")
    print(f"\nDECODED: {res['text']}")
    print(f"Agreement: {100 * res['agreement']:.0f}%   Heatmap: {res['heatmap_path']}")


if __name__ == "__main__":
    main()
