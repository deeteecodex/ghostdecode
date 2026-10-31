"""
make_synthetic.py - generate a Ghost-Font-style video with a KNOWN hidden text.

How it works (same idea as the real thing):
  * Two full-frame fields of evenly spaced (Poisson-disk) dots.
  * Field A scrolls UP, field B scrolls DOWN, both wrapping around.
  * Each frame, field A's dots are shown only inside the letter shapes and
    field B's dots only outside them. Every dot keeps one fixed direction, so
    a single frame is uniform noise and only the motion reveals the letters.
  * The letter mask drifts slowly around the screen, like the real videos.

Usage:
    python tools/make_synthetic.py "HELLO WORLD" --out samples/hello.mp4  (about 40 s)
    python tools/make_synthetic.py "LINE ONE|LINE TWO" --out samples/two.mp4

Use | to force line breaks; otherwise lines are chosen automatically.
Requires: numpy, opencv-python, pillow
"""
import argparse
import math
import os
import sys
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "arialbd.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def find_font():
    for f in FONT_CANDIDATES:
        try:
            ImageFont.truetype(f, 20)
            return f
        except OSError:
            pass
    sys.exit("No bold font found. Edit FONT_CANDIDATES in make_synthetic.py.")


def poisson_disk_sample(width, height, radius, rng, k=30):
    """Points filling the area with no two closer than `radius` (Bridson)."""
    cell = radius / np.sqrt(2)
    gw, gh = int(np.ceil(width / cell)), int(np.ceil(height / cell))
    grid = -np.ones((gh, gw), dtype=int)
    pts, active = [], []

    def fits(p):
        gy, gx = int(p[1] // cell), int(p[0] // cell)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < gh and 0 <= nx < gw and grid[ny, nx] != -1:
                    q = pts[grid[ny, nx]]
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < radius ** 2:
                        return False
        return True

    first = (rng.uniform(0, width), rng.uniform(0, height))
    pts.append(first)
    active.append(0)
    grid[int(first[1] // cell), int(first[0] // cell)] = 0
    while active:
        idx = active[rng.integers(0, len(active))]
        bx, by = pts[idx]
        for _ in range(k):
            ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(radius, 2 * radius)
            c = (bx + dist * np.cos(ang), by + dist * np.sin(ang))
            if 0 <= c[0] < width and 0 <= c[1] < height and fits(c):
                pts.append(c)
                active.append(len(pts) - 1)
                grid[int(c[1] // cell), int(c[0] // cell)] = len(pts) - 1
                break
        else:
            active.remove(idx)
    return np.array(pts, dtype=np.float32)


def best_layout(text, W, H, font_path):
    """Pick line breaks and the biggest font size that fits the frame."""
    if "|" in text:
        candidates = [[s.strip() for s in text.split("|")]]
    else:
        candidates = []
        words = text.split()
        for n in range(1, 5):
            if n > len(words):
                break
            lines = textwrap.wrap(text, width=max(1, math.ceil(len(text) / n) + 1),
                                    break_long_words=False, break_on_hyphens=False)
            candidates.append(lines)
    best = None
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    for lines in candidates:
        for size in range(300, 20, -4):
            font = ImageFont.truetype(font_path, size)
            widest = max(probe.textlength(l, font=font) for l in lines)
            if widest <= 0.84 * W and len(lines) * size * 1.05 <= 0.80 * H:
                if best is None or size > best[0]:
                    best = (size, lines)
                break
    if best is None:
        sys.exit("Text too long to fit the frame.")
    return best


def render_mask(lines, size, W, H, font_path):
    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    stroke = max(1, size // 40)
    line_h = int(size * 1.05)
    y = (H - line_h * len(lines)) // 2
    for line in lines:
        bb = d.textbbox((0, 0), line, font=font, stroke_width=stroke)
        x = (W - (bb[2] - bb[0])) // 2 - bb[0]
        d.text((x, y), line, fill=255, font=font, stroke_width=stroke, stroke_fill=255)
        y += line_h
    return np.array(img) > 127


def render_frame(p, W, H, bg=235, dot=40):
    img = np.full((H, W), bg, dtype=np.uint8)
    xi = np.clip(p[:, 0].astype(int), 1, W - 2)
    yi = np.clip(p[:, 1].astype(int), 1, H - 2)
    for ox, oy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        img[yi + oy, xi + ox] = dot
    return img


def main():
    ap = argparse.ArgumentParser(description="Make a synthetic Ghost Font video")
    ap.add_argument("text")
    ap.add_argument("--out", default="synthetic.mp4")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=478)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="px/frame per field (above ~1.2 the small dots alias and decoding degrades)")
    ap.add_argument("--drift", type=float, default=8.0,
                    help="how far the text wanders (px); 0 = stationary")
    ap.add_argument("--radius", type=float, default=4.0, help="dot spacing (px)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    W, H = a.width, a.height
    font_path = find_font()
    size, lines = best_layout(a.text, W, H, font_path)
    print(f"Text: {lines}  (font size {size})")
    base_mask = render_mask(lines, size, W, H, font_path).astype(np.uint8)

    print("Generating dot fields (a few seconds)...")
    A = poisson_disk_sample(W, H, a.radius, np.random.default_rng(a.seed))
    B = poisson_disk_sample(W, H, a.radius, np.random.default_rng(a.seed + 100))
    print(f"  {len(A)} + {len(B)} dots")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H), isColor=False)
    if not writer.isOpened():
        sys.exit("Could not open video writer.")

    for i in range(a.frames):
        t = i / max(a.frames, 1)
        dx = a.drift * math.sin(2 * math.pi * t + 0.7)
        dy = 0.6 * a.drift * math.sin(2 * math.pi * 1.3 * t + 2.1)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        mask = cv2.warpAffine(base_mask, M, (W, H)) > 0

        def inside(p):
            return mask[np.clip(p[:, 1].astype(int), 0, H - 1),
                        np.clip(p[:, 0].astype(int), 0, W - 1)]

        shown = np.vstack([A[inside(A)], B[~inside(B)]])
        writer.write(render_frame(shown, W, H))
        A[:, 1] = (A[:, 1] - a.speed) % H   # field A scrolls up
        B[:, 1] = (B[:, 1] + a.speed) % H   # field B scrolls down
        if i % 25 == 0:
            print(f"  frame {i}/{a.frames}")
    writer.release()
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
