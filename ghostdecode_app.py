"""
GhostDecode - small desktop app around ghostdecode.py.

  * Double-click: opens a window, press "Choose video...".
  * Drag a video file onto GhostDecode.exe: it opens and starts decoding at once.

Decoding takes about one to four minutes depending on the video size.
"""
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import ghostdecode as gd

VIDEO_TYPES = [("Video files", "*.mp4 *.mov *.webm *.avi *.mkv *.m4v"), ("All files", "*.*")]


def progress_percent(stage, done, total):
    """Map decoder stages onto one 0-100 progress bar."""
    frac = done / total if total else 0
    if stage == "loading":
        return 3 * frac
    if stage == "motion":
        return 3 + 85 * frac
    if stage == "reading":
        return 88 + 12 * frac
    return 0


def status_text(stage, done, total):
    if stage == "loading":
        return "Loading video..."
    if stage == "motion":
        return f"Analysing motion... {done}/{total}"
    if stage == "reading":
        return f"Reading text... {done}/{total}"
    return ""


def describe(res):
    """Human-readable lines for a decode result."""
    if res["text"] is None:
        return "No text found.", ["Try a different video, or one recorded straight from the generator."]
    lines = [f"Agreement: {round(100 * res['agreement'])}% of {res['total_reads']} reads "
             f"across {len(res['windows'])} time windows."]
    others = res["candidates"][1:]
    if others:
        lines.append("Other readings: " + "; ".join(
            f"{c['text']} ({round(100 * c['share'])}%)" for c in others))
    lines.append("Note: output is capitals only, and punctuation can be misread.")
    return res["text"], lines


class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.working = False
        self.last_text = ""

        root.title("GhostDecode")
        root.geometry("680x470")
        root.minsize(560, 400)
        frm = ttk.Frame(root, padding=16)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Ghost Font decoder", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(frm, wraplength=620, text=(
            "Choose a Ghost Font video, or drag a video file onto GhostDecode.exe. "
            "It reads the hidden text from the motion, no AI involved.")
        ).pack(anchor="w", pady=(2, 12))

        row = ttk.Frame(frm)
        row.pack(fill="x")
        self.btn = ttk.Button(row, text="Choose video...", command=self.choose)
        self.btn.pack(side="left")
        self.file_lbl = ttk.Label(row, text="")
        self.file_lbl.pack(side="left", padx=10)

        self.bar = ttk.Progressbar(frm, maximum=100)
        self.bar.pack(fill="x", pady=(16, 4))
        self.status = ttk.Label(frm, text="Ready.")
        self.status.pack(anchor="w")

        self.headline = tk.Text(frm, height=3, wrap="word", font=("Segoe UI", 18, "bold"),
                                relief="flat", state="disabled")
        self.headline.pack(fill="x", pady=(14, 4))
        self.details = tk.Text(frm, height=6, wrap="word", font=("Segoe UI", 10),
                               relief="flat", state="disabled")
        self.details.pack(fill="both", expand=True)

        bottom = ttk.Frame(frm)
        bottom.pack(fill="x", pady=(8, 0))
        self.copy_btn = ttk.Button(bottom, text="Copy text", command=self.copy, state="disabled")
        self.copy_btn.pack(side="right")

    # ---- helpers
    def _set(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def choose(self):
        path = filedialog.askopenfilename(title="Choose a Ghost Font video", filetypes=VIDEO_TYPES)
        if path:
            self.start(path)

    def copy(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_text)
        self.status.configure(text="Copied to clipboard.")

    # ---- decoding (runs in a worker thread so the window stays responsive)
    def start(self, path):
        if self.working:
            return
        self.working = True
        self.last_text = ""
        self.btn.state(["disabled"])
        self.copy_btn.state(["disabled"])
        self.bar["value"] = 0
        self.file_lbl.configure(text=os.path.basename(path))
        self._set(self.headline, "")
        self._set(self.details, "")
        self.status.configure(text="Starting... this takes one to a few minutes.")
        threading.Thread(target=self._work, args=(path,), daemon=True).start()
        self.root.after(100, self._poll)

    def _work(self, path):
        def cb(stage, done, total):
            self.q.put(("progress", stage, done, total))
        try:
            self.q.put(("done", gd.decode(path, progress=cb)))
        except gd.GhostDecodeError as e:
            self.q.put(("error", str(e)))
        except Exception as e:  # keep the window alive whatever happens
            self.q.put(("error", f"Unexpected error: {e}"))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "progress":
                    _, stage, done, total = msg
                    self.bar["value"] = progress_percent(stage, done, total)
                    self.status.configure(text=status_text(stage, done, total))
                elif msg[0] == "done":
                    head, lines = describe(msg[1])
                    self.last_text = msg[1]["text"] or ""
                    self._set(self.headline, head)
                    self._set(self.details, "\n".join(lines))
                    self.bar["value"] = 100
                    self.status.configure(text="Done.")
                    self._finish()
                elif msg[0] == "error":
                    self._set(self.headline, "Couldn't decode that video")
                    self._set(self.details, msg[1])
                    self.status.configure(text="Error.")
                    self._finish()
        except queue.Empty:
            pass
        if self.working:
            self.root.after(100, self._poll)

    def _finish(self):
        self.working = False
        self.btn.state(["!disabled"])
        if self.last_text:
            self.copy_btn.state(["!disabled"])


def main():
    root = tk.Tk()
    app = App(root)
    path = next((a for a in sys.argv[1:] if os.path.isfile(a)), None)
    if path:
        root.after(300, lambda: app.start(path))
    root.mainloop()


if __name__ == "__main__":
    main()
