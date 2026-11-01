"""
GhostDecode - desktop app around ghostdecode.py.

  Tab "Decode":  choose a video (or drag one onto GhostDecode.exe) and read it.
  Tab "Ask AI":  chat with an LLM that runs the decoder as a tool. Works with
                 cloud providers (bring your own key) or a local model such as
                 Ollama, which needs no key and no internet.

Your API key is typed into a masked field, kept in memory only, and never
written to disk. Only non-secret settings (provider, model, address, folder)
are remembered, in your user profile folder, never next to the program.
"""
import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from urllib.parse import urlparse

import agent
import ghostdecode as gd

VIDEO_TYPES = [("Video files", "*.mp4 *.mov *.webm *.avi *.mkv *.m4v"), ("All files", "*.*")]

# (label shown in the app, provider id, address shown as the default)
PROVIDER_CHOICES = [
    ("OpenAI", "openai", "https://api.openai.com/v1"),
    ("Anthropic (Claude)", "anthropic", "https://api.anthropic.com/v1"),
    ("Google Gemini", "gemini", "https://generativelanguage.googleapis.com/v1beta/openai"),
    ("xAI Grok", "grok", "https://api.x.ai/v1"),
    ("Local model - Ollama (no key, no internet)", "ollama", "http://localhost:11434/v1"),
    ("Other OpenAI-compatible", "custom", ""),
]
LABEL_TO_ID = {label: pid for label, pid, _ in PROVIDER_CHOICES}
ID_TO_LABEL = {pid: label for label, pid, _ in PROVIDER_CHOICES}
DEFAULT_ENDPOINT = {pid: url for _, pid, url in PROVIDER_CHOICES}

VIDEO_NAME_RE = re.compile(r"\.(mp4|mov|webm|avi|mkv|m4v)\b", re.IGNORECASE)

# ---------------------------------------------------------------- settings (never the key)
SETTINGS_FIELDS = ("provider", "model", "endpoint", "reasoning_effort", "folder")


def settings_path():
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "GhostDecode", "settings.json")


def load_settings(path=None):
    try:
        with open(path or settings_path(), encoding="utf-8") as f:
            data = json.load(f)
        return {k: data[k] for k in SETTINGS_FIELDS if isinstance(data.get(k), str)}
    except Exception:
        return {}


def save_settings(values, path=None):
    """Save the non-secret settings only. Anything else (such as a key) is dropped."""
    clean = {k: values[k] for k in SETTINGS_FIELDS if isinstance(values.get(k), str)}
    path = path or settings_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)
    except OSError:
        pass


# ---------------------------------------------------------------- shared helpers
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


def privacy_note(endpoint):
    """One honest line about where the user's messages go."""
    url = agent.normalize_endpoint(endpoint)
    if not url:
        return "", "gray"
    if agent.is_local_url(url):
        return "Local address: nothing leaves this computer.", "#1a7f37"
    host = urlparse(url).hostname or "the provider"
    return (f"Cloud: your messages and the decoded text are sent to {host}. "
            "The video itself never leaves this computer."), "#9a6700"


def error_hint(msg):
    low = msg.lower()
    if "reasoning_effort" in low:
        return "Hint: under Connection, set 'Reasoning effort' to none, then Disconnect and Connect again."
    if "401" in low or "invalid_api_key" in low or "incorrect api key" in low:
        return "Hint: the provider rejected the API key. Disconnect and enter it again."
    if "404" in low or "model" in low and "not found" in low:
        return "Hint: check the model name (for Ollama, use 'Find installed models')."
    if "couldn't reach" in low:
        return "Hint: for a local model, make sure Ollama is running."
    return ""


# ---------------------------------------------------------------- Decode tab
class DecodeTab:
    def __init__(self, root, parent, on_video=None):
        self.root = root
        self.on_video = on_video
        self.q = queue.Queue()
        self.working = False
        self.last_text = ""

        frm = ttk.Frame(parent, padding=16)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Ghost Font decoder", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(frm, wraplength=640, text=(
            "Choose a Ghost Font video, or drag a video file onto GhostDecode.exe. "
            "This tab reads the hidden text from the motion and runs fully offline.")
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

    def start(self, path):
        if self.working:
            return
        if self.on_video:
            self.on_video(path)
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


# ---------------------------------------------------------------- Ask AI tab
class AskTab:
    def __init__(self, root, parent):
        self.root = root
        self.q = queue.Queue()
        self.backend = None          # holds the API key, in memory only
        self.tools = None
        self.busy = False
        self.shown_model = False
        self.cache_dir = tempfile.mkdtemp(prefix="ghostdecode_")   # removed on exit
        saved = load_settings()

        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill="both", expand=True)

        box = ttk.LabelFrame(frm, text="Connection", padding=8)
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)

        self.provider_var = tk.StringVar(value=ID_TO_LABEL.get(saved.get("provider"), PROVIDER_CHOICES[0][0]))
        self.model_var = tk.StringVar(value=saved.get("model", ""))
        self.endpoint_var = tk.StringVar(value=saved.get("endpoint", ""))
        self.effort_var = tk.StringVar(value=saved.get("reasoning_effort", ""))
        self.folder_var = tk.StringVar(value=saved.get("folder", ""))

        ttk.Label(box, text="Provider").grid(row=0, column=0, sticky="w", pady=2)
        self.provider_box = ttk.Combobox(box, textvariable=self.provider_var, state="readonly",
                                         values=[label for label, _, _ in PROVIDER_CHOICES])
        self.provider_box.grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)
        self.provider_box.bind("<<ComboboxSelected>>", self.on_provider)

        ttk.Label(box, text="Model").grid(row=1, column=0, sticky="w", pady=2)
        self.model_box = ttk.Combobox(box, textvariable=self.model_var)
        self.model_box.grid(row=1, column=1, sticky="ew", pady=2)
        self.find_btn = ttk.Button(box, text="Find installed models", command=self.find_models)
        self.find_btn.grid(row=1, column=2, padx=(6, 0))

        ttk.Label(box, text="Address").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(box, textvariable=self.endpoint_var).grid(row=2, column=1, columnspan=2, sticky="ew", pady=2)
        self.endpoint_var.trace_add("write", lambda *_: self.update_privacy())

        ttk.Label(box, text="API key").grid(row=3, column=0, sticky="w", pady=2)
        self.key_entry = ttk.Entry(box, show="\u2022")
        self.key_entry.grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Label(box, text="memory only, never saved", foreground="gray").grid(row=3, column=2, padx=(6, 0))

        ttk.Label(box, text="Reasoning effort").grid(row=4, column=0, sticky="w", pady=2)
        self.effort_box = ttk.Combobox(box, textvariable=self.effort_var, width=10,
                                       values=["", "none", "minimal", "low", "medium", "high"])
        self.effort_box.grid(row=4, column=1, sticky="w", pady=2)
        ttk.Label(box, text="optional; some OpenAI models need 'none' to use tools",
                  foreground="gray").grid(row=4, column=2, padx=(6, 0))

        ttk.Label(box, text="Video folder").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Entry(box, textvariable=self.folder_var, state="readonly").grid(row=5, column=1, sticky="ew", pady=2)
        ttk.Button(box, text="Browse...", command=self.browse).grid(row=5, column=2, padx=(6, 0))

        btns = ttk.Frame(box)
        btns.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.connect_btn = ttk.Button(btns, text="Connect", command=self.connect)
        self.connect_btn.pack(side="left")
        self.disconnect_btn = ttk.Button(btns, text="Disconnect", command=self.disconnect, state="disabled")
        self.disconnect_btn.pack(side="left", padx=6)
        self.newchat_btn = ttk.Button(btns, text="New chat", command=self.new_chat, state="disabled")
        self.newchat_btn.pack(side="left")
        self.status = ttk.Label(btns, text="Not connected.")
        self.status.pack(side="left", padx=12)

        self.privacy = ttk.Label(frm, text="", wraplength=700)
        self.privacy.pack(anchor="w", pady=(6, 2))

        chat_frame = ttk.Frame(frm)
        chat_frame.pack(fill="both", expand=True)
        self.chat = tk.Text(chat_frame, wrap="word", height=12, font=("Segoe UI", 10), state="disabled")
        sb = ttk.Scrollbar(chat_frame, command=self.chat.yview)
        self.chat.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.chat.pack(side="left", fill="both", expand=True)
        self.chat.tag_configure("user", foreground="#0b4f9e", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("tool", foreground="#666666", font=("Consolas", 9))
        self.chat.tag_configure("assistant", foreground="#111111")
        self.chat.tag_configure("error", foreground="#b00020")
        self.chat.tag_configure("warn", foreground="#9a6700", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("note", foreground="#666666", font=("Segoe UI", 9, "italic"))

        self.bar = ttk.Progressbar(frm, maximum=100)
        self.bar.pack(fill="x", pady=(6, 0))

        entry_row = ttk.Frame(frm)
        entry_row.pack(fill="x", pady=(6, 0))
        self.input = ttk.Entry(entry_row)
        self.input.pack(side="left", fill="x", expand=True)
        self.input.bind("<Return>", lambda e: self.send())
        self.send_btn = ttk.Button(entry_row, text="Send", command=self.send, state="disabled")
        self.send_btn.pack(side="left", padx=(6, 0))
        self.input.configure(state="disabled")

        if not saved.get("endpoint"):
            self.endpoint_var.set(DEFAULT_ENDPOINT[self.provider_id()])
        self.refresh_provider_widgets()
        self.update_privacy()
        self.note("Pick a provider, press Connect, then ask: what does <your video>.mp4 say?")

    # ----- small helpers
    def provider_id(self):
        return LABEL_TO_ID.get(self.provider_var.get(), "openai")

    def note(self, text, tag="note"):
        self.chat.configure(state="normal")
        self.chat.insert("end", text + "\n", tag)
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def set_status(self, text):
        self.status.configure(text=text)

    def update_privacy(self):
        text, color = privacy_note(self.endpoint_var.get())
        self.privacy.configure(text=text, foreground=color)

    def refresh_provider_widgets(self):
        pid = self.provider_id()
        needs_key = agent.PROVIDERS[pid].get("needs_key", True)
        self.key_entry.configure(state="normal" if needs_key or pid == "custom" else "disabled")
        self.find_btn.configure(state="normal" if pid in ("ollama", "custom") else "disabled")

    def suggest_folder(self, video_path):
        """Called by the Decode tab: default the video folder to where the last video was."""
        if not self.folder_var.get():
            self.folder_var.set(os.path.dirname(os.path.abspath(video_path)))

    # ----- settings widgets
    def on_provider(self, _event=None):
        pid = self.provider_id()
        self.endpoint_var.set(DEFAULT_ENDPOINT[pid])
        self.model_var.set(agent.PROVIDERS[pid].get("model") or "")
        self.model_box.configure(values=[])
        self.key_entry.configure(state="normal")
        self.key_entry.delete(0, "end")
        self.refresh_provider_widgets()
        self.update_privacy()
        if pid == "ollama":
            self.find_models(quiet=True)

    def find_models(self, quiet=False):
        models = agent.list_models(self.endpoint_var.get(), timeout=1.5 if quiet else 5)
        self.model_box.configure(values=models)
        if models:
            if not self.model_var.get():
                self.model_var.set(models[0])
            self.set_status(f"Found {len(models)} model(s). Pick one that supports tools.")
        elif not quiet:
            self.set_status("No models found. Is Ollama running? Install one with: ollama pull <name>")

    def browse(self):
        folder = filedialog.askdirectory(title="Folder that contains your Ghost Font videos")
        if folder:
            self.folder_var.set(folder)
            if self.tools:
                try:
                    self.tools.set_root(folder)
                except agent.ToolError as e:
                    self.set_status(str(e))

    # ----- connection
    def connect(self):
        folder = self.folder_var.get().strip()
        if not os.path.isdir(folder):
            self.set_status("Choose the folder that holds your videos first.")
            return
        key = self.key_entry.get().strip()
        try:
            backend = agent.make_backend(
                self.provider_id(), key, self.model_var.get().strip(),
                self.endpoint_var.get().strip(), self.effort_var.get().strip() or None)
        except agent.AgentError as e:
            self.set_status(str(e))
            return
        finally:
            key = None
        self.key_entry.delete(0, "end")            # the key now lives only in `backend`
        self.backend = backend
        self.tools = agent.Tools(folder, self.cache_dir,
                                 progress=lambda s, d, t: self.q.put(("progress", s, d, t)))
        self.shown_model = False
        save_settings({
            "provider": self.provider_id(), "model": self.model_var.get().strip(),
            "endpoint": self.endpoint_var.get().strip(),
            "reasoning_effort": self.effort_var.get().strip(), "folder": folder})
        self.connect_btn.configure(state="disabled")
        self.disconnect_btn.configure(state="normal")
        self.newchat_btn.configure(state="normal")
        self.send_btn.configure(state="normal")
        self.input.configure(state="normal")
        self.provider_box.configure(state="disabled")
        pid = self.provider_id()
        self.set_status("Connected. Key held in memory only." if key_used(pid, self.endpoint_var.get())
                        else "Connected (no key needed).")
        self.note("Connected. Ask away.")
        if pid == "ollama" or agent.is_local_url(self.endpoint_var.get()):
            self.note("Local models vary a lot at tool use. If one answers without running the "
                      "decoder, you'll see a warning; try a larger model.")
        self.input.focus_set()

    def disconnect(self):
        self.backend = None
        self.tools = None
        self.connect_btn.configure(state="normal")
        self.disconnect_btn.configure(state="disabled")
        self.newchat_btn.configure(state="disabled")
        self.send_btn.configure(state="disabled")
        self.input.configure(state="disabled")
        self.provider_box.configure(state="readonly")
        self.set_status("Disconnected. The key was discarded.")

    def new_chat(self):
        if self.backend and not self.busy:
            self.backend.reset()
            self.chat.configure(state="normal")
            self.chat.delete("1.0", "end")
            self.chat.configure(state="disabled")
            self.note("New chat started.")

    # ----- chatting (worker thread keeps the window responsive)
    def send(self):
        text = self.input.get().strip()
        if not text or self.busy or not self.backend:
            return
        self.input.delete(0, "end")
        self.note("You: " + text, "user")
        self.busy = True
        self.send_btn.configure(state="disabled")
        self.bar["value"] = 0
        self.set_status("Thinking...")
        threading.Thread(target=self._work, args=(text,), daemon=True).start()
        self.root.after(100, self._poll)

    def _work(self, text):
        decoded = {"called": False}

        def on_tool(name, args):
            self.q.put(("tool_call", name, args))
            result = self.tools.run(name, args)
            if name == "decode_ghost_font":
                decoded["called"] = True
            self.q.put(("tool_result", name, result))
            return result

        try:
            answer = self.backend.ask(text, on_tool)
            self.q.put(("answer", answer, getattr(self.backend, "reported_model", None)))
            if VIDEO_NAME_RE.search(text) and not decoded["called"]:
                self.q.put(("warn", "The model answered without running the decoder, so that "
                                    "answer did NOT come from the decoder. Don't trust it."))
        except agent.AgentError as e:
            self.q.put(("error", str(e)))
        except Exception as e:
            self.q.put(("error", f"Unexpected error: {e}"))
        self.q.put(("end",))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, stage, done, total = msg
                    self.bar["value"] = progress_percent(stage, done, total)
                    self.set_status(status_text(stage, done, total))
                elif kind == "tool_call":
                    shown = ", ".join(f"{k}={v!r}" for k, v in (msg[2] or {}).items())
                    self.note(f"  \u25b6 running tool: {msg[1]}({shown})", "tool")
                    if msg[1] == "decode_ghost_font":
                        self.set_status("Decoding... this takes one to a few minutes.")
                elif kind == "tool_result":
                    res = msg[2]
                    if "error" in res:
                        self.note(f"  \u2716 tool error: {res['error']}", "tool")
                    elif msg[1] == "decode_ghost_font":
                        self.note(f"  \u2714 decoder read: {res['text']!r} "
                                  f"(agreement {round(100 * res['agreement'])}%)", "tool")
                    else:
                        self.note(f"  \u2714 {len(res.get('videos', []))} video(s) found", "tool")
                elif kind == "answer":
                    self.note("AI: " + (msg[1] or "(no answer)"), "assistant")
                    if msg[2] and not self.shown_model:
                        self.shown_model = True
                        self.note(f"  (the API says it answered with model: {msg[2]})", "note")
                elif kind == "warn":
                    self.note("\u26a0 " + msg[1], "warn")
                elif kind == "error":
                    self.note("Error: " + msg[1], "error")
                    hint = error_hint(msg[1])
                    if hint:
                        self.note(hint, "note")
                elif kind == "end":
                    self.busy = False
                    self.bar["value"] = 0
                    self.set_status("Connected." if self.backend else "Disconnected.")
                    if self.backend:
                        self.send_btn.configure(state="normal")
        except queue.Empty:
            pass
        if self.busy:
            self.root.after(100, self._poll)

    def shutdown(self):
        self.backend = None
        shutil.rmtree(self.cache_dir, ignore_errors=True)


def key_used(provider, endpoint):
    return agent.key_required(provider, agent.normalize_endpoint(endpoint)) or provider == "custom"


# ---------------------------------------------------------------- the window
class App:
    def __init__(self, root):
        self.root = root
        root.title("GhostDecode")
        root.geometry("780x760")
        root.minsize(640, 600)
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)
        decode_frame = ttk.Frame(self.nb)
        ask_frame = ttk.Frame(self.nb)
        self.nb.add(decode_frame, text="  Decode  ")
        self.nb.add(ask_frame, text="  Ask AI  ")
        self.decode_frame = decode_frame
        self.ask = AskTab(root, ask_frame)
        self.decode = DecodeTab(root, decode_frame, on_video=self.ask.suggest_folder)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def start(self, path):
        self.nb.select(self.decode_frame)
        self.decode.start(path)

    def close(self):
        self.ask.shutdown()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    path = next((a for a in sys.argv[1:] if os.path.isfile(a)), None)
    if path:
        root.after(300, lambda: app.start(path))
    root.mainloop()


if __name__ == "__main__":
    main()
