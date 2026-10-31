"""
Smoke tests. They need the Tesseract OCR engine and Pillow installed:

    pip install pillow pytest
    pytest -q

The end-to-end test builds a short synthetic Ghost Font video with a known
text and checks that the decoder reads it back (takes about a minute).
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ghostdecode as gd  # noqa: E402

try:
    gd.setup_tesseract()
    HAVE_TESSERACT = True
except gd.GhostDecodeError:
    HAVE_TESSERACT = False

needs_tesseract = pytest.mark.skipif(not HAVE_TESSERACT, reason="Tesseract not installed")


def test_key_normalises_text():
    assert gd.key("Hello, World 9:30!") == "HELLOWORLD930"


@needs_tesseract
def test_missing_video_raises_clean_error():
    with pytest.raises(gd.GhostDecodeError):
        gd.decode(os.path.join(ROOT, "does_not_exist.mp4"))


@needs_tesseract
def test_decodes_synthetic_video(tmp_path):
    pytest.importorskip("PIL")
    video = tmp_path / "t.mp4"
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "make_synthetic.py"),
         "HELLO WORLD", "--frames", "60", "--out", str(video)],
        check=True, capture_output=True)
    res = gd.decode(str(video))
    assert res["text"] == "HELLO WORLD"
    assert res["agreement"] >= 0.5
    assert os.path.isfile(res["heatmap_path"])
