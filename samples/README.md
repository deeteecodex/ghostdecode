# samples

Generated sample videos go here. They aren't committed to git (video files are
ignored). Make one in about 40 seconds:

    pip install pillow
    python tools/make_synthetic.py "HELLO WORLD" --out samples/hello.mp4
    python ghostdecode.py samples/hello.mp4
