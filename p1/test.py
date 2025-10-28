import os
import shlex
from pathlib import Path

if __name__ == '__main__':
    data_dir = Path(__file__).resolve().parent / "data"
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            if file.lower().endswith(".png"):
                img_path = Path(root) / file
                cmd = f"python -m p1.main {shlex.quote(str(img_path))}"
                os.system(cmd)
