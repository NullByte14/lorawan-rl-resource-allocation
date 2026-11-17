"""Create a checked portable ZIP, excluding environments/caches/demo outputs."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
EXCLUDED = {".venv", ".tmp", ".mplcache", ".mplconfig", "__pycache__",
            "quick_results", "quick_figures", "reproduced_results", "reproduced_figures"}


def main():
    paths = sorted(p for p in HERE.rglob("*") if p.is_file()
                   and not EXCLUDED.intersection(p.relative_to(HERE).parts)
                   and p.name != "MANIFEST.json")
    files = {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in paths}
    manifest_path = HERE/"MANIFEST.json"
    manifest_path.write_text(json.dumps(dict(algorithm="SHA256", files=files), indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(HERE/"verify.py")], cwd=HERE, check=True)
    destination = HERE.parent/"lorawan-feedback-experiment-bundle.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths+[manifest_path]:
            archive.write(path, HERE.name+"/"+path.relative_to(HERE).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        members = len(archive.namelist())
    print(json.dumps(dict(bundle=str(destination), members=members,
                          bytes=destination.stat().st_size,
                          sha256=hashlib.sha256(destination.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()
