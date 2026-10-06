"""Download pinned local inference assets; never send questions to a paid API."""

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "judge"
MODEL_REVISION = "7dabda4d13d513e3e842b20f0d435c732f172cbe"
MODEL_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_URL = f"https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/{MODEL_REVISION}/{MODEL_NAME}"
MODEL_SHA256 = "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
BINARY_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11433/llama-b11433-bin-win-cpu-x64.zip"
BINARY_SHA256 = "301943380e45deae3991d42dddd03fae65c6a4963a5d882328a60208b43aa9e0"


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_checked(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"Checksum mismatch in existing {destination.name}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if sha256_file(partial) != expected_sha256:
        raise ValueError(f"Downloaded checksum mismatch: {destination.name}")
    partial.replace(destination)
    print(f"Verified {destination.name}", flush=True)


def main() -> None:
    if sys.platform == "win32":
        archive = CACHE / "llama-b11433-win-cpu.zip"
        download_checked(BINARY_URL, archive, BINARY_SHA256)
        binary_dir = CACHE / "llama"
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if not (binary_dir / name).resolve().is_relative_to(binary_dir.resolve()):
                    raise ValueError("Archive member escapes the inference directory")
            bundle.extractall(binary_dir)
        print(f"Inference binaries: {binary_dir}", flush=True)
    download_checked(MODEL_URL, CACHE / MODEL_NAME, MODEL_SHA256)
    print("Start llama-server on loopback with this model and alias evidencerag-local.", flush=True)


if __name__ == "__main__":
    main()
