"""Check released LightGBM files without importing LightGBM or running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_header(path: Path) -> dict[str, str]:
    header = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.rstrip("\n")
            if line.startswith("Tree="):
                break
            if "=" in line:
                key, value = line.split("=", 1)
                header[key] = value
    return header


def verify_models(root: Path, manifest_path: Path) -> list[str]:
    root = root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_features = manifest["feature_columns"]
    checked = []
    for model in manifest["models"]:
        relative = Path(model["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe model path: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != model["bytes"] or sha256_file(path) != model["sha256"]:
            raise ValueError(f"Model checksum differs: {relative}")
        header = model_header(path)
        if header.get("feature_names", "").split() != expected_features:
            raise ValueError(f"Feature order differs: {relative}")
        if len(header.get("tree_sizes", "").split()) != model["trees"]:
            raise ValueError(f"Tree count differs: {relative}")
        checked.append(relative.as_posix())
    if not checked:
        raise ValueError("No models in manifest.")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest or args.root / "models" / "manifest.json"
    for path in verify_models(args.root, manifest):
        print(f"OK {path}")


if __name__ == "__main__":
    main()
