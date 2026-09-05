#!/usr/bin/env python3
"""Check release files; use --archive for byte-identical distributed outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
GENERATED_FILES = {
    f"figures/figure{n}/output/figure{n}.{suffix}"
    for n in (2, 3) for suffix in ("pdf", "png")
} | {
    f"figures/figure{n}/output/figure{n}_{suffix}"
    for n in (2, 3) for suffix in ("values.csv", "caption.md")
} | {f"results/figure{n}_results.json" for n in (2, 3)}
REQUIRED_FILES = GENERATED_FILES | {
    "README.md", "VERSION", "CHANGELOG.md", "ACCEPTANCE.md", "MANIFEST.sha256",
    "requirements.txt", "requirements-tested.txt", "run_all.sh", "verify_release.py",
    "docs/SOLVER_GUIDE.md", "docs/REPRODUCIBILITY.md", "docs/DATA_DICTIONARY.md",
    "figures/validation.py", "LICENSE", "CITATION.cff",
    "tests/test_solver_smoke.py", "tests/test_release_validation.py",
}
IGNORED_DIRECTORIES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".AppleDouble"}


def distributed_files(root: Path = ROOT) -> dict[str, Path]:
    """Exclude known local files, but reject unexplained additions and symlinks."""
    files = {}
    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(root).as_posix()
            if (path.name in IGNORED_DIRECTORIES or path.name in {".DS_Store", ".LSOverride"}
                    or path.name.startswith("._") or path.suffix in {".pyc", ".pyo"}
                    or relative == ".vscode/settings.json"):
                continue
            if path.is_symlink():
                raise RuntimeError(f"release must not contain symlinks: {relative}")
            if path.is_dir():
                visit(path)
            elif path.is_file() and relative != "MANIFEST.sha256":
                files[relative] = path
    visit(root)
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(root: Path) -> dict[str, str]:
    entries = {}
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError("malformed manifest line") from exc
        path = PurePosixPath(relative)
        if (not re.fullmatch(r"[0-9a-f]{64}", expected) or path.is_absolute()
                or ".." in path.parts or path.as_posix() != relative
                or relative in entries or relative == "MANIFEST.sha256"):
            raise RuntimeError(f"invalid or duplicate manifest entry: {relative}")
        entries[relative] = expected
    return entries


def check_manifest(root: Path = ROOT, *, archive: bool = False) -> None:
    entries = read_manifest(root)
    files = distributed_files(root)
    if set(entries) != set(files):
        raise RuntimeError(f"manifest file set mismatch; unlisted={sorted(set(files)-set(entries))}, "
                           f"missing={sorted(set(entries)-set(files))}")
    for relative, expected in entries.items():
        if not archive and relative in GENERATED_FILES:
            continue
        if sha256(files[relative]) != expected:
            raise RuntimeError(f"manifest hash mismatch: {relative}")


def check_metadata(root: Path = ROOT) -> str:
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"missing required release files: {missing}")
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"20\d{6}-v\d+", version):
        raise RuntimeError("invalid VERSION")
    for n in (2, 3):
        registry = json.loads((root / f"results/figure{n}_results.json").read_text())
        if registry["release_version"] != version:
            raise RuntimeError(f"Figure {n} has a stale release version")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--archive", action="store_true", help="check all original distributed bytes")
    mode.add_argument("--write-manifest", action="store_true", help="maintainer: deliberately replace release hashes")
    args = parser.parse_args()
    if args.write_manifest:
        files = distributed_files()
        (ROOT / "MANIFEST.sha256").write_text(
            "".join(f"{sha256(path)}  {name}\n" for name, path in sorted(files.items())),
            encoding="utf-8",
        )
        print("Wrote MANIFEST.sha256; review it before publishing this release.")
        return
    version = check_metadata()
    check_manifest(archive=args.archive)
    if args.archive:
        print(f"PASS: {version} archive file set and all distributed hashes match.")
    else:
        print(f"PASS: {version} file set, source hashes, and release metadata match.")
        print("Generated-output hashes are checked only with --archive; run_all.sh validates recomputed numerics.")


if __name__ == "__main__":
    main()
