#!/usr/bin/env python3
"""Validate public metadata, source hygiene, outputs, and file hashes."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
TEXT_SUFFIXES = {".py", ".md", ".sh", ".txt", ".json", ".csv"}

REQUIRED_FILES = {
    "README.md",
    "VERSION",
    "CHANGELOG.md",
    "ACCEPTANCE.md",
    "MANIFEST.sha256",
    "requirements.txt",
    "run_all.sh",
    "verify_release.py",
    "docs/SOLVER_GUIDE.md",
    "results/figure2_results.json",
    "results/figure3_results.json",
    "figures/figure2/output/figure2.pdf",
    "figures/figure2/output/figure2.png",
    "figures/figure2/output/figure2_values.csv",
    "figures/figure2/output/figure2_caption.md",
    "figures/figure3/output/figure3.pdf",
    "figures/figure3/output/figure3.png",
    "figures/figure3/output/figure3_values.csv",
    "figures/figure3/output/figure3_caption.md",
}


def text_files() -> list[Path]:
    """Return distributed text files, excluding generated cache directories."""

    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix in TEXT_SUFFIXES
    )


def check_required_files() -> None:
    missing = sorted(name for name in REQUIRED_FILES if not (ROOT / name).is_file())
    if missing:
        raise RuntimeError(f"missing required release files: {missing}")


def check_source_hygiene() -> None:
    cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
    version_pattern = re.compile(r"20\d{6}-v\d+")
    versions: set[str] = set()

    blocked_fragments = (
        "/" + "Users/",
        "/work" + "space/",
        "Drop" + "box",
        "source_" + "solver",
        "locked_" + "table",
        "m_" + "flags",
        "uniform_" + "completion",
        "SUBMISSION_" + "CONVENTION",
        "back-" + "compat",
        "frozen " + "core",
        "supplied " + "Figure",
    )
    issue_tag = re.compile(r"\[[A-Z]-\d")

    for path in text_files():
        text = path.read_text(encoding="utf-8")
        versions.update(version_pattern.findall(text))
        if path.suffix == ".py" and cjk.search(text):
            raise RuntimeError(f"non-English code text found in {path.relative_to(ROOT)}")
        if issue_tag.search(text):
            raise RuntimeError(f"development issue tag found in {path.relative_to(ROOT)}")
        for fragment in blocked_fragments:
            if fragment in text:
                raise RuntimeError(
                    f"blocked development fragment {fragment!r} found in "
                    f"{path.relative_to(ROOT)}"
                )

    if versions != {VERSION}:
        raise RuntimeError(
            f"release must contain one public version; found {sorted(versions)}"
        )

    caches = [path for path in ROOT.rglob("__pycache__") if path.is_dir()]
    if caches:
        raise RuntimeError("Python cache directories must not be distributed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifest() -> None:
    entries: dict[str, str] = {}
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        entries[relative] = expected

    distributed = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    if set(entries) != distributed:
        missing = sorted(distributed - set(entries))
        extra = sorted(set(entries) - distributed)
        raise RuntimeError(f"manifest file set mismatch; missing={missing}, extra={extra}")

    for relative, expected in entries.items():
        observed = sha256(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"manifest hash mismatch: {relative}")


def main() -> None:
    check_required_files()
    check_source_hygiene()
    check_manifest()
    print(f"PASS: public release {VERSION} is complete and internally consistent.")


if __name__ == "__main__":
    main()
