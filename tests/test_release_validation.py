"""Regression tests for damaged outputs and release integrity boundaries."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from figures.validation import require_finite, validate_csv
from verify_release import check_manifest, distributed_files, sha256

ROOT = Path(__file__).resolve().parents[1]


class CsvValidationTests(unittest.TestCase):
    def test_every_cell_is_checked(self):
        for figure in (2, 3):
            registry = json.loads((ROOT / f"results/figure{figure}_results.json").read_text())
            original = ROOT / f"figures/figure{figure}/output/figure{figure}_values.csv"
            validate_csv(original, registry, figure=figure)
            with original.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "values.csv"
                for index, row in enumerate(rows):
                    for column in row:
                        with self.subTest(figure=figure, row=index, column=column):
                            damaged = copy.deepcopy(rows)
                            damaged[index][column] = "999"
                            with path.open("w", newline="") as stream:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                                writer.writerows(damaged)
                            with self.assertRaises(ValueError):
                                validate_csv(path, registry, figure=figure)

    def test_nonfinite_registry_values_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                require_finite({"cases": [{"residual": value}]})

    def test_nonfinite_csv_and_bad_schema_are_rejected(self):
        registry = json.loads((ROOT / "results/figure3_results.json").read_text())
        original = (ROOT / "figures/figure3/output/figure3_values.csv").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "values.csv"
            for text in (original.replace("0.8333333333333334", "nan"),
                         original.replace("PR_acceptance", "TR_acceptance"),
                         original + original.splitlines()[1] + "\n"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    validate_csv(path, registry, figure=3)


class ManifestTests(unittest.TestCase):
    def test_output_rerun_does_not_hide_source_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run_all.sh"
            source.write_text("original source")
            output = root / "results/figure2_results.json"
            output.parent.mkdir()
            output.write_text("original output")
            files = distributed_files(root)
            (root / "MANIFEST.sha256").write_text("".join(
                f"{sha256(path)}  {name}\n" for name, path in files.items()))
            for local in (".git/config", ".venv/pyvenv.cfg", ".vscode/settings.json",
                          "results/.DS_Store", "solver/__pycache__/ignored.pyc"):
                path = root / local
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("local")
            check_manifest(root, archive=True)
            output.write_text("recomputed output")
            check_manifest(root)
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                check_manifest(root, archive=True)
            source.write_text("tampered source")
            with self.assertRaisesRegex(RuntimeError, "run_all.sh"):
                check_manifest(root)

    def test_unlisted_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "MANIFEST.sha256").write_text("")
            (root / "unexpected.txt").write_text("unlisted")
            with self.assertRaisesRegex(RuntimeError, "unlisted"):
                check_manifest(root)

    def test_external_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outside").symlink_to(ROOT / "README.md")
            with self.assertRaisesRegex(RuntimeError, "symlinks"):
                distributed_files(root)


if __name__ == "__main__":
    unittest.main()
