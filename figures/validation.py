"""Validate every CSV cell against its numerical registry."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def require_finite(value: object, path: str = "registry") -> None:
    """Reject non-finite JSON numbers before ordering or residual checks."""
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")


def validate_csv(path: Path, registry: dict, *, figure: int) -> None:
    """Check schema, row order, and all values; numerical tolerance is 1e-12."""
    if figure not in (2, 3):
        raise ValueError("expected Figure 2 or 3")
    expected_rows = []
    for label in registry["case_order"]:
        case = registry["cases"][label]
        tr, pr = case["TR"], case["PR"]
        expected = {
            "case": label, "prior": case["prior"],
            "c_p": case["parameters"]["c_p"],
            "delta0": case["parameters"]["delta0"],
            "TR_acceptance": tr["acceptance"], "PR_acceptance": pr["acceptance"],
            "U_R_TR": tr["U_R" if figure == 2 else "U_R_mix"],
            "U_R_PR": pr["U_R"], "d_PR": pr["d"], "c_PR": pr["c"],
            "U_R_PR_minus_TR": (case["comparison"]["U_R_PR_minus_TR"]
                                  if figure == 2 else case["U_R_PR_minus_TR"]),
        }
        if figure == 2:
            expected.update(d_TR=tr["d"], c_TR=tr["c"],
                            U_S_TR=tr["U_S"], U_S_PR=pr["U_S"])
        else:
            expected["TR_type"] = tr["type"]
        expected_rows.append(expected)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or set(columns) != set(expected_rows[0]):
            raise ValueError(f"{path.name}: missing, duplicate, or unexpected columns")
        rows = list(reader)
    if len(rows) != len(expected_rows):
        raise ValueError(f"{path.name}: wrong number of rows")
    for row, expected in zip(rows, expected_rows):
        if set(row) != set(expected):
            raise ValueError(f"{path.name}: malformed row")
        for column, target in expected.items():
            actual = row[column]
            location = f"{path.name}:{expected['case']}:{column}"
            if isinstance(target, (int, float)):
                try:
                    value = float(actual)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{location}: invalid number") from exc
                if not math.isfinite(value) or not math.isclose(
                    value, target, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"{location}: CSV {actual!r} != registry {target!r}")
            elif actual != target:
                raise ValueError(f"{location}: CSV {actual!r} != registry {target!r}")
