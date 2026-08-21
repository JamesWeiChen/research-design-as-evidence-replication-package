"""Shared monochrome style and validated export helpers."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader

INK = "#111111"
DARK_GRAY = "#4D4D4D"
MID_GRAY = "#777777"
LIGHT_GRAY = "#D8D8D8"
PALE_GRAY = "#EFEFEF"

TR_LINE = {"color": DARK_GRAY, "linestyle": (0, (5, 3)), "linewidth": 1.8}
PR_LINE = {"color": INK, "linestyle": "-", "linewidth": 1.8}
TR_REFERENCE = {
    "color": MID_GRAY,
    "linestyle": (0, (5, 3)),
    "linewidth": 0.9,
}
PR_REFERENCE = {
    "color": MID_GRAY,
    "linestyle": (0, (1.5, 2.2)),
    "linewidth": 0.9,
}
CUTOFF_REFERENCE = {
    "color": MID_GRAY,
    "linestyle": (0, (5, 3)),
    "linewidth": 0.9,
}


def format_significant(value: float, digits: int = 3, *, signed: bool = False) -> str:
    """Return fixed-point text rounded to the requested significant digits."""

    if digits < 1:
        raise ValueError("digits must be positive")
    if not math.isfinite(value):
        raise ValueError("value must be finite")

    magnitude = abs(value)
    if magnitude == 0.0:
        decimal_places = digits - 1
    else:
        exponent = math.floor(math.log10(magnitude))
        decimal_places = digits - exponent - 1
        rounded = round(magnitude, decimal_places)
        if rounded > 0.0:
            rounded_exponent = math.floor(math.log10(rounded))
            decimal_places = digits - rounded_exponent - 1

    if decimal_places >= 0:
        body = f"{magnitude:.{decimal_places}f}"
    else:
        scale = 10.0 ** (-decimal_places)
        body = f"{round(magnitude / scale) * scale:.0f}"

    if value < 0.0:
        return f"-{body}"
    if signed and value > 0.0:
        return f"+{body}"
    return body


def apply_nhb_style() -> None:
    """Apply the font, line, and export settings used by both figures."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    """Apply the common open-axis treatment."""

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3.2, width=0.8)


def _validate_figure_files(pdf: Path, png: Path) -> None:
    """Check that staged files are readable and contain one figure page."""

    with pdf.open("rb") as stream:
        reader = PdfReader(stream)
        if len(reader.pages) != 1:
            raise RuntimeError("figure PDF must contain exactly one page")
        reader.pages[0]
    with Image.open(png) as image:
        if image.format != "PNG":
            raise RuntimeError("figure raster output is not PNG")
        image.verify()


def save_pdf_png(fig: plt.Figure, output_stem: Path) -> None:
    """Atomically save a validated vector PDF and a 350-dpi PNG."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    final_pdf = output_stem.with_suffix(".pdf")
    final_png = output_stem.with_suffix(".png")
    stage_pdf = final_pdf.with_name(f".{final_pdf.name}.staging")
    stage_png = final_png.with_name(f".{final_png.name}.staging")

    with tempfile.TemporaryDirectory(prefix="nhb-figure-render-") as temp_dir:
        temp_root = Path(temp_dir)
        temp_pdf = temp_root / final_pdf.name
        temp_png = temp_root / final_png.name
        fig.savefig(
            temp_pdf,
            bbox_inches="tight",
            metadata={
                "Creator": "NHB replication package",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        fig.savefig(
            temp_png,
            dpi=350,
            bbox_inches="tight",
            metadata={"Software": "NHB replication package"},
        )
        _validate_figure_files(temp_pdf, temp_png)

        try:
            shutil.copyfile(temp_pdf, stage_pdf)
            shutil.copyfile(temp_png, stage_png)
            _validate_figure_files(stage_pdf, stage_png)
            os.replace(stage_pdf, final_pdf)
            os.replace(stage_png, final_png)
        finally:
            stage_pdf.unlink(missing_ok=True)
            stage_png.unlink(missing_ok=True)
