#!/usr/bin/env sh
set -eu

RELEASE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$RELEASE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nhb-matplotlib"

cd "$RELEASE_ROOT"

python -m unittest discover -s tests -p 'test_*.py'
python figures/figure2/make_figure2.py
python figures/figure2/verify_figure2.py
python precision/compute_figure3.py
python figures/figure3/make_figure3.py
python figures/figure3/verify_figure3.py
python verify_release.py
