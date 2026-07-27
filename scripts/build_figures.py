#!/usr/bin/env python
"""Regenerate every example figure and refresh the copies the docs serve.

The examples write vector PDFs (and two animated GIFs) into ``examples/figures/``,
but the documentation embeds rasters from ``docs/img/examples/``. Nothing used to
connect the two, so a change to the plotting theme updated the PDFs and left the
site showing the previous palette. This script is that connection: it runs each
example once, captures every figure as a PNG at the path the docs reference, and
copies the GIFs across.

Usage::

    python scripts/build_figures.py            # all examples
    python scripts/build_figures.py q_gaussian learnable_q   # a subset
    python scripts/build_figures.py --check    # report stale doc images

``--check`` regenerates nothing; it compares modification times. That makes it a
useful local check before committing, but *not* a CI gate: a fresh clone gives
every file the same checkout timestamp, so the comparison is meaningless there.
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
FIGURES = EXAMPLES / "figures"
DOCS_IMG = ROOT / "docs" / "img" / "examples"

#: Raster resolution for the docs copies. High enough to stay crisp on a
#: HiDPI screen without bloating the repository.
DPI = 150

#: Animations are written by Matplotlib's writer rather than ``savefig``, so they
#: are copied verbatim instead of being re-rastered.
GIFS = ("optimization.gif", "attention_q_learning.gif")


def example_scripts() -> list[Path]:
    """Every runnable example, in a stable order."""
    return sorted(p for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))


def run_example(script: Path) -> list[Path]:
    """Run one example, mirroring each saved figure into the docs image folder.

    Returns:
        The doc-image paths written while the example ran.
    """
    written: list[Path] = []
    original = plt.Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        original(self, fname, *args, **kwargs)
        # Animation writers call savefig once per frame with an open file object
        # rather than a path. Those frames are not figures the docs embed -- the
        # finished GIF is copied separately -- so only mirror real paths.
        if not isinstance(fname, str | Path):
            return
        target = DOCS_IMG / f"{Path(fname).stem}.png"
        # Drop any PDF-specific kwargs the caller passed before re-saving.
        kwargs.pop("format", None)
        original(self, str(target), dpi=DPI, bbox_inches="tight")
        written.append(target)

    plt.Figure.savefig = savefig  # type: ignore[method-assign]
    argv = sys.argv[:]
    try:
        sys.argv = [str(script)]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        plt.Figure.savefig = original  # type: ignore[method-assign]
        sys.argv = argv
        plt.close("all")
    return written


def copy_gifs() -> list[Path]:
    """Copy the animated figures into the docs image folder."""
    copied = []
    for name in GIFS:
        source = FIGURES / name
        if source.exists():
            target = DOCS_IMG / name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def check_staleness() -> int:
    """Report doc images older than the figure they are derived from.

    Modification-time based, so it is meaningful only in a working tree where
    the figures were actually rebuilt -- not after a fresh clone.
    """
    stale = []
    for image in sorted(DOCS_IMG.iterdir()):
        source = FIGURES / f"{image.stem}{'.gif' if image.suffix == '.gif' else '.pdf'}"
        if source.exists() and source.stat().st_mtime > image.stat().st_mtime:
            stale.append(image.name)
    if stale:
        print("Stale doc images (regenerate with scripts/build_figures.py):")
        for name in stale:
            print(f"  {name}")
        return 1
    print(f"All {len(list(DOCS_IMG.iterdir()))} doc images are current.")
    return 0


def main() -> int:
    """Parse arguments and regenerate (or check) the documentation figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="example stems to rebuild; default all")
    parser.add_argument("--check", action="store_true", help="only report stale doc images")
    args = parser.parse_args()

    DOCS_IMG.mkdir(parents=True, exist_ok=True)
    if args.check:
        return check_staleness()

    scripts = example_scripts()
    if args.names:
        wanted = set(args.names)
        scripts = [s for s in scripts if s.stem in wanted]
        missing = wanted - {s.stem for s in scripts}
        if missing:
            parser.error(f"no such example(s): {', '.join(sorted(missing))}")

    for script in scripts:
        print(f"── {script.name}")
        for path in run_example(script):
            print(f"   wrote {path.relative_to(ROOT)}")

    for path in copy_gifs():
        print(f"   copied {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
