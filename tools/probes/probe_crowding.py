"""Crowding of the decision makers in a run's long-run probes, against the field
(section 278).

The long-run probe (lib/runners/training_progress.py) stores per probe and
whole probe year each decision maker's annual mean crowding (Lloyd's mean
crowding over the mean, lib/diagnostics/crowding.py: 1 = an even spread). This
prints one row per probe (the mean over its probe years) and the field reading
(NS-IBTS GOV in the Skagerrak, median and survey range) where one exists; the
last column of the header row is the model / field ratio of the latest probe.
Diagnostic only: no value may be fitted to it.

Usage:
    python3 tools/probes/probe_crowding.py results/<run-name> [--last N]

A probe from before section 278 has no crowding and is skipped.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.diagnostics.crowding import FIELD_CROWDING  # noqa: E402


def probe_means(record):
    """{group: mean over the probe's whole years} of one probe record, or None."""
    years = record.get("crowding") or []
    if not years:
        return None
    groups = sorted({g for y in years for g in y})
    return {g: sum(y[g] for y in years if g in y) / sum(1 for y in years if g in y)
            for g in groups}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--last", type=int, default=10, help="probes to print (default 10)")
    args = parser.parse_args(argv)
    path = Path(args.run_dir) / "progress" / "probe.jsonl"
    if not path.exists():
        sys.exit(f"no probe history at {path}")
    rows = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        means = probe_means(record)
        if means is not None:
            rows.append((record.get("step"), record.get("evaluation"), means))
    if not rows:
        sys.exit("no probe with crowding (section 278) in this history")
    rows = rows[-max(1, args.last):]
    groups = sorted({g for _, _, m in rows for g in m})
    width = max(len(g) for g in groups)
    print("crowding = Lloyd's mean crowding / mean (1 = even spread), mean over each probe's years")
    print(f"{'group':<{width}}  {'field (range)':>20}  " + "  ".join(f"{f'u{s}':>7}" for s, _, _ in rows)
          + "  latest/field")
    for g in groups:
        field = FIELD_CROWDING.get(g)
        ftxt = f"{field[0]:.2f} ({field[1]:.1f}-{field[2]:.1f})" if field else "-"
        cells = "  ".join(f"{m[g]:7.1f}" if g in m else f"{'-':>7}" for _, _, m in rows)
        latest = rows[-1][2].get(g)
        ratio = f"{latest / field[0]:10.1f}x" if field and latest is not None else ""
        print(f"{g:<{width}}  {ftxt:>20}  {cells}  {ratio}")
    print("field: " + "; ".join(f"{g} {v[3]}" for g, v in FIELD_CROWDING.items()))
    print("A GOV haul is a 3.4 km tow and the survey region is wider than the box: an order of "
          "magnitude, not a target.")


if __name__ == "__main__":
    main()
