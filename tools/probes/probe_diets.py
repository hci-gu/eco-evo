"""Realised diets of the long-run probe against the booked diets (section 248).

The viewer's long-run probe (lib/runners/training_progress.py) stores, per
probe and whole probe year, each decision maker's tonnes eaten per prey
and its ration (t eaten per t of mean biomass per year). This prints, for
one probe of a run, every predator's realised diet shares next to its
booked shares (M2 x B at the start densities, over the booked pairs - the
"booked" column of budget_gate's diet gate) and its ration next to the
field Q/B. It is the independent test of section 245's encounter diets on
trained policies. Diagnostic only: no value may be fitted to it.

Usage:
    python3 tools/probes/probe_diets.py results/<run-name> [--evaluation N]

Without --evaluation the latest probe is read. A probe from before
section 248 has no diets.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.diagnostics.field_diet import FIELD_QB  # noqa: E402

DIET_BAND = 3.0          # the diet gate's band (budget_gate)
DIET_MIN_SHARE = 0.05
PREY_MIN_SHARE = 0.10   # the prey-side check's share of the prey's booked M2 (budget_gate)


def start_densities(manifest):
    """t/km2 of every listed, unmuted group at its inference start."""
    meta = manifest.get('project_metadata', {})
    area = float(meta['reference_grid_width']) * float(meta['reference_grid_height'])
    out = {}
    for entries in manifest.values():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if (isinstance(e, dict) and 'group_id' in e and not e.get('muted')
                    and e.get('inference_initial_biomass') is not None):
                out[e['group_id']] = float(e['inference_initial_biomass']) / area
    return out


def booked_flows(library, density, pred):
    """{prey: M2 x B} over the predator's checked pairs on present prey."""
    out = {}
    for key, pair in (library.get('interaction_definitions') or {}).items():
        if not key.startswith(f"{pred}_preys_on_") or not (pair or {}).get('preys_on'):
            continue
        prey = key[len(f"{pred}_preys_on_"):]
        m2 = float(pair.get('predation_mortality', 0.0) or 0.0)
        if prey in density and m2 > 0.0:
            out[prey] = m2 * density[prey]
    return out


def year_biomass(record, progress):
    """Mean biomass (t) per group and probe year: the record's own
    ``year_biomass`` (section 249), else - for an older probe that is the
    run's latest - the daily trajectory times the decision makers' start
    biomass. None when neither is available."""
    if record.get('year_biomass'):
        return record['year_biomass']
    latest = progress / 'probe_latest.json'
    if not latest.exists():
        return None
    traj = json.loads(latest.read_text())
    if traj.get('evaluation') != record.get('evaluation'):
        return None
    every, per_year = int(traj['every_ticks']), int(traj['ticks_per_year'])
    days = per_year // every
    out = []
    for y in range(len(record['diets'])):
        row = {}
        for f, b0 in (record.get('initial_biomass') or {}).items():
            series = traj['series'].get(f)
            if series:
                part = series[1 + days * y: 1 + days * (y + 1)]
                row[f] = sum(part) / len(part) * b0
        out.append(row)
    return out


def prey_side(record, diets, library, density, progress):
    """Realised M2 per prey (tonnes eaten / mean biomass) against the booked
    M2, pair by pair (section 249). A pair carrying >= PREY_MIN_SHARE of
    its prey's booked M2 outside the band is OFF. Returns the OFF list,
    or None when the probe has no biomass to divide by."""
    biomass = year_biomass(record, progress)
    print()
    print("=== prey side: realised M2 (t eaten / mean prey biomass, /yr) vs booked M2 ===")
    if biomass is None:
        print("no yearly biomass in this probe (before section 249 and not the latest probe)")
        return None
    booked = {}
    for key, pair in (library.get('interaction_definitions') or {}).items():
        if '_preys_on_' not in key or not (pair or {}).get('preys_on'):
            continue
        pred, prey = key.split('_preys_on_', 1)
        m2 = float(pair.get('predation_mortality', 0.0) or 0.0)
        if m2 > 0.0 and pred in diets[0] and prey in density:
            booked.setdefault(prey, {})[pred] = m2
    failures = []
    for prey in sorted(booked):
        if not all(prey in yb for yb in biomass):
            continue
        row = booked[prey]
        total = sum(row.values())
        reals = []
        for y, yb in enumerate(biomass):
            b = yb[prey]
            reals.append({p: (diets[y].get(p, {}).get('tonnes', {}).get(prey, 0.0) / b if b > 0 else 0.0)
                          for p in row})
        print(f"{prey:>20}: booked {total:.4g}; realised " +
              " / ".join(f"{sum(r.values()):.4g}" for r in reals))
        for p, m2 in sorted(row.items(), key=lambda kv: -kv[1]):
            share = m2 / total
            cells = []
            for y, r in enumerate(reals):
                ratio = r[p] / m2
                off = share >= PREY_MIN_SHARE and not (1.0 / DIET_BAND <= ratio <= DIET_BAND)
                if off:
                    failures.append((p, prey, y + 1, r[p], m2))
                cells.append(f"{r[p]:8.4g} {ratio:5.2f}{'*' if off else ' '}")
            print(f"{'':>22}{p:>20} booked {m2:8.4g} ({share:4.2f})  " + "  ".join(cells))
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run', type=Path, help='results/<run-name>')
    parser.add_argument('--evaluation', type=int, default=None,
                        help='probe number (default: the latest)')
    args = parser.parse_args(argv)
    progress = args.run / 'progress'
    records = [json.loads(line) for line in (progress / 'probe.jsonl').read_text().splitlines()
               if line.strip()]
    if not records:
        sys.exit(f"{progress / 'probe.jsonl'}: no probes")
    if args.evaluation is None:
        record = records[-1]
    else:
        match = [r for r in records if r.get('evaluation') == args.evaluation]
        if not match:
            sys.exit(f"no probe {args.evaluation} in {progress / 'probe.jsonl'}")
        record = match[0]
    diets = record.get('diets')
    if not diets:
        sys.exit(f"probe {record.get('evaluation')} (update {record.get('step')}) has no diets "
                 f"(recorded before section 248, or shorter than a year)")
    config = json.loads((progress / 'config.json').read_text())
    with open(config['project'], encoding='utf-8-sig') as fh:
        density = start_densities(yaml.safe_load(fh))
    with open(config['library'], encoding='utf-8') as fh:
        library = yaml.safe_load(fh)

    print(f"run {args.run.name}, probe {record.get('evaluation')} at update {record.get('step')}: "
          f"{len(diets)} probe year(s)")
    print("share = realised share of the year's tonnes eaten; booked = M2 x B share over the booked")
    print(f"pairs; ratio outside 1/{DIET_BAND:g}-{DIET_BAND:g} on a share >= {DIET_MIN_SHARE:g} "
          f"is OFF (the diet gate's band); unbooked prey are marked '-'.")
    failures = []
    for pred in sorted(diets[0]):
        book = booked_flows(library, density, pred)
        book_sum = sum(book.values())
        rations = [y.get(pred, {}).get('ration') for y in diets]
        qb = FIELD_QB.get(pred)
        ration_txt = " / ".join("-" if r is None else f"{r:.2f}" for r in rations)
        print()
        print(f"{pred}: ration {ration_txt} t/t/yr"
              + (f" (field Q/B {qb[0]:g}, {qb[1]})" if qb else ""))
        prey_ids = sorted(set(book) | {q for y in diets for q in y.get(pred, {}).get('tonnes', {})})
        if not prey_ids:
            continue
        head = " ".join(f"{'year ' + str(k + 1):>14}" for k in range(len(diets)))
        print(f"  {'prey':>24} {'booked':>7} {head}")
        for q in prey_ids:
            k = book.get(q, 0.0) / book_sum if book_sum > 0.0 else 0.0
            cells = []
            for n, y in enumerate(diets):
                tonnes = y.get(pred, {}).get('tonnes', {})
                total = sum(tonnes.values())
                e = tonnes.get(q, 0.0) / total if total > 0.0 else 0.0
                if q not in book:
                    cells.append(f"{e:6.3f}      -")
                    continue
                ratio = e / k if k > 0.0 else float('inf')
                off = (max(e, k) >= DIET_MIN_SHARE
                       and not (1.0 / DIET_BAND <= ratio <= DIET_BAND))
                if off:
                    failures.append((pred, q, n + 1, e, k))
                cells.append(f"{e:6.3f} {ratio:5.2f}{'*' if off else ' '}")
            booked_txt = f"{k:>7.3f}" if q in book else f"{'-':>7}"
            print(f"  {q:>24} {booked_txt} " + " ".join(f"{c:>14}" for c in cells))
    print()
    print("cells: share ratio (* = OFF)")
    prey_failures = prey_side(record, diets, library, density, progress)
    if failures:
        print("OFF: " + "; ".join(f"{p} -> {q} year {n} {e:.3f} vs booked {k:.3f}"
                                  for p, q, n, e, k in failures))
    else:
        print("Every booked pair within the band in every probe year.")
    if prey_failures:
        print("Prey side OFF: " + "; ".join(f"{p} -> {q} year {n} M2 {r:.3g} vs booked {m:.3g}"
                                            for p, q, n, r, m in prey_failures))
    elif prey_failures is not None:
        print("Prey side: every pair that matters to its prey within the band in every year.")


if __name__ == '__main__':
    main()
