#!/usr/bin/env python3
"""Generate a standalone interactive HTML page from a run's biomass.jsonl.

Usage
-----
    python tools/biomass_html.py <run_dir> [-o <out.html>]

``<run_dir>`` is a directory inside ``results/`` (or anywhere) that
contains a ``biomass.jsonl`` produced by ``train.py``'s probe. The
output HTML uses Plotly via CDN, so it works offline only after the
first load (Plotly's JS is fetched once). It mirrors the live pygame
plot panel:

  * Tabs (sub-plots) for the metrics actually present in the file,
    matching the live pygame visualiser's tab strip exactly (train
    mode): ``reward`` (= log10_ratio), ``biomass`` (= ratio %),
    ``energy``, ``move``, ``rest``, ``eat``, ``predation``,
    ``starvation``. The loss tabs read the nested
    ``loss_breakdown[fid][<cause>]`` field and plot it as a percentage
    of total biomass loss per FG.
  * One trace per functional group (FG); legend entries are
    click-toggleable (same UX as the live plot's per-FG checkboxes).
  * X-axis = sample index across the file (each JSONL line is one
    probe rollout, so this corresponds 1:1 to live ``iter``).

The script has no third-party Python deps — Plotly is loaded purely
client-side, so this runs on the same vanilla Python the project
already uses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple


# Metrics we know how to plot, in the order they should appear as tabs.
# Order mirrors the live pygame visualiser's tab strip (train mode) 1:1:
#   reward (= log10_ratio), biomass (= ratio %), energy, move, rest,
#   eat, predation, starvation.
# The fourth tuple element points at the matching random-action
# baseline series inside the optional ``rnd`` sub-dict of each JSONL
# record (empty string if no baseline is available for that metric).
_METRICS: Tuple[Tuple[str, str, str, str], ...] = (
    # (tab key, top-level JSONL field, plot title,        rnd sub-field)
    # Mirrors the live visualizer tab order in train mode: reward,
    # biomass, energy, move, rest, eat, predation, starvation. ``bh``/``b0``
    # are omitted because they are not live-viz tabs. Loss tabs read nested
    # ``loss_breakdown[fid][<cause>]`` fractions and scale them to percent.
    # OBS: titlarna nedan är TRAIN-lägets titlar (probe-medelvärden per
    # ARS-step). I inference-läget överlagras de av ``_INFER_TITLES``
    # nedan (per-tick ögonblicksvärden). ``_build_html(..., mode=...)``
    # väljer set.
    # ``reward``-fliken: pekar nu på ``record["reward"]`` (den faktiska
    # ARS-rewarden per-FG som skrivs av ``_probe_biomass``). Fallback:
    # om ``reward`` saknas i alla records används ``log10_ratio`` som
    # bakåtkompatibilitet för gamla jsonl-filer. Titeln är generisk här
    # och byts dynamiskt i ``_build_html`` baserat på ``__meta__``-
    # headern (reward-formel) om den finns.
    ("reward",     "reward",                 "ARS reward per FG",                  ""),
    ("biomass",    "ratio",                  "bh / b0 per FG — biomass tab",       "ratio"),
    ("energy",     "energy_ratio",           "eh / e0 per FG — energy tab",        "energy_ratio"),
    ("move",       "move_frac",              "mean move-action % per DM",          "move_frac"),
    ("rest",       "rest_frac",              "mean rest-action % per DM",          "rest_frac"),
    ("eat",        "eat_frac",               "mean eat-action % per DM",           "eat_frac"),
    ("predation",  "loss_breakdown.predation",  "predation share of total loss (%) per FG",  "loss_breakdown.predation"),
    ("starvation", "loss_breakdown.starvation", "starvation share of total loss (%) per FG", "loss_breakdown.starvation"),
)


# Inference-lägets plot-titlar. Speglar ``_tab_labels`` i
# ``lib/viz/pygame_viz.py`` när ``mode="inference"``: värdena är per-tick
# ögonblick, inte medelvärden/kumulativa andelar. reward-fliken finns
# inte i inference (ingen reward-signal) men lämnas kvar för robusthet
# ifall någon record ändå har ``log10_ratio``.
_INFER_TITLES: Dict[str, str] = {
    "reward":     "log10(bh / b0) per FG — reward tab",
    "biomass":    "bh / b0 per FG — biomass tab (per-tick)",
    "energy":     "eh / e0 per FG — energy tab (per-tick)",
    "move":       "move-action % per DM (per-tick)",
    "rest":       "rest-action % per DM (per-tick)",
    "eat":        "eat-action % per DM (per-tick)",
    "predation":  "predation share of tick loss (%) per FG",
    "starvation": "starvation share of tick loss (%) per FG",
}


def _extract_field(record: dict, field: str, rnd: bool = False) -> dict:
    """Hämtar ett ``{fid: value}``-dict från en JSONL-record.

    Stöder två sorters fält:
      * Platt fält, t.ex. ``"ratio"`` → ``record["ratio"]`` (eller
        ``record["rnd"]["ratio"]`` om ``rnd=True``).
      * Nested loss-breakdown, t.ex. ``"loss_breakdown.predation"`` →
        bygger ``{fid: record["loss_breakdown"][fid]["predation"] * 100}``.
        Värdena skalas till procent (0..100) här, så att y-axeln matchar
        live visualizer plot tabs ``predation``/``starvation``.
    """
    root = (record.get("rnd") or {}) if rnd else record
    if "." not in field:
        d = root.get(field) or {}
        return d if isinstance(d, dict) else {}
    parent, child = field.split(".", 1)
    pd = root.get(parent) or {}
    if not isinstance(pd, dict):
        return {}
    out: Dict[str, float] = {}
    for fid, sub in pd.items():
        if isinstance(sub, dict) and isinstance(sub.get(child), (int, float)):
            out[fid] = float(sub[child]) * 100.0
    return out


def _load_jsonl(path: str) -> List[dict]:
    """Läs biomass.jsonl. ``__meta__``-rader (fresh-start-header med
    reward-formel-flaggor) filtreras bort ur den returnerade listan;
    använd ``_load_jsonl_with_meta`` om du behöver meta-headern."""
    records, _meta = _load_jsonl_with_meta(path)
    return records


def _load_jsonl_with_meta(path: str) -> Tuple[List[dict], dict]:
    """Läs biomass.jsonl och returnera ``(records, meta)``.

    ``meta`` är det senaste ``__meta__``-objektet som setts i filen
    (typiskt skrivet som första rad av ``train.py`` vid fresh start med
    reward-formel-flaggor: ``legacy_reward``, ``integral_reward``,
    ``alpha``/``beta``/``cappa``/``survival_bonus``). Saknas ``__meta__``
    (gamla jsonl-filer) returneras ``{}``.
    """
    records: List[dict] = []
    meta: dict = {}
    with open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARN: skipping malformed line {ln}: {e}",
                      file=sys.stderr)
                continue
            if isinstance(obj, dict) and "__meta__" in obj:
                m = obj.get("__meta__")
                if isinstance(m, dict):
                    meta = m
                continue
            records.append(obj)
    return records, meta


def _collect_keys(records: List[dict], field: str,
                  rnd_field: str = "") -> List[str]:
    """Stable ordering: insertion order across the whole file.

    Pulls keys from the top-level ``field`` dict on each record; if
    ``rnd_field`` is set, also unions in keys from each record's
    ``rnd.<rnd_field>`` sub-dict so the random-action baseline FGs are
    represented even on records where the main field is empty.
    """
    seen: Dict[str, None] = {}
    for r in records:
        d = _extract_field(r, field, rnd=False)
        for k in d.keys():
            if k not in seen:
                seen[k] = None
        if rnd_field:
            rd = _extract_field(r, rnd_field, rnd=True)
            for k in rd.keys():
                if k not in seen:
                    seen[k] = None
    return list(seen.keys())


def _fg_colour_rgb(idx: int) -> Tuple[int, int, int]:
    """Replicates ``lib/viz/pygame_viz.py::_fg_colour_for`` exactly:
    golden-ratio hue spacing, HSV(hue, 0.65, 0.95) → (R, G, B) ints in
    0..255. The live viz uses the same formula keyed off the FG's
    position in ``env.fg_ids``.
    """
    import colorsys
    hue = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _fg_colour_for(idx: int) -> str:
    """Plotly-compatible ``"rgb(r,g,b)"`` string for FG index ``idx``."""
    r, g, b = _fg_colour_rgb(idx)
    return f"rgb({r},{g},{b})"


def _fg_rnd_colour_for(idx: int) -> str:
    """Dimmed colour for the ``_rnd`` random-action baseline trace —
    matches live viz's per-channel ``max(0, c - 90)`` darkening of the
    base FG colour (see ``pygame_viz.py`` ~line 220).
    """
    r, g, b = _fg_colour_rgb(idx)
    return f"rgb({max(0, r - 90)},{max(0, g - 90)},{max(0, b - 90)})"


def _build_traces(records: List[dict], field: str, fg_ids: List[str],
                  rnd_field: str = "",
                  colour_index: Dict[str, int] | None = None) -> List[dict]:
    """One trace per FG, plus optionally one ``<fid>_rnd`` dashed trace
    per FG when the random-action baseline is present in the records.

    X = sample index (1..N), Y = metric value. Missing values become
    ``None`` so Plotly draws a gap rather than a wrong point.

    ``colour_index`` maps each FG-id to a stable global index used to
    pick the same per-FG colour as the live pygame visualiser. If not
    provided, falls back to this metric's local order.
    """
    n = len(records)
    xs = list(range(1, n + 1))
    traces: List[dict] = []
    if colour_index is None:
        colour_index = {fid: i for i, fid in enumerate(fg_ids)}

    # Main policy traces.
    for fid in fg_ids:
        col = _fg_colour_for(colour_index.get(fid, 0))
        ys: List[object] = []
        for r in records:
            d = _extract_field(r, field, rnd=False)
            v = d.get(fid, None)
            ys.append(v if isinstance(v, (int, float)) else None)
        traces.append({
            "x": xs, "y": ys, "name": fid,
            "type": "scatter", "mode": "lines",
            "showlegend": True,
            "legendgroup": fid,
            "line": {"width": 1.6, "color": col},
            "hovertemplate": (
                f"<b>{fid}</b><br>iter=%{{x}}<br>"
                f"{field}=%{{y}}<extra></extra>"),
        })

    # Random-action baseline traces (dashed) — only if any record
    # actually has values for this rnd sub-field, otherwise we skip the
    # whole block to keep the legend clean.
    if rnd_field:
        any_rnd = False
        for r in records:
            rd = _extract_field(r, rnd_field, rnd=True)
            if any(isinstance(v, (int, float)) for v in rd.values()):
                any_rnd = True
                break
        if any_rnd:
            for fid in fg_ids:
                col = _fg_rnd_colour_for(colour_index.get(fid, 0))
                ys = []
                for r in records:
                    rd = _extract_field(r, rnd_field, rnd=True)
                    v = rd.get(fid, None)
                    ys.append(v if isinstance(v, (int, float)) else None)
                traces.append({
                    "x": xs, "y": ys, "name": fid + "_rnd",
                    "type": "scatter", "mode": "lines",
                    "showlegend": True,
                    "legendgroup": fid + "_rnd",
                    "line": {"width": 1.2, "dash": "dash", "color": col},
                    "visible": "legendonly",
                    "hovertemplate": (
                        f"<b>{fid}_rnd</b><br>iter=%{{x}}<br>"
                        f"{field}=%{{y}}<extra></extra>"),
                })
    return traces


def _reward_title_from_meta(meta: dict) -> str:
    """Bygg en reward-flik-titel som beskriver den aktiva rewardformeln.

    ``meta`` kommer från ``__meta__``-headern i biomass.jsonl (skriven
    av ``train.py`` vid fresh start). Om headern saknas eller är tom
    faller vi tillbaka till en generisk titel.
    """
    if not isinstance(meta, dict) or not meta:
        return "ARS reward per FG"
    legacy = bool(meta.get("legacy_reward", False))
    integral = bool(meta.get("integral_reward", True))
    if legacy:
        # Legacy linjärkombination: α·Δlog b + β·survival − γ·loss.
        alpha = meta.get("alpha")
        beta = meta.get("beta")
        cappa = meta.get("cappa")
        parts = []
        if alpha is not None:
            parts.append(f"α={alpha}")
        if beta is not None:
            parts.append(f"β={beta}")
        if cappa is not None:
            parts.append(f"γ={cappa}")
        pstr = f" ({', '.join(parts)})" if parts else ""
        return ("ARS reward per FG — legacy "
                "(α·Δlog b + β·survival − γ·loss)" + pstr)
    # Ny total-energi-reward.
    kind = "integral" if integral else "final-value"
    return f"ARS reward per FG — total-energy ({kind})"


def _build_html(run_dir: str, records: List[dict],
                mode: str = "train",
                meta: dict | None = None) -> str:
    """Bygg standalone HTML-plot.

    ``mode`` styr flik-titlar, sidhuvud och x-axel-etikett:
      * ``"train"`` (default): probe-medelvärden per ARS-step, en
        record per rollout — samma semantik som ``biomass.jsonl`` från
        ``train.py``. Titlar från ``_METRICS`` (t.ex. "mean move-action
        % per DM", "predation share of total loss (%)"). X-axel =
        "sample index (one per probe rollout)".
      * ``"inference"``: per-tick ögonblick, en record per tick — samma
        semantik som live-vizen i inference-läget. Titlar från
        ``_INFER_TITLES`` (t.ex. "move-action % per DM (per-tick)",
        "predation share of tick loss (%)"). X-axel = "tick".
    """
    is_infer = str(mode).lower().startswith("infer")
    # Build a stable global FG-id -> colour index, shared across all
    # tabs so the same FG keeps the same colour everywhere AND matches
    # the live pygame visualiser (which keys colours off the position
    # of each FG in ``env.fg_ids``).
    #
    # ``env.fg_ids`` insertion order is faithfully preserved in the
    # ``b0`` field of every record (``_probe_biomass`` iterates
    # ``env.fgs`` to build it), so we use ``b0`` as the canonical
    # order. We fall back to ``bh`` then ``ratio`` for safety, and
    # finally union in any extras from other metrics / rnd baselines
    # so unknown FGs still get a stable colour.
    global_order: Dict[str, None] = {}
    for canonical in ("b0", "bh", "ratio"):
        for r in records:
            d = r.get(canonical) or {}
            for k in d.keys():
                if k not in global_order:
                    global_order[k] = None
        if global_order:
            break
    # Union in any stragglers (extras, rnd-only ids, etc.) so they
    # still get a deterministic colour slot.
    for _tab_key, _field, _label, _rnd in _METRICS:
        for fid in _collect_keys(records, _field, rnd_field=_rnd):
            if fid not in global_order:
                global_order[fid] = None
    colour_index: Dict[str, int] = {fid: i for i, fid in enumerate(global_order)}

    # Which metrics actually exist in this file? Skip empty ones.
    metric_blocks: List[Tuple[str, str, List[str], List[dict]]] = []
    for tab_key, field, label, rnd_field in _METRICS:
        eff_field = field
        # ``reward``-fliken: primärt fält ``reward`` (den faktiska ARS-
        # rewarden som skrivs av ``_probe_biomass`` i nya train.py).
        # Om ingen record har det (gamla jsonl-filer) faller vi
        # tillbaka till ``log10_ratio`` så gamla filer fortsätter
        # plotta något meningsfullt under reward-fliken.
        if tab_key == "reward":
            has_reward = any(
                isinstance(r.get("reward"), dict) and r.get("reward")
                for r in records
            )
            if not has_reward:
                eff_field = "log10_ratio"
        fg_ids = _collect_keys(records, eff_field, rnd_field=rnd_field)
        if not fg_ids:
            continue
        traces = _build_traces(records, eff_field, fg_ids, rnd_field=rnd_field,
                               colour_index=colour_index)
        # Titel: reward-fliken får dynamisk titel från ``meta`` (vilken
        # rewardformel som var aktiv). Övriga flikar behåller sina
        # _METRICS/_INFER_TITLES-defaulttitlar.
        if tab_key == "reward" and not is_infer:
            effective_label = _reward_title_from_meta(meta or {})
            if eff_field == "log10_ratio":
                # Visa att vi fallade tillbaka pga saknad meta/reward-data.
                effective_label += " — log10(bh/b0) fallback"
        else:
            effective_label = (_INFER_TITLES.get(tab_key, label)
                               if is_infer else label)
        metric_blocks.append((tab_key, effective_label, fg_ids, traces))

    if not metric_blocks:
        raise SystemExit("No known metric fields found in biomass.jsonl "
                         "(expected one of: ratio, log10_ratio, "
                         "energy_ratio, move_frac, rest_frac, eat_frac, "
                         "loss_breakdown).")

    # Summary line for the page header.
    n_records = len(records)
    gens = sorted({r.get("gen") for r in records if r.get("gen") is not None})
    iters = sorted({r.get("iter") for r in records if r.get("iter") is not None})
    gen_range = f"{gens[0]}–{gens[-1]}" if gens else "?"
    iter_range = f"{iters[0]}–{iters[-1]}" if iters else "?"
    run_name = os.path.basename(os.path.abspath(run_dir.rstrip("/")))

    # Serialise data + metadata for the page.
    payload = {
        "run_name": run_name,
        "n_records": n_records,
        "gen_range": gen_range,
        "iter_range": iter_range,
        "metrics": [
            {"key": k, "label": l, "fg_ids": fg, "traces": tr}
            for (k, l, fg, tr) in metric_blocks
        ],
    }
    # ``ensure_ascii=False`` gör att unicode-tecken (α, β, γ, Δ, · osv.
    # i reward-flikens titel) hamnar som läsbara UTF-8-tecken i den
    # genererade HTML:en istället för ``\uXXXX``-escapes. Sidan
    # deklarerar ``<meta charset="utf-8">`` så det renderas korrekt.
    payload_json = json.dumps(payload, ensure_ascii=False)

    # Mode-beroende texter i sidhuvudet. Train-läget pratar om
    # "records" (en per probe-rollout) och "gen/iter"; inference-läget
    # har en record per tick så vi visar "ticks" och döljer gen-raden.
    page_title = (f"Inference plots — {run_name}" if is_infer
                  else f"Biomass training plots — {run_name}")
    header_h1 = page_title
    if is_infer:
        header_meta = (f"{n_records} ticks · tick range {iter_range} · "
                       "per-tick snapshots (no averaging) · click a legend "
                       "entry to toggle that FG on/off · double-click to "
                       "isolate it")
        xaxis_title = "tick"
        footer_src = (f"Generated from live inference viz buffer "
                      f"(run dir: {run_dir}).")
    else:
        header_meta = (f"{n_records} records · gen {gen_range} · iter "
                       f"{iter_range} · click a legend entry to toggle that "
                       "FG on/off · double-click to isolate it (hide all "
                       "others) — same as the live visualiser's per-FG "
                       "checkboxes")
        xaxis_title = "sample index (one per probe rollout)"
        footer_src = (f"Generated from "
                      f"<code>{os.path.join(run_dir, 'biomass.jsonl')}</code>.")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{page_title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    background: #1a1a22;
    color: #e6e6ef;
  }}
  header {{
    padding: 12px 20px;
    background: #24242e;
    border-bottom: 1px solid #3a3a48;
  }}
  header h1 {{ margin: 0 0 4px 0; font-size: 18px; font-weight: 600; }}
  header .meta {{ font-size: 12px; color: #a0a0b0; }}
  #tabs {{
    display: flex;
    gap: 2px;
    padding: 8px 20px 0 20px;
    background: #1a1a22;
    border-bottom: 1px solid #3a3a48;
  }}
  .tab {{
    padding: 6px 14px;
    background: #2a2a36;
    color: #b0b0c0;
    border: 1px solid #3a3a48;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    cursor: pointer;
    font-size: 13px;
    user-select: none;
  }}
  .tab:hover {{ background: #34344a; color: #e6e6ef; }}
  .tab.active {{ background: #3a3a55; color: #ffffff; font-weight: 600; }}
  #plot {{ width: 100%; height: calc(100vh - 110px); }}
  .footer {{
    padding: 6px 20px;
    font-size: 11px;
    color: #707080;
    border-top: 1px solid #3a3a48;
  }}
</style>
</head>
<body>
<header>
  <h1>{header_h1}</h1>
  <div class="meta">{header_meta}</div>
</header>
<div id="tabs"></div>
<div id="plot"></div>
<div class="footer">
  {footer_src}
  Plotly via CDN; works offline after first load.
</div>

<script>
const DATA = {payload_json};

const tabsEl = document.getElementById("tabs");
const plotEl = document.getElementById("plot");
let activeIdx = 0;

function layoutFor(metric) {{
  return {{
    paper_bgcolor: "#1a1a22",
    plot_bgcolor: "#24242e",
    font: {{ color: "#e6e6ef" }},
    margin: {{ l: 60, r: 20, t: 40, b: 50 }},
    title: {{ text: metric.label, font: {{ size: 14 }} }},
    xaxis: {{
      title: {json.dumps(xaxis_title)},
      gridcolor: "#3a3a48",
      zerolinecolor: "#3a3a48",
    }},
    yaxis: {{
      title: metric.key,
      gridcolor: "#3a3a48",
      zerolinecolor: "#5a5a6a",
    }},
    showlegend: true,
    legend: {{
      bgcolor: "rgba(0,0,0,0)",
      bordercolor: "#3a3a48",
      borderwidth: 1,
      // Click a legend entry to toggle that FG on/off; double-click
      // to isolate it (hide all others). Matches the live viz's
      // per-FG checkbox behaviour.
      itemclick: "toggle",
      itemdoubleclick: "toggleothers",
    }},
    hovermode: "closest",
  }};
}}

function render(idx) {{
  activeIdx = idx;
  const metric = DATA.metrics[idx];
  Plotly.react(plotEl, metric.traces, layoutFor(metric),
               {{ responsive: true, displaylogo: false }});
  // Refresh tab styling.
  [...tabsEl.children].forEach((el, i) => {{
    el.classList.toggle("active", i === idx);
  }});
}}

DATA.metrics.forEach((m, i) => {{
  const btn = document.createElement("div");
  btn.className = "tab" + (i === 0 ? " active" : "");
  btn.textContent = m.key;
  btn.title = m.label;
  btn.addEventListener("click", () => render(i));
  tabsEl.appendChild(btn);
}});

render(0);
window.addEventListener("resize", () => Plotly.Plots.resize(plotEl));
</script>
</body>
</html>
"""
    return html


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description=("Build an interactive HTML page (Plotly) from a "
                     "training run's biomass.jsonl."))
    p.add_argument("run_dir", help="Path to the run directory containing "
                                   "biomass.jsonl")
    p.add_argument("-o", "--output", default=None,
                   help="Output HTML path (default: <run_dir>/plots.html)")
    args = p.parse_args(argv)

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
        return 2
    jsonl = os.path.join(run_dir, "biomass.jsonl")
    if not os.path.isfile(jsonl):
        print(f"ERROR: missing biomass.jsonl in {run_dir}", file=sys.stderr)
        return 2

    records, meta = _load_jsonl_with_meta(jsonl)
    if not records:
        print(f"ERROR: no records in {jsonl}", file=sys.stderr)
        return 1

    html = _build_html(run_dir, records, meta=meta)
    out = args.output or os.path.join(run_dir, "plots.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote {out} ({len(records)} records, "
          f"{len(html)/1024:.1f} KiB).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
