"""The final per-pair table (rule A + band overlap, every member's f_m), written as a YAML edit block, applied to
temporary library copies, run through the diet gate (tools/probes/budget_gate.py, predator and prey side,
equal cells and the field spread, per bed class) and the loader. Nothing in the repository's library, code,
tests or manifest changes; library copies live in <scratch_dir>.

Usage: python3 final_gate.py <repo_root> <benth_profile.json> <scratch_dir> <tables_dir>

Inputs: research_notes/"Overlapp i fiskparen"/tables/band_pairs.json (A_band central: the 11 non-benthivore
pairs, as proposal_table.py prints them), <benth_profile.json> (this note: the benthivore pairs),
tables/wc_benth_window.json (this note: the water column -> benthivores window).
The gate machinery is research_notes/"En synregel for fiskparen"/scripts/gate_variants.py exec'd up to its run
list (run_gate: budget_gate.py exec'd from its own file with library_path=<copy>; per-class gate).
"""
import contextlib
import copy
import io
import json
import os
import re
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
BPJ = json.load(open(sys.argv[2]))
SCR = os.path.abspath(sys.argv[3])
TAB = os.path.abspath(sys.argv[4])
RN = os.path.join(ROOT, "research_notes")
os.makedirs(SCR, exist_ok=True)
os.makedirs(os.path.join(TAB, "gates"), exist_ok=True)
BAND = json.load(open(os.path.join(RN, "Overlapp i fiskparen", "tables", "band_pairs.json")))["variants"]
WIN = json.load(open(os.path.join(TAB, "wc_benth_window.json")))
GV = os.path.join(RN, "En synregel for fiskparen", "scripts", "gate_variants.py")
src = open(GV).read()
head = src[:src.index("RUNS = [")]
ARGV = list(sys.argv)
sys.argv = [GV, ROOT, os.path.join(RN, "Overlapp i fiskparen", "tables", "band_pairs.json"), SCR, TAB]
NS = {"__name__": "gate_variants_head", "__file__": GV}
os.chdir(ROOT)
sys.path.insert(0, ROOT)
exec(compile(head, GV, "exec"), NS)
sys.argv = ARGV
LIB0 = NS["LIB0"]
INTER = LIB0["interaction_definitions"]
DEPTHS = [(99.9, 0), (100.0, 1), (149.9, 1), (150.0, 2), (249.9, 2), (250.0, 3)]


def g3(x):
    return float(f"{x:.3g}")


def table(cls):
    return [{"depth_m": d, "factor": g3(max(cls[k], 0.0))} for d, k in DEPTHS]


STOMACH = {"planktivorous_fish_preys_on_krill": 0.57, "mackerel_preys_on_krill": 0.42,
           "mackerel_preys_on_planktivorous_fish": 0.16, "whiting_preys_on_krill": None}
REMOVE = object()


def overlap_rows():
    """the 11 non-benthivore pairs of the Overlapp i fiskparen proposal (A_band central)."""
    out = {}
    A = BAND["A_band"]
    for g, r in A["krill"].items():
        if g == "benthivorous_fish":
            continue
        key = f"{g}_preys_on_krill"
        e = dict(max_intake_rate=g3(r["a_new"]), depth_access_table=table(r["classes_lib_units"]))
        e["dark_ratio"] = STOMACH[key] if key in STOMACH else g3(r["nd"])
        if e["dark_ratio"] is None:
            e["dark_ratio"] = REMOVE
        out[key] = e
    for name, r in A["fish"].items():
        if name.startswith("benthivorous_fish"):
            continue
        key = name.replace("->", "_preys_on_")
        e = dict(max_intake_rate=g3(r["a_new"]))
        if key != "mackerel_preys_on_planktivorous_fish":           # derived shape flat +-1 % [J]
            e["depth_access_table"] = table([r["a_new"] * c for c in r["cls_rel"]])
        e["dark_ratio"] = STOMACH.get(key, g3(r["nd"]))
        out[key] = e
    return out


def benth_rows(var, benthic=True):
    R = BPJ["variants"][var]
    kr = R["krill"]
    out = {"benthivorous_fish_preys_on_krill": dict(max_intake_rate=g3(kr["a_new"]),
                                                     depth_access_table=table(kr["classes_lib_units"]),
                                                     dark_ratio=g3(kr["nd"]))}
    for prey in ("planktivorous_fish", "whiting"):
        r = R[f"benthivorous_fish->{prey}"]
        out[f"benthivorous_fish_preys_on_{prey}"] = dict(max_intake_rate=g3(r["a_new"]),
                                                          depth_access_table=table(r["classes"]),
                                                          dark_ratio=g3(r["nd"]))
    if benthic:
        for prey in ("benthic_infauna", "mobile_epifauna", "norway_lobster"):
            out[f"benthivorous_fish_preys_on_{prey}"] = dict(max_intake_rate=g3(R[f"benthivorous_fish->{prey}"]["a_new"]))
    return out


def overlap_benth_rows():
    """the benthivore rows as the Overlapp i fiskparen proposal had them (vision only), for the reproduction."""
    A = BAND["A_band"]
    r = A["krill"]["benthivorous_fish"]
    out = {"benthivorous_fish_preys_on_krill": dict(max_intake_rate=g3(r["a_new"]), depth_access_table=table(r["classes_lib_units"]))}
    for prey in ("planktivorous_fish", "whiting"):
        rr = A["fish"][f"benthivorous_fish->{prey}"]
        out[f"benthivorous_fish_preys_on_{prey}"] = dict(max_intake_rate=g3(rr["a_new"]),
                                                          depth_access_table=table([rr["a_new"] * c for c in rr["cls_rel"]]))
    return out


W_CENTRAL = [g3(x) for x in WIN["A_band|ling anglerfish|k+1.7"]]
WC_WIN = {"predatory_fish_preys_on_benthivorous_fish": dict(stage_access=W_CENTRAL)}


def build(edits):
    lib = copy.deepcopy(LIB0)
    inter = lib["interaction_definitions"]
    for key, e in edits.items():
        for k, v in e.items():
            if v is REMOVE:
                inter[key].pop(k, None)
            else:
                inter[key][k] = v
    return lib


def merge(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out.setdefault(k, {}).update(v)
    return out


FINAL = merge(overlap_rows(), benth_rows("central"), WC_WIN)
VARS = {
    "repro_overlap_A_band_T": merge(overlap_rows(), overlap_benth_rows()),
    "final": FINAL,
    "final_nobenthic": merge(overlap_rows(), benth_rows("central", benthic=False), WC_WIN),
    "final_win243": merge(overlap_rows(), benth_rows("central")),
    "final_deep": merge(overlap_rows(), benth_rows("deep"), WC_WIN),
    "final_bedbound": merge(overlap_rows(), benth_rows("bedbound"), WC_WIN),
    "final_rest50": merge(overlap_rows(), benth_rows("rest50"), WC_WIN),
    "final_deep_rest50": merge(overlap_rows(), benth_rows("deep_rest50"), WC_WIN),
    "final_sw_contact": merge(overlap_rows(), benth_rows("sw_contact"), WC_WIN),
    "final_pout_albert": merge(overlap_rows(), benth_rows("pout_albert"), WC_WIN),
    "final_win_lo": merge(overlap_rows(), benth_rows("central"),
                          {"predatory_fish_preys_on_benthivorous_fish": dict(stage_access=[g3(x) for x in WIN["A_band_Hs5|ling excluded|k+1.5"]])}),
    "final_win_hi": merge(overlap_rows(), benth_rows("central"),
                          {"predatory_fish_preys_on_benthivorous_fish": dict(stage_access=[g3(x) for x in WIN["A_band_Hs40|ling anglerfish|k+2.0"]])}),
}

# ------------------------------------------------------------------ YAML edit block
COMMENT = {
    "predatory_fish_preys_on_krill": "band (saithe exp 15 m) + rule A: R_sat on 25 mm krill |C0| 0.4 0.57 m vs 0.87 (x0.45); n/d rule",
    "whiting_preys_on_krill": "band 10 m + rule A: x1.00 (2 L cap binds); no dark_ratio (stomachs: no diel difference, Onsrud et al. 2004)",
    "planktivorous_fish_preys_on_krill": "band + rule A: x1.00 (2 L cap binds); dark_ratio 0.57 from the stomachs (245)",
    "mackerel_preys_on_krill": "band + rule A: x1.00; dark_ratio 0.42 from the stomachs (245)",
    "benthivorous_fish_preys_on_krill": "member search profiles (this note) + rule A: x0.52; LRD bed-bound, haddock >= 35 cm rise by day; n/d integral",
    "predatory_fish_preys_on_planktivorous_fish": "band x0.49 x rule A x5.93 = x2.90 (saithe 0.92 of a); n/d rule",
    "predatory_fish_preys_on_benthivorous_fish": "band x1.45 x rule A x5.24 = x7.60 (saithe 0.89, hake 0.05; 243 time factors replaced by the profile); window aw on the band a's member shares, a(L) exponent from the band a (kexp +1.7)",
    "predatory_fish_preys_on_whiting": "band x0.46 x rule A x6.02 = x2.76; n/d rule",
    "predatory_fish_preys_on_mackerel": "band x0.13 x rule A x1.19 = x0.155 (spurdog summer surface 0.76 of a); OFF in every reading",
    "whiting_preys_on_planktivorous_fish": "band (whiting 10 m) x0.75 x rule A x2.06 = x1.55; n/d rule",
    "whiting_preys_on_benthivorous_fish": "band x0.20 x rule A x2.51 = x0.49; FLAGGED OPEN until whiting's height above the bed is measured",
    "mackerel_preys_on_planktivorous_fish": "band x0.61 x rule A x1.53 = x0.93; no table (derived shape flat +-1 %); dark_ratio 0.16 from the stomachs (228)",
    "benthivorous_fish_preys_on_planktivorous_fish": "rule A at the bed x1.25 x member search profiles x0.87 = x1.08 (cod 0.68 / haddock 0.32; pout 0.85 of a); n/d integral",
    "benthivorous_fish_preys_on_whiting": "rule A at the bed x1.19 x member search profiles x0.90 = x1.07 (cod 0.94 of a); n/d integral",
    "benthivorous_fish_preys_on_benthic_infauna": "x tau_bed 0.71: members' search-time share in bed contact (0-2 m; flatfish and LRD 1, cod 0.70, haddock 0.56 by size)",
    "benthivorous_fish_preys_on_mobile_epifauna": "x tau_bed 0.74 (epifauna member weights a_enc x e)",
    "benthivorous_fish_preys_on_norway_lobster": "x tau_bed 0.70 (cod; with haddock / LRD weights 0.69)",
}
ORDER = ["predatory_fish_preys_on_krill", "whiting_preys_on_krill", "planktivorous_fish_preys_on_krill",
         "benthivorous_fish_preys_on_krill", "mackerel_preys_on_krill", "predatory_fish_preys_on_planktivorous_fish",
         "predatory_fish_preys_on_benthivorous_fish", "predatory_fish_preys_on_whiting", "predatory_fish_preys_on_mackerel",
         "whiting_preys_on_planktivorous_fish", "whiting_preys_on_benthivorous_fish", "mackerel_preys_on_planktivorous_fish",
         "benthivorous_fish_preys_on_planktivorous_fish", "benthivorous_fish_preys_on_whiting",
         "benthivorous_fish_preys_on_benthic_infauna", "benthivorous_fish_preys_on_mobile_epifauna",
         "benthivorous_fish_preys_on_norway_lobster"]
lines = ["# YAML edits for fgconfig/fg_library.yaml interaction_definitions (NOT applied).",
         "# Rule A + band overlap with every member's search profile f_m(z); one section, every pair at once.",
         "# Per pair: the keys to set (others unchanged). 'dark_ratio: REMOVE' = delete the key.",
         "# light_threshold_deg unchanged on every pair (krill pairs 0 / mackerel -5; fish pairs -5; benthivore",
         "# fish pairs 0). depth_access_table rows as the krill pairs (loader renormalises to an equal-cell mean 1).",
         "# The three benthic rows are recommendation [J] (section 'Benthic pairs'); final_nobenthic leaves them out.",
         ""]
for key in ORDER:
    e = FINAL[key]
    old = INTER[key]
    lines.append(f"  {key}:")
    lines.append(f"    # {COMMENT[key]}")
    lines.append(f"    # was: max_intake_rate {old.get('max_intake_rate')}, dark_ratio {old.get('dark_ratio')}, "
                 f"light_threshold_deg {old.get('light_threshold_deg')}, stage_access {old.get('stage_access')}, "
                 f"table {'yes' if old.get('depth_access_table') else 'no'}")
    lines.append(f"    max_intake_rate: {e['max_intake_rate']}")
    if "dark_ratio" in e:
        if e["dark_ratio"] is REMOVE:
            lines.append("    # dark_ratio: none today, none proposed" if "dark_ratio" not in old else "    dark_ratio: REMOVE")
        else:
            lines.append(f"    dark_ratio: {e['dark_ratio']}")
    if "stage_access" in e:
        lines.append(f"    stage_access: [{e['stage_access'][0]}, {e['stage_access'][1]}]")
    if "depth_access_table" in e:
        lines.append("    depth_access_table:")
        for row in e["depth_access_table"]:
            lines.append(f"      - {{depth_m: {row['depth_m']}, factor: {row['factor']}}}")
    lines.append("")
open(os.path.join(TAB, "yaml_edits.txt"), "w").write("\n".join(lines))
print("\n".join(lines))

# ------------------------------------------------------------------ gate runs
PREY_SIDE = re.compile(r"^\s+(\S+) booked\s+(\S+) \((\S+)\)\s+implied\s+(\S+) ratio\s+(\S+)\s+(\S+)(?:\s+field\s+(\S+) \(\s*(\S+)\))?")
SUMMARY = {}
for label, edits in VARS.items():
    lib = build(edits)
    lp = os.path.join(SCR, f"fg_library_{label}.yaml")
    with open(lp, "w") as fh:
        yaml.safe_dump(lib, fh, sort_keys=False)
    txt, out = NS["run_gate"](lp, {})
    with open(os.path.join(TAB, "gates", f"budget_gate_{label}.txt"), "w") as fh:
        fh.write(f"# budget_gate.py on a temporary library copy, variant {label}; edits: "
                 + "; ".join(f"{k} " + ",".join(f"{kk}={'REMOVE' if vv is REMOVE else (vv if kk != 'depth_access_table' else 'table')}"
                                                for kk, vv in v.items()) for k, v in edits.items()) + "\n" + txt)
    prey = {}
    cur = None
    for ln in txt.splitlines():
        m0 = re.match(r"^\s+(\w+): booked M2 (\S+) /yr, implied (\S+) \(ratio (\S+)\)", ln)
        if m0:
            cur = m0.group(1)
            prey[cur] = dict(total=float(m0.group(4)), pairs={})
            continue
        m = PREY_SIDE.match(ln)
        if m and cur:
            prey[cur]["pairs"][m.group(1)] = dict(booked=float(m.group(2)), ratio=float(m.group(5)), verdict=m.group(6),
                                                  field=float(m.group(8)) if m.group(8) else None)
    off = [ln for ln in txt.splitlines() if ln.startswith("Diet gate") and "OFF" in ln]
    SUMMARY[label] = dict(equal=out["equal"], field=out["field"], cls=out["class"], prey=prey, off=off)
    print(f"{label}: done", flush=True)
json.dump(SUMMARY, open(os.path.join(TAB, "final_gate.json"), "w"), indent=1, default=float)


# ------------------------------------------------------------------ tables
def eq_ratio(S, pred, prey):
    book = None
    t = S["equal"].get(pred)
    return t


GT = {}
for ln in open(os.path.join(TAB, "gates", "budget_gate_final.txt")):
    pass
print("\n" + "=" * 112)
print("Predator side, encounter / booked share (equal cells; field spread in brackets for WC and whiting); * = OFF")
print("=" * 112)


def shares_from_txt(label):
    """(pred -> prey -> (enc, booked, ratio, verdict)) from the gate text, equal cells and field."""
    txt = open(os.path.join(TAB, "gates", f"budget_gate_{label}.txt")).read()
    eq, fld = {}, {}
    part = txt.split("=== diet gate: encounter-implied")[1].split("=== diet gate at the field spread")[0]
    cur = None
    for ln in part.splitlines():
        p = ln.split()
        if len(p) >= 3 and p[1] == "prey":
            cur = p[0]
            eq[cur] = {}
            continue
        if cur and len(p) >= 8:
            try:
                eq[cur][p[0]] = (float(p[3]), float(p[4]), float(p[5]), p[-1])
            except ValueError:
                pass
    part = txt.split("=== diet gate at the field spread")[1].split("=== prey-side check")[0]
    cur = None
    for ln in part.splitlines():
        m = re.match(r"^\s*(\w+) \(\w+, GOV", ln)
        if m:
            cur = m.group(1)
            fld[cur] = {}
            continue
        m = re.match(r"^\s+(\w+) enc\s+(\S+) booked\s+(\S+) ratio\s+(\S+)\s+(\S+)", ln)
        if m and cur:
            fld[cur][m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5))
    return eq, fld


ROWS = {"predatory_fish": ["krill", "planktivorous_fish", "benthivorous_fish", "whiting", "mackerel", "zooplankton"],
        "whiting": ["krill", "benthivorous_fish", "mobile_epifauna", "planktivorous_fish", "zooplankton"],
        "benthivorous_fish": ["benthic_infauna", "mobile_epifauna", "norway_lobster", "krill", "planktivorous_fish", "whiting"],
        "mackerel": ["krill", "planktivorous_fish", "zooplankton"], "planktivorous_fish": ["krill", "zooplankton"]}
for label in VARS:
    eq, fld = shares_from_txt(label)
    print(f"  {label}:")
    for pred, preys in ROWS.items():
        cells = []
        for p in preys:
            if p not in eq.get(pred, {}):
                continue
            e, b, r, v = eq[pred][p]
            s = f"{p[:10]} {e:.3f}/{b:.3f}={r:.2f}{'*' if v == 'OFF' else ''}"
            if pred in fld and p in fld[pred]:
                fe, fb, fr, fv = fld[pred][p]
                s += f" ({fr:.2f}{'*' if fv == 'OFF' else ''})"
            cells.append(s)
        print(f"    {pred:18s} " + "; ".join(cells))
    cl = SUMMARY[label]["cls"].get("predatory_fish")
    if cl:
        print(f"    WC krill encounter share per bed class (< 100 / 100-150 / 150-250 / >= 250 m): "
              + " / ".join(f"{c['krill']['enc']:.2f}" for c in cl) + f" (booked {cl[2]['krill']['book']:.2f})")
    clb = SUMMARY[label]["cls"].get("benthivorous_fish")
    if clb:
        print("    benthivores per bed class, encounter shares infauna / epifauna / Nephrops / krill / plk / whiting: "
              + " | ".join("/".join(f"{c[p]['enc']:.3f}" for p in ("benthic_infauna", "mobile_epifauna", "norway_lobster",
                                                                   "krill", "planktivorous_fish", "whiting") if p in c)
                           for c in clb))
    print("    prey side (implied / booked M2, equal cells; field): "
          + "; ".join(f"{k} {v['total']:.2f} [" + ", ".join(f"{pp[:5]} {d['ratio']:.2f}{'*' if d['verdict'] == 'OFF' else ''}"
                                                            + (f" ({d['field']:.2f})" if d['field'] is not None else "")
                                                            for pp, d in v["pairs"].items()) + "]"
                      for k, v in SUMMARY[label]["prey"].items()
                      if k in ("krill", "planktivorous_fish", "benthivorous_fish", "whiting", "mackerel",
                               "benthic_infauna", "mobile_epifauna", "norway_lobster")))
    for o in SUMMARY[label]["off"]:
        print(f"    {o}")

# ------------------------------------------------------------------ loader on the final library
print("\n" + "=" * 112)
print("Loader: lib.config.config_loader.load_project_config(mareld2.yaml, library_path=<final copy>)")
print("=" * 112)
import warnings  # noqa: E402

from lib.config.config_loader import load_project_config  # noqa: E402

for label in ("final", "final_nobenthic"):
    lp = os.path.join(SCR, f"fg_library_{label}.yaml")
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf):
        warnings.simplefilter("always")
        res_new = load_project_config(os.path.join(ROOT, "mareld2.yaml"), library_path=lp, seed=1)
        res_old = load_project_config(os.path.join(ROOT, "mareld2.yaml"),
                                      library_path=os.path.join(ROOT, "fgconfig", "fg_library.yaml"), seed=1)
    fgs_new = res_new[0] if isinstance(res_new, tuple) else res_new
    fgs_old = res_old[0] if isinstance(res_old, tuple) else res_old

    def by_id(fgs):
        if isinstance(fgs, dict):
            return fgs
        return {getattr(f, "id", getattr(f, "name", str(i))): f for i, f in enumerate(fgs)}
    nn, oo = by_id(fgs_new), by_id(fgs_old)
    print(f"  {label}: loaded {len(nn)} groups; warnings {len(w)}: " + "; ".join(str(x.message)[:120] for x in w[:5]))
    for k in nn:
        gn = getattr(nn[k], "params", {}).get("growth_rate", None) if hasattr(nn[k], "params") else None
        go = getattr(oo.get(k), "params", {}).get("growth_rate", None) if k in oo and hasattr(oo[k], "params") else None
        if gn is None:
            gn = getattr(nn[k], "growth_rate", None)
            go = getattr(oo.get(k), "growth_rate", None)
        print(f"    {str(k):26s} growth_rate {gn} (live library {go}){'  CHANGED' if gn != go else ''}")
    for ln in buf.getvalue().splitlines():
        if re.search(r"warn|refus|error|stab", ln, re.I):
            print("    loader: " + ln[:160])
