"""budget_gate (tools/probes/budget_gate.py, unchanged source, library_path=<temporary copy>) for the saithe-profile
and juvenile-layer variants of band_hs.py, with the combined capture-p x schooling factors of
research_notes/"Fangst och stim i fiskparen" (central reading) on top. Nothing in the repository is written;
the copies live in <scratch_dir>.

Usage: python3 gate_hs.py <repo_root> <band_hs.json> <window_hs.json> <scratch_dir> <out_dir>
       (imports lib/ through budget_gate.py, so not -I)

Per variant V (ratio method against this note's reproduction of the section 269 central, exp 15 m):
  WC fish pairs (-> planktivores, benthivores, whiting, mackerel): max_intake_rate = library a x eq_V / eq_c,
      depth_access_table class k = library table_k x classes_V,k / classes_c,k;
  WC -> krill: the same with the krill eq and classes;
  WC -> benthivores stage_access = window_hs.json [V|ident] (ident 'gad' = library identity, 'had' = haddock only);
  then x the combined central factor on the nine fish-prey pairs (Hs-independent: p and schooling).
The 'library' run must reproduce today's gate and 'combined_central' the combined note's central.
No value is chosen from the gate.
"""
import copy
import json
import os
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
BH = json.load(open(sys.argv[2]))
WH = json.load(open(sys.argv[3]))
SCR = os.path.abspath(sys.argv[4])
OUTD = os.path.abspath(sys.argv[5])
os.makedirs(SCR, exist_ok=True)
os.makedirs(OUTD, exist_ok=True)
RN = os.path.join(ROOT, "research_notes")
COMB = json.load(open(os.path.join(RN, "Fangst och stim i fiskparen", "tables", "gates", "combined.json")))
FAC = COMB["factors"]["central"]
GATE_PY = os.path.join(RN, "Fangstsannolikhet for fiskbyten", "scripts", "gate_p.py")
gsrc = open(GATE_PY).read()
BG = os.path.join(ROOT, "tools", "probes", "budget_gate.py")
GNS = {"__name__": "gate_head", "ROOT": ROOT, "os": os, "sys": sys, "BG": BG, "BG_SRC": open(BG).read(),
       "OLD": "apply_natural_mortality=True, migration=True)"}
assert GNS["BG_SRC"].count(GNS["OLD"]) == 1
exec(compile("import contextlib, io, re\n" + gsrc[gsrc.index("def run_gate"):gsrc.index("summary = {}")], GATE_PY, "exec"), GNS)
run_gate, parse = GNS["run_gate"], GNS["parse"]

LIB0 = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))
WC_FISH = ["predatory_fish->planktivorous_fish", "predatory_fish->benthivorous_fish", "predatory_fish->whiting",
           "predatory_fish->mackerel"]
C = BH["central_repro"]


def key(name):
    p, q = name.split("->")
    return f"{p}_preys_on_{q}"


def g4(x):
    return float(f"{x:.4g}")


def build(var, ident="gad", combined=True, profile=True):
    lib = copy.deepcopy(LIB0)
    inter = lib["interaction_definitions"]
    if profile:
        V = BH[var]
        base = BH["noclip"] if var.startswith("noclip_") else C
        rows = [("predatory_fish->krill", V["krill"]["eq"] / base["krill"]["eq"], V["krill"]["classes"], base["krill"]["classes"])]
        rows += [(n, V["fish"][n]["eq"] / base["fish"][n]["eq"], V["fish"][n]["classes"], base["fish"][n]["classes"]) for n in WC_FISH]
        for name, r, cls, cls0 in rows:
            e = inter[key(name)]
            e["max_intake_rate"] = g4(e["max_intake_rate"] * r)
            if e.get("depth_access_table"):
                idx = [0, 1, 1, 2, 2, 3]
                e["depth_access_table"] = [dict(depth_m=row["depth_m"], factor=g4(row["factor"] * cls[i] / cls0[i]))
                                           for row, i in zip(e["depth_access_table"], idx)]
        w = WH[f"{var}|{ident}"]
        inter["predatory_fish_preys_on_benthivorous_fish"]["stage_access"] = [float(f"{w[0]:.3g}"), float(f"{w[1]:.3g}")]
    if combined:
        for name, f in FAC.items():
            e = inter[key(name)]
            e["max_intake_rate"] = g4(e["max_intake_rate"] * f)
    return lib


RUNS = [("library", None, "gad", False, False), ("combined_central", None, "gad", True, False),
        ("repro_central", "central_repro", "gad", True, True)]
for v in ("exp_Hs7", "exp_Hs10", "exp_Hs20", "exp_Hs30", "exp_Hs40", "gamma2_s6", "gamma2_s10", "uniform_35",
          "juv_H0.5", "exp_Hs10_juvH0.5", "gamma2_s6_juvH0.5", "noclip_juv_night_aglen"):
    RUNS.append((v, v, "gad", True, True))
for v in ("central_repro", "juv_H0.5", "exp_Hs10_juvH0.5", "gamma2_s6_juvH0.5"):
    RUNS.append((v + "_had", v, "had", True, True))
RUNS.append(("juv_H0.5_had_noCombined", "juv_H0.5", "had", False, True))

summary = {}
edits = {}
for name, var, ident, comb, prof in RUNS:
    lib = build(var, ident, comb, prof)
    path = os.path.join(SCR, f"fg_library_hs_{name}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(lib, fh, sort_keys=False, allow_unicode=True)
    txt = run_gate(path)
    open(os.path.join(OUTD, f"budget_gate_{name}.txt"), "w").write(txt)
    summary[name] = parse(txt)
    inter = lib["interaction_definitions"]
    edits[name] = {k: {kk: inter[k].get(kk) for kk in ("max_intake_rate", "stage_access", "depth_access_table")}
                   for k in [key(n) for n in ["predatory_fish->krill"] + WC_FISH]}
    print(f"{name}: done", flush=True)
json.dump({"gate": summary, "edits": edits}, open(os.path.join(OUTD, "gate_hs.json"), "w"), indent=1)

V = [r[0] for r in RUNS]
print("\nWC pair a after all factors (krill / planktivores / benthivores / whiting) and WC -> benthivores window")
for n in V:
    e = edits[n]
    print(f"  {n:28s} {e['predatory_fish_preys_on_krill']['max_intake_rate']:.4g} / "
          f"{e['predatory_fish_preys_on_planktivorous_fish']['max_intake_rate']:.4g} / "
          f"{e['predatory_fish_preys_on_benthivorous_fish']['max_intake_rate']:.4g} / "
          f"{e['predatory_fish_preys_on_whiting']['max_intake_rate']:.4g};  window "
          f"{e['predatory_fish_preys_on_benthivorous_fish']['stage_access']}")


def cell(d, k):
    r = d.get(k)
    return f"{r[0]:6.2f}{'*' if r[1] else ' '}" if r else "     - "


print("\nGate per run (* = OFF). Prey side: implied / booked M2. Predator side: encounter / booked share (equal cells; field spread)")
cols = [("prey plk", "prey_total", "planktivorous_fish"), ("prey ben", "prey_total", "benthivorous_fish"),
        ("prey whi", "prey_total", "whiting"), ("prey krill", "prey_total", "krill"),
        ("WC>ben prey", "prey_pair", "predatory_fish->benthivorous_fish"),
        ("WC>whi prey", "prey_pair", "predatory_fish->whiting"),
        ("WC>plk prey", "prey_pair", "predatory_fish->planktivorous_fish"),
        ("WC>ben eq", "diet", "predatory_fish->benthivorous_fish"),
        ("WC>ben fld", "diet_field", "predatory_fish->benthivorous_fish"),
        ("WC>krill eq", "diet", "predatory_fish->krill"), ("WC>krill fld", "diet_field", "predatory_fish->krill"),
        ("WC>whi eq", "diet", "predatory_fish->whiting"), ("WC>whi fld", "diet_field", "predatory_fish->whiting"),
        ("WC>plk eq", "diet", "predatory_fish->planktivorous_fish"), ("WC>plk fld", "diet_field", "predatory_fish->planktivorous_fish")]
print(f"  {'run':28s} " + " ".join(f"{c[0]:>12s}" for c in cols))
for n in V:
    row = []
    for _, sec, k in cols:
        if sec == "prey_total":
            x = summary[n]["prey_total"].get(k)
            row.append(f"{x:11.2f} " if x is not None else "          - ")
        else:
            row.append(f"{cell(summary[n][sec], k):>12s}")
    print(f"  {n:28s} " + " ".join(row))
print("\nOFF pairs per run (prey side / equal cells / field spread):")
for n in V:
    offs = sorted({k for sec in ("prey_pair", "diet", "diet_field") for k, r in summary[n][sec].items() if r[1]})
    print(f"  {n:28s} {', '.join(offs)}")
