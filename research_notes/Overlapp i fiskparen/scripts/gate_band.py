"""Diet gate (tools/probes/budget_gate.py, unchanged source) on temporary library copies: today's library, the
sibling note's rule A on the in-layer overlap ("literal reading"), and rule A / library vision on the BAND
overlap of band_pairs.py, plus a per-bed-class gate.

Usage: python3 gate_band.py <repo_root> <band_pairs.json> <one_rule_pairs.json (sibling)> <scratch_dir> <tables_dir>

The machinery is research_notes/"En synregel for fiskparen"/scripts/gate_variants.py, exec'd up to its own run list
(make_library, run_gate: budget_gate.py exec'd with library_path=<copy>; only max_intake_rate and, for the krill
pairs, the depth_access_table class a's change; flavour "T" also gives each fish pair the depth_access_table of
its own class a's; "rev" = review capture p). Library copies live in <scratch_dir>; nothing in the repository
changes.
"""
import json
import os
import sys

ROOT = os.path.abspath(sys.argv[1])
BANDJ, SIBJ, SCR, TAB = sys.argv[2], sys.argv[3], os.path.abspath(sys.argv[4]), os.path.abspath(sys.argv[5])
GV = os.path.join(ROOT, "research_notes", "En synregel for fiskparen", "scripts", "gate_variants.py")
src = open(GV).read()
head = src[:src.index("RUNS = [")]
ARGV = list(sys.argv)
sys.argv = [GV, ROOT, BANDJ, SCR, TAB]
NS = {"__name__": "gate_variants_head", "__file__": GV}
os.chdir(ROOT)
sys.path.insert(0, ROOT)
exec(compile(head, GV, "exec"), NS)
sys.argv = ARGV
BAND = json.load(open(BANDJ))
SIB = json.load(open(SIBJ))
os.makedirs(os.path.join(TAB, "gates"), exist_ok=True)

RUNS = [("lib", None, None, False, False),
        ("A_inlayer", SIB, "A", False, False), ("A_inlayer_T", SIB, "A", False, True),
        ("lib_band", BAND, "lib_band", False, False), ("lib_band_T", BAND, "lib_band", False, True),
        ("A_band", BAND, "A_band", False, False), ("A_band_T", BAND, "A_band", False, True),
        ("A_band_rev_T", BAND, "A_band", True, True)]
for v in ("A_band_noclip", "A_band_Hs5", "A_band_Hs40", "A_band_fup0.25", "A_band_Hn1.0", "A_band_Hw5", "A_band_Hw30",
          "A_band_Ke2e-4", "A_band_Ke0.1"):
    RUNS.append((v + "_T", BAND, v, False, True))
# extra library edits per run (brackets the band a's depend on; not proposals):
#   WC -> benthivores window: section 243's stage_access [0.42, 0.12] was a-weighted with the 243 member shares
#   (saithe 0.45 ...); the band a is saithe 0.89, so the saithe-weighted windows bracket it: pre-243 [0.30, 0.038],
#   the audit's aw [0.312, 0.0070] ("Vattenpelarens bottenfiskbyte").
#   Whiting's bed pair carries H_bed / H_whiting (section 232): with whiting's band at Hw it scales by 10 / Hw.
EXTRA = {"A_band_win243pre_T": ("A_band", {"predatory_fish_preys_on_benthivorous_fish": {"stage_access": [0.30, 0.038]}}),
         "A_band_winaw_T": ("A_band", {"predatory_fish_preys_on_benthivorous_fish": {"stage_access": [0.312, 0.0070]}}),
         "A_band_Hw5_epi_T": ("A_band_Hw5", {"whiting_preys_on_mobile_epifauna": {"max_intake_rate": 0.0116 * 10 / 5}}),
         "A_band_Hw30_epi_T": ("A_band_Hw30", {"whiting_preys_on_mobile_epifauna": {"max_intake_rate": 0.0116 * 10 / 30}})}
for lab, (v, _) in EXTRA.items():
    RUNS.append((lab, BAND, v, False, True))
SUMMARY = {}
for label, res, var, prev, tables in RUNS:
    if res is not None:
        NS["RES"] = res
    lib, fixes = NS["make_library"](var, prev, tables)
    for key, kv in (EXTRA[label][1] if label in EXTRA else {}).items():
        lib["interaction_definitions"][key].update(kv)
        fixes[key + " " + ",".join(kv)] = float(list(kv.values())[0][0] if isinstance(list(kv.values())[0], list)
                                                else list(kv.values())[0])
    lp = os.path.join(SCR, f"fg_library_{label}.yaml")
    with open(lp, "w") as fh:
        NS["yaml"].safe_dump(lib, fh, sort_keys=False)
    cls_fac = {name: r["cls_rel"] for name, r in res["variants"][var]["fish"].items()} if var is not None else {}
    txt, out = NS["run_gate"](lp, cls_fac)
    with open(os.path.join(TAB, "gates", f"budget_gate_{label}.txt"), "w") as fh:
        fh.write(f"# budget_gate.py on a temporary library copy, variant {label}; changed a's: "
                 + ", ".join(f"{k} {v:.4g}" for k, v in fixes.items()) + "\n" + txt)
    SUMMARY[label] = dict(out, fixes=fixes)
    print(f"{label}: done", flush=True)
json.dump(SUMMARY, open(os.path.join(TAB, "gate_band.json"), "w"), indent=1,
          default=float)
