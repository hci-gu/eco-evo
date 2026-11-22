"""One table per variant from run_all.sh's outputs: class tables, G, the field-spread diet gate and prey-side krill,
and the policy-free intake G by month. Usage: python3 -I tabulate.py <tables_dir>"""
import json
import os
import re
import sys

import numpy as np

T = sys.argv[1]
V = json.load(open(os.path.join(T, "variants.json")))
SHARE = np.array(V["share"])
RUNS, SRC = V["runs"], V.get("src", {})


def g3(x):
    return f"{float(f'{x:.3g}'):g}" if x > 0 else "0"


def G(c):
    c = np.asarray(c, float)
    w = SHARE[:2] / SHARE[:2].sum()
    return float(w @ c[:2] / c[2]), float(c[0] / c[2])


def gate(path):
    txt = open(path).read()
    out = {}
    blk = txt[txt.index("predatory_fish (saithe, GOV in the box"):]
    blk = blk[:blk.index("whiting (whiting, GOV")]
    for prey in ("krill", "planktivorous_fish", "benthivorous_fish", "whiting"):
        m = re.search(rf"\s{prey} enc\s+([\d.]+) booked\s+([\d.]+) ratio\s+([\d.]+)\s+(\w+)", blk)
        out[prey] = (float(m.group(1)), float(m.group(3)), m.group(4))
    m = re.search(r"Diet gate at the field spread: (.*)", txt)
    out["field_off"] = m.group(1).strip()
    m = re.search(r"Diet gate: (.*)", txt)
    out["equal_off"] = m.group(1).strip()
    # prey side: totals implied / booked on krill and the planktivores, and the prey-side OFF list
    for prey in ("krill", "planktivorous_fish", "benthivorous_fish", "whiting"):
        m = re.search(rf"^\s+{prey}: booked M2 ([\d.]+) /yr, implied ([\d.e-]+) \(ratio ([\d.]+)\)", txt, re.M)
        out[f"prey_{prey}"] = float(m.group(3))
    m = re.search(r"Prey-side check: (.*)", txt)
    out["prey_off"] = m.group(1).strip()
    return out


def off_sets(g_):
    """pair names in the three OFF lists: field spread, equal cells, prey side."""
    pat = re.compile(r"(\w+) -> (\w+)")
    return {k: set(pat.findall(g_[k])) for k in ("field_off", "equal_off", "prey_off")}


def intake(path):
    txt = open(path).read()
    res = {}
    for mon in ("Jan", "Apr", "Jul", "Oct"):
        blk = txt[txt.index(f"== {mon}:"):]
        m = re.search(r"\n\s+library\s+I ([\d. ]+?)\s+G\s+([\d.]+)\s+([\d.]+)\s+krill ([\d. ]+?)\s+S", blk)
        res[mon] = (m.group(1).split(), float(m.group(2)), float(m.group(3)), m.group(4).split())
    return res


print("Variants (pre-296 class a's in library units; the loader renormalises each table to an equal-cell mean 1).")
print("G = a(< 150 m, cell-weighted) / a(150-250 m); R0 = a(< 100 m) / a(150-250 m) (caller's x27 for krill).")
print("Intake G = policy-free intake per tonne I(< 150 m) / I(150-250 m) (Jan / Apr / Jul / Oct), intake_by_class.py.")
print("Gate = budget_gate's diet gate at the field spread (saithe GOV bed shares), WC encounter / booked, krill and")
print("planktivores; scope wc = the four WC pairs written (a and table), all = also whiting, planktivores and mackerel ->")
print("krill, wcshape = the four WC tables only at the live a (section 296 level kept). OFF sets are diffed by pair name")
print("against the library's (budget_gate's field-spread, equal-cell and prey-side lists).\n")
for name, r in RUNS.items():
    k, p = r["krill"]["predatory_fish"], r["fish"]["plk"]
    gk, rk = G(k["classes"])
    gp, rp = G(p["classes"])
    print(f"== {name}: {SRC.get(name, '')}")
    print(f"   krill        " + " / ".join(g3(x) for x in k["classes"]) + f"   G {gk:.2f}  R0 {rk:.1f}  a x"
          f"{k['a'] / RUNS['library']['krill']['predatory_fish']['a']:.3f}  n/d {k['nd']:.4f}")
    print(f"   planktivores " + " / ".join(g3(x) for x in p["classes"]) + f"   G {gp:.2f}  R0 {rp:.1f}  a x"
          f"{p['a'] / RUNS['library']['fish']['plk']['a']:.3f}")
    for key, lab in (("benth", "benthivores"), ("whit", "whiting")):
        f = r["fish"][key]
        gf, rf = G(f["classes"])
        print(f"   {lab:12s} " + " / ".join(g3(x) for x in f["classes"]) + f"   G {gf:.2f}  a x"
              f"{f['a'] / RUNS['library']['fish'][key]['a']:.3f}")
    for sc in ("wc", "all", "wcshape"):
        gp_ = os.path.join(T, "gate", f"{name}_{sc}.txt")
        ip_ = os.path.join(T, "intake", f"{name}_{sc}.txt")
        if not (os.path.exists(gp_) and os.path.exists(ip_)):
            continue
        g_ = gate(gp_)
        it = intake(ip_)
        print(f"   [{sc}] intake G " + " / ".join(f"{it[m][1]:.2f}" for m in it)
              + "   I by class Jul " + " / ".join(it["Jul"][0]) + "   krill share of encounters Jul " + " / ".join(it["Jul"][3]))
        print(f"   [{sc}] field-spread gate: krill enc {g_['krill'][0]:.3f} ratio {g_['krill'][1]:.2f} {g_['krill'][2]};"
              f" planktivores {g_['planktivorous_fish'][0]:.3f} ratio {g_['planktivorous_fish'][1]:.2f}"
              f" {g_['planktivorous_fish'][2]}; benthivores ratio {g_['benthivorous_fish'][1]:.2f}"
              f" {g_['benthivorous_fish'][2]}; whiting ratio {g_['whiting'][1]:.2f} {g_['whiting'][2]}")
        print(f"   [{sc}] prey side implied/booked M2: krill {g_['prey_krill']:.2f}, planktivores"
              f" {g_['prey_planktivorous_fish']:.2f}, benthivores {g_['prey_benthivorous_fish']:.2f}, whiting"
              f" {g_['prey_whiting']:.2f}")
        offs = off_sets(g_)
        lib = off_sets(gate(os.path.join(T, "gate", f"library_{sc}.txt")))
        diffs = []
        for k, lab in (("field_off", "field spread"), ("equal_off", "equal cells"), ("prey_off", "prey side")):
            new_ = offs[k] - lib[k]
            gone = lib[k] - offs[k]
            if new_:
                diffs.append(f"{lab} NEW OFF: " + ", ".join(f"{a} -> {b}" for a, b in sorted(new_)))
            if gone:
                diffs.append(f"{lab} back ON: " + ", ".join(f"{a} -> {b}" for a, b in sorted(gone)))
        print(f"   [{sc}] OFF sets vs library: " + ("; ".join(diffs) if diffs else "unchanged"))
    print()
