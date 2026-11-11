"""Diet gate (tools/probes/budget_gate.py, unchanged source) on temporary library copies carrying the
one-rule attack rates of one_rule_pairs.py, plus a per-bed-class gate.

Usage: python3 gate_variants.py <repo_root> <one_rule_pairs.json> <scratch_dir> <tables_dir>

For every variant a copy of fgconfig/fg_library.yaml is written to <scratch_dir> (yaml safe_load / dump;
comments dropped) in which ONLY max_intake_rate (and, for the krill pairs, the depth_access_table class
factors) of the in-scope pairs change:
    krill pairs        a = library a x (new equal-cell a / reproduced a); table = new class a's
    fish pairs         a = library a x ratio (the rule's equal-cell annual mean / the library encounter)
    p variant "rev"    review capture p (Juanes et al. 2002 review; Synregel note item 3): krill 0.6 / 0.5,
                       section 220 fish pairs 0.45 / 0.2, benthivore fish pairs 0.45 / 0.3.
    flavour "T"        the fish pairs also get a depth_access_table from the rule's class a's (the rule's
                       own light-by-bed consequence); the plain flavour keeps the library's no-table form.
budget_gate.py is exec'd from its own file with one textual change (library_path=<copy> passed to the
probe builder); nothing in the repository is written. After it, a per-class gate: the predator spread
evenly over the cells of one bed class (16 x 16 probe grid), each prey at its manifest density x the
field shape budget_gate uses x the pair's visible share; fish pairs either flat (plain) or class-resolved
(the rule's class factors, which is what the T flavour puts into the engine).
"""
import contextlib
import io
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(sys.argv[1])
RES = json.load(open(sys.argv[2]))
SCR = os.path.abspath(sys.argv[3])
TAB = os.path.abspath(sys.argv[4])
os.makedirs(SCR, exist_ok=True)
BG = os.path.join(ROOT, "tools", "probes", "budget_gate.py")
BG_SRC = open(BG).read()
OLD = "apply_natural_mortality=True, migration=True)"
assert BG_SRC.count(OLD) == 1
LIB0 = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))
INTER = "interaction_definitions"
TABLE_DEPTHS = [(99.9, 0), (100.0, 1), (149.9, 1), (150.0, 2), (249.9, 2), (250.0, 3)]
KRILL_PRED = ["predatory_fish", "whiting", "planktivorous_fish", "benthivorous_fish", "mackerel"]
P_REV = {"krill": 0.6 / 0.5, "fish220": 0.45 / 0.2, "benth": 0.45 / 0.3}
BENTH = {"benthivorous_fish->planktivorous_fish", "benthivorous_fish->whiting"}


def pair_key(name):
    pred, prey = name.split("->")
    return f"{pred}_preys_on_{prey}"


def table(vals):
    return [{"depth_m": d, "factor": float(f"{max(vals[k], 0.0):.6g}")} for d, k in TABLE_DEPTHS]


def make_library(var, prev, tables):
    lib = json.loads(json.dumps(LIB0))
    inter = lib[INTER]
    fixes = {}
    if var is not None:
        v = RES["variants"][var]
        for g in KRILL_PRED:
            r = v["krill"][g]
            key = f"{g}_preys_on_krill"
            a = inter[key]["max_intake_rate"] * r["ratio"] * (P_REV["krill"] if prev else 1.0)
            inter[key]["max_intake_rate"] = float(f"{a:.6g}")
            inter[key]["depth_access_table"] = table(r["classes_lib_units"])
            fixes[key] = a
        for name, r in v["fish"].items():
            key = pair_key(name)
            pf = (P_REV["benth"] if name in BENTH else P_REV["fish220"]) if prev else 1.0
            a = inter[key]["max_intake_rate"] * r["ratio"] * pf
            inter[key]["max_intake_rate"] = float(f"{a:.6g}")
            if tables:
                inter[key]["depth_access_table"] = table(r["cls_rel"])
            fixes[key] = a
    elif prev:   # library vision, review p only (the sibling note's V_p, library-wide)
        for g in KRILL_PRED:
            key = f"{g}_preys_on_krill"
            inter[key]["max_intake_rate"] *= P_REV["krill"]
            fixes[key] = inter[key]["max_intake_rate"]
        for name in RES["variants"]["A"]["fish"]:
            key = pair_key(name)
            inter[key]["max_intake_rate"] *= P_REV["benth"] if name in BENTH else P_REV["fish220"]
            fixes[key] = inter[key]["max_intake_rate"]
    return lib, fixes


def run_gate(libpath, cls_fac):
    src = BG_SRC.replace(OLD, f"apply_natural_mortality=True, migration=True, library_path={libpath!r})")
    ns = {"__name__": "budget_gate_variant", "__file__": BG}
    buf = io.StringIO()
    cwd = os.getcwd()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, BG, "exec"), ns)
    os.chdir(cwd)
    env = ns["env"]
    density = ns["density"]
    from lib.diagnostics.field_diet import bed_class, field_prey_shape
    depth = ns["_depth"]
    cls = bed_class(depth)
    out = {"equal": ns["gate_rows"], "field": ns["field_rows"], "class": {}, "class_flat": {}}
    for i, pid in enumerate(env.dm_ids):
        if pid not in KRILL_PRED:
            continue
        pairs = env.fgs[pid].params.get("interaction", {}) or {}
        menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j] and env.global_fg_order[j] in density]
        book = {env.global_fg_order[j]: float((pairs.get(f"{pid}_preys_on_{env.global_fg_order[j]}") or {})
                                              .get("predation_mortality", 0.0) or 0.0) * density[env.global_fg_order[j]]
                for j in menu}
        bs = sum(book.values())
        for flav, use in (("class", True), ("class_flat", False)):
            per = []
            for k in range(4):
                cells = cls == k
                enc = {}
                for j in menu:
                    prey = env.global_fg_order[j]
                    mult = (np.asarray(env.pair_depth_mult[i, j], float) if env._has_pair_depth else np.ones(depth.shape))
                    a = float(env.max_intake_mat[i, j]) * mult[cells]
                    f = cls_fac.get(f"{pid}->{prey}") if use else None
                    if f is not None and not np.allclose(mult, mult.flat[0]):
                        f = None   # the pair already carries a table (T flavour): do not apply twice
                    if f is not None:
                        a = a * f[k] / float(np.sum(np.asarray(f) * np.array([(cls == kk).mean() for kk in range(4)])))
                    bv = density[prey] * field_prey_shape(prey, env.fgs[prey], depth)[cells] * ns["_visible_share"](i, j, prey)
                    t3 = float(env._type3_pred_mask[i, j, 0, 0]) > 0.0
                    enc[prey] = float(np.mean(a * bv * (bv if t3 else 1.0)))
                es = sum(enc.values())
                per.append({p: dict(enc=enc[p] / es if es > 0 else 0.0, book=book[p] / bs) for p in enc})
            out[flav][pid] = per
    return buf.getvalue(), out


RUNS = [("lib", None, False, False), ("lib_rev", None, True, False)]
for var in RES["variants"]:
    RUNS.append((var, var, False, False))
    RUNS.append((var + "_rev", var, True, False))
for var in ("A", "B", "A_Ke2e-4", "A_Ke0.1", "B_Ke2e-4", "B_Ke0.1"):
    RUNS.append((var + "_T", var, False, True))
    RUNS.append((var + "_rev_T", var, True, True))
SUMMARY = {}
for label, var, prev, tables in RUNS:
    lib, fixes = make_library(var, prev, tables)
    lp = os.path.join(SCR, f"fg_library_{label}.yaml")
    with open(lp, "w") as fh:
        yaml.safe_dump(lib, fh, sort_keys=False)
    cls_fac = {}
    if var is not None:
        for name, r in RES["variants"][var]["fish"].items():
            cls_fac[name] = r["cls_rel"]
    txt, out = run_gate(lp, cls_fac)
    with open(os.path.join(TAB, "gates", f"budget_gate_{label}.txt"), "w") as fh:
        fh.write(f"# budget_gate.py on a temporary library copy, variant {label}; changed a's: "
                 + ", ".join(f"{k} {v:.4g}" for k, v in fixes.items()) + "\n" + txt)
    SUMMARY[label] = out
    print(f"{label}: done")


def ser(x):
    if isinstance(x, tuple):
        return [ser(v) for v in x]
    if isinstance(x, dict):
        return {k: ser(v) for k, v in x.items()}
    return x


json.dump(ser(SUMMARY), open(os.path.join(TAB, "gate_variants.json"), "w"), indent=1, default=float)
