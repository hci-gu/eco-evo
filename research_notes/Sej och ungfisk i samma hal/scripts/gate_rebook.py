"""COPY of research_notes/"Vattenpelarens vuxna bottenfisk"/scripts/gate_adult.py (2026-11-11), itself a copy of
research_notes/"Rekryternas bottendjup"/scripts/gate_ben_shape.py. Changed only in its run list, its library edits
and one extra argument: every run starts from that note's CENTRAL edit (pA+Aprof+hake_out: combined capture-p x
schooling factors, WC -> benthivores a x0.945 -> 0.117 and stage_access [0.3818, 0.01653]; benthivores at the field
spread with J on the recruit_depth_table 2.71 / 1.00 / 0.21 / 0.05 and A on the GOV annual shape) and varies
  (i)  the pair's predation_mortality (the re-booking of this note, scripts/sms_rebook.py), and
  (ii) as a DIAGNOSTIC only, f_J of the pair x the within-class co-occurrence factor W of scripts/gov_within_class.py
       (the J part of the encounter scaled, as research_notes/"Sejens hojd over botten"/scripts/cooccurrence_check.py).

Usage: python3 gate_rebook.py <repo_root> <gov_juv_bed.json> <decompose_members.json> <scratch_dir> <out_dir>
                              <within_class.json>
       (imports lib/ through budget_gate.py, so not -I; the jsons are research_notes outputs)

Original docstring follows.
budget_gate with the benthivores placed by bed depth: J on a juvenile shape, A on the adult shape (GOV, this
note's gov_juv_bed.py), on library copies carrying the combined capture-p x schooling factors of
research_notes/"Fangst och stim i fiskparen" (central), as research_notes/"Sejens hojd over botten"/scripts/gate_hs.py.

Usage: python3 gate_ben_shape.py <repo_root> <gov_juv_bed.json> <scratch_dir> <out_dir>
       (imports lib/ through budget_gate.py, so not -I; the json is this note's own output)

tools/probes/budget_gate.py and lib/diagnostics/field_diet.py are NOT edited: the gate source is read, patched
in memory by exact string replacement (each anchor asserted to occur once) and executed, as gate_p.py does.
Patches (all switchable per run):
  P0 library_path=<copy> (every run).
  P1 field spread: the benthivores as PREY at density x [js f_J shape_J(c) + (1 - js) f_A shape_A(c)] per cell,
     js the annual-mean start share (as _visible_share), shape_X the bed-class table at each cell's depth renormalised
     to mean 1 on the gate's grid (as the loader treats a depth table). Every other prey unchanged.
  P2 equal cells: the same per-cell benthivore prey with the predator uniform, encounter = a mean_c(mult(c) bv(c))
     (Type III: bv^2) - changes only pairs with a depth_access_table (mult not constant).
  P3 the benthivores as PREDATOR at the field spread with their whole-group class shares (B shape x cell shares).
No value is chosen from the gate; it is a check.
"""
import contextlib
import copy
import io
import json
import os
import re
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(sys.argv[1])
SH = json.load(open(sys.argv[2]))
DM = json.load(open(sys.argv[3]))
SCR = os.path.abspath(sys.argv[4])
OUTD = os.path.abspath(sys.argv[5])
WC = json.load(open(sys.argv[6]))["rows"]
os.makedirs(SCR, exist_ok=True)
os.makedirs(OUTD, exist_ok=True)
RN = os.path.join(ROOT, "research_notes")
FAC = json.load(open(os.path.join(RN, "Fangst och stim i fiskparen", "tables", "gates", "combined.json")))[
    "factors"]["central"]
BG = os.path.join(ROOT, "tools", "probes", "budget_gate.py")
SRC0 = open(BG).read()
CELL60 = np.array(SH["_cell_shares"])

A_LIB = "apply_natural_mortality=True, migration=True)"
A_FIELD_BV = ("        bv = (density[prey_id] * field_prey_shape(prey_id, env.fgs[prey_id], _depth)\n"
              "              * _visible_share(i, j, prey_id))\n")
A_DEPTH = "_depth = habitat.load_depth_map(_manifest, PROJECT, (H, W))\n"
A_EQ_ENC = "        enc = a * bv * (bv if type3 else 1.0)\n"
A_EQ_HEAD = "density = _start_densities()\n"
for anchor in (A_LIB, A_FIELD_BV, A_DEPTH, A_EQ_ENC, A_EQ_HEAD):
    assert SRC0.count(anchor) == 1, anchor

HELPERS = '''
from lib.diagnostics import field_diet as _fd
from lib.world import habitat as _hab
with open(PROJECT, encoding='utf-8-sig') as _fh:
    _man0 = yaml.safe_load(_fh)
_depth0 = _hab.load_depth_map(_man0, PROJECT, (H, W))


def _class_map(f4):
    m = np.asarray(f4, dtype=np.float64)[_fd.bed_class(_depth0)]
    return m / m.mean()


def _ben_cells(i, j):
    js = float(env.stage_start[:, j].mean())
    return (js * float(env.stage_f_j[i, j]) * _class_map(_BEN['J'])
            + (1.0 - js) * float(env.stage_f_a[i, j]) * _class_map(_BEN['A']))


def _enc_eq(i, j, prey_id, b, a, bv, type3):
    if not (_BEN and _BEN.get('eq') and prey_id == 'benthivorous_fish' and _depth0 is not None):
        return a * bv * (bv if type3 else 1.0)
    mult = (np.asarray(env.pair_depth_mult[i, j], dtype=np.float64)
            if getattr(env, '_has_pair_depth', False) else 1.0)
    bvc = b * _ben_cells(i, j)
    return float(env.max_intake_mat[i, j]) * float(np.mean(mult * bvc * (bvc if type3 else 1.0)))

'''
FIELD_HOOK = '''
if _BEN and _BEN.get('pred'):
    FIELD_BED_SHARES = dict(FIELD_BED_SHARES)
    FIELD_BED_SHARES['benthivorous_fish'] = (tuple(_BEN['pred']), 'benthivores, GOV B shape (this note)')


def _prey_cells(i, j, prey_id):
    if _BEN and _BEN.get('field') and prey_id == 'benthivorous_fish' and _depth is not None:
        return _ben_cells(i, j)
    return field_prey_shape(prey_id, env.fgs[prey_id], _depth) * _visible_share(i, j, prey_id)

'''
SRC_T = (SRC0.replace(A_EQ_HEAD, HELPERS + A_EQ_HEAD)
         .replace(A_EQ_ENC, "        enc = _enc_eq(i, j, prey_id, b, a, bv, type3)\n")
         .replace(A_DEPTH, A_DEPTH + FIELD_HOOK)
         .replace(A_FIELD_BV, "        bv = density[prey_id] * _prey_cells(i, j, prey_id)\n"))


def run_gate(libpath, ben):
    src = SRC_T.replace(A_LIB, f"apply_natural_mortality=True, migration=True, library_path={libpath!r})")
    buf = io.StringIO()
    ns = {"__name__": "__main__", "__file__": BG, "_BEN": ben}
    cwd = os.getcwd()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, BG, "exec"), ns)
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    return buf.getvalue()


NUM = r"([\d.]+(?:e[-+]?\d+)?)"


def parse(txt):
    out = {"diet": {}, "diet_field": {}, "prey": {}}
    eq = txt[txt.index("=== diet gate: encounter"):txt.index("=== diet gate at the field spread")]
    pred = None
    for line in eq.splitlines():
        m = re.match(r"\s*([a-z_]+)\s+prey\s+B\s+B_vis", line)
        if m:
            pred = m.group(1)
            continue
        m = re.match(r"\s*([a-z_]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+|inf)\s+([\d.]+)\s+(ok|OFF)", line)
        if m and pred:
            out["diet"][f"{pred}->{m.group(1)}"] = (float(m.group(2)), float(m.group(3)), float(m.group(4)),
                                                   m.group(6) == "OFF")
    fs = txt[txt.index("=== diet gate at the field spread"):txt.index("=== prey-side check")]
    for line in fs.splitlines():
        m = re.match(r"\s*([a-z_]+) \(", line)
        if m:
            pred = m.group(1)
            continue
        m = re.match(r"\s*([a-z_]+) enc\s+([\d.]+) booked\s+([\d.]+) ratio\s+([\d.]+|inf)\s+(ok|OFF)", line)
        if m and pred:
            out["diet_field"][f"{pred}->{m.group(1)}"] = (float(m.group(2)), float(m.group(3)), float(m.group(4)),
                                                         m.group(5) == "OFF")
    ps = txt[txt.index("=== prey-side check"):txt.index("=== total ration")]
    prey = None
    for line in ps.splitlines():
        m = re.match(r"\s*([a-z_]+): booked M2 " + NUM + r" /yr, implied " + NUM, line)
        if m:
            prey = m.group(1)
            out["prey"][prey] = {}
            continue
        m = re.match(r"\s*([a-z_]+) booked\s+" + NUM + r" \(\s*[\d.]+\)\s+(.*)$", line)
        if m and prey:
            rest = m.group(3)
            imp = re.search(r"implied\s+" + NUM, rest)
            fld = re.search(r"field\s+" + NUM, rest)
            out["prey"][prey][m.group(1)] = (float(m.group(2)), float(imp.group(1)) if imp else None,
                                             float(fld.group(1)) if fld else None, "OFF" in rest)
    return out


def prey_totals(p):
    """Equal-cell and field-spread totals: implied / booked M2 (field: a predator without a field row counts at
    its equal-cell implied value, one the gate skips at its booked value)."""
    book = sum(v[0] for v in p.values())
    eq = sum(v[1] if v[1] is not None else v[0] for v in p.values())
    fld = sum(v[2] if v[2] is not None else (v[1] if v[1] is not None else v[0]) for v in p.values())
    return eq / book, fld / book


LIB0 = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))
PAIR = "predatory_fish_preys_on_benthivorous_fish"
FL = DM["flows"]
P0 = DM["p0"]
JS = DM["JS"]
APROF = {}
for k, v in DM["members"]["predatory_fish->benthivorous_fish"].items():
    sp = k.split("|")[0]
    APROF.setdefault(sp, []).append((v["share"], v["Aprof"]))
APROF = {sp: sum(s * f for s, f in l) / sum(s for s, f in l) for sp, l in APROF.items()}


def pair_edit(g=None, stage_p=False, aprof=False, fA0=(), fJ0=(), ling_win=None):
    """member factors -> (a multiplier, f_J, f_A) relative to the section 269 library pair.
    g: {member: multiplier on its a} (0 = out of the pair); fA0 / fJ0: members whose A / J window is set to 0;
    ling_win: a WIN key for ling's own window (decompose_members.json windows)."""
    g = g or {}
    num_a = den_a = nj = na = nj0 = na0 = 0.0
    for sp, r in FL.items():
        s = r["a_share"]
        gm = g.get(sp, 1.0)
        fj, fa, pj, pa = r["fJ"], r["fA"], r["pJ"], r["pA"]
        if sp == "ling" and ling_win:
            w = DM["windows"][ling_win]
            fj, fa, pj, pa = w["fJ"], w["fA"], w["pJ"], w["pA"]
        nj0 += s * r["fJ"]
        na0 += s * r["fA"]
        mj = (pj / P0 if (stage_p and pj == pj) else 1.0) * (0.0 if sp in fJ0 else 1.0)
        ma = (pa / P0 if (stage_p and pa == pa) else 1.0) * (APROF.get(sp, 1.0) if aprof else 1.0) * (0.0 if sp in fA0 else 1.0)
        num_a += s * gm
        nj += s * gm * fj * mj
        na += s * gm * fa * ma
    am = num_a
    fj_new = 0.392 * (nj / am) / nj0
    fa_new = 0.0388 * (na / am) / na0
    return am, fj_new, fa_new


def lib_copy(tag, edit=None, m2=None, fj_mult=1.0):
    lib = copy.deepcopy(LIB0)
    inter = lib["interaction_definitions"]
    for name, f in FAC.items():
        p, q = name.split("->")
        e = inter[f"{p}_preys_on_{q}"]
        e["max_intake_rate"] = float(f"{e['max_intake_rate'] * f:.4g}")
    info = None
    if edit is not None:
        am, fj, fa = pair_edit(**edit)
        e = inter[PAIR]
        e["max_intake_rate"] = float(f"{e['max_intake_rate'] * am:.4g}")
        e["stage_access"] = [float(f"{fj * fj_mult:.4g}"), float(f"{fa:.4g}")]
        if m2 is not None:
            e["predation_mortality"] = m2
        info = dict(a=e["max_intake_rate"], stage_access=e["stage_access"], a_mult=am,
                    m2=e["predation_mortality"], fj_mult=fj_mult)
    path = os.path.join(SCR, f"fg_library_rebook_{tag}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(lib, fh, sort_keys=False, allow_unicode=True)
    return path, info


def S(key, which="central"):
    return SH[key][which]


def ben(J, A, field=True, eq=False, pred=None):
    return {"J": list(J), "A": list(A), "field": field, "eq": eq,
            "pred": None if pred is None else list(np.asarray(pred) * CELL60 / (np.asarray(pred) * CELL60).sum())}


C = "h|box|q"
AC = S(f"{C}|ann|all|A")
JIMP = [2.71, 1.00, 0.21, 0.05]           # the proposed recruit_depth_table (Rekryternas bottendjup)
SHAPE = ben(JIMP, AC)
HAKE_OUT = {"hake": 0.0}
CENTRAL = dict(stage_p=True, aprof=True, g=HAKE_OUT)       # gate_adult.py 'pA+Aprof+hake_out'
RB = json.load(open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[6])), "rebook.json")))
BOOK = {"0.0833 (library)": 0.0833,
        "field central": round(RB["field"]["booking"], 4),
        "field MC 5 %": round(RB["field"]["booking_mc"][0], 4),
        "field MC 95 %": round(RB["field"]["booking_mc"][1], 4),
        "SMS stomach-level (pred side)": round(RB["sms"]["booking"]["predator side|stomach level x0.45"], 4),
        "SMS level (prey side)": round(RB["sms"]["booking"]["prey side|SMS level"], 4)}
WJ = WC["box|pres|J|W"]
WW = WC["wide|pres|J|W"]
WYEAR = WC["box|pres|J|W/E_yearLOO"]
VARIANTS = []
for lab, m2 in BOOK.items():
    VARIANTS.append((f"book_{m2:.4f}", dict(edit=CENTRAL, m2=m2), lab))
for lab, wv in ((f"W box {WJ['central']:.2f}", WJ["central"]), (f"W box lo {WJ['lo']:.2f}", WJ["lo"]),
                (f"W box hi {WJ['hi']:.2f}", min(WJ["hi"], 1.5)), (f"W wide {WW['central']:.2f}", WW["central"]),
                (f"W box within-year {WYEAR['central']:.2f}", WYEAR["central"])):
    VARIANTS.append((lab.replace(" ", "_"), dict(edit=CENTRAL, fj_mult=wv), "diagnostic: f_J x W, booking 0.0833"))
    VARIANTS.append((lab.replace(" ", "_") + "+book", dict(edit=CENTRAL, fj_mult=wv, m2=BOOK["field central"]),
                     "diagnostic: f_J x W, field-central booking"))
RUNS = []
EDITS = {}
LABEL = {}
for tag, kw, lab in VARIANTS:
    path, info = lib_copy(tag.replace(" ", "_"), **kw)
    EDITS[tag] = info
    LABEL[tag] = lab
    RUNS.append((tag, path, SHAPE))
    RUNS.append((tag + "+eq", path, ben(JIMP, AC, eq=True)))
summary = {}
for name, path, b in RUNS:
    txt = run_gate(path, b)
    open(os.path.join(OUTD, f"budget_gate_{name}.txt"), "w").write(txt)
    summary[name] = parse(txt)
    print(f"{name}: done", flush=True)
json.dump(dict(summary=summary, edits=EDITS, labels=LABEL), open(os.path.join(OUTD, "gate_rebook.json"), "w"), indent=1)
print("\nPair edits (a after the combined factor x2.96 and the adult-part corrections; stage_access; booking):")
for tag, info in EDITS.items():
    print(f"  {tag:34s} a {info['a']:.4g} (x{info['a_mult']:.3f}), stage_access {info['stage_access']} (f_J x"
          f" {info['fj_mult']:.3f}), predation_mortality {info['m2']}  [{LABEL[tag]}]")
print("\nJ / A shapes used (f per class < 100 / 100-150 / 150-250 / >= 250 m, mean 1 on the 60x60 map; the gate "
      "renormalises on its 16x16 grid):")
for name, _p, b in RUNS:
    if b:
        print(f"  {name:26s} J {'/'.join(f'{x:.2f}' for x in b['J'])}  A {'/'.join(f'{x:.2f}' for x in b['A'])}"
              + (f"  pred shares {'/'.join(f'{x:.3f}' for x in b['pred'])}" if b['pred'] else ""))


def pr(d, k):
    r = d.get(k)
    return f"{r[2]:5.2f}{'*' if r[3] else ' '}" if r else "    - "


def pp(p, prey, pred):
    v = p.get(prey, {}).get(pred)
    if not v:
        return "     -      -"
    eq = v[1] / v[0] if v[1] is not None else float("nan")
    fl = v[2] / v[0] if v[2] is not None else float("nan")
    return f"{eq:6.2f} {fl:6.2f}" + ("*" if v[3] else " ")


print("\nPREY SIDE on the benthivores: implied / booked M2, equal cells then field spread (* = OFF on the equal-cell"
      " prey check)")
PREDS = ["predatory_fish", "whiting", "seals", "porpoises"]
print(f"  {'run':26s} " + " ".join(f"{p:>14s}" for p in PREDS) + "   total eq / field")
for name, _, _ in RUNS:
    p = summary[name]["prey"]
    t = prey_totals(p["benthivorous_fish"])
    print(f"  {name:26s} " + " ".join(f"{pp(p, 'benthivorous_fish', q):>14s}" for q in PREDS)
          + f"   {t[0]:5.2f} / {t[1]:5.2f}")

print("\nPREDATOR SIDE (encounter share / booked share of the predator's diet), equal cells | field spread:")
KEYS = ["predatory_fish->benthivorous_fish", "whiting->benthivorous_fish", "seals->benthivorous_fish",
        "porpoises->benthivorous_fish"]
print(f"  {'run':26s} " + " ".join(f"{k.split('->')[0]:>16s}" for k in KEYS))
for name, _, _ in RUNS:
    s = summary[name]
    print(f"  {name:26s} " + " ".join(f"{pr(s['diet'], k)} | {pr(s['diet_field'], k)}" for k in KEYS))

print("\nOther prey totals (implied / booked M2, equal cells / field spread):")
OTH = ["planktivorous_fish", "krill", "whiting", "mobile_epifauna", "norway_lobster", "benthic_infauna"]
print(f"  {'run':26s} " + " ".join(f"{k[:13]:>14s}" for k in OTH))
for name, _, _ in RUNS:
    p = summary[name]["prey"]
    print(f"  {name:26s} " + " ".join(
        "{:6.2f}/{:5.2f}  ".format(*prey_totals(p[k])) if k in p else "        -     " for k in OTH))

print("\nBenthivores as predator at the field spread (P3 runs only): encounter / booked share")
for name, _, b in RUNS:
    if b and b.get("pred"):
        rows = {k: v for k, v in summary[name]["diet_field"].items() if k.startswith("benthivorous_fish->")}
        print(f"  {name}: " + "; ".join(f"{k.split('->')[1]} {v[2]:.2f}{'*' if v[3] else ''}" for k, v in rows.items()))

print("\nOFF pairs per run (equal-cell diet / field diet / prey side):")
for name, _, _ in RUNS:
    s = summary[name]
    offs = sorted({k for k, v in s["diet"].items() if v[3]} | {k + " (f)" for k, v in s["diet_field"].items() if v[3]}
                  | {f"{q}->{prey} (prey)" for prey, d in s["prey"].items() for q, v in d.items() if v[3]})
    print(f"  {name:26s} {', '.join(offs)}")
