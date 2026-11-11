"""Who the fish predators of the benthivores actually take: prey species and stage composition of the
accessible tonnes in the two fish pairs on the benthivores, and the bound a member-specific refuge
(flatfish burial) could put on a whole-group floor.

Usage: python3 -I take_composition.py <repo_root> <scratch_dir>

Method (no new values; the library windows' own machinery):
- research_notes/"Fonsterviktning"/scripts/aw_stage_access.py is exec'd up to its run block, exactly as
  research_notes/"Bottenfiskens hojdprofil"/scripts/wc_benth_window.py does, so prey_dist, lweights,
  inside_share, the windows and the prey length tables are the ones the library stage_access came from.
- Water column -> benthivores: members and shares of the section 269 band-A a (band_pairs.json, variant
  A_band: saithe 0.892, pollack 0.007 on saithe's window, hake 0.051, anglerfish 0.015, ling 0.025 on
  anglerfish's window), saithe's diet identity = haddock + cod, a(L) x L^1.7 (kexp central of the 269
  window). Reproduces the library [0.392, 0.0388].
- Whiting -> benthivores: whiting member, identity all, kexp 0 (Fonsterviktning central). Reproduces
  [0.038, 0.0064].
- Per prey species s and stage S: contribution c_S(s) = sum_m share_m x mean_q k_q(s) inside_q(s) / den_q,S;
  f_S = sum_s c_S(s). Take share of s = (JB c_J(s) + (1 - JB) c_A(s)) / f_eff, JB = 0.15 (box J/B).
- Bound: if a member set M had its own floor x and every other member 1 (one rest share per cell, as the
  engine), the pair-equivalent floor is 1 - w_M (1 - x), w_M = M's share of the take.
"""
import contextlib
import io
import json
import os
import sys

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
RN = os.path.join(ROOT, "research_notes")
AW = os.path.join(RN, "Fönsterviktning", "scripts", "aw_stage_access.py")
src = open(AW).read()
head = src[:src.index('print("a(L) shape check')]
ARGV = list(sys.argv)
sys.argv = [AW, RN, SCR]
NS = {"__name__": "aw_head", "__file__": AW}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, AW, "exec"), NS)
sys.argv = ARGV
sa = NS["sa"]
prey_dist, lweights, inside_share, GROUP_W = NS["prey_dist"], NS["lweights"], NS["inside_share"], NS["GROUP_W"]
UNKEYED = NS["UNKEYED"]
IDENT = {m: ident for m, _, ident in NS["WC_BEN"]}
IDENT["whiting"] = None
JB = 0.15

BAND = json.load(open(os.path.join(RN, "Overlapp i fiskparen", "tables", "band_pairs.json")))["variants"]
mem = BAND["A_band"]["fish"]["predatory_fish->benthivorous_fish"]["members"]
SH = {}
for k, v in mem.items():
    sp = k.split("|")[0]
    SH[sp] = SH.get(sp, 0.0) + v
MAP = {"saithe": "saithe", "pollack": "saithe", "hake": "hake", "anglerfish": "anglerfish", "ling": "anglerfish"}


def decompose(members, kexp):
    """members: [(window member, share)]; returns {stage: {species: contribution}}."""
    out = {"J": {}, "A": {}}
    for stage in ("J", "A"):
        for m, s in members:
            window = sa.WINDOWS[m]
            lw = lweights(m, "ben", "aw", kexp) if window[0] == "ratio" else None
            ident = IDENT[m]
            for q in (1, 2, 3, 4):
                den = 0.0
                num = {}
                for sp, w in GROUP_W["ben"](q).items():
                    dist, share = prey_dist(q, sp, stage)
                    k = w * share
                    if k <= 0:
                        continue
                    den += k
                    if dist is not None and (ident is None or sp in ident):
                        num[sp] = k * inside_share(dist, window, lw)
                for sp, v in num.items():
                    out[stage][sp] = out[stage].get(sp, 0.0) + s * v / den / 4.0
    return out


def report(name, members, kexp, lib):
    d = decompose(members, kexp)
    fj, fa = sum(d["J"].values()), sum(d["A"].values())
    feff = JB * fj + (1 - JB) * fa
    print(f"\n== {name}: [f_J, f_A] = [{fj:.4f}, {fa:.5f}] (library {lib}); f_eff {feff:.5f} at JB {JB}")
    print(f"   stage split of the take: J {JB * fj / feff:.3f}, A {(1 - JB) * fa / feff:.3f}")
    print("   take share by prey species and stage (share of f_eff):")
    rows = []
    for sp in sorted(set(d["J"]) | set(d["A"])):
        j = JB * d["J"].get(sp, 0.0) / feff
        a = (1 - JB) * d["A"].get(sp, 0.0) / feff
        rows.append((sp, j, a))
    rows.sort(key=lambda r: -(r[1] + r[2]))
    for sp, j, a in rows:
        tag = "flatfish" if sp in sa.FLAT - {"grey gurnard", "starry ray"} else (
            "ray/gurnard" if sp in ("grey gurnard", "starry ray") else "gadoid")
        print(f"     {sp:15s} J {j:.3f}  A {a:.3f}  total {j + a:.3f}   [{tag}{', unkeyed (all adult)' if sp in UNKEYED else ''}]")
    flat = sum(j + a for sp, j, a in rows if sp in sa.FLAT - {"grey gurnard", "starry ray"})
    rays = sum(j + a for sp, j, a in rows if sp in ("grey gurnard", "starry ray"))
    gadJ = sum(j for sp, j, a in rows if sp in sa.GADOIDS)
    gadA = sum(a for sp, j, a in rows if sp in sa.GADOIDS)
    print(f"   flatfish {flat:.3f}; rays + gurnard {rays:.3f}; gadoid J {gadJ:.3f}; gadoid A {gadA:.3f}")
    print("   pair-equivalent floor 1 - w_flat (1 - x_flat), other members at 1:")
    for x in (0.0, 0.2, 0.4, 0.6, 0.8):
        print(f"     x_flat {x:.1f}: {1 - flat * (1 - x):.3f}")
    # per-member split of the take (which predator member takes what)
    return dict(fJ=fj, fA=fa, feff=feff, flat=flat, rays=rays, gadJ=gadJ, gadA=gadA,
                rows=[list(r) for r in rows])


res = {}
wc_members = [(MAP[sp], s) for sp, s in SH.items()]
tot = sum(s for _, s in wc_members)
wc_members = [(m, s / tot) for m, s in wc_members]
print("water column band-A member shares:", {k: round(v, 3) for k, v in SH.items()})
res["wc"] = report("water column -> benthivores (269 band-A shares, kexp 1.7)", wc_members, 1.7, [0.392, 0.0388])
print("\n   by predator member (share of the WC take; J / A):")
for sp, s in SH.items():
    d = decompose([(MAP[sp], s / tot)], 1.7)
    fj, fa = sum(d["J"].values()), sum(d["A"].values())
    fl = sum(JB * d["J"].get(x, 0) + (1 - JB) * d["A"].get(x, 0) for x in sa.FLAT - {"grey gurnard", "starry ray"})
    print(f"     {sp:11s} J {JB * fj / res['wc']['feff']:.3f}  A {(1 - JB) * fa / res['wc']['feff']:.3f}"
          f"   of which flatfish {fl / res['wc']['feff']:.3f}")
res["whiting"] = report("whiting -> benthivores (kexp 0)", [("whiting", 1.0)], 0.0, [0.038, 0.0064])

# Sensitivity (not a proposal): hake's in-area diet holds 0.4 % benthivores and no flatfish
# (research_notes/"Fonsterviktning" open point 2); with hake's identity set to haddock + cod the flatfish
# share of the WC take is recomputed. The window itself is not re-derived here.
IDENT["hake"] = sa.GADOIDS
res["wc_hake_gadoids"] = report("SENSITIVITY water column -> benthivores, hake identity haddock + cod", wc_members, 1.7,
                                "(not the library)")
IDENT["hake"] = None

# Booked flows: M2 x B of each fish pair on the benthivores (library), to weight the two pairs
B_BEN = 1.53  # t/km2, benthivore start density (AGENTS.md section 4)
M2 = {"wc": 0.0833, "whiting": 0.1067}
print("\nBooked M2 on the benthivores (library): water column 0.0833, whiting 0.1067 /yr;"
      " flatfish share of the combined fish-predator booking:")
w = {k: M2[k] / sum(M2.values()) for k in M2}
fl = sum(w[k] * res[k]["flat"] for k in w)
print(f"   weights wc {w['wc']:.3f}, whiting {w['whiting']:.3f}; flatfish {fl:.3f}")
res["booked_flat"] = fl
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "take_composition.json"), "w"), indent=1)
