"""Krill ration as a fraction of the GER Cmax, from krill-dominated ICES stomach records (single fish and pools).

Input: research_notes/"Krill over grunda bottnar"/stomachs/tables/records_linked.csv.gz (o-umlaut in the real name).

Method [D]:
- The ICES gadoid gastric evacuation model (WGSAM 2024 Table 3-7; saithe = whiting parameters; cod own):
  R (g/h) = rho e^(delta T) L^lambda E^-0.85 S^0.5 / M, M = prey armament.
- research_notes/"Fiskgrupper i rutan"/predators/scripts/calc_ger_cmax.py defines Cmax with the same model at a
  sustained stomach fill S_max = 2.5 %BW (FB4 cod calibration; 5 %BW the upper bound). At the same L, T, the ratio
  of the ration on a krill stomach to Cmax on 4.184 kJ/g fish prey, in ENERGY, is
      R_krill / Cmax = sqrt(S / S_max) x (E_krill / 4.184)^0.15 / M.
  T, rho, lambda cancel. E_krill 4.0-5.0 kJ/g [J] -> factor 0.99-1.03 (ignored, folded into the band).
  M (krill armament relative to fish) 1.0-1.5 [J]: Tanasichuk et al. 1991 found hake digested euphausiids most
  rapidly of their prey [A], so M <= 1 is possible; 1.5 is a crustacean-shell bound [J].
- S = mean stomach content per stomach in the record (pools: all stomachs, empties included). The krill mass is the
  recorded g_krill when present, else counts x 0.15 / 0.23 g (Meganyctiphanes adults; the three weighed saithe pools
  give 0.17-0.19 g per krill [D]).
- The pool mean of S over all stomachs (empties included) is a lower bound on S of the feeding fish; sqrt(mean S)
  >= mean sqrt(S) (Jensen) overstates the ration slightly when S varies among fish. Pools of 10-50 fish damp both.
- Length: pooled records of 1980-82, 1987, 1991 give the lower bound of a length class: saithe 10-cm classes
  (values 30, 40, 50, 70) -> midpoint L + 5; the other species 5-cm classes -> L + 2.5 [J]. Records without a
  length are skipped (three weighed saithe pools of 1982 among them).
- Krill mass for small whiting (< 20 cm) is overstated if they ate juvenile krill or Thysanoessa (~0.05 g) [J]:
  read whiting from 20 cm only.
- Krill-dominated: krill >= 0.8 of the counted prey items. Also reported: weighed vs counted records, the weighed
  saithe pools without a length (bracketed 35-55 cm), and shares on the model-ceiling scale (GER ratio / 1.2 whiting,
  / 1.23 saithe; other species / 1.2 [J]).

Usage: python3 -I krill_pool_rations.py <records_linked.csv.gz>
"""
import csv, gzip, math, sys
LW = {"whiting": (0.0058, 3.10), "saithe": (0.0089, 3.05), "norway pout": (0.0065, 3.17),
      "haddock": (0.0083, 3.028), "cod": (0.0076, 3.08), "hake": (0.0060, 3.10)}
SMAX = (0.025, 0.05)
M = (1.0, 1.5)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


out = []
nolen = []
CEIL = {"whiting": 1.2, "saithe": 1.23, "norway pout": 1.2, "cod": 1.2, "haddock": 1.2, "hake": 1.2}  # model ceiling / GER Cmax; non-model species use the whiting factor [J]
with gzip.open(sys.argv[1], "rt") as fh:
    for r in csv.DictReader(fh):
        sp = r["sp"].strip().lower()
        if sp not in LW:
            continue
        kc, allc, gk, N = f(r["krill_count"]), f(r["all_count"]), f(r["g_krill"]), f(r["Number"]) or 1.0
        if not (kc or gk):
            continue
        share = (kc / allc) if (kc and allc) else None
        if share is None or share < 0.8:
            continue
        L0 = f(r["L_cm"])
        if not L0:
            if sp == "saithe" and gk:
                for Lb in (35.0, 45.0, 55.0):
                    Wb = LW[sp][0] * Lb ** LW[sp][1]
                    Sg = gk / N
                    nolen.append((r["year"], r["month"], N, Sg, Lb, 100 * Sg / Wb, math.sqrt(Sg / (0.05 * Wb)) / 1.5, math.sqrt(Sg / (0.025 * Wb))))
            continue
        classed = r["year"] in ("1981", "1982", "1980", "1987", "1991") and abs(L0 - round(L0 / 5) * 5) < 1e-6 and N > 1
        L = L0 + (5.0 if sp == "saithe" else 2.5) if classed else L0
        a, b = LW[sp]
        W = a * L ** b
        if gk:
            S = (gk / N, gk / N)
        else:
            S = (kc * 0.15 / N, kc * 0.23 / N)
        s_pct = tuple(100 * s / W for s in S)
        lo = math.sqrt(S[0] / (SMAX[1] * W)) / M[1]
        hi = math.sqrt(S[1] / (SMAX[0] * W)) / M[0]
        cen = math.sqrt(0.5 * (S[0] + S[1]) / (SMAX[0] * W)) / 1.0
        out.append((sp, r["year"], r["month"], L, W, N, f(r["fed"]), s_pct, lo, cen, hi, r["bed_true"] or r["bed"], r["area"], r["lat"], r["lon"], r["depth"], bool(gk)))

print("Krill-dominated records (krill >= 0.8 of counted prey): S = mean krill content per stomach (%BW, krill 0.15-0.23 g"
      " or weighed); ration / GER Cmax = sqrt(S/S_max)/M, low (S_max 5 %, M 1.5) / central (S_max 2.5 %, M 1) / high")
for sp in LW:
    rows = sorted((o for o in out if o[0] == sp), key=lambda o: -o[9])
    if not rows:
        continue
    print(f"== {sp}: {len(rows)} records")
    for o in rows[:15]:
        print(f"  {o[1]}-{o[2]:>2s} L {o[3]:5.1f} W {o[4]:6.0f} N {o[5]:3.0f} fed {o[6] if o[6] is not None else float('nan'):4.0f}  S {o[7][0]:5.2f}-{o[7][1]:5.2f} %BW"
              f"  ratio {o[8]:.2f} / {o[9]:.2f} / {o[10]:.2f}   bed {o[11]} {o[12]} ({o[13]},{o[14]}) depth {o[15]}")
    cens = sorted(o[9] for o in rows)
    wN = sum(o[5] for o in rows)
    wmean = sum(o[9] * o[5] for o in rows) / wN
    if sp == "whiting":
        big = [o for o in rows if o[3] >= 20]
        wb = sum(o[5] for o in big)
        cb = sorted(o[9] for o in big)
        print(f"  whiting >= 20 cm: {len(big)} records, {wb:.0f} stomachs; stomach-weighted mean central "
              f"{sum(o[9] * o[5] for o in big) / wb:.2f}, low {sum(o[8] * o[5] for o in big) / wb:.2f}; record median {cb[len(cb)//2]:.2f};"
              f" share >= 0.5 {sum(c >= 0.5 for c in cb)/len(cb):.2f}, >= 1 {sum(c >= 1 for c in cb)/len(cb):.2f}")
    for label, sub in (("weighed only", [o for o in rows if o[16]]), ("counted only", [o for o in rows if not o[16]])):
        if sub:
            ws = sum(o[5] for o in sub)
            print(f"  {label}: {len(sub)} records, {ws:.0f} stomachs, stomach-weighted mean central {sum(o[9] * o[5] for o in sub) / ws:.2f}, low {sum(o[8] * o[5] for o in sub) / ws:.2f}")
    base = [o for o in rows if not (sp == "whiting" and o[3] < 20)]
    wsb = sum(o[5] for o in base)
    for thr in (0.25, 0.5):
        sl = sum(o[5] for o in base if o[8] / CEIL[sp] > thr) / wsb
        sc = sum(o[5] for o in base if o[9] / CEIL[sp] > thr) / wsb
        print(f"  model-ceiling scale (/ {CEIL[sp]}): share of stomachs above {thr}: low band {sl:.2f}, central {sc:.2f}"
              f"{' (whiting >= 20 cm)' if sp == 'whiting' else ''}")
    print(f"  model-ceiling scale stomach-weighted mean: low {sum(o[8] * o[5] for o in base) / wsb / CEIL[sp]:.2f}, central {sum(o[9] * o[5] for o in base) / wsb / CEIL[sp]:.2f}")
    print(f"  stomach-weighted mean low {sum(o[8] * o[5] for o in rows) / wN:.2f};")
    print(f"  stomach-weighted mean central ratio {wmean:.2f} over {wN:.0f} stomachs; record median {cens[len(cens)//2]:.2f}; "
          f"share of records with central >= 0.5: {sum(c >= 0.5 for c in cens)/len(cens):.2f}, >= 1: {sum(c >= 1 for c in cens)/len(cens):.2f}")
print()
print("Weighed saithe pools without a length (bracketed at 35 / 45 / 55 cm): year-month, N, krill g per stomach, L, S %BW, ratio low / central (GER Cmax)")
for o in nolen:
    print(f"  {o[0]}-{o[1]:>2s} N {o[2]:3.0f} krill {o[3]:5.1f} g  L {o[4]:.0f}  S {o[5]:5.2f} %BW  ratio {o[6]:.2f} / {o[7]:.2f}  (model scale / 1.23: {o[6]/1.23:.2f} / {o[7]/1.23:.2f})")
