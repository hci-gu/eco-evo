"""Nephrops size window for fish predation (part B, Q2). All outputs are derivations (J).

Inputs (passed as arguments, never imported):
  1. tables/nep34_catch_lfd_2023_read.csv  (digitised WGNSSK 2024 Fig. 10.1.4, thousands per 1 mm CL)
  2. tables/declerck_sterk1980_table5_transcribed.csv
Usage: python3 -I nephrops_size_window.py <lfd_csv> <declerck_csv>

Parameters (sources in tables/part_B_size_recolonisation.txt):
  ICES stock annex nep.27.3a (FU 3-4): males Linf 73 mm, K 0.14 (WGNSSK 2024: 0.138); mature females
  Linf 65, K 0.10; immature females as males (WGNSSK 2024); female L50 maturity 28.5-29 mm;
  W(g) = a CL(mm)^b, males a 0.00045 b 3.113, females a 0.00108 b 2.847;
  M 0.3 males and immature females, 0.2 mature females.
  Settlement size: first postlarva CL 3.08 mm (Smith 1987 thesis, Table 4.1a).
  UWTV floor 17 mm CL (stock annex: knife-edge selectivity at 17 mm).
  Harvest rate (removals / UWTV abundance, % by number) 2021-2023: 5.1, 4.7, 4.2 (WGNSSK 2024 table).
  Armstrong et al. 1991 (ICES CM 1991/G:15) Table 11 cod-predation selectivity by prawn pseudo age,
  mapped to CL with their male length-at-age (1:10, 2:17, 3:24, 4:30, 5:35, 6:39, 7:41 mm).
"""
import csv
import math
import sys

LINF_M, K_M = 73.0, 0.138
LINF_F_MAT, K_F_MAT = 65.0, 0.10
L50_F = 29.0
A_M, B_M = 0.00045, 3.113
A_F, B_F = 0.00108, 2.847
L_SETTLE = 3.08
L_UWTV = 17.0
DL = 0.05


def w_m(L):
    return A_M * L ** B_M


def w_f(L):
    return A_F * L ** B_F


def cl_from_w(w, a, b):
    return (w / a) ** (1.0 / b)


def years(L1, L2, linf, k):
    return math.log((linf - L1) / (linf - L2)) / k


# ---------------------------------------------------------------- A. De Clerck & Sterk prey CL
def declerck(path):
    mids = [8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 56]
    rows = [r for r in csv.reader(l for l in open(path) if not l.startswith("#"))]
    hdr, rows = rows[0], rows[1:]
    print("\nA. De Clerck & Sterk 1980 Table 5: prey CL (class mid-points) per predator length class")
    print("season,predator,pred_len_cm,n,median_CL,p90_CL,max_CL_class_mid,share_CL_le_30")
    pooled = {}
    for r in rows:
        counts = [int(x) for x in r[3:18]]
        n = sum(counts)
        if n == 0:
            continue
        vals = sorted(sum(([m] * c for m, c in zip(mids, counts)), []))
        med = vals[(n - 1) // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        p90 = vals[min(n - 1, int(math.ceil(0.9 * n)) - 1)]
        le30 = sum(c for m, c in zip(mids, counts) if m <= 29) / n
        print(f"{r[0]},{r[1]},{r[2]},{n},{med},{p90},{max(vals)},{le30:.2f}")
        lo = int(r[2].split("-")[0])
        grp = "26-45" if lo < 46 else ("46-65" if lo < 66 else "66-105")
        key = (r[0], r[1], grp)
        pooled.setdefault(key, [0] * 15)
        pooled[key] = [p + c for p, c in zip(pooled[key], counts)]
    print("pooled groups: season,predator,group,n,median,p90,max,share_le_30,share_le_39,share_ge_40")
    for key, counts in pooled.items():
        n = sum(counts)
        vals = sorted(sum(([m] * c for m, c in zip(mids, counts)), []))
        med = vals[(n - 1) // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        p90 = vals[min(n - 1, int(math.ceil(0.9 * n)) - 1)]
        s30 = sum(c for m, c in zip(mids, counts) if m <= 29) / n
        s39 = sum(c for m, c in zip(mids, counts) if m <= 38) / n
        print(f"{key[0]},{key[1]},{key[2]},{n},{med},{p90},{max(vals)},{s30:.2f},{s39:.2f},{1 - s39:.2f}")


# ---------------------------------------------------------------- B. Dorman 1992 mean prey weight -> CL
def dorman():
    print("\nB. Dorman et al. 1992 (IYFS Feb 1991): mean weight per Nephrops in cod stomachs -> CL via annex L-W")
    print("   (stomach weights are partly digested -> CL is a LOWER bound on the live size)")
    data = {"Skagerrak": [(350, 11.1, 1), (400, 6.4, 3), (500, 11.2, 1), (600, 70.0, 6), (700, 59.4, 5)],
            "Kattegat": [(150, 2.2, 1), (200, 2.5, 1), (350, 5.7, 2), (400, 37.0, 4), (500, 52.3, 4),
                         (600, 41.6, 2), (700, 42.6, 3)]}
    print("area,cod_class_mm,n,mean_w_g,CL_if_male_mm,CL_if_female_mm")
    for area, rows in data.items():
        for c, w, n in rows:
            mw = w / n
            print(f"{area},{c},{n},{mw:.2f},{cl_from_w(mw, A_M, B_M):.1f},{cl_from_w(mw, A_F, B_F):.1f}")
        tw = sum(r[1] for r in rows)
        tn = sum(r[2] for r in rows)
        print(f"{area},all,{tn},{tw / tn:.2f},{cl_from_w(tw / tn, A_M, B_M):.1f},{cl_from_w(tw / tn, A_F, B_F):.1f}")


# ---------------------------------------------------------------- C. catch LFD shares
def read_lfd(path):
    rows = list(csv.DictReader(open(path)))
    return {int(r["CL_mm"]): float(r["catch_thousands"]) for r in rows}


def catch_shares(lfd):
    print("\nC. Catch LFD 2023 Division 3.a (gear-selected, both sexes): shares of numbers and weight")
    tot_n = sum(lfd.values())
    wmean = lambda L: 0.5 * (w_m(L) + w_f(L))
    tot_w = sum(n * wmean(L) for L, n in lfd.items())
    print(f"   total catch numbers read off figure: {tot_n / 1000:.1f} million")
    print(f"   total catch weight (sex-mean L-W): {tot_w / 1e3:.0f} t")
    for X in (25, 30, 35, 40, 45, 50):
        sn = sum(n for L, n in lfd.items() if L <= X) / tot_n
        sw = sum(n * wmean(L) for L, n in lfd.items() if L <= X) / tot_w
        print(f"   CL <= {X} mm: numbers {sn:.3f}, weight {sw:.3f}")


# ---------------------------------------------------------------- D. equilibrium population
def grow_params(L, sex):
    if sex == "m" or L < L50_F:
        return LINF_M, K_M, 0.3, w_m(L) if sex == "m" else w_f(L)
    return LINF_F_MAT, K_F_MAT, 0.2, w_f(L)


def population(sel, F, sex, M_juv=None):
    """Steady state with constant recruitment at L_SETTLE. Returns dict L_bin(1 mm) -> (numbers, biomass)."""
    out = {}
    N = 1.0
    L = L_SETTLE
    while True:
        linf, k, M, w = grow_params(L, sex)
        if L + DL >= linf - 1.0:
            break
        if M_juv is not None and L < L_UWTV:
            M = M_juv
        dtdl = 1.0 / (k * (linf - L))
        Z = M + F * sel(L)
        dens = N * dtdl * DL  # standing numbers in [L, L+DL)
        b = int(math.floor(L + 0.5))
        n0, w0 = out.get(b, (0.0, 0.0))
        out[b] = (n0 + dens, w0 + dens * w)
        N *= math.exp(-Z * dtdl * DL)
        L += DL
    return out


def combined(sel, F, M_juv=None):
    pm = population(sel, F, "m", M_juv)
    pf = population(sel, F, "f", M_juv)
    keys = sorted(set(pm) | set(pf))
    return {k: (pm.get(k, (0, 0))[0] + pf.get(k, (0, 0))[0], pm.get(k, (0, 0))[1] + pf.get(k, (0, 0))[1])
            for k in keys}


def fit_selectivity(lfd, harvest_rate, M_juv=None, iters=60):
    """Gear selectivity s(L) proportional to C(L)/N(L), normalised to max 1 (smoothed tail),
    F scaled so that sum(F s N)/sum(N, L>=17) = harvest_rate (instantaneous approximation)."""
    s = {L: 1.0 for L in range(0, 80)}
    F = 0.05
    for _ in range(iters):
        sel = lambda L: s.get(int(math.floor(L + 0.5)), 0.0)
        pop = combined(sel, F, M_juv)
        raw = {}
        for L in range(0, 80):
            n = pop.get(L, (0, 0))[0]
            c = lfd.get(L, 0.0)
            raw[L] = c / n if n > 0 else 0.0
        # above 50 mm the read-off counts are small and noisy: hold s at its 45-50 mm mean
        top = max(raw.values())
        s = {L: raw[L] / top for L in raw}
        plateau = sum(s[L] for L in range(45, 51)) / 6.0
        for L in range(51, 80):
            s[L] = plateau
        m = max(s.values())
        s = {L: v / m for L, v in s.items()}
        sel = lambda L: s.get(int(math.floor(L + 0.5)), 0.0)
        pop = combined(sel, F, M_juv)
        n17 = sum(v[0] for L, v in pop.items() if L >= L_UWTV)
        caught = sum(F * sel(L) * v[0] for L, v in pop.items())
        F *= harvest_rate / (caught / n17)
    return s, F, combined(lambda L: s.get(int(math.floor(L + 0.5)), 0.0), F, M_juv)


ARM_SEL = {  # Armstrong et al. 1991 Table 11; prawn pseudo-age t spans CL at age t .. t+1 (males)
    "cod_age4": [(10, 17, 0.06), (17, 24, 0.58), (24, 30, 0.94), (30, 35, 1.00), (35, 39, 0.83),
                 (39, 41, 0.68), (41, 99, 0.62)],
    "cod_age6plus": [(10, 17, 0.03), (17, 24, 0.41), (24, 30, 0.79), (30, 35, 1.00), (35, 39, 0.64),
                     (39, 41, 0.52), (41, 99, 0.48)],
    "cod_age2": [(10, 17, 0.28), (17, 24, 1.00), (24, 30, 1.00), (30, 35, 0.80), (35, 39, 0.04),
                 (39, 99, 0.0)],
}


def arm_sel(name, L):
    for lo, hi, v in ARM_SEL[name]:
        if lo <= L < hi:
            return v
    return 0.0


def report_population(lfd):
    print("\nD. Equilibrium population (constant recruitment at 3.08 mm, VBGF + M from annex, gear")
    print("   selectivity from the 2023 catch LFD, F scaled to the harvest rate). 50:50 sex ratio at settlement.")
    for hr in (0.042, 0.047, 0.051):
        for mj in (None, 0.6):
            s, F, pop = fit_selectivity(lfd, hr, mj)
            tag = f"harvest {hr:.3f}, M(<17 mm) {'0.3 (as adults)' if mj is None else mj}"
            n17 = sum(v[0] for L, v in pop.items() if L >= L_UWTV)
            b17 = sum(v[1] for L, v in pop.items() if L >= L_UWTV)
            ball = sum(v[1] for v in pop.values())
            mean_w17 = b17 / n17
            print(f"  [{tag}] F_full {F:.3f}; mean weight of animals >= 17 mm {mean_w17:.1f} g;"
                  f" biomass < 17 mm / biomass >= 17 mm = {(ball - b17) / b17:.3f}")
            line = "   biomass share of the >=17 mm stock with CL <= X:"
            for X in (25, 30, 35, 40, 45):
                sh = sum(v[1] for L, v in pop.items() if L_UWTV <= L <= X) / b17
                line += f" {X}:{sh:.2f}"
            print(line)
            line = "   numbers share of the >=17 mm stock with CL <= X:"
            for X in (25, 30, 35, 40, 45):
                sh = sum(v[0] for L, v in pop.items() if L_UWTV <= L <= X) / n17
                line += f" {X}:{sh:.2f}"
            print(line)
            for name in ARM_SEL:
                eff = sum(v[1] * arm_sel(name, L) for L, v in pop.items() if L >= L_UWTV) / b17
                effall = sum(v[1] * arm_sel(name, L) for L, v in pop.items()) / ball
                print(f"   Armstrong-selectivity-weighted biomass share ({name}): {eff:.2f} of >=17 mm,"
                      f" {effall:.2f} of all post-settlement biomass")
            if hr == 0.047 and mj is None:
                print("   model vs observed catch shape (relative, CL: model_catch_share obs_share):")
                sel = lambda L: s.get(int(math.floor(L + 0.5)), 0.0)
                cm = {L: F * sel(L) * v[0] for L, v in pop.items()}
                tc, to = sum(cm.values()), sum(lfd.values())
                print("   " + " ".join(f"{L}:{cm.get(L, 0) / tc:.3f}/{lfd.get(L, 0) / to:.3f}"
                                       for L in range(20, 66, 5)))
                print("   gear selectivity s(L): " + " ".join(f"{L}:{s[L]:.2f}" for L in range(20, 56, 5)))


# ---------------------------------------------------------------- E. years in windows
def report_years():
    print("\nE. Years to grow through CL windows (VBGF, t = ln((Linf-L1)/(Linf-L2))/K)")
    for (a, b) in [(3.08, 17), (10, 17), (17, 30), (17, 35), (17, 40), (17, 45), (30, 40), (40, 50)]:
        tm = years(a, b, LINF_M, K_M)
        if b <= L50_F:
            tf = tm
        elif a >= L50_F:
            tf = years(a, b, LINF_F_MAT, K_F_MAT)
        else:
            tf = years(a, L50_F, LINF_M, K_M) + years(L50_F, b, LINF_F_MAT, K_F_MAT)
        print(f"   {a:>5} -> {b:>3} mm: males {tm:.1f} yr, females {tf:.1f} yr")
    print("   Armstrong et al. 1991 Irish Sea length-at-age (independent, found): 1:10 2:17 3:24 4:30(m)/27(f)"
          " 5:35/30 6:39/32 7:41/34 mm -> 17->30 mm males 2 yr; 17->40 mm males ~4.5 yr")


if __name__ == "__main__":
    lfd = read_lfd(sys.argv[1])
    declerck(sys.argv[2])
    dorman()
    catch_shares(lfd)
    report_population(lfd)
    report_years()
