#!/usr/bin/env python3
"""Print the markdown tables of stock_recruitment.md from the result CSVs.

x_data in Table 2 is the Beverton-Holt S_today/b that reproduces the data-anchored
ratio (R/S at today's SSB)/(R/S at the period's lowest SSB): x = (1 - q)/(q - r) with
r = S_min/S_today; "inf" when q <= r (recruitment flat or falling over the range),
"n/a" when today's SSB is below the period's lowest.

Usage:
    python3 -I scripts/sr_tables_md.py COMP_CSV GROUPS_CSV > tables_fragment.md
    e.g. python3 -I research_notes/Beståndsreglering/scripts/sr_tables_md.py \
             research_notes/Beståndsreglering/tables/sr_compensation.csv \
             research_notes/Beståndsreglering/tables/sr_groups.csv
"""
import csv
import sys

ORDER = [
    ("her.27.3a47d", "plankt."), ("her.27.20-24", "plankt."), ("spr.27.3a4", "plankt."),
    ("nop.27.3a4", "plankt."), ("whb.27.1-91214", "plankt."),
    ("had.27.46a20", "benth."), ("cod.27.47d20", "benth. (old whole-stock SAM, 2022)"),
    ("cod.27.46a7d20_sum", "benth. (sum of 3 substocks)"),
    ("cod.27.46a7d20_viking", "benth. (used)"), ("cod.27.46a7d20_southern", "-"),
    ("cod.27.46a7d20_northwestern", "-"), ("cod.27.21", "- (Kattegat, relative SSB)"),
    ("ple.27.420", "benth."), ("ple.27.21-32", "- (Kattegat-Baltic)"), ("wit.27.3a47d", "benth."),
    ("pok.27.3a46", "pred."), ("mac.27.nea", "pred."), ("hom.27.2a3a4a5b6a7a-ce-k8", "pred. (used, SS3)"),
    ("hom.27.4bc7d", "-"), ("hke.27.3a46-8abd", "pred. (SS3)"), ("anf.27.3a46", "pred. (SS3)"),
    ("dgs.27.nea", "pred. (structural)"), ("whg.27.47d", "whiting"),
]


def f(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except ValueError:
        return x or "-"


def x_floor(q, r):
    """BH x = S_t/b reproducing the data-anchored ratio q = (R/S)(S_t)/(R/S)(S_min),
    r = S_min/S_t: (1 + r x)/(1 + x) = q  ->  x = (1 - q)/(q - r)."""
    if r >= 1:
        return "n/a"
    if q <= r:
        return "inf"
    if q >= 1:
        return "0"
    x = (1 - q) / (q - r)
    return f"{x:.1f}" if x < 100 else "inf"


def main():
    comp = {(r["stock"], r["period"]): r for r in csv.DictReader(open(sys.argv[1]))}
    print("Table 1. Full series (stationary set = prop, const, BH, Ricker, HS; all = + trend + shift).\n")
    print("| Stock | Group | R age | Year classes (n) | Best stationary | Best overall | dAICc prop | dAICc const "
          "| Shift: a_post/a_pre (dAICc) | e stationary [90 %] | e all [90 %] | sigma, AR1 (BH) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st, g in ORDER:
        r = comp.get((st, "full"))
        if not r:
            continue
        sh = (f"{f(r['a_post_over_pre_bh_shift'])} ({f(r['d_aicc_bestnoshift_vs_bestshift'], 1)})"
              if r["a_post_over_pre_bh_shift"] else "-")
        print(f"| {st} | {g} | {r['rec_age']} | {r['yc_first']}-{r['yc_last']} ({r['n']}) | {r['best_noshift']} "
              f"| {r['best_all']} | {f(r['d_aicc_prop_vs_best'], 1)} | {f(r['d_aicc_const_vs_best'], 1)} | {sh} "
              f"| {f(r['elast_avg'])} [{f(r['elast_avg_lo'])}, {f(r['elast_avg_hi'])}] "
              f"| {f(r['elast_avg_all'])} [{f(r['elast_avg_all_lo'])}, {f(r['elast_avg_all_hi'])}] "
              f"| {f(r['sigma_bh'])}, {f(r['ar1_bh'])} |")
    print("\nTable 2. Year classes 2000 on, and compensation at today's SSB.\n")
    print("| Stock | S_today (kt) | S_today/Blim | S_min/S_t - S_max/S_t | Best stationary | Best overall "
          "| dAICc prop | dAICc const | e stationary [90 %] | e all [90 %] | Sbp/S_today (HS) "
          "| (R/S)_today/(R/S)_Smin, all [90 %] | x_data [90 %] | Grade |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st, g in ORDER:
        r = comp.get((st, "post2000"))
        if not r:
            continue
        St = float(r["S_today"])
        kt = f"{St / 1000:.0f}" if St > 1000 else f"{St:.3g} (rel.)"
        sbp = float(r["hs_sbp"]) / St
        flag = f" ({r['hs_flag']} bound)" if r["hs_flag"] else ""
        grade = "D"
        if st.startswith(("hke", "anf", "hom.27.2a")):
            grade = "D (SS3 caveat)"
        if st.startswith("dgs"):
            grade = "J (structural R)"
        rr = float(r["S_min"]) / St
        xd = (f"{x_floor(float(r['ratio_minobs_avg_all']), rr)} [{x_floor(float(r['ratio_minobs_avg_all_hi']), rr)}, "
              f"{x_floor(float(r['ratio_minobs_avg_all_lo']), rr)}]")
        print(f"| {st} | {kt} | {f(r['S_today_over_Blim'])} | {float(r['S_min']) / St:.2f}-{float(r['S_max']) / St:.2f} "
              f"| {r['best_noshift']} | {r['best_all']} | {f(r['d_aicc_prop_vs_best'], 1)} | {f(r['d_aicc_const_vs_best'], 1)} "
              f"| {f(r['elast_avg'])} [{f(r['elast_avg_lo'])}, {f(r['elast_avg_hi'])}] "
              f"| {f(r['elast_avg_all'])} [{f(r['elast_avg_all_lo'])}, {f(r['elast_avg_all_hi'])}] "
              f"| {sbp:.2f}{flag} | {f(r['ratio_minobs_avg_all'])} [{f(r['ratio_minobs_avg_all_lo'])}, "
              f"{f(r['ratio_minobs_avg_all_hi'])}] | {xd} | {grade} |")
    print("\nTable 3. Group elasticity e (members clipped to [-1, 0]); x = -e/(1+e) = A_today/K_R of a "
          "Beverton-Holt term with the same elasticity; 1+e = rho(A_today)/rho(0).\n")
    print("| Group | e post-2000 stationary [90 %] | e post-2000 all [90 %] | e full stationary [90 %] "
          "| e full all [90 %] | x (post-2000 all) | 1+e (post-2000 all) |")
    print("|---|---|---|---|---|---|---|")
    G = [r for r in csv.DictReader(open(sys.argv[2])) if r["member"] == "GROUP"]
    for g in ("planktivorous_fish", "benthivorous_fish", "predatory_fish", "whiting"):
        cells = {}
        for r in G:
            if r["group"] == g:
                cells[(r["period"], r["column"])] = r
        def c(p, k):
            r = cells[(p, k)]
            return f"{f(r['elast'])} [{f(r['elast_lo'])}, {f(r['elast_hi'])}]"
        ra = cells[("post2000", "elast_avg_all")]
        print(f"| {g} | {c('post2000', 'elast_avg')} | {c('post2000', 'elast_avg_all')} | {c('full', 'elast_avg')} "
              f"| {c('full', 'elast_avg_all')} | {ra['x_bh']} | {f(ra['ratio_bh'])} |")


if __name__ == "__main__":
    main()
