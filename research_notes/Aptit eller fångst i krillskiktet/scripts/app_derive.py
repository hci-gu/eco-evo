"""Derived numbers for lit_appetite.md, read from tables/appetite_raw.csv (no other input).

Usage: python3 -I app_derive.py <appetite_raw.csv>
"""
import csv, math, sys

def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    def val(src, region, season_start, method_start, quantity):
        hits = [r for r in rows if r["source"] == src and r["region"].startswith(region)
                and r["season"].startswith(season_start) and r["method"].startswith(method_start)
                and r["quantity"] == quantity]
        if len(hits) != 1:
            raise SystemExit(f"lookup not unique: {src} {region} {season_start} {method_start} {quantity}: {len(hits)}")
        return float(hits[0]["value"])

    print("== Uglem 2020, Hitra: what the 'pellet weight' column is")
    # minimum SD of an all-fish mean (zeros included) at prevalence p: sqrt(p m^2 - (p m)^2), m = mean/p
    for season, gear, n_lab in [("autumn", "FA, jig", "autumn jig"), ("autumn", "FA, gillnet", "autumn net"),
                                ("spring", "FA, jig", "spring jig"), ("spring", "FA, gillnet", "spring net"),
                                ("summer", "FA, jig", "summer jig"), ("summer", "FA, gillnet", "summer net")]:
        p = val("Uglem2020", "Hitra", season, gear, "pellet prevalence") / 100
        mean = val("Uglem2020", "Hitra", season, gear, "pellet mass in stomach (fish with pellets)")
        m_pos = mean / p
        sd_min = math.sqrt(p * m_pos ** 2 - (p * m_pos) ** 2)
        print(f"  {n_lab:11s} p {p:.2f} mean {mean:5.0f} g -> if all-fish mean, SD >= {sd_min:5.0f} g")
    print("  reported SDs: autumn jig 77, autumn net 163, spring jig 106, spring net 63, summer jig 87, summer net 112")
    m6 = [val("Uglem2020", "Hitra", s, g, "pellet mass in stomach (fish with pellets)")
          for s in ("autumn", "spring", "summer") for g in ("FA, jig", "FA, gillnet")]
    print(f"  mean of the six occasion means {sum(m6)/6:.1f} g (text: 105 g)")

    print("== Uglem 2020, Hitra: pellet meal as % of mean FA body weight (W from Table 1)")
    W = {("autumn", "FA, jig"): 2.14, ("autumn", "FA, gillnet"): 3.1, ("spring", "FA, jig"): 3.09,
         ("spring", "FA, gillnet"): 3.19, ("summer", "FA, jig"): 3.18, ("summer", "FA, gillnet"): 2.92}
    for (s, g), w in W.items():
        m = val("Uglem2020", "Hitra", s, g, "pellet mass in stomach (fish with pellets)")
        print(f"  {s:6s} {g:12s} {m:5.0f} g / {w:.2f} kg = {100*m/(1000*w):.1f} % BW")

    print("== Uglem 2020, Hitra: HSI FA/UA within gear")
    for s, gu, gf in [("autumn", "UA 5-10 km from farms, jig", "FA <500 m from farm, jig"),
                      ("autumn", "UA, gillnet", "FA, gillnet"),
                      ("spring", "UA, gillnet", "FA, gillnet"),
                      ("summer", "UA, jig", "FA, jig"),
                      ("summer", "UA, gillnet", "FA, gillnet")]:
        u = val("Uglem2020", "Hitra", s, gu, "HSI UA")
        f = val("Uglem2020", "Hitra", s, gf, "HSI FA")
        print(f"  {s:6s} {gf.split(',')[-1].strip():8s} FA {f:4.1f} / UA {u:4.1f} = {f/u:.2f}")

    print("== Hoines et al. 1992, Karmoy: haddock and cod stomach contents")
    hs = val("Hoines1992", "Karmoy", "Mar-Apr", "stomachs, >30 cm, all", "mean stomach content")
    ho = val("Hoines1992", "Karmoy", "Oct, Dec", "stomachs, >30 cm, all", "mean stomach content")
    print(f"  haddock >30 cm herring season / outside = {hs:.2f} / {ho:.2f} = {hs/ho:.1f}")
    e_in = val("Hoines1992", "Karmoy", "Mar-Apr", "stomachs, >30 cm, n 496", "empty stomachs")
    e_out = val("Hoines1992", "Karmoy", "outside", "stomachs, >30 cm, n 470", "empty stomachs")
    print(f"  haddock >30 cm empty share: herring season {100*e_in/496:.1f} %, outside {100*e_out/470:.1f} %")
    print("  (Feb-based ratios below are NOT used: Feb values lie inside the +-0.5 g read-off error)")
    for size in ("40-49 cm", ">50 cm"):
        f = val("Hoines1992", "Karmoy", "Feb", f"Fig 7 read off, {size}", "mean stomach content")
        m = val("Hoines1992", "Karmoy", "Mar", f"Fig 7 read off, {size}", "mean stomach content")
        a = val("Hoines1992", "Karmoy", "Apr", f"Fig 7 read off, {size}", "mean stomach content")
        print(f"  haddock {size:8s} Feb {f:.1f} g -> Mar {m:.1f} g (x{m/f:.0f}) -> Apr {a:.1f} g (x{a/f:.0f}, same gear as Feb)")
    for size in ("30-49 cm", "50-69 cm"):
        m = val("Hoines1992", "Karmoy", "Mar", f"Fig 8 read off, {size}", "mean stomach content")
        j = val("Hoines1992", "Karmoy", "Jul", f"Fig 8 read off, {size}", "mean stomach content")
        print(f"  cod {size} Mar / Jul = {m:.0f} / {j:.0f} = {m/j:.1f}")

    print("== Nedreaas 1985, Taelavaag: stomach content of 30-39 cm saithe, March vs summer (Fig 5 read off)")
    mar = [val("Nedreaas1985", "Taelavaag", "1 Mar 1983", f"Fig 5 read off, {lg}", "mean stomach content")
           for lg in ("30-34 cm", "35-39 cm")]
    summ = [float(r["value"]) for r in rows if r["source"] == "Nedreaas1985" and r["season"].startswith("Jun-Sep")
            and r["quantity"] == "mean stomach content"]
    w = 0.0085 * 34.5 ** 3  # g, the paper's W = 0.0085 L^3 at the 30-39 cm midpoint
    print(f"  March {min(mar):.1f}-{max(mar):.1f} g; Jun-Sep occasion means {min(summ):.1f}-{max(summ):.1f} g")
    print(f"  at W = 0.0085 L^3, L 34.5 cm: {w:.0f} g; March content = {100*min(mar)/w:.2f}-{100*max(mar)/w:.2f} % BW wet")

    print("== Fordham/Trippel etc.: no numbers derived (secondary)")

if __name__ == "__main__":
    main()
