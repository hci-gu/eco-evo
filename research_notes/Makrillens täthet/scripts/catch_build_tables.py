"""Build the 3.a mackerel / horse mackerel catch tables.

Inputs (read as data only):
  ICES_CSV   ICES Official Nominal Catches 2006-2024 (ICESCatchDataset2006-2024.csv)
  NO_DIR     folder with no_mac_hom_<year>.csv written by catch_no_extract.py
  OUT_DIR    tables folder

Outputs (OUT_DIR):
  catch_3a_by_country_year.csv   long table: source, species, area, country, year, tonnes, flag
  catch_3a_means.csv             per species x area x country: means over 2015-2024, 2016-2024,
                                 2020-2024 (ICES) and 2015-2025 (Norwegian sales notes)
  (Norwegian series: final sales notes 'Sluttseddeldokument' only, Norwegian-flag vessels
  only, main area 09 / area group 27_3_A.)
  catch_3a_by_month_or_quarter.csv  Norwegian monthly catch (month of last catch date) in
                                 Norwegian main area 09 Skagerrak (= ICES 27.3.a, NK), by
                                 species x year x month x gear group; plus WGWIDE western
                                 horse mackerel quarterly 3.a catches 2023-2025 (typed from
                                 the reports, see notes)
  catch_no_3a_month_profile.csv  monthly and quarterly shares of the Norwegian 3.a catch,
                                 pooled over year windows
  catch_no_3a_by_gear_year.csv   Norwegian 3.a catch by year x gear (detailed) x species

Usage: python3 -I catch_build_tables.py ICES_CSV NO_DIR OUT_DIR
"""
import csv
import glob
import os
import sys
from collections import defaultdict

SPECIES = ("MAC", "HOM", "JAX")


def parse_val(s):
    s = s.strip()
    flag = ""
    if s.endswith("c"):
        flag = "c"
        s = s[:-1].strip()
    try:
        return float(s), flag
    except ValueError:
        return None, s


def main(ices_csv, no_dir, out_dir):
    long_rows = []
    with open(ices_csv, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        years = hdr[4:]
        for row in rd:
            sp, area, ctry = row[0], row[1], row[2]
            if sp not in SPECIES or not (area == "27.3.a" or area.startswith("27.3.a.")
                                         or area == "27.3.a_NK"):
                continue
            for y, v in zip(years, row[4:]):
                val, flag = parse_val(v)
                long_rows.append(["ICES_ONC_2006-2024", sp, area, ctry, int(y), val, flag])

    # Norwegian sales notes (final notes only; landing notes can double-count, see doc)
    no = []
    for fp in sorted(glob.glob(os.path.join(no_dir, "no_mac_hom_*.csv"))):
        with open(fp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["document_type"] != "Sluttseddeldokument":
                    continue
                if not (r["area_group"].startswith("27_3_A") or r["main_area"] == "09"):
                    continue
                if r["vessel_nationality"] != "NOR":  # foreign vessels landing in Norway
                    continue
                r["t"] = float(r["round_weight_kg"]) / 1000.0
                no.append(r)
    no_year = defaultdict(float)
    for r in no:
        no_year[(r["species_fao"], int(r["year"]))] += r["t"]
    for sp in ("MAC", "HOM"):  # a scanned year without rows is a true zero
        for y in range(2010, 2027):
            no_year[(sp, y)] += 0.0
    for (sp, y), t in sorted(no_year.items()):
        flag = "partial_year_to_2026-10-30" if y == 2026 else ""
        long_rows.append(["NO_Fiskeridir_sales_notes", sp, "27.3.a (main area 09)", "NO", y,
                          round(t, 3), flag])

    long_rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))
    with open(os.path.join(out_dir, "catch_3a_by_country_year.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "species_fao", "area", "country", "year", "tonnes", "flag"])
        w.writerows(long_rows)

    # Means
    def mean_of(src, sp, area, ctry, y0, y1):
        vals = {r[4]: r[5] for r in long_rows if r[0] == src and r[1] == sp and r[2] == area
                and r[3] == ctry and y0 <= r[4] <= y1}
        xs = [vals.get(y) for y in range(y0, y1 + 1)]
        if any(x is None for x in xs):
            return ""
        return round(sum(xs) / len(xs), 2)

    keys = sorted({(r[0], r[1], r[2], r[3]) for r in long_rows})
    with open(os.path.join(out_dir, "catch_3a_means.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "species_fao", "area", "country", "mean_2015_2024",
                    "mean_2016_2024", "mean_2020_2024", "mean_2015_2025"])
        for src, sp, area, ctry in keys:
            # subdivisions (and _NK) are only reported from 2016; a 2015 mean would
            # average in a structural zero
            sub = area != "27.3.a" and src.startswith("ICES")
            m1 = "" if sub else mean_of(src, sp, area, ctry, 2015, 2024)
            m2 = mean_of(src, sp, area, ctry, 2016, 2024)
            m3 = mean_of(src, sp, area, ctry, 2020, 2024)
            m4 = mean_of(src, sp, area, ctry, 2015, 2025) if src.startswith("NO_") else ""
            if all(x in ("", 0, 0.0) for x in (m1, m2, m3, m4)):
                continue
            w.writerow([src, sp, area, ctry, m1, m2, m3, m4])
        # Totals across countries (ICES), per subdivision level
        for sp in SPECIES:
            for area in ("27.3.a", "27.3.a.20", "27.3.a.21"):
                tot = defaultdict(float)
                for r in long_rows:
                    if r[0].startswith("ICES") and r[1] == sp and r[2] == area and r[5]:
                        tot[r[4]] += r[5]
                row = []
                for y0, y1 in ((2015, 2024), (2016, 2024), (2020, 2024)):
                    if area != "27.3.a" and y0 == 2015:
                        row.append("")
                        continue
                    row.append(round(sum(tot[y] for y in range(y0, y1 + 1)) / (y1 - y0 + 1), 2))
                w.writerow(["ICES_ONC_2006-2024", sp, area, "ALL"] + row + [""])

    # Derived subdivision split per year (2016-2024: DK, SE, DE report 3.a.20/3.a.21).
    # Norway, Faroes, Netherlands report only 27.3.a(_NK); their catch is put in 3.a.20
    # (judgement: Norwegian main area 09 is the Norwegian Skagerrak; see notes).
    by = defaultdict(float)
    for r in long_rows:
        if not r[0].startswith("ICES") or r[5] is None:
            continue
        sp, area, ctry, y, t = r[1], r[2], r[3], r[4], r[5]
        if area == "27.3.a.20":
            by[(sp, y, "3.a.20_reported")] += t
        elif area == "27.3.a.21":
            by[(sp, y, "3.a.21_reported")] += t
        elif area == "27.3.a_NK":
            by[(sp, y, "3.a_NK")] += t
        elif area == "27.3.a":
            by[(sp, y, "3.a_total")] += t
    with open(os.path.join(out_dir, "catch_3a_subdiv_estimate.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species_fao", "year", "3a_total", "3a20_reported", "3a21_reported",
                    "3a_NK", "3a20_est_incl_NK", "3a21_est", "share_3a21"])
        for sp in ("MAC", "HOM"):
            for y in range(2016, 2025):
                tot = by[(sp, y, "3.a_total")]
                a20 = by[(sp, y, "3.a.20_reported")]
                a21 = by[(sp, y, "3.a.21_reported")]
                nk = by[(sp, y, "3.a_NK")]
                w.writerow([sp, y, round(tot, 2), round(a20, 2), round(a21, 2), round(nk, 2),
                            round(a20 + nk, 2), round(a21, 2),
                            round(a21 / tot, 3) if tot else ""])

    # Monthly table
    mon = defaultdict(float)
    for r in no:
        mon[(r["species_fao"], int(r["year"]), int(r["catch_month"]), r["gear_group"])] += r["t"]
    with open(os.path.join(out_dir, "catch_3a_by_month_or_quarter.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "species_fao", "area", "country", "year", "period_type",
                    "period", "gear_group", "tonnes", "note"])
        for (sp, y, m, g), t in sorted(mon.items()):
            note = "month 0 = no catch date" if m == 0 else ""
            if y == 2026:
                note = (note + "; " if note else "") + "partial year (data to 2026-10-30)"
            w.writerow(["NO_Fiskeridir_sales_notes", sp, "27.3.a (main area 09)", "NO", y,
                        "catch_month", m, g, round(t, 3), note])
        whm = {  # WGWIDE Table 7.2.1.1, western horse mackerel, division 3.a, all countries
            2023: (0, 0, 22, 3, "WGWIDE 2024 Table 7.2.1.1 (row labelled '3')"),
            2024: (1, 0, 13, 11, "WGWIDE 2025 Table 7.2.1.1"),
            2025: (0, 1, 11, 43, "WGWIDE 2026 Table 7.2.1.1"),
        }
        for y, (q1, q2, q3, q4, src) in whm.items():
            for q, t in zip((1, 2, 3, 4), (q1, q2, q3, q4)):
                w.writerow(["WGWIDE_WHM_quarterly", "HOM", "27.3.a", "ALL", y, "quarter", q,
                            "all", t, src])

    # Profiles
    def profile(sp, y0, y1):
        tot = defaultdict(float)
        for (s, y, m, g), t in mon.items():
            if s == sp and y0 <= y <= y1:
                tot[m] += t
        return tot

    with open(os.path.join(out_dir, "catch_no_3a_month_profile.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species_fao", "years", "total_t", "mean_t_per_year"]
                   + [f"share_m{m:02d}" for m in range(1, 13)]
                   + ["share_q1", "share_q2", "share_q3", "share_q4", "share_no_date"])
        for sp in ("MAC", "HOM"):
            for y0, y1 in ((2010, 2011), (2012, 2025), (2015, 2025), (2015, 2024),
                           (2020, 2025)):
                p = profile(sp, y0, y1)
                tot = sum(p.values())
                if tot <= 0:
                    continue
                ms = [p.get(m, 0.0) / tot for m in range(1, 13)]
                qs = [sum(ms[3 * k:3 * k + 3]) for k in range(4)]
                w.writerow([sp, f"{y0}-{y1}", round(tot, 1), round(tot / (y1 - y0 + 1), 1)]
                           + [round(x, 3) for x in ms] + [round(x, 3) for x in qs]
                           + [round(p.get(0, 0.0) / tot, 3)])

    gy = defaultdict(float)
    for r in no:
        gy[(r["species_fao"], int(r["year"]), r["gear_group"], r["gear"])] += r["t"]
    with open(os.path.join(out_dir, "catch_no_3a_by_gear_year.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species_fao", "year", "gear_group", "gear", "tonnes"])
        for k, t in sorted(gy.items()):
            w.writerow(list(k) + [round(t, 3)])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
