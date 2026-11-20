"""Write tables/fill_series_raw.csv from the values transcribed in lit_fill_series.md.

Values were read by hand from the sources (tables or figures); this script only lays them out.
Run: python3 -I scripts/write_fill_series_csv.py <output.csv>
"""
import csv
import sys

COLS = ["species", "area", "years", "maturity_or_size", "quantity", "month_or_quarter",
        "value", "sd_or_range", "n", "unit", "source", "tag", "note"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
rows = []

# Saithe, Mathers et al. 1992 MEPS 86:31, doi:10.3354/meps086031
M92 = "Mathers et al. 1992 MEPS 86:31 doi:10.3354/meps086031"
for mon, k in [("Aug-1988", 1.13), ("Oct-1988", 1.09), ("Apr-1989", 0.87), ("Aug-1989", 1.12)]:
    rows.append(["saithe", "N North Sea Beryl oilfield", "1988-1989", "ages 2+-5+ ~41-43 cm, maturity not reported",
                 "Fulton K raw (K not in Table 3 size corrections)", mon, k, "fig read +-0.02", "", "g cm-3 x100", M92, "F",
                 "Fig 3A; min Apr after Feb-Mar spawning"])
for mon, g, rng in [("Aug-1988", 0.3, "fig read, bar near 0, +-0.3"), ("Oct-1988", 0.3, "fig read, bar near 0, +-0.3"),
                    ("Apr-1989", 16.5, "fig read +-0.5"), ("Aug-1989", 3.3, "fig read 3.2-3.5")]:
    rows.append(["saithe", "N North Sea Beryl oilfield", "1988-1989", "ages 2+-5+", "GLI gall bladder/liver x100",
                 mon, g, rng, "", "%", M92, "F", "feeding indicator, not reserve; bile stored when not feeding"])
rows.append(["saithe", "N North Sea Beryl Alpha", "1989", "41.2+-1.0 cm", "LSI liver/whole weight", "Aug-1989",
             5.40, "SE 0.49", 18, "%", M92, "F", "Table 2; only LSI values given"])
rows.append(["saithe", "N North Sea Beryl Bravo", "1989", "43.5+-3.1 cm", "LSI liver/whole weight", "Aug-1989",
             5.95, "SE 0.76", 5, "%", M92, "F", "Table 2"])

# Saithe, Uglem et al. 2020 Aquac Res 51:1720, doi:10.1111/are.14519, Hitra UA only
U20 = "Uglem et al. 2020 Aquac Res 51:1720 doi:10.1111/are.14519"
for season, gear, n, tl, k, hsi in [("autumn", "jig", 30, "68+-4", "0.95+-0.13", "6.9+-3.1"),
                                     ("autumn", "net", 30, "68+-3", "0.94+-0.10", "6.8+-1.4"),
                                     ("spring", "net", 31, "66+-6", "0.84+-0.07", "6.2+-4.4"),
                                     ("summer", "jig", 20, "62+-6", "0.86+-0.07", "6.0+-2.2"),
                                     ("summer", "net", 20, "69+-4", "0.90+-0.08", "6.8+-2.7")]:
    for q, v in [("HSI liver/whole weight", hsi), ("Fulton K", k)]:
        val, sd = v.split("+-")
        rows.append(["saithe", "mid-Norway coast Hitra, unassociated control", "2013-2015",
                     f"TL {tl} cm, mostly immature (8.2% of spring fish GSI>2)", q, season, val, f"SD {sd}", n,
                     "%" if q.startswith("HSI") else "g cm-3 x100", U20, "F", f"gear {gear}; months not given"])

# Saithe, Dempster et al. 2011 PLoS ONE 6:e15646, summer 2007
D11 = "Dempster et al. 2011 PLoS ONE 6:e15646 doi:10.1371/journal.pone.0015646"
for area, fl, n, ua, fa, fua, ffa in [("Ryfylke", 54.2, 30, 8.5, 9.3, 1.08, 1.17),
                                       ("Hitra", 34.3, 88, 6.0, 8.7, 1.15, 1.28),
                                       ("Oksfjord", 43.4, 97, 5.2, 9.3, 1.10, 1.17)]:
    rows.append(["saithe", f"Norway coast {area}, unassociated", "2007", f"mean FL {fl} cm, mostly immature",
                 "HSI liver/(wet - stomach)", "Jun-Sep", ua, "fig read +-0.2", n, "%", D11, "F",
                 "Fig 3b; Ryfylke UA possibly farm influenced"])
    rows.append(["saithe", f"Norway coast {area}, farm-associated", "2007", "mostly immature",
                 "HSI liver/(wet - stomach)", "Jun-Sep", fa, "fig read +-0.2", "", "%", D11, "F",
                 "Fig 3b; food-unlimited reference"])
    rows.append(["saithe", f"Norway coast {area}, unassociated", "2007", f"mean FL {fl} cm",
                 "Fulton K", "Jun-Sep", fua, "fig read +-0.01", n, "g cm-3 x100", D11, "F", "Fig 3a"])

# Whiting, Erdogan Saglam & Saglam 2012 TrJFAS 12:831, doi:10.4194/1303-2712-v12_4_11
S12 = "Erdogan Saglam & Saglam 2012 TrJFAS 12:831 doi:10.4194/1303-2712-v12_4_11"
order = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
series = {"female": [6.0, 6.25, 6.3, 5.6, 6.15, 5.55, 2.5, 3.0, 2.35, 3.0, 4.0, 4.2],
          "male": [4.4, 5.5, 4.7, 4.3, 3.9, 4.4, 2.0, 2.5, 2.3, 1.85, 3.1, 3.9],
          "both sexes": [5.2, 5.85, 5.5, 5.0, 5.05, 5.0, 2.2, 2.7, 2.3, 2.4, 3.55, 4.0]}
for sex, vals in series.items():
    for mon, v in zip(order, vals):
        rows.append(["whiting (M. m. euxinus)", "SE Black Sea Sinop-Giresun, gillnet 25-70 m", "Nov 2010-Oct 2011",
                     f"{sex}, TL 10-23 cm mean 14.9 (small form, mostly mature)", "HSI liver/total weight",
                     mon, v, "fig read +-0.1", "", "%", S12, "F",
                     "Fig 6; spawning Mar-Jul; Apr->May step possibly sampling artefact [J]; abstract says 2009-2010"])

# Whiting, Lauerburg 2016 PhD Hamburg, Table II-VI
L16 = "Lauerburg 2016 PhD Univ Hamburg (ediss 6770) Table II-VI"
for sex, tl, a, b, n in [("F", "25-29.9", 3.76, 3.13, "99/147"), ("F", "30-34.9", 2.88, 3.72, "65/219"),
                         ("F", "35-39.9", 2.58, 4.36, "18/120"), ("F", "40-49.9", 2.80, 6.10, "3/42"),
                         ("M", "25-29.9", 2.41, 2.44, "91/161"), ("M", "30-34.9", 2.10, 2.48, "45/135"),
                         ("M", "35-39.9", 2.04, 3.67, "8/15")]:
    n1, n2 = n.split("/")
    rows.append(["whiting", "North Sea (GSBTS boxes)", "2007", f"{sex} TL {tl} cm", "median HSI liver/somatic mass",
                 "Jul-Aug", a, "", n1, "%", L16, "F", "year confounded with season; 2007 low-food year; denominator somatic (Methods) vs total (table caption)"])
    rows.append(["whiting", "North Sea (VECTORS cruise)", "2012", f"{sex} TL {tl} cm", "median HSI liver/somatic mass",
                 "Apr", b, "", n2, "%", L16, "F", "spawning season; female liver includes vitellogenin; denominator somatic (Methods) vs total (table caption)"])

# Whiting, Jonsson 2011 Matis report 36-11
J11 = "Jonsson 2011 Matis report 36-11"
for mon, k, f, tl in [("Oct", 0.87, 0.150, "47+-7"), ("Mar", 0.91, 0.077, "46+-4"),
                      ("Jun", 0.77, 0.136, "40+-2"), ("Aug", 0.72, 0.100, "48+-3")]:
    rows.append(["whiting", "Iceland SE of Vestmannaeyjar", "Oct 2010-Aug 2011", f"TL {tl} cm",
                 "Fulton K gutted", mon, k, "fig read +-0.02 (Mynd 4)", 10, "g cm-3 x100", J11, "F",
                 "spawning May-mid Jul; Mar > Aug P=0.02; Jun fish smaller"])
    rows.append(["whiting", "Iceland SE of Vestmannaeyjar", "Oct 2010-Aug 2011", f"TL {tl} cm",
                 "fillet fat", mon, f, "fig read +-0.005 (Mynd 10)", "pooled 10 fish", "% wet", J11, "F",
                 "lean fillet; Mar significantly lower than Oct and Jun"])

# Hake, Costa 2013 OJMS 3:12, doi:10.4236/ojms.2013.31002
C13 = "Costa 2013 OJMS 3:12 doi:10.4236/ojms.2013.31002"
hake = {"female < 38 cm": [3.8, 2.9, 3.1, 3.4, 3.1, 3.3, 5.7, 4.1, 5.7, 8.5, 5.5, 3.6],
        "female > 38 cm": [5.2, 4.3, 3.2, 3.4, 3.3, 3.6, 5.8, 4.1, 4.2, 5.7, 5.0, 3.6],
        "male > 26 cm": [4.6, 2.7, 2.4, 2.6, 2.4, 8.0, 2.5, 3.8, 4.0, 4.4, 2.9, 1.7]}
for grp, vals in hake.items():
    for mon, v in zip(MONTHS, vals):
        rows.append(["hake", "Portuguese coast (landings + on-board)", "2005-2010 pooled by month", grp,
                     "HSIt liver/total weight", mon, v, "fig read +-0.3", "", "%", C13, "F",
                     "Fig 7; length split not maturity; single-month spikes (F<38 Oct, M Jun) caveat"])

with open(sys.argv[1], "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    w.writerows(rows)
print(len(rows), "rows")
