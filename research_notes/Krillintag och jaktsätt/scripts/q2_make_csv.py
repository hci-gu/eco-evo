"""Q2: write tables/q2_raw.csv from the numbers in q2_evidence.md (csv module; quotes in sources).

Usage (from scripts/): python3 -I q2_make_csv.py ../tables/q2_raw.csv
"""
import csv
import sys

HEADER = ["question", "species", "quantity", "low", "central", "high", "unit", "source", "tag"]

R03 = "Rindorf 2003 MEPS 249:265 doi:10.3354/meps249265"
RG05 = "Rindorf & Gislason 2005 JEMBE doi:10.1016/j.jembe.2005.04.013"
MT97 = "Mergardt & Temming 1997 ICES JMS 54:226 doi:10.1006/jmsc.1996.0190"
P00 = "Pedersen 2000 J Fish Biol 57:402 doi:10.1006/jfbi.2000.1316"
ONS = "Onsrud et al. 2004 ICES JMS 61:1278"
AE00 = "Adlerstein & Ehrich 2000 ICES CM 2000/K:01"
AJ12 = "Armannsson & Jonsson 2012 ICES JMS 69:1372 doi:10.1093/icesjms/fss121"
SO16 = "Skilbrei & Ottera 2016 ICES JMS 73:1186 doi:10.1093/icesjms/fsv261"
SIB = "research_notes Sejens hojd over botten / Sejens profil over djupa bottnar"
DP = "de Pontual et al. 2013 ICES JMS 70:1416 doi:10.1093/icesjms/fst102"
MOD = "Modica et al. 2011 MEPS 440:177 doi:10.3354/meps09341"
BOZ = "Bozzano et al. 2005 Fish Res 73:29 doi:10.1016/j.fishres.2005.01.006"
R02 = "Rindorf 2002 J Fish Biol 61:579 doi:10.1111/j.1095-8649.2002.tb00897.x"
DL = "q2_daylength.py (astronomy)"

ROWS = [
    # whiting
    ("whiting", "site bed depth of the 5 diel stomach sites (Sept; 20-35 cm; n=7738)", 44, 53, 137, "m", R03, "F"),
    ("whiting", "civil-night window share of 24 h in which bed prey were eaten (4 of 5 sites, early Sept; R&G Table 1 clock times)", 0.369, 0.38, 0.392, "fraction of 24 h", R03 + "; " + RG05 + "; " + DL, "D"),
    ("whiting", "benthic foraging at the 137 m site (no bed prey in stomachs)", 0, 0, 0, "fraction of 24 h", R03, "F"),
    ("whiting", "deviance share explained by the common diel term (bt), crab", None, 0.31, None, "fraction", R03, "F"),
    ("whiting", "deviance share explained by location-specific diel terms (abloc,t), crab", None, 0.05, None, "fraction", R03, "F"),
    ("whiting", "deviance share explained by the common diel term (bt), krill", None, 0.093, None, "fraction", R03, "F"),
    ("whiting", "deviance share explained by location-specific diel terms (abloc,t), krill", None, 0.219, None, "fraction", R03, "F"),
    ("whiting", "civil twilight hours per day (both ends), early Sept, R&G Table 1", 1.23, None, 1.32, "h", RG05 + "; " + DL, "D"),
    ("whiting", "fish-feeding window share of 24 h: bounds twilight only vs two 4-h sampling bins (width not resolved)", 0.051, None, 0.333, "fraction of 24 h", RG05 + "; " + DL, "D"),
    ("whiting", "clupeid intake plateau (25-35 cm whiting)", 0.10, None, 0.17, "fraction of max daily ration", RG05, "F"),
    ("whiting", "gadoid intake plateau (25-35 cm whiting)", None, 0.06, None, "fraction of max daily ration", RG05, "F"),
    ("whiting", "intake of other food (crustaceans, benthos)", 0.08, None, 0.22, "fraction of max daily ration", RG05, "F"),
    ("whiting", "correlation of fish vs other-food intake across sites (n.s.)", -0.30, None, 0.73, "r", RG05, "F"),
    ("whiting", "between-site range of fish intake (% of max) by length group", 17, 29, 36, "fold", RG05, "F"),
    ("whiting", "between-site range of other-food intake (max)", None, None, 3, "fold", RG05, "F"),
    ("whiting", "maximum daily intake used (warm site; cold site ~3.1)", 3.1, None, 6.2, "% body weight per day", RG05, "F"),
    ("whiting", "sandeel feeding peak clock hour (May-Jun, 57.5 N, 25-30 cm)", 22, 23, 24, "h of day", MT97, "F"),
    ("whiting", "minimum feeding clock hour", 8, 9, 10, "h of day", MT97, "F"),
    ("whiting", "specific daily ration, in situ evacuation (pelagic and demersal)", 3.99, None, 10.31, "% body weight per day", P00, "A"),
    ("whiting", "share of krill in stomach weight, day and night (Oslofjord, 120 m bed)", None, 0.82, 0.99, "fraction", ONS, "F"),
    ("whiting", "height above bed by day (krill layer z 70-110 m over 120 m bed)", 10, None, 50, "m", ONS, "D"),
    ("whiting", "height above bed at night (upper 50 m over 120 m bed)", 70, None, 120, "m", ONS, "D"),
    ("whiting", "swimming speed day (night 10-11)", 14, None, 16, "cm/s", "Onsrud et al. 2005 (held)", "S"),
    ("whiting", "time-of-day effect on GOV catch rates of whiting > 20 cm (not significant)", 0, 0, 0, "effect", AE00, "F"),
    # saithe
    ("saithe", "DST fish length (n=51 recovered, 268018 hourly depths)", 42, 54, 83, "cm", AJ12, "F"),
    ("saithe", "median residence depth, summer-autumn to Mar-Apr", None, 50, 140, "m", AJ12, "F"),
    ("saithe", "share of time in rapid vertical runs (> 3 m/min), 19 tags", 0.002, 0.15, 0.33, "fraction of time", AJ12, "F"),
    ("saithe", "share of days within free vertical range (above 0.22, below 0.02, both 0.13)", None, 0.63, None, "fraction of days", AJ12, "F"),
    ("saithe", "underestimate of daily vertical range by hourly vs 1-min data", None, 0.70, None, "fraction", AJ12, "F"),
    ("saithe", "fish-months with no diel vertical migration (farm fish, 38-59 cm)", None, 0.382, None, "fraction", SO16, "F"),
    ("saithe", "fish-months deeper by day", None, 0.174, None, "fraction", SO16, "F"),
    ("saithe", "fish-months deeper by night", None, 0.236, None, "fraction", SO16, "F"),
    ("saithe", "fish-months switching between day-deep and night-deep", None, 0.045, None, "fraction", SO16, "F"),
    ("saithe", "observations away from the farm in daytime", None, 0.16, None, "fraction", SO16, "F"),
    ("saithe", "individuals with a diel pattern part of the time (30 of 40)", None, 0.75, None, "fraction", SO16, "F"),
    ("saithe", "December activity (SD of depth) at 7-10 h (night 18-4 h: 3.8, 3.3-4.4)", 7.9, 9.1, 10.3, "m", SO16, "F"),
    ("saithe", "residence depth summer (winter 60-90)", 25, None, 50, "m", SO16, "F"),
    ("saithe", "day scale height above bed, exponential profile (held)", 9, 15, 31, "m", SIB, "S"),
    ("saithe", "night scale height above bed (Neilson 2003, day 15 m at same site)", None, 9, None, "m", SIB, "S"),
    ("saithe", "night surface hauls holding adults over beds >= 150 m (11/82; day 0/63)", None, 0.134, None, "fraction of hauls", SIB, "S"),
    ("saithe", "day-length (active window) share of 24 h at 58.15 N (manifest), 21 Dec / 5 Sep / 21 Jun", 0.267, 0.573, 0.760, "fraction of 24 h", DL, "D"),
    ("saithe", "stomach content near salmon farms relative to away (Dempster 2011)", None, 2.35, None, "ratio", "research_notes Mattnad hos rovfisk (Dempster et al. 2011)", "S"),
    # hake
    ("hake", "DST fish length (259 tagged 2006-07, 9 recovered, <= 81 d; mostly immature, no adult > 40 cm)", 28.9, None, 39.5, "cm", DP, "F"),
    ("hake", "temperature change during diel vertical movements", 7, None, None, "deg C", "de Pontual et al. 2012 J Fish Biol 81:728 doi:10.1111/j.1095-8649.2012.03345.x", "A"),
    ("hake", "SST reached during diel vertical movements to forage", None, 20, None, "deg C", DP, "F"),
    ("hake", "pelagic-net catches of juveniles taken at night", None, 0.955, None, "fraction", BOZ, "S"),
    ("hake", "juvenile low-feeding phase around sunset (1-3 h; catch times GMT, printed sunset likely local time)", 0.04, None, 0.125, "fraction of 24 h", MOD, "D"),
    ("hake", "empty stomachs at 20:25 GMT (rest phase)", None, 0.345, None, "fraction", MOD, "F"),
    ("hake", "stomach content minimum (20:25) vs maximum (05:37)", 0.81, None, 3.49, "% body wet weight", MOD, "F"),
    ("hake", "euphausiids in zooplankton prey at night (afternoon: mysids 0.954)", None, 0.698, None, "fraction by number", MOD, "F"),
    ("hake", "night window share of 24 h (pelagic euphausiid mode), 27 Jun 41 N", None, 0.324, None, "fraction of 24 h", MOD + "; " + DL, "D"),
    ("hake", "juvenile daily ration (square-root model; exponential model 1.5)", 4.08, None, 5.09, "% body wet weight per day", MOD, "F"),
    ("hake", "day bed hauls holding hake in the area (32/43, Jun-Jul)", None, 0.744, None, "fraction of hauls", SIB, "S"),
    # satiation
    ("whiting", "effect of stomach fullness on meal size (none) vs on P(fresh food) (depressed)", None, None, None, "qualitative", R02, "A"),
]


def main():
    out = sys.argv[1]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for sp, q, lo, c, hi, unit, src, tag in ROWS:
            w.writerow(["Q2", sp, q, "" if lo is None else lo, "" if c is None else c,
                        "" if hi is None else hi, unit, src, tag])
    print(f"wrote {len(ROWS)} rows to {out}")


if __name__ == "__main__":
    main()
