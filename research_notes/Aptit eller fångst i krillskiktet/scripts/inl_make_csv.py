"""Write tables/inlayer_raw.csv: one row per sourced or derived number of lit_inlayer.md.

Run: python3 -I inl_make_csv.py ../tables/inlayer_raw.csv
Tags: F full text read, A abstract only, S secondary (another note, a review, or a fetch-tool quote summary),
D derived here (arithmetic, script named in the method column, or a value read off a figure), J judgement.
"""
import csv
import sys

HEADER = ["source", "doi", "species", "region", "season", "method", "quantity", "value", "unit", "tag"]
rows = []


def add(source, doi, species, region, season, method, quantity, value, unit, tag):
    rows.append([source, doi, species, region, season, method, quantity, value, unit, tag])


# ---------------------------------------------------------------- Onsrud et al. 2004, Table 3
ONS = "Onsrud et al. 2004 ICES JMS 61:1278"
ONSD = "10.1016/j.icesjms.2004.09.005"
REG = "Oslofjord 120 m site"
T3 = [
    ("whiting", "Nov 2000", "day", 2.67, 1.37, 42.31, 1.82, 0.98, 2.89, 0.91, 25.00),
    ("whiting", "Nov 2000", "night", 3.00, 1.20, 35.19, 3.47, 3.47, 1.88, 0.91, 14.81),
    ("whiting", "Mar 2001", "day", 3.28, 1.03, 75.51, 3.95, 4.28, 3.03, 1.17, 8.16),
    ("whiting", "Mar 2001", "night", 3.25, 0.97, 79.69, 5.92, 11.39, 2.77, 1.09, 6.25),
    ("whiting", "Jun 2001", "day", 3.08, 1.21, 42.86, 2.17, 2.33, 3.32, 1.25, 7.14),
    ("whiting", "Jun 2001", "night", 2.23, 1.03, 34.62, 1.44, 1.01, 2.92, 1.75, 26.92),
    ("Norway pout", "Nov 2000", "day", 3.27, 1.42, 59.09, 1.54, 0.78, 3.45, 1.00, 22.73),
    ("Norway pout", "Nov 2000", "night", 2.93, 1.08, 44.07, 1.54, 0.71, 3.72, 0.84, 15.25),
    ("Norway pout", "Mar 2001", "day", 4.00, 1.31, 75.00, 2.17, 1.33, 3.69, 0.85, 0.00),
    ("Norway pout", "Mar 2001", "night", 3.62, 0.86, 47.62, 2.10, 1.20, 3.41, 1.26, 0.00),
    ("herring", "Nov 2000", "day", 2.91, 0.84, 31.03, 1.90, 1.10, 4.00, 0.69, 0.00),
    ("herring", "Nov 2000", "night", 2.29, 0.46, 17.24, 2.20, 0.83, 4.08, 0.79, 0.00),
    ("herring", "Mar 2001", "day", 3.43, 0.89, 50.00, 5.13, 4.64, 5.00, 0.00, 6.25),
    ("herring", "Mar 2001", "night", 2.70, 0.47, 20.00, 1.50, 0.70, 2.79, 1.40, 0.00),
]
M = "trawl stomachs, Table 3 (p. 1287)"
for sp, mo, per, fm, fs, fo, mc, sc, dm, ds, em in T3:
    s = "%s %s" % (mo, per)
    add(ONS, ONSD, sp, REG, s, M, "stomach fullness mean (scale 1 empty - 5 full)", fm, "score", "F")
    add(ONS, ONSD, sp, REG, s, M, "stomach fullness SD", fs, "score", "F")
    add(ONS, ONSD, sp, REG, s, M, "stomachs with krill structures", fo, "%", "F")
    add(ONS, ONSD, sp, REG, s, M, "krill per stomach mean (conditional on krill present, see D rows)", mc, "krill/stomach", "F")
    add(ONS, ONSD, sp, REG, s, M, "krill per stomach SD", sc, "krill/stomach", "F")
    add(ONS, ONSD, sp, REG, s, M, "digestion state of krill mean (1 undigested - 5 fully digested)", dm, "score", "F")
    add(ONS, ONSD, sp, REG, s, M, "digestion state of krill SD", ds, "score", "F")
    add(ONS, ONSD, sp, REG, s, M, "empty stomachs", em, "%", "F")

add(ONS, ONSD, "whiting", REG, "Nov 2000 / Mar, Jun 2001", "text p. 1284", "krill share of stomach content by weight, mean day and night", 82, "%", "F")
add(ONS, ONSD, "whiting", REG, "Nov 2000 / Mar, Jun 2001", "text p. 1284", "krill share of stomach content by weight, maximum", 99, "%", "F")
add(ONS, ONSD, "whiting (21 cm)", REG, "Nov 2000 / Mar, Jun 2001", "text p. 1284", "maximum krill in one stomach", 40, "krill", "F")
add(ONS, ONSD, "fish (all)", REG, "Mar 2001", "38 kHz, hull vs bottom transducer 80 m away", "Sv difference below 75 m, under vessel vs 80 m away", 9, "dB", "F")
add(ONS, ONSD, "fish (all)", REG, "Mar 2001", "38 kHz, hull vs bottom transducer 80 m away", "fish biomass under anchored vessel relative to 80 m away", 8, "x", "F")
add(ONS, ONSD, "fish (all)", REG, "all surveys", "echograms", "time for fish targets to reappear after a passing ship (200-400 m)", "10-20", "min", "F")
add(ONS, ONSD, "M. norvegica", REG, "all surveys", "echograms", "day krill layer thickness (below ~70 m)", "20-40", "m", "F")
add(ONS, ONSD, "fish (all)", REG, "all surveys", "methods", "minimum total length for stomach removal at sea (smaller fish frozen whole)", 25, "cm", "F")

S1 = "scripts/inl_onsrud_counts.py"
add(ONS, ONSD, "whiting, pout, herring", REG, "all cells", S1, "count statistic is conditional on krill present (reported SD below the all-stomach minimum) in cells", "11 of 14 (3 undecided, none contradicts)", "cells", "D")
for sp, s, vm in (("whiting", "Nov 2000 day", 0.53), ("whiting", "Nov 2000 night", 3.47), ("whiting", "Mar 2001 day", 4.64),
                  ("whiting", "Mar 2001 night", 21.91), ("whiting", "Jun 2001 day", 2.50), ("whiting", "Jun 2001 night", 0.71),
                  ("Norway pout", "Nov 2000 day", 0.40), ("Norway pout", "Nov 2000 night", 0.33), ("Norway pout", "Mar 2001 day", 0.82),
                  ("Norway pout", "Mar 2001 night", 0.69)):
    add(ONS, ONSD, sp, REG, s, S1, "variance/mean of krill count among stomachs with krill", vm, "-", "D")
add(ONS, ONSD, "whiting", REG, "Mar 2001 night; Jun 2001 day and night", S1, "conditional SD vs the log-series limit (largest zero-truncated NB SD); June n ~6-12 and ~9, not robust", "11.39 vs 8.51; 2.33 vs 2.00; 1.01 vs 0.89", "krill/stomach", "D")
add(ONS, ONSD, "whiting", REG, "Nov 2000 day", S1, "krill-free stomachs, observed vs implied by zero-truncated Poisson fit to positives", "0.577 vs 0.261", "share", "D")
add(ONS, ONSD, "Norway pout", REG, "Nov 2000 day", S1, "krill-free stomachs, observed vs implied by zero-truncated Poisson fit to positives", "0.409 vs 0.392", "share", "D")
add(ONS, ONSD, "Norway pout", REG, "Mar 2001 day", S1, "krill-free stomachs, observed vs implied by zero-truncated NB fit to positives", "0.250 vs 0.245", "share", "D")
add(ONS, ONSD, "whiting", REG, "Nov 2000", S1, "digestion state night minus day (negative = fresher at night)", -1.01, "score", "D")
add(ONS, ONSD, "whiting", REG, "Mar 2001", S1, "digestion state night minus day", -0.26, "score", "D")
add(ONS, ONSD, "whiting", REG, "Jun 2001", S1, "digestion state night minus day", -0.40, "score", "D")

# ---------------------------------------------------------------- Onsrud, Kaartvedt & Breien 2005
O5 = "Onsrud, Kaartvedt & Breien 2005 CJFAS 62:1822"
O5D = "10.1139/f05-090"
TT = "split-beam target tracking"
add(O5, O5D, "whiting and Norway pout (deep fish)", REG, "Nov 2000", TT, "swimming speed mode, 80-110 m, day / during ascent", "15-16", "cm/s", "F")
add(O5, O5D, "whiting and Norway pout (deep fish)", REG, "Nov 2000", TT, "swimming speed mode, 80-110 m, night", 10, "cm/s", "F")
add(O5, O5D, "fish in 10-30 m (herring, sprat, whiting)", REG, "Nov 2000 night", TT, "swimming speed range of most fish", "16-24", "cm/s", "F")
add(O5, O5D, "deep fish", REG, "Nov 2000", TT, "swimming speed in body lengths, day / night", "1 / 0.7", "BL/s", "F")
add(O5, O5D, "unidentified fish at upper krill-layer edge", REG, "Nov 2000", TT, "one tracked search cycle: slow spiralling ascent / straight rapid descent", "11 / 39", "cm/s", "F")
add(O5, O5D, "deep fish", REG, "Nov 2000 1400-1716 h", TT, "tracks descending vs ascending during dusk", "245 vs 109", "tracks", "F")
add(O5, O5D, "deep fish", REG, "Nov 2000", TT, "typical track length", "10-20", "pings", "F")
add(O5, O5D, "Norway pout (inferred)", REG, "Nov 2000 night", TT, "lowest gliding speeds on curved paths", 0.3, "BL/s", "F")
add(O5, O5D, "M. norvegica vs deep fish", REG, "night", "text, citing Klevjer & Kaartvedt 2003", "krill speed mode as share of deep-fish night speed mode", "30-40", "%", "F")

# ---------------------------------------------------------------- Albert 1994
AL = "Albert 1994 ICES JMS 51:45"
ALD = "10.1006/jmsc.1994.1005"
AR = "Norwegian Deep incl. Skagerrak slope"
M8 = "trawl stomachs, Table 8 (page images)"
for grp, n, emp, mnw, mnf in (("< 14 cm", 277, 33.6, 42.8, 29.3), (">= 14 cm", 188, 24.5, 59.1, 47.2)):
    add(AL, ALD, "Norway pout " + grp, AR, "winter 1986-87", M8, "stomachs examined", n, "stomachs", "F")
    add(AL, ALD, "Norway pout " + grp, AR, "winter 1986-87", M8, "empty stomachs", emp, "%", "F")
    add(AL, ALD, "Norway pout " + grp, AR, "winter 1986-87", M8, "M. norvegica share of contents by weight", mnw, "%W", "F")
    add(AL, ALD, "Norway pout " + grp, AR, "winter 1986-87", M8, "M. norvegica frequency of occurrence in non-empty stomachs", mnf, "%", "F")
add(AL, ALD, "Norway pout < 16 cm", AR, "summer 1985-86", M8, "empty stomachs", 27.8, "%", "F")
add(AL, ALD, "Norway pout >= 16 cm", AR, "summer 1985-86", M8, "empty stomachs", 18.6, "%", "F")
add(AL, ALD, "Norway pout", AR, "pooled", "text p. 53", "non-empty stomachs with only one prey category", 43, "%", "F")
add(AL, ALD, "Norway pout", AR, "pooled", "text p. 53", "stomachs with no more than three prey categories, by predator group", "78-92", "%", "F")
add(AL, ALD, "Norway pout", AR, "one 22-h series (season not stated on the pages read)", "7 hauls, text p. 54", "M. norvegica share of contents by weight", "70-99", "%W", "F")
add(AL, ALD, "Norway pout", AR, "one 22-h series", "Fig. 11", "stomachs per haul", "16, 18, 19, 26, 18, 13, 29", "stomachs", "F")
add(AL, ALD, "Norway pout", AR, "one 22-h series", "Fig. 11 read off (200 dpi)", "empty stomachs at ~0.5, 5, 9, 11.5, 14, 17, 20 h", "0, 22, 21, 45, 72, 30, 30", "%", "D")
add(AL, ALD, "Norway pout", AR, "one 22-h series", "Fig. 11 read off; text says 'halved'", "relative dry weight of contents (non-empty) at the same hours", "2.2, 1.9, 1.65, 1.35, 1.0, 1.0, 2.0", "dry mg per fish weight, units as labelled", "D")
add(AL, ALD, "Norway pout", AR, "one 22-h series", "Fig. 11 read off", "share of content weight at digestion stage 5 (almost complete): 0-5 h -> 14-17 h -> 20 h", "<5 -> 40-50 -> ~3", "%", "D")

# ---------------------------------------------------------------- Wilson, Jump & Buchheister 2009
W = "Wilson, Jump & Buchheister 2009 MEPS 392:239"
WD = "10.3354/meps08237"
WR = "western Gulf of Alaska shelf, 48-site grid"
WS = "September 2000, 2001, 2003"
F2 = "Fig. 2: night stomach krill weight^0.25 vs site krill abundance (ind/m2)^0.25 (Eq. 3)"
add(W, WD, "walleye pollock Age-0 (<=120 mm SL)", WR, WS, F2, "R2 of krill weight on predator length and krill abundance", 0.53, "-", "F")
add(W, WD, "walleye pollock Age-0", WR, WS, F2, "p", "<0.001", "-", "F")
add(W, WD, "capelin", WR, WS, F2, "R2 of krill weight on krill abundance", 0.23, "-", "F")
add(W, WD, "capelin", WR, WS, F2, "p", "<0.001", "-", "F")
add(W, WD, "walleye pollock Age-1+ (121-250 mm SL)", WR, WS, F2, "R2 of krill weight on krill abundance and year", 0.38, "-", "F")
add(W, WD, "walleye pollock Age-1+", WR, WS, F2, "p", 0.012, "-", "F")
add(W, WD, "eulachon", WR, WS, F2, "R2 of krill weight on predator length (no krill-abundance term retained)", 0.10, "-", "F")
add(W, WD, "eulachon", WR, WS, F2, "p", 0.043, "-", "F")
for sp, dr in (("walleye pollock Age-0", 1.60), ("walleye pollock Age-1+", 1.67), ("capelin", 0.68), ("eulachon", 0.50)):
    add(W, WD, sp, WR, WS, "MAXIMS diel model, one value per group (Eq. 2 constant)", "daily ration", dr, "%BW/d", "F")
for sp, e in (("walleye pollock Age-0", 3.2), ("walleye pollock Age-1+", 2.3), ("capelin", 19), ("eulachon", 31)):
    add(W, WD, sp, WR, WS, "stomachs", "empty stomachs", e, "%", "F")
add(W, WD, "pollock, capelin, eulachon", WR, WS, "stomachs", "fish examined / trawl hauls", "4267 / 378", "-", "F")
add(W, WD, "pollock, capelin, eulachon", WR, WS, "site consumption vs night standing stock", "maximum daily consumption as share of site standing stock", 12, "%", "F")
add(W, WD, "pollock, capelin, eulachon", WR, "2000 / 2001 / 2003", "site consumption vs night standing stock", "median daily consumption as share of site standing stock", "0.015 / 0.026 / 0.018", "%", "F")
add(W, WD, "walleye pollock Age-1+", WR, WS, "Table 2 (fixed ration)", "per capita krill consumption in northeastern shelf areas", "7-59", "krill/fish/d", "F")
add(W, WD, "walleye pollock Age-0", WR, WS, "diet by 3-h bin", "krill share of stomach content mass in any 3-h bin, minimum", 60, "%", "F")
S2 = "scripts/inl_wilson_table2.py"
add(W, WD, "krill (Thysanoessa mainly)", WR, WS, S2, "area-mean krill density, 14 area-years (stock x plankton krill weight / area)", "1.2-13.7", "g/m2", "D")
add(W, WD, "walleye pollock Age-1+", WR, WS, S2, "Spearman rho, area krill g/m2 vs per capita krill numbers (n 12, fixed ration built in)", 0.41, "-", "D")
add(W, WD, "walleye pollock Age-1+", WR, WS, S2 + "; Fig. 2 read off", "site krill abundance span of the fitted relation", "~39-2400", "ind/m2", "D")
add(W, WD, "capelin", WR, WS, S2 + "; Fig. 2 read off", "site krill abundance span of the fitted relation", "~0-6600", "ind/m2", "D")

# ---------------------------------------------------------------- Kock et al. 1994 (icefish, krill specialist proxy)
K = "Kock, Wilhelms, Everson & Groger 1994 MEPS 108:43"
KD = "10.3354/meps108043"
KR = "South Georgia shelf and Shag Rocks"
KM = "stomach content wet weight / fish weight (Fig. 7 text)"
for yr, med, q in (("Jan-Feb 1985", 1.07, "0.31-2.40"), ("Jan-Feb 1991 (krill scarce)", 0.27, "0.0-1.21"), ("Jan 1992 (krill high)", 2.19, "0.63-4.43")):
    add(K, KD, "Champsocephalus gunnari", KR, yr, KM, "median feeding intensity", med, "%BW", "F")
    add(K, KD, "Champsocephalus gunnari", KR, yr, KM, "feeding intensity 25-75 % percentiles", q, "%BW", "F")
add(K, KD, "Champsocephalus gunnari", KR, "Jan-Feb 1991", "stations", "empty stomachs per station, often above", 40, "%", "F")
add(K, KD, "Champsocephalus gunnari", KR, "1985, 1991, 1992", "stomachs", "stomachs examined", "ca 8700", "stomachs", "F")
add("Main, Collins, Mitchell & Belchier 2009 Polar Biol 32:569", "10.1007/s00300-008-0552-7", "Champsocephalus gunnari", "South Georgia",
    "summers incl. 2003-04 (krill poor)", "stomachs (abstract)", "stomachs examined", 2239, "stomachs", "A")
add("Main, Collins, Mitchell & Belchier 2009 Polar Biol 32:569", "10.1007/s00300-008-0552-7", "Champsocephalus gunnari", "South Georgia",
    "2005 after the 2003-04 krill-poor summer", "stock assessment (abstract)", "reduction in estimated stock biomass", ">80", "%", "A")

# ---------------------------------------------------------------- Orlova et al. 2005 (cod), fetch-tool quotes only
OR = "Orlova, Dolgov, Rudneva & Nesterova 2005 ICES JMS 62:1463"
ORD = "10.1016/j.icesjms.2005.08.003"
add(OR, ORD, "euphausiids", "Barents Sea, central/east and Murmansk coast", "1987", "near-bottom net index (autumn-winter), not paired with stomachs", "mean density of concentrations", "1900-3000", "ind/1000 m3", "S")
add(OR, ORD, "euphausiids", "Murmansk Bank slopes", "Jun-Aug 1997", "as above", "concentrations where cod fed on krill", "1-5", "ind/m3", "S")
add(OR, ORD, "cod", "Barents Sea", "1999", "stomach fullness index", "partial index of euphausiids / total index", "6.8 / 250.9", "prodecimille", "S")
add(OR, ORD, "cod", "Barents Sea", "1995 (euphausiid feeding, low capelin)", "stomach fullness index", "total index did not exceed", "1.5-2", "as printed (scale not verified)", "S")

# ---------------------------------------------------------------- Johannesen et al. 2012 (cod, negative)
JO = "Johannesen et al. 2012 MEPS 458:181"
JOD = "10.3354/meps09818"
add(JO, JOD, "cod >= 30 cm", "Barents Sea ecosystem survey", "Aug-Sep 2004-2009", "stomachs", "krill share of individual consumption (krill excluded from the density analysis)", 10, "%", "F")
add(JO, JOD, "krill", "Barents Sea ecosystem survey", "Aug-Sep 2004-2009", "0-group pelagic trawl", "data lost by keeping night hauls only (diel catchability of krill)", "2/3", "share", "F")

# ---------------------------------------------------------------- Shaw et al. 1989 (Pacific hake, DFO MS Rep 2027)
SH = "Shaw, Tanasichuk, Ware & McFarlane 1989 Can MS Rep Fish Aquat Sci 2027"
add(SH, "none (report)", "plankton layer with euphausiids", "Trevor Channel, Barkley Sound", "Aug 1987", "echogram", "depth of dense plankton layer", "66-73", "m", "F")
add(SH, "none (report)", "Pacific hake 23-44 cm", "Trevor Channel, Barkley Sound", "Aug 1987", "echogram", "depth of hake layer beneath it", "77-91", "m", "F")
add(SH, "none (report)", "Pacific hake", "Trevor Channel, Barkley Sound", "Aug 1987", "stomachs", "euphausiid share of prey volume", 86, "%", "F")
add(SH, "none (report)", "Pacific hake", "Trevor Channel / offshore north / La Perouse", "Aug 1987", "stomachs, everted excluded", "mean food volume per stomach", "7.8 / 16.8 / 26.9", "cc", "F")
add(SH, "none (report)", "Pacific herring", "SW Vancouver Island", "Aug 1987", "stomachs", "empty stomachs (diet euphausiids only)", 39, "%", "F")

# ---------------------------------------------------------------- secondary carried from sibling notes
add("Livingston 1983 Fish Bull 81:629 (via Krillintag och jaktsatt S5)", "none", "Pacific hake ~500 mm", "Washington-Oregon", "14-15 July 1967",
    "diel stomach series", "peak stomach content weight / fish weight, at 1800 h", 2.5, "%BW", "S")

with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(HEADER)
    w.writerows(rows)
print("rows written:", len(rows))
