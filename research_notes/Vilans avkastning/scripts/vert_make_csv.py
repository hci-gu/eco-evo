"""Write tables/vertical_refuge_raw.csv (one row per sourced number) for lit_vertical_refuge.md.

Usage: python3 -I vert_make_csv.py <out.csv>
Values are transcribed from the sources named in lit_vertical_refuge.md; nothing is computed here.
"""
import csv, sys

H = ["source", "doi", "species", "predator", "method", "quantity", "value", "unit", "tag"]
A94 = ("Albert 1994 ICES JMS 51:45", "10.1006/jmsc.1994.1005")
A95 = ("Albert 1995 ICES JMS 52:873", "10.1006/jmsc.1995.0084")
O04 = ("Onsrud et al. 2004 ICES JMS 61:1278", "10.1016/j.icesjms.2004.09.005")
O05 = ("Onsrud Kaartvedt & Breien 2005 CJFAS 62:1822", "10.1139/f05-090")
K09 = ("Kaartvedt Rostad & Klevjer 2009 MEPS 390:237", "10.3354/meps08196")
L89 = ("Lough et al. 1989 MEPS 56:1", "10.3354/meps056001")
AE = ("Adlerstein & Ehrich 2000 ICES CM 2000/K:01", "NA (ICES CM)")
B97 = ("Bromley Watson & Hislop 1997 ICES JMS 54:846", "10.1006/jmsc.1996.0211")
S11 = ("Staby Rostad & Kaartvedt 2011 MEPS 441:241", "10.3354/meps09363")
D12 = ("Dypvik Klevjer & Kaartvedt 2012 Mar Biol 159:443", "10.1007/s00227-011-1822-4")

rows = [
 (*A94, "Norway pout", "-", "echo sounding + bottom trawl, Norwegian Deep 1984-87", "height above bed of pout echo traces, maximum (seldom exceeded)", "20", "m", "F"),
 (*A94, "Norway pout", "-", "echo sounding", "day position after dawn sinking: near-bottom zone", "1.5", "m above bed", "F"),
 (*A94, "Norway pout", "-", "echo sounding", "night: traces homogeneous in the lowest layer", "5-10", "m above bed", "F"),
 (*A94, "Norway pout", "-", "bottom trawl GLM (147 stations)", "time-of-day factor on catch", "not significant (any age group)", "-", "F"),
 (*A94, "Norway pout", "-", "20-h repeated trawls, one locality (Fig. 11, read by eye)", "stomach dry weight relative to fish weight (non-empty), maximum near 01 h", "~2.2", "mg/g", "F"),
 (*A94, "Norway pout", "-", "20-h repeated trawls (Fig. 11, read by eye)", "stomach dry weight relative to fish weight (non-empty), minimum ~14-17 h", "~1.0", "mg/g", "F"),
 (*A94, "Norway pout", "-", "20-h repeated trawls (Fig. 11, read by eye)", "share of empty stomachs near 01 h", "~0", "%", "F"),
 (*A94, "Norway pout", "-", "20-h repeated trawls (Fig. 11, read by eye)", "share of empty stomachs 06-09 h", "~20", "%", "F"),
 (*A94, "Norway pout", "-", "20-h repeated trawls (Fig. 11, read by eye)", "share of empty stomachs ~14 h (peak)", "~70", "%", "F"),
 (*A94, "Norway pout", "-", "20-h repeated trawls (Fig. 11, read by eye)", "share of empty stomachs ~17 h and 20:20 h", "~30", "%", "F"),
 (*A94, "Norway pout < 14 cm", "-", "stomachs, winter (Table 8)", "share empty stomachs", "33.6", "%", "F"),
 (*A94, "Norway pout >= 14 cm", "-", "stomachs, winter (Table 8)", "share empty stomachs", "24.5", "%", "F"),
 (*A94, "Norway pout < 16 cm", "-", "stomachs, summer (Table 8)", "share empty stomachs", "27.8", "%", "F"),
 (*A94, "Norway pout >= 16 cm", "-", "stomachs, summer (Table 8)", "share empty stomachs", "18.6", "%", "F"),
 ("Dickson 1960 World Fishing 9:38 (as cited by Albert 1994 p. 55)", "NA", "Norway pout", "-", "horizontally divided trawl", "main catch height above bed by day", "2-4", "m", "S"),
 ("Dickson 1960 World Fishing 9:38 (as cited by Albert 1994 p. 55)", "NA", "Norway pout", "-", "horizontally divided trawl", "main catch height above bed by night", "0-2", "m", "S"),
 (*A95, "Norway pout 11-15 cm", "-", "14 trawls over 48 h, 140 m coastal bank, July", "stomach dry weight day / night", "2", "ratio", "F"),
 (*A95, "haddock 15-26 cm", "-", "14 trawls over 48 h, July", "share with newly eaten food at midnight and dawn", "<10", "%", "F"),
 (*A95, "haddock 15-26 cm", "-", "14 trawls over 48 h, July", "share with newly eaten food in the evening", ">60", "%", "F"),
 (*A95, "haddock 15-26 cm", "-", "14 trawls over 48 h, July", "stomach dry weight feeding / non-feeding periods", "2", "ratio", "F"),
 (*O04, "Norway pout", "-", "trawl stomachs day vs night, Oslofjord 120 m", "difference in stomach fullness or normalised wet weight day/night", "none (chi-square, Kruskal-Wallis p > 0.05)", "-", "F"),
 (*O04, "herring > 25 cm", "-", "trawl stomachs, Nov 2000 (Table 3; read in research_notes/Sillens vinterfoda)", "stomach fullness by day (1 empty - 5 distended)", "2.91", "index", "S"),
 (*O04, "herring > 25 cm", "-", "trawl stomachs, Nov 2000 (Table 3; read in research_notes/Sillens vinterfoda)", "stomach fullness by night (1 empty - 5 distended)", "2.29", "index", "S"),
 (*O04, "herring", "-", "lab value cited (Blaxter 1964)", "lower light threshold for feeding by biting", "3.6e-2 to 7e-3", "lx", "S"),
 (*O05, "Norway pout and whiting (80-110 m layer)", "-", "split-beam target tracking, Oslofjord, Nov 2000", "swimming speed mode by day", "14-16", "cm/s", "F"),
 (*O05, "Norway pout and whiting (80-110 m layer)", "-", "split-beam target tracking, Oslofjord, Nov 2000", "swimming speed mode by night", "10-11", "cm/s", "F"),
 (*O05, "Norway pout and whiting", "-", "target tracking + trawl lengths", "swimming speed by day", "1", "body lengths/s", "F"),
 (*O05, "Norway pout and whiting", "-", "target tracking + trawl lengths", "swimming speed by night", "0.7", "body lengths/s", "F"),
 (*O05, "deep fish layer (pout, whiting)", "-", "38/120 kHz echograms", "day depth of deep fish scattering layer", "below ~90", "m (bed 120 m)", "F"),
 (*O05, "krill M. norvegica", "-", "120 kHz echograms", "day depth of krill layer", "70-90", "m", "F"),
 (*K09, "sprat", "whiting, cod", "acoustics + CTD, Bunnefjorden, 4 winters", "lower tolerable O2 saturation of sprat", "~7", "% O2 sat", "F"),
 (*K09, "sprat", "whiting, cod", "acoustics + CTD", "O2 saturation below which gadoid predators were scarce", "15-20", "% O2 sat", "F"),
 (*K09, "sprat", "whiting, cod", "trawl stomachs 10 Feb 2005 day, 55-85 m (Table 1)", "share of sprat stomachs containing prey", "93", "%", "F"),
 (*K09, "sprat", "whiting, cod", "trawl stomachs 10 Feb 2005 night, 30-85 m (Table 1)", "share of sprat stomachs containing prey", "47-53", "%", "F"),
 (*K09, "sprat", "whiting, cod", "trawl stomachs 28 Jan 2004 day, 70-80 m (Table 1)", "share of sprat stomachs containing prey", "0", "%", "F"),
 (*K09, "whiting 28-36 cm", "-", "trawl stomachs 28 Jan 2004 night (Table 2)", "share of whiting stomachs with prey (mainly sprat)", "64-89", "%", "F"),
 (*K09, "whiting 31-32 cm", "-", "trawl stomachs 14 Jan 2004 day, 70-90 m (Table 2)", "share of whiting stomachs with prey", "60-77", "%", "F"),
 (*K09, "whiting 31-32 cm", "-", "trawl stomachs 14 Jan 2004 day, 70-90 m (Table 2)", "sprat per whiting stomach", "0.73-1.46", "n", "F"),
 (*K09, "whiting 28-36 cm", "-", "trawl stomachs 14 Jan 2004 night (Table 2)", "share of whiting stomachs with prey", "50-75", "%", "F"),
 (*L89, "juvenile cod and haddock 3-9 cm (settled 0-group)", "-", "submersible, Georges Bank 70-100 m, Aug 1986", "height off bottom of the part of the population that rose after sunset", "3-5", "m", "F"),
 (*L89, "juvenile cod and haddock 3-9 cm (settled 0-group)", "-", "submersible, day", "day position", "within a few cm of the bottom", "-", "F"),
 (*AE, "whiting < 20 cm (0-group)", "-", "GOV, 30 hauls, northern North Sea, Nov 1997", "night vs day catch rate", "night higher, within 2x", "-", "F"),
 (*AE, "Norway pout; haddock < 20 and > 20 cm", "-", "GOV, 30 hauls, Nov 1997", "day vs night catch rate", "day higher, within 2x", "-", "F"),
 (*AE, "whiting > 20 cm", "-", "GOV, 30 hauls, Nov 1997", "time-of-day effect", "not significant", "-", "F"),
 (*S11, "Maurolicus muelleri (PROXY)", "gadoids (blue whiting, saithe, haddock)", "bottom-mounted 38 kHz, 15 months, Masfjorden", "winter reverse DVM upward shift of part of the deep layer", "20-30", "m", "F"),
 (*S11, "Maurolicus muelleri (PROXY)", "-", "authors' calculation, K 0.1 /m", "light increase from 20-30 m shallower position", "7-20", "x", "F"),
 (*D12, "Benthosema glaciale (PROXY)", "-", "bottom-mounted echosounder + trawl, Masfjorden autumn", "daytime depth of inverse-migrating (feeding) group", "200-270", "m", "A"),
 ("Linehan Gregory & Schneider 2001 JEMBE 263:25", "NA (not checked)", "age-0 cod (coastal 0.7-20 m)", "piscivorous fish (coastal)", "tethering", "predation deep vs shallow, day and dusk", "~3", "x", "S"),
 ("Sogard & Olla 1996 MEPS 133:43 (via lit_activity_budgets.md)", "10.3354/meps133043", "juvenile walleye pollock (PROXY)", "-", "lab thermal gradient, ration series", "starved fish occupied colder water by", "3-4", "C", "S"),
 ("Sogard & Olla 1996 MEPS 133:43 (via lit_activity_budgets.md)", "10.3354/meps133043", "juvenile walleye pollock (PROXY)", "-", "lab thermal gradient, ration series", "energy saving in colder water", "up to 34", "%", "S"),
 ("Aglen et al. 1997 ICES CM 1997/Y:16 (via Overlapp i fiskparen)", "NA", "haddock large and medium (Barents)", "-", "acoustics", "day height range above bed (down at night; small haddock the reverse)", "0-100", "m", "S"),
 ("research_notes/Sejens hojd over botten section 3", "NA", "small haddock (Barents)", "-", "acoustics (as cited there)", "night rise", "30-40", "m", "S"),
 ("Vollset et al. 2013 MEPS 475:191 (via lit_activity_budgets.md)", "10.3354/meps10129", "cod larvae (PROXY stage)", "-", "lab column, 16 h unfed vs fed", "unfed deeper than fed", "4.3 in a 115", "cm", "S"),
 ("Biro Post & Parkinson 2003 Ecology 84:2419 (via lit_activity_budgets.md)", "10.1890/02-0416", "age-0 rainbow trout (PROXY)", "cannibal trout", "whole-lake food manipulation", "survival high-food vs low-food lakes", "+68", "%", "S"),
 ("research_notes/Sillens vinterprofil summary (D on GOV/SAM)", "NA", "herring 1-2 ringers (box)", "-", "GOV Q1/Q3, SAM catchability", "winter day bottom-10-m share multiplier k", "2 (1.5-3)", "x", "S"),
 ("research_notes/Planktivorernas golv E21 (SMS 2023)", "NA", "herring age 1", "fish predators", "SMS key run partial M2", "Q1 fish-predator M2 (1.58 x annual mean)", "0.38", "/yr", "S"),
 ("Hobson et al. 2007 MEPS 347:101 (via Overlapp i fiskparen)", "10.3354/meps07047", "cod", "-", "DSTs North Sea", "share of fish-days in near-inactive on-bed class", "~70", "%", "S"),
]

with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(H)
    for r in rows:
        assert len(r) == len(H), r
        w.writerow(r)
print(len(rows), "rows")
