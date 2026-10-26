"""Independent check: share of each North Sea benthic group's production eaten
by fish, from the balanced Ecopath model of Mackinson & Daskalov 2007 (Cefas
Sci. Ser. Tech. Rep. 142; Table 3.3 parameters, Table 3.4 diet matrix).

Not used to set any value. Run with ``python3 -I``:
    python3 -I scripts/md2007_fish_share.py <tech142.pdf> <scratch_dir>
It calls ``pdftotext -bbox`` on the diet-matrix pages (31-42) and assigns each
number to the predator column whose header number lies closest in x.
Output: tables/md2007_fish_share_stdout.txt and tables/md2007_fish_share.csv.
"""
import csv
import html
import os
import re
import subprocess
import sys

PDF, SCRATCH = sys.argv[1], sys.argv[2]
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "tables", "md2007_fish_share.csv")

# Table 3.3 (balanced values): group -> (name, B t/km2, P/B, Q/B, EE);
# Table 3.5: landings + discards (t over the model area, 570 000 km2).
PARAMS = {
    54: ("Large crabs", 1.354, 0.5541, 2.7704, 0.9322, 6555 + 573),
    55: ("Nephrops", 1.1, 0.37, 1.85, 0.9937, 10317 + 10),
    56: ("Epifaunal macrobenthos (mobile grazers)", 78.0, 0.3884, 1.942,
         0.4315, 3762),
    57: ("Infaunal macrobenthos", 136.0, 1.0, 3.3333, 0.2791, 51756 + 28443),
    58: ("Shrimp", 0.50, 3.0, 10.0, 0.4486, 24738),
    59: ("Small mobile epifauna (swarming crustaceans)", 30.0, 1.9, 5.4286,
         0.9249, 0),
    60: ("Small infauna (polychaetes)", 150.0, 0.9, 3.0, 0.9079, 0),
}
AREA_KM2 = 570000.0
# DEFECT IN THE SOURCE PDF: the lower half of the diet-matrix page for
# predator columns 13-24 (pdf page 34) repeats the numbers of columns 1-12
# (page 32) row for row, so the gadoid columns (cod, whiting, haddock,
# saithe, Norway pout ...) are lost for every invertebrate prey. Fish
# consumption is therefore taken as the remainder:
#   fish = EE x P - catch - (consumption by non-fish predators),
# with the non-fish predators (columns 1-4 and 49-68) read from pages that
# are not duplicated. Columns 13-24 on page 34 are skipped.
CORRUPT = {34: set(range(13, 25))}
PREY_ROWS = list(PARAMS)
# Predator groups 1-68: fish are 5-48 (sharks .. misc. filterfeeding
# pelagic fish); 1-4 mammals and birds, 49 squid, 50 larvae, 51+ invertebrates.
FISH = set(range(5, 49))


def words(page):
    xml = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), "-bbox", PDF, "-"],
        capture_output=True, text=True, check=True).stdout
    out = []
    for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" '
                         r'xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', xml):
        x0, y0, x1, y1 = map(float, m.groups()[:4])
        out.append(((x0 + x1) / 2, (y0 + y1) / 2, html.unescape(m.group(5))))
    return out


def parse_page(page):
    ws = words(page)
    # header: the line with "Predator" and integers
    hdr = [w for w in ws if w[2] == "Predator"]
    if not hdr:
        return {}
    y_h = hdr[0][1]
    cols = sorted([(w[0], int(w[2])) for w in ws
                   if abs(w[1] - y_h) < 3 and w[2].isdigit()])
    if not cols:
        return {}
    # prey rows: a word equal to the prey number in the left margin, x small
    result = {}
    x_first_col = cols[0][0]
    for num in PREY_ROWS:
        cands = [w for w in ws if w[2] == str(num) and w[0] < x_first_col - 60
                 and w[1] > y_h + 5]
        if not cands:
            continue
        y = cands[0][1]
        vals = [w for w in ws if abs(w[1] - y) < 2.5
                and re.fullmatch(r"\d*\.\d+", w[2]) and w[0] > x_first_col - 25]
        for x, _, t in vals:
            col = min(cols, key=lambda c: abs(c[0] - x))
            result.setdefault(num, {})[col[1]] = float(t)
    return result


def main():
    diet = {}
    for page in range(31, 43):
        for prey, d in parse_page(page).items():
            for pred, v in d.items():
                if pred in CORRUPT.get(page, ()):
                    continue
                diet.setdefault(prey, {})[pred] = v
    # predator B and Q/B from Table 3.3: read the layout text of pages 28-30
    txt = subprocess.run(["pdftotext", "-f", "26", "-l", "30", "-layout", PDF,
                          "-"], capture_output=True, text=True,
                         check=True).stdout
    predBQ = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(\d{1,2})\s+(.+?)\s{2,}([\d.]+)\s+([\d.]+)(?:\s*\([^)]*\))?"
                     r"\s+([\d.]+)(?:\s*\([^)]*\))?\s+([\d.]+)", line)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 68 and n not in predBQ:
                predBQ[n] = (m.group(2).strip(), float(m.group(4)),
                             float(m.group(6)))
    # Rows whose layout breaks the regex (name runs into the trophic level,
    # or the name wraps), checked by eye against Table 3.3 (balanced values).
    OVERRIDE = {
        15: ("Juvenile whiting (0-1)", 0.222, 6.58),
        17: ("Juvenile haddock (0-1)", 0.284, 5.39),
        48: ("Misc. filterfeeding pelagic fish", 0.0298, 10.19),
        52: ("Herbivorous & omnivorous zooplankton", 16.0, 30.0),
        56: ("Epifaunal macrobenthos (mobile grazers)", 78.0, 1.942),
        59: ("Small mobile epifauna", 30.0, 5.4286),
        60: ("Small infauna (polychaetes)", 150.0, 3.0),
        62: ("Meiofauna", 4.1071, 125.0),
    }
    predBQ.update(OVERRIDE)
    lines = []
    p = lines.append
    p("Mackinson & Daskalov 2007 North Sea Ecopath (1991, balanced): consumption")
    p("of each benthic group by predator class, as a share of its production.")
    p(f"predators parsed with B and Q/B: {len(predBQ)} of 68")
    rows = []
    rows_bounds = []
    for prey in PREY_ROWS:
        name, B, PB, _, EE, catch_t = PARAMS[prey]
        P = B * PB
        catch = catch_t / AREA_KM2
        pred_total = EE * P - catch
        nonfish, fish_read, top = 0.0, 0.0, []
        for pred, frac in diet.get(prey, {}).items():
            if pred not in predBQ:
                continue
            _, Bj, QBj = predBQ[pred]
            c = Bj * QBj * frac
            if pred in FISH:
                fish_read += c
            else:
                nonfish += c
            top.append((c, predBQ[pred][0]))
        top.sort(reverse=True)
        fish = max(pred_total - nonfish, 0.0)
        p(f"{prey} {name}: B {B}, P/B {PB}, P {P:.3f} t/km2/yr, EE {EE}; "
          f"catch {catch:.4f} ({catch / P:.1%} of P); all predation "
          f"{pred_total:.3f} ({pred_total / P:.1%}); non-fish predators "
          f"{nonfish:.3f} ({nonfish / P:.1%}); FISH (remainder) {fish:.3f} "
          f"= {fish / P:.1%} of P (fish columns readable outside 13-24: "
          f"{fish_read:.3f})")
        gad = sum(predBQ[j][1] * predBQ[j][2] for j in range(13, 25))
        lo_f, hi_f = fish_read, min(fish, fish_read + gad)
        p(f"    BOUNDS fish share of P: {lo_f / P:.1%} (readable columns only)"
          f" - {hi_f / P:.1%} (min of the EE remainder and readable + the "
          f"whole consumption of columns 13-24, {gad:.2f} t/km2/yr)")
        rows_bounds.append((name, lo_f / P, hi_f / P))
        p("    readable consumers: " + "; ".join(
            f"{n} {c:.3f}" for c, n in top[:6]))
        rows.append(dict(group=name, md_group=prey, B=B, PB=PB, P=round(P, 4),
                         EE=EE, catch=round(catch, 5),
                         predation_total=round(pred_total, 4),
                         nonfish_predation=round(nonfish, 4),
                         fish_predation_remainder=round(fish, 4),
                         fish_share_of_P=round(fish / P, 4)))
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p("")
    p("The EE remainder is NOT reliable where it exceeds what the lost gadoid")
    p("columns can eat (polychaetes, swarming crustaceans): EE values of the")
    p("balanced model are partly Ecopath closures. Use the bounds; the earlier")
    p("note's diet-matrix values (Bentos och torskfiskar/evidence_C) were")
    p("computed before the page-34 duplication was noticed.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
