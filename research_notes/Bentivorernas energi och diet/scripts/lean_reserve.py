"""Lean body + reserve for benthivorous_fish (resume 223.9 decision 2a).

Usage: python3 -I lean_reserve.py <out_dir>

Per-species values (kJ/g wet = MJ/kg) are the ones chosen in
research_notes/"Fiskgrupper i rutan"/benthivores/scripts/group_params.py (ELEAN, EWHOLE, ME),
which in turn rest on research_notes/"Torskfiskars energiinnehall" (Swedish letters)/
energy_density.md and evidence_benthivores_A_bioenergetics.md. Sources per species are in
SOURCES below. Nothing here is fitted to model output.

Engine convention (read from the code, not assumed):
- predation.py / interactions.py: a prey with prey_includes_reserve gives the predator
  assimilation x (energy_content + fill_ref x ME) statically (energy_gain_mat, the gates, h)
  and, at run time, assimilation x (energy_content + actual reserve per tonne).
- population_change._charge_growth_to_reserve: each new tonne is debited energy_content from
  the reserve and dilutes the reserve, so in steady state a tonne produced costs
  energy_content + s x ME (s = the reserve fill the group lives at).
- growth_budget: s* from h(s) C = cost + P (ec + s ME); mass-balance ceiling g <= ME / ec.
So energy_content must be the LEAN body (the tissue a new tonne is built of) and ME the
mobilisable lipid store, exactly as planktivorous_fish (4000 + 6250) and predatory_fish
(4100 + 2400 at fill 0.48).
"""
import csv
import os
import sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

# Long-run 2001-25 NS-IBTS biomass shares (section 209; tables/long_run_species_by_group.csv).
W = {"haddock": 38.6, "cod": 31.4, "long rough dab": 12.6, "dab": 4.5, "plaice": 3.8,
     "witch": 1.7, "starry ray": 1.5, "hagfish": 1.3, "grey gurnard": 0.9, "lemon sole": 0.9,
     "greater weever": 0.8, "others": 2.5}
tot = sum(W.values())
W = {k: v / tot for k, v in W.items()}

# species: (lean kJ/g, whole-body annual mean kJ/g, max reserve MJ/t, tag, source)
S = {
    "haddock": (3.8, 4.6, 2000, "J",
                "cod proxy: no wild haddock HSI or whole-body ED obtained (Pedersen & Hislop 2001 "
                "closed, not OA). Fillet 3.29 kJ/g, protein 18.3-19.4 % (Matvaretabellen); muscle "
                "lipid ~1 % (gadoid reserve = liver; Nanton et al. 2003 cultured haddock muscle lipid "
                "0.8-1.9 %, liver the store). Whole 4.2-4.75"),
    "cod": (3.8, 4.75, 2000, "F/S",
            "lean: whole-minus-liver 3.67 kJ/g in E Baltic cod (Ryberg et al. 2020) to pre-spawning "
            "muscle 4.1 (Rojbek et al. 2012 water 81 %, Dutil et al. 1995 line); liver 920 MJ/t "
            "mean, 1665 wild autumn max, ~2070 well fed (Dutil 1995; Lambert & Dutil 1997); whole "
            "4.62-5.0 (Holdway & Beamish 1984; FB4 Hansson et al. 1996)"),
    "plaice": (3.6, 4.6, 1850, "F/J",
               "Fonds et al. 1985 Table 1 whole body 3.0 (K 0.75) - 6.1 (K 1.25) kJ/g; adult females "
               "5.81 Dec -> 4.10 Mar (Rijnsdorp 1983); lipid 3-22 % of DW, stored in the whole body; "
               "lean 3.5-4.0 J, reserve 5.8 - 4.1 = 1.7 kJ/g"),
    "dab": (3.6, 4.5, 1500, "J", "no data; plaice analogue, leaner"),
    "long rough dab": (2.6, 3.2, 1000, "A/J",
                       "MacKinnon 1972: a 35 cm female stores 92 of ~300 kcal in summer (31 %) "
                       "-> reserve 850-1100 MJ/t at 350-450 g; whole 2.8-3.6"),
    "witch": (4.0, 5.6, 1500, "F/J", "whole 5.6 kJ/g Oct (Spitz et al. 2010, an autumn value)"),
    "lemon sole": (4.0, 5.8, 1850, "F/J", "whole 5.8 kJ/g Oct (Spitz et al. 2010)"),
    "grey gurnard": (3.8, 4.4, 1500, "S/J", "whole 4.4 (Plimmer 1921 via JNCC Report 633)"),
    "starry ray": (3.6, 4.0, 1500, "J", "no data; elasmobranch liver store"),
    "hagfish": (3.5, 4.5, 1500, "J", "no data"),
    "greater weever": (4.0, 5.3, 1500, "F/J", "whole 5.3 Oct (Spitz et al. 2010)"),
    "others": (3.6, 4.5, 1500, "J", "group-like"),
}

lean = sum(W[k] * S[k][0] for k in W) * 1000
whole = sum(W[k] * S[k][1] for k in W) * 1000
me = sum(W[k] * S[k][2] for k in W)
fill = (whole - lean) / me

rows = []
with open(os.path.join(out, "lean_reserve_species.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["species", "weight", "lean_kJ_g", "whole_kJ_g", "max_reserve_MJ_t",
                 "implied_fill", "tag", "source"])
    for k in W:
        le, wh, m, tag, src = S[k]
        wr.writerow([k, round(W[k], 4), le, wh, m, round((wh - le) * 1000 / m, 3), tag, src])

print(f"{'species':16s} {'w':>6s} {'lean':>5s} {'whole':>6s} {'ME':>6s} {'fill':>6s}")
for k in W:
    le, wh, m, tag, _ = S[k]
    print(f"{k:16s} {W[k]:6.3f} {le:5.2f} {wh:6.2f} {m:6.0f} {(wh - le) * 1000 / m:6.3f} {tag}")
print()
print(f"weighted lean energy_content      {lean:7.0f} MJ/t (range 3400-3850, J)")
print(f"weighted whole body (annual mean) {whole:7.0f} MJ/t (3900-4800)")
print(f"weighted max_energy_reserve       {me:7.0f} MJ/t (1500-2100)")
print(f"reference fill (whole - lean)/ME  {fill:7.3f}")
print()
# Proposed library values (rounded; J): lean 3600, ME 1800, fill 0.50.
EC, ME, FILL = 3600.0, 1800.0, 0.50
print(f"PROPOSED energy_content {EC:.0f}, max_energy_reserve {ME:.0f}, reserve_reference_fill {FILL}")
print(f"  static prey quality = {EC:.0f} + {FILL} x {ME:.0f} = {EC + FILL * ME:.0f} MJ/t "
      f"(today: whole body 4500, no reserve eaten)")
print(f"  fill check: (weighted whole {whole:.0f} - {EC:.0f}) / {ME:.0f} = {(whole - EC) / ME:.3f}")
print(f"  mass-balance ceiling ME/ec = {ME / EC:.3f} per tick (today 1800/4500 = 0.400)")
print()
print("Per tonne at reserve fill s (engine): prey quality ec + s ME, production cost ec + s ME")
print(f"{'s':>5s} {'today eaten':>12s} {'new eaten':>10s} {'today cost/t':>13s} {'new cost/t':>11s}")
for s in (0.30, 0.50, 0.60, 0.72, 0.758, 0.80):
    print(f"{s:5.3f} {4500:12.0f} {EC + s * ME:10.0f} {4500 + s * 1800:13.0f} {EC + s * ME:11.0f}")
