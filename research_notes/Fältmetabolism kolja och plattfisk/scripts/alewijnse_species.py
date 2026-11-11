"""Summarise the Alewijnse (2022/2023) otolith C_resp dataset for the benthivore members and
comparison species, and convert C_resp to O2 consumption with the Chung et al. 2019 cod calibration
C_resp = C (1 - exp(-k VO2)), C 0.243, k 8.80e-3 (VO2 in mg O2/kg/h), and with the linear form used by
Jones et al. 2023 (VO2 = (C_resp - 0.041)/9.71e-4).
Usage: python3 -I alewijnse_species.py <path to alewijnse_master_data.csv>
"""
import csv, math, sys, statistics as st

SPP = ["Gadus_morhua", "Melanogrammus_aeglefinus", "Merlangius_merlangus", "Pollachius_virens",
       "Micromesistius_poutassou", "Boreogadus_saida", "Hippoglossoides_platessoides",
       "Glyptocephalus_cynoglossus", "Limanda_limanda", "Microstomus_kitt", "Lepidorhombus_whiffiagonis",
       "Solea_solea", "Scophthalmus_maximus", "Cyclopterus_lumpus", "Scomber_scombrus", "Sprattus_sprattus",
       "Trachurus_trachurus", "Merluccius_merluccius"]
C, K = 0.243, 8.80e-3

def vo2_exp(c):
    if c >= C:
        return float("nan")
    return -math.log(1 - c / C) / K

def vo2_lin(c):
    return (c - 0.041) / 9.71e-4

rows = list(csv.DictReader(open(sys.argv[1])))
print("species | n | C_resp mean (sd among fish) | min-max | T mean (range) C | mass median (range) g | "
      "VO2 exp-cal at mean C_resp | VO2 lin-cal | sources | lat range | sample types | years")
for s in SPP:
    r = [x for x in rows if x["species"] == s]
    if not r:
        continue
    c = [float(x["c_resp_mean"]) for x in r]
    t = [float(x["temp_mean"]) for x in r]
    m = [float(x["body_mass"]) for x in r if x["body_mass"] not in ("", "NA")]
    lat = [float(x["lat"]) for x in r if x["lat"] not in ("", "NA")]
    srcs = sorted(set(x["source"] for x in r))
    typ = sorted(set(x["sample_type"] for x in r))
    yrs = sorted(set(x["year"] for x in r))
    cm = st.mean(c)
    sd = st.stdev(c) if len(c) > 1 else 0
    mm = (f"{st.median(m):.0f} ({min(m):.0f}-{max(m):.0f})" if m else "-")
    la = f"{min(lat):.1f}-{max(lat):.1f}" if lat else "-"
    print(f"{s} | {len(r)} | {cm:.3f} ({sd:.3f}) | {min(c):.3f}-{max(c):.3f} | {st.mean(t):.1f} "
          f"({min(t):.1f}-{max(t):.1f}) | {mm} | {vo2_exp(cm):.0f} | {vo2_lin(cm):.0f} | {srcs} | {la} | {typ} | "
          f"{yrs[0]}-{yrs[-1]}")

print()
print("Per-fish detail for the benthivore members and cod:")
for s in ["Gadus_morhua", "Melanogrammus_aeglefinus", "Hippoglossoides_platessoides", "Glyptocephalus_cynoglossus",
          "Limanda_limanda", "Microstomus_kitt"]:
    for x in rows:
        if x["species"] == s:
            c = float(x["c_resp_mean"])
            print(f"  {s} {x['specimen_id']} C_resp {c:.3f}+-{float(x['c_resp_sd']):.3f} T {float(x['temp_mean']):.1f} "
                  f"mass {x['body_mass']} g lat {x['lat']} lon {x['long']} year {x['year']} {x['sample_type']} "
                  f"n {x['n']} {x['source']} VO2exp {vo2_exp(c):.0f} diet {float(x['diet_d13c_mean']):.1f} "
                  f"DIC {float(x['dic_d13c_mean']):.2f} oto {x['oto_d13c_mean']}")
