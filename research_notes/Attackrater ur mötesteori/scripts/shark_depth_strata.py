"""List the depth strata sampled for zooplankton at SHARK station A17 and, where a visit
has several strata, the share of copepod abundance (ind m-2) per stratum.

Usage: python3 -I shark_depth_strata.py <a17_zoo.txt>
Input is untrusted data (SHARK download), only parsed.
"""
import sys
import pandas as pd

d = pd.read_csv(sys.argv[1], sep="\t", low_memory=False)
d["zmin"] = pd.to_numeric(d.sample_min_depth_m, errors="coerce")
d["zmax"] = pd.to_numeric(d.sample_max_depth_m, errors="coerce")
print("parameters:", d.parameter.value_counts().to_dict())
strata = d.groupby(["zmin", "zmax"]).agg(n_visits=("visit_id", "nunique"),
                                         years=("visit_year", lambda s: f"{s.min()}-{s.max()}"))
print(strata.to_string())
ab = d[d.parameter.str.contains("Abundance", case=False, na=False)].copy()
ab["value"] = pd.to_numeric(ab.value, errors="coerce")
ab = ab[ab.unit.astype(str).str.contains("ind/m3")]
cop = ab[ab.scientific_name.astype(str).str.contains(
    "Calanus|Pseudocalanus|Paracalanus|Temora|Acartia|Centropages|Oithona|Copepoda|Metridia|Microcalanus|Clausocalanus",
    regex=True)]
cop = cop.assign(cal=cop.scientific_name.str.contains("Calanus finmarchicus|Calanus helgolandicus|^Calanus$", regex=True))
per = cop.groupby(["visit_id", "sample_date", "zmin", "zmax", "cal"]).value.sum().reset_index()
per["ind_m2"] = per.value * (per.zmax - per.zmin)
multi = per.groupby("visit_id").filter(lambda g: g[["zmin", "zmax"]].drop_duplicates().shape[0] > 1)
if len(multi):
    t = multi.pivot_table(index=["visit_id", "sample_date", "cal"], columns=["zmin", "zmax"],
                          values="ind_m2", aggfunc="sum")
    print("\nvisits with several strata, copepod ind m-2 per stratum (cal=True: Calanus):")
    print(t.to_string())
    tot = t.sum(axis=1)
    print("\nshare per stratum:")
    print(t.div(tot, axis=0).round(2).to_string())
else:
    print("no visit with several strata")
