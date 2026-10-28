"""Haul-scale conspecific density vs stomach content, Baltic cod and flounder.

Data: Lindmark et al. (bioRxiv 10.1101/2024.04.22.590538, CC-BY 4.0), GitHub
maxlindmark/cod-interactions, data/clean/aggregated_stomach_data.csv (BITS
2015-2022, SW Baltic Proper, stomachs matched to haul-level biomass density in
kg/km2 of small cod, large cod and flounder from the same haul).

Does three things:
 1. SD of ln(density) as the authors scale it, to turn their standardized
    coefficients (read from their Fig. 4) into elasticities.
 2. Own quasi-Poisson GLM (log link) of relative prey weight on ln(own-group
    density) + ln(other densities) + year + quarter + depth + oxygen +
    predator length, haul-clustered SEs. No spatial random field (theirs has
    one), so this is a cross-check, not a replication.
 3. Spatial patchiness of the haul densities: Lloyd's mean crowding /
    mean = sum(d^2)/sum(d)/mean(d) per year-quarter (the field analogue of the
    model's "biomass-weighted own density / grid mean"), and quantiles of
    d / mean(d).
Usage: python3 -I lindmark_density_feeding.py <aggregated_stomach_data.csv> <out_dir>
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BENTHIC = ["amphipoda_tot", "bivalvia_tot", "gadus_morhua_tot", "gobiidae_tot",
           "mysidae_tot", "non_bio_tot", "other_crustacea_tot", "other_tot",
           "other_pisces_tot", "platichthys_flesus_tot", "polychaeta_tot",
           "saduria_entomon_tot"]


def prepare(path):
    d = pd.read_csv(path)
    d = d.dropna(subset=["group", "oxy"]).copy()
    tot_cols = [c for c in d.columns if c.endswith("_tot")]
    d["tot_weight"] = d[tot_cols].sum(axis=1)
    d["benthic_weight"] = d[BENTHIC].sum(axis=1)
    body = d["pred_weight_g"] - d["tot_weight"]
    d["tot_rel"] = d["tot_weight"] / body
    d["ben_rel"] = d["benthic_weight"] / body
    for c in ["kg_km2_small_cod", "kg_km2_large_cod", "kg_km2_flounder"]:
        pos = d.loc[d[c] > 0, c].min()
        d[c + "_f"] = np.where(d[c] == 0, 0.5 * pos, d[c])
        d["ln_" + c] = np.log(d[c + "_f"])
    d["fyear"] = d["year"].astype(str)
    d["fq"] = d["quarter"].astype(str)
    return d


def sd_table(d):
    rows = []
    for c in ["kg_km2_small_cod", "kg_km2_large_cod", "kg_km2_flounder"]:
        rows.append(dict(variable="ln " + c, sd_all_rows=d["ln_" + c].std()))
    return pd.DataFrame(rows)


def glm(d, group, resp, own, others):
    dd = d[d["group"] == group].copy()
    dd = dd[np.isfinite(dd[resp])]
    f = (f"{resp} ~ ln_{own} + " + " + ".join("ln_" + o for o in others)
         + " + C(fyear) + C(fq) + depth + oxy + pred_length_cm")
    m = smf.glm(f, data=dd, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["haul_id"])[0]})
    b = m.params["ln_" + own]
    se = m.bse["ln_" + own]
    return dict(group=group, response=resp, own_density=own, n=len(dd),
                n_hauls=dd["haul_id"].nunique(), elasticity=b, se=se,
                ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se,
                cut_per_doubling=1 - 2 ** b, ratio_at_13x=13 ** b)


def patchiness(d):
    h = d.drop_duplicates("haul_id")[["haul_id", "year", "quarter",
                                      "kg_km2_large_cod", "kg_km2_flounder",
                                      "kg_km2_small_cod"]]
    rows = []
    for c in ["kg_km2_large_cod", "kg_km2_flounder", "kg_km2_small_cod"]:
        for (y, q), g in h.groupby(["year", "quarter"]):
            x = g[c].values
            if x.sum() <= 0 or len(x) < 10:
                continue
            m = x.mean()
            rows.append(dict(var=c, year=y, quarter=q, n_hauls=len(x),
                             mean_t_km2=m / 1000.0,
                             lloyd_crowding_over_mean=(x ** 2).sum() / x.sum() / m,
                             p50_over_mean=np.quantile(x, 0.5) / m,
                             p90_over_mean=np.quantile(x, 0.9) / m,
                             max_over_mean=x.max() / m,
                             max_t_km2=x.max() / 1000.0))
    return pd.DataFrame(rows)


def main():
    path, out = sys.argv[1], sys.argv[2]
    d = prepare(path)
    sd = sd_table(d)
    print(sd.to_string(index=False))
    res = [
        glm(d, "large cod", "tot_rel", "kg_km2_large_cod",
            ["kg_km2_small_cod", "kg_km2_flounder"]),
        glm(d, "large cod", "ben_rel", "kg_km2_large_cod",
            ["kg_km2_small_cod", "kg_km2_flounder"]),
        glm(d, "small cod", "ben_rel", "kg_km2_small_cod",
            ["kg_km2_large_cod", "kg_km2_flounder"]),
        glm(d, "flounder", "ben_rel", "kg_km2_flounder",
            ["kg_km2_small_cod", "kg_km2_large_cod"]),
    ]
    res = pd.DataFrame(res)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    p = patchiness(d)
    summ = p.groupby("var").agg(n_surveys=("year", "size"),
                                mean_t_km2=("mean_t_km2", "median"),
                                lloyd_median=("lloyd_crowding_over_mean", "median"),
                                lloyd_min=("lloyd_crowding_over_mean", "min"),
                                lloyd_max=("lloyd_crowding_over_mean", "max"),
                                p90_over_mean=("p90_over_mean", "median"),
                                max_over_mean=("max_over_mean", "median"),
                                max_t_km2=("max_t_km2", "max"))
    print(summ.round(2).to_string())
    sd.to_csv(f"{out}/lindmark_sd_log_density.csv", index=False)
    res.to_csv(f"{out}/lindmark_glm_elasticities.csv", index=False)
    p.to_csv(f"{out}/lindmark_patchiness_by_survey.csv", index=False)
    summ.to_csv(f"{out}/lindmark_patchiness_summary.csv")


if __name__ == "__main__":
    main()
