"""Crowding of a group over the grid, against the field (section 278).

Lloyd's mean crowding over the mean: the density the average tonne of a
group has around it, over the group's mean density,

    crowding = n * sum(b^2) / (sum b)^2,

n the number of cells. 1 is an even spread; a group packed into k equal
cells of n reads n / k. The long-run probe records each decision maker's
annual mean of the daily value (``run_probe``'s ``crowding``); it is a
diagnostic, never a reward. tools/probes/probe_crowding.py prints a run's
probes against the field values below.

Field values: NS-IBTS GOV, Q1 and Q3 2021-25, Skagerrak + box rectangles,
wing-spread swept area, q = 1, the same statistic per survey (year x
quarter), median and range over the 10 surveys. research_notes/"Interferens
hos bottenfisk" scripts/ibts_patchiness.py (benthivores) and
scripts/ibts_crowding_groups.py (every group). A haul (0.07-0.15 km2, a
3.4 km tow) is not a 1 km cell and the survey region is wider than the box,
so the field reading is a scale-matched order of magnitude, not a target.
"""
import numpy as np

# group: (median, low, high, source note)
FIELD_CROWDING = {
    "benthivorous_fish": (3.06, 2.11, 3.80, "NS-IBTS GOV, guild B"),
    "whiting": (4.06, 1.75, 5.32, "NS-IBTS GOV"),
    "predatory_fish": (4.95, 2.84, 21.32, "NS-IBTS GOV, the group's members (saithe alone 14.0)"),
    "planktivorous_fish": (4.36, 1.68, 9.42, "NS-IBTS GOV, guild P; GOV catches pelagics poorly"),
    "mackerel": (6.33, 1.79, 25.0, "NS-IBTS GOV, mackerel + horse mackerel; GOV q ~0.1"),
}


def crowding(biomass):
    """Lloyd's mean crowding over the mean of one group's biomass field, or
    None when the group has no (finite, positive) biomass."""
    b = np.asarray(biomass, dtype=np.float64).ravel()
    total = float(b.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return float(b.size * float(np.square(b).sum()) / total ** 2)
