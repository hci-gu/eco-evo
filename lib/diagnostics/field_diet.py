"""Field reference values for the diet diagnostics (sections 245, 248).

Shared by ``tools/probes/budget_gate.py`` (total-ration row, field-spread
diet gate) and ``tools/probes/probe_diets.py`` (realised diets of the
long-run probe). Diagnostic inputs only: nothing in the model is set from
them, and no library value may be fitted to the comparisons they feed.
"""
import numpy as np

# Field consumption per tonne, t/t/yr, and its source (section 245;
# research_notes/"Vitlingens ranson" (a-ring) summary.md and its sources).
FIELD_QB = {
    # Section 268: the source-consistent Q/B at the field fill (5.8-6.6; the
    # members' 5.0 pays only the lowest metabolism, "Planktivorernas energibudget").
    'planktivorous_fish': (6.2, "source-consistent Q/B (5.8-6.6), section 268"),
    'benthivorous_fish': (4.46, '223.9 budget'),
    'whiting': (4.2, 'SMS 2023, Lauerburg 2018 (3.3-5.6)'),
    'predatory_fish': (2.83, 'params_split_223 (2.6-3.6)'),
    # Section 268: the annual stock 2.2 per tonne PRESENT in the box (SMS
    # quarterly ration shares; "Makrillens ranson"); need 4.99.
    'mackerel': (2.75, 'stock Q/B 2.2 presence-weighted, section 268'),
}

# Bed-depth classes of the krill pairs' depth tables (section 245): < 100,
# 100-150, 150-250, >= 250 m.
BED_CLASS_EDGES_M = (100.0, 150.0, 250.0)

# Section 248: each group's biomass share per bed class in NS-IBTS GOV
# hauls inside the box (57.7-58.4 N, 9.9-11.3 E; 2000-20, Q1 + Q3 pooled,
# 354 hauls; GOV has none deeper than 300 m): mean kg/h per class x the
# box's cell share per class. The water column carries its dominant member,
# saithe (0.80 of the group, 0.94 of its krill a); research_notes/"Krill i
# Skagerrak" a_revisit/ (scripts/gov_by_bed.py), robust to the haul key
# (263.7). Whiting since section 264: that script keyed hauls on (year,
# quarter, ship, haul number), which added same-numbered hauls elsewhere in
# the North Sea to box hauls (was 0.533 / 0.211 / 0.251 / 0.006); the key
# with ShootLat gives the values below, research_notes/"Vitlingens uteblivna
# predation" (scripts/whiting_bed_shares_fixedkey.py).
FIELD_BED_SHARES = {
    'predatory_fish': ((0.043, 0.127, 0.782, 0.048), 'saithe, GOV in the box'),
    'whiting': ((0.769, 0.204, 0.027, 0.001), 'whiting, GOV in the box'),
}


# Section 270: a staged prey placed by stage at the field spread. The
# benthivores' adults (>= length at age 1) by bed class in NS-IBTS GOV
# 2000-25 in the box, Q1 and Q3 mean, Walker q (class shares); their J sit on
# the group's recruit_depth_table (``recruit_weight_map``, the recruits'
# arrival shape), uniform without one. research_notes/"Rekryternas
# bottendjup" summary.md.
FIELD_STAGE_BED_SHARES = {
    'benthivorous_fish': ((0.308, 0.441, 0.191, 0.060), 'adults, GOV in the box, annual'),
}


def bed_class(depth):
    """Class index 0-3 of every cell's bed depth (m)."""
    return np.digitize(np.asarray(depth, dtype=np.float64), BED_CLASS_EDGES_M)


def field_cell_weights(depth, shares):
    """Cell weights summing to 1: each class's share spread evenly over its
    cells. A class without cells drops out and the rest are renormalised."""
    cls = bed_class(depth)
    w = np.zeros(cls.shape, dtype=np.float64)
    for k, share in enumerate(shares):
        cells = cls == k
        if cells.any():
            w[cells] = float(share) / cells.sum()
    total = w.sum()
    if total <= 0.0:
        raise ValueError("no cell lies in a bed class with a positive share")
    return w / total


def field_prey_shape(prey_id, fg, depth):
    """Section 252: a prey's spatial shape for the field-spread diet gate, mean 1
    over the cells: an open pool's own depth shape (``outside_weight_map``), a
    group with GOV bed-depth shares spread as in the field, otherwise uniform.
    Never the spawned start map (random colonies, independent of depth)."""
    depth = np.asarray(depth, dtype=np.float64)
    shape = getattr(fg, 'outside_weight_map', None)
    if shape is not None:
        shape = np.asarray(shape, dtype=np.float64)
        return shape / shape.mean()
    if prey_id in FIELD_BED_SHARES:
        return field_cell_weights(depth, FIELD_BED_SHARES[prey_id][0]) * depth.size
    return np.ones(depth.shape, dtype=np.float64)


def field_stage_shapes(prey_id, fg, depth):
    """Section 270: ``(J shape, adult shape)`` of a staged prey for the
    field-spread gate, each mean 1 over the cells, or None when the prey has
    no field stage shares. J follows the group's recruit import shape
    (``recruit_weight_map``; uniform without one), the adults their GOV bed
    shares."""
    if prey_id not in FIELD_STAGE_BED_SHARES:
        return None
    depth = np.asarray(depth, dtype=np.float64)
    adult = field_cell_weights(depth, FIELD_STAGE_BED_SHARES[prey_id][0]) * depth.size
    juv = getattr(fg, 'recruit_weight_map', None)
    if juv is None:
        juv = np.ones(depth.shape, dtype=np.float64)
    else:
        juv = np.asarray(juv, dtype=np.float64)
        juv = juv / juv.mean()
    return juv, adult
