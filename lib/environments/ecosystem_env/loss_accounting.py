"""Per-FG biomass-loss breakdown shared by train.py, inference.py and the viz.

The environment accumulates the biomass each cause removed over a rollout:
``loss_starvation`` (catabolism below the maintenance level),
``loss_predation`` (eaten by a decision maker), ``loss_impact`` (impact
mortality) and ``loss_natural`` (the residual natural mortality M1,
``natural_mortality``). M1 was missing from the breakdown until section
132, so a group whose dominant loss was M1 showed up as "100 % predation"
in the live plot.
"""

LOSS_CAUSES = (
    ("starvation", "loss_starvation"),
    ("predation", "loss_predation"),
    ("impact", "loss_impact"),
    ("natural", "loss_natural"),
)


def loss_shares(env, fid):
    """{'starvation', 'predation', 'impact', 'natural', 'total'} for ``fid``.

    The four shares sum to 1.0 when anything was lost, otherwise all are
    0.0; ``total`` is the biomass removed (tonnes).
    """
    amounts = {name: float((getattr(env, attr, None) or {}).get(fid, 0.0))
               for name, attr in LOSS_CAUSES}
    total = sum(amounts.values())
    out = {name: (value / total if total > 0.0 else 0.0)
           for name, value in amounts.items()}
    out["total"] = total
    return out
