"""Model-side yardsticks for the demersal piscivores' fill (research only).

Everything here is arithmetic on library values (fg_library.yaml / mareld2.yaml as of
2026-11-20, copied as constants below) and on field numbers from earlier notes. The
fix104 feeding levels are model OUTPUT and are used only as a yardstick (the size of the
gap to close), never to set a value.

Run: python3 -I scripts/model_side.py > tables/model_side_stdout.txt
"""

TICKS_PER_YEAR = 1460
NET = 0.80 - 0.12          # assimilation_factor (every predator pair) - sda_fraction

GROUPS = {
    # ceiling MJ/t/tick (ingested) at the layer's annual mean; cost = RM x feeding_cost
    "whiting": dict(ceiling=31.34, rm=3.87, feed=1.50, lean=3900.0, me=2000.0, u=0.07,
                    P=0.61, layer="bottom",
                    fix104_feed=0.383, fix104_uncapped=0.664),
    "water_column": dict(ceiling=29.79, rm=3.29, feed=2.14, lean=4180.0, me=2000.0, u=0.07,
                         P=0.42, layer="at_50m",
                         fix104_feed=0.438, fix104_uncapped=0.731),
}
LAYERS = {  # mareld2.yaml simulation_settings.temperature.layers (monthly, C)
    "at_50m": [7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3],
    "bottom": [7.8, 6.99, 6.59, 6.58, 6.69, 6.96, 7.22, 7.41, 7.7, 8.04, 8.36, 8.5],
}
Q10_COST, Q10_INTAKE = 2.3, 2.25


def kjkgd(mj_t_tick):
    return mj_t_tick * 4.0  # 1 MJ/t/tick = 4 kJ/kg/d


def growth_demand(g, s):
    """MJ/t/tick the derived growth law asks at fill s (reserve_room: s* = 1, so
    G(s) = P/1460 x (s-u)/(1-u); price = lean + s x ME, no growth_cost on these groups)."""
    frac = max(s - g["u"], 0.0) / (1.0 - g["u"])
    return g["P"] / TICKS_PER_YEAR * frac * (g["lean"] + s * g["me"])


def section(t):
    print("\n" + t + "\n" + "-" * len(t))


section("1. Units: ceiling, cost, need at fill 1 (annual-mean temperature)")
for name, g in GROUPS.items():
    cost = g["rm"] * g["feed"]
    need1 = cost + growth_demand(g, 1.0)
    print(f"{name:13s} ceiling {g['ceiling']:.2f} MJ/t/tick = {kjkgd(g['ceiling']):.1f} kJ/kg/d; "
          f"cost {cost:.2f} ({kjkgd(cost):.1f} kJ/kg/d) = {cost/(NET*g['ceiling']):.3f} of ceiling (ingested); "
          f"cost + P at fill 1 = {need1:.2f} -> feeding level {need1/(NET*g['ceiling']):.3f} "
          f"(fix104 realised {g['fix104_feed']:.3f}, uncapped {g['fix104_uncapped']:.3f})")

section("2. Feeding level a fish needs to hold fill s (equilibrium, no season)")
print("fill s:        " + " ".join(f"{s:6.2f}" for s in (0.1, 0.3, 0.5, 0.71, 0.73, 0.9, 1.0)))
for name, g in GROUPS.items():
    cost = g["rm"] * g["feed"]
    row = [(cost + growth_demand(g, s)) / (NET * g["ceiling"]) for s in (0.1, 0.3, 0.5, 0.71, 0.73, 0.9, 1.0)]
    print(f"{name:13s} " + " ".join(f"{x:6.3f}" for x in row))
print("Reading: the whole fill range 0.1-1.0 lies inside a feeding-level band of ~0.1 of the ceiling,\n"
      "so a fill below 1 needs a potential feeding level within that band; the model's uncapped\n"
      "potential (fix104) is ~0.25-0.3 above its top.")

section("3. Refill speed: model surplus vs the field refill time")
for name, g in GROUPS.items():
    surplus = (g["fix104_uncapped"] - g["fix104_feed"]) * g["ceiling"] * NET  # MJ/t/tick
    for dfill, label in ((0.43, "whiting proxy 0.30 -> 0.73"), (0.25, "saithe NIFES-like ~0.25 ME")):
        days_model = dfill * g["me"] / surplus / 4.0
        out = []
        for days_field in (60, 90, 150, 180):
            s_field = dfill * g["me"] / (days_field * 4.0)
            out.append(f"{days_field} d -> surplus {s_field:.2f} MJ/t/tick, potential "
                       f"{g['fix104_feed'] + s_field/(NET*g['ceiling']):.3f}")
        print(f"{name:13s} {label}: model surplus {surplus:.2f} MJ/t/tick refills in {days_model:.0f} d; field: "
              + "; ".join(out))

section("4. Seasonal ration (SMS, stomach-based) -> implied reserve swing, closure imposed")
# SMS quarterly rations (%BW/d), ICES WGSAM 2024 Table 3-8 via
# research_notes/"Fodonivå och mattnad i falt"/tables/A_ration_over_cmax_stdout.txt
SMS = {
    "whiting 120 g": ("whiting", [0.915, 0.976, 1.458, 1.233]),
    "whiting 250 g": ("whiting", [0.725, 0.773, 1.155, 0.977]),
    "saithe 1.5 kg": ("water_column", [0.439, 1.27, 0.673, 0.787]),
}
for label, (gname, q) in SMS.items():
    g = GROUPS[gname]
    temps = LAYERS[g["layer"]]
    qT = [sum(temps[3 * i:3 * i + 3]) / 3 for i in range(4)]
    tmean = sum(temps) / 12
    for e_diet in (4.0, 4.5, 5.5):
        gain = [x / 100 * e_diet * 1000 / 4 * NET for x in q]          # MJ/t/tick net
        cost = [g["rm"] * g["feed"] * Q10_COST ** ((t - tmean) / 10) for t in qT]
        grow = growth_demand(g, 0.5)                                    # flat over the year
        net = [a - c - grow for a, c in zip(gain, cost)]
        annual = sum(net) / 4
        # closure: remove the annual mean imbalance (the fill is cyclic), keep the shape
        shape = [(n - annual) * 365.25 for n in net]                    # MJ/t per quarter
        traj, f = [], 0.0
        for d in shape:
            f += d / g["me"]
            traj.append(f)
        lo = min([0.0] + traj)
        traj = [x - lo for x in [0.0] + traj]
        print(f"{label:14s} E {e_diet:.1f} kJ/g: annual imbalance {annual:+.2f} MJ/t/tick "
              f"({annual/(g['rm']*g['feed']):+.0%} of cost); quarter net-mean (ME) "
              + " ".join(f"{d/g['me']:+.2f}" for d in shape)
              + "; fill from the minimum at 1 Jan/1 Apr/1 Jul/1 Oct/31 Dec: "
              + " ".join(f"{x:.2f}" for x in traj))
print("Caveats: SMS consumption = stomach content x a temperature-dependent evacuation model with\n"
      "North Sea temperatures, so part of the Q3 rise is temperature in that model, not food; growth\n"
      "is spread evenly (field whiting grow mainly Q2-Q3, which deepens the summer trough); saithe\n"
      "SMS is the North Sea stock (mostly adults that spawn in Q1), the box fish are mostly immature.")
