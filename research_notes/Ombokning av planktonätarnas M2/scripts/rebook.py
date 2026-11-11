"""Re-booking the planktivores' and mackerel's M2 on krill and zooplankton from field-consistent rations.

Library only - no policy, no rollout, nothing written back. Each predator's checked pairs are scaled by
Q/B_new / Q/B_booked (the booked diet shares kept), then:
  - the prey's derived growth_rate through lib/world/growth_budget.derived_growth_rate (as the loader does);
  - for the krill open pool, the steady state B/T = (1 - e^-k)/(1 - (1 + g - m) e^-k) per tick, at the booked
    predation m, for g re-derived (= new booked M2) and for g kept (2.11, near the field P/B 2).
Ratios (research_notes/"Planktivorernas energibudget", "Makrillens ranson"):
  planktivores  5.0 -> 5.8 / 6.2 / 6.6 at the field fill  (x 1.16 / 1.24 / 1.32)
  mackerel      annual 2.2 -> presence-weighted SMS 2.75 (x 1.25) / energetic need 4.99 (x 2.27)
usage: python3 rebook.py"""
import copy, math, os, sys
sys.path.insert(0, '/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
import yaml
from lib.world.growth_budget import derived_growth_rate

L = yaml.safe_load(open('fgconfig/fg_library.yaml'))
S, I = L['species_definitions'], L['interaction_definitions']
ACTIVE = [s for s in S if s != 'seabirds']          # mareld2.yaml mutes only seabirds
TPY = 1460
PREYS = ('krill', 'zooplankton')
CASES = [("library (booked)", 1.0, 1.0),
         ("plankt 5.8, mack x1.25", 5.8 / 5.0, 2.75 / 2.2),
         ("plankt 6.2, mack x1.25", 6.2 / 5.0, 2.75 / 2.2),
         ("plankt 6.2, mack x2.27", 6.2 / 5.0, 4.99 / 2.2),
         ("plankt 6.6, mack x2.27", 6.6 / 5.0, 4.99 / 2.2)]


def m2_on(prey, inter):
    out = {}
    for pred in ACTIVE:
        e = inter.get(f"{pred}_preys_on_{prey}") or {}
        if e.get('preys_on') and float(e.get('predation_mortality') or 0) > 0:
            out[pred] = float(e['predation_mortality'])
    return out


def pool_ratio(g_yr, m_yr, k_yr):
    ek = math.exp(-k_yr / TPY)
    return (1 - ek) / (1 - (1 + (g_yr - m_yr) / TPY) * ek)


k_krill = float(S['krill']['outside_exchange_rate']) * TPY
g0 = {p: derived_growth_rate(p, S, I, active_predators=ACTIVE) for p in PREYS}
print(f"krill exchange k {k_krill:.1f} /yr; booked pairs:")
for p in PREYS:
    print(f"  {p}: " + ", ".join(f"{k} {v:g}" for k, v in m2_on(p, I).items()))
print()
print(f"{'case':26s} | {'krill M2':>8s} {'g /yr':>6s} {'B/T g=M2':>8s} {'B/T g kept':>10s} | "
      f"{'zoo M2':>6s} {'zoo g /tick':>11s} {'dg %':>6s}")
for name, fp, fm in CASES:
    inter = copy.deepcopy(I)
    for prey in PREYS:
        for pred, f in (('planktivorous_fish', fp), ('mackerel', fm)):
            e = inter.get(f"{pred}_preys_on_{prey}")
            if e and e.get('preys_on'):
                e['predation_mortality'] = float(e['predation_mortality']) * f
    mk = sum(m2_on('krill', inter).values()); mz = sum(m2_on('zooplankton', inter).values())
    gk = derived_growth_rate('krill', S, inter, active_predators=ACTIVE)
    gz = derived_growth_rate('zooplankton', S, inter, active_predators=ACTIVE)
    print(f"{name:26s} | {mk:8.3f} {gk * TPY:6.3f} {pool_ratio(gk * TPY, mk, k_krill):8.3f} "
          f"{pool_ratio(g0['krill'] * TPY, mk, k_krill):10.3f} | {mz:6.3f} {gz:11.6f} "
          f"{100 * (gz / g0['zooplankton'] - 1):6.2f}")
print()
print("realised krill M2 in fix96's probe 4.7 /yr: realised / booked = "
      + ", ".join(f"{4.7 / sum(m2_on('krill', I).values()):.2f} (library)" for _ in [0]))
