"""Hide-floor readings for the planktivore pairs, and what a floor does to the winter visible share.

Usage: python3 -I floor_readings.py

Part 1 - the only refuge that could survive the combined derivation (research_notes/"Fangst och stim i
fiskparen") inside the same cell and tick: a RESTING planktivore schooled more tightly than the
population mean by day (sigma_rest = 1 instead of the population sigma). Night terms unchanged
(schools dissolve in the dark whatever the activity). Per pair:
    floor = sum_t s_t F_rest,t / sum_t s_t F_t,  F = 1 - sigma (1 - S),
s_t, S_t (central, library p) and sigma from research_notes/"Stimskydd i motes-a"/scripts/
school_factor_stdout.txt; S scaled to the combined derivation's new p (S ~ 1/p, capped at 1);
pout sigma day 0.4 / night 0 and blue whiting 0.2 / 0 from "Vitlinglojans stim" (the combined set).
Readings: A = herring and sprat day terms only; B = pout day terms too. An UPPER bound on that
protection: the population sigma already contains the resting fish, and Pitcher et al. 1996 saw herring
keep feeding inside schools under gadoid attack (evidence.md E14).

Part 2 - visible share v = 1 - r (1 - floor) at the winter rest shares implied by the observed v at
floor 0.25 (0.28-0.50 in Nov-Feb, fix78/fix80, resume 223.27, 223.31; descriptive, never a target).
"""
# pair: (p_lib, p_new, [(prey, period, share, S_central_at_p_lib, sigma_pop)])
PAIRS = {
    "WC -> planktivores": (0.2, 0.592, [
        ("pout", "day", 0.544, 0.283, 0.4), ("herring", "day", 0.281, 0.14, 0.9),
        ("blue whiting", "day", 0.105, 0.617, 0.2), ("pout", "night", 0.041, 1.42, 0.0),
        ("herring", "night", 0.015, 0.65, 0.0), ("sprat", "day", 0.011, 0.14, 0.9)]),
    "whiting -> planktivores": (0.2, 0.566, [
        ("pout", "day", 0.523, 0.813, 0.4), ("herring", "day", 0.233, 0.455, 0.9),
        ("pout", "night", 0.158, 1.54, 0.0), ("blue whiting", "day", 0.071, 1.83, 0.2),
        ("sprat", "day", 0.009, 0.455, 0.9)]),
    "mackerel -> planktivores": (0.2, 0.617, [
        ("herring", "day", 0.489, 0.373, 0.9), ("herring", "night", 0.474, 0.99, 0.0),
        ("sprat", "day", 0.019, 0.373, 0.9), ("sprat", "night", 0.018, 0.99, 0.0)]),
    "benthivores -> planktivores": (0.3, 0.634, [
        ("pout", "day", 0.745, 0.391, 0.4), ("pout", "night", 0.105, 1.56, 0.0),
        ("herring", "day", 0.131, 0.275, 0.9), ("herring", "night", 0.019, 1.1, 0.0)]),
}


def F(sigma, S):
    return 1.0 - sigma * (1.0 - min(S, 1.0))


print("Part 1: activity-schooling floor (sigma_rest = 1 on the day terms named), combined p")
print(f"{'pair':30s} {'F avg':>6s} {'floor A (clupeids)':>19s} {'floor B (+pout)':>16s}")
for pair, (pl, pn, terms) in PAIRS.items():
    num_a = num_b = den = 0.0
    for prey, per, s, S0, sig in terms:
        S = S0 * pl / pn
        f_avg = F(sig, S)
        f_a = F(1.0, S) if (per == "day" and prey in ("herring", "sprat")) else f_avg
        f_b = F(1.0, S) if (per == "day" and prey in ("herring", "sprat", "pout")) else f_avg
        den += s * f_avg
        num_a += s * f_a
        num_b += s * f_b
    print(f"{pair:30s} {den:6.3f} {num_a / den:19.3f} {num_b / den:16.3f}")

print("\nPart 2: visible share v = 1 - r (1 - floor)")
floors = [0.25, 0.35, 0.5, 0.75, 0.9, 0.95, 1.0]
print(f"{'v at 0.25':>9s} {'rest r':>7s} " + " ".join(f"{f:>6.2f}" for f in floors))
for v in (0.28, 0.40, 0.50):
    r = (1 - v) / 0.75
    print(f"{v:9.2f} {r:7.3f} " + " ".join(f"{1 - r * (1 - f):6.3f}" for f in floors))
print("\nratio of v (floor 1 / floor 0.25) = per-capita fish-predation factor on the planktivores in a "
      "cell, below saturation:")
for v in (0.28, 0.40, 0.50):
    print(f"   v {v:.2f} -> x{1 / v:.2f}")
print("\nannual-mean rest share 0.043 (fix97, research_notes/\"Gom och sinnen\" section 4):")
for f in floors:
    print(f"   floor {f:.2f}: v {1 - 0.043 * (1 - f):.3f}")
