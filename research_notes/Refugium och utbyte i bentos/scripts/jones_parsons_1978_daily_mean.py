"""24-h mean bottom availability of Pandalus borealis from Jones & Parsons
(ICNAF Res. Doc. 78/XI/87, Table 2; Davis Strait, August 1978, ~368 m).
Table 2 gives catch as % of the noon plateau. The parenthesised values at
0400 (40.4) and 1600 (100.0) are point values; the other values are printed
between hour labels and are read here as hourly INTERVAL values (key = start
hour of the interval, GMT; local solar time = GMT - 4 h). Typed in by hand.
Judgement (J): treats the % as the share of the population inside the trawl
opening (~11 m), i.e. the share reachable by bottom feeders, relative to noon."""
pct = {4: 40.8, 5: 42.6, 6: 45.5, 7: 48.8, 8: 52.4, 9: 56.8, 10: 62.1,
       11: 68.5, 12: 76.3, 13: 86.2, 14: 96.2, 15: 100.0,
       16: 100.0, 17: 100.0, 18: 100.0, 19: 100.0,
       20: 87.7, 21: 74.1, 22: 64.1, 23: 56.5, 0: 50.5, 1: 45.5, 2: 42.9,
       3: 40.4}
assert len(pct) == 24
mean = sum(pct.values()) / 24.0
print(f"24-h mean availability (fraction of noon level): {mean/100:.3f}")
print(f"24-h mean fraction off bottom relative to noon: {1-mean/100:.3f}")
for start in (0, 6, 12, 18):
    hrs = [(h + 4) % 24 for h in range(start, start + 6)]
    m = sum(pct[h] for h in hrs) / 6.0
    print(f"local {start:02d}-{start+6:02d} h: mean availability {m/100:.3f}")
