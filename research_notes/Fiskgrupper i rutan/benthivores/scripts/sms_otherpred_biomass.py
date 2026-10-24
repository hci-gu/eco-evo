"""Biomass of SMS 'other predators' (N * west, mean over quarters). Usage: python3 -I sms_otherpred_biomass.py other_pred_N.in west.in CODE y0 y1"""
import sys, re
def blocks(fn, code):
    out, cur = {}, None
    for line in open(fn):
        m = re.match(r"#\s*(\w+) year:\s*(\d+)", line)
        if m:
            cur = (m.group(1), int(m.group(2))); out[cur] = []; continue
        if line.startswith("#") or cur is None or cur[0] != code:
            continue
        v = [float(x) for x in line.split()]
        if len(v) == 11:
            out[cur].append(v)
    return {k[1]: v for k, v in out.items() if k[0] == code}
N = blocks(sys.argv[1], sys.argv[3]); W = blocks(sys.argv[2], sys.argv[3])
y0, y1 = int(sys.argv[4]), int(sys.argv[5])
tot = []
for y in range(y0, y1 + 1):
    b = 0.0
    for q in range(4):
        b += sum(n * w for n, w in zip(N[y][q], W[y][q]) if n > 0 and w > 0) / 4
    tot.append(b); print(sys.argv[3], y, f"{b/1000:.1f} kt (N thousands x kg)")
print("mean", f"{sum(tot)/len(tot)/1000:.1f} kt")
