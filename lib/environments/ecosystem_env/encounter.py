"""Encounter observation (section 275; research_notes/Observabilitetsparen/
design.md section 5): what a group knows about another group in its own cell
and in the four neighbours is what it MET during the tick.

Each observation pair carries components [share, kappa] per region (own cell,
neighbour). kappa is the expected number of encounters per tick per t/km2 of
the target's visible density (1 km cells: per t per cell); ``inf`` marks a
diffuse target (plankton or krill layer, bed fauna, the own group), met
wherever it is present. With n the target's density in the region:

    P(met)     = sum_k share_k (1 - exp(-kappa_k n))
    P(entered) = sum_k share_k            (1 in the own cell)

One uniform u per (tick, observer, target, region, cell) picks the outcome:
u < P(met) shows n; P(met) <= u < P(entered) shows 0 (it was there and met
nothing); u >= P(entered) shows the observer's own-cell reading of that
channel after its own draw (it was never there; neighbours only). The
juvenile channel of a staged target follows the target's draw; the own
biomass, energy and own juveniles in the own cell are never drawn.

The uniforms come from a counter-based hash identical to ``lib/gpu/random``:
key = fold_in(fold_in(world_key, tick), STREAM), counter =
((d G + j) 5 + r) C + cell (d the observer's DM index, j the target's global
index, r 0 for the own cell and 1 + the direction's position for a
neighbour, cell = row W + column), so both engines draw the same numbers.
"""
import numpy as np

STREAM = 7001
REGIONS = 5
_MASK = np.uint64(0xFFFFFFFF)


def hash32(x):
    """Bit-identical numpy twin of ``lib.gpu.random.hash32``."""
    x = np.asarray(x).astype(np.uint64) & _MASK
    x = ((x ^ (x >> np.uint64(16))) * np.uint64(0x7FEB352D)) & _MASK
    x = ((x ^ (x >> np.uint64(15))) * np.uint64(0x846CA68B)) & _MASK
    return (x ^ (x >> np.uint64(16))) & _MASK


def fold_in(keys, value):
    """Twin of ``lib.gpu.random.fold_in``."""
    value = np.asarray(value).astype(np.uint64) + np.uint64(0x9E3779B9)
    return hash32(np.asarray(keys).astype(np.uint64) ^ hash32(value))


def tick_key(world_key, tick):
    return fold_in(fold_in(world_key, tick), STREAM)


def uniform_at(key, counters):
    """Uniforms in (0, 1) at integer ``counters`` under one ``key`` (float32,
    the 23-bit construction of ``lib.gpu.random.uniform``)."""
    bits = hash32(np.asarray(key, dtype=np.uint64) ^ hash32(
        np.asarray(counters).astype(np.uint64) + np.uint64(1)))
    return ((bits >> np.uint64(9)).astype(np.float32) + np.float32(0.5)) * np.float32(
        1.0 / 8388608.0)


def hash32_u32(x):
    """``hash32`` on uint32 arrays: numpy's uint32 multiplies wrap modulo
    2^32, so the bits equal ``hash32`` (half the memory of uint64)."""
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(0x7FEB352D)
    x = x ^ (x >> np.uint32(15))
    x = x * np.uint32(0x846CA68B)
    return x ^ (x >> np.uint32(16))


def counter_base(d, j, region, n_groups, n_cells):
    return ((int(d) * int(n_groups) + int(j)) * REGIONS + int(region)) * int(n_cells)


def entered_share(shares):
    """P(entered) = sum of the shares, in float32 (both engines alike)."""
    return np.float32(sum(float(np.float32(s)) for s in shares))


def diffuse_share(shares, kappas):
    """The summed share of the diffuse (kappa inf) components, float32."""
    total = np.float32(0.0)
    for s, k in zip(shares, kappas):
        if np.isinf(k):
            total = np.float32(total + np.float32(s))
    return total


def met_probability(density, shares, kappas):
    """P(met) in float32: the diffuse share where the target is present,
    then each finite component's s (1 - exp(-kappa n)) in library order,
    capped at P(entered) - the operation order of the tensor engine."""
    density = np.asarray(density, dtype=np.float32)
    p = diffuse_share(shares, kappas) * (density > 0).astype(np.float32)
    for s, k in zip(shares, kappas):
        if not np.isinf(k):
            p = p + np.float32(s) * -np.expm1(-np.float32(k) * density)
    return np.minimum(p, entered_share(shares))


def outcome(u, density, shares, kappas):
    """(met, entered) boolean fields for one region of one pair."""
    return u < met_probability(density, shares, kappas), u < entered_share(shares)


def build_plan(enc, d, n_groups, shape):
    """Static draw plan of one observer (both engines use the same rules).

    Returns {"kind": {(s, r): ("draw", n) | "away"}, "draws": [(s, r)] in
    order, "groups": [(start, end, shares [n, K], kappas [n, K])],
    "diffuse" [n], "entered" [n], "cell_hash" [n, H, W] uint32}. A diffuse
    pair in the own cell is shown as read (no draw: met wherever present);
    a region never entered (P(entered) 0) reads the own cell (no draw). The
    draws are ordered by their number of finite components."""
    H, W = shape
    C = H * W
    items, kind = [], {}
    for s, comp in enumerate(enc["components"]):
        if comp is None:
            continue
        for r in range(REGIONS):
            if r == 0 and s == 0:
                continue        # the own biomass in the own cell is known
            shares, kappas = comp["own" if r == 0 else "neighbour"]
            finite = [(x, k) for x, k in zip(shares, kappas) if not np.isinf(k)]
            if r == 0 and not finite:
                continue
            ent = entered_share(shares)
            if ent == 0.0:
                kind[(s, r)] = "away"
                continue
            items.append({"key": (s, r), "finite": finite, "entered": ent,
                          "diffuse": diffuse_share(shares, kappas),
                          "base": counter_base(d, enc["targets"][s], r, n_groups, C)})
    items.sort(key=lambda it: len(it["finite"]))
    for n, it in enumerate(items):
        kind[it["key"]] = ("draw", n)
    groups, start = [], 0
    while start < len(items):
        width, end = len(items[start]["finite"]), start
        while end < len(items) and len(items[end]["finite"]) == width:
            end += 1
        if width:
            groups.append((start, end,
                           np.array([[x for x, _ in it["finite"]] for it in items[start:end]], np.float32),
                           np.array([[k for _, k in it["finite"]] for it in items[start:end]], np.float32)))
        start = end
    cells = np.arange(C, dtype=np.int64)
    cell_hash = (hash32(np.asarray([it["base"] for it in items], dtype=np.int64)[:, None]
                        + cells[None] + 1).astype(np.uint32).reshape(len(items), H, W)
                 if items else np.zeros((0, H, W), np.uint32))
    return {"kind": kind, "draws": [it["key"] for it in items], "groups": groups,
            "diffuse": np.asarray([it["diffuse"] for it in items], np.float32),
            "entered": np.asarray([it["entered"] for it in items], np.float32),
            "cell_hash": cell_hash}


def plan_outcomes(plan, key, densities):
    """(met, entered) [n, H, W] for the plan's draws at ``densities``."""
    if not plan["draws"]:
        return None, None
    bits = hash32_u32(np.uint32(int(key) & 0xFFFFFFFF) ^ plan["cell_hash"])
    u = ((bits >> np.uint32(9)).astype(np.float32) + np.float32(0.5)) * np.float32(1.0 / 8388608.0)
    p = plan["diffuse"][:, None, None] * (densities > 0).astype(np.float32)
    for start, end, shares, kappas in plan["groups"]:
        n = densities[start:end]
        part = p[start:end]
        for k in range(shares.shape[1]):
            part = part + shares[:, k, None, None] * -np.expm1(-kappas[:, k, None, None] * n)
        p[start:end] = part
    entered = plan["entered"][:, None, None]
    p = np.minimum(p, entered)
    return u < p, u < entered


def parse_components(value, where):
    """Validate a library component list [[share, kappa], ...] -> two tuples."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{where}: expected a non-empty list of [share, kappa]")
    shares, kappas = [], []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{where}: each component is [share, kappa], got {item!r}")
        s, k = float(item[0]), float(item[1])
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"{where}: share must be in [0, 1], got {s}")
        if not k >= 0.0:      # also refuses NaN
            raise ValueError(f"{where}: kappa must be >= 0 (or .inf), got {k}")
        shares.append(s)
        kappas.append(k)
    if sum(shares) > 1.0 + 1e-6:
        raise ValueError(f"{where}: the shares sum to {sum(shares):.6f} > 1")
    return tuple(shares), tuple(kappas)


def rescale_kappas(kappas, tick_hours, library_hours):
    """kappa per tick scales with the path per tick (exact for the own cell's
    confined search; the neighbour shares keep their 6 h P(enter))."""
    f = float(tick_hours) / float(library_hours)
    return tuple(k if np.isinf(k) else k * f for k in kappas)
