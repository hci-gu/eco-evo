"""The time a move takes (section 276; research_notes/Observabilitetsparen
design.md 2.2).

A decision maker's move action shifts a share of a cell one cell (1 km) in
the tick. Before section 276 that share ate nothing for the whole tick:
a 1 km move cost a whole tick of food, the same for a porpoise (25 min at
0.65 m/s) as for a Norway pout (2 h). With ``relocation_speed`` U a move
takes the share tau = min(1, 1 km / (U x tick length)) of the tick and the
moving share forages for the rest, 1 - tau, in the cell it leaves (the
tick runs predation before movement), with the prey choice of the
policy's eat actions; it pays tau x movement_cost + (1 - tau) x
feeding_cost (x the resting metabolism), and its share of the tick's
intake travels with it.

``relocation_speed`` is a library species key, tick-independent:

- a number, U in m/s, or
- a list of ``[share, U]`` members (shares summing to 1), giving the
  member-weighted tau, sum share x min(1, 1 km / (U x tick)). A group of
  fast and slow members is NOT the tau of their mean speed: the slow
  members' long moves dominate the time.

Absent means tau = 1, the old rule, bit-identical.
"""
import math

SPEED_KEY = "relocation_speed"
TAU_KEY = "relocation_tau"      # derived by the loader, never in the library
CELL_METRES = 1000.0   # every grid cell is 1 km (a 30 x 30 world is a 30 km box)


def parse_speed(value, label="relocation_speed"):
    """[(share, U m/s)] from a library value, or None when absent.

    Raises ValueError on a malformed value: a speed must be > 0, member
    shares in (0, 1] summing to 1 (+-0.01)."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        members = [(1.0, float(value))]
    elif isinstance(value, (list, tuple)):
        members = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"{label}: each member is [share, speed m/s], got {item!r}")
            members.append((float(item[0]), float(item[1])))
        if not members:
            raise ValueError(f"{label}: an empty member list")
    else:
        raise ValueError(f"{label}: a speed in m/s or a list of [share, speed], got {value!r}")
    for share, speed in members:
        if not (math.isfinite(speed) and speed > 0.0):
            raise ValueError(f"{label}: a speed must be > 0 m/s, got {speed!r}")
        if not (0.0 < share <= 1.0):
            raise ValueError(f"{label}: a member share must be in (0, 1], got {share!r}")
    total = sum(share for share, _ in members)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"{label}: the member shares sum to {total:.4g}, not 1")
    return members


def tau(value, tick_hours, label="relocation_speed"):
    """The share of a tick a one-cell move takes; 1.0 without a speed."""
    members = parse_speed(value, label)
    if members is None:
        return 1.0
    seconds = float(tick_hours) * 3600.0
    total = sum(share for share, _ in members)
    return float(sum(share * min(1.0, CELL_METRES / (speed * seconds))
                     for share, speed in members) / total)
