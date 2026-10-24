"""Shared definitions for the benthivorous_fish ICES stomach scripts (no I/O)."""
PRED = {"126436": "cod", "126437": "haddock", "127137": "long_rough_dab", "127139": "dab",
        "127143": "plaice", "127136": "witch", "127140": "lemon_sole", "150637": "grey_gurnard",
        "105865": "starry_ray", "101170": "hagfish", "127082": "greater_weever", "127141": "flounder"}

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")

def area_of(lat, lon):
    """Same areas as predators/scripts/predator_diet_pools.py (Kattegat, Skagerrak, NNS, CSNS)."""
    if lat != lat or lon != lon:
        return "na"
    if 55.5 <= lat < 57.75 and 10.3 <= lon <= 13:
        return "Kattegat"
    if lat >= 57.0 and lon <= 13:
        lon_line = 8.6 + (lat - 57.12) * (7.05 - 8.6) / (58.0 - 57.12)
        if lat <= 59.2 and lon >= lon_line:
            return "Skagerrak"
    if lat >= 57.5 and -4 <= lon <= 13:
        return "NNS"
    if 51 <= lat < 57.5 and -4 <= lon <= 9:
        return "CSNS"
    if lon > 13 or (lat < 57.5 and lon > 9):
        return "Baltic_or_Belts"
    return "other"

def box(lat, lon):
    return 57.7 <= lat <= 58.4 and 9.9 <= lon <= 11.3
