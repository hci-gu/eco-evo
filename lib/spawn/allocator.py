"""Allokator: mappa vikter -> per-cell-biomass med respekt för "indivisible weight".

Kontrakt
========
``distribute_with_floor(weights, total_b, min_per_cell, allowed_mask=None) -> np.ndarray``

Tar en vikt-array (typiskt från ``lib.spawn.strategies.make_weights``, summa = 1
över eligible cells) och fördelar ``total_b`` ton över griden så att:

1. **Summan av output är exakt ``total_b``** (modulo flyttalsbrus).
2. **Varje aktiv cell har biomass >= ``min_per_cell``** (ingen aktiv cell
   under sub-tröskel-masken).
3. Celler utanför ``allowed_mask`` är alltid 0.
4. Om en cell skulle få < min_per_cell (men > 0) baserat på vikten, så
   släcks den ("zero-or-floor"-princip) — vi behåller bara så många aktiva
   celler som ``total_b`` räcker till med golvet hållet.
5. Output-storlek följer ``weights.shape``.

Algoritm (greedy-fill)
----------------------
1. Sortera celler efter fallande vikt.
2. Beräkna ``n_max = floor(total_b / min_per_cell)`` (max antal aktiva
   celler som golvet tillåter). Om ``min_per_cell <= 0`` → ``n_max = inf``.
3. Välj de top-N cellerna med högsta vikt där ``N = min(n_max, antal celler
   med weight > 0)``.
4. **Proportionell fördelning** över de N valda cellerna: var och en får
   ``total_b * (w_i / sum(w_top_N))`` ton.
5. Om någon vald cell hamnar under ``min_per_cell``, släck den (sätt till 0)
   och fördela om dess vikt över de övriga. Iterera tills alla aktiva
   celler är >= ``min_per_cell`` eller listan är tom.
6. Edge case: om ``total_b < min_per_cell`` och ``min_per_cell > 0``,
   lägg hela ``total_b`` i den högst viktade cellen (1 aktiv cell, under
   golv-tröskeln — biologiskt en "rest-population" som annars skulle dö).

Edge cases
----------
- ``min_per_cell == 0``: ren proportionell fördelning över alla celler med
  weight > 0 (motsvarar dagens beteende för phyto/zoo/benthic).
- ``total_b == 0``: returnerar nollarray.
- ``weights.sum() == 0``: returnerar nollarray (inget att fördela).
- ``allowed_mask`` ges: vikter utanför mask-en zeroas ut innan algoritmen
  körs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def distribute_with_floor(
    weights: np.ndarray,
    total_b: float,
    min_per_cell: float,
    allowed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fördela ``total_b`` ton över griden enligt ``weights`` med per-cell-golv.

    Parameters
    ----------
    weights : np.ndarray, shape (H, W)
        Icke-negativa vikter (behöver inte vara normaliserade).
    total_b : float
        Total biomass i ton att fördela.
    min_per_cell : float
        Minsta tillåtna biomassa per aktiv cell (``10 * min_split_biomass``
        i konventionen). 0 → ingen nedre gräns.
    allowed_mask : np.ndarray or None
        Boolean mask, samma shape som ``weights``. Celler där mask=False
        får alltid 0.

    Returns
    -------
    np.ndarray, shape (H, W), dtype float64
        Per-cell-biomass i ton. Summan == ``total_b`` (modulo flyttalsbrus)
        om ``total_b >= min_per_cell``; annars exakt ``total_b`` i en cell.
    """
    w = np.asarray(weights, dtype=np.float64)
    out = np.zeros_like(w)

    if total_b <= 0.0:
        return out

    # Applicera allowed_mask.
    if allowed_mask is not None:
        mask = np.asarray(allowed_mask, dtype=bool)
        w = np.where(mask, w, 0.0)

    # Behåll endast positiva vikter.
    w = np.where(w > 0.0, w, 0.0)
    total_w = float(w.sum())
    if total_w <= 0.0:
        return out

    flat_w = w.ravel()
    flat_out = out.ravel()

    # Sortera index efter fallande vikt.
    order = np.argsort(-flat_w, kind="stable")

    # Antal celler med strikt positiv vikt.
    n_pos = int(np.count_nonzero(flat_w))

    if min_per_cell <= 0.0:
        # Ren proportionell fördelning.
        flat_out[:] = flat_w * (total_b / total_w)
        return out

    # Edge case: total_b räcker inte ens till en cell vid golvet.
    if total_b < min_per_cell:
        # Lägg allt i högst viktade cellen ("rest-population"-beteendet).
        flat_out[order[0]] = total_b
        return out

    # Greedy-fill: börja med så många celler som golvet tillåter.
    n_max = int(np.floor(total_b / min_per_cell))
    n_active = min(n_max, n_pos)

    while n_active > 0:
        chosen = order[:n_active]
        w_chosen = flat_w[chosen]
        sum_chosen = float(w_chosen.sum())
        if sum_chosen <= 0.0:
            n_active -= 1
            continue

        alloc = w_chosen * (total_b / sum_chosen)

        # Om alla allokationer är >= golvet, klart.
        if float(alloc.min()) >= min_per_cell:
            flat_out[:] = 0.0
            flat_out[chosen] = alloc
            return out

        # Annars: minska antalet aktiva celler med 1 (släcker minsta vikt).
        n_active -= 1

    # Fallback: 1 aktiv cell med hela total_b (golv-violation accepteras
    # eftersom alternativet är att förlora biomassa).
    flat_out[order[0]] = total_b
    return out


def summarize(per_cell: np.ndarray, min_per_cell: float) -> Tuple[int, float, float, float]:
    """Diagnostik: (n_active, sum, min_active, max_active).

    ``n_active`` = antal celler med biomass > 0. ``min_active`` är minsta
    biomassa bland aktiva celler, eller 0 om inga aktiva. Nyttigt vid
    tester och i loggar.
    """
    flat = per_cell.ravel()
    active = flat[flat > 0.0]
    n_active = int(active.size)
    total = float(flat.sum())
    if n_active == 0:
        return 0, total, 0.0, 0.0
    return n_active, total, float(active.min()), float(active.max())
