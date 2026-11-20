# Krill availability by season (2026-11-20)

Main-agent fork (user: "Gör steg 1, krillens tillgänglighet per säsong"; saved by the main agent). Nothing in the
library, code, tests, manifest, AGENTS.md or the resume was changed, and nothing was committed. The frozen fix105
world only sized the field terms, through process-local patches in `scripts/seasonal_krill_test.py`.
- **Trail:** `evidence.md` (S1-S4).
- **Tables:** `tables/monthly_multipliers_stdout.txt` (policy-free), `tables/fill_decomp_d_*`,
  `tables/fill_summary_stdout.txt`.
- **`data/`:** not to be committed.

## Question

Field krill intake is high in Q1-Q2 and lowest in Q3 on every quarterly series. The chain's krill encounter peaks in
Q3-Q4. Is the krill available to the box's fish seasonal in a way the model lacks? If so, can it be sized well enough
to enter as a physical term? The term must be forcing data of the krill's own distribution, never a calendar factor
on a or a behaviour switch.

## Field per quarter

| term | Q1 | Q2 | Q3 | Q4 | tags, sources |
|---|---|---|---|---|---|
| stock level (library season) | 1.0 | 0.6 | 1.1 | 1.3 | J on F: Bohle & Moksness 1991 winter IKMT; NSEC spring trawls; Buchholz & Prado-Fiedler 1987 growth; spawning Apr-Oct, peak July (Boysen & Buchholz 1984 [A]). Q2 stomach pulses (whiting Apr 1991, saithe 1991) and Bergstad 2003 (deep fish eat krill in March, little in September) argue against the Q2 trough; no monthly series |
| presence over < 100 m beds, of >= 250 m | 0.43 | 0.43 (Apr), 0.05 (May-Jun) | 0.05 (Jul-Aug), 0.43 (Sep) | 0.43 | D on F / J: winter nights Bohle & Moksness; 0 of 14 near-bed day tows Jun-Jul (HERAS); library annual 0.30 |
| day layer | 70-150 m; "deeper in winter" | Trench below 150 m in May | 131-154 m over 157-166 m beds (Jul); 90-150 m at 250 m (Aug) | - | F/S; Oslofjord 40-100 m by season and weather [A]; unsized for the box |
| energy, kJ/g WW | ~6 | 4.3 | 4.3 | ~6 | D on F (Kattegat DW x DW/WW; Barents Sea 4.5 / 6.6) |
| size (year-class I DW, mg) | 50-53 (Feb), 19-26 (Mar) | 27-28 (May) | 30-34 (Jul), 53-60 (Sep) | 62-64 (Nov) | F |

## Model mapping

| term | model | status |
|---|---|---|
| stock season | `outside_season` | carried; shape [J], Q2 phase disputed by stomachs |
| presence over < 100 m beds | `outside_depth_table` | annual mean only; the half-year season is **missing** (summer x6 too high, Sep-Apr x0.7 too low) |
| day-layer depth | band overlap, `depth_access_table` | missing; no box data to size it |
| energy | `energy_content` 5000 | **missing** (one value) |
| ingestion Q10 | the predator's | carried; a fish term that lifts the chain's summer encounter |

## Policy-free multiplier on the krill encounter (quarter means, library annual mean = 1)

| predator (share of krill encounter on < 100 m beds) | library | + shallow presence | + energy | + flat stock |
|---|---|---|---|---|
| whiting, GOV placement (0.88) | 1.00 / 0.60 / 1.10 / 1.30 | 1.32 / 0.38 / 0.69 / 1.71 | 1.53 / 0.32 / 0.58 / 1.99 | 1.53 / 0.53 / 0.53 / 1.53 |
| planktivores, equal cells (0.48) | same | 1.16 / 0.49 / 0.90 / 1.50 | 1.35 / 0.41 / 0.75 / 1.75 | 1.35 / 0.68 / 0.68 / 1.35 |
| water column, GOV (0.17) | same | 1.03 / 0.58 / 1.06 / 1.34 | 1.20 / 0.48 / 0.89 / 1.56 | 1.20 / 0.81 / 0.81 / 1.20 |
| mackerel, equal cells (0.14) | same | 1.02 / 0.59 / 1.08 / 1.32 | 1.19 / 0.49 / 0.90 / 1.54 | 1.19 / 0.82 / 0.82 / 1.19 |

Whiting month by month with shallow presence: May-June 0.17 and July-August 0.32 of the annual mean, against 1.3-1.7
from September to March. The shallow-bed season alone moves whiting's krill into the field phase (low in summer).
The flat stock season removes the library's Q4 peak.

## Frozen sizing (copy d + field terms; fix105, seeds 0/1, one year)

| arm | whiting fill: year, minimum | whiting realised | planktivores fill | water column fill |
|---|---|---|---|---|
| copy d | 0.94, 0.90 (Oct) | 0.381 | 0.88 | 0.96 |
| + shallow presence | 0.90, 0.79 (Sep) | 0.369 (Aug 0.25) | 0.88 | 0.96 |
| + energy | 0.90, 0.79 (Sep) | 0.373 | 0.89 | 0.96 |
| + flat stock | 0.88, 0.69 (Sep) | 0.362 | 0.87 | 0.94 |
| field | min 0.30 (May-Jul), 0.69-0.73 Sep-Feb | - | min 0.20 (Apr) | - |

- **Whiting** runs a real summer deficit: realised 0.25 in August, below need. Its fill falls July-September and
  refills by December.
  - The minimum is in September, about 2 months after the field minimum, and the dip is about 1/3 of the field's depth.
  - May-June shows no deficit, although the multiplier is lowest then. The frozen whiting meet krill and epifauna
    elsewhere in spring. That is the sated-placement caveat of 294.6, so retraining is the judge.
- **Planktivores and the water column** hardly move. Their krill encounter sits mainly over deeper beds.
- **Diet gate and booked M2.** Each term keeps its annual mean, so `budget_gate` is unchanged by construction. The
  realised krill M2 under the arms was not computed.

## Verdict

1. **The krill available to whiting is seasonal, and the model lacks it.**
   - The main term is the krill's absence over beds < 100 m from May to August.
   - Field whiting stand on those beds: 0.77 of the GOV catch in the box, which carries 0.88 of their krill
     encounter.
   - The library holds only the annual mean of that presence.
   - With it, whiting's krill encounter is 0.17-0.32 of its annual mean from May to August and 1.3-1.7 from September
     to March: the field phase.
   - The frozen world shows a summer deficit, but late (September) and shallow (fill 0.79 against the field's 0.30).
2. **The krill energy season (6 / 4.3 kJ/g) is missing too.** It deepens the summer contrast by ~x0.85 and the winter
   peak by ~x1.17. Frozen, it hardly moves the fill.
3. **The library stock season's Q2 trough is [J] and disputed by Q2 stomach pulses.** Flattening it deepens whiting's
   September minimum to 0.69, but weakens the water column's winter. Leave it until monthly data exist.
4. **Proposed library change** (engine work: neither term is expressible in today's keys):
   - **A seasonal presence for the open pool.** For example a `outside_depth_season` table of monthly multipliers per
     bed class, here < 100 m: Sep-Apr 0.43/0.303, May-Aug 0.05/0.303, renormalised to the box total each month.
     Source: resume 262 and "Krill over grunda bottnar".
   - **A monthly `energy_content` for krill:** Q4-Q1 6, Q2-Q3 4.3, normalised to 5000.
   - Both are forcing of the prey's distribution and quality, like `outside_season`, not a behaviour switch. Both need
     the GPU engine too, and checkpoints become invalid (payoffs change).
5. **Unmeasured:**
   - the summer absence rests on 0 of 14 tows, only 2 inside the box, not aimed at krill;
   - the day-layer depth by season in the box;
   - the monthly stock (Boysen & Buchholz 1984 closed);
   - whether May-June whiting switch beds or prey.

**Settling datum:** daytime 120/200 kHz transects across the 60-120 m isobaths in the box in December-February and
June, with whiting stomachs by bed class in Q2-Q3.

Sources (web): Boysen & Buchholz 1984 (https://epic.awi.de/id/eprint/5229/); Oslofjord krill echosounder thesis 2021
(https://www.duo.uio.no/handle/10852/102828); Onsrud & Kaartvedt 1998, MEPS 171:209
(https://int-res.com/abstracts/meps/v171/p209-219).
