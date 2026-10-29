# Age-0 and age-1 share of the fish groups per quarter, Mareld box (2026-10-29)

Question (user): the data behind a juvenile stage (alternative B, resume 223.31): how large a share
of the planktivore, benthivore and whiting biomass are this year's and last year's fish, per quarter,
and when the age-0 fish appear.

Scripts: `scripts/datras_q2q4_fetch.py` (NS-IBTS HH/HL/CA from DATRAS, box hauls; CA for the
Skagerrak rectangles 43-47 F7-G1), `scripts/juvenile_share_by_quarter.py`. Output
`tables/juvenile_share_by_quarter_stdout.txt`, `tables/juvenile_share_by_species.csv`,
`tables/juvenile_share_by_group.csv`. Downloaded data in `data/` (untracked, untrusted).

## Data and method [D]

- **1991-96, all four quarters.** The quarterly NS-IBTS ran in Q2 and Q4 only in 1991-97 (1997 has no
  box hauls). Box GOV hauls: Q1 154, Q2 146, Q3 163, Q4 57. Unstratified (haul mean per year, then
  year mean). Ages from that period's own CA records per quarter; Q4 has few aged fish (489 rows), so
  for species with < 50 aged fish in Q4 the key is the Q1 key shifted one year (Q1 age 1 = the Q4
  age-0 cohort a few months later).
- **2001-25, Q1 (Jan-Feb) and Q3 (Aug).** Depth-stratified as the earlier box scripts; ages from the
  2021-25 CA records.
- Biomass from length-weight; catchability-at-length (Walker et al. 2017) for cod, haddock, whiting
  and Norway pout; herring, sprat and flatfish raw (no q published). Group weights: planktivores by
  the pool weights of params_planktivorous_fish.md (herring 0.78-0.88), benthivores by the model's
  member weights (haddock 38.6, cod 31.4, long rough dab 12.6, ...). Species without an age key count
  as 0 (long rough dab, rays, gurnard: a lower bound; their 0-group is barely in the GOV).

## Group shares of biomass [D]

Age 0 | age 1, q-corrected within species (raw in the csv):

| group | period | Q1 Jan-Feb | Q2 May | Q3 Aug | Q4 Nov |
|---|---|---|---|---|---|
| planktivores | 1991-96 | 0 / 0.77 | 0 / 0.51 | 0.53 / 0.22 | 0.76 / 0.11 |
| planktivores | 2001-25 | 0 / 0.41 | - | 0.18 / 0.35 | - |
| benthivores | 1991-96 | 0 / 0.35 | 0 / 0.34 | 0.40 / 0.21 | 0.35 / 0.18 |
| benthivores | 2001-25 | 0 / 0.12 | - | 0.10 / 0.16 | - |
| whiting | 1991-96 | 0 / 0.50 | 0 / 0.65 | 0.48 / 0.34 | 0.78 / 0.19 |
| whiting | 2001-25 | 0 / 0.12 | - | 0.12 / 0.39 | - |

Age 0 + age 1 together (the juvenile compartment if it holds the fish until their second summer):
planktivores 1991-96 0.77 / 0.51 / 0.75 / 0.87, 2001-25 0.41 / - / 0.52 / -; benthivores 0.35 /
0.34 / 0.61 / 0.53 and 0.12 / - / 0.26 / -; whiting 0.50 / 0.65 / 0.82 / 0.97 and 0.12 / - / 0.50 / -.

## Per species, age-0 share and mean length (cm) [D]

| species | 1991-96 Q3 | Q4 | 2001-25 Q3 | age-0 length Q3 -> Q4 |
|---|---|---|---|---|
| herring | 0.64 | 0.90 | 0.21 | 12.2 -> 13.9 |
| sprat | 0.02 | 0.20 | 0.01 | 10.1 -> 11.0 |
| Norway pout | 0.74 | 0.52 | 0.29 | 9.2 -> 9.9 |
| haddock | 0.90 | 0.73 | 0.23 | 12.5 -> 16.3 |
| cod | 0.11 | 0.16 | 0.02 | 11.0 -> 15.6 |
| whiting | 0.48 | 0.78 | 0.12 | 13.3 -> 16.4 |

## Reading [D/J]

- **Timing.** No age-0 fish in Q1 (by the age convention) or in Q2 (May: larvae and post-larvae, not
  in the GOV). They are in by August: haddock settle mid-Jul to mid-Aug (`../Sejens diet pa unga
  torskfiskar/evidence_settlement.md`), 0-group herring metamorphose and school from ~Jun-Jul. So the
  pulse enters between May and August and grows through the autumn (herring 12 -> 14 cm, haddock
  12.5 -> 16 cm Aug -> Nov).
- **Size.** Age-0 fish in August are 9-13 cm: inside a mackerel's ~10 cm gape at the small end, all
  inside saithe's 19 cm; by November haddock and whiting age 0 (16 cm) have left the mackerel window.
  Age 1 (15-25 cm) is inside the saithe gape only.
- **Weight of the stage.** The juvenile share is large: by biomass the age-0 cohort is 10-50 % of the
  August group and age 0 + 1 is 25-80 %, depending on period. That is no small correction: a
  juvenile compartment holds a big part of each group.
- **Period.** 1991-96 shares are 2-4x the 2001-25 ones. Part is real (strong 1990s herring and
  haddock year classes, a different stock state), part method (unstratified Q2/Q4 means, fewer
  hauls). Use 1991-96 for the seasonal SHAPE (Q2 and Q4 exist only then) and 2001-25 for the LEVEL.
- **Catchability.** The haddock q-correction raises the age-0 share most (2001-25 Q3 benthivores
  0.032 raw -> 0.097 corrected; Walker q 0.012 below 11 cm against 0.12 for adults, the small-length
  q is the least certain part). Herring has no q: the GOV under-catches pelagic adults, which inflates
  its juvenile share; HERAS (pelagic trawl, Jun-Jul) gives herring below 20 cm 0.44 of the biomass,
  close to the GOV age-1 share of 0.35-0.39 in August.
