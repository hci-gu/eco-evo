# Primary production and C:chl for the open Skagerrak (yardsticks for A17)

Purpose: pin down the two yardsticks the model's phytoplankton is compared
with: (A) annual primary production (PP) in the open Skagerrak near SHARK
A17 (58.27 N 10.51 E, ~345 m), with its seasonal split and f-ratio; (B) the
phytoplankton C:chl that turns A17 chlorophyll into biomass. Literature
check and SHARK data pull, 2026-10-24.

Grades: **[F]** full text read; **[A]** abstract only; **[S]** search
snippet or secondary citation only; **[N]** not found / no access;
**[D]** derived by me from raw data. "(Own calculation)" = my arithmetic,
"(Own judgement)" = my reading. Builds on
`../Litteraturbaserad growth rate per FG/plankton.md` section 3,
`../Djupklorofyllmaximum/subsurface_chlorophyll_A17.md` and
`../Detritus och remineralisering/detritus_remineralisation.md`; their
coastal PP values (Kattegat 290, front 190, Gullmar 241, northern North Sea
175) are not repeated except as comparators.

---

## A. Annual primary production

### A1. Values found

| # | Area (distance from A17) | Value (g C m-2 yr-1) | Method, years | Grade |
|---|---|---|---|---|
| 1 | Skagerrak/Kattegat region of POLCOMS-ERSEM (contains A17) | **131** (ERA40-forced run, 40-yr mean), **138** (CNTRL run) | 3-D model, net PP (uptake minus phytoplankton respiration, 0-200 m). Holt et al. 2012, Biogeosciences 9:97-117, Tables 1-2, doi:10.5194/bg-9-97-2012 | [F] |
| 2 | Norwegian Trench region, same model | **102** (ERA40), **96** (CNTRL) | as above | [F] |
| 3 | Literature range quoted for Skagerrak/Kattegat by the same paper | 135-220 | Holt Table 1. The footnote letters look swapped: "f Joint et al. 2001" sits on Skagerrak/Kattegat and "g Rydberg et al. 2006" on the Celtic Sea (160). Rydberg 2006 is the Baltic-entrance paper, Joint 2001 the Celtic shelf-break one, so 135-220 is most probably Rydberg's (Own judgement) | [F] of Holt; attribution inferred |
| 4 | Skagerrak (general) | ~130 | "annual primary productivity of 130 g Corg m-2 yr-1", Frontiers 2023 (doi:10.3389/fmars.2023.1141448) citing Beckmann & Liebezeit 1988 (Mitt. Geol.-Palaont. Inst. Univ. Hamburg 65:99-116). The source is a May/June 1986 organic-carbon survey, not a PP time series | [S] (citing paper read, original not seen) |
| 5 | NORWECOM North Sea/Skagerrak eutrophication report | 160 | snippet: the model's oxygen budget "considers the oxygen consumption that results from an annual primary production of 160 g C m-2 yr-1". Region not verified; the IMR host (imr.brage.unit.no) did not resolve | [S] |
| 6 | N Kattegat / S Skagerrak frontal region, 57 25'-58 00' N | ~190 | 15 cruises 1984-1993, P-I curves x light, particulate 14C only. **7 of 12 months interpolated** (Nov-Feb from Steemann Nielsen's 1950s values). Few stations > 58 N: 192-199 mg C m-2 d-1 in Mar/Apr, 410 in Sep. Heilmann et al. 1994, MEPS 112:213-223, doi:10.3354/meps112213 | [F] |
| 7 | Baltic Sea Transition Zone (6 Danish stations, 14-51 m deep) | mean **190** (155-226), +-17 between years | 1385 P-I profiles 1998-2012 (Lyngsgaard et al. 2017, Estuar. Coasts 40, doi:10.1007/s12237-017-0215-4, Table 1) | [F] |
| 8 | Kattegat + Belt Sea, 6-8 Danish sites, 20-50 yr | 50-500 overall (open waters lowest, fjords highest); a jump from 1998 "most likely" a method change | Rydberg, Aertebjerg & Edler 2006, J. Sea Res. 56:1-16, doi:10.1016/j.seares.2006.03.009 | [S] |
| 9 | SMHI Anholt E (open Kattegat, 56.67 N 12.12 E), SHARK `Carbon production` (mg C m-2 d-1) 2006-2021 | 38-83 per year; 62 from the monthly climatology | [D] from SHARK. The parameter is reported for a 0-10 m hose sample; whether it is euphotic-depth integrated or only the 0-10 m layer is not documented (attenuation, incubation and calculation fields are empty). It is 3x below every published Kattegat value (rows 6-8) and is **not used** below | [D], integration unverified |
| 10 | OSPAR QSR 2023 pilot assessment of primary productivity (units SK "Skagerrak", NT "Norwegian Trench") | no absolute values; anomalies only (satellite OC-CCI PP 1997-2016: decreasing in most units) | OSPAR 2023, p00950 | [F] |

No 14C time series exists for the open Skagerrak in SHARK: the only
`Primary production` stations there are Baltic and Kattegat ones (Anholt E,
Fladen, BY-stations, Bothnian Sea); none at A17, Å17 or Släggö [D]. The
Gullmar series (Lindahl et al. 1998) is not in SHARK.

### A2. Daily rates measured in the open (central) Skagerrak

| Date | Where | PP (mg C m-2 d-1) | Notes | Grade |
|---|---|---|---|---|
| 20-28 Mar 1999 | Koster-Hirtshals transect | mean **2150** (all water masses); 0-20 m chl 229-239 mg m-2 in the bloom water masses | spring diatom bloom; copepods graze < 3 %, whole zooplankton 17 % of daily PP. Maar et al. 2002, MEPS 239:11-29, doi:10.3354/meps239011 | [F] |
| 8-14 Jun 1990 | central Skagerrak transect (Denmark-Norway) | apparent net PP ~**320**, about half in the subsurface maximum (15-30 m) | Bjørnsen et al. 1993, MEPS 95:279-294, doi:10.3354/meps095279; authors think SCM production is underestimated | [F] |
| May 1987 | 4 stations, open Skagerrak | not extracted | ~80 % of surface N uptake as NH4+urea, new production ~20 % | Rosenberg et al. 1990, MEPS 61:215, doi:10.3354/meps061215 | [A] |
| 25 Aug-3 Sep 2000 | central stations T2-T4 (400-560 m deep) | 471-1068, budget mean **740**; coastal K2/T1 1083-2071 | DCM held up to 95 % of column PP; Ceratium furca dominated. Maar 2003 PhD thesis (NERI), Paper V manuscript, Table 2 (pub.: Maar et al. 2004, Mar. Biol. 144:917) | [F] thesis |
| Jul 2016 | NE North Sea, 56.2-58 N 6.2-8.3 E (120-250 km W of A17) | **476 +- 110** (n = 128); column chl 29.9 +- 7.8 mg m-2 | Bendtsen & Richardson 2018, Biogeosciences 15:7315, doi:10.5194/bg-15-7315-2018 | [F] |
| Aug | north of Dogger Bank | 424 | DCM 58 % of column PP. Weston et al. 2005, J. Plankton Res. 27:909, doi:10.1093/plankt/fbi064 | [A] |

Illustrative annual integral from these rates (Own calculation, +-40 %):
Jan 50, Feb 150, Mar 1200 (a 3-4 week bloom at ~2000 inside a month),
Apr 500, May 600, Jun 450, Jul 450, Aug 600, Sep 450, Oct 400, Nov 150,
Dec 50 mg C m-2 d-1 gives **~155 g C m-2 yr-1**: Mar ~24 %, Apr-Sep ~60 %,
Oct-Feb ~16 %. Most of these daily rates are "potential" PP from
incubator P-I curves; they are the same kind of number as rows 6-7.

### A3. Is the open Skagerrak below, at or above the Kattegat/Gullmar values?

**Below** (Own judgement, from the evidence above):
- The two coastal/frontal values (190-290, Gullmar 241) sit where fronts,
  the Baltic outflow and coastal upwelling lift nutrients into the light
  (Heilmann 1994 [F]: biomass maxima at the most weakly stratified frontal
  stations; Maar thesis [F]: coastal K2/T1 2-3x the central stations).
- Over the trench the summer nutricline sits at 20-30 m (A17 table) and
  vertical mixing is weak: f < 0.02 over the Norwegian Trench in July
  (Bendtsen & Richardson 2018 [F]).
- The one model that resolves both gives Norwegian Trench 96-102 against
  Skagerrak/Kattegat 131-138 (Holt 2012 [F]); the model's Skagerrak errors
  are large (poor Baltic exchange), so treat it as a lower bound.
- The secondary 130 (row 4) and 160 (row 5) agree.

### A4. Seasonal split and depth split

- **Spring bloom share.** BSTZ Feb-Mar 14 % (10-18 %) of annual PP,
  Jul-Aug 30 % (Lyngsgaard 2017 [F]); southern Kattegat 1989: ~19 % in the
  spring bloom, ~30 % in subsurface (pycnocline) populations in summer
  (Richardson & Christoffersen 1991, via plankton.md [F]). In the Skagerrak
  the bloom is short (3-4 weeks) and very intense (2150 mg C m-2 d-1, Maar
  2002 [F]). Expected open-Skagerrak bloom share 15-25 % (Own judgement).
- **Subsurface share.** These numbers use different definitions and are
  not directly comparable with the model's "70 % below the 6-17 m mixed
  layer":
  - central Skagerrak, June 1990: ~50 % of column PP in the 15-30 m SCM
    (Bjørnsen 1993 [F]);
  - central Skagerrak, Aug-Sep 2000: 50-80 % of column chl and up to 95 %
    of PP below the warm surface layer (Richardson et al. 2003, J. Plankton
    Res. 25:799, doi:10.1093/plankt/25.7.799 [A]; Maar thesis [F]);
  - BSTZ: 6-30 % (mean 17 %) of ANNUAL PP in the pycnocline-bottom layer
    (Lyngsgaard, Markager & Richardson 2014, L&O 59:1679,
    doi:10.4319/lo.2014.59.5.1679 [A]);
  - central North Sea: DCM 60 % of PP in the summer stratified period
    (64 g C m-2) and 66 % of its new production (Fernand et al. 2013,
    Biogeochemistry 113:153, doi:10.1007/s10533-013-9831-7 [S]); model:
    up to 30 % of annual NPP below 15 m locally, 10 % North Sea-wide (van
    Leeuwen et al. 2013, Biogeochemistry 113:137 [S]).
  - Reading (Own judgement): a high subsurface share is right for May-Sep
    at A17 (50-95 %), but the March bloom is surface production, so an
    annual subsurface share of ~40-60 % (not 70 %) is what the evidence
    supports.
- **Autumn.** Heilmann 1994 [F] shows spring and autumn peaks (Oct 1990
  front 879 mg C m-2 d-1); Gullmar mouth autumn blooms of Ceratium and
  Gyrodinium (~800 in the second half of the season, sibling note
  `zoo_autumn_closure.md`). No open-Skagerrak autumn PP measurement was
  found [N].

### A5. f-ratio (new / total production)

| Situation | f | Source | Grade |
|---|---|---|---|
| NE North Sea, open area over the Norwegian Trench, July | < 0.02 | Bendtsen & Richardson 2018 | [F] |
| same, shelf edge | > 0.10 | same | [F] |
| central Skagerrak June 1990, surface layer | 0.04-0.11 | Bjørnsen 1993, Table 3 | [F] |
| same, SCM | 0.23 to > 1 (PP underestimated); new production 100-460 mg C m-2 d-1 by six methods | Bjørnsen 1993, Table 6 | [F] |
| open Skagerrak, May 1987 | ~0.20 (NO3 share of N uptake), coastal ~0.5 | Rosenberg et al. 1990 | [A] |
| Skagerrak late summer 2000 | 0.18-0.36; export ratio 23-39 % | Richardson et al. (in press 2003), quoted in the Maar thesis | [F] thesis, primary [N] |
| North Sea DCM, annual | DCM = 37 % of annual new production of the stratified North Sea | Weston et al. 2005 | [A] |

Reading (Own judgement): the spring bloom is mostly new production; the
summer is mostly regenerated (surface f < 0.1, SCM 0.2-0.4, trench < 0.02).
An **annual regional f of 0.2-0.4** is the target. (Not an f-ratio but
related: the Skagerrak takes up 1.2 mol C m-2 yr-1 = 14 g C of
atmospheric CO2, Hjalmarsson et al. 2010, J. Mar. Syst. 82:245, via
Binczewska et al. 2018, BG 15:5909 [S].) The model's f ~0.7
(section 203; 0.9 in 201.3) is too high: its PP is nearly all new
nitrogen, so it has too little recycling inside the layers.

---

## B. Phytoplankton C:chl

### B1. SHARK A17: microscopy carbon vs chl of the same hose sample [D]

The data: SHARK `Phytoplankton` at Å17/A17, 2001-2022, 0-10 m hose
samples (Utermöhl counts, biovolume to carbon by HELCOM/Menden-Deuer &
Lessard), paired with the `Chlorophyll` datatype's 0-10 m hose chl of the
same date. 222 paired visits, 16-22 per month. AU = autotrophs, MX =
mixotrophs (heterotrophs HT excluded). Downloaded 2026-10-24 with the
request in `../Djupklorofyllmaximum/` (`dataTypes` "Phytoplankton" /
"Chlorophyll", table views `sharkweb_all`).

| Month | n | chl ug/l | C AU+MX ug C/l | median C:chl (AU+MX) | IQR | ratio of means | dominant carbon |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 1.01 | 21 | 14 | 11-29 | 21 | diatoms (Mediophyceae) 36 %, dinoflagellates 21 % |
| 2 | 20 | 3.05 | 105 | 15 | 10-27 | 34 | diatoms 87 % |
| 3 | 18 | 3.49 | 109 | 28 | 15-35 | 31 | diatoms 73 % |
| 4 | 21 | 1.38 | 43 | 28 | 12-48 | 31 | dinoflagellates 29 % |
| 5 | 18 | 0.91 | 48 | 34 | 23-68 | 53 | unclassified flagellates 36 %, dinoflagellates 22 % |
| 6 | 19 | 0.79 | 53 | 52 | 28-76 | 67 | diatoms 37 %, dinoflagellates 33 % |
| 7 | 22 | 0.49 | 26 | 51 | 29-85 | 53 | dinoflagellates 44 % |
| 8 | 18 | 0.71 | 38 | 42 | 22-95 | 54 | dinoflagellates 33 % |
| 9 | 18 | 0.91 | 38 | 35 | 23-63 | 42 | dinoflagellates 41 % |
| 10 | 17 | 1.80 | 38 | 21 | 13-26 | 21 | dinoflagellates 30 % |
| 11 | 16 | 1.51 | 26 | 17 | 10-22 | 17 | dinoflagellates 38 % |
| 12 | 19 | 0.78 | 23 | 24 | 14-36 | 29 | dinoflagellates 34 % |

All-year median 28. A17 surface total N is 10.5-15.8 umol/l (annual mean
12.1, 0-10 m, 2001-2025) [D].

Biases (Own judgement):
- Microscopy carbon misses picoplankton (< 2-3 um) and undercounts small
  fragile flagellates. These carry a large share of summer surface carbon
  when chl is 0.4-0.8 ug/l, so the Jun-Sep values are a **lower bound**.
- Winter and the diatom bloom are counted well, so those values are close
  to the truth.
- Diatom carbon by biovolume has its own +-30-50 % uncertainty.

### B2. Literature

| Source | Area | C:chl (g/g) | Basis | Grade |
|---|---|---|---|---|
| Jakobsen & Markager 2016, L&O 61:1853, doi:10.1002/lno.10338 | Danish waters, 7578 samples 1990-2014 | winter ~15 at all stations; summer 20-96, the amplitude inversely related to annual mean TN; sinusoidal monthly model; +0.8 +- 0.2 yr-1 trend as N loads fell | microscopy carbon / chl | [A] (paper closed) |
| Lyngsgaard et al. 2017 (citing J&M) | BSTZ | "up to 7-fold from 10 in the winter and up to almost 60 during summer"; dinoflagellates higher than diatoms | same | [F] |
| Maar et al. 2002 | Skagerrak spring bloom | 30 assumed, "representative for a spring bloom in Danish waters" (Olesen & Lundsgaard 1995) | assumption | [F] |
| Bjørnsen et al. 1993 | central Skagerrak, June SCM | 30 assumed | assumption | [F] |
| Maar thesis (Paper V) | Skagerrak late summer 2000, Ceratium furca bloom | **108** (POC:chl regression, r2 0.41, n 12); sedimented matter 124 | suspended POC, includes detritus and heterotrophs | [F] |
| Bendtsen & Richardson 2018 | NE North Sea July | 50 assumed | assumption | [F] |
| Ståhl et al. 2004 (Skagerrak sediments) | - | 60 assumed (after Sun et al. 1991) | assumption | [S] |
| Skogen et al. eutrophication status report | North Sea-Baltic | in situ C from chl, 15-176 | - | [S] |
| Cloern, Grenz & Vidergar-Lucas 1995, L&O 40:1313, doi:10.4319/lo.1995.40.7.1313 | culture fit | Chl:C = 0.003 + 0.0154 exp(0.050 T) exp(-0.059 I) mu', I in mol photons m-2 d-1, mu' = N/(K_N + N) | 219 culture measurements | [F] of the equation in a 2024 review (Front. Mar. Sci. 11:1466072); original [N] (paywalled) |

### B3. Surface vs subsurface maximum: is the SCM a biomass or a photoacclimation maximum?

Evidence for a **real biomass maximum** in the Skagerrak:
- Bjørnsen 1993 [F]: microscopy found subsurface maxima of autotrophic
  flagellates and diatoms in the nutricline, growing in intensity over 6
  days. The SCM was supported partly by new production, and its net PP was
  split about evenly between biomass build-up, remineralisation and
  sedimentation.
- Maar thesis [F]: chl > 45 um correlated with Ceratium furca abundance
  (r2 0.92); the strongest DCM held 20 ug chl/l at 23-27 m.
- Richardson 2003 [A]: oxygen saturation maxima just above the peaks show
  in situ net photosynthesis.
- Lyngsgaard et al. 2014, MEPS 517:15, doi:10.3354/meps11015 [F, sibling
  note `../Djupklorofyllmaximum/`; abstract re-read]: the pycnocline population is a
  distinct, large-celled Ceratium community acclimated/adapted to low
  light.

Evidence that chl **overstates** SCM biomass:
- Chl-normalised Pmax: 1.72 at the SCM vs 4.76 ug C (ug chl h)-1 at the
  surface (NE North Sea, Bendtsen & Richardson 2018 [F]).
- Celtic Sea: cell maxima decoupled from pigment maxima (Hickman et al.
  2009, L&O 54:525 [A]).
- Cullen 2015, Annu. Rev. Mar. Sci. 7:207 [A]: photoacclimation raises chl
  relative to biomass at depth, and the SCM is nearly always partly this.
- Cloern model at A17 conditions (Own calculation, next section): SCM
  C:chl is 0.2-0.5 of the mixed-layer value from May to August.

Reading (Own judgement): the A17 SCM (Apr-Sep, 15-30 m) is a **real
biomass maximum**, but chl overstates it by a factor of ~2 relative to the
surface. For `subsurface_phytoplankton` this means:
- comparing its biomass with A17 subsurface chl x a surface C:chl
  overstates the target ~2x;
- its biomass target should use C:chl ~25-30;
- the shade-adapted producer is right in kind, but needs less carbon than
  the chlorophyll suggests.

### B4. Cloern ratio at A17 (Own calculation)

Inputs:
- light: daily PAR from `lib/world/daylight.py` (clear-sky Haurwitz PAR at
  58.15 N, manifest cloud transmission), Kd 0.29 /m in the mixed layer and
  0.17 /m below it, manifest mixed-layer depths;
- temperature and nitrate from the A17 tables;
- SCM depths 10/15/15/20/30/20/15 m for Mar-Sep;
- K_N 0.5-1.0 umol/l.

Results:
- Mixed layer: 46-64 in Nov-Feb, 180-270 in May-Aug.
- SCM: 44-78 in Apr-Aug.
- Ratio SCM / mixed layer: 0.70 in Mar, 0.21-0.29 in Apr-Jul, 0.43-0.49 in
  Aug, 0.69-0.75 in Sep.

The absolute values are not usable. In winter they are 3x the A17
microscopy values. In summer, with N = 0.1 umol/l, the Monod term pushes
Chl:C to its 0.003 floor. The model is a culture fit. Only the ratio is
used, as qualitative support for a factor 2-4 lower C:chl at the SCM.

---

## Recommendation

### R1. Open-Skagerrak PP target
- **Central 150 g C m-2 yr-1, range 120-200.**
- Below the frontal/coastal 190-290.
- Above the Norwegian Trench model's 96-102, which errs low in this region.
- Consistent with Holt 131-138, the secondary 130 and 160, and the ~155
  integral of the measured daily rates.
- The model's 57-61 is ~2-2.5x under even the low end (Own calculation:
  120/60 = 2.0). Definitions (particulate vs total 14C, potential incubator
  vs in situ PP, gross vs net) account for 20-50 % at most. No source puts
  the open Skagerrak below ~100.
- Seasonal shape: spring bloom 15-25 %, May-Sep 55-65 %, Oct-Feb 15-20 %.
- Subsurface share 50-95 % in May-Sep but ~40-60 % annually.
- Annual f-ratio 0.2-0.4, summer surface < 0.1.

### R2. Monthly C:chl (g C / g chl a)

(a) **Surface / mixed layer** (central, low-high).
- Central: the SHARK A17 ratio of means for Oct-Apr, because a monthly
  mean biomass needs mean(C)/mean(chl), not the median of the visit
  ratios (bloom visits carry both high chl and high C:chl).
- Central for May-Sep: raised above the measured value, for picoplankton
  and flagellates and for A17's low TN, which puts it at the
  high-amplitude end of Jakobsen & Markager.
- Low ~ the SHARK median.

| | J | F | M | A | M | J | J | A | S | O | N | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| central | 21 | 34 | 31 | 35 | 55 | 70 | 70 | 60 | 50 | 25 | 20 | 29 |
| low | 14 | 15 | 25 | 28 | 35 | 50 | 50 | 45 | 35 | 20 | 15 | 20 |
| high | 30 | 45 | 40 | 50 | 75 | 90 | 95 | 90 | 70 | 35 | 30 | 35 |
| SHARK ratio of means [D] | 21 | 34 | 31 | 31 | 53 | 67 | 53 | 54 | 42 | 21 | 17 | 29 |

(b) **0-30 m column**, chl-weighted. The column value is
share_ML x surface value + share_sub x subsurface value, where:
- the shares come from the A17 profile split at the manifest mixed-layer
  depth;
- the subsurface value (below the mixed layer) is 28 in Mar, 25 in Apr-May
  and 30 in Jun-Sep (ranges 15-20 to 40-45);
- Oct-Feb have no stratified SCM, so the whole column takes the surface
  value.

(Own calculation)

| | J | F | M | A | M | J | J | A | S | O | N | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chl share below M | 0.34 | 0.46 | 0.75 | 0.84 | 0.84 | 0.89 | 0.90 | 0.82 | 0.68 | 0.54 | 0.28 | 0.35 |
| column C:chl central | 21 | 34 | 29 | 27 | 30 | 34 | 34 | 35 | 36 | 25 | 20 | 29 |
| low | 14 | 15 | 21 | 17 | 18 | 23 | 23 | 24 | 25 | 20 | 15 | 20 |
| high | 30 | 45 | 40 | 42 | 45 | 50 | 50 | 53 | 53 | 35 | 30 | 35 |

### R3. A17 targets recomputed (Own calculation)

0-30 m column biomass in t WW/km2 (= g WW m-2):
mg chl m-2 x C:chl / 1000 / 0.1 (C:WW 0.1). Check: March 73 x 50 = 36.5;
73 x 29 = 21.0.

| Month | col chl mg m-2 | at C:chl 50 (old) | low | **central** | high | central / old |
|---|---|---|---|---|---|---|
| 1 | 23 | 11.5 | 3.2 | **4.8** | 6.9 | 0.42 |
| 2 | 66 | 33.0 | 9.9 | **22.4** | 29.7 | 0.68 |
| 3 | 73 | 36.5 | 15.5 | **21.0** | 29.2 | 0.58 |
| 4 | 52 | 26.0 | 8.9 | **13.8** | 21.6 | 0.53 |
| 5 | 34 | 17.0 | 6.2 | **10.1** | 15.5 | 0.59 |
| 6 | 39 | 19.5 | 9.0 | **13.4** | 19.4 | 0.69 |
| 7 | 26 | 13.0 | 6.0 | **8.8** | 13.0 | 0.68 |
| 8 | 25 | 12.5 | 6.1 | **8.8** | 13.3 | 0.70 |
| 9 | 25 | 12.5 | 6.2 | **9.1** | 13.3 | 0.73 |
| 10 | 40 | 20.0 | 8.0 | **10.0** | 14.0 | 0.50 |
| 11 | 34 | 17.0 | 5.1 | **6.8** | 10.2 | 0.40 |
| 12 | 17 | 8.5 | 3.4 | **4.9** | 5.9 | 0.58 |
| mean | | 18.9 | 7.3 | **11.2** | 16.0 | 0.59 |

Surface (mixed-layer) targets:

| | J | F | M | A | M | J | J | A | S | O | N | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| measured 0-10 m phytoplankton C (AU+MX), ug C/l [D] | 21 | 105 | 109 | 43 | 48 | 53 | 26 | 38 | 38 | 38 | 26 | 23 |
| A17 surface chl x surface C:chl central, ug C/l (Own calculation) | 25 | 132 | 102 | 52 | 52 | 56 | 32 | 39 | 40 | 45 | 31 | 22 |
| range | 17-36 | 58-175 | 82-131 | 41-74 | 33-70 | 40-72 | 23-44 | 29-58 | 28-55 | 36-63 | 23-46 | 15-27 |
| mixed-layer stock at the manifest depth, t WW/km2 (central) | 3.3 | 11.9 | 6.1 | 3.1 | 3.1 | 3.4 | 1.9 | 2.7 | 4.0 | 4.5 | 5.2 | 3.3 |

- The measured carbon row is a direct measurement (microscopy, a lower
  bound in summer) on the 0-10 m hose. Where the conversion falls below
  it, the measured value is the floor.
- The conversion row uses the 0-5 m surface chl of the current
  comparison. In Feb it is higher than the hose carbon because the 0-5 m
  chl (3.88) exceeds the 0-10 m hose chl (3.05).

What moves (Own calculation; the month-by-month C:chl behind section
203's "30 winter, 100 summer" is not documented, so the second and third
points are conditional):
- **Winter and the bloom.** The softer change of the two:
  - Nov and Jan fall to 0.40-0.42 of the C:chl 50 targets, Oct and Dec
    to 0.50-0.58;
  - Feb-Mar stay near 0.6-0.7 (the diatom bloom has the higher
    ratio-of-means C:chl of 31-34);
  - April falls to 0.53.
  - The winter C:chl (14-24 by visit, 21-29 by month as a ratio of means)
    is measured, not guessed.
- **Summer.** If the old guess used 100 in Jun-Sep, the new column targets
  are 0.34-0.36 of it. The model's Jun-Sep column ratios of 0.28-0.38
  would become ~0.8-1.1, so the summer column gap largely disappears.
- **Annual mean.** The target falls from 18.9 to ~11 t/km2 (range 7-16).
  With col/obs 0.59 at C:chl 50 (section 203.1) the model would sit near
  1.0 (rough, ratio of means).
- **What stays wrong.** The surface: Jul-Aug A17 26-39 ug C/l (measured
  and converted) vs the model's 0.09-0.13 ug chl/l (~5-7 ug C/l), i.e.
  still 4-7x low. Also PP (2-2.5x) and the f-ratio (0.7 vs 0.2-0.4).

### R4. Main uncertainties
1. **No open-Skagerrak 14C annual series exists.** The PP target rests on
   one 3-D model (with its worst skill in exactly this region), secondary
   citations and a handful of cruise days. +-30 % is honest; a factor 1.5
   is possible.
2. **What "PP" means.** Particulate vs total 14C, potential (incubator P-I)
   vs in situ, gross vs net. The model's PP is a gross-of-grazing growth
   term. Differences of 20-50 % come from definitions alone.
3. **Summer surface C:chl** depends on picoplankton carbon that
   microscopy does not see. The 50 -> 70 raise is a judgement. Use the
   SHARK median (~50) as the floor and 90 as the ceiling.
4. **SCM C:chl** has no direct measurement at A17 or in the Skagerrak.
   25-30 rests on two assumptions in the literature (30), the Pmax^B ratio
   (0.36) and the Cloern ratio (0.2-0.5). Range 15-45. The column target
   moves almost linearly with it in May-Sep.
5. **C:WW 0.1** is untouched here (Mackinson & Daskalov 2007 1:10; their
   report calls the conversion a major uncertainty). Any change scales
   every biomass target equally.
6. **Interannual variability.** The monthly IQRs of C:chl at A17 span about
   2x, and the Gullmar PP spans 180-339 between years.

### Not found / no access [N]
- Jakobsen & Markager 2016: full text (fit parameters, TN relation).
- Rydberg et al. 2006: full text.
- Cloern 1995: original.
- Pettersson 1991 (JEMBE, Skagerrak seasonal C and N uptake): abstract.
- Karlson et al. 1996 (J. Sea Res., Skagerrak SCM): abstract.
- Hjalmarsson et al. 2010: abstract.
- The NORWECOM IMR reports (host unreachable).
- Fernand et al. 2013 and van Leeuwen et al. 2013: full texts.
