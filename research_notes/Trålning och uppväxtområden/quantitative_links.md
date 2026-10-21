# Is there a quantitative link from trawl intensity to juvenile gadoid survival, recruitment or K? (Q2)

Labels:
- **[Primary]**: peer-reviewed literature.
- **[Report]**: manuals, ICES and grey literature.
- **[Own]**: my computation or inference.

Fetch status is FULL, ABS or SNIP, as in the companion notes.

## Takeaway

- **No published function maps trawl intensity (SAR or effort) to juvenile gadoid survival, recruitment or K.** This
  holds for fitted functions and for assumed ones. The chain has two halves, and they have never been joined for
  gadoids.
- **Half 1, SAR to benthic state: well quantified.**
  - Relative benthic status RBS = B/K = 1 - SAR·d/r (Pitcher et al. 2017, 2022; Hiddink et al. 2017, 2019, 2026).
  - ICES WGFBIT uses it as RBS_tot and RBS_sen.
  - **Neither ICES nor any paper I found feeds RBS into fish productivity.** WGFBIT's extensions go to bioturbation and
    sediment carbon, not fish.
- **Half 2, habitat to fish: generic forms only, none with gadoid parameters.**
  1. Bioeconomic K(H) = f + gH and r(H) = a + bH (Foley et al. 2012). This is the form the proposed coupling uses.
  2. Lindholm et al. 2001: a habitat-complexity class to 0-group cod survival, through three density-dependent mortality
     curves. The input is habitat complexity, not SAR, and the parameters were not retrievable.
  3. Ecosim mediation functions: user-drawn and "difficult to quantify or scale from empirical or experimental
     studies" (Harvey 2014).
  4. Atlantis: a recruitment habitat scalar equal to the proportion of suitable habitat in the cell, plus a saturating
     habitat-refuge term on prey availability. Biogenic cover counts as habitat (CSIRO user guide 2025).
  5. van Denderen et al. 2013 and Hiddink et al. 2008: food-web links through benthic prey, generic benthivores.
- **The proposed form K_eff = K·(1 + share·ΔH) is therefore structurally the Foley/Atlantis form.** It is a linear
  habitat scalar on capacity or recruitment, with a facultative-habitat intercept f/K_today = 1 - share·H_0. Its slope
  (share × elasticity) has no empirical estimate for any gadoid.

## A. SAR to benthic state (RBS)

- **Equation** [Primary, FULL via `research_notes/Bottentrålning och bentos/depletion_and_recovery.md`].
  - Pitcher et al. 2022, *PNAS* 119:e2109449119, doi [10.1073/pnas.2109449119](https://doi.org/10.1073/pnas.2109449119):
    "B/K = 1 - F D/R where F < R/D, otherwise B/K = 0".
  - For mixed gears, "the cumulative RBS was estimated by summing the depletion (Fd/R)".
  - At the 1-3 km2 cell scale, R = r d/[-ln(1 - d)].
- **Origin** [Primary, ABS]. Pitcher CR, Ellis N, Jennings S, Hiddink JG, Mazor T, Kaiser MJ, et al. 2017. Estimating
  the sustainability of towed fishing-gear impacts on seabed habitats: a simple quantitative risk assessment method
  applicable to data-limited fisheries. *Methods Ecol. Evol.* 8:472-480.
  doi [10.1111/2041-210X.12705](https://doi.org/10.1111/2041-210X.12705).
  - RBS is "derived by solving the logistic population growth equation for the equilibrium state".
- **European assessment** [Primary, ABS]. Hiddink JG, Valanko S, Batts L, Beukhof E, et al. 2026. Assessment of bottom
  trawl impacts on the status of seabed communities in European seas. *Fish Fish.* 27:285-299.
  doi [10.1111/faf.70054](https://doi.org/10.1111/faf.70054).
  - Indicators: RBS_tot (whole community) and RBS_sen (the 10 % most sensitive fauna).
  - "A strong correlation emerged between the mean trawling intensity and RBS tot and RBS sen, allowing the use of SAR
    to estimate ecosystem status."
  - RBS_tot is above 0.75 in 79 % of habitat-region combinations, RBS_sen in only 46 %.
- **Longevity version (PD method)** [Primary, ABS here; FULL via Rijnsdorp et al. 2020 in the earlier note].
  - Hiddink et al. 2019, *J. Appl. Ecol.* 56:1075-1084, doi [10.1111/1365-2664.13278](https://doi.org/10.1111/1365-2664.13278).
  - I = 1 - Σ_i K_i (1 - Σ_m T_m d_m/r_i) over longevity classes, with r_i ≈ 5.31/longevity.
  - This is the natural way to get a sensitive-fauna H (RBS_sen).
- **ICES WGFBIT** [Report, SNIP of abstract]. ICES 2025, *Working Group on Fisheries Benthic Impact and Trade-offs
  (WGFBIT; outputs from 2024 meeting)*, ICES Scientific Reports 7:18, doi [10.17895/ices.pub.28351412](https://doi.org/10.17895/ices.pub.28351412).
  - Uses RBStot and RBSsen.
  - The ecosystem-function extension (BFIAT) predicts bioturbation and feeds sediment carbon models.
  - **No fish-productivity link** appears in the summary. The full report was not read, so this is "not found", not
    "confirmed absent".
- **Footprint context** [Primary, ABS]. Eigaard OR, Bastardie F, Hintzen NT, et al. 2017. The footprint of bottom
  trawling in European waters: distribution, intensity, and seabed integrity. *ICES J. Mar. Sci.* 74:847-865.
  doi [10.1093/icesjms/fsw194](https://doi.org/10.1093/icesjms/fsw194).
  - "Highest intensities were recorded in the Skagerrak-Kattegat."

## B. Habitat to fish productivity (generic forms)

- **Foley NS, Armstrong CW, Kahui V, Mikkelsen E, Reithe S. 2012.** A review of bioeconomic modelling of
  habitat-fisheries interactions. *Int. J. Ecol.* 2012:861635. doi [10.1155/2012/861635](https://doi.org/10.1155/2012/861635).
  [Primary, FULL, review and theory]
  - Growth: F(X,H) = r(H) X (1 - X/K(H)), with r(H) = a + bH and K(H) = f + gH.
  - Habitat is "essential" if a = 0 and/or f = 0, and "facultative" if a, f > 0.
  - Case 1 (habitat acts on K only): F = aX(1 - X/(f + gH)).
  - Applications are mangroves, marshes, seagrass and cold-water coral with redfish. **There is no gadoid
    parameterisation.**
  - The proposed coupling is Case 1. Written as K_eff = K_today·(1 + s·(H - H_0)), it has g = s·K_today and
    f = K_today·(1 - s·H_0). Here H_0 is today's (noll) H, so K = K_today at today's state.
  - The habitat is facultative (f > 0) for any s < 1/H_0.
- **Lindholm et al. 2001**, *Conserv. Biol.* 15:424-437,
  doi [10.1046/j.1523-1739.2001.015002424.x](https://doi.org/10.1046/j.1523-1739.2001.015002424.x).
  [Primary, ABS, spatial model]
  - Habitat classes run "pebble-cobble substratum with emergent epifauna > pebble-cobble > sand".
  - "Density-dependent natural mortality was specified as three alternative functional response curves ... during the
    first 12 months of demersal existence".
  - Applied to a theoretical hard-bottom patch and to Stellwagen Bank.
  - Forms and parameters were not retrievable. The survival contrast is anchored in the Lindholm 1999 tank result:
    roughly 2-3x lower 4-h mortality with dense sponge than on sand (`mechanism_evidence.md`).
  - The model does not take SAR as input.
- **Fogarty MJ. 2005.** Impacts of fishing activities on benthic habitat and carrying capacity: approaches to assessing
  and managing risk. In: Barnes PW, Thomas JP (eds), *Benthic Habitats and the Effects of Fishing*. AFS Symposium 41.
  [Report/chapter, SNIP] Chapter DOI and pages unconfirmed.
  - Fishing removes biomass and degrades habitat that limits K, which shifts reference points.
  - No equations were obtained. This is probably what Collie et al. 2017 mean by "Models suggest that reduction in the
    carrying capacity ... could lead to lower equilibrium yield".
- **Ecosim mediation** [Primary, ABS].
  - Harvey CJ. 2014. Mediation functions in Ecopath with Ecosim: handle with care. *Can. J. Fish. Aquat. Sci.*
    71:1020-1029. doi [10.1139/cjfas-2013-0594](https://doi.org/10.1139/cjfas-2013-0594).
  - A mediating group's biomass scales a predator-prey vulnerability. Linear, hyperbolic and sigmoid shapes are user
    choices; results were "nonintuitive" as the lower limit approached 0.
  - "these functions are difficult to quantify or scale from empirical or experimental studies".
  - "Hyperbolic mediation curves may be the most conservative when empirical or theoretical knowledge is unavailable".
  - Christensen V, Walters CJ. 2004. Ecopath with Ecosim: methods, capabilities and limitations. *Ecol. Model.*
    172:109-139. doi [10.1016/j.ecolmodel.2003.09.003](https://doi.org/10.1016/j.ecolmodel.2003.09.003). Not fetched.
  - No North Sea or Skagerrak gadoid mediation parameters were found.
- **Atlantis** [Report, FULL]. Audzijonyte A, Gorton R, Kaplan I, Kasper JM, Fulton EA, et al. 2025. *Atlantis User's
  Guide Part I: General Overview, Physics & Ecology*. CSIRO.
  [PDF](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf)
  - **Recruitment, section 10.9.3, item 6:** "Habitat scalar on recruitment acts as an additional multiplier on all
    environmental scalars above. It is applied if species recruitment is habitat dependent (XXX_rec_HabDepend=1). The
    habitat scalar is calculated as a proportion of suitable habitat in the cell."
  - **Refuge, section 10.2.4:** an optional prey refuge for age-structured groups, a function of the cell's suitable
    cover with juvenile and adult parameters Acov, Bcov and Kcov. Cover is
    (δ_substrate · p_substrate + p_biogenic)(1 + p_canyon), where p_biogenic is cover by epibenthic groups flagged
    isCover.
  - So **Atlantis has the structure** for "trawled epibenthos lowers juvenile refuge and recruitment". The guide gives
    no default parameter values; they are set per model.
- **van Denderen et al. 2013**, *Proc. R. Soc. B* 280:20131883, doi [10.1098/rspb.2013.1883](https://doi.org/10.1098/rspb.2013.1883).
  [Primary, FULL]
  - Fish feed on susceptible and resistant benthos; benthos r = 1/yr, B_max 2.5 (arbitrary units); fish mu = 0.1/yr.
  - The susceptible benthos is 40x more trawl-vulnerable.
  - The fish response to trawling is negative only if susceptible benthos is the most profitable prey. This is a food
    pathway, and generic.
- **Not checked in primary sources:**
  - Fogarty & Murawski 1998 (*Ecol. Appl.* 8:S6-S22);
  - Armstrong & Falk-Petersen 2008 (*ICES J. Mar. Sci.* 65:817-821, doi [10.1093/icesjms/fsn092](https://doi.org/10.1093/icesjms/fsn092));
  - Ecospace habitat capacity (Christensen et al. 2014, *Ecosystems* 17:1397-1412, doi [10.1007/s10021-014-9803-3](https://doi.org/10.1007/s10021-014-9803-3));
  - OSMOSE.
  None of them is known to me to carry gadoid parameters. That is not verified.

## C. [Own] The benthic index H on the Mareld grid

Inputs and method:
- The repo's bottom-trawl effort map (`maps/fishing_bottom_otter_trawls.npz`, grid mean 3.01 MW-h/km2/yr).
- SAR = 0.75 x effort, from `swept_area.md`.
- Park polygons area-weighted from `vindparker.npz`.
- RBS_i = max(0, 1 - d·SAR_i/r) per 1 km cell.
- "Park untrawled" sets the park's area fraction of each cell to H = 1 at equilibrium.
- The depth band uses the EMODnet DTM, recomputed in scratch (`nursery_areas.md`).

| d | r (/yr) | H grid today | H park today | ΔH grid (park closed, effort removed) | ΔH, 150-300 m band of the grid | ΔH in park cells at 150-300 m |
|---|---|---|---|---|---|---|
| 0.10 | 0.82 (Hiddink 2017; the model's benthos r) | 0.72 | 0.49 | **0.074** | 0.17 | 0.58 |
| 0.10 | 0.68 (Pitcher mud) | 0.67 | 0.38 | 0.089 | 0.20 | 0.70 |
| 0.10 | 0.48 (Pitcher mud, lower CL) | 0.55 | 0.19 | 0.117 | 0.26 | 0.91 |
| 0.06 | 0.82 (Hiddink otter-trawl mean d) | 0.84 | 0.69 | 0.045 | 0.10 | 0.35 |

- The park is 14.5 % of the grid area, but it carries 27 % of the effort (SAR 4.2 against 1.9 outside). That is why
  ΔH_grid (0.05-0.12) is well above 14.5 % x the grid-mean depletion.
- **Displacement matters more than the parameters.**
  - RBS is linear in SAR until it clips at 0. If the park's effort is *redistributed* outside rather than removed, the
    grid-mean H barely moves: **ΔH_grid 0.005 at r 0.82 and 0.010 at r 0.68**. The small gain comes only from cells
    clipped at H = 0, and partial park cells.
  - Georges Bank shows effort concentrating at closure boundaries (Murawski et al. 2005; see
    `benthic_recovery_and_wind_farms.md`).
  - So the `projekt` effort map must state whether the park's effort leaves the grid or moves within it.
- **Sensitive-fauna H** (RBS_sen; sea pens, longevity of 10-30 yr, r about 0.2-0.5) is much lower today and rises
  more. It also recovers over 10-20+ yr rather than 4 yr (`benthic_recovery_and_wind_farms.md`).
- **The model already simulates H.**
  - `benthic_community` has logistic r 0.82 /yr (library growth_rate 0.000562 per 6-h tick x 1460).
  - It is depleted per cell by the same trawl map at d x SAR, so B/K per cell is the model's own RBS_tot, plus the
    effect of gadoid and herring predation.
  - Note: `research_notes/Bottentrålning och bentos/depletion_and_recovery.md` (15 Oct) gives the model's benthos r as
    1.46 /yr. That predates a library change; the library now reads 0.000561644 per tick, i.e. 0.82 /yr.
  - Any H used for a K coupling should be read from that state, normalised to the noll-scenario value. It should not
    be recomputed separately.
