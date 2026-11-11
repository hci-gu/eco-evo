# January light deficit - evidence (2026-11-11)

Grades: [F] field data or full text, [L] lab, [S] secondary, [D] derived here, [J] judgement, [M] model fact (code,
library, manifest). Scripts in `scripts/`, stdout and json in `tables/`. Nothing in the library, code, manifest,
tests, AGENTS.md or resume was changed; no model output was read; no downloads.

## Model facts

M1 [M] `lib/world/daylight.py` (section 137): a pair with `dark_ratio` rho < 1 has a(t) = a m(t),
   m(t) = (rho + (1 - rho) F(t)) / annual mean, F = tick mean of a logistic in the sun's elevation around
   `light_threshold_deg` (width 2 deg). Day length and twilight only; no irradiance, no cloud, no water.
M2 [M] `interactions.py` 246-261: the (ticks, N_dm, N_all) table `light_mult_table` is filled once per env from
   `daylight.attack_multiplier`; `attack_rate()` multiplies `max_intake_mat` by the tick's row. GPU
   (`lib/gpu/ecosystem.py` 233-234, 962) copies the same table (`self.light_mult`) - no GPU code of its own.
   `natural_mortality.py` 102 also calls `attack_multiplier` (M1 visual share, unset in the library).
M3 [M] Observation: `light_obs_table = daylight.light_schedule(lat, tick)` at the default threshold -6 deg - F, a
   0-1 day/twilight fraction per tick, one channel (section 137). The DMs never observe intensity or the month.
M4 [M] Manifest `mareld2.yaml`: latitude 58.15, Kd 0.29 (producers' mixed layer only), cloud transmission
   Jan..Dec .66 .66 .73 .78 .75 .74 .74 .73 .70 .66 .60 .69 (SMHI Nordkoster global radiation 2011-2025 over the
   Haurwitz clear sky, resume 138.3 [F there]). The producers' light path (daylight.clear_sky_par x cloud) exists;
   the fish pairs do not use it.
M5 [M] Library pairs with a dark_ratio (17; rho / threshold / M2): planktivores -> zooplankton 0.03 / 0 / 1.612;
   benthivores -> planktivores 0.141 / 0 / 0.1023; WC -> planktivores 0.0594 / -5 / 0.174; mackerel -> planktivores
   0.16 / -5 / 0.0186; whiting -> planktivores 0.212 / -5 / 0.0545; benthivores -> whiting 0.168 / 0 / 0.18;
   WC -> benthivores 0.165 / -5 / 0.0833; whiting -> benthivores 0.565 / -5 / 0.1067; WC -> whiting 0.137 / -5 /
   0.157; WC -> mackerel 0.923 / -5 / 0.021; WC -> zooplankton 0.51 / -5 / 0.0017; mackerel -> zooplankton
   0.24 / -5 / 0.09; whiting -> zooplankton 0.78 / -5 / 0.0039; planktivores -> krill 0.57 / 0 / 2.071; benthivores
   -> krill 0.036 / 0 / 0.035; WC -> krill 0.0245 / 0 / 0.119; mackerel -> krill 0.42 / -5 / 0.125. Whiting -> krill
   (M2 0.187) has none (stomachs: no diel difference). No mammal or seabird pair has a dark_ratio.
M6 [M] Presence in the engine (library `outside_season`): WC .22 .16 .23 .52 .97 1.49 1.80 1.80 1.78 1.53 .97 .52;
   mackerel .12 .11 .10 .32 1.05 2.10 2.12 2.19 1.85 1.24 .57 .22; planktivores .80 .80 .81 .89 1.07 1.26 1.33 1.33
   1.14 .95 .82 .81; krill 1 1 1 .6 .6 .6 1.1 1.1 1.1 1.3 1.3 1.3. Whiting and benthivores resident.
M7 [M] The derivation chain's light (research_notes/"En synregel for fiskparen"/scripts/one_rule_pairs.py
   `trans_month`, SECCHI; "Krill i Skagerrak"/wc_depth/scripts/krill_pairs_wc.py `surface_lux`, E0):
   surface lx = 1.3e5 sin(h) (twilight table below 5 deg) x a constant cloud factor 0.5; K(0-2) 0.30, K(2-20) =
   1.56 / Secchi(month), K(20-50) 0.109, K(50-65) 0.091, K_deep 0.06, the same every month except K(2-20);
   Secchi (m) Jan..Dec 5.0 7.0 7.0 8.5 8.0 7.0 10.0 9.5 8.0 9.0 9.0 11.5. Gadoid eye Ke 0.02 lx; clupeid eye Ke 1 lx
   with a biting threshold 0.016 lx. Time grid: 2 days per month, 15-min steps.
M8 [S] research_notes/"Sillens vinterprofil" section 4: derived/engine in January 0.66 (WC -> planktivores) and
   0.84 (whiting -> planktivores) with the July herring profile, Nov-Mar 0.88 / 0.94.

## Light data (sources)

E1 [F] Monthly Secchi (research_notes/"Ljus under 50 m"/tables/secchi_and_deep_water_stdout.txt, SHARK):
   A17 January 5.0 m (n 1), December 11.0 (n 1); A16 December 12.0 (n 1), no January or November visit. The
   derivation's Jan 5.0 and Dec 11.5 are these three visits. Slaggo (75 m, coastal, 2018-2024, n 12-19 per month):
   Jan 5.5 (14), Feb 5.0 (17), Nov 5.8 (18), Dec 5.0 (13), Jul 8.0 (12) - no Dec/Jan step.
E2 [F data / D analysis] SMHI CTD PAR casts (QSP-2000; par_casts.json of "Ljus under 50 m"), daytime (sun > 5 deg),
   bed >= 60 m, reliable to PAR >= 0.5 umol m-2 s-1 (winter_light.py): K(2-20) January median 0.192 (n 10, casts
   from 2018, 2019, 2024, 2025; 0.182 over beds >= 100 m, n 8); Jan-Feb 0.193 (n 13, p25-p75 0.171-0.230); Nov-Feb
   0.191 (n 15); Apr-Sep 0.167 (n 42). Same with the 0.14 floor: Jan 0.199. Measured at the actual low winter sun,
   so the path-length effect is inside.
E3 [F, weak] K(20-50) winter: n 3 (Jan 0.101, Feb 0.132 / 0.164; Jan-Feb median 0.132) vs Apr-Sep 0.117 (n 34).
   The winter casts with low surface PAR stop at 13-25 m (sensor floor), so K(20-50) is rare in winter.
E4 [F data, J use] PAR at 2 m / Haurwitz clear-sky PAR: January median 0.047 (n 12), Nov-Feb 0.067, Apr-Sep 0.207.
   Spread 0.009-0.65 in January; the 2018-2019 casts are 0.01-0.07, the 2025 casts 0.40-0.65. Instantaneous, ship
   sensor (hull shadow at low sun likely), cloud okta 6-9 on most January casts. Not a climatology; used only as a
   dark bracket (R5).
E5 [F, resume 138.3] SMHI Nordkoster monthly global radiation 2011-2025 / Haurwitz: Jan 0.66, Nov 0.60, Dec 0.69,
   summer 0.73-0.78 (M4).
E6 [F/D] K(2-20) x Secchi = 1.56 (n 6 same-visit pairs at A17, "Ljus under 50 m" section 1); the derivation's
   K = 1.56 / Secchi.
E7 [F] No PAR above the sensor floor below ~65 m in any month; K below 65 m (0.06, 0.05-0.09) is a judgement
   ("Ljus under 50 m" section 2) and is held constant through the year.
E8 [F] CTD chlorophyll fluorescence at 50-150 m on the Feb 2023 casts 0.33-0.50 mg m-3 (Apr-Sep median 0.04-0.13):
   winter mixing can make the deep water darker in some winters (par_profiles_thr0.14_stdout.txt); no K measured.

## Derived (light_pairs.py, flow_shift.py)

Readings: R0 the derivation as is; R1 R0 with K(2-20) for Jan = the January cast median 0.192 and for Dec = the
Nov-Feb cast median 0.191 (the two n = 1 Secchi months replaced, nothing else); R2 Secchi from the Slaggo monthly
shape scaled to the box median 8.2 m (x1.262; Jan 6.9, Dec 6.3, Jul 10.1 m); R3 R1 + surface = 118.4 lm/W x
Haurwitz (air mass) x the manifest's monthly cloud (E5) instead of 1.3e5 sin h x 0.5; R4 R3 + K(20-50) 0.132 in
Dec-Feb (E3); R5 R3 + Nov-Feb surface darkened to the casts' winter/summer ratio (E4; bracket); R6 R2 + R3's surface.

D1 [D] Reproduction: R0 gives January derived/engine 0.63 (WC -> planktivores) and 0.81 (whiting -> planktivores)
   at the library night/day, vs "Sillens vinterprofil" 0.66 / 0.84 at its own (combined) night/day (M8).
D2 [D] Daytime median light at 100 m (lx): R0 Jan 0.022, Dec 0.43, Jul 1.38; R1 Jan 0.19, Dec 0.16; R3 Jan 0.16,
   Dec 0.13, Jul 1.83; R6 Jan 0.090, Dec 0.047. The R0 January light below 20 m is ~8.7x darker than the casts'
   K gives, and December ~2.6x brighter (exp of the K difference over 18 m).
D3 [D] Residual derived/engine (night/day matched to the library rho), gadoid band pairs (WC -> planktivores,
   -> benthivores, -> whiting; benthivores -> planktivores, -> whiting): R0 Jan 0.63-0.70, Dec 0.91-1.00
   (the V); R1 Jan 0.85-0.91, Dec 0.81-0.90; R3 Jan 0.83-0.88, Dec 0.79-0.87; R2 / R6 Jan 0.77-0.85,
   Dec 0.69-0.80; R5 Jan 0.74-0.80, Dec 0.70-0.79. Nov-Mar mean 0.83-0.92 in every reading; Jul-Aug 1.08-1.15.
D4 [D] Whiting pairs: whiting -> planktivores Dec-Jan 0.84-0.92 (R1-R6), whiting -> benthivores 0.90-0.95.
   Zooplankton pairs (shallow, saturating light): 0.93-1.00 except R5. Planktivores -> zooplankton at 25 m
   (bracket) R0 Jan 0.83, R3 0.98.
D5 [D] Clupeid krill pairs: planktivores -> krill (stomach rho 0.57 imposed on the rule's series) Dec-Jan
   0.82-0.84 (R1, R3), 0.70-0.77 (R2, R4-R6), Jul 1.27-1.32 - the largest swing beyond m(t); mackerel -> krill and
   WC -> mackerel carry non-light seasonal terms of the derivation (spurdog's Jun-Sep surface profile in the
   EATERS; mackerel members) and are not interpretable as light residuals.
D6 [D] Annual equal-time a of the derivation under R1-R6 relative to R0: within -1.7 % to +1.5 % for every pair but
   mackerel -> krill (0.86-1.00; clupeid eye near its threshold, surface-convention sensitive). Replacing the n = 1
   Secchi months does not move the library a's.
D7 [D] Flow timing (linear regime, presence-weighted, booked flows M2 x inference biomass): Dec + Jan share of the
   annual flow engine -> derived (R3): WC -> planktivores 2.5 -> 1.9 % (17 t/yr of 2958), planktivores -> krill
   13.8 -> 11.2 % (986 t/yr of 37 278), whiting -> planktivores 9.6 -> 8.5 % (10 t), benthivores -> planktivores
   8.5 -> 7.4 % (19 t); every other pair <= 8 t/yr. Total |shift| over the year <= 3.3 % of a pair's flow for the
   band pairs, 6.4 % planktivores -> krill.
D8 [D] Predator Dec-Jan intake, derived / engine timing (all booked pairs; non-light pairs flat): WC 0.86 (R3;
   0.77-0.88 over readings); planktivores 0.91 (0.86-0.92); whiting 0.985 (0.98-0.99); benthivores 0.993;
   mackerel 0.94-1.07 (absent in winter). Ranges over R1-R6. Prey Dec-Jan predation: krill 0.85 (0.74-0.86), planktivores 0.93
   (0.89-0.94), whiting 0.91 (0.86-0.93), benthivores 0.96 (0.94-0.96), zooplankton 0.996.
D9 [D] Prey stocks: krill is an open pool, exchange k 20 /yr (outside_exchange_rate 0.0137 per tick) against a
   booked M2 ~2.5 /yr; a 15 % Dec-Jan cut in M2 moves the winter standing stock by ~2.5 x 0.15 / (20 + 2.5) ~ 2 %.
   Planktivores: total booked M2 ~0.36 /yr; -7 % for two months ~0.4 % of the stock. Neither matters for the prey.
D10 [D, design] The residuals D3-D5 are whole-m(t) ratios. The rule's intensity effect sits in the DAY term (the
   night term is R_nv in every season), so a monthly factor carrying it belongs on the light term,
   m(t) = (rho + (1 - rho) F(t) s(month)) / mean, not on the whole m(t); for rho <= 0.2 the difference is second
   order, for whiting -> benthivores (0.565), planktivores -> krill (0.57) and mackerel -> krill (0.42) it is not.
   The s(month) values were not computed here.
