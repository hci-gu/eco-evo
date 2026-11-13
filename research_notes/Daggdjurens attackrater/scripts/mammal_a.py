"""Encounter-theory attack rates a for the porpoise and seal pairs.

Research note research_notes/"Daggdjurens attackrater". Reads only the depth map
(maps/depth.npz); writes nothing outside its own output path. No model output is
used anywhere.

    python3 -I mammal_a.py <repo> <out.json>        # ~1 min, stdout -> tables/

Unit chain (mirrors research_notes/"Attackrater ur motesteori"/scripts/
pred_encounter_a.py): a [km2 per t of predator per 6 h tick of FORAGING time] =
search cross-section x speed x T x N_t x p x (time share at the prey's depth) / H
/ 1e6. The policy's eat share (time spent foraging rather than resting or
travelling) is the engine's T and is NOT in a. What IS in a and was not needed for
the fish: the dive cycle (surface time, descent and ascent), and the reach of the
bed from the surface.

Every factor is a (low, central, high) triple; the Monte Carlo draws each from a
triangular distribution on the log scale (mode = central). Shares and
probabilities are drawn on the linear scale and clipped to [0, 1].
"""
import json
import math
import sys

import numpy as np

T = 21600.0  # s per tick
RNG = np.random.default_rng(20261113)
NMC = 4000

# ------------------------------------------------------------------ factors
# Grades and sources: summary.md, section "Inputs". (low, central, high)
P = {
    # ---------------- porpoise
    "m_porp": (40.0, 45.0, 62.0),       # kg: SCANS-IV conversion 45 (manifest); Rojano-Donate 2024 model 40; RD 2018 ~62
    "v_porp": (0.8, 1.2, 1.6),          # m/s through water while searching; closing speed 1.4 (Wisniewska 2016); captive dives 0.76-0.91
    "Theta_porp": (6.0, 20.0, 45.0),    # deg, half-angle of the searched cone: beam alone (-3 dB 13 x 11 deg, Koblitz 2012) .. head scanning (J)
    "R_plk": (19.0, 35.0, 60.0),        # m, single planktivore on axis, sonar equation at SL 178/191/205 (mammals.md)
    "R_gad": (23.0, 50.0, 90.0),        # m, 25-35 cm swimbladdered gadoid (mammals.md)
    "c_bed": (0.25, 0.5, 1.0),          # J: bed clutter shortens detection of fish within metres of the bed
    "e_flat": (0.1, 0.3, 0.6),          # J: buried flatfish in the sonar's bed dead zone (D5); porpoises grub (E14)
    "R_school": (86.0, 200.0, 300.0),   # m, school (TS -10..+10) at SL 178-205 (mammals.md)
    "k_porp": (1.0, 5.0, 38.0),         # J: herring taken per school met: 1 (fish rule) .. a full stomach 1.9 kg / 50 g
    "p_porp": (0.5, 0.85, 0.97),        # per encounter: buzz success 0.91-0.99 (Table S1 0.58-0.99) x pre-buzz escape q 0.8-1
    "f_sub": (0.55, 0.69, 0.80),        # submerged share of the dive cycle: 61 s dive / 27 s surface (Rojano-Donate 2024)
    "w_porp": (1.0, 1.3, 1.6),          # m/s vertical transit (J)
    "Tmax_porp": (150.0, 220.0, 330.0), # s, longest routine dive (95 % < 127 s, max 270 s, RD 2024; deep dives 132-152 m Skagerrak)
    "tb0_porp": (40.0, 70.0, 100.0),    # s, wanted bottom time (bottom-feeding dives 74-78 s in 11 m water, RD 2024)
    "phi_b_day": (0.1, 0.4, 0.6),       # share of day foraging dives that go to the bed where reachable (bottom 25/62, RD 2024)
    "phi_b_night": (0.0, 0.1, 0.3),     # night: "primarily pelagic dives" (Wisniewska 2016)
    "O_pel_day": (0.4, 0.8, 1.0),       # J: share of a pelagic day dive spent inside the 10-100 m herring layer
    "O_pel_night": (0.6, 1.0, 1.0),     # J: herring dispersed in the upper 30-60 m at night
    # ---------------- seal
    "m_seal": (45.0, 55.0, 80.0),       # kg: population mean 55 (library unit), adult 80
    "v_seal": (0.5, 1.0, 1.4),          # m/s at the bed: 1.4 typical (Vance 2021), ~0.5 near the vibrissal threshold (Niesterok 2017)
    "w_vib": (0.2, 0.5, 1.0),           # m, swath of whiskers + near vision for still prey (J)
    "tau_trail": (15.0, 30.0, 35.0),    # s, trail age still followed (Dehnhardt 2001, Wieskotten 2010)
    "u_gad": (0.1, 0.3, 0.6),           # m/s, swimming speed of an active near-bed gadoid (J)
    "f_follow": (0.4, 0.7, 0.9),        # J: share of crossed trails followed to the fish (glide phases, Wieskotten 2010b)
    "act_ben": (0.2, 0.4, 0.7),         # share of time a benthivore gadoid is swimming (cod ~0.3 active, Hobson 2007; haddock more)
    "act_whi": (0.4, 0.7, 0.9),         # whiting, saithe juveniles, pout: mostly active (J)
    "p_seal_fish": (0.35, 0.6, 0.85),   # per encounter, active fish: Bowen 2002 per attempt 0.64 cryptic / 0.375 schooling sand lance
    "p_seal_flat": (0.6, 0.8, 0.9),     # buried flounder 0.799 per attempt (Bowen 2002)
    "k_seal": (0.6, 1.2, 3.0),          # fish per school contact: 3.2 attempts x 0.375 (Bowen 2002)
    "w_seal": (0.8, 1.1, 1.4),          # m/s vertical transit (J)
    "Tmax_seal": (270.0, 360.0, 480.0), # s: mean dives 3.3-4.4 min (Bjorge 1995, Bowen 2002, Vance 2021) at 10-37 m; longer at depth (J)
    "tb0_seal": (100.0, 150.0, 200.0),  # s wanted bottom time (J)
    "srf_seal": (0.18, 0.25, 0.35),     # surface time / dive time: 80 % of time diving (Bowen 2002 table 2)
    "O_whi_seal": (0.1, 0.3, 0.6),      # J: share of whiting inside a seal's near-bed search band (whiting 20-45 m off the bed, Onsrud 2004)
    "f_pel_seal": (0.0, 0.5, 1.0),      # J: pelagic search where the bed is out of reach, relative to benthic search
    # ---------------- prey
    "H_her_day": (40.0, 70.0, 100.0),   # m, herring day layer 10-100 m (HERAS box)
    "H_her_night": (30.0, 40.0, 60.0),  # m, upper 30-60 m at night
    "sig_day": (0.6, 0.9, 1.0),         # schooled share of herring and sprat by day (Stimskydd)
    "w_her": (0.03, 0.05, 0.13),        # t x 1e-3 (kg) per herring: box 1-2 ringers 50 g; Borjesson 26 cm ~130 g
    "M_s": (1.0, 1.25, 10.0),           # t per school (Misund 1995; Petitgas 2001 upper)
    "H_whi": (15.0, 30.0, 45.0),        # m, whiting layer above the bed
    "z_whi": (10.0, 20.0, 30.0),        # m, centre of the whiting layer above the bed
    "her_bed_shallow": (0.3, 0.5, 0.66),# herring share in the bottom 10 m over beds < 150 m (0.66 -> 0.05 by bed depth)
}

PS = {"herring_sprat": 0.575, "pout": 0.350, "blue_whiting": 0.075}   # 243.3 pool shares
BENTH_MIX_PORP = {"gadoid": 0.64, "flat": 0.36}   # inside the porpoise window (D6)
BENTH_MIX_SEAL = {"gadoid": 0.57, "flat": 0.43}   # inside the seal window (D6)

PROB = {"Theta_porp", "c_bed", "e_flat", "p_porp", "f_sub", "phi_b_day", "phi_b_night", "O_pel_day",
        "O_pel_night", "f_follow", "act_ben", "act_whi", "p_seal_fish", "p_seal_flat", "O_whi_seal",
        "f_pel_seal", "sig_day", "her_bed_shallow", "srf_seal"}


def draw(n):
    out = {}
    for k, (lo, c, hi) in P.items():
        if lo == hi:
            out[k] = np.full(n, c)
            continue
        if k in PROB or lo <= 0:
            x = RNG.triangular(lo, c, hi, n)
        else:
            x = np.exp(RNG.triangular(math.log(lo), math.log(c), math.log(hi), n))
        out[k] = x
    return out


def central():
    return {k: np.array([v[1]]) for k, v in P.items()}


def area_in_slab(R, H):
    """Detection disc of radius R clipped to a slab of thickness H (predator centred)."""
    R = np.asarray(R, float)
    h = np.broadcast_to(np.asarray(H, float) / 2.0, R.shape)
    full = np.pi * R * R
    s = np.clip(h / np.maximum(R, 1e-9), 0, 1)
    clip = 2.0 * (R * R * np.arcsin(s) + h * np.sqrt(np.maximum(R * R - h * h, 0)))
    return np.where(R <= h, full, clip)


def bed_share(D, w, Tmax, tb0, srf):
    """Share of the foraging dive cycle spent in the bed band at depth D (m).

    Dive length = min(Tmax, 2D/w + tb0); bottom time = dive - 2D/w (0 if negative);
    cycle = dive x (1 + srf). D: (cells,), others (n,) -> (n, cells)."""
    D = np.asarray(D)[None, :]
    w, Tmax, tb0, srf = (np.asarray(x)[:, None] for x in (w, Tmax, tb0, srf))
    trans = 2.0 * D / w
    dive = np.minimum(Tmax, trans + tb0)
    tb = np.clip(dive - trans, 0, None)
    return tb / (dive * (1.0 + srf))


def porp_srf(q):
    """surface / dive ratio implied by the submerged share."""
    return 1.0 / q["f_sub"] - 1.0


def porpoise(q, D):
    Nt = 1000.0 / q["m_porp"]
    v = q["v_porp"]
    th = np.radians(q["Theta_porp"])
    p = q["p_porp"]
    base = v * T * Nt / 1e6                       # km2-chain without cross-section and time shares
    col = lambda x: np.asarray(x)[:, None]
    srf = porp_srf(q)
    sb = bed_share(D, q["w_porp"], q["Tmax_porp"], q["tb0_porp"], srf)   # (n, cells)
    sbw = bed_share(np.maximum(np.asarray(D) - P["z_whi"][1], 5.0), q["w_porp"], q["Tmax_porp"], q["tb0_porp"], srf)
    reach = (sb > 0).astype(float)
    reach_w = (sbw > 0).astype(float)
    res = {}

    # ---- planktivores
    rho = q["R_plk"] * np.sin(th)
    vol = lambda H: area_in_slab(rho, H) / H                     # m2/m = m per unit path, per unit density
    # bed form (prey in a thin slab at the bed): cross-section per unit path ~ 2 rho_bed
    rho_bed_plk = q["R_plk"] * q["c_bed"] * np.sin(th)
    strip_plk = area_in_slab(rho_bed_plk, 5.0) / 5.0
    out = {}
    for period in ("day", "night"):
        phib = q["phi_b_" + period]
        Oo = q["O_pel_" + period]
        pel_share = 1.0 - col(phib) * reach                     # share of foraging dives that are pelagic
        bed_time = col(phib) * sb                                     # share of foraging time in the bed band
        if period == "night":
            disp = col(vol(q["H_her_night"]) * base * p * q["f_sub"] * Oo * PS["herring_sprat"])
            her = disp * pel_share
        else:
            H = q["H_her_day"]
            disp1 = vol(H) * base * p * q["f_sub"] * Oo * PS["herring_sprat"]      # all dispersed
            rho_s = q["R_school"] * np.sin(th)
            school = area_in_slab(rho_s, H) / H * base * (q["k_porp"] * q["w_her"] * 1e-3 / q["M_s"]) \
                * q["f_sub"] * Oo * PS["herring_sprat"]
            S = np.minimum(1.0, school / np.maximum(disp1, 1e-30))
            her = col(disp1 * (1 - q["sig_day"] * (1 - S))) * pel_share
            out["S_school"] = S
        pout = col(strip_plk * base * p * PS["pout"]) * bed_time
        out[period] = her + pout
        out[period + "_her"] = her
        out[period + "_pout"] = pout
    a_cells = 0.5 * (out["day"] + out["night"])
    res["porpoises->planktivorous_fish"] = dict(cells=a_cells, nd=out["night"].mean(1) / np.maximum(out["day"].mean(1), 1e-30),
                                              S=out["S_school"], parts={k: out[k].mean(1) for k in ("day_her", "day_pout", "night_her", "night_pout")})

    # bed time averaged over day and night
    bed_time = 0.5 * (col(q["phi_b_day"]) + col(q["phi_b_night"])) * sb
    bed_time_w = 0.5 * (col(q["phi_b_day"]) + col(q["phi_b_night"])) * sbw
    # ---- benthivores: gadoids above the dead zone, flatfish partly in it
    rho_g = q["R_gad"] * q["c_bed"] * np.sin(th)
    strip_g = area_in_slab(rho_g, 5.0) / 5.0
    det = BENTH_MIX_PORP["gadoid"] * strip_g + BENTH_MIX_PORP["flat"] * strip_g * q["e_flat"]
    res["porpoises->benthivorous_fish"] = dict(cells=col(det * base * p) * bed_time)
    # ---- whiting: a layer H_whi about z_whi above the bed; no clutter at that height
    rho_w = q["R_gad"] * np.sin(th)
    strip_w = area_in_slab(rho_w, q["H_whi"]) / q["H_whi"]
    res["porpoises->whiting"] = dict(cells=col(strip_w * base * p) * bed_time_w)
    # ---- water column (accessible part: saithe juveniles and small hake within metres of the bed)
    res["porpoises->predatory_fish"] = dict(cells=col(strip_g * base * p) * bed_time)
    return res


def seal(q, D):
    Nt = 1000.0 / q["m_seal"]
    v = q["v_seal"]
    base = v * T * Nt / 1e6
    col = lambda x: np.asarray(x)[:, None]
    sb = bed_share(D, q["w_seal"], q["Tmax_seal"], q["tb0_seal"], q["srf_seal"])
    unreach = (sb <= 0).astype(float)
    trail = lambda act, u: q["w_vib"] + act * q["f_follow"] * (2.0 / np.pi) * u * q["tau_trail"]
    res = {}
    # benthivores: gadoids (active part leaves trails), flatfish still / buried
    w_g = trail(q["act_ben"], q["u_gad"])
    det = BENTH_MIX_SEAL["gadoid"] * w_g * q["p_seal_fish"] + BENTH_MIX_SEAL["flat"] * q["w_vib"] * q["p_seal_flat"]
    res["seals->benthivorous_fish"] = dict(cells=col(det * base) * sb, width=w_g)
    # whiting: active, partly above the seal's band
    w_w = trail(q["act_whi"], q["u_gad"])
    res["seals->whiting"] = dict(cells=col(w_w * base * q["p_seal_fish"] * q["O_whi_seal"]) * sb)
    # water column: saithe juveniles near the bed, schooling (Bowen school success)
    # x 0.625 = 0.375 / 0.6: Bowen's schooling / single-fish success ratio for loosely schooling saithe (J)
    res["seals->predatory_fish"] = dict(cells=col(w_w * base * q["p_seal_fish"] * 0.625) * sb)
    # planktivores: pout near the bed (active, trail) + herring schools near the bed
    pout = w_w * base * q["p_seal_fish"] * PS["pout"]
    # herring near the bed in schools: a school met per unit path ~ (school width + trail) x school density;
    # per (t/km2) of herring: (2 R_s,eff) x k w / M_s; R_s,eff = school half-length 12 m + trail
    Rs_eff = 12.0 + q["f_follow"] * (2.0 / np.pi) * q["u_gad"] * q["tau_trail"]
    sch = 2.0 * Rs_eff * base * (q["k_seal"] * q["w_her"] * 1e-3 / q["M_s"])
    disp = w_w * base * q["p_seal_fish"]
    S = np.minimum(1.0, sch / disp)
    her = disp * (1 - q["sig_day"] * 0.5 * (1 - S))   # schools by day only (half the time)
    shallow = (np.asarray(D) < 150.0).astype(float)[None, :]
    her_bed = col(her * PS["herring_sprat"]) * (col(q["her_bed_shallow"]) * shallow + 0.05 * (1 - shallow))
    near = col(pout) * sb + her_bed * sb
    # pelagic search where the bed is out of reach: dispersed herring at night in a 3-D layer; detection
    # radius = half the trail swath of a herring cruising at 0.5 m/s; submerged share 1/(1 + srf); J share
    rho_pel = 0.5 * (q["w_vib"] + q["f_follow"] * (2.0 / np.pi) * 0.5 * q["tau_trail"])
    pel = col(area_in_slab(rho_pel, q["H_her_night"]) / q["H_her_night"] * base * q["p_seal_fish"]
              * PS["herring_sprat"] / (1 + q["srf_seal"]) * q["f_pel_seal"] * 0.5) * unreach
    res["seals->planktivorous_fish"] = dict(cells=near + pel, S=S)
    return res


# ------------------------------------------------------------------ the engine pairs today
B = {  # box start densities, t/km2 (mareld2.yaml inference biomass / 3600)
    "planktivorous_fish": 17000 / 3600, "benthivorous_fish": 5500 / 3600, "whiting": 1605 / 3600,
    "predatory_fish": 2220 / 3600,
}
JUV = {"benthivorous_fish": 0.15, "whiting": 0.17}   # annual-mean juvenile start share (library start_share)
TODAY = {  # a, h, stage_access, E prey (MJ/t), assimilation
    "porpoises->planktivorous_fish": (0.0225, 28.6, (1, 1), 6495, 0.9),
    "porpoises->benthivorous_fish": (0.12, 28.6, (1, 0.63), 4500, 0.84),
    "porpoises->whiting": (0.2, 28.6, (1, 0.95), 4800, 0.84),
    "porpoises->predatory_fish": (0.2, 28.6, (0.054, 0.054), 5080, 0.84),
    "seals->benthivorous_fish": (0.1, 21.2, (1, 0.5), 4500, 0.85),
    "seals->whiting": (0.05, 22.6, (1, 0.84), 4800, 0.85),
    "seals->predatory_fish": (0.05, 23.9, (0.029, 0.029), 5080, 0.85),
    "seals->planktivorous_fish": (0.002, 30.54, (1, 1), 6495, 0.88),
}


def bvis(pair):
    prey = pair.split("->")[1]
    fJ, fA = TODAY[pair][2]
    j = JUV.get(prey, 0.0)
    return B[prey] * (j * fJ + (1 - j) * fA)


def ration(pred, a_map, typ3=None):
    """Shared-handling intake (t/t/tick) with every prey attacked (p = 1), T = 1,
    hunger 1, at the box start densities; returns %BM/day and fraction of the
    ceiling of the most-eaten prey."""
    lam, hl = {}, 0.0
    for pair, a in a_map.items():
        if not pair.startswith(pred):
            continue
        b = bvis(pair)
        l = a * b * b if (typ3 and pair in typ3) else a * b
        lam[pair] = l
        hl += TODAY[pair][1] * l
    f = {k: l / (1 + hl) for k, l in lam.items()}
    tot = sum(f.values())
    return 400.0 * tot, f, hl


def main():
    repo = sys.argv[1]
    out_json = sys.argv[2]
    D = np.load(f"{repo}/maps/depth.npz")["depth"].astype(float).ravel()
    east = np.load(f"{repo}/maps/depth.npz")["depth"].astype(float)[:, 40:].ravel()
    classes = [(0, 100), (100, 125), (125, 150), (150, 200), (200, 250), (250, 1000)]

    qc = central()
    qm = draw(NMC)
    res_c = {**porpoise(qc, D), **seal(qc, D)}
    res_m = {**porpoise(qm, D), **seal(qm, D)}
    res_ce = {**porpoise(qc, east), **seal(qc, east)}

    rows = {}
    print("=" * 100)
    print("Derived a (km2 / t predator / tick of foraging), equal-cell box mean over maps/depth.npz")
    print(f"{'pair':34s} {'today':>8s} {'central':>9s} {'MC p5':>9s} {'MC p50':>9s} {'MC p95':>9s} {'x today':>8s} {'east3rd':>9s}")
    for pair in TODAY:
        c = float(res_c[pair]["cells"].mean())
        m = res_m[pair]["cells"].mean(1)
        e = float(res_ce[pair]["cells"].mean())
        p5, p50, p95 = np.percentile(m, [5, 50, 95])
        rows[pair] = dict(today=TODAY[pair][0], central=c, p5=p5, p50=p50, p95=p95, east=e)
        print(f"{pair:34s} {TODAY[pair][0]:8.4f} {c:9.4f} {p5:9.4f} {p50:9.4f} {p95:9.4f} {c / TODAY[pair][0]:8.1f} {e:9.4f}")

    print("\nPer bed-depth class (central a; class mean), and the depth_access_table (class / equal-cell mean)")
    hdr = " ".join(f"{lo:>4d}-{hi:<4d}" for lo, hi in classes)
    print(f"{'pair':34s} {'share':>6s} {hdr}")
    print(f"{'(cell share)':34s} {'':6s} " + " ".join(f"{((D >= lo) & (D < hi)).mean():9.3f}" for lo, hi in classes))
    for pair in TODAY:
        cells = res_c[pair]["cells"][0]
        mean = cells.mean()
        vals = [cells[(D >= lo) & (D < hi)].mean() for lo, hi in classes]
        rows[pair]["class_a"] = vals
        rows[pair]["class_table"] = [v / mean if mean > 0 else 0 for v in vals]
        print(f"{pair:34s} {'a':>6s} " + " ".join(f"{v:9.4f}" for v in vals))
        print(f"{'':34s} {'table':>6s} " + " ".join(f"{v / mean if mean > 0 else 0:9.3f}" for v in vals))

    pp = res_c["porpoises->planktivorous_fish"]
    print("\nPorpoises -> planktivores, central parts (box mean): " +
          ", ".join(f"{k} {float(v[0]):.4f}" for k, v in pp["parts"].items()))
    print(f"  night/day ratio central {float(pp['nd'][0]):.1f}; MC p5-p95 {np.percentile(res_m['porpoises->planktivorous_fish']['nd'], [5, 95])}")
    print(f"  school factor S (day) central {float(pp['S'][0]):.4f}; MC p5-p95 {np.percentile(res_m['porpoises->planktivorous_fish']['S'], [5, 95])}")
    print(f"Seals -> planktivores school factor S central {float(res_c['seals->planktivorous_fish']['S'][0]):.3f}")
    print(f"Seal trail swath (benthivore gadoids) central {float(res_c['seals->benthivorous_fish']['width'][0]):.2f} m")

    print("\nSaturation at the box start densities (B_vis = density x stage access; floors ignored, rest shares ~0.02)")
    print(f"{'pair':34s} {'B_vis':>7s} {'aHB today':>10s} {'aHB deriv':>10s} {'aHB p5':>8s} {'aHB p95':>8s} {'knee today':>10s} {'knee deriv':>10s}")
    for pair in TODAY:
        a0, h = TODAY[pair][0], TODAY[pair][1]
        b = bvis(pair)
        r = rows[pair]
        typ3 = pair == "seals->planktivorous_fish"
        if typ3:   # today's pair is Type III; the derived encounter a is Type II
            k0, kd = 1 / math.sqrt(a0 * h), 1 / (r["central"] * h)
            x0, xd = a0 * h * b * b, r["central"] * h * b
        else:
            k0, kd = 1 / (a0 * h), 1 / (r["central"] * h) if r["central"] > 0 else float("inf")
            x0, xd = a0 * h * b, r["central"] * h * b
        r.update(B_vis=b, aHB_today=x0, aHB_derived=xd, knee_today=k0, knee_derived=kd,
                 aHB_p5=r["p5"] * h * b, aHB_p95=r["p95"] * h * b)
        tag = " (Type III today; derived read as Type II)" if typ3 else ""
        print(f"{pair:34s} {b:7.3f} {x0:10.3f} {r['central'] * h * b:10.2f} {r['p5'] * h * b:8.2f} {r['p95'] * h * b:8.2f} {k0:10.3f} {1 / (r['central'] * h):10.4f}{tag}")

    print("\nImplied ration at T = 1 (whole tick foraging), hunger 1, every prey attacked, box start densities")
    today = {k: v[0] for k, v in TODAY.items()}
    der = {k: rows[k]["central"] for k in TODAY}
    lo = {k: rows[k]["p5"] for k in TODAY}
    out = {}
    for pred, field in (("porpoises", "7-10 (Rojano-Donate 2024 ~10 at 40 kg; Kastelein captive 4-9.5)"),
                        ("seals", "4-6 (Harkonen & Heide-Jorgensen 1991 5.1; routine 3-8)")):
        r0, f0, _ = ration(pred, today, typ3={"seals->planktivorous_fish"})
        r1, f1, _ = ration(pred, der)
        r2, f2, _ = ration(pred, lo)
        ceil = 400.0 / 28.6 if pred == "porpoises" else 400.0 / 21.2
        print(f"{pred}: today {r0:.1f} %BM/d, derived central {r1:.1f}, derived p5 {r2:.1f}; ceiling ~{ceil:.1f}; field {field}")
        for name, f in (("today", f0), ("derived", f1)):
            tot = sum(f.values())
            print(f"   diet shares {name}: " + ", ".join(f"{k.split('->')[1]} {v / tot:.2f}" for k, v in f.items()))
        out[pred] = dict(today=r0, derived=r1, derived_p5=r2, ceiling=ceil,
                         diet_today={k: v / sum(f0.values()) for k, v in f0.items()},
                         diet_derived={k: v / sum(f1.values()) for k, v in f1.items()})

    print("\nPorpoise cell on few planktivores (fix99 episodes: 0.03-0.3 t in a 1 km2 cell), single prey, T = 1")
    for b in (0.03, 0.1, 0.3, 1.0, 4.72):
        f0 = 0.0225 * b / (1 + 0.0225 * 28.6 * b)
        f1 = rows["porpoises->planktivorous_fish"]["central"] * b / (1 + rows["porpoises->planktivorous_fish"]["central"] * 28.6 * b)
        f5 = rows["porpoises->planktivorous_fish"]["p5"] * b / (1 + rows["porpoises->planktivorous_fish"]["p5"] * 28.6 * b)
        print(f"  B {b:5.2f} t/km2: today {400 * f0:5.2f} %BM/d ({f0 * 28.6:.2f} of ceiling); derived central {400 * f1:5.2f} ({f1 * 28.6:.2f}); p5 {400 * f5:5.2f} ({f5 * 28.6:.2f})")
    print("  break-even on planktivores (eat cost 90 MJ/t/tick, 6495 x 0.9 per t): "
          f"{90 / (6495 * 0.9):.4f} t/t/tick = {400 * 90 / (6495 * 0.9):.2f} %BM/d")

    json.dump(dict(rows=rows, ration=out, P=P), open(out_json, "w"), indent=1, default=lambda x: float(x) if np.ndim(x) == 0 else list(map(float, x)))


if __name__ == "__main__":
    main()
