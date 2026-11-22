"""Variant definitions for variants.py: each tied to a named field source (../evidence.md, ../q1_evidence.md).

Krill-layer variants give the gap (m) between the day layer's BOTTOM edge and the bed at anchor beds; between the
anchors the gap is linear, outside them the section 262 library layer (whose gaps are 40 m at 150 m, 4 m at 160 m,
13 m at 170 m, 47.5 m at 200 m, 105 m at 250 m). The 170 m anchor keeps the library gap (in-box Dana tows, 157-166 m
beds), so every krill variant changes beds 170-250 m only.
"""


def defs(h):
    la, mix, L0 = h["layer_anchor"], h["saithe_mix"], h["LAYER0"]
    g170 = h["lib_gap"](170.0)
    g250 = h["lib_gap"](250.0)
    # Q1 E-far: Kaartvedt et al. 2002 Stn D (Drobak sound, 200 m bed): positive day tows' bottom edge ~87 m above the
    # bed in Aug, 0 at 150 m in Nov (deepest positive 107 m), deepest positive 152-158 m in May; Balino & Aksnes 1993
    # Masfjorden ~195 m bed 45-65 m up; Degnbol & Munch-Petersen 1985 250 m bed 90-150 m. Central gap at 200 m: 87 m.
    far = la([(170.0, g170), (200.0, 87.0), (250.0, g250)])
    # Q1 E-near (analogue): Scotian Shelf basins, Cochrane et al. 1994 / Herman et al. 1993: the layer maximum 20-30 m
    # above 190-250 m beds, its lower edge at the 7 m acoustic bottom gap. Gap 7 m from 190 to 245 m beds; library at
    # >= 250 m (Degnbol 1985 kept at the class edge). Contradicted in the box by 0 of 4 day tows with the footrope
    # 0-3 m above ~204 m beds (Dana); supported in kind by GOV near-bed presence in Q3 (3/16, 200-250 m beds).
    near = la([(170.0, g170), (190.0, 7.0), (245.0, 7.0)])
    return {
        "krill_far_oslofjord200": dict(layer=far, src="Q1: Kaartvedt et al. 2002 Stn D (200 m bed) gap 87 m;"
                                       " Balino & Aksnes 1993; Degnbol 1985 at 250 m [F/D]"),
        "krill_near_scotian": dict(layer=near, src="Q1: Cochrane et al. 1994, Herman et al. 1993 (Scotian basins,"
                                   " analogue) gap 7 m over 190-245 m beds [F/D]"),
        "krill_near_Q3only": dict(layer=dict(summer=near, winter=L0, months=[6, 7, 8]),
                                  src="Q1: GOV Q3 near-bed presence over 150-250 m beds (DATRAS) [D] x the Scotian"
                                      " geometry, Jul-Sep only; library the rest of the year"),
        "saithe_runs0.15": dict(saithe=mix(0.15), src="Q2: <= 0.15 of the day in the krill layer: Armannsson &"
                                " Jonsson 2012 rapid vertical runs 0.15 of the time (upper bound; not linked to"
                                " prey) [F/J]"),
        "Ke_2e-3_F1": dict(Ke=2e-3, src="Q3: Aksnes & Utne 1997 fit of Anthony 1981 cod thresholds in downwelling lx"
                           " at F 1 (Torskfiskars syn i morker E1) [D]"),
        "Ke_0.06_F30": dict(Ke=0.06, src="Q3: same fit at F 30 (upper edge of the bracket) [D]"),
        "Ke_2.4e-4_scot": dict(Ke=2.4e-4, src="Q3: scotopic-branch refit of Anthony 1981 (Torskfiskars syn i morker"
                               " E1 row 3; its plateau R^2 is ~1/3 of the bright one, not modelled) [D/J]"),
        "Rnv_fish_0.05": dict(rnv_fish=0.05, src="Q3: burbot dark reactive distance scaled by predator length"
                              " (Michels 2020; Torskfiskars syn i morker E2), low edge [D/J]"),
        "Rnv_fish_0.5": dict(rnv_fish=0.5, src="Q3: same, scaled by prey size, high edge [D/J]"),
        "near_scotian+Ke_2e-3": dict(layer=near, Ke=2e-3, src="combination of krill_near_scotian and Ke_2e-3_F1"),
        "sens_Rnv_krill0.25": dict(rnv_krill=0.25, src="SENSITIVITY ONLY [J], no field source: dark detection of"
                                   " krill 0.25 m (krill_pairs_wc.py 'biolum' centre); Q3 found no measurement"),
        "sens_dark_near": dict(Ke=1e9, layer=near, src="DECOMPOSITION ONLY: sens_dark with the krill_near_scotian"
                               " layer (geometry left once the krill reach the bed)"),
        "sens_dark": dict(Ke=1e9, src="DECOMPOSITION ONLY: every visual member at R_nv (geometry and member mix)"),
    }
