"""Stage access helper (no output): the rest water column's [f_J, f_A] on benthivores, planktivores and
whiting under the library method and the corrected bookkeeping, for clearance_test.py.

Imported, not run. load(arsungarnas_dir) imports research_notes/"Arsungarnas andel" (A-ring)
scripts/stage_access.py (windows, box length mixes, prey stage tables) and returns an object with
access(pair, variant) -> (f_J by quarter, f_A by quarter).

Variants (see stage_access_check.py): "lib" (the library method), "bw+all" (species by stage biomass,
normalised over the whole group), "aw" (bw+all with each predator member's window averaged over its
length mix with weights biomass(L) x a(L), a(L) ~ L^-2, anglerfish L^-3).

Pairs and member a-shares: benthivores and whiting as the library (saithe 0.77, spurdog 0.10, hake
0.07, velvet belly 0.04, anglerfish 0.01); planktivores as research_notes/"Makrillens bytesstorlek"
mackerel_stage_access.py REST_PLK (saithe 0.33, hake 0.03, spurdog 0.02, velvet belly 0.01, renormalised).
"saithe" alone is available as a pair for each group.
"""
import importlib.util
import os
import sys

import numpy as np

EXP = {"saithe": 2.0, "spurdog": 2.0, "hake": 2.0, "anglerfish": 3.0}


class Access:
    def __init__(self, ad):
        ad = os.path.abspath(ad)
        p = os.path.join(ad, "scripts", "stage_access.py")
        spec = importlib.util.spec_from_file_location("stage_access_mod", p)
        argv = sys.argv
        sys.argv = [p, os.path.join(ad, "tables", "stage_length_biomass.csv"), "/nonexistent_out_not_used"]
        mk = os.makedirs
        os.makedirs = lambda *a, **k: None
        try:
            sa = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sa)
        finally:
            os.makedirs = mk
            sys.argv = argv
        self.sa = sa
        self.orig = dict(sa.PRED_LEN)
        self.aw_len = {}
        for m, k in EXP.items():
            Ls, ws = self.orig[m]
            w2 = ws * Ls ** (-k)
            self.aw_len[m] = (Ls, w2 / w2.sum())
        rp = [("saithe", 0.33), ("hake", 0.03), ("spurdog", 0.02), ("velvet belly", 0.01)]
        t = sum(w for _, w in rp)
        wc = [("saithe", 0.77), ("spurdog", 0.10), ("hake", 0.07), ("velvet belly", 0.04), ("anglerfish", 0.01)]
        self.pairs = {
            "ben": ("ben", [(m, w, sa.GADOIDS if m == "saithe" else (sa.FLAT | {"haddock"} if m == "spurdog" else None))
                            for m, w in wc]),
            "plk": ("plk", [(m, w / t, None) for m, w in rp]),
            "wht": ("wht", [(m, w, None) for m, w in wc]),
            "saithe_ben": ("ben", [("saithe", 1.0, sa.GADOIDS)]),
            "saithe_plk": ("plk", [("saithe", 1.0, None)]),
            "saithe_wht": ("wht", [("saithe", 1.0, None)]),
        }

    def _weights(self, group, q):
        sa = self.sa
        return {"plk": sa.POOL_W[q], "ben": sa.BEN_W, "wht": {"whiting": 1.0}}[group]

    def _stage_share(self, q, sp, stage):
        sa = self.sa
        d = sa.SL[(sa.SL.period == sa.PERIOD[q]) & (sa.SL.quarter == q) & (sa.SL.species == sp)]
        if d.empty:
            return 0.0 if stage == "J" else 1.0
        J, A = d.J.sum(), d.A.sum()
        return (J if stage == "J" else A) / (J + A) if J + A > 0 else 0.0

    def _inside(self, q, sp, window, stage, lens):
        sa = self.sa
        d = sa.SL[(sa.SL.period == sa.PERIOD[q]) & (sa.SL.quarter == q) & (sa.SL.species == sp)]
        if d.empty or d[stage].sum() <= 0:
            return None
        L = d.L_cm.values + 0.5
        v = d[stage].values / d[stage].sum()
        if window[0] == "ratio":
            Ls, ws = lens[window[2]]
            return sum(wp * v[L <= window[1] * Lp].sum() for Lp, wp in zip(Ls, ws))
        return v[(L >= window[1]) & (L <= window[2])].sum()

    def member_f(self, q, group, member, ident, stage, variant):
        sa = self.sa
        lens = dict(self.orig)
        if variant == "aw":
            lens.update(self.aw_len)
        window = sa.WINDOWS[member]
        num = den = 0.0
        for sp, w in self._weights(group, q).items():
            if w <= 0:
                continue
            if variant == "lib":
                if ident is not None and sp not in ident:
                    continue
                fin = self._inside(q, sp, window, stage, lens)
                if fin is None:
                    continue
                num += w * fin
                den += w
                continue
            wt = w * self._stage_share(q, sp, stage)
            if wt <= 0:
                continue
            fin = self._inside(q, sp, window, stage, lens)
            den += wt
            if fin is not None and (ident is None or sp in ident):
                num += wt * fin
        return num / den if den > 0 else np.nan

    def access(self, pair, variant):
        group, members = self.pairs[pair]
        out = {}
        for stage in ("J", "A"):
            out[stage] = [sum(s * self.member_f(q, group, m, ident, stage, variant) for m, s, ident in members)
                          for q in (1, 2, 3, 4)]
        return out["J"], out["A"]
