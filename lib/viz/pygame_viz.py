"""Real-time pygame visualiser for Mareld training & inference.

Public entry point is :class:`LiveVisualizer`. The class is intentionally
defensive: any failure inside the visualiser must NEVER take down training
or inference, so the public methods catch all exceptions and degrade to
no-ops (after logging once to stderr).

Usage (train.py)::

    viz = LiveVisualizer(fg_ids=[...], grid_shape=(H, W), mode="train")
    ...
    viz.update_biomass(env.fgs, tick=t, extra={"gen": g, "iter": i, "T": T})
    viz.update_reward(fg_id, reward_value, step=global_step)
    if not viz.pump_events():
        break
    viz.close()

Usage (inference.py)::

    viz = LiveVisualizer(fg_ids=[...], grid_shape=(H, W), mode="inference")
    ...
    viz.update_biomass(env.fgs, tick=t)
    if not viz.pump_events():
        break
    viz.close()

Keys:
    space   pause / resume
    l       toggle log-scale on heatmaps
    r       toggle log-scale on the reward/biomass plot y-axis
    q / esc quit visualiser (training continues)
    1..9    solo-focus one FG (press same digit again to clear)
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


_VIRIDIS_256: Optional[np.ndarray] = None


def _build_viridis_lut() -> np.ndarray:
    """Tiny analytic approximation of viridis (256x3 uint8).

    Avoids a matplotlib dependency. Good enough for live visualisation.
    """
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    # Polynomial fits to the viridis colormap; not exact but visually close.
    r = np.clip(  0.2777 + 0.1056*x + 2.6532*x**2 - 6.9056*x**3 + 5.4419*x**4, 0.0, 1.0)
    g = np.clip(  0.0054 + 1.4040*x - 0.3334*x**2 - 0.4439*x**3 + 0.3322*x**4, 0.0, 1.0)
    b = np.clip(  0.3340 + 1.3838*x - 3.5235*x**2 + 3.0941*x**3 - 1.0884*x**4, 0.0, 1.0)
    lut = np.stack([r, g, b], axis=-1)
    return (lut * 255.0 + 0.5).astype(np.uint8)


def _viridis_lut() -> np.ndarray:
    global _VIRIDIS_256
    if _VIRIDIS_256 is None:
        _VIRIDIS_256 = _build_viridis_lut()
    return _VIRIDIS_256


class _NullViz:
    """Drop-in replacement when pygame is unavailable or init failed.

    All methods are no-ops; :meth:`pump_events` returns True so callers
    keep running.
    """

    enabled = False

    def update_biomass(self, *a, **kw): pass
    def update_reward(self, *a, **kw): pass
    def update_series(self, *a, **kw): pass
    def update_loss_breakdown(self, *a, **kw): pass
    def update_diet_breakdown(self, *a, **kw): pass
    def update_action_fracs(self, *a, **kw): pass
    def update_status(self, *a, **kw): pass
    def begin_rollout_recording(self, *a, **kw): pass
    def end_rollout_recording(self, *a, **kw): pass
    def set_b0_defaults(self, *a, **kw): pass
    def get_b0_override(self, *a, **kw): return None
    def get_b0_overrides(self, *a, **kw): return {}
    def consume_b0_change(self, *a, **kw): return False
    def set_ticks_default(self, *a, **kw): pass
    def get_ticks_override(self, *a, **kw): return None
    def consume_ticks_change(self, *a, **kw): return False
    def set_neval_ticks_default(self, *a, **kw): pass
    def get_neval_ticks_override(self, *a, **kw): return None
    def get_neval_ticks(self, *a, **kw): return None
    def consume_neval_ticks_change(self, *a, **kw): return False
    def set_spawn_templates(self, *a, **kw): pass
    def set_spawn_defaults(self, *a, **kw): pass
    def set_save_dir(self, *a, **kw): pass
    def set_reward_label(self, *a, **kw): pass
    def get_spawn_overrides(self, *a, **kw): return {}
    def consume_spawn_change(self, *a, **kw): return False
    def pump_events(self): return True
    def wait_for_close(self, *a, **kw): pass
    def close(self): pass


class LiveVisualizer:
    """Live pygame window with per-FG biomass heatmaps + rolling reward plot.

    The visualiser owns its own pygame display; do not call from worker
    processes. Constructed in the main process only.
    """

    # Frame budget: render at most this often (seconds between frames).
    # Train calls update_biomass once per ARS iter which is already slow,
    # so this mostly protects inference (per-tick rendering).
    _MIN_FRAME_INTERVAL = 1.0 / 30.0

    def __new__(cls, *args, **kwargs):
        # Headless fallback: if no DISPLAY and SDL_VIDEODRIVER isn't set,
        # silently switch to a dummy driver so CI / nohup runs don't crash.
        if not os.environ.get("DISPLAY") and not os.environ.get("SDL_VIDEODRIVER"):
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        try:
            import pygame  # noqa: F401
        except Exception as e:
            print(f"[viz] pygame unavailable ({e!r}); --visual disabled.",
                  file=sys.stderr)
            return _NullViz()
        return super().__new__(cls)

    def __init__(
        self,
        fg_ids: Sequence[str],
        grid_shape: Sequence[int],
        mode: str = "train",
        reward_window: int = 500,
        cell_px: int = 6,
        fps_cap: int = 30,
        title: Optional[str] = None,
        plot_fg_ids: Optional[Sequence[str]] = None,
        extra_plot_ids: Optional[Sequence[str]] = None,
        ndm_ids: Optional[Sequence[str]] = None,
    ):
        import pygame
        self._pg = pygame
        self.enabled = True
        self.mode = mode
        self.fg_ids = list(fg_ids)
        if not self.fg_ids:
            raise ValueError("LiveVisualizer requires at least one fg_id.")
        # Subset of FGs to show in the reward/time-series plot (e.g. only
        # decision makers in train mode). Heatmaps still cover all FGs.
        if plot_fg_ids is None:
            self.plot_fg_ids = list(self.fg_ids)
        else:
            self.plot_fg_ids = [f for f in plot_fg_ids if f in self.fg_ids]
        # ``extra_plot_ids`` are extra series shown in the plot panel only —
        # no heatmap, no totals/B0 tracking. Used e.g. by --rnd-baseline to
        # overlay random-action baselines (named ``<fid>_rnd``) alongside
        # the trained policies in the biomass/energy tabs.
        self._extra_plot_ids: list = list(extra_plot_ids or [])
        for eid in self._extra_plot_ids:
            if eid not in self.plot_fg_ids:
                self.plot_fg_ids.append(eid)
        # Non-decision-makers: only shown on the 'biomass' tab. On every
        # other tab (reward, energy, ...) they are filtered out by
        # ``_active_plot_ids``. ``_rnd``-baseline series follow the same
        # rule based on their stripped base id.
        self._ndm_ids: set = set(ndm_ids or [])
        self.grid_h, self.grid_w = int(grid_shape[0]), int(grid_shape[1])
        # Heatmaps ska alltid renderas i samma fysiska storlek som ett
        # 32x32-rutnt hade haft. Referens: cell_px=6 vid 32x32 -> 192 px.
        # For andra gridstorlekar skalar vi cell_px sa block-storleken
        # (cell_px * max(grid_w, grid_h)) halls konstant ~ 32 * cell_px.
        _ref_grid = 32
        _ref_cell_px = int(cell_px)
        _ref_block_px = _ref_grid * _ref_cell_px
        _max_dim = max(self.grid_w, self.grid_h, 1)
        self.cell_px = max(1, int(round(_ref_block_px / _max_dim)))
        self.reward_window = int(reward_window)
        self._MIN_FRAME_INTERVAL = 1.0 / max(1, int(fps_cap))

        # Tabs: each tab is an independent time series shown in the plot
        # panel. The user cycles through tabs with TAB / Shift+TAB. The
        # primary tab depends on mode:
        #   train     -> reward, biomass%, energy%
        #   inference -> biomass%, energy%
        # Each tab has its own per-FG rolling buffer of (step, value).
        if self.mode == "train":
            self._tabs = ["reward", "biomass", "energy", "move", "rest", "eat",
                          "predation", "starvation", "impacts"]
        else:
            self._tabs = ["biomass", "energy", "move", "rest", "eat",
                          "predation", "starvation", "impacts"]
        # Fliketiketterna beror på läget: i inference matas nu ögonblicks-
        # värden per tick till alla serier (biomass/energy/move/rest/eat +
        # predation/starvation/impacts), så "avg"/"share of total" är
        # missvisande där. I train pushas fortfarande medelvärden över
        # probe-rollouten (en punkt per ARS-step), så där behåller vi
        # medelvärdes-formuleringen.
        if self.mode == "train":
            self._tab_labels = {
                "reward": "reward",
                "biomass": "avg biomass (% of start)",
                "energy": "avg energy (% of start)",
                "move": "avg move action (%)",
                "rest": "avg rest action (%)",
                "eat": "avg eat action (%)",
                "predation": "predation share of total loss (%)",
                "starvation": "starvation share of total loss (%)",
                "impacts": "impact share of total loss (%)",
            }
        else:
            self._tab_labels = {
                "reward": "reward",
                "biomass": "biomass (% of start)",
                "energy": "energy (% of start)",
                "move": "move action (%)",
                "rest": "rest action (%)",
                "eat": "eat action (%)",
                "predation": "predation share of tick loss (%)",
                "starvation": "starvation share of tick loss (%)",
                "impacts": "impact share of tick loss (%)",
            }
        self._active_tab = 0
        # Per-FG enable flag for plot panel (checkbox state). Toggled via
        # legend click; applies globally across all plot tabs.
        # Default: alla huvud-FG-serier ikryssade, alla ``_rnd``-baseline-
        # serier okryssade vid uppstart. Användaren kan enkelt aktivera
        # dem via legend-klick när de vill jämföra mot random-action-
        # baselinen, utan att grafen blir plottrig från start.
        self._plot_enabled: Dict[str, bool] = {}
        for fid in self.fg_ids:
            self._plot_enabled[fid] = True
        for eid in self._extra_plot_ids:
            self._plot_enabled[eid] = not str(eid).endswith("_rnd")
        self._legend_rects: list = []
        _all_series_ids = list(self.fg_ids) + [
            eid for eid in self._extra_plot_ids if eid not in self.fg_ids
        ]
        # I inference-läge sparas hela rollouten (obegränsad deque) så
        # att scrollbaren under plot-arean kan panorera tillbaka till
        # tick 0 utan att tidiga punkter tappats. I träningsläget
        # behålls ``reward_window`` som maxlen — där pushas en punkt
        # per ARS-step och rolling-fönster undviker minnesläckor över
        # långa körningar.
        _series_maxlen = (None
                          if str(self.mode).lower().startswith("infer")
                          else self.reward_window)
        self._series: Dict[str, Dict[str, deque]] = {
            tab: {fid: deque(maxlen=_series_maxlen) for fid in _all_series_ids}
            for tab in self._tabs
        }
        # Back-compat alias: legacy callers (and internal code) treat the
        # ``reward`` tab buffer as the historical _reward_buf. In inference
        # mode the first tab is biomass, so update_reward writes there.
        self._reward_buf: Dict[str, deque] = self._series[self._tabs[0]]
        # Click hit-boxes for tab headers, recomputed each frame.
        self._tab_rects: list = []
        # Latest biomass arrays + totals for heatmap rendering.
        self._biomass: Dict[str, np.ndarray] = {}
        self._totals: Dict[str, float] = {fid: 0.0 for fid in self.fg_ids}
        # Initial (rollout start) totals per FG; captured on tick==0 so that
        # the title can display 'B0=... B=...' for context. Reset each new
        # rollout (train probe runs fresh per ARS-iter, inference is one run).
        self._b0: Dict[str, float] = {fid: 0.0 for fid in self.fg_ids}
        # Per-FG biomassaförlustfraktioner (predation/starvation/impact)
        # som visas som en fjärde textrad ovanför heatmapen, på formatet
        # ``pr/st/im=X/Y/Z%``. Uppdateras via :meth:`update_loss_breakdown`
        # från probe-rollouten i train.py och från inference.py per tick.
        # Varje värde är en fraktion i [0, 1]; saknas FG → ingen rad ritas.
        self._loss_breakdown: Dict[str, Dict[str, float]] = {}
        # Per-FG dietuppdelning (predator-DM → {prey_id: andel}) som visas
        # som en femte textrad ovanför heatmapen, på formatet
        # ``<abbr1>/<abbr2>/… = X/Y/…%``. Uppdateras via
        # :meth:`update_diet_breakdown` från probe-rollouten i train.py och
        # från inference.py per tick. Värden är fraktioner i [0, 1] som
        # summerar till 1 när någon konsumtion skett (annars saknas FG).
        self._diet_breakdown: Dict[str, Dict[str, float]] = {}
        # Per-FG senaste move/rest/eat-fraktioner (procent 0..100) som
        # heatmap-headern visar på raden ``mv/rs/et = X/Y/Z%``. Detta är
        # separerat från ``_series['move'|'rest'|'eat']`` så att plot-
        # flikarna kan matas med en annan x-skala (t.ex. en gång per probe
        # med global ARS-step) medan headern fortfarande uppdateras per
        # tick under rollouten. Uppdateras via ``update_action_fracs``.
        self._action_fracs: Dict[str, Dict[str, float]] = {}
        # Per-FG startbiomass-slider (issue: b0-slider per heatmap). Tre dicts:
        # ``_b0_defaults`` är det gridskaleberäknade b0 (= inference_initial_biomass
        # * biomass_scale, med 1-ton-floor) som härleds från projektfilen och
        # sätts av train.py/inference.py via :meth:`set_b0_defaults`. Värdet
        # används som referens: slider-range = [0, 4 * default]; sliderns
        # mittposition motsvarar default. ``_b0_overrides[fid]`` är det
        # aktuella, av användaren satta värdet (None = använd default).
        # ``_b0_dirty`` sätts varje gång slidern släpps på ett nytt värde
        # och konsumeras av ``consume_b0_change`` så inference.py vet att
        # en ny inspelning ska startas.
        self._b0_defaults: Dict[str, float] = {}
        self._b0_overrides: Dict[str, float] = {}
        self._b0_dirty: bool = False
        # Slider-rektanglar registreras varje frame i ``_draw_one_heatmap``
        # och konsumeras av mus-handlerna (click/motion/up) i ``pump_events``.
        # Varje entry är (track_rect, fid).
        self._slider_rects: list = []
        self._dragging_slider: Optional[str] = None
        # Rollout-längd-slider (ticks). ``_ticks_default`` är värdet från
        # CLI (--ticks i inference / --n_eval_ticks i train), satt av
        # ``set_ticks_default``. ``_ticks_override`` är användarens valda
        # värde via slidern (None = default). ``_ticks_dirty`` sätts när
        # slidern släpps på ett nytt värde; konsumeras av inference.py.
        # Range = [3, 10000] med log-skala mappning för bättre kontroll.
        # ``_ticks_track_rect`` är (x, y, w, h) för den enda track-rect:en,
        # registrerad varje frame av ``_draw_status_bar``.
        self._ticks_min: int = 3
        self._ticks_max: int = 10000
        self._ticks_default: int = 200
        self._ticks_override: Optional[int] = None
        self._ticks_dirty: bool = False
        self._ticks_track_rect: Optional[tuple] = None
        self._dragging_ticks: bool = False
        # Lås för ticks-slidern. Default låst — användaren måste klicka
        # på låsikonen för att kunna ändra värdet. Rect lagras vid varje
        # frame för hit-test.
        self._ticks_locked: bool = True
        self._ticks_lock_rect: Optional[tuple] = None
        # Separat slider för ARS-träningens ``n_eval_ticks`` (rollout-
        # längd per delta-evaluering). Endast meningsfull i träningsläget,
        # men staten finns alltid så API:t kan anropas oberoende av mode.
        # Range = [3, 10000] log-skala, samma som ticks-slidern. Värdet
        # konsumeras av train.py **mellan** ARS-iterationer (säkert),
        # inte mitt i en iteration.
        self._neval_min: int = 3
        self._neval_max: int = 10000
        self._neval_default: int = 15
        self._neval_override: Optional[int] = None
        self._neval_dirty: bool = False
        self._neval_track_rect: Optional[tuple] = None
        self._dragging_neval: bool = False
        # Lås för n_eval_ticks-slidern (default låst).
        self._neval_locked: bool = True
        self._neval_lock_rect: Optional[tuple] = None
        # Uppspelnings-scrub-slider (timeline). Placeras i playback-baren
        # mellan speed-knapparna och frame-indikatorn. ``_playback_track_rect``
        # sätts varje frame av ``_draw_playback_bar`` (None när slidern
        # är disabled). ``_dragging_playback`` följer drag-state.
        self._playback_track_rect: Optional[tuple] = None
        self._dragging_playback: bool = False
        # Per-FG spawn-strategi-overrides (issue: dropdowns under varje
        # heatmap). ``_spawn_templates`` är hela det globala
        # ``project_data['spawn_templates']``-trädet:
        # ``{<mode>: {<name>: {<params>}}}``. ``_spawn_modes`` är
        # listan över tillgängliga modes (samma som fgconfig).
        # ``_spawn_overrides[fid]`` = {"mode": <mode_str>, "template": <name_or_None>}.
        # ``template=None`` innebär att FG:n behåller projektfilens
        # spawn-konfiguration (default-läget) — då ignoreras även
        # ``mode``-fältet. ``_spawn_dirty`` sätts vid val och konsumeras
        # av inference.py mellan rollouts (samma mönster som b0/ticks).
        self._spawn_templates: Dict[str, Dict[str, dict]] = {}
        self._spawn_modes: tuple = (
            "uniform", "perlin", "colony", "env_driven")
        self._spawn_mode_labels: Dict[str, str] = {
            "uniform": "Uniform", "perlin": "Perlin",
            "colony": "Colony", "env_driven": "Env-driven",
        }
        self._spawn_overrides: Dict[str, Dict[str, Optional[str]]] = {}
        # Per-FG default-mode från projektfilens spawn-block. Visas i
        # Mode-dropdownen när användaren inte aktivt valt något annat
        # (dvs ingen entry i ``_spawn_overrides`` för fid). När
        # användaren aktivt väljer "(default)" registreras fid i
        # ``_spawn_explicit_default`` så vi kan rita "(default)"
        # istället för default-modets namn.
        self._spawn_defaults: Dict[str, str] = {}
        self._spawn_explicit_default: set = set()
        self._spawn_dirty: bool = False
        # Hit-rects per FG, registrerade varje frame av _draw_one_heatmap:
        # (rect, fid, which) där which='mode'|'template'.
        self._spawn_dropdown_rects: list = []
        # Aktiv popup: (fid, which, options, rect_anchor) eller None.
        self._spawn_popup: Optional[dict] = None
        self._spawn_popup_item_rects: list = []
        # Stable colour per FG (used for the plot legend).
        self._fg_colour: Dict[str, tuple] = {
            fid: _fg_colour_for(i, len(self.fg_ids))
            for i, fid in enumerate(self.fg_ids)
        }
        # ``_rnd``-baseline series inherit the base FG colour but dimmed so
        # they're visually distinguishable from the trained-policy curve.
        for eid in self._extra_plot_ids:
            base = eid[:-4] if eid.endswith("_rnd") else None
            if base and base in self._fg_colour:
                r, g, b = self._fg_colour[base]
                self._fg_colour[eid] = (max(0, r - 90), max(0, g - 90), max(0, b - 90))
            else:
                self._fg_colour[eid] = _fg_colour_for(
                    len(self._fg_colour), len(self._fg_colour) + 1)

        # State.
        self._paused = False
        self._log_heatmap = False
        self._log_plot = False
        self._solo: Optional[str] = None
        self._quit = False
        self._last_frame_ts = 0.0
        self._status: Dict[str, object] = {}
        self._tick = 0
        self._fps_clock_t = time.monotonic()
        self._fps_frames = 0
        self._fps_value = 0.0

        # ---- Layout -------------------------------------------------------
        # Grid of heatmaps: choose columns so cells stay readable. With 8 FGs
        # we get 4 cols x 2 rows; with 1 FG, 1x1; etc.
        n = len(self.fg_ids)
        self._cols = min(4, n)
        self._rows = (n + self._cols - 1) // self._cols
        hm_w = self.grid_w * self.cell_px
        hm_h = self.grid_h * self.cell_px
        # Five info lines above the heatmap: FG id, 'B0 = … B = …',
        # 'mv/rs/et = …' (DM only), 'pr/st/im = …' (DM only) and the
        # diet breakdown '<abbr>/<abbr>/… = X/Y/…%' (DM only), followed
        # av en horisontell b0-slider mellan info-blocket och heatmapen.
        # NDM panels leave the three action/loss/diet lines blank to keep
        # heatmap origin aligned across panels.
        # title_h = 5 textrader (~14 px var) + slider (~22 px) + luft.
        title_h = 104
        self._slider_h = 16
        self._slider_margin = 4
        cbar_h = 16  # colorbar strip (gradient + 0/max labels)
        # Två extra rader under colorbaren: "Mode: …" och "Tpl: …".
        # Varje rad ~14 px text + 2 px padding ⇒ 32 px för båda.
        spawn_dd_h = 34
        self._spawn_dd_h = spawn_dd_h
        pad = 8
        self._cbar_h = cbar_h
        self._panel_w = hm_w + 2 * pad
        self._panel_h = hm_h + title_h + cbar_h + spawn_dd_h + 2 * pad
        heatmap_block_w = self._cols * self._panel_w
        heatmap_block_h = self._rows * self._panel_h
        # Reserve enough horizontal room in the plot panel for the legend:
        # roughly 7 px per character of the longest FG id, plus the swatch
        # and gutter. This avoids the legend overflowing the window with
        # long names like ``pelagic_fish_rnd``.
        all_legend_ids = list(self.plot_fg_ids) or list(self.fg_ids)
        max_name_len = max((len(self._display_name(fid))
                            for fid in all_legend_ids), default=8)
        self._legend_w = 38 + 7 * max_name_len + 8  # checkbox + swatch + text + pad
        # Bredd för tab-strippen: approximera font-bredden till ~7 px per
        # tecken och lägg på 10 px padding + 4 px gutter per flik, plus
        # en startmarginal. Denna bredd måste rymmas inom plot-panelens
        # legend-fria område, annars wrappar/försvinner flikar. Vi
        # garanterar därför att plot_w är minst så stor att hela
        # tab-strippen (+ legend + några pixlar marginal) ryms på en rad.
        tab_strip_w = 6 + sum(7 * len(t) + 10 + 4 for t in self._tabs) + 8
        plot_w = max(420, heatmap_block_w // 2 + self._legend_w,
                     tab_strip_w + self._legend_w + 4)
        plot_h = heatmap_block_h
        # Status-baren rymmer textraden, rollout-längd-slidern och
        # n_eval_ticks-slidern. ~18 px per slider-rad + 22 px för
        # textraden ⇒ 62 px räcker för alla tre. I inference-läget visas
        # ingen n_eval_ticks-slider, så raden tas bort (44 px räcker).
        if str(mode).lower().startswith("infer"):
            status_h = 44
        else:
            status_h = 62
        log_h = 0
        # Playback bar (under heatmaps) for replaying the most recent
        # probe/inference rollout. Holds the buttons << / >> / play / pause
        # / speed +/- and a frame indicator. Allokeras alltid (även när
        # ingen film ännu finns) så fönsterstorleken är konstant.
        playback_h = 32
        self._playback_h = playback_h
        self._win_w = heatmap_block_w + plot_w + pad
        self._win_h = status_h + heatmap_block_h + playback_h + pad + log_h
        self._heatmap_origin = (0, status_h)
        self._plot_rect = (heatmap_block_w + pad // 2,
                           status_h,
                           plot_w - pad // 2,
                           plot_h)
        self._status_rect = (0, 0, self._win_w, status_h)
        self._playback_rect = (0, status_h + heatmap_block_h,
                               heatmap_block_w, playback_h)
        # ---- Playback state ----------------------------------------------
        # Den senast färdigställda rollouten lagras som en lista av frame-
        # snapshots (dicts med biomass/totals/b0/loss_breakdown/diet_breakdown/
        # action_fracs/tick/status). Under live-träningens probe-rollout
        # ackumuleras frames i ``_pending_rollout``; när ``end_rollout_recording``
        # kallas flyttas listan över till ``_current_rollout`` och knapparna
        # blinkar tills användaren klickar (eller en ny film tar över).
        self._recording = False
        self._pending_rollout: list = []
        self._current_rollout: list = []
        self._playback_mode = "live"  # "live" | "paused" | "playing"
        self._playback_idx = 0
        self._playback_fps = 10.0  # justerbar via knappar (1..60)
        self._playback_last_advance = 0.0
        self._playback_blink = False
        self._playback_blink_until = 0.0
        self._playback_rects: list = []  # [(rect, action_name)]

        # ---- Plot horizontal scroll --------------------------------------
        # I inference-läget: när rolloutens x-axel är längre än ett fast
        # fönster (``_plot_window_width`` ticks) ritas en horisontell
        # scrollbar under plot-arean; ``_plot_scroll_offset`` är vänsterkant
        # (i tick-koordinater) för det synliga fönstret. ``None`` = auto
        # (följ senaste data-änden, dvs. samma beteende som tidigare).
        self._plot_window_width: int = 100
        self._plot_scroll_offset: Optional[float] = None
        self._plot_scroll_track_rect: Optional[tuple] = None
        self._plot_scroll_thumb_rect: Optional[tuple] = None
        self._dragging_plot_scroll: bool = False
        self._plot_scroll_drag_dx: float = 0.0
        # "Save plot HTML"-knapp under plot-arean (endast inference).
        # Sätts av ``_draw_plot`` varje frame; klick-test i ``_handle_click``
        # ropar ``_save_inference_plot_html`` som öppnar en Tk-fildialog och
        # skriver samma format som train-lägets ``plots.html``.
        self._save_plot_button_rect: Optional[tuple] = None
        self._save_plot_button_flash_until: float = 0.0
        self._save_plot_button_flash_msg: str = ""
        # Default-katalog för "Save plot HTML"-dialogen. Sätts av
        # ``set_save_dir`` (t.ex. från inference.py med ``args.checkpoints``
        # = ``results/<run-name>/``) så Tk-fildialogen startar där.
        self._save_dir: Optional[str] = None

        # ---- Init pygame --------------------------------------------------
        try:
            # MEMORY LEAK FIX: undvik full ``pygame.init()`` som ocksa drar
            # igang SDL audio-subsystemet (mixer + PulseAudio-mainloop i
            # egen trad). Memray summary visade ~201 MB lackt minne via
            # ``pa_mainloop_iterate`` / ``pa_mainloop_run`` / ``thread`` ->
            # ``internal_thread_func`` -> ``start_thread`` (565 trad-
            # skapelser under sessionen). Vi anvander aldrig ljud, sa
            # initiera bara display + font.
            pygame.display.init()
            pygame.font.init()
            # MEMORY LEAK FIX: tvinga ren software-surface i stallet for
            # SDL2:s default GL-backed window. Memray summary visade att
            # ~230 MB lackt minne kom fran pg_flip -> SDL_UpdateWindowSurface
            # -> SDL_UpdateWindowTexture -> GL_UpdateTexture (278 flips,
            # bara 108 GL_UpdateTexture-anrop returnerar minnet). Med
            # SWSURFACE-flaggan gar flip via SDL_UpdateRects istallet och
            # undviker hela GL-texture-pathen som lacker pa manga drivers.
            self._screen = pygame.display.set_mode(
                (self._win_w, self._win_h), pygame.SWSURFACE)
            pygame.display.set_caption(
                title or f"Mareld viz [{mode}]")
            self._font = pygame.font.SysFont("dejavusansmono,monospace", 12)
            self._font_big = pygame.font.SysFont("dejavusansmono,monospace", 14, bold=True)
        except Exception as e:
            print(f"[viz] pygame init failed ({e!r}); --visual disabled.",
                  file=sys.stderr)
            self.enabled = False
            return

        # Pre-build LUT.
        self._lut = _viridis_lut()
        # Initial paint.
        self._render_full()


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_biomass(
        self,
        fgs: Mapping[str, object],
        tick: int,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Record latest per-FG biomass arrays.

        ``fgs`` maps fg_id -> object with a ``.biomass`` ndarray attribute
        (matches the FG objects used in :mod:`lib.environments.ecosystem`).
        Also accepts raw ndarrays for testing.
        """
        if not self.enabled:
            return
        try:
            new_rollout = (int(tick) == 0)
            self._tick = int(tick)
            for fid in self.fg_ids:
                if fid not in fgs:
                    continue
                obj = fgs[fid]
                arr = obj.biomass if hasattr(obj, "biomass") else obj
                arr = np.asarray(arr, dtype=np.float32)
                # När en FG-cell kollapsat lämnar upprepade decay-multiplikationer
                # kvar pyttesmå rester (ofta float32-subnormaler ~1e-45, men
                # även "normala" mikrovärden långt under realistisk biomass).
                # Heatmapens per-frame max-normalisering förstoras då upp till
                # full skala och fluktuerar visuellt brusigt mellan tickar
                # trots att cellen i praktiken är död. Vi nollar därför **per
                # cell** så snart biomassan understiger en tröskel kopplad till
                # FG:ns minsta odelbara enhet (``min_split_biomass`` i ton;
                # halva den nivån = mindre än en halv odelbar individ kvar i
                # cellen). Saknas ``min_split_biomass`` (kontinuerligt läge)
                # använder vi en liten numerisk tröskel som bara fångar
                # subnormaler/brus.
                msb = float(getattr(obj, 'min_split_biomass', 0.0)) \
                    if hasattr(obj, 'biomass') else 0.0
                dead_eps = 0.5 * msb if msb > 0.0 else 1e-20
                if dead_eps > 0.0:
                    # Kopiera först för att inte mutera env:s underliggande
                    # array; nolla sedan alla celler under tröskeln.
                    arr = np.where(arr <= dead_eps,
                                   np.float32(0.0), arr).astype(np.float32,
                                                                copy=False)
                total = float(arr.sum())
                self._biomass[fid] = arr
                self._totals[fid] = total
                if new_rollout:
                    self._b0[fid] = total
            if extra:
                self._status.update(extra)
            # Spela in en frame om vi är mitt i en rollout-inspelning.
            # Capture tas EFTER att all state uppdaterats men FÖRE
            # _maybe_render så snapshotten reflekterar samma frame som
            # live-renderingen visar.
            if self._recording:
                try:
                    self._pending_rollout.append(self._capture_frame())
                    # Säkerhetscap mot oavsiktligt obegränsade rollouts.
                    # Capen är primärt en säkerhetsventil mot oavsiktligt
                    # obegränsade rollouts — den verkliga läckan (gamla
                    # filmer som hängde kvar mellan probes) åtgärdas i
                    # ``end_rollout_recording`` via explicit frigöring +
                    # ``gc.collect()``. 10000 frames ≈ 1.1 GB per film på
                    # 8 FG / 60×60 vilket är acceptabelt så länge bara en
                    # film lever åt gången.
                    if len(self._pending_rollout) > 10000:
                        # In-place decimering (var 2:a frame) så att den
                        # gamla listan kan frigöras direkt istället för att
                        # leva kvar parallellt med en ny kopia.
                        del self._pending_rollout[1::2]
                except Exception as e:
                    self._log_once(f"frame capture failed: {e!r}")
            self._maybe_render()
        except Exception as e:
            self._log_once(f"update_biomass failed: {e!r}")

    def update_reward(self, fg_id: str, value: float, step: int) -> None:
        # Back-compat shim: writes into the first tab's buffer (which is
        # ``reward`` in train mode and ``biomass`` in inference mode).
        self.update_series(self._tabs[0] if self.enabled else "reward",
                           fg_id, value, step)

    def update_series(self, tab: str, fg_id: str, value: float, step: int) -> None:
        """Append ``(step, value)`` to the named tab's buffer for ``fg_id``.

        Unknown tab names are silently ignored so callers can push data for
        tabs that may or may not exist in the current mode (e.g. ``reward``
        only exists in train mode).
        """
        if not self.enabled:
            return
        try:
            buf = self._series.get(tab)
            if buf is None or fg_id not in buf:
                return
            buf[fg_id].append((int(step), float(value)))
        except Exception as e:
            self._log_once(f"update_series failed: {e!r}")

    def update_loss_breakdown(
        self,
        breakdown: Mapping[str, Mapping[str, float]],
    ) -> None:
        """Record per-FG biomass-loss fractions (predation/starvation/impact).

        ``breakdown`` maps fg_id -> {'predation': p, 'starvation': s,
        'impact': i} where each value is a fraction in [0, 1] summing to
        1.0 when there was any loss at all (else all zero). Unknown keys
        are tolerated. Used to draw a ``pr/st/im=X/Y/Z%`` line above each
        heatmap.
        """
        if not self.enabled:
            return
        try:
            for fid, lb in breakdown.items():
                if not isinstance(lb, Mapping):
                    continue
                self._loss_breakdown[fid] = {
                    'predation':  float(lb.get('predation', 0.0)),
                    'starvation': float(lb.get('starvation', 0.0)),
                    'impact':     float(lb.get('impact', 0.0)),
                }
        except Exception as e:
            self._log_once(f"update_loss_breakdown failed: {e!r}")

    def update_diet_breakdown(
        self,
        breakdown: Mapping[str, Mapping[str, float]],
    ) -> None:
        """Record per-DM diet fractions (which prey FG was eaten how much).

        ``breakdown`` maps predator_fg_id -> {prey_fg_id: fraction in [0,1]}.
        Fractions are normalised over the predator's prey set so the menu
        in the rendered ``<abbr>/… = X/…%`` line sums to ~100% when any
        intake has occurred. Predators with no recorded intake are simply
        omitted (no diet line is drawn for them).
        """
        if not self.enabled:
            return
        try:
            for pred_id, prey_map in breakdown.items():
                if not isinstance(prey_map, Mapping):
                    continue
                self._diet_breakdown[pred_id] = {
                    str(k): float(v) for k, v in prey_map.items()
                }
        except Exception as e:
            self._log_once(f"update_diet_breakdown failed: {e!r}")

    def update_action_fracs(
        self,
        fid: str,
        move: float,
        rest: float,
        eat: float,
    ) -> None:
        """Record latest per-FG action fractions (procent 0..100) för
        heatmap-headerns ``mv/rs/et``-rad. Frikopplat från plot-serierna
        så att flikarna ``move``/``rest``/``eat`` kan matas med en annan
        x-skala (t.ex. en gång per probe med ``viz_step``) utan att
        headern slutar uppdateras per tick.
        """
        if not self.enabled:
            return
        try:
            self._action_fracs[str(fid)] = {
                'move': float(move),
                'rest': float(rest),
                'eat':  float(eat),
            }
        except Exception as e:
            self._log_once(f"update_action_fracs failed: {e!r}")

    def update_status(self, **kw) -> None:
        if not self.enabled:
            return
        self._status.update(kw)

    # ---- b0-slider API -----------------------------------------------
    def set_b0_defaults(self, defaults: Mapping[str, float]) -> None:
        """Registrera per-FG gridskaleberäknad startbiomass (b0_default).

        Värdet kommer från projektfilens ``inference_initial_biomass``
        (under fliken Inference i fgconfig) multiplicerat med
        ``biomass_scale = (H*W)/(ref_H*ref_W)`` och med 1-ton-floor — se
        ``lib/config/config_loader.py``. Slidern ritas med range
        ``[0, 4 * default]`` och mittposition motsvarar default.
        FGs utan default (eller default <= 0) får en disabled placeholder.
        """
        if not self.enabled:
            return
        for fid, v in dict(defaults).items():
            try:
                self._b0_defaults[str(fid)] = float(v)
            except (TypeError, ValueError):
                continue

    def get_b0_override(self, fid: str) -> Optional[float]:
        """Returnera användarens valda b0 för FG, eller None om slidern
        står på default-positionen (ingen explicit override)."""
        return self._b0_overrides.get(str(fid))

    def get_b0_overrides(self) -> Dict[str, float]:
        """Returnera en kopia av alla aktiva b0-overrides."""
        return dict(self._b0_overrides)

    def consume_b0_change(self) -> bool:
        """Returnera True och nollställ dirty-flaggan om slidern dragits
        sedan senaste anrop. Används av ``inference.py`` för att veta att
        en ny inspelning ska startas."""
        if not self.enabled:
            return False
        dirty = bool(self._b0_dirty)
        self._b0_dirty = False
        return dirty

    # ---- ticks-slider API (rollout length) ----------------------------
    # ------------------------------------------------------------------
    # Spawn-strategy overrides (per-FG dropdowns under each heatmap)
    # ------------------------------------------------------------------
    def set_spawn_templates(self, templates: Mapping[str, Mapping[str, dict]]) -> None:
        """Install the global ``spawn_templates`` tree from the project file.

        Structure mirrors what fgconfig writes::

            {<mode>: {<name>: {<params>}}}

        Used to populate the per-FG "Tpl" dropdown under each heatmap.
        Modes with no templates still appear in the Mode dropdown but
        the Tpl dropdown will be empty (only "(default)" available).
        Safe to call before or after `__init__`; an empty mapping
        clears all stored templates.
        """
        if not isinstance(templates, Mapping):
            self._spawn_templates = {}
            return
        out: Dict[str, Dict[str, dict]] = {}
        for mode, by_name in templates.items():
            if not isinstance(by_name, Mapping):
                continue
            mode_key = str(mode)
            cleaned: Dict[str, dict] = {}
            for name, params in by_name.items():
                if isinstance(params, Mapping):
                    cleaned[str(name)] = dict(params)
            if cleaned:
                out[mode_key] = cleaned
        self._spawn_templates = out

    def set_spawn_defaults(self, defaults: Mapping[str, str]) -> None:
        """Registrera varje FG:s default-spawn-mode från projektfilen.

        ``defaults`` är ``{fid: <mode_str>}``. Modet visas i Mode-
        dropdownen så länge användaren inte aktivt valt något annat
        (dvs ingen entry i ``_spawn_overrides``). Användaren kan
        fortfarande aktivt välja "(default)" — då visas "(default)"
        som etikett och projektfilens spawn-block används (samma
        beteende som tidigare).
        """
        if not isinstance(defaults, Mapping):
            self._spawn_defaults = {}
            return
        out: Dict[str, str] = {}
        for fid, mode in defaults.items():
            if mode is None:
                continue
            m = str(mode)
            if m in self._spawn_modes:
                out[str(fid)] = m
        self._spawn_defaults = out

    def get_spawn_overrides(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Return a copy of per-FG spawn-strategy overrides.

        Entry ``{"mode": <m>, "template": <name>}`` means: respawn FG
        ``fid`` using the template ``<name>`` stored under ``<m>``.
        FGs absent from the dict (or with ``template`` == None) keep
        the project file's spawn configuration.
        """
        return {fid: dict(ov) for fid, ov in self._spawn_overrides.items()
                if ov.get("template")}

    def consume_spawn_change(self) -> bool:
        """Return True iff a spawn dropdown was changed since the last call."""
        d = bool(self._spawn_dirty)
        self._spawn_dirty = False
        return d

    def set_reward_label(self, text: Optional[str]) -> None:
        """Sätt reward-flikens y-axel-/rubriktext (t.ex. "ARS reward per
        FG — total-energi (integral)"). Anropas typiskt av train.py
        efter LiveVisualizer-init så plot-panelens rubrik speglar den
        aktiva rewardformeln istället för det generiska "reward"."""
        if not self.enabled:
            return
        try:
            if text:
                self._tab_labels["reward"] = str(text)
        except Exception:
            pass

    def set_save_dir(self, path: Optional[str]) -> None:
        """Registrera default-katalog för "Save plot HTML"-dialogen.

        Anropas typiskt från ``inference.py`` med ``args.checkpoints``
        (``results/<run-name>/``) så att Tk-fildialogen öppnas i den
        mapp som användaren angav med ``--run-name``. ``None`` eller
        icke-existerande katalog ignoreras och Tk faller tillbaka till
        sin egen default (CWD)."""
        if not self.enabled:
            return
        self._save_dir = str(path) if path else None

    def set_ticks_default(self, ticks: int) -> None:
        """Registrera CLI-värdet (default) för rollout-längd.

        Klampas till ``[_ticks_min, _ticks_max]`` = [3, 10000]. Anropas
        en gång av train.py (``--n_eval_ticks``) och inference.py
        (``--ticks``) när viz initieras.
        """
        if not self.enabled:
            return
        try:
            v = int(ticks)
        except (TypeError, ValueError):
            return
        self._ticks_default = max(self._ticks_min,
                                  min(self._ticks_max, v))

    def get_ticks_override(self) -> Optional[int]:
        """Returnera användarens valda rollout-längd, eller None om
        slidern står på default-positionen."""
        return self._ticks_override

    def get_ticks(self) -> int:
        """Effektiv rollout-längd: override om satt, annars default."""
        if self._ticks_override is not None:
            return int(self._ticks_override)
        return int(self._ticks_default)

    def consume_ticks_change(self) -> bool:
        """Returnera True och nollställ dirty-flaggan om ticks-slidern
        dragits sedan senaste anrop. Används av ``inference.py`` för att
        veta att en ny inspelning ska startas."""
        if not self.enabled:
            return False
        dirty = bool(self._ticks_dirty)
        self._ticks_dirty = False
        return dirty

    def _ticks_pos_from_value(self, v: int) -> float:
        """Log-skala mappning value -> position-frac i [0, 1]."""
        lo, hi = float(self._ticks_min), float(self._ticks_max)
        v = max(lo, min(hi, float(v)))
        import math
        return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))

    def _ticks_value_from_pos(self, frac: float) -> int:
        """Log-skala mappning position-frac i [0, 1] -> value (int)."""
        frac = max(0.0, min(1.0, float(frac)))
        import math
        lo, hi = float(self._ticks_min), float(self._ticks_max)
        v = math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
        return int(round(max(lo, min(hi, v))))

    # ---- n_eval_ticks-slider API (ARS rollout length) ----------------
    def set_neval_ticks_default(self, ticks: int) -> None:
        """Registrera CLI-värdet (--n_eval_ticks) som default för
        ARS-rollout-längd. Anropas en gång av train.py när viz initieras."""
        if not self.enabled:
            return
        try:
            v = int(ticks)
        except (TypeError, ValueError):
            return
        self._neval_default = max(self._neval_min,
                                  min(self._neval_max, v))

    def get_neval_ticks_override(self) -> Optional[int]:
        """Returnera användarens valda n_eval_ticks, eller None."""
        return self._neval_override

    def get_neval_ticks(self) -> int:
        """Effektivt n_eval_ticks: override om satt, annars default."""
        if self._neval_override is not None:
            return int(self._neval_override)
        return int(self._neval_default)

    def consume_neval_ticks_change(self) -> bool:
        """Returnera True och nollställ dirty om slidern dragits sedan
        senaste anrop."""
        if not self.enabled:
            return False
        dirty = bool(self._neval_dirty)
        self._neval_dirty = False
        return dirty

    def _neval_pos_from_value(self, v: int) -> float:
        lo, hi = float(self._neval_min), float(self._neval_max)
        v = max(lo, min(hi, float(v)))
        import math
        return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))

    def _neval_value_from_pos(self, frac: float) -> int:
        frac = max(0.0, min(1.0, float(frac)))
        import math
        lo, hi = float(self._neval_min), float(self._neval_max)
        v = math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
        return int(round(max(lo, min(hi, v))))

    # ---- Rollout recording / playback --------------------------------
    def begin_rollout_recording(self) -> None:
        """Starta inspelning av en ny probe/inference-rollout.

        Frames samlas i ``_pending_rollout`` via :meth:`update_biomass`
        (en snapshot per anrop). Vid :meth:`end_rollout_recording`
        flyttas listan över till ``_current_rollout`` och uppspelnings-
        knapparna blir aktiva. Under inspelningen är knapparna låsta
        (gråa); klick ignoreras.
        """
        if not self.enabled:
            return
        self._recording = True
        self._pending_rollout = []
        # Rensa per-FG plot-serier mellan inspelningar — MEN endast i
        # inference-läge. I inference används ``step=tick`` (0..N) som
        # x-värde och varje ny rollout startar om från 0, så gamla par
        # från förra rolloutens slut + nya pars start gör att
        # ``pg.draw.aalines`` (utan clipping mot plot-rect) ritar
        # långa fel-linjer tvärs över skärmen in i heatmap-området.
        # I träningsläget pushas däremot en punkt per probe med
        # globalt monotont ökande ``viz_step`` (ARS-step) och hela
        # historiken över probes ska vara synlig i grafritarrutan —
        # där får vi INTE rensa, annars ritas ingenting (endast 1
        # punkt kvar per probe, och aalines kräver ≥2 punkter).
        if str(self.mode).lower().startswith("infer"):
            try:
                for tab_buf in self._series.values():
                    for fid in tab_buf:
                        tab_buf[fid].clear()
            except Exception as e:
                self._log_once(
                    f"begin_rollout_recording series-reset failed: {e!r}")
            # Nollställ plot-scroll så nya rolloutens fönster startar i
            # auto-läge (följer senaste data) tills användaren själv drar.
            self._plot_scroll_offset = None
            self._dragging_plot_scroll = False
        # MEMORY LEAK FIX (blind, riktad): rensa per-FG breakdown-dicts
        # mellan probes. update_loss_breakdown / update_diet_breakdown /
        # update_action_fracs gör ``dict.update`` med nya nycklar varje
        # tick utan att rensa gamla. Om FG-id varierar mellan generationer
        # (t.ex. spawn-varianter) ackumuleras nycklar. Dessutom innehåller
        # _diet_breakdown nästlade dicts (predator -> {prey: frac}) där
        # prey-set kan växa. Eftersom probe-rolloutsen är ENDA källan
        # till dessa data i train-läget är det säkert att nolla dem här.
        try:
            self._loss_breakdown.clear()
            self._diet_breakdown.clear()
            self._action_fracs.clear()
        except Exception as e:
            self._log_once(
                f"begin_rollout_recording breakdown-reset failed: {e!r}")

    def end_rollout_recording(self) -> None:
        """Avsluta inspelning; den nya filmen tar över ``_current_rollout``."""
        if not self.enabled:
            return
        self._recording = False
        if self._pending_rollout:
            # Frigör föregående films frame-listor EXPLICIT innan vi byter
            # in den nya. Utan detta kunde två filmer (gamla + nya) leva
            # parallellt en stund — och om replay-state höll en referens
            # till den gamla via t.ex. ``_playback_idx``-rendering kunde
            # CPython:s ref-cykel försena frigöringen ytterligare. Detta
            # var en huvudkomponent av den minnesläcka som rapporterades.
            try:
                old = self._current_rollout
                self._current_rollout = []
                if old:
                    old.clear()
                del old
            except Exception:
                pass
            self._current_rollout = self._pending_rollout
            self._pending_rollout = []
            # Tvinga generationell GC så att de gamla numpy-buffertarna
            # faktiskt återlämnas till allokatorn istället för att vänta
            # på nästa triggernivå. Billigt jämfört med en typisk probe.
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            # Blinka knapparna ett par sekunder så användaren ser att en
            # ny film är tillgänglig; auto-hoppa INTE in i replay-läget
            # (per användarens önskemål) — live-vyn fortsätter visas tills
            # användaren själv klickar play/step.
            try:
                self._playback_blink_until = time.monotonic() + 3.0
            except Exception:
                self._playback_blink_until = 0.0
            # Om användaren tidigare rört slidern under träningen befann
            # vi oss i replay-läge (paused/playing). Nya filmen ska då
            # visa rolloutsens SISTA steg (samma tillstånd som live-vyn
            # visar om slidern aldrig rörts), inte hoppa till frame 1.
            n_new = len(self._current_rollout)
            if self._playback_mode != "live":
                self._playback_idx = max(0, n_new - 1)
                self._playback_mode = "paused"
            else:
                self._playback_idx = 0

    def _capture_frame(self) -> dict:
        """Bygg en snapshot av all state som ``_draw_one_heatmap`` läser."""
        # Kopior är essentiella: live-state muteras efter capture.
        biomass = {fid: arr.copy() for fid, arr in self._biomass.items()}
        totals = dict(self._totals)
        b0 = dict(self._b0)
        loss = {fid: dict(v) for fid, v in self._loss_breakdown.items()}
        diet = {fid: dict(v) for fid, v in self._diet_breakdown.items()}
        actf = {fid: dict(v) for fid, v in self._action_fracs.items()}
        return {
            "biomass": biomass,
            "totals": totals,
            "b0": b0,
            "loss_breakdown": loss,
            "diet_breakdown": diet,
            "action_fracs": actf,
            "tick": int(self._tick),
            "status": dict(self._status),
        }

    def pump_events(self) -> bool:
        """Process pygame events; return False if user asked to quit viz.

        Must be called from the main thread (the one that created the
        SDL display). ``train.py`` wires this up via
        ``trainer.pump_callback`` so the queue is drained every ~50 ms
        while the worker pool runs, which is what stops the OS from
        marking the window as "not responding".

        We also call :meth:`_maybe_render` after draining the queue so
        that user actions taken mid-iteration (e.g. switching tabs by
        click or TAB key, toggling legend entries, pausing) become
        visible immediately rather than only after the current
        ``train_step`` returns. ``_maybe_render`` is frame-rate capped
        via ``_MIN_FRAME_INTERVAL`` so repeated pump calls do not
        starve the worker pool.
        """
        if not self.enabled:
            return True
        try:
            pg = self._pg
            # ALWAYS pump SDL first — this answers the WM ping and
            # transfers OS-level events into pygame's queue. Doing this
            # before ``event.get()`` minimises the window in which a
            # click can sit unseen in the OS layer.
            pg.event.pump()
            interacted = False
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self._quit = True
                elif event.type == pg.KEYDOWN:
                    self._handle_key(event.key)
                    interacted = True
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                    interacted = True
                elif event.type == pg.MOUSEMOTION and self._dragging_slider is not None:
                    fid = self._dragging_slider
                    v = self._slider_value_from_x(fid, event.pos[0])
                    if v is not None:
                        self._b0_overrides[fid] = float(v)
                        interacted = True
                elif event.type == pg.MOUSEMOTION and self._dragging_ticks:
                    v = self._ticks_value_from_x(event.pos[0])
                    if v is not None:
                        self._ticks_override = int(v)
                        interacted = True
                elif event.type == pg.MOUSEMOTION and self._dragging_neval:
                    v = self._neval_value_from_x(event.pos[0])
                    if v is not None:
                        self._neval_override = int(v)
                        interacted = True
                elif event.type == pg.MOUSEMOTION and self._dragging_playback:
                    idx = self._playback_idx_from_x(event.pos[0])
                    if idx is not None:
                        self._playback_idx = idx
                        self._playback_mode = "paused"
                        interacted = True
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1 \
                        and self._dragging_playback:
                    self._dragging_playback = False
                    interacted = True
                elif event.type == pg.MOUSEMOTION and self._dragging_plot_scroll:
                    new_off = self._plot_scroll_offset_from_x(event.pos[0])
                    if new_off is not None:
                        self._plot_scroll_offset = float(new_off)
                        interacted = True
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1 \
                        and self._dragging_plot_scroll:
                    self._dragging_plot_scroll = False
                    interacted = True
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1 \
                        and self._dragging_slider is not None:
                    # Släppt: markera dirty så inference.py kan trigga ny
                    # inspelning. För train.py konsumeras värdet av nästa
                    # probe utan att läsa flaggan.
                    self._dragging_slider = None
                    self._b0_dirty = True
                    interacted = True
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1 \
                        and self._dragging_ticks:
                    self._dragging_ticks = False
                    self._ticks_dirty = True
                    interacted = True
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1 \
                        and self._dragging_neval:
                    self._dragging_neval = False
                    self._neval_dirty = True
                    interacted = True
            # Only render on actual user interaction here — a full
            # heatmap+plot redraw can easily cost 50-150 ms and during
            # that time the main thread is blocked, which is the main
            # reason subsequent clicks felt "lost". The trainer pumps
            # us every ~50 ms via ``pump_callback``; the regular live
            # preview render happens between iterations from train.py.
            if interacted:
                self._last_frame_ts = 0.0
                self._render_full()
                # Pump again right after the redraw so any clicks the
                # user issued while we were drawing are picked up on
                # the very next pump tick instead of next iteration.
                pg.event.pump()
            # Auto-advance vid replay-play: stega frame när play-fps
            # tidsbudgeten passerats. Render-anrop sker via _render_full
            # direkt så användaren ser framgång även mellan ARS-steg.
            if (self._playback_mode == "playing"
                    and self._current_rollout
                    and not self._recording):
                now = time.monotonic()
                dt = max(1.0 / 60.0, 1.0 / max(1.0, self._playback_fps))
                if (now - self._playback_last_advance) >= dt:
                    self._playback_last_advance = now
                    if self._playback_idx + 1 < len(self._current_rollout):
                        self._playback_idx += 1
                    else:
                        # Nått slutet — pausa på sista framen.
                        self._playback_mode = "paused"
                    self._last_frame_ts = 0.0
                    self._render_full()
                    pg.event.pump()
            # Blinkning: rendera om ungefär 4 Hz medan blinkningen är
            # aktiv så användaren faktiskt ser knapparna pulsa.
            if (self._playback_blink_until > time.monotonic()
                    and not self._recording):
                if (time.monotonic() - self._last_frame_ts) > 0.25:
                    self._last_frame_ts = 0.0
                    self._render_full()
            # While paused, keep the window responsive without spinning.
            # Playback-knapparna (inkl. step/play/speed) ska fortsätta
            # fungera medan träningen är pausad, så vi gör samma
            # auto-advance + render som ovanför.
            while self._paused and not self._quit:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self._quit = True
                    elif event.type == pg.KEYDOWN:
                        self._handle_key(event.key)
                    elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)
                if (self._playback_mode == "playing"
                        and self._current_rollout
                        and not self._recording):
                    now = time.monotonic()
                    dt = max(1.0 / 60.0, 1.0 / max(1.0, self._playback_fps))
                    if (now - self._playback_last_advance) >= dt:
                        self._playback_last_advance = now
                        if self._playback_idx + 1 < len(self._current_rollout):
                            self._playback_idx += 1
                        else:
                            self._playback_mode = "paused"
                self._last_frame_ts = 0.0
                self._render_full()
                pg.time.wait(50)
            return not self._quit
        except Exception as e:
            self._log_once(f"pump_events failed: {e!r}")
            return True

    def wait_for_close(self, banner: Optional[str] = None) -> None:
        """Block until the user closes the window (or presses Q/ESC).

        Used at the end of train/inference --visual runs so the final
        frame stays on screen instead of the window disappearing
        immediately. Ctrl+C in the terminal still aborts (KeyboardInterrupt
        propagates out).
        """
        if not self.enabled or self._quit:
            return
        pg = self._pg
        try:
            # Stash a banner the status bar can pick up, if desired.
            self._final_banner = banner or "run finished — close window to exit (Q/ESC)"
            # Render one last full frame so the banner is visible.
            try:
                self._render_full()
            except Exception:
                pass
            while not self._quit:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self._quit = True
                    elif event.type == pg.KEYDOWN:
                        if event.key in (pg.K_q, pg.K_ESCAPE):
                            self._quit = True
                        else:
                            self._handle_key(event.key)
                    elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)
                        # Klick på en spawn-dropdown/popup-item kan ha
                        # satt ``_spawn_dirty`` — i så fall ska vi
                        # bryta wait-loopen så inference.py får spela
                        # in en ny rollout med den nya spawn-strategin.
                        if self._spawn_dirty:
                            return
                    elif (event.type == pg.MOUSEMOTION
                          and self._dragging_slider is not None):
                        fid = self._dragging_slider
                        v = self._slider_value_from_x(fid, event.pos[0])
                        if v is not None:
                            self._b0_overrides[fid] = float(v)
                    elif (event.type == pg.MOUSEMOTION
                          and self._dragging_ticks):
                        v = self._ticks_value_from_x(event.pos[0])
                        if v is not None:
                            self._ticks_override = int(v)
                    elif (event.type == pg.MOUSEMOTION
                          and self._dragging_neval):
                        v = self._neval_value_from_x(event.pos[0])
                        if v is not None:
                            self._neval_override = int(v)
                    elif (event.type == pg.MOUSEMOTION
                          and self._dragging_playback):
                        idx = self._playback_idx_from_x(event.pos[0])
                        if idx is not None:
                            self._playback_idx = idx
                            self._playback_mode = "paused"
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_playback):
                        self._dragging_playback = False
                        # Avbryt INTE wait-loopen — scrubbing är en
                        # ren visuell operation, ingen ny rollout behövs.
                    elif (event.type == pg.MOUSEMOTION
                          and self._dragging_plot_scroll):
                        new_off = self._plot_scroll_offset_from_x(event.pos[0])
                        if new_off is not None:
                            self._plot_scroll_offset = float(new_off)
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_plot_scroll):
                        self._dragging_plot_scroll = False
                        # Ren visuell operation; avbryt inte wait-loopen.
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_slider is not None):
                        self._dragging_slider = None
                        self._b0_dirty = True
                        # Avbryt wait_for_close-loopen så inference.py kan
                        # läsa dirty-flaggan och spela in en ny rollout.
                        return
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_ticks):
                        self._dragging_ticks = False
                        self._ticks_dirty = True
                        # Avbryt wait_for_close-loopen så inference.py
                        # spelar in en ny rollout med nya ticks-värdet.
                        return
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_neval):
                        self._dragging_neval = False
                        self._neval_dirty = True
                        # n_eval_ticks-slidern påverkar bara träning;
                        # i inference-läget triggar den ingen re-recording
                        # (vi avbryter inte wait-loopen för den här).
                        # Men vi avbryter ändå för konsistens — main()
                        # i inference.py reagerar bara om en relevant
                        # slider är dirty, så detta är ofarligt.
                        pass
                # Playback auto-advance (samma logik som i pump_events),
                # annars händer ingenting när användaren trycker play efter
                # att inference-rollouten är klar.
                if (self._playback_mode == "playing"
                        and self._current_rollout
                        and not self._recording):
                    now = time.monotonic()
                    dt = max(1.0 / 60.0, 1.0 / max(1.0, self._playback_fps))
                    if (now - self._playback_last_advance) >= dt:
                        self._playback_last_advance = now
                        if self._playback_idx + 1 < len(self._current_rollout):
                            self._playback_idx += 1
                        else:
                            self._playback_mode = "paused"
                self._last_frame_ts = 0.0
                try:
                    self._render_full()
                except Exception:
                    pass
                pg.time.wait(50)
        except KeyboardInterrupt:
            # Ctrl+C: let it propagate after we mark the window for closing.
            self._quit = True
            raise

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self._pg.display.quit()
            self._pg.quit()
        except Exception:
            pass
        self.enabled = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _display_name(fid: str) -> str:
        """Format an FG id as a human-readable label.

        Replaces underscores with spaces and capitalises the first letter,
        e.g. ``pelagic_fish`` -> ``Pelagic fish``.
        """
        if not fid:
            return fid
        s = str(fid).replace("_", " ").strip()
        return s[:1].upper() + s[1:] if s else s

    @staticmethod
    def _fmt_compact(v: float) -> str:
        """Compact human-readable number for colorbar max labels."""
        try:
            v = float(v)
        except Exception:
            return "?"
        if not np.isfinite(v):
            return "?"
        a = abs(v)
        if a >= 1e9:
            return f"{v/1e9:.2f}G"
        if a >= 1e6:
            return f"{v/1e6:.2f}M"
        if a >= 1e3:
            return f"{v/1e3:.1f}k"
        if a >= 1.0:
            return f"{v:.1f}"
        if a == 0.0:
            return "0"
        return f"{v:.2g}"

    _logged_errors: set = set()

    def _log_once(self, msg: str) -> None:
        if msg in self._logged_errors:
            return
        self._logged_errors.add(msg)
        print(f"[viz] {msg}", file=sys.stderr)

    def _handle_key(self, key: int) -> None:
        pg = self._pg
        if key in (pg.K_q, pg.K_ESCAPE):
            self._quit = True
        elif key == pg.K_SPACE:
            self._paused = not self._paused
        elif key == pg.K_l:
            self._log_heatmap = not self._log_heatmap
        elif key == pg.K_r:
            self._log_plot = not self._log_plot
        elif key == pg.K_TAB:
            mods = self._pg.key.get_mods()
            n = len(self._tabs)
            if mods & self._pg.KMOD_SHIFT:
                self._active_tab = (self._active_tab - 1) % n
            else:
                self._active_tab = (self._active_tab + 1) % n
        elif pg.K_1 <= key <= pg.K_9:
            idx = key - pg.K_1
            if idx < len(self.fg_ids):
                target = self.fg_ids[idx]
                self._solo = None if self._solo == target else target

    def _slider_value_from_x(self, fid: str, mx: int) -> Optional[float]:
        """Map mouse x-coordinate to a b0-värde för denna FG:s slider.

        Returnerar None om FG saknar default (slidern är disabled).
        """
        b0_default = self._b0_defaults.get(fid)
        if b0_default is None or b0_default <= 0.0:
            return None
        # Slider-tracken börjar vid panel_x + 4 (pad) och har bredden
        # grid_w * cell_px. Vi söker upp track-rektangeln via senaste
        # registrerade hit_rect så vi inte måste duplicera layout-mat.
        for rect, rfid in self._slider_rects:
            if rfid != fid:
                continue
            rx, _ry, rw, _rh = rect
            # Justera tillbaka för det 2-px padding vi la till runt hit_rect.
            track_x = rx + 2
            track_w = rw - 4
            frac = (mx - track_x) / max(1.0, float(track_w))
            frac = max(0.0, min(1.0, frac))
            return frac * 4.0 * float(b0_default)
        return None

    def _draw_lock_icon(self, x: int, y: int, size: int,
                        locked: bool) -> tuple:
        """Rita en liten lås-ikon med övre vänstra hörnet i (x, y).

        Returnerar hit-rect (x, y, size, size). Färgschema: gul/orange
        låsöra + ljus kropp om olåst, dämpat grått om låst.
        """
        pg = self._pg
        s = int(size)
        # Kropp (rektangel i nedre delen).
        body_h = int(s * 0.55)
        body_y = y + s - body_h
        body_x = x + 1
        body_w = s - 2
        # Bygel (ovan kroppen).
        sh_w = int(s * 0.6)
        sh_x = x + (s - sh_w) // 2
        sh_y = y + 1
        sh_h = int(s * 0.55)
        if locked:
            body_col = (160, 160, 170)
            shackle_col = (160, 160, 170)
        else:
            body_col = (230, 200, 90)
            shackle_col = (230, 200, 90)
        # Bygel: rita som en ofylld halvcirkel/rektangel-ring.
        pg.draw.rect(self._screen, shackle_col,
                     (sh_x, sh_y, sh_w, sh_h), 2,
                     border_radius=max(2, sh_w // 2))
        if not locked:
            # Olåst: bryt bygeln på höger sida genom att täcka med
            # bakgrundsfärg (status-bar bakgrund ≈ (20,20,28) i panelen).
            # Vi ritar ett litet avsnitt som "öppnar" bygeln.
            pg.draw.rect(self._screen, (28, 28, 36),
                         (sh_x + sh_w - 3, sh_y + sh_h // 2 - 1,
                          3, sh_h // 2 + 2))
        # Kropp.
        pg.draw.rect(self._screen, body_col,
                     (body_x, body_y, body_w, body_h),
                     border_radius=2)
        pg.draw.rect(self._screen, (30, 30, 36),
                     (body_x, body_y, body_w, body_h), 1,
                     border_radius=2)
        # Litet nyckelhål.
        kh_x = body_x + body_w // 2
        kh_y = body_y + body_h // 2
        pg.draw.circle(self._screen, (30, 30, 36), (kh_x, kh_y), 1)
        return (x, y, s, s)

    @staticmethod
    def _hit_rect(rect, pos) -> bool:
        if rect is None:
            return False
        try:
            mx, my = pos
        except Exception:
            return False
        rx, ry, rw, rh = rect
        return rx <= mx < rx + rw and ry <= my < ry + rh

    def _hit_slider(self, pos) -> Optional[str]:
        try:
            mx, my = pos
        except Exception:
            return None
        for rect, fid in self._slider_rects:
            rx, ry, rw, rh = rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                return fid
        return None

    def _hit_ticks_slider(self, pos) -> bool:
        """True om klicket landade i ticks-sliderns track-rect."""
        rect = self._ticks_track_rect
        if rect is None:
            return False
        try:
            mx, my = pos
        except Exception:
            return False
        rx, ry, rw, rh = rect
        return rx <= mx < rx + rw and ry <= my < ry + rh

    def _ticks_value_from_x(self, mx: int) -> Optional[int]:
        """Mappa mus-x till ticks-värde via tracken (log-skala)."""
        rect = self._ticks_track_rect
        if rect is None:
            return None
        rx, _ry, rw, _rh = rect
        track_x = rx + 2
        track_w = max(1, rw - 4)
        frac = (mx - track_x) / float(track_w)
        return self._ticks_value_from_pos(frac)

    def _hit_neval_slider(self, pos) -> bool:
        rect = self._neval_track_rect
        if rect is None:
            return False
        try:
            mx, my = pos
        except Exception:
            return False
        rx, ry, rw, rh = rect
        return rx <= mx < rx + rw and ry <= my < ry + rh

    def _neval_value_from_x(self, mx: int) -> Optional[int]:
        rect = self._neval_track_rect
        if rect is None:
            return None
        rx, _ry, rw, _rh = rect
        track_x = rx + 2
        track_w = max(1, rw - 4)
        frac = (mx - track_x) / float(track_w)
        return self._neval_value_from_pos(frac)

    def _handle_click(self, pos) -> None:
        try:
            mx, my = pos
        except Exception:
            return
        # Aktiv spawn-popup: klick inuti = val, klick utanför = stäng.
        if self._spawn_popup is not None:
            for rect, value in self._spawn_popup_item_rects:
                rx, ry, rw, rh = rect
                if rx <= mx < rx + rw and ry <= my < ry + rh:
                    self._apply_spawn_popup_selection(value)
                    return
            # Klick utanför stänger popup utan att välja.
            self._spawn_popup = None
            self._spawn_popup_item_rects = []
            return
        # Spawn-dropdown-rektanglar (Mode / Tpl per FG).
        for rect, fid, which in self._spawn_dropdown_rects:
            rx, ry, rw, rh = rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                self._open_spawn_popup(fid, which, rect)
                return
        # Lås-ikoner: togglar lås-state. Klick på låsikon ska aldrig
        # routas vidare till slidern eller heatmaps.
        if self._hit_rect(self._ticks_lock_rect, pos):
            self._ticks_locked = not self._ticks_locked
            return
        if self._hit_rect(self._neval_lock_rect, pos):
            self._neval_locked = not self._neval_locked
            return
        # Ticks-slider: kolla först av allt (sitter i status-baren ovanför
        # heatmapsen och får inte routas vidare som heatmap/tab-klick).
        # Slidern är aktiv endast om låset är upplåst.
        if self._hit_ticks_slider(pos):
            if self._ticks_locked:
                return
            v = self._ticks_value_from_x(mx)
            if v is not None:
                self._ticks_override = int(v)
                self._dragging_ticks = True
            return
        # n_eval_ticks-slider: samma princip (separat rad i status-baren).
        if self._hit_neval_slider(pos):
            if self._neval_locked:
                return
            v = self._neval_value_from_x(mx)
            if v is not None:
                self._neval_override = int(v)
                self._dragging_neval = True
            return
        # b0-slidrar: kolla först om klicket landade i en slider — då
        # initieras dragning och vi hoppar över övrig klick-routing.
        fid = self._hit_slider(pos)
        if fid is not None:
            v = self._slider_value_from_x(fid, mx)
            if v is not None:
                self._b0_overrides[fid] = float(v)
                self._dragging_slider = fid
                # Markera inte ``_b0_dirty`` förrän musen släpps; annars
                # skulle inference.py kunna trigga en ny inspelning för
                # varje liten mus-darrning under dragningen.
            return
        # Save-plot-HTML-knapp (inference): kollas före scrollbaren så
        # den fångas även om rect överlappar i pixel-marginalerna.
        if self._save_plot_button_rect is not None \
                and self._hit_rect(self._save_plot_button_rect, pos):
            try:
                self._save_inference_plot_html()
            except Exception as e:
                self._log_once(f"save_inference_plot_html failed: {e!r}")
                self._save_plot_button_flash_msg = f"Save failed: {e}"
                import time as _t
                self._save_plot_button_flash_until = _t.time() + 3.0
            return
        # Plot horisontell scrollbar: klick på thumb startar drag, klick
        # utanför thumb men i tracken paginerar ett fönster åt vänster/höger.
        hit_ps = self._hit_plot_scroll(pos)
        if hit_ps:
            thumb = self._plot_scroll_thumb_rect
            span = getattr(self, "_plot_scroll_span_cache", None)
            if hit_ps == "thumb" and thumb is not None:
                self._plot_scroll_drag_dx = float(mx - thumb[0])
                self._dragging_plot_scroll = True
            elif hit_ps == "track" and span is not None and thumb is not None:
                # Klick vid sidan om thumb: paginera med ~fönsterbredd.
                thumb_x = thumb[0]
                step_x = float(self._plot_window_width)
                data_xmin, data_xmax = span
                cur = (self._plot_scroll_offset
                       if self._plot_scroll_offset is not None
                       else float(data_xmax) - float(self._plot_window_width))
                if mx < thumb_x:
                    cur -= step_x
                else:
                    cur += step_x
                max_off = float(data_xmax) - float(self._plot_window_width)
                cur = max(float(data_xmin), min(max_off, cur))
                self._plot_scroll_offset = float(cur)
            return
        for rect, i in self._tab_rects:
            rx, ry, rw, rh = rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                self._active_tab = int(i)
                return
        # Legend checkbox toggles (global across tabs).
        for rect, fid in getattr(self, "_legend_rects", []):
            rx, ry, rw, rh = rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                self._plot_enabled[fid] = not self._plot_enabled.get(fid, True)
                return
        # Playback-knappar. ``toggle_train_pause`` är alltid aktiv;
        # övriga är gråade/inaktiva under pågående probe / utan film.
        disabled = self._recording or not self._current_rollout
        # Scrub-slider (timeline): klick + ev. drag-start. Endast aktiv
        # när film finns och recording inte pågår.
        if self._hit_playback_slider(pos) and not disabled:
            idx = self._playback_idx_from_x(mx)
            if idx is not None:
                self._playback_idx = idx
                self._playback_mode = "paused"
                self._playback_blink_until = 0.0
                self._dragging_playback = True
            return
        for rect, action in getattr(self, "_playback_rects", []):
            rx, ry, rw, rh = rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                if disabled and action != "toggle_train_pause":
                    return
                self._handle_playback_action(action)
                return

    def _open_spawn_popup(self, fid: str, which: str, anchor_rect) -> None:
        """Öppna en popup-meny med val för Mode eller Tpl för FG ``fid``.

        För ``which='mode'`` listas alla SPAWN_MODES.
        För ``which='template'`` listas "(default)" + alla namn som är
        sparade under aktuellt valt mode (eller "uniform" om inget mode
        är valt än).
        """
        if which == "mode":
            # "(default)" ligger nu under Mode — väljs den återgår FG:n
            # till projektfilens spawn-konfiguration (template-valet
            # nollställs och en ny probe triggas).
            options = [("(default)", None)]
            for m in self._spawn_modes:
                options.append((self._spawn_mode_labels.get(m, m), m))
        else:
            ov = self._spawn_overrides.get(fid, {})
            # Använd aktivt valt mode; annars FG:s default-mode från
            # projektfilen så Tpl-popupen filtrerar mot rätt strategi
            # redan innan användaren rört Mode-dropdownen.
            cur_mode = (ov.get("mode")
                        or self._spawn_defaults.get(fid)
                        or "uniform")
            # Templates-popupen listar enbart sparade mallar för aktuellt
            # mode. "(default)" har flyttats till Mode-dropdownen.
            options = []
            mode_tpls = self._spawn_templates.get(cur_mode, {})
            for name in sorted(mode_tpls.keys()):
                options.append((name, name))
        self._spawn_popup = {
            "fid": fid,
            "which": which,
            "options": options,
            "anchor": tuple(anchor_rect),
        }
        self._spawn_popup_item_rects = []

    def _apply_spawn_popup_selection(self, value) -> None:
        """Apply the user's choice from the active spawn popup."""
        pop = self._spawn_popup
        if not pop:
            return
        fid = pop["fid"]
        which = pop["which"]
        ov = dict(self._spawn_overrides.get(fid, {}))
        trigger_probe = False
        if which == "mode":
            if value is None:
                # "(default)" valt under Mode → återgå till projekt-
                # filens spawn-konfiguration. Detta är ett aktivt val
                # och triggar en ny probe. Markera FG som "explicit
                # default" så etiketten visar "(default)" istället för
                # FG:s sparade strategi-namn.
                ov["mode"] = None
                ov["template"] = None
                self._spawn_explicit_default.add(fid)
                trigger_probe = True
            else:
                # Användaren har aktivt valt ett mode (≠ default) → ta
                # bort ev. explicit-default-flagga.
                self._spawn_explicit_default.discard(fid)
                ov["mode"] = str(value)
                # Byte av mode återställer template-valet — sparade
                # templates är per-mode och gamla namnet är inte garanterat
                # giltigt under det nya moden.
                ov["template"] = None
                # Edge case: om det bara finns exakt ett template
                # sparat för det nyvalda moden, auto-välj det och
                # trigga probe direkt.
                mode_tpls = self._spawn_templates.get(str(value), {})
                if len(mode_tpls) == 1:
                    only_name = next(iter(mode_tpls.keys()))
                    ov["template"] = only_name
                    trigger_probe = True
                # Annars: bara byte av mode (filterval) — ingen probe.
        else:
            # Aktivt template-val triggar alltid en probe.
            self._spawn_explicit_default.discard(fid)
            ov["template"] = value
            if value is not None and not ov.get("mode"):
                # Om mode inte var explicit satt, ärv FG:ns default-mode
                # från projektfilen så Tpl-listan filtrerats korrekt.
                ov["mode"] = (self._spawn_defaults.get(fid) or "uniform")
            trigger_probe = True
        self._spawn_overrides[fid] = ov
        if trigger_probe:
            self._spawn_dirty = True
        self._spawn_popup = None
        self._spawn_popup_item_rects = []

    def _draw_spawn_popup(self) -> None:
        """Render the active spawn-popup overlay (called after heatmaps)."""
        pop = self._spawn_popup
        if not pop:
            return
        pg = self._pg
        options = pop["options"]
        if not options:
            return
        ax, ay, aw, ah = pop["anchor"]
        item_h = 16
        pad_x = 6
        # Width: max text width + padding, but at least the anchor width.
        font = self._font
        label_widths = [font.size(lbl)[0] for lbl, _ in options]
        menu_w = max(aw, max(label_widths) + pad_x * 2 + 4)
        menu_h = item_h * len(options) + 2
        # Anchor below the dropdown rect; clamp inside window.
        mx0 = max(0, min(ax, self._win_w - menu_w))
        my0 = ay + ah
        if my0 + menu_h > self._win_h:
            my0 = max(0, ay - menu_h)
        # Background.
        pg.draw.rect(self._screen, (50, 50, 62), (mx0, my0, menu_w, menu_h))
        pg.draw.rect(self._screen, (160, 160, 180),
                     (mx0, my0, menu_w, menu_h), 1)
        self._spawn_popup_item_rects = []
        for i, (label, value) in enumerate(options):
            ry = my0 + 1 + i * item_h
            item_rect = (mx0 + 1, ry, menu_w - 2, item_h)
            # Highlight current selection.
            cur = self._spawn_overrides.get(pop["fid"], {})
            if pop["which"] == "mode":
                is_current = (cur.get("mode") or "uniform") == value
            else:
                is_current = cur.get("template") == value
            if is_current:
                pg.draw.rect(self._screen, (75, 85, 110), item_rect)
            txt = font.render(label, True, (230, 230, 240))
            self._screen.blit(txt, (mx0 + pad_x, ry + 1))
            self._spawn_popup_item_rects.append((item_rect, value))

    def _hit_playback_slider(self, pos) -> bool:
        return self._hit_rect(self._playback_track_rect, pos)

    def _hit_plot_scroll(self, pos) -> str:
        """Returnera 'thumb', 'track' eller '' beroende på var klicket
        landade i plot-scrollbaren."""
        thumb = self._plot_scroll_thumb_rect
        track = self._plot_scroll_track_rect
        if thumb is not None and self._hit_rect(thumb, pos):
            return "thumb"
        if track is not None and self._hit_rect(track, pos):
            return "track"
        return ""

    def _plot_scroll_offset_from_x(self, mx: int) -> Optional[float]:
        """Mappa musens x-koord till en ny plot-scroll-offset (tick-koord
        för fönstrets vänsterkant). Använder cachad data-span från senaste
        ``_draw_plot``. Returnerar ``None`` om scroll inte är aktiv."""
        track = self._plot_scroll_track_rect
        thumb = self._plot_scroll_thumb_rect
        span = getattr(self, "_plot_scroll_span_cache", None)
        if track is None or thumb is None or not span:
            return None
        tx, _ty, tw, _th = track
        thw = thumb[2]
        data_xmin, data_xmax = span
        full_span = float(data_xmax - data_xmin)
        win_w_x = float(self._plot_window_width)
        if full_span <= win_w_x:
            return None
        # Placera thumb så att muspekaren behåller sitt grepp om thumb
        # (drag_dx = mx_down - thumb_x lagras vid klick).
        new_thumb_x = mx - int(self._plot_scroll_drag_dx)
        max_x = tx + tw - thw
        new_thumb_x = max(tx, min(max_x, new_thumb_x))
        denom = max(1, tw - thw)
        frac = (new_thumb_x - tx) / float(denom)
        frac = max(0.0, min(1.0, frac))
        return float(data_xmin) + frac * (full_span - win_w_x)

    def _playback_idx_from_x(self, mx: int) -> Optional[int]:
        """Mappar muspos x till en frame-index i ``_current_rollout``.

        Returnerar None om slidern är inaktiv (ingen film eller
        recording pågår) eller ingen track finns registrerad.
        """
        if self._playback_track_rect is None:
            return None
        n = len(self._current_rollout)
        if n == 0 or self._recording:
            return None
        rx, _ry, rw, _rh = self._playback_track_rect
        # Trackens "riktiga" bredd är rw - 4 (vi padda hit-rect 2 px på var sida).
        track_x = rx + 2
        track_w = max(1, rw - 4)
        frac = (mx - track_x) / float(track_w)
        frac = max(0.0, min(1.0, frac))
        return int(round(frac * (n - 1)))

    def _handle_playback_action(self, action: str) -> None:
        # Träningspausen är oberoende av om någon film finns — hantera
        # den först så knappen fungerar även innan första probe-rollouten
        # är klar.
        if action == "toggle_train_pause":
            self._paused = not self._paused
            self._playback_blink_until = 0.0
            return
        n = len(self._current_rollout)
        if n == 0:
            return
        # Klick stänger av blinkningen.
        self._playback_blink_until = 0.0
        if action == "toggle_play":
            if self._playback_mode == "playing":
                self._playback_mode = "paused"
            else:
                # Gå in i replay vid första play-klick (eller fortsätt
                # från nuvarande frame om vi redan var i replay). Om vi
                # står på sista framen (klippet har spelats klart),
                # börja om från frame 0 — annars skulle play omedelbart
                # studsa tillbaka till "paused" via auto-advance.
                if self._playback_mode == "live":
                    self._playback_idx = 0
                elif self._playback_idx >= n - 1:
                    self._playback_idx = 0
                self._playback_mode = "playing"
                self._playback_last_advance = time.monotonic()
        elif action == "step_fwd":
            if self._playback_mode == "live":
                self._playback_idx = 0
            else:
                self._playback_idx = min(n - 1, self._playback_idx + 1)
            self._playback_mode = "paused"
        elif action == "step_back":
            if self._playback_mode == "live":
                self._playback_idx = n - 1
            else:
                self._playback_idx = max(0, self._playback_idx - 1)
            self._playback_mode = "paused"
        elif action == "speed_up":
            self._playback_fps = min(60.0, self._playback_fps + 1.0
                                     if self._playback_fps < 10
                                     else self._playback_fps + 5.0)
        elif action == "speed_down":
            self._playback_fps = max(1.0, self._playback_fps - 1.0
                                     if self._playback_fps <= 10
                                     else self._playback_fps - 5.0)
        elif action == "to_live":
            self._playback_mode = "live"

    def _maybe_render(self) -> None:
        now = time.monotonic()
        if (now - self._last_frame_ts) < self._MIN_FRAME_INTERVAL:
            return
        self._last_frame_ts = now
        self._render_full()
        # FPS bookkeeping.
        self._fps_frames += 1
        if (now - self._fps_clock_t) >= 1.0:
            self._fps_value = self._fps_frames / (now - self._fps_clock_t)
            self._fps_clock_t = now
            self._fps_frames = 0

    # ---- Rendering ---------------------------------------------------
    def _render_full(self) -> None:
        screen = self._screen
        screen.fill((18, 18, 22))
        self._draw_status_bar()
        # Replay-state-swap: när vi inte är i live-läget ritar
        # heatmap-griden frame N av ``_current_rollout`` i stället för
        # live-state. Vi swappar in snapshotten innan ``_draw_heatmaps``
        # och återställer efteråt, så plot-panelen och status-baren
        # förblir oförändrade (plot-flikarna lämnas medvetet orörda enligt
        # användarens önskemål — de visar globala kurvor över hela körningen).
        # Under pågående inspelning visar vi alltid live-state, även om
        # användaren tidigare hade pausat/spelat upp en gammal rollout
        # med uppspelningsverktygen. Annars skulle nyinspelade frames
        # inte synas förrän rolloutten är klar (eller alls, i inference
        # där "live"-knappen saknas).
        swap = None
        if (not self._recording
                and self._playback_mode != "live"
                and self._current_rollout
                and 0 <= self._playback_idx < len(self._current_rollout)):
            swap = self._swap_in_frame(self._current_rollout[self._playback_idx])
        try:
            self._draw_heatmaps()
        finally:
            if swap is not None:
                self._swap_out_frame(swap)
        self._draw_plot()
        self._draw_playback_bar()
        # Spawn-popup ritas sist så den ligger ovanpå allt annat.
        self._draw_spawn_popup()
        self._pg.display.flip()

    def _swap_in_frame(self, frame: dict) -> dict:
        """Tillfälligt ersätt live-state med en snapshot. Returnerar
        ett ``saved``-dict som ``_swap_out_frame`` använder för att
        återställa."""
        saved = {
            "biomass": self._biomass,
            "totals": self._totals,
            "b0": self._b0,
            "loss_breakdown": self._loss_breakdown,
            "diet_breakdown": self._diet_breakdown,
            "action_fracs": self._action_fracs,
            "tick": self._tick,
        }
        self._biomass = frame.get("biomass", {})
        self._totals = frame.get("totals", {})
        self._b0 = frame.get("b0", {})
        self._loss_breakdown = frame.get("loss_breakdown", {})
        self._diet_breakdown = frame.get("diet_breakdown", {})
        self._action_fracs = frame.get("action_fracs", {})
        self._tick = int(frame.get("tick", 0))
        return saved

    def _swap_out_frame(self, saved: dict) -> None:
        self._biomass = saved["biomass"]
        self._totals = saved["totals"]
        self._b0 = saved["b0"]
        self._loss_breakdown = saved["loss_breakdown"]
        self._diet_breakdown = saved["diet_breakdown"]
        self._action_fracs = saved["action_fracs"]
        self._tick = saved["tick"]

    def _draw_playback_bar(self) -> None:
        """Rita uppspelningskontroller under heatmap-griden.

        Layout: [<<] [play/pause] [>>] [-]  speed=X fps  [+]   live/replay
                frame = i/N  (blinkar när ny film finns)

        Knapparna är gråa (disabled) medan en probe pågår
        (``_recording = True``) eller om ingen film finns ännu.
        """
        pg = self._pg
        x, y, w, h = self._playback_rect
        # Bakgrund.
        pg.draw.rect(self._screen, (22, 22, 28), (x, y, w, h))
        pg.draw.line(self._screen, (50, 50, 60),
                     (x, y), (x + w, y), 1)

        disabled = self._recording or not self._current_rollout
        # Blink-effekt på hela rad när ny rollout precis blivit klar och
        # användaren ännu inte interagerat. Toggle 2 Hz.
        blinking = (not disabled
                    and self._playback_blink_until > time.monotonic()
                    and self._playback_mode == "live")
        blink_on = blinking and (int(time.monotonic() * 2) % 2 == 0)

        # Knapp-spec: (label, action_id, width).
        is_playing = (self._playback_mode == "playing")
        buttons = [
            ("<<", "step_back", 32),
            ("|>" if not is_playing else "||", "toggle_play", 32),
            (">>", "step_fwd", 32),
            ("-", "speed_down", 24),
            (f"{self._playback_fps:.0f} fps", None, 60),
            ("+", "speed_up", 24),
        ]
        # ``live``/``replay``-knappen är bara meningsfull i train-läge,
        # där en pågående probe-rollout kan "ta över" bufferten och
        # användaren vill kunna hoppa tillbaka till live-vyn. Vid
        # inferens finns ingen live-vy att återgå till — filmen ÄR
        # rolloutten — så vi döljer knappen där.
        if self.mode == "train":
            buttons.append(
                ("live" if self._playback_mode == "live" else "replay",
                 "to_live", 56)
            )
        # Paus/resume av själva träningsloopen visas endast i train-läge.
        # Vid inferens finns ingen träning att pausa — inference-loopen
        # är redan klar när ``wait_for_close`` körs och uppspelnings-
        # kontrollerna (|>, <<, >>) räcker för att granska filmen.
        if self.mode == "train":
            buttons.append(
                ("resume train" if self._paused else "pause train",
                 "toggle_train_pause", 100)
            )

        self._playback_rects = []
        bx = x + 8
        by = y + (h - 22) // 2
        for label, action, bw in buttons:
            rect = (bx, by, bw, 22)
            if action is None:
                # Etikett (speed-indikator) — ingen klickzon, ingen ram.
                col = (210, 210, 220)
                surf = self._font.render(label, True, col)
                self._screen.blit(surf,
                                  (bx + (bw - surf.get_width()) // 2,
                                   by + (22 - surf.get_height()) // 2))
                bx += bw + 4
                continue
            # Knappfärg. ``toggle_train_pause`` är alltid aktiv (även
            # under recording / utan film), så den hanteras separat.
            always_active = (action == "toggle_train_pause")
            if disabled and not always_active:
                bg = (40, 40, 46)
                fg = (110, 110, 120)
                border = (60, 60, 70)
            else:
                if blink_on and not always_active:
                    bg = (90, 70, 30)
                    border = (220, 180, 60)
                else:
                    bg = (50, 50, 60)
                    border = (90, 90, 105)
                fg = (235, 235, 245)
                # Highlight aktiv play-toggle.
                if action == "toggle_play" and is_playing:
                    bg = (60, 90, 60)
                # Highlight när träningen är pausad.
                if action == "toggle_train_pause" and self._paused:
                    bg = (90, 50, 50)
                    border = (200, 100, 100)
            pg.draw.rect(self._screen, bg, rect)
            pg.draw.rect(self._screen, border, rect, 1)
            surf = self._font.render(label, True, fg)
            self._screen.blit(surf,
                              (bx + (bw - surf.get_width()) // 2,
                               by + (22 - surf.get_height()) // 2))
            self._playback_rects.append((rect, action))
            bx += bw + 4

        # Frame-indikator till höger.
        n = len(self._current_rollout)
        if self._playback_mode == "live":
            ind = f"live  (saved frames: {n})"
        else:
            ind = f"frame = {self._playback_idx + 1}/{n}"
        if self._recording:
            ind = f"recording... ({len(self._pending_rollout)} frames)"
        ind_surf = self._font.render(ind, True, (200, 200, 215))
        # Reservera stabil bredd för indikator-zonen så att scrub-
        # slidern inte ändrar storlek när texten växlar mellan
        # "live ...", "frame = i/N" och "recording... (...)". Vi
        # beräknar maxbredden av samtliga möjliga varianter med
        # de nuvarande räknarvärdena och använder den för att
        # placera scrub-slidens högerkant.
        _max_n = max(n, 1)
        _max_pending = len(self._pending_rollout)
        _ind_variants = [
            f"live  (saved frames: {_max_n})",
            f"frame = {_max_n}/{_max_n}",
            f"recording... ({_max_pending} frames)",
        ]
        # MEMORY LEAK FIX: anvand font.size()[0] istallet for
        # font.render(...).get_width(). Memray --leaks-flamegraph visade
        # att denna callsite ensam stod for 218 MB lackt SDL-minne
        # (TTF_Render_Internal -> AllocateAlignedPixels -> SDL_malloc)
        # eftersom den allokerar en full Surface per anrop bara for att
        # mata textbredden, och Surface-objektet kastas direkt utan att
        # SDL aterlamnar pixelbufferten. font.size() returnerar (w,h)
        # utan att allokera nagon Surface alls.
        _ind_reserve_w = max(
            self._font.size(s)[0] for s in _ind_variants)
        ind_zone_left = x + w - _ind_reserve_w - 10
        # Höger-justera den faktiska texten inom den reserverade zonen
        # så att den alltid hamnar vid samma högerkant.
        ind_x = x + w - ind_surf.get_width() - 10
        self._screen.blit(ind_surf,
                          (ind_x,
                           y + (h - ind_surf.get_height()) // 2))

        # Scrub-slider (timeline) mellan knapparna och frame-indikatorn.
        # Disabled samma villkor som transport-knapparna (recording eller
        # ingen film). Klick/drag sätter ``_playback_idx`` och pausar.
        track_x = bx + 8
        # Använd den reserverade indikator-zonens vänsterkant (stabil)
        # istället för den dynamiska ``ind_x`` (som flyttar sig när
        # indikator-texten ändras), så scrub-slidern behåller exakt
        # samma bredd oavsett playback-mode/recording-state.
        track_right = ind_zone_left - 10
        track_w = track_right - track_x
        track_h = 6
        track_y = y + (h - track_h) // 2
        if track_w >= 60:
            # Bakgrund.
            pg.draw.rect(self._screen, (50, 50, 60),
                         (track_x, track_y, track_w, track_h))
            border_col = (60, 60, 70) if disabled else (110, 110, 125)
            pg.draw.rect(self._screen, border_col,
                         (track_x, track_y, track_w, track_h), 1)
            # Filled del (progress) + handle, om vi har en film.
            if n > 0 and not self._recording:
                if self._playback_mode == "live":
                    idx_for_pos = n - 1
                else:
                    idx_for_pos = max(0, min(n - 1, self._playback_idx))
                frac = (idx_for_pos / max(1, n - 1)) if n > 1 else 0.0
                fill_w = int(round(frac * track_w))
                fill_col = (90, 140, 200) if not disabled else (70, 80, 95)
                if fill_w > 0:
                    pg.draw.rect(self._screen, fill_col,
                                 (track_x, track_y, fill_w, track_h))
                hx = track_x + fill_w
                hy = track_y + track_h // 2
                knob_col = (230, 230, 240) if not disabled else (110, 110, 120)
                pg.draw.circle(self._screen, knob_col, (hx, hy), 6)
                pg.draw.circle(self._screen, (30, 30, 40), (hx, hy), 6, 1)
            # Hit-rect (lite tjockare för enkel klick).
            self._playback_track_rect = (track_x - 2, track_y - 8,
                                         track_w + 4, track_h + 16)
        else:
            self._playback_track_rect = None

    def _draw_status_bar(self) -> None:
        pg = self._pg
        x, y, w, h = self._status_rect
        parts = [f"mode = {self.mode}", f"tick = {self._tick}"]
        for k in ("gen", "iter", "T"):
            if k in self._status:
                v = self._status[k]
                if isinstance(v, float):
                    parts.append(f"{k} = {v:.3f}")
                else:
                    parts.append(f"{k} = {v}")
        if self._paused:
            parts.append("[PAUSED]")
        if self._log_heatmap:
            parts.append("hm:log")
        if self._log_plot:
            parts.append("plot:log")
        if self._solo:
            parts.append(f"solo = {self._solo}")
        parts.append(f"grid = {self.grid_w}x{self.grid_h}")
        parts.append(f"fps = {self._fps_value:4.1f}")
        text = "  ".join(parts)
        surf = self._font_big.render(text, True, (230, 230, 235))
        self._screen.blit(surf, (x + 6, y + 3))

        # ---- Rollout-längd-slider (rad 2 i status-baren) -------------
        # Layout: vänster label "rollout ticks = N", track, höger gränser.
        # Range = [_ticks_min, _ticks_max] (log-skala). Vid drag visas
        # det realtidsuppdaterade värdet inline.
        cur = self.get_ticks()
        marker = "*" if self._ticks_override is not None else ""
        label = f"Probe ticks{marker} = {cur}"
        lbl_surf = self._font.render(label, True, (210, 210, 220))
        row2_y = y + 22
        row2_h = 18
        self._screen.blit(lbl_surf, (x + 6, row2_y + 2))
        lo_surf = self._font.render(str(self._ticks_min), True, (150, 150, 160))
        hi_surf = self._font.render(str(self._ticks_max), True, (150, 150, 160))
        # Gemensam label-bredd så att Probe-ticks- och Perturbation-
        # ticks-slidrarna får exakt samma track-startposition (och
        # därmed samma längd). Vi mäter "Perturbation ticks"-etiketten
        # här utan markör/värde-suffix — den faktiska etiketten kan
        # variera i längd p.g.a. ``*``-markören och värdet, men vi
        # vill ha en stabil, gemensam baseline. Använd max av båda
        # rendererade bredderna nedan.
        cur2_preview = self.get_neval_ticks()
        marker2_preview = "*" if self._neval_override is not None else ""
        # MEMORY LEAK FIX: anvand font.size()[0] istallet for
        # font.render(...).get_width() for ren matning. Memray --leaks
        # visade att _draw_status_bar stod for 228 MB lackt SDL-minne
        # via TTF_Render_Internal -> AllocateAlignedPixels -> SDL_malloc
        # (samma antipattern som redan fixad _draw_playback_bar:1968).
        _lbl2_preview_w = self._font.size(
            f"Perturbation ticks{marker2_preview} = {cur2_preview}")[0]
        _label_w = max(lbl_surf.get_width(), _lbl2_preview_w)
        # Track-rect: börjar efter label (+ liten gutter), slutar före
        # hi-label (+ gutter). Mappar mot fönsterbredden.
        try:
            _lo2_preview_w = self._font.size(str(self._neval_min))[0]
        except Exception:
            _lo2_preview_w = lo_surf.get_width()
        _lo_w = max(lo_surf.get_width(), _lo2_preview_w)
        track_x = x + 6 + _label_w + 12 + _lo_w + 6
        # Begränsa sliderns högerkant så att den slutar där plot-rutan
        # börjar (annars sträcker den sig över hela fönsterbredden och
        # täcker grafritarrutan, vilket ser fult ut och är onödigt långt).
        try:
            plot_left = int(self._plot_rect[0])
        except Exception:
            plot_left = x + w
        # Lämna plats för låsikon (16 px + 6 px gutter) mellan track och
        # hi-label. I inference-läget döljs låsikonerna och
        # n_eval_ticks-slidern helt, så ingen plats reserveras.
        _is_inference = str(self.mode).lower().startswith("infer")
        if _is_inference:
            # Tvinga ticks-slidern olåst när låset inte är synligt.
            self._ticks_locked = False
            lock_size = 0
            lock_gutter = 0
        else:
            lock_size = 14
            lock_gutter = 6
        # Använd max av båda hi-labels bredd så att högerkanten
        # (och därmed track-längden) blir identisk för båda raderna.
        # MEMORY LEAK FIX: font.size() istallet for font.render().get_width().
        try:
            _hi2_preview_w = self._font.size(str(self._neval_max))[0]
        except Exception:
            _hi2_preview_w = hi_surf.get_width()
        _hi_w = max(hi_surf.get_width(), _hi2_preview_w)
        right_limit = (plot_left - (_hi_w + 6) - 4
                       - (lock_size + lock_gutter))
        track_w_max = right_limit - track_x
        track_w = max(80, track_w_max)
        track_y = row2_y + (row2_h - 6) // 2
        track_h = 6
        # Lo-label strax före tracken; hi-label strax efter.
        self._screen.blit(lo_surf, (track_x - 6 - lo_surf.get_width(),
                                    row2_y + 2))
        # Hi-label hamnar efter låsikonen (track + gutter + lock + gutter).
        self._screen.blit(hi_surf,
                          (track_x + track_w + lock_gutter + lock_size + 6,
                           row2_y + 2))
        # Track-bakgrund.
        pg.draw.rect(self._screen, (60, 60, 70),
                     (track_x, track_y, track_w, track_h))
        pg.draw.rect(self._screen, (110, 110, 125),
                     (track_x, track_y, track_w, track_h), 1)
        # Default-markering (liten tick).
        try:
            df = self._ticks_pos_from_value(self._ticks_default)
            dx = int(track_x + df * (track_w - 1))
            pg.draw.line(self._screen, (140, 140, 150),
                         (dx, track_y - 2), (dx, track_y + track_h + 2), 1)
        except Exception:
            pass
        # Handle på aktuell position.
        try:
            frac = self._ticks_pos_from_value(cur)
        except Exception:
            frac = 0.0
        hx = int(track_x + frac * (track_w - 1))
        hcol = (220, 220, 90) if self._ticks_override is not None else (200, 200, 210)
        pg.draw.circle(self._screen, hcol, (hx, track_y + track_h // 2), 6)
        pg.draw.circle(self._screen, (40, 40, 50),
                       (hx, track_y + track_h // 2), 6, 1)
        # Spara track-rect för hit-test (vi gör hit-rect lite tjockare).
        self._ticks_track_rect = (track_x - 2, track_y - 6,
                                  track_w + 4, track_h + 12)
        # Lås-ikon direkt höger om tracken (mellan track och hi-label).
        # Döljs i inference-läget.
        if _is_inference:
            self._ticks_lock_rect = None
        else:
            lock_x = track_x + track_w + lock_gutter
            lock_y = row2_y + (row2_h - lock_size) // 2
            self._ticks_lock_rect = self._draw_lock_icon(
                lock_x, lock_y, lock_size, self._ticks_locked)

        # ---- n_eval_ticks-slider (rad 3 i status-baren) --------------
        # Endast meningsfull under träning — döljs helt i inference.
        if _is_inference:
            self._neval_track_rect = None
            self._neval_lock_rect = None
            return
        # Identisk layout som ticks-slidern men med "n_eval_ticks" som
        # label. Aktiv främst under träning; värdet konsumeras av
        # train.py mellan ARS-iterationer.
        cur2 = self.get_neval_ticks()
        marker2 = "*" if self._neval_override is not None else ""
        label2 = f"Perturbation ticks{marker2} = {cur2}"
        lbl2_surf = self._font.render(label2, True, (210, 210, 220))
        row3_y = y + 40
        row3_h = 18
        self._screen.blit(lbl2_surf, (x + 6, row3_y + 2))
        lo2_surf = self._font.render(str(self._neval_min), True,
                                     (150, 150, 160))
        hi2_surf = self._font.render(str(self._neval_max), True,
                                     (150, 150, 160))
        # Använd samma gemensamma label-bredd som rad 2 så att
        # track-startpositionen (och därmed sliderlängden) blir
        # identisk för Probe- och Perturbation-ticks-slidrarna.
        _label2_w = max(lbl2_surf.get_width(), lbl_surf.get_width())
        _lo2_w = max(lo2_surf.get_width(), lo_surf.get_width())
        track2_x = (x + 6 + _label2_w + 12
                    + _lo2_w + 6)
        _hi2_w = max(hi2_surf.get_width(), hi_surf.get_width())
        right_limit2 = (plot_left - (_hi2_w + 6) - 4
                        - (lock_size + lock_gutter))
        track2_w_max = right_limit2 - track2_x
        track2_w = max(80, track2_w_max)
        track2_y = row3_y + (row3_h - 6) // 2
        track2_h = 6
        self._screen.blit(lo2_surf,
                          (track2_x - 6 - lo2_surf.get_width(),
                           row3_y + 2))
        self._screen.blit(
            hi2_surf,
            (track2_x + track2_w + lock_gutter + lock_size + 6, row3_y + 2))
        pg.draw.rect(self._screen, (60, 60, 70),
                     (track2_x, track2_y, track2_w, track2_h))
        pg.draw.rect(self._screen, (110, 110, 125),
                     (track2_x, track2_y, track2_w, track2_h), 1)
        try:
            df2 = self._neval_pos_from_value(self._neval_default)
            dx2 = int(track2_x + df2 * (track2_w - 1))
            pg.draw.line(self._screen, (140, 140, 150),
                         (dx2, track2_y - 2),
                         (dx2, track2_y + track2_h + 2), 1)
        except Exception:
            pass
        try:
            frac2 = self._neval_pos_from_value(cur2)
        except Exception:
            frac2 = 0.0
        hx2 = int(track2_x + frac2 * (track2_w - 1))
        hcol2 = ((220, 220, 90) if self._neval_override is not None
                 else (200, 200, 210))
        pg.draw.circle(self._screen, hcol2,
                       (hx2, track2_y + track2_h // 2), 6)
        pg.draw.circle(self._screen, (40, 40, 50),
                       (hx2, track2_y + track2_h // 2), 6, 1)
        self._neval_track_rect = (track2_x - 2, track2_y - 6,
                                  track2_w + 4, track2_h + 12)
        # Lås-ikon för n_eval_ticks-slidern.
        lock2_x = track2_x + track2_w + lock_gutter
        lock2_y = row3_y + (row3_h - lock_size) // 2
        self._neval_lock_rect = self._draw_lock_icon(
            lock2_x, lock2_y, lock_size, self._neval_locked)

    def _draw_heatmaps(self) -> None:
        ox, oy = self._heatmap_origin
        # Slider-rektangellistan byggs om varje frame så stale entries
        # från föregående layout aldrig kan trigga drag i fel panel.
        self._slider_rects = []
        # Samma sak för spawn-dropdown-rektanglarna.
        self._spawn_dropdown_rects = []
        for idx, fid in enumerate(self.fg_ids):
            row = idx // self._cols
            col = idx % self._cols
            px = ox + col * self._panel_w
            py = oy + row * self._panel_h
            self._draw_one_heatmap(fid, px, py)

    @staticmethod
    def _abbrev_fg(name: str) -> str:
        """Förkortning för FG-namn enligt diet-radens regler.

        - Mer än ett ord (separerat på ``_`` eller mellanslag):
          begynnelsebokstav per ord, t.ex. ``pelagic_fish`` → ``pf``.
        - Ett enda ord skrivet i singular (slutar ej på 's'):
          första och sista bokstaven, t.ex. ``cod`` → ``cd``.
        - Ett enda ord i plural (slutar på 's'):
          första och näst sista bokstaven, t.ex. ``gadoids`` → ``gd``.

        Resultatet är gemener. Tomt namn → tom sträng.
        """
        if not name:
            return ""
        s = str(name).strip().lower()
        # Dela på understreck och mellanslag.
        parts = [p for p in s.replace('-', '_').replace(' ', '_').split('_')
                 if p]
        if len(parts) > 1:
            return ''.join(p[0] for p in parts if p)
        word = parts[0]
        if len(word) == 1:
            return word
        if word.endswith('s'):
            # Plural → första + näst sista bokstaven.
            return word[0] + word[-2]
        # Singular → första + sista bokstaven.
        return word[0] + word[-1]

    def _draw_one_heatmap(self, fid: str, px: int, py: int) -> None:
        pg = self._pg
        pad = 4
        # Five info lines above the heatmap: FG name, 'B0 = … B = …', the
        # action distribution 'mv/rs/et = …', biomass-loss breakdown
        # 'pr/st/im = …' and the diet breakdown '<abbr>/… = X/…%' (last
        # three blank for NDMs), följt av en b0-slider mellan info-blocket
        # och heatmapen. title_h måste rymma alla fem rader + slider.
        title_h = 104
        arr = self._biomass.get(fid)
        # Panel background.
        pg.draw.rect(self._screen, (28, 28, 34),
                     (px, py, self._panel_w, self._panel_h))
        # Solo dimming.
        dim = (self._solo is not None and self._solo != fid)
        # Title text (FG id on line 1, biomass info on line 2).
        total = self._totals.get(fid, 0.0)
        b0 = self._b0.get(fid, 0.0)
        colour = self._fg_colour[fid]
        tcol = tuple(int(c * (0.35 if dim else 1.0)) for c in colour)
        name_surf = self._font.render(self._display_name(fid), True, tcol)
        self._screen.blit(name_surf, (px + pad, py + 1))
        def _fmt_b(v: float) -> str:
            # <10 -> 2 decimaler, <100 -> 1 decimal, annars 0.
            a = abs(v)
            if a < 10.0:
                s = f"{v:,.2f}"
            elif a < 100.0:
                s = f"{v:,.1f}"
            else:
                s = f"{v:,.0f}"
            return s.replace(",", " ")
        # När användaren satt en b0-override via slidern visas det
        # värdet i info-raden (markerat med ``*``); annars visas den
        # senast registrerade rollout-starten ``self._b0[fid]``.
        b0_default = self._b0_defaults.get(fid)
        b0_override = self._b0_overrides.get(fid)
        if b0_override is not None:
            b0_shown = b0_override
            b0_marker = "*"
        else:
            b0_shown = b0
            b0_marker = ""
        info_txt = f"B0{b0_marker} = {_fmt_b(b0_shown)}  B = {_fmt_b(total)}"
        info_col = (180, 180, 190) if not dim else (90, 90, 95)
        info_surf = self._font.render(info_txt, True, info_col)
        info_y = py + 1 + name_surf.get_height()
        self._screen.blit(info_surf, (px + pad, info_y))

        # Third info line: action distribution mv/rs/et for DMs. The
        # numbers come from the same per-tab rolling buffers that feed
        # the move/rest/eat plot tabs (stored as percent 0..100) och
        # visas som heltalsprocent ('mv/rs/et=X/Y/Z%') för att matcha
        # stilen i 'pr/st/im'-raden nedanför. NDMs har ingen policy och
        # får en tom rad så heatmap-origin förblir aligned.
        act_y = info_y + info_surf.get_height()
        is_ndm = fid in self._ndm_ids
        if not is_ndm:
            # Header läser senaste mv/rs/et från ``_action_fracs`` (uppdateras
            # per tick via ``update_action_fracs``) i stället för från plot-
            # serie-buffrarna. Det gör att plot-flikarna kan matas med en
            # annan x-skala (t.ex. end-of-probe med ``viz_step``) utan att
            # headern slutar uppdateras per tick.
            _af = self._action_fracs.get(fid)
            if _af is not None:
                mv = float(_af.get('move', 0.0))
                rs = float(_af.get('rest', 0.0))
                et = float(_af.get('eat',  0.0))
            else:
                mv = rs = et = None
            if mv is not None and rs is not None and et is not None:
                act_txt = (f"mv/rs/et = {mv:.0f}/{rs:.0f}/{et:.0f}%")
                act_surf = self._font.render(act_txt, True, info_col)
                self._screen.blit(act_surf, (px + pad, act_y))
                # Fourth info line: biomass-loss breakdown pr/st/im as
                # percentages. Only rendered when a loss-breakdown record
                # exists for this FG (probe rollout or inference tick has
                # pushed it). NDMs always skip this line.
                lb = self._loss_breakdown.get(fid)
                if lb is not None:
                    pr = float(lb.get('predation', 0.0)) * 100.0
                    st = float(lb.get('starvation', 0.0)) * 100.0
                    im = float(lb.get('impact', 0.0)) * 100.0
                    loss_txt = (f"pr/st/im = {pr:.0f}/{st:.0f}/{im:.0f}%")
                    loss_surf = self._font.render(loss_txt, True, info_col)
                    loss_y = act_y + act_surf.get_height()
                    self._screen.blit(loss_surf, (px + pad, loss_y))
                    # Fifth info line: diet breakdown — vilka byten den
                    # här DM-FG:n har ätit under rollouten, normaliserat
                    # till procentandelar. Formateras dynamiskt utifrån
                    # antalet byten med >0 intake: '<a1>/<a2>/… = X/Y/…%'.
                    diet = self._diet_breakdown.get(fid)
                    if diet:
                        items = [(pid, float(v)) for pid, v in diet.items()
                                 if float(v) > 0.0]
                        if items:
                            tot = sum(v for _, v in items)
                            if tot > 0.0:
                                items.sort(key=lambda kv: kv[1], reverse=True)
                                abbrs = [self._abbrev_fg(pid)
                                         for pid, _ in items]
                                pcts = [v / tot * 100.0 for _, v in items]
                                diet_txt = (
                                    "/".join(abbrs)
                                    + " = "
                                    + "/".join(f"{p:.0f}" for p in pcts)
                                    + "%"
                                )
                                diet_surf = self._font.render(
                                    diet_txt, True, info_col)
                                diet_y = loss_y + loss_surf.get_height()
                                self._screen.blit(
                                    diet_surf, (px + pad, diet_y))

        # ---- b0-slider --------------------------------------------------
        # Horisontell slider placerad mellan info-blocket ovanför och själva
        # heatmapen nedanför. Range = [0, 4 * b0_default]; mittposition
        # motsvarar projektets gridskalade default. När ingen default är
        # satt (FG saknar inference_initial_biomass) ritas en disabled
        # placeholder så heatmap-origin förblir aligned med andra paneler.
        slider_h = int(getattr(self, '_slider_h', 16))
        slider_y = py + title_h - slider_h - 4
        slider_x = px + pad
        slider_w = self.grid_w * self.cell_px
        track_y = slider_y + slider_h // 2 - 2
        track_h = 4
        # Track-rektangel registreras alltid (även för disabled-sliders),
        # men dragning aktiveras endast när b0_default > 0 (annars ingen
        # meningsfull range).
        if b0_default is not None and b0_default > 0.0:
            v_max = 4.0 * float(b0_default)
            v_cur = float(b0_override if b0_override is not None else b0_default)
            v_cur = max(0.0, min(v_max, v_cur))
            frac = v_cur / v_max if v_max > 0.0 else 0.0
            track_col = (60, 60, 80) if not dim else (40, 40, 50)
            fill_col = tuple(int(c * (0.35 if dim else 0.7)) for c in colour)
            knob_col = tuple(int(c * (0.35 if dim else 1.0)) for c in colour)
            pg.draw.rect(self._screen, track_col,
                         (slider_x, track_y, slider_w, track_h))
            pg.draw.rect(self._screen, fill_col,
                         (slider_x, track_y, int(slider_w * frac), track_h))
            knob_x = int(slider_x + slider_w * frac)
            knob_r = max(4, slider_h // 2)
            pg.draw.circle(self._screen, knob_col,
                           (knob_x, track_y + track_h // 2), knob_r)
            # Markera default-positionen (= oediterat initialvärde) med
            # ett litet streck så användaren ser var "ursprungsvärdet"
            # ligger. Range är [0, 4 × default], så default-fraktionen är
            # 1/4 av tracken (inte mitten).
            def_x = int(slider_x + slider_w * 0.25)
            pg.draw.line(self._screen, (220, 220, 230),
                         (def_x, track_y - 3),
                         (def_x, track_y + track_h + 3), 1)
            # Hit-rektangel för mus: täcker hela slider-raden, inte bara
            # tracken, så det är lätt att klicka.
            hit_rect = (slider_x - 2, slider_y - 2,
                        slider_w + 4, slider_h + 4)
            self._slider_rects.append((hit_rect, fid))
        else:
            # Disabled placeholder.
            pg.draw.rect(self._screen, (40, 40, 50),
                         (slider_x, track_y, slider_w, track_h))

        if arr is None:
            return
        # Heatmap surface.
        hm_x = px + pad
        hm_y = py + title_h
        H, W = arr.shape
        if (H, W) != (self.grid_h, self.grid_w):
            # Defensive: skip if shape doesn't match expected grid.
            return

        v = arr.astype(np.float32, copy=False)
        if self._log_heatmap:
            v = np.log1p(np.maximum(v, 0.0))
        vmax = float(v.max())
        if vmax <= 1e-12:
            idx = np.zeros_like(v, dtype=np.uint8)
        else:
            idx = np.clip((v / vmax) * 255.0, 0, 255).astype(np.uint8)
        rgb = self._lut[idx]  # (H, W, 3)
        if dim:
            rgb = (rgb.astype(np.uint16) * 90 // 255).astype(np.uint8)
        # pygame expects (W, H, 3); transpose first two axes.
        try:
            small = pg.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        except Exception:
            return
        scaled = pg.transform.scale(
            small, (W * self.cell_px, H * self.cell_px))
        self._screen.blit(scaled, (hm_x, hm_y))
        # Border.
        pg.draw.rect(self._screen, (60, 60, 70),
                     (hm_x, hm_y, W * self.cell_px, H * self.cell_px), 1)

        # ---- Colorbar legend under the heatmap ----------------------------
        # Per-FG normalisation: shows what the colour gradient maps to,
        # from 0 (left, dark) to vmax (right, bright). vmax reflects the
        # *current* per-FG max biomass in this tick (log1p when hm:log is on).
        cbar_y = hm_y + H * self.cell_px + 3
        cbar_w = W * self.cell_px
        cbar_strip_h = 6
        try:
            grad = self._lut[np.arange(256, dtype=np.uint8)]  # (256, 3)
            grad_img = np.broadcast_to(grad[np.newaxis, :, :],
                                       (cbar_strip_h, 256, 3)).copy()
            if dim:
                grad_img = (grad_img.astype(np.uint16) * 90 // 255).astype(np.uint8)
            cbar_surf = pg.surfarray.make_surface(
                np.transpose(grad_img, (1, 0, 2)))
            cbar_scaled = pg.transform.scale(
                cbar_surf, (cbar_w, cbar_strip_h))
            self._screen.blit(cbar_scaled, (hm_x, cbar_y))
            pg.draw.rect(self._screen, (60, 60, 70),
                         (hm_x, cbar_y, cbar_w, cbar_strip_h), 1)
        except Exception:
            pass
        # Labels: 0 left, vmax right; include "(log1p)" hint when toggled.
        lbl_col = (170, 170, 180) if not dim else (90, 90, 95)
        zero_lbl = self._font.render("0", True, lbl_col)
        self._screen.blit(zero_lbl, (hm_x, cbar_y + cbar_strip_h + 1))
        max_txt = self._fmt_compact(vmax)
        if self._log_heatmap:
            max_txt = f"log1p≤{max_txt}"
        max_lbl = self._font.render(max_txt, True, lbl_col)
        self._screen.blit(
            max_lbl,
            (hm_x + cbar_w - max_lbl.get_width(),
             cbar_y + cbar_strip_h + 1))

        # ---- Spawn-strategi-dropdowns under heatmapen ---------------------
        # Två rader: Mode och Tpl. Klick öppnar popup-meny som hanteras
        # av _handle_click. Default-läget visar "(default)" och betyder
        # att projektfilens spawn-konfiguration behålls vid nästa probe.
        dd_y0 = cbar_y + cbar_strip_h + 1 + self._font.get_height() + 4
        dd_w = cbar_w
        dd_row_h = 15
        ov = self._spawn_overrides.get(fid, {})
        cur_mode_raw = ov.get("mode")
        cur_tpl = ov.get("template")
        # Etikett-logik:
        #   * Användaren har aktivt valt "(default)" → visa "(default)".
        #   * Användaren har valt en mode/template → visa modets etikett.
        #   * Annars (ingen override) → visa FG:s default-mode från
        #     projektfilen om känt, annars "(default)".
        if fid in self._spawn_explicit_default:
            mode_label = "(default)"
            tpl_label = "—"
        elif cur_mode_raw is None:
            default_mode = self._spawn_defaults.get(fid)
            if default_mode:
                mode_label = self._spawn_mode_labels.get(
                    default_mode, default_mode)
            else:
                mode_label = "(default)"
            tpl_label = "—"
        else:
            mode_label = self._spawn_mode_labels.get(cur_mode_raw, cur_mode_raw)
            tpl_label = cur_tpl if cur_tpl else "—"
        dd_bg = (38, 38, 48) if not dim else (28, 28, 35)
        dd_border = (90, 90, 105) if not dim else (50, 50, 60)
        dd_text = (210, 210, 220) if not dim else (110, 110, 120)
        for i, (label_prefix, value_text, which) in enumerate(
                (("Mode: ", mode_label, "mode"),
                 ("Tpl:  ", tpl_label, "template"))):
            rx, ry = hm_x, dd_y0 + i * (dd_row_h + 2)
            rect = (rx, ry, dd_w, dd_row_h)
            pg.draw.rect(self._screen, dd_bg, rect)
            pg.draw.rect(self._screen, dd_border, rect, 1)
            txt = f"{label_prefix}{value_text}"
            # Truncate to fit.
            max_w = dd_w - 14
            txt_surf = self._font.render(txt, True, dd_text)
            if txt_surf.get_width() > max_w:
                # Crude truncation.
                while len(txt) > 4 and txt_surf.get_width() > max_w:
                    txt = txt[:-2]
                    txt_surf = self._font.render(txt + "…", True, dd_text)
                txt = txt + "…"
            self._screen.blit(txt_surf, (rx + 4, ry + 1))
            # Triangle indicator on the right.
            tri_x = rx + dd_w - 10
            tri_y = ry + dd_row_h // 2
            pg.draw.polygon(self._screen, dd_text,
                            [(tri_x, tri_y - 2),
                             (tri_x + 6, tri_y - 2),
                             (tri_x + 3, tri_y + 2)])
            self._spawn_dropdown_rects.append((rect, fid, which))

    def _active_plot_ids(self) -> list:
        """Return ``plot_fg_ids`` filtered for the active tab.

        Non-decision-makers (``self._ndm_ids``) only appear on the
        ``biomass`` tab; on every other tab they are filtered out. The
        same filter applies to their ``<id>_rnd`` baseline counterparts.
        On the action tabs (move/rest/eat) only DMs have meaningful data,
        so NDMs are likewise filtered out there.
        """
        active = self._tabs[self._active_tab]
        # NDMs ingår på biomass-tabben och på loss-tabbarna (de kan
        # förlora biomass både till predation och impacts även utan
        # eget beslutsfattande). Övriga tabbar filtreras till DMs.
        if (active in ("biomass", "predation", "starvation", "impacts")
                or not self._ndm_ids):
            return list(self.plot_fg_ids)
        out = []
        for fid in self.plot_fg_ids:
            base = fid[:-4] if fid.endswith("_rnd") else fid
            if base in self._ndm_ids:
                continue
            out.append(fid)
        return out

    def _save_inference_plot_html(self) -> None:
        """Bygger en interaktiv HTML-plot i samma format som
        ``<run_dir>/plots.html`` (från ``tools/biomass_html.py``) men från
        de per-tick-serier som live-viz:en har buffrat under den senaste
        inference-rolloutens gång. Öppnar en Tk-fildialog med default-
        filnamnet ``inferenceplot.html``.

        Enhetskonvertering (viz-serier -> jsonl-schema):
          * ``biomass`` / ``energy``: viz lagrar 100·ratio (procent);
            skrivs som ``ratio`` / ``energy_ratio`` i 0..1.
          * ``move`` / ``rest`` / ``eat``: viz lagrar procent; skrivs
            direkt som ``move_frac`` / ``rest_frac`` / ``eat_frac`` i %.
          * ``predation`` / ``starvation`` / ``impacts``: viz lagrar %;
            skrivs som ``loss_breakdown[fid][<cause>]`` i fraktion 0..1
            (``_extract_field`` multiplicerar med 100 vid rendering).
        """
        # Bygg tick-indexerad union av alla stegkoordinater som finns i
        # någon serie. Varje record motsvarar en tick.
        series = self._series
        all_steps: set = set()
        for tab in ("biomass", "energy", "move", "rest", "eat",
                    "predation", "starvation", "impacts"):
            if tab not in series:
                continue
            for fid, buf in series[tab].items():
                for step, _v in buf:
                    all_steps.add(int(step))
        if not all_steps:
            self._save_plot_button_flash_msg = "No data to save"
            import time as _t
            self._save_plot_button_flash_until = _t.time() + 2.5
            return
        steps_sorted = sorted(all_steps)

        # Per-(tab, fid) dict: step -> value, för snabb slagning.
        def _index(tab: str) -> Dict[str, Dict[int, float]]:
            out: Dict[str, Dict[int, float]] = {}
            if tab not in series:
                return out
            for fid, buf in series[tab].items():
                d: Dict[int, float] = {}
                for step, v in buf:
                    d[int(step)] = float(v)
                out[fid] = d
            return out

        bio = _index("biomass")
        eng = _index("energy")
        mv = _index("move")
        rs = _index("rest")
        et = _index("eat")
        pr = _index("predation")
        st = _index("starvation")
        im = _index("impacts")

        records = []
        for step in steps_sorted:
            rec: dict = {"iter": int(step)}
            ratio: Dict[str, float] = {}
            energy_ratio: Dict[str, float] = {}
            move_frac: Dict[str, float] = {}
            rest_frac: Dict[str, float] = {}
            eat_frac: Dict[str, float] = {}
            loss_breakdown: Dict[str, Dict[str, float]] = {}
            for fid, d in bio.items():
                if step in d:
                    ratio[fid] = d[step] / 100.0
            for fid, d in eng.items():
                if step in d:
                    energy_ratio[fid] = d[step] / 100.0
            for fid, d in mv.items():
                if step in d:
                    move_frac[fid] = d[step]
            for fid, d in rs.items():
                if step in d:
                    rest_frac[fid] = d[step]
            for fid, d in et.items():
                if step in d:
                    eat_frac[fid] = d[step]
            all_fids = set(pr) | set(st) | set(im)
            for fid in all_fids:
                lp = pr.get(fid, {}).get(step)
                ls = st.get(fid, {}).get(step)
                li = im.get(fid, {}).get(step)
                if lp is None and ls is None and li is None:
                    continue
                loss_breakdown[fid] = {
                    "predation":  (lp / 100.0) if lp is not None else 0.0,
                    "starvation": (ls / 100.0) if ls is not None else 0.0,
                    "impact":     (li / 100.0) if li is not None else 0.0,
                }
            if ratio:
                rec["ratio"] = ratio
                # OBS: ``log10_ratio`` (reward-fliken) och ``b0``/``bh``
                # utelämnas medvetet i inference-läget — inferens har
                # ingen reward-signal, så reward-fliken ska inte dyka
                # upp i den sparade HTML:en. FG-ordning härleds istället
                # från ``ratio`` (``_build_html`` fallback:ar dit).
            if energy_ratio:
                rec["energy_ratio"] = energy_ratio
            if move_frac:
                rec["move_frac"] = move_frac
            if rest_frac:
                rec["rest_frac"] = rest_frac
            if eat_frac:
                rec["eat_frac"] = eat_frac
            if loss_breakdown:
                rec["loss_breakdown"] = loss_breakdown
            records.append(rec)

        # Fildialog (Tk). Kör i samma tråd; pygame-fönstret pausar under
        # dialogen, vilket är acceptabelt eftersom knappen bara går att
        # klicka på när ingen inspelning pågår.
        default_name = "inferenceplot.html"
        out_path = self._ask_save_path(default_name)
        if not out_path:
            return
        # Ladda tools/biomass_html som modul (samma trick som train.py).
        import importlib.util, os
        tool_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "tools", "biomass_html.py")
        tool_path = os.path.normpath(tool_path)
        spec = importlib.util.spec_from_file_location(
            "_biomass_html_inference", tool_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {tool_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        run_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        # ``mode="inference"`` byter flik-titlar/sidhuvud/x-axel-etikett
        # från train-lägets "mean … per DM" / "sample index" till
        # per-tick-semantik ("… per-tick", "tick") som matchar live-vizen.
        html = mod._build_html(run_dir, records, mode="inference")
        with open(out_path, "w") as f:
            f.write(html)
        import time as _t
        self._save_plot_button_flash_msg = f"Saved: {os.path.basename(out_path)}"
        self._save_plot_button_flash_until = _t.time() + 3.0

    def _ask_save_path(self, default_name: str) -> str:
        """Öppnar en Tk ``asksaveasfilename``-dialog. Returnerar tom
        sträng om användaren avbryter eller om Tk inte är tillgängligt."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            self._log_once(f"tkinter unavailable: {e!r}")
            return ""
        root = tk.Tk()
        try:
            root.withdraw()
            root.update_idletasks()
            # ``initialdir`` sätts från ``_save_dir`` (typiskt
            # ``results/<run-name>/`` via ``set_save_dir`` från
            # inference.py). Om katalogen inte finns eller inte satts
            # låter vi Tk välja sin egen default (CWD).
            initial_dir = ""
            if self._save_dir:
                import os as _os
                if _os.path.isdir(self._save_dir):
                    initial_dir = self._save_dir
            kwargs = dict(
                parent=root,
                title="Save inference plot as HTML",
                initialfile=default_name,
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
            )
            if initial_dir:
                kwargs["initialdir"] = initial_dir
            path = filedialog.asksaveasfilename(**kwargs)
        finally:
            try:
                root.destroy()
            except Exception:
                pass
        return path or ""

    def _draw_plot(self) -> None:
        pg = self._pg
        x, y, w, h = self._plot_rect
        # Background.
        pg.draw.rect(self._screen, (24, 24, 30), (x, y, w, h))
        pg.draw.rect(self._screen, (60, 60, 70), (x, y, w, h), 1)

        # Choose data source from the active tab. Each tab has its own
        # per-FG buffer; callers push values via update_series(tab, ...).
        active = self._tabs[self._active_tab]
        buffers = self._series[active]
        base_label = self._tab_labels.get(active, active)
        if active == "reward" and self._log_plot:
            ylabel = "reward (log10 ratio)"
        else:
            ylabel = base_label

        # ---- Tab headers (clickable) -------------------------------------
        # Alla flikar ritas på en och samma rad. Plot-panelens bredd
        # (``plot_w`` i ``__init__``) är dimensionerad så att hela
        # tab-strippen ryms inom det legend-fria området, oavsett antal
        # flikar (8 i inference-läget, 9 i train-läget).
        row_h = 16
        tab_pad_x = 6
        tab_x = x + tab_pad_x
        tab_y = y + 2
        self._tab_rects = []
        strip_h = row_h + 2
        pg.draw.rect(self._screen, (24, 24, 30), (x, y, w, strip_h), 0)
        for i, tab in enumerate(self._tabs):
            is_active = (i == self._active_tab)
            col = (240, 240, 250) if is_active else (140, 140, 150)
            surf = self._font.render(tab, True, col)
            tw = surf.get_width() + 10
            rect = (tab_x, tab_y, tw, 14)
            if is_active:
                pg.draw.rect(self._screen, (45, 45, 60), rect)
            pg.draw.rect(self._screen, (60, 60, 70), rect, 1)
            self._screen.blit(surf, (tab_x + 5, tab_y + 1))
            self._tab_rects.append((rect, i))
            tab_x += tw + 4

        # Title (under tab strip).
        title_y = y + strip_h + 2
        tsurf = self._font.render(ylabel, True, (200, 200, 210))
        self._screen.blit(tsurf, (x + 6, title_y))

        active_ids = self._active_plot_ids()

        # Plot area geometry — computed up-front so the legend is always
        # drawn (and clickable) even when there is no data to plot, e.g.
        # when every FG has been toggled off. Otherwise the legend would
        # disappear together with the lines and the user couldn't re-
        # enable any series without blindly guessing checkbox positions.
        plot_pad_l = 36
        plot_pad_r = getattr(self, "_legend_w", 110)  # room for legend
        plot_pad_t = strip_h + 20  # tab strip (variable rows) + ylabel line
        # Reservera ~14 px längst ned för horisontell scrollbar (inference).
        # I train-läget används inte scrollbaren; behåll ursprunglig padding.
        scroll_bar_h = 12 if str(self.mode).lower().startswith("infer") else 0
        # Save-knappen (inference) ritas under scrollbaren; reservera 22 px.
        save_btn_h = 22 if str(self.mode).lower().startswith("infer") else 0
        plot_pad_b = 16 + (scroll_bar_h + 4 if scroll_bar_h else 0) \
            + (save_btn_h + 4 if save_btn_h else 0)
        px0 = x + plot_pad_l
        py0 = y + plot_pad_t
        pw = w - plot_pad_l - plot_pad_r
        ph = h - plot_pad_t - plot_pad_b
        if pw <= 4 or ph <= 4:
            return

        # First pass: full data x-range across all enabled buffers.
        data_xmin = None
        data_xmax = None
        for fid in active_ids:
            if self._solo is not None and self._solo != fid:
                continue
            if not self._plot_enabled.get(fid, True):
                continue
            if buffers[fid]:
                bx0 = buffers[fid][0][0]
                bx1 = buffers[fid][-1][0]
                data_xmin = bx0 if data_xmin is None else min(data_xmin, bx0)
                data_xmax = bx1 if data_xmax is None else max(data_xmax, bx1)
        have_data = data_xmin is not None
        xmin = xmax = 0.0
        ymin = ymax = 0.0
        # Aktivera scrollbaren när data-intervallet överstiger fönstret
        # och läget är inference. Annars visar vi hela intervallet som förut.
        scroll_active = False
        if have_data and scroll_bar_h > 0:
            full_span = float(data_xmax - data_xmin)
            if full_span > float(self._plot_window_width):
                scroll_active = True

        if have_data:
            if scroll_active:
                win_w_x = float(self._plot_window_width)
                max_off = float(data_xmax) - win_w_x
                if self._plot_scroll_offset is None:
                    # Följ senaste data (samma beteende som förut / live).
                    off = max_off
                else:
                    off = float(self._plot_scroll_offset)
                off = max(float(data_xmin), min(max_off, off))
                # Om vi var i auto-läge, håll det (så under pågående record
                # följer fönstret senaste ticket automatiskt).
                if self._plot_scroll_offset is not None:
                    self._plot_scroll_offset = off
                xmin = off
                xmax = off + win_w_x
            else:
                xmin = float(data_xmin)
                xmax = float(data_xmax)
                if xmax - xmin < 1:
                    xmax = xmin + 1

            # Y-range: bara från punkter inom det synliga x-intervallet så
            # skalan följer det som faktiskt syns i fönstret.
            all_vals: list = []
            for fid in active_ids:
                if self._solo is not None and self._solo != fid:
                    continue
                if not self._plot_enabled.get(fid, True):
                    continue
                for s, v in buffers[fid]:
                    if xmin <= s <= xmax:
                        all_vals.append(v)
            if not all_vals:
                have_data = False
            else:
                arr = np.asarray(all_vals, dtype=np.float64)
                if self._log_plot:
                    arr = np.sign(arr) * np.log10(np.abs(arr) + 1e-12)
                ymin = float(arr.min())
                ymax = float(arr.max())
                if not np.isfinite(ymin) or not np.isfinite(ymax):
                    have_data = False
                elif ymax - ymin < 1e-9:
                    ymax = ymin + 1.0

        if have_data:
            # Y-axis tick labels (5 st: max, 3/4, mid, 1/4, min).
            for frac, val in (
                (0.0,  ymax),
                (0.25, ymin + 0.75 * (ymax - ymin)),
                (0.5,  (ymin + ymax) / 2),
                (0.75, ymin + 0.25 * (ymax - ymin)),
                (1.0,  ymin),
            ):
                yy = int(py0 + frac * ph)
                pg.draw.line(self._screen, (50, 50, 60),
                             (px0, yy), (px0 + pw, yy), 1)
                lab = self._font.render(f"{val:+.3g}", True, (160, 160, 170))
                self._screen.blit(lab, (x + 2, yy - 7))

            # Zero line if in range.
            if ymin < 0.0 < ymax:
                yy = int(py0 + (ymax - 0.0) / (ymax - ymin) * ph)
                pg.draw.line(self._screen, (90, 90, 110),
                             (px0, yy), (px0 + pw, yy), 1)

            # Plot lines. Klipp till synligt x-fönster med en padding-punkt
            # på varje sida så linjesegment som skär fönstrets kanter
            # fortfarande når hela vägen ut.
            for fid in active_ids:
                if self._solo is not None and self._solo != fid:
                    continue
                if not self._plot_enabled.get(fid, True):
                    continue
                buf = buffers[fid]
                if len(buf) < 2:
                    continue
                # Bygg indexlista av synliga punkter + en granne på varje
                # sida (för att linjen ska nå fönsterkanten).
                start_i = 0
                end_i = len(buf) - 1
                for i, (s, _v) in enumerate(buf):
                    if s >= xmin:
                        start_i = max(0, i - 1)
                        break
                for j in range(len(buf) - 1, -1, -1):
                    if buf[j][0] <= xmax:
                        end_i = min(len(buf) - 1, j + 1)
                        break
                pts = []
                for k in range(start_i, end_i + 1):
                    step, val = buf[k]
                    v = val
                    if self._log_plot:
                        v = float(np.sign(v) * np.log10(abs(v) + 1e-12))
                    fx = (step - xmin) / (xmax - xmin)
                    fy = (ymax - v) / (ymax - ymin)
                    xi = int(px0 + fx * pw)
                    yi = int(py0 + fy * ph)
                    # Klipp x till plot-arean så linjer inte spiller över.
                    if xi < px0:
                        xi = px0
                    elif xi > px0 + pw:
                        xi = px0 + pw
                    pts.append((xi, yi))
                if len(pts) < 2:
                    continue
                try:
                    pg.draw.aalines(self._screen, self._fg_colour[fid], False, pts)
                except Exception:
                    pg.draw.lines(self._screen, self._fg_colour[fid], False, pts, 1)

        # ---- Horisontell scrollbar (inference) ---------------------------
        # Ritas alltid när scroll_bar_h > 0 (reserverar utrymme), men blir
        # bara interaktiv/synlig thumb när data-intervallet överstiger
        # fönstret (scroll_active). ``_plot_scroll_track_rect`` sätts alltid
        # så hit-tests kan avgöra klick även vid mindre data.
        self._plot_scroll_track_rect = None
        self._plot_scroll_thumb_rect = None
        # Cacha data-span för scrollbar-drag (används av
        # ``_plot_scroll_offset_from_x`` mellan render-anrop).
        if have_data and scroll_active:
            self._plot_scroll_span_cache = (float(data_xmin), float(data_xmax))
        else:
            self._plot_scroll_span_cache = None
        if scroll_bar_h > 0 and pw > 4:
            sb_y = py0 + ph + 4
            sb_h = scroll_bar_h
            track = (px0, sb_y, pw, sb_h)
            pg.draw.rect(self._screen, (30, 30, 38), track)
            pg.draw.rect(self._screen, (60, 60, 70), track, 1)
            if scroll_active:
                full_span = float(data_xmax - data_xmin)
                win_w_x = float(self._plot_window_width)
                thumb_w = max(20, int(pw * (win_w_x / full_span)))
                # Position i track baserat på offset.
                off_frac = (float(xmin) - float(data_xmin)) / max(
                    1e-9, full_span - win_w_x)
                off_frac = max(0.0, min(1.0, off_frac))
                thumb_x = int(px0 + off_frac * (pw - thumb_w))
                thumb_rect = (thumb_x, sb_y + 1, thumb_w, sb_h - 2)
                pg.draw.rect(self._screen, (110, 110, 130), thumb_rect)
                pg.draw.rect(self._screen, (170, 170, 190), thumb_rect, 1)
                self._plot_scroll_thumb_rect = thumb_rect
                self._plot_scroll_track_rect = track
            else:
                # Ingen scrollning möjlig — rita en tunn markör som fyller
                # hela tracken (visar att allt data är synligt).
                inner = (px0 + 1, sb_y + 1, pw - 2, sb_h - 2)
                pg.draw.rect(self._screen, (55, 55, 65), inner)

        # ---- Save-plot-HTML-knapp (inference) ---------------------------
        # Ritas under scrollbaren. Enabled endast när inspelning INTE
        # pågår OCH det finns data att spara. Vid klick öppnas en
        # Tk-fildialog med default-filnamn ``inferenceplot.html``.
        self._save_plot_button_rect = None
        if save_btn_h > 0 and pw > 4:
            btn_y = py0 + ph + 4 + (scroll_bar_h + 4 if scroll_bar_h else 0)
            btn_w = min(180, pw)
            btn_x = px0
            btn_rect = (btn_x, btn_y, btn_w, save_btn_h)
            enabled = bool(have_data) and not self._recording
            bg = (50, 70, 55) if enabled else (35, 35, 42)
            border = (100, 160, 120) if enabled else (70, 70, 80)
            pg.draw.rect(self._screen, bg, btn_rect)
            pg.draw.rect(self._screen, border, btn_rect, 1)
            # Text: normal etikett, eller flash-meddelande efter save.
            import time as _t
            now = _t.time()
            if now < self._save_plot_button_flash_until \
                    and self._save_plot_button_flash_msg:
                label = self._save_plot_button_flash_msg
                col = (220, 230, 210)
            else:
                label = "Save plot as HTML..."
                col = (220, 230, 220) if enabled else (120, 120, 130)
            surf = self._font.render(label, True, col)
            tx = btn_x + max(4, (btn_w - surf.get_width()) // 2)
            ty = btn_y + (save_btn_h - surf.get_height()) // 2
            self._screen.blit(surf, (tx, ty))
            if enabled:
                self._save_plot_button_rect = btn_rect

        # Legend with a per-FG checkbox that toggles plotting across all
        # tabs. Drawn unconditionally so users can always re-enable a
        # series that they previously toggled off (even when *all* series
        # are currently disabled and there is nothing to plot).
        lx = px0 + pw + 8
        ly = py0
        self._legend_rects = []
        for fid in active_ids:
            col = self._fg_colour[fid]
            faded = (self._solo is not None and self._solo != fid)
            enabled = self._plot_enabled.get(fid, True)
            c = tuple(int(v * (0.3 if faded or not enabled else 1.0)) for v in col)
            # Checkbox.
            cb_rect = (lx, ly + 3, 12, 12)
            pg.draw.rect(self._screen, (200, 200, 210), cb_rect, 1)
            if enabled:
                pg.draw.line(self._screen, (220, 220, 230),
                             (lx + 2, ly + 9), (lx + 5, ly + 12), 2)
                pg.draw.line(self._screen, (220, 220, 230),
                             (lx + 5, ly + 12), (lx + 11, ly + 4), 2)
            self._legend_rects.append((cb_rect, fid))
            # Colour swatch.
            pg.draw.rect(self._screen, c, (lx + 16, ly + 4, 10, 10))
            lab = self._font.render(self._display_name(fid), True, c)
            self._screen.blit(lab, (lx + 30, ly))
            ly += 14
            if ly > y + h - 14:
                break


def _fg_colour_for(idx: int, n: int) -> tuple:
    """Distinct-ish RGB colour per FG, deterministic by index."""
    # Golden-ratio hue spacing for visually well-separated colours.
    import colorsys
    hue = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)
