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
        self._tab_labels = {
            "reward": "reward",
            "biomass": "avg biomass (% of start)",
            "energy": "avg energy (% of start)",
            "move": "move action (%)",
            "rest": "rest action (%)",
            "eat": "eat action (%)",
            "predation": "predation share of total loss (%)",
            "starvation": "starvation share of total loss (%)",
            "impacts": "impact share of total loss (%)",
        }
        self._active_tab = 0
        # Per-FG enable flag for plot panel (checkbox state). Toggled via
        # legend click; applies globally across all plot tabs.
        self._plot_enabled: Dict[str, bool] = {
            fid: True for fid in (list(self.fg_ids) + list(self._extra_plot_ids))
        }
        self._legend_rects: list = []
        _all_series_ids = list(self.fg_ids) + [
            eid for eid in self._extra_plot_ids if eid not in self.fg_ids
        ]
        self._series: Dict[str, Dict[str, deque]] = {
            tab: {fid: deque(maxlen=self.reward_window) for fid in _all_series_ids}
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
        # by one blank line beneath the last info line. NDM panels leave
        # the three action/loss/diet lines blank to keep the heatmap
        # origin aligned across panels.
        title_h = 84
        cbar_h = 16  # colorbar strip (gradient + 0/max labels)
        pad = 8
        self._cbar_h = cbar_h
        self._panel_w = hm_w + 2 * pad
        self._panel_h = hm_h + title_h + cbar_h + 2 * pad
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
        plot_w = max(420, heatmap_block_w // 2 + self._legend_w)
        plot_h = heatmap_block_h
        status_h = 26
        log_h = 0
        self._win_w = heatmap_block_w + plot_w + pad
        self._win_h = status_h + heatmap_block_h + pad + log_h
        self._heatmap_origin = (0, status_h)
        self._plot_rect = (heatmap_block_w + pad // 2,
                           status_h,
                           plot_w - pad // 2,
                           plot_h)
        self._status_rect = (0, 0, self._win_w, status_h)

        # ---- Init pygame --------------------------------------------------
        try:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self._screen = pygame.display.set_mode((self._win_w, self._win_h))
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
            # While paused, keep the window responsive without spinning.
            while self._paused and not self._quit:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self._quit = True
                    elif event.type == pg.KEYDOWN:
                        self._handle_key(event.key)
                    elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)
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

    def _handle_click(self, pos) -> None:
        try:
            mx, my = pos
        except Exception:
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
        self._draw_heatmaps()
        self._draw_plot()
        self._pg.display.flip()

    def _draw_status_bar(self) -> None:
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
        self._screen.blit(surf, (x + 6, y + 5))

    def _draw_heatmaps(self) -> None:
        ox, oy = self._heatmap_origin
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
        # three blank for NDMs), följt av en tom rad så det blir luft
        # mellan sista info-raden och heatmapen. title_h måste rymma alla
        # fem rader + den tomma raden.
        title_h = 84
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
        info_txt = f"B0 = {_fmt_b(b0)}  B = {_fmt_b(total)}"
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
        pg.draw.rect(self._screen, (24, 24, 30), (x, y, w, 18))
        tab_x = x + 6
        self._tab_rects = []
        for i, tab in enumerate(self._tabs):
            label = tab
            is_active = (i == self._active_tab)
            col = (240, 240, 250) if is_active else (140, 140, 150)
            surf = self._font.render(label, True, col)
            tw = surf.get_width() + 10
            rect = (tab_x, y + 2, tw, 14)
            if is_active:
                pg.draw.rect(self._screen, (45, 45, 60), rect)
            pg.draw.rect(self._screen, (60, 60, 70), rect, 1)
            self._screen.blit(surf, (tab_x + 5, y + 3))
            self._tab_rects.append((rect, i))
            tab_x += tw + 4

        # Title (under tab strip).
        tsurf = self._font.render(ylabel, True, (200, 200, 210))
        self._screen.blit(tsurf, (x + 6, y + 20))

        active_ids = self._active_plot_ids()

        # Plot area geometry — computed up-front so the legend is always
        # drawn (and clickable) even when there is no data to plot, e.g.
        # when every FG has been toggled off. Otherwise the legend would
        # disappear together with the lines and the user couldn't re-
        # enable any series without blindly guessing checkbox positions.
        plot_pad_l = 36
        plot_pad_r = getattr(self, "_legend_w", 110)  # room for legend
        plot_pad_t = 38  # tab strip (18) + ylabel line
        plot_pad_b = 16
        px0 = x + plot_pad_l
        py0 = y + plot_pad_t
        pw = w - plot_pad_l - plot_pad_r
        ph = h - plot_pad_t - plot_pad_b
        if pw <= 4 or ph <= 4:
            return

        # Determine y-range across all buffers (only from enabled series).
        all_vals: list = []
        for fid in active_ids:
            if self._solo is not None and self._solo != fid:
                continue
            if not self._plot_enabled.get(fid, True):
                continue
            for _, v in buffers[fid]:
                all_vals.append(v)
        have_data = bool(all_vals)
        ymin = ymax = xmin = xmax = 0.0
        if have_data:
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
            # Determine x-range (use sample index per buffer; aligned by step).
            xmins, xmaxs = [], []
            for fid in active_ids:
                if self._solo is not None and self._solo != fid:
                    continue
                if not self._plot_enabled.get(fid, True):
                    continue
                if buffers[fid]:
                    xmins.append(buffers[fid][0][0])
                    xmaxs.append(buffers[fid][-1][0])
            if xmins:
                xmin = min(xmins)
                xmax = max(xmaxs)
                if xmax - xmin < 1:
                    xmax = xmin + 1
            else:
                have_data = False

        if have_data:
            # Y-axis tick labels (min, mid, max).
            for frac, val in ((0.0, ymax), (0.5, (ymin + ymax) / 2), (1.0, ymin)):
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

            # Plot lines.
            for fid in active_ids:
                if self._solo is not None and self._solo != fid:
                    continue
                if not self._plot_enabled.get(fid, True):
                    continue
                buf = buffers[fid]
                if len(buf) < 2:
                    continue
                pts = []
                for step, val in buf:
                    v = val
                    if self._log_plot:
                        v = float(np.sign(v) * np.log10(abs(v) + 1e-12))
                    fx = (step - xmin) / (xmax - xmin)
                    fy = (ymax - v) / (ymax - ymin)
                    pts.append((int(px0 + fx * pw),
                                int(py0 + fy * ph)))
                try:
                    pg.draw.aalines(self._screen, self._fg_colour[fid], False, pts)
                except Exception:
                    pg.draw.lines(self._screen, self._fg_colour[fid], False, pts, 1)

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
