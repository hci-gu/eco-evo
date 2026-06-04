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
    def update_status(self, *a, **kw): pass
    def pump_events(self): return True
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

        # Per-FG rolling buffers: (step, value) for the line plot.
        self._reward_buf: Dict[str, deque] = {
            fid: deque(maxlen=self.reward_window) for fid in self.fg_ids
        }
        # Latest biomass arrays + totals for heatmap rendering.
        self._biomass: Dict[str, np.ndarray] = {}
        self._totals: Dict[str, float] = {fid: 0.0 for fid in self.fg_ids}
        # Initial (rollout start) totals per FG; captured on tick==0 so that
        # the title can display 'B0=... B=...' for context. Reset each new
        # rollout (train probe runs fresh per ARS-iter, inference is one run).
        self._b0: Dict[str, float] = {fid: 0.0 for fid in self.fg_ids}
        # Stable colour per FG (used for the plot legend).
        self._fg_colour: Dict[str, tuple] = {
            fid: _fg_colour_for(i, len(self.fg_ids))
            for i, fid in enumerate(self.fg_ids)
        }

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
        title_h = 32  # two lines: FG id + 'B0=… B=…'
        cbar_h = 16  # colorbar strip (gradient + 0/max labels)
        pad = 8
        self._cbar_h = cbar_h
        self._panel_w = hm_w + 2 * pad
        self._panel_h = hm_h + title_h + cbar_h + 2 * pad
        heatmap_block_w = self._cols * self._panel_w
        heatmap_block_h = self._rows * self._panel_h
        plot_w = max(360, heatmap_block_w // 2)
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
                self._biomass[fid] = arr
                total = float(arr.sum())
                self._totals[fid] = total
                if new_rollout:
                    self._b0[fid] = total
            if extra:
                self._status.update(extra)
            self._maybe_render()
        except Exception as e:
            self._log_once(f"update_biomass failed: {e!r}")

    def update_reward(self, fg_id: str, value: float, step: int) -> None:
        if not self.enabled:
            return
        try:
            if fg_id not in self._reward_buf:
                return
            self._reward_buf[fg_id].append((int(step), float(value)))
        except Exception as e:
            self._log_once(f"update_reward failed: {e!r}")

    def update_status(self, **kw) -> None:
        if not self.enabled:
            return
        self._status.update(kw)

    def pump_events(self) -> bool:
        """Process pygame events; return False if user asked to quit viz."""
        if not self.enabled:
            return True
        try:
            pg = self._pg
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self._quit = True
                elif event.type == pg.KEYDOWN:
                    self._handle_key(event.key)
            # While paused, keep the window responsive without spinning.
            while self._paused and not self._quit:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self._quit = True
                    elif event.type == pg.KEYDOWN:
                        self._handle_key(event.key)
                self._render_full()
                pg.time.wait(50)
            return not self._quit
        except Exception as e:
            self._log_once(f"pump_events failed: {e!r}")
            return True

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
        elif pg.K_1 <= key <= pg.K_9:
            idx = key - pg.K_1
            if idx < len(self.fg_ids):
                target = self.fg_ids[idx]
                self._solo = None if self._solo == target else target

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
        parts = [f"mode={self.mode}", f"tick={self._tick}"]
        for k in ("gen", "iter", "T"):
            if k in self._status:
                v = self._status[k]
                if isinstance(v, float):
                    parts.append(f"{k}={v:.3f}")
                else:
                    parts.append(f"{k}={v}")
        if self._paused:
            parts.append("[PAUSED]")
        if self._log_heatmap:
            parts.append("hm:log")
        if self._log_plot:
            parts.append("plot:log")
        if self._solo:
            parts.append(f"solo={self._solo}")
        parts.append(f"grid={self.grid_w}x{self.grid_h}")
        parts.append(f"fps={self._fps_value:4.1f}")
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

    def _draw_one_heatmap(self, fid: str, px: int, py: int) -> None:
        pg = self._pg
        pad = 4
        title_h = 32
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
        info_txt = (f"B0={b0:,.0f}  B={total:,.0f}").replace(",", " ")
        info_col = (180, 180, 190) if not dim else (90, 90, 95)
        info_surf = self._font.render(info_txt, True, info_col)
        self._screen.blit(info_surf, (px + pad, py + 1 + name_surf.get_height()))

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

    def _draw_plot(self) -> None:
        pg = self._pg
        x, y, w, h = self._plot_rect
        # Background.
        pg.draw.rect(self._screen, (24, 24, 30), (x, y, w, h))
        pg.draw.rect(self._screen, (60, 60, 70), (x, y, w, h), 1)

        # Choose data source per mode.
        if self.mode == "train":
            buffers = self._reward_buf
            ylabel = "reward (log10 ratio)" if self._log_plot else "reward"
        else:
            # Inference: synthesise from biomass totals over recent ticks.
            # Reuse _reward_buf as a generic time series; train uses reward,
            # inference can push biomass via update_reward(fid, total, t).
            buffers = self._reward_buf
            ylabel = "total biomass"

        # Title.
        tsurf = self._font.render(ylabel, True, (200, 200, 210))
        self._screen.blit(tsurf, (x + 6, y + 4))

        # Determine y-range across all buffers.
        all_vals: list = []
        for fid in self.plot_fg_ids:
            if self._solo is not None and self._solo != fid:
                continue
            for _, v in buffers[fid]:
                all_vals.append(v)
        if not all_vals:
            return
        arr = np.asarray(all_vals, dtype=np.float64)
        if self._log_plot:
            arr = np.sign(arr) * np.log10(np.abs(arr) + 1e-12)
        ymin = float(arr.min())
        ymax = float(arr.max())
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return
        if ymax - ymin < 1e-9:
            ymax = ymin + 1.0

        # Determine x-range (use sample index per buffer; aligned by step).
        xmins, xmaxs = [], []
        for fid in self.plot_fg_ids:
            if self._solo is not None and self._solo != fid:
                continue
            if buffers[fid]:
                xmins.append(buffers[fid][0][0])
                xmaxs.append(buffers[fid][-1][0])
        if not xmins:
            return
        xmin = min(xmins)
        xmax = max(xmaxs)
        if xmax - xmin < 1:
            xmax = xmin + 1

        plot_pad_l = 36
        plot_pad_r = 110  # room for legend
        plot_pad_t = 22
        plot_pad_b = 16
        px0 = x + plot_pad_l
        py0 = y + plot_pad_t
        pw = w - plot_pad_l - plot_pad_r
        ph = h - plot_pad_t - plot_pad_b
        if pw <= 4 or ph <= 4:
            return

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
        for fid in self.plot_fg_ids:
            if self._solo is not None and self._solo != fid:
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

        # Legend.
        lx = px0 + pw + 8
        ly = py0
        for fid in self.plot_fg_ids:
            col = self._fg_colour[fid]
            faded = (self._solo is not None and self._solo != fid)
            c = tuple(int(v * (0.3 if faded else 1.0)) for v in col)
            pg.draw.rect(self._screen, c, (lx, ly + 4, 10, 10))
            lab = self._font.render(self._display_name(fid)[:18], True, c)
            self._screen.blit(lab, (lx + 14, ly))
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
