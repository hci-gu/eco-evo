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
    n       toggle rolling heatmap normalisation (per-frame maximum)
            instead of the fixed rollout-start maximum. Also a button
            in the status bar, "norm: fixed / rolling".
    r       toggle log-scale on the reward/biomass plot y-axis
    p       fold the plot panel in under the heatmaps / bring it back
            (also the button at the top of the splitter)
    wheel   over the heatmaps: scroll them when the grid is taller than
            the window (also the scrollbar at their right edge;
            MARELD_VIZ_MAX_HEIGHT=<px> overrides the start height cap)

The window is resizable and can be maximised. The vertical splitter
between the heatmaps and the plot is dragged to share the width between
them; the heatmaps grow with their share.
    q / esc quit visualiser (training continues)
    1..9    solo-focus one FG (press same digit again to clear)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from lib.viz import i18n
from lib.world import daylight
from lib.world.tick_time import (
    DAYS_PER_YEAR, HOURS_PER_DAY, LIBRARY_TICK_HOURS, resolve_tick_hours,
)


_VIRIDIS_256: Optional[np.ndarray] = None

# Width of the daylight frame around each heatmap (section 142).
_LIGHT_FRAME_PX = 2

# Viewer preferences kept between runs (section 213): today only the
# display language. A small JSON file next to this module, ignored by git
# like fgconfig/recent_projects.txt; MARELD_VIEWER_SETTINGS points it
# elsewhere (the test suite gives every test its own file).
_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "viewer_settings.json")


def _settings_path() -> str:
    return os.environ.get("MARELD_VIEWER_SETTINGS") or _SETTINGS_FILE


def _load_settings() -> dict:
    """The saved preferences, or {} when there are none or they are unreadable."""
    try:
        with open(_settings_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_settings(**changes) -> None:
    """Merge ``changes`` into the saved preferences (never fatal)."""
    data = _load_settings()
    data.update(changes)
    path = _settings_path()
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except OSError as e:
        print(f"[viz] Could not save viewer settings to {path}: {e}")

# Calendar band under the inference plot (section 155): a month row and
# a clock row, profiler-style blocks with the label inside when it fits.
# The bound that decides a group's viability margin on the progress
# tab (training_progress.probe_viability, section 223.18).
_VIABILITY_BINDING = {"floor": "floor", "ceiling": "ceiling",
                      "drift": "drift", "extinct": "extinct"}
# training_progress.FLOOR_NOTE_MAX_YEARS (that module pulls in torch).
_FLOOR_NOTE_MAX_YEARS = 50.0
# Tabs fed from the probe history instead of per-update buffers.
_PROBE_TABS = ("progress", "viability")
# Display names of tabs whose internal id says something else (section
# 223.21): the "progress" tab shows one long run of the frozen policies,
# not training progress. The id, <run>/progress and probe.jsonl stay.
_TAB_NAMES = {"progress": "long run"}
# The viability tab's fixed axis (section 223.20): margins are <= 1 by
# construction; anything below -1 (and an extinct group) is drawn at -1.
_VIABILITY_AXIS = (-1.0, 1.0)
# Tabs the R key's log axis never applies to.
_LINEAR_TABS = ("biomass", "viability")

_CAL_ROW_H = 14
_CAL_BAND_H = 2 * _CAL_ROW_H + 3
# Clock block lengths tried, finest first; the finest one whose block is
# at least _CAL_MIN_BLOCK_PX wide is drawn (never finer than a tick).
_CAL_CLOCK_STEPS_H = (1, 2, 3, 6, 12, 24)
_CAL_MIN_BLOCK_PX = 18
# Below this a day block is not outlined; the light shading remains.
_CAL_MIN_DAY_PX = 8
# Month blocks are tinted by season (Dec-Feb, Mar-May, Jun-Aug, Sep-Nov).
_CAL_SEASON_RGB = ((62, 84, 128), (66, 112, 72), (132, 114, 52),
                   (134, 84, 48))
# Clock row shading: the mean light level F of the column's ticks.
_CAL_NIGHT_RGB = np.array((26, 30, 56), dtype=np.float64)
_CAL_DAY_RGB = np.array((112, 100, 58), dtype=np.float64)
_HOURS_PER_YEAR = DAYS_PER_YEAR * HOURS_PER_DAY
_MONTH_START_H = np.cumsum((0,) + daylight.MONTH_DAYS[:-1]) * HOURS_PER_DAY


def _plot_x_hours(cfg: Mapping, x):
    """Hours after 1 January 00:00 of the start year at inference plot x.

    The series point at step ``s`` is the state after tick ``s`` ran,
    i.e. at the start of tick ``s + 1`` - the instant the status bar
    describes when it shows ``tick = s + 1``. Not wrapped at a year, so
    a run that crosses New Year keeps increasing.
    """
    hours = resolve_tick_hours(cfg["tick_hours"])
    start = int(cfg.get("start_tick", 0))
    return (start + np.asarray(x, dtype=np.float64) + 1.0) * hours


def _date_parts(hours: float) -> tuple:
    """``(month, day_of_month, hour)`` of an instant in year-hours.

    ``month`` is 0-based, the day of the month 1-based; the year-hours
    may be unwrapped (past the first New Year).
    """
    h = int(np.floor(hours)) % _HOURS_PER_YEAR
    day0, hour = divmod(h, HOURS_PER_DAY)
    month = daylight.month_of_day(day0)
    dom = day0 - int(_MONTH_START_H[month]) // HOURS_PER_DAY + 1
    return month, dom, hour


def _calendar_label(hours: float, lang: str = i18n.DEFAULT) -> str:
    """``"Mar 14 06:00"`` for an instant in (unwrapped) year-hours."""
    month, dom, hour = _date_parts(hours)
    return f"{i18n.month_name(lang, month)[:3]} {dom} {hour:02d}:00"


def _month_blocks(h0: float, h1: float) -> list:
    """``(start_h, end_h, month)`` of every month overlapping [h0, h1]."""
    out = []
    for year in range(int(h0 // _HOURS_PER_YEAR),
                      int(h1 // _HOURS_PER_YEAR) + 1):
        base = year * _HOURS_PER_YEAR
        for m, days in enumerate(daylight.MONTH_DAYS):
            start = base + int(_MONTH_START_H[m])
            end = start + days * HOURS_PER_DAY
            if end > h0 and start < h1:
                out.append((start, end, m))
    return out


def _clock_step_hours(tick_hours: int, px_per_hour: float) -> Optional[int]:
    """Clock block length to draw, or None when even a day is too narrow."""
    for step in _CAL_CLOCK_STEPS_H:
        if step >= tick_hours and step * px_per_hour >= _CAL_MIN_BLOCK_PX:
            return step
    return None


def _column_light(cfg: Mapping, h_edges: np.ndarray) -> np.ndarray:
    """Mean light F between consecutive year-hour edges, exactly.

    F is the per-tick light level (the observation channel); the mean
    over a column is taken through the cumulative sum of the year's
    schedule, so a column covering many ticks shows their average
    rather than an aliased sample of one time of day.
    """
    hours = resolve_tick_hours(cfg["tick_hours"])
    sched = daylight.light_schedule(cfg["latitude_deg"], hours)
    n = len(sched)
    cum = np.concatenate(([0.0], np.cumsum(sched, dtype=np.float64)))
    ticks = np.asarray(h_edges, dtype=np.float64) / hours
    whole = np.floor(ticks / n)
    rest = ticks - whole * n
    c = whole * cum[-1] + np.interp(rest, np.arange(n + 1), cum)
    span = np.diff(ticks)
    return np.where(span > 0, np.diff(c) / np.maximum(span, 1e-12), 0.0)


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


# ---------------------------------------------------------------------------
# Rollout-frame compression.
#
# Varje "frame" i ``_pending_rollout`` / ``_current_rollout`` dominerar RAM-
# fotavtrycket via sina ``biomass``-arrays (en per FG, form H×W). För en
# typisk rollout (t.ex. 5000 frames × 6 FG × 128×128 × float64 ≈ 3.7 GB) är
# det värt att både (1) casta ner till float32 och (2) lossless-komprimera
# arrays som lagras i historien. Live-state (``self._biomass``) rörs inte;
# endast frame-snapshots komprimeras och dekomprimeras on demand vid
# scrubbing/playback via ``_swap_in_frame``.
#
# Codec-val styrs av env-flaggan ``ECO_ROLLOUT_COMPRESS`` (``zstd`` (default
# om tillgängligt), ``lz4``, eller ``none``). Om det önskade biblioteket
# saknas faller vi tyst tillbaka till nästa alternativ.
# ---------------------------------------------------------------------------


def _select_rollout_compressor() -> Optional[tuple]:
    """Välj (codec_name, compress_fn, decompress_fn) enligt env, eller None.

    ``compress_fn(bytes) -> bytes`` och ``decompress_fn(bytes) -> bytes``.
    Returnerar ``None`` om användaren begärt ``none`` eller inget lib finns.
    """
    want = os.environ.get("ECO_ROLLOUT_COMPRESS", "zstd").strip().lower()
    if want in ("none", "off", "0", "false", ""):
        return None
    candidates = []
    if want == "zstd":
        candidates = ["zstd", "lz4"]
    elif want == "lz4":
        candidates = ["lz4", "zstd"]
    else:
        candidates = [want]
    for codec in candidates:
        if codec == "zstd":
            try:
                import zstandard as zstd  # type: ignore
                cctx = zstd.ZstdCompressor(level=3)
                dctx = zstd.ZstdDecompressor()
                return ("zstd", cctx.compress, dctx.decompress)
            except Exception:
                continue
        if codec == "lz4":
            try:
                import lz4.block as _lz4b  # type: ignore
                return ("lz4", _lz4b.compress, _lz4b.decompress)
            except Exception:
                try:
                    import lz4.frame as _lz4f  # type: ignore
                    return ("lz4", _lz4f.compress, _lz4f.decompress)
                except Exception:
                    continue
    return None


_ROLLOUT_COMPRESSOR: Optional[tuple] = _select_rollout_compressor()


class _CompressedArray:
    """En lat, lossless-komprimerad numpy-array-container.

    Lagrar komprimerade bytes + shape/dtype och rekonstruerar arrayen först
    när ``.array`` läses. Storleken i minnet reduceras typiskt 5–20× för
    glesa/spatialt korrelerade biomassa-fält. Trådsäker för läsning
    (dekompression skapar en ny buffer per anrop, ingen delad muterbar
    state).
    """

    __slots__ = ("_shape", "_dtype", "_buf", "_decompress")

    def __init__(self, arr: np.ndarray, compressor: tuple) -> None:
        # ``arr`` förväntas redan vara i önskad dtype (t.ex. float32).
        # Se till att den är C-contig för att .tobytes() ska bli oambigu.
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        _codec, compress_fn, decompress_fn = compressor
        self._shape = arr.shape
        self._dtype = arr.dtype
        self._buf = compress_fn(arr.tobytes())
        self._decompress = decompress_fn

    @property
    def array(self) -> np.ndarray:
        raw = self._decompress(self._buf)
        return np.frombuffer(raw, dtype=self._dtype).reshape(self._shape)

    # Bekvämlighet: många call-sites gör ``arr.shape`` / ``arr.dtype`` innan
    # de bryr sig om värdena.
    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype


def _maybe_unwrap_array(v):
    """Returnera underliggande ndarray om ``v`` är en ``_CompressedArray``.

    Anropas i ``_swap_in_frame`` så resten av ritkoden alltid ser vanliga
    numpy-arrays och inte behöver bry sig om lagringsformatet.
    """
    if isinstance(v, _CompressedArray):
        return v.array
    return v


class _SeriesBuffer:
    """Append-only ``(step, value)`` series backed by growable numpy arrays.

    Drop-in for the ``deque`` the plot tabs used to hold: ``append``,
    ``clear``, ``len``, truthiness, iteration and indexing all yield
    ``(int step, float value)`` tuples, and ``maxlen`` drops the oldest
    points like a bounded deque. Unlike a deque, :meth:`arrays` hands the
    plot code contiguous numpy views, so drawing a frame costs O(visible
    window) instead of a Python loop over the whole rollout (inference
    keeps every tick, which made the frame rate fall linearly with the
    rollout length).

    ``is_sorted`` stays True while steps are non-decreasing; the plot only
    uses binary search and per-pixel decimation when it holds.
    """

    __slots__ = ("maxlen", "_steps", "_vals", "_start", "_end", "is_sorted")

    def __init__(self, maxlen: Optional[int] = None) -> None:
        self.maxlen = None if maxlen is None else max(1, int(maxlen))
        self._steps = np.empty(256, dtype=np.int64)
        self._vals = np.empty(256, dtype=np.float64)
        self._start = 0
        self._end = 0
        self.is_sorted = True

    def append(self, item) -> None:
        step, value = item
        step = int(step)
        if self._end > self._start and step < self._steps[self._end - 1]:
            self.is_sorted = False
        if self._end == len(self._steps):
            n = self._end - self._start
            cap = len(self._steps)
            if n * 2 > cap:
                cap *= 2
            steps = np.empty(cap, dtype=np.int64)
            vals = np.empty(cap, dtype=np.float64)
            steps[:n] = self._steps[self._start:self._end]
            vals[:n] = self._vals[self._start:self._end]
            self._steps, self._vals = steps, vals
            self._start, self._end = 0, n
        self._steps[self._end] = step
        self._vals[self._end] = float(value)
        self._end += 1
        if self.maxlen is not None and self._end - self._start > self.maxlen:
            self._start = self._end - self.maxlen
            if not self.is_sorted:
                s = self._steps[self._start:self._end]
                self.is_sorted = bool(np.all(s[1:] >= s[:-1]))

    def clear(self) -> None:
        self._start = 0
        self._end = 0
        self.is_sorted = True

    def arrays(self) -> tuple:
        """Return ``(steps, values)`` as read-only-by-convention views."""
        return (self._steps[self._start:self._end],
                self._vals[self._start:self._end])

    def __len__(self) -> int:
        return self._end - self._start

    def __iter__(self):
        steps, vals = self.arrays()
        return zip(steps.tolist(), vals.tolist())

    def __getitem__(self, i: int) -> tuple:
        n = self._end - self._start
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError("series index out of range")
        j = self._start + i
        return (int(self._steps[j]), float(self._vals[j]))


def _series_arrays(buf) -> tuple:
    """``(steps, values, is_sorted)`` for a series buffer or a plain list."""
    if isinstance(buf, _SeriesBuffer):
        steps, vals = buf.arrays()
        return steps, vals, buf.is_sorted
    if not buf:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, True
    arr = np.asarray(list(buf), dtype=np.float64).reshape(-1, 2)
    steps, vals = arr[:, 0], arr[:, 1]
    return steps, vals, bool(np.all(steps[1:] >= steps[:-1]))


def _axis_labels(values: Sequence[float], step: float) -> list:
    """Plain decimal y-axis labels: no scientific notation, no plus sign.

    One number of decimals for the whole axis: three significant digits
    of the label ``step``, then the trailing zeros all labels share are
    dropped - 60000, 45000; 0.75, 0.50; -50.7, -30.6; 0.0012, 0.0009.
    ``-0`` prints as ``0``.
    """
    step = abs(float(step))
    if step > 0 and np.isfinite(step):
        decimals = int(min(12, max(0, 2 - np.floor(np.log10(step)))))
    else:
        decimals = 2
    out = [f"{float(v):.{decimals}f}" for v in values]
    if decimals > 0:
        common = min(len(t) - len(t.rstrip("0")) for t in out)
        common = min(common, decimals)
        if common:
            out = [t[:-common] for t in out]
        out = [t.rstrip(".") if t.endswith(".") else t for t in out]
    return [t.lstrip("-") if float(t) == 0.0 else t for t in out]


def _decimate_polyline(xs: np.ndarray, ys: np.ndarray) -> tuple:
    """Reduce a polyline with non-decreasing integer ``xs`` to at most four
    points per pixel column (first, min, max, last).

    Pixel-exact: every segment inside one column is vertical, so the
    column's ink is the span [min, max] plus the links to its neighbours,
    which first/last preserve.
    """
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    ends = np.r_[starts[1:], len(xs)] - 1
    col_x = xs[starts]
    out_y = np.stack([ys[starts],
                      np.minimum.reduceat(ys, starts),
                      np.maximum.reduceat(ys, starts),
                      ys[ends]], axis=1).ravel()
    out_x = np.repeat(col_x, 4)
    return out_x, out_y


class _NullViz:
    """Drop-in replacement when pygame is unavailable or init failed.

    All methods are no-ops; :meth:`pump_events` returns True so callers
    keep running.
    """

    enabled = False

    def update_biomass(self, *a, **kw): pass
    def update_reward(self, *a, **kw): pass
    def update_series(self, *a, **kw): pass
    def set_training_progress(self, *a, **kw): pass
    def set_progress_message(self, *a, **kw): pass
    def update_loss_breakdown(self, *a, **kw): pass
    def update_diet_breakdown(self, *a, **kw): pass
    def update_predation_breakdown(self, *a, **kw): pass
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
    def set_eval_interval_default(self, *a, **kw): pass
    def get_eval_interval_override(self, *a, **kw): return None
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
    # Below this scrub-slider width the playback bar switches its frame
    # indicator to the compact texts (section 191).
    _MIN_SCRUB_W = 120
    _playback_compact = False
    # Most frames a recorded rollout keeps. Past it every other frame is
    # dropped and the capture stride doubles, so a 100000-tick inference
    # rollout is still a film with evenly spaced frames (stride 16).
    _MAX_ROLLOUT_FRAMES = 10000

    def __new__(cls, *args, **kwargs):
        # Linux headless fallback. macOS/Windows native displays do not use
        # DISPLAY, and Wayland desktops may only provide WAYLAND_DISPLAY.
        if (sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
                and not os.environ.get("WAYLAND_DISPLAY")
                and not os.environ.get("SDL_VIDEODRIVER")):
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
        tick_hours: Optional[int] = None,
        capture: bool = True,
        heatmap_norm: str = "rolling",
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
        # How much real time one tick is (project_metadata.tick_hours,
        # section 97). The engine is tick-agnostic, so this is purely for
        # display: it turns the tick counter into simulated time. Callers
        # without a project file (manual_play, simple_inference) leave it
        # None and get the historical 6 h.
        self.tick_hours = resolve_tick_hours(tick_hours)
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
                          "predation", "starvation", "impacts", "natural",
                          "progress", "viability"]
        else:
            self._tabs = ["biomass", "energy", "move", "rest", "eat",
                          "predation", "starvation", "impacts", "natural"]
        # Inference receives per-tick values. Training uses final biomass
        # and energy ratios, plus action averages over each probe rollout.
        if self.mode == "train":
            self._tab_labels = {
                "reward": "reward",
                "biomass": "biomass (% of start)",
                "energy": "energy (% of start)",
                "move": "avg move action (%)",
                "rest": "avg rest action (%)",
                "eat": "avg eat action (%)",
                "predation": "predation share of total loss (%)",
                "starvation": "starvation share of total loss (%)",
                "impacts": "impact share of total loss (%)",
                "natural": "natural mortality (M1) share of total loss (%)",
                "progress": "Long run of the current policies (biomass, % of start)",
                "viability": "Viability margin per probe (0 = bound, 1 = spawn level / no drift)",
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
                "natural": "natural mortality (M1) share of tick loss (%)",
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
        self._legend_all_rect = None
        _all_series_ids = list(self.fg_ids) + [
            eid for eid in self._extra_plot_ids if eid not in self.fg_ids
        ]
        # I inference-läge sparas hela rollouten (obegränsad buffert) så
        # att scrollbaren under plot-arean kan panorera tillbaka till
        # tick 0 utan att tidiga punkter tappats. I träningsläget
        # behålls ``reward_window`` som maxlen — där pushas en punkt
        # per ARS-step och rolling-fönster undviker minnesläckor över
        # långa körningar.
        _series_maxlen = (None
                          if str(self.mode).lower().startswith("infer")
                          else self.reward_window)
        self._series: Dict[str, Dict[str, "_SeriesBuffer | list"]] = {
            tab: {fid: ([] if tab in _PROBE_TABS else _SeriesBuffer(maxlen=_series_maxlen))
                  for fid in _all_series_ids}
            for tab in self._tabs
        }
        self._progress_message = "Waiting for the first probe"
        # The English template and fields behind ``_progress_message`` and
        # behind a formatted tab title, so both are drawn in the display
        # language (section 212); the English strings stay as they were.
        self._progress_fmt: tuple = (self._progress_message, {})
        # The latest probe's viability block (section 223.18), shown as a
        # second line on the progress tab; None hides it.
        self._probe_viability = None
        # The viability tab's V per probe and its extinct marks (section
        # 223.20): [(step, V clipped to the axis)], [(step, fid)].
        self._viability_v: list = []
        self._viability_extinct: list = []
        self._viability_scores: dict = {}  # step -> V unclipped, None = extinct
        # The long run's length in library ticks (the probe world is built
        # at LIBRARY_TICK_HOURS); None until the first probe (223.21).
        self._long_run_ticks = None
        self._tab_label_fmt: Dict[str, tuple] = {}
        # Back-compat alias: legacy callers (and internal code) treat the
        # ``reward`` tab buffer as the historical _reward_buf. In inference
        # mode the first tab is biomass, so update_reward writes there.
        self._reward_buf: Dict[str, _SeriesBuffer] = self._series[self._tabs[0]]
        # Click hit-boxes for tab headers, recomputed each frame.
        self._tab_rects: list = []
        # Latest biomass arrays + totals for heatmap rendering.
        self._biomass: Dict[str, np.ndarray] = {}
        self._totals: Dict[str, float] = {fid: 0.0 for fid in self.fg_ids}
        # Initial (rollout start) totals per FG; captured on tick==0 so that
        # the title can display 'B0=... B=...' for context. Reset each new
        # rollout (train probe runs fresh per ARS-iter, inference is one run).
        self._b0: Dict[str, float] = {fid: 0.0 for fid in self.fg_ids}
        # Rollout-start per-cell maxima, captured on the first frame of
        # each rollout. This is the DEFAULT heatmap reference (section 94).
        self._heatmap_vmax0: Dict[str, float] = {fid: 0.0 for fid in self.fg_ids}
        # Rolling normalisation ('n'): take the reference from the frame
        # being drawn instead of from the start of the rollout. The two
        # answer different questions and neither is always right, which is
        # why it is a toggle rather than a choice made once:
        #   fixed    absolute. Declining biomass darkens, and two ticks or
        #            two FGs can be compared by eye. A field that collapses
        #            to a thousandth goes black and its structure with it.
        #   rolling  relative. Always uses the full colour range, so the
        #            pattern stays readable however small the numbers get -
        #            at the price of the brightness meaning nothing across
        #            frames, since the reference moves under it.
        # The colourbar always prints the reference in use, so a frame can
        # be read without knowing which mode produced it. The starting
        # mode is ``heatmap_norm`` (--norm, default rolling; section 197).
        if heatmap_norm not in ("rolling", "fixed"):
            raise ValueError(
                f"heatmap_norm must be 'rolling' or 'fixed', not {heatmap_norm!r}")
        self._rolling_heatmap: bool = heatmap_norm == "rolling"
        # Hit-rect for the status-bar button that toggles it, registered
        # every frame by ``_draw_status_bar``.
        self._hm_norm_button_rect: Optional[tuple] = None
        # Display language (section 212): the flag button at the right end
        # of the status bar opens a menu of the three flags. The strings in
        # this file are the English source; ``_t`` looks the translation up
        # at draw time (lib/viz/i18n.py), so no state depends on it.
        # The last language picked, remembered between runs.
        saved_lang = _load_settings().get("language")
        self._lang: str = (saved_lang if saved_lang in i18n.CODES
                           else i18n.DEFAULT)
        self._lang_button_rect: Optional[tuple] = None
        self._lang_menu_open: bool = False
        self._lang_item_rects: list = []
        # Was the pointer inside the plot area at the previous motion
        # event? Kept so the movement that LEAVES the area also forces a
        # repaint and the hover tooltip is erased instead of stranded.
        self._plot_hover: bool = False
        # Data arrived that is not on screen yet: the update_* methods
        # other than update_biomass only store, and between probes
        # (--eval-every > 1) nothing else repaints, so the reward curve
        # and the status row waited for a hover over the plot.
        # ``pump_events`` repaints while it is set (frame-capped);
        # ``_render_full`` clears it.
        self._dirty: bool = False
        # The species in each multi-species FG (the library's
        # ``member_species``, read off the FG objects by
        # ``update_biomass``), shown as a tooltip over the panel's info
        # lines. ``_member_rects`` holds (rect, fid) per visible panel,
        # rebuilt every frame by ``_draw_heatmaps``; ``_member_hover``
        # plays the role of ``_plot_hover`` for them.
        self._members: Dict[str, list] = {}
        self._member_rects: list = []
        self._member_hover: bool = False
        # (code, prey name) for each prey code on the panel's diet line
        # this frame, listed under the species in the same tooltip;
        # filled by ``_draw_one_heatmap``.
        self._panel_legend: Dict[str, list] = {}
        # One global visual biomass scale for inference. 1.0 means:
        # heatmap max = rollout-start per-cell max. After slider adjustment,
        # biomass plot max = 100% * scale. The slider zooms both together.
        #
        # Heatmaps stay comparable across ticks, including near extinction.
        # The line plot still auto-scales until the slider is adjusted.
        # Clicking the label restores 1x start colours and the auto plot axis.
        self._biomass_display_scale: float = 1.0
        self._biomass_scale_manual: bool = False
        self._biomass_scale_min: float = 0.05
        self._biomass_scale_max: float = 5.0
        self._biomass_scale_track_rect: Optional[tuple] = None
        self._biomass_scale_label_rect: Optional[tuple] = None
        self._dragging_biomass_scale: bool = False
        # Per-FG biomass-loss fractions shown above the heatmap as
        # ``pr/st/im=X/Y/Z%``. Updated from train probe rollouts and
        # inference ticks. Values are fractions in [0, 1].
        self._loss_breakdown: Dict[str, Dict[str, float]] = {}
        # Per-FG dietuppdelning (predator-DM → {prey_id: andel}) som visas
        # som en femte textrad ovanför heatmapen, på formatet
        # ``<abbr1>/<abbr2>/… = X/Y/…%``. Uppdateras via
        # :meth:`update_diet_breakdown` från probe-rollouten i train.py och
        # från inference.py per tick. Värden är fraktioner i [0, 1] som
        # summerar till 1 när någon konsumtion skett (annars saknas FG).
        self._diet_breakdown: Dict[str, Dict[str, float]] = {}
        # Per-FG predation by predator (prey -> {predator: share of the
        # prey's predation loss}), listed in the panel's hover tooltip;
        # :meth:`update_predation_breakdown`, fed like the diet.
        self._predation_breakdown: Dict[str, Dict[str, float]] = {}
        # (predator id, share) behind each panel's tooltip block, rebuilt
        # by ``_draw_one_heatmap`` every frame.
        self._panel_predators: Dict[str, list] = {}
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
        # Range = [3, 25000] (train) / [3, 100000] (inference) med
        # log-skala mappning för bättre kontroll. In inference a CLI
        # length above the max is kept (see ``set_ticks_default``).
        # ``_ticks_track_rect`` är (x, y, w, h) för den enda track-rect:en,
        # registrerad varje frame av ``_draw_status_bar``.
        self._ticks_min: int = 3
        self._ticks_max: int = (100000
                                if str(self.mode).lower().startswith("infer")
                                else 25000)
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
        # Evaluation probe interval: how many training updates between
        # two survival progress evaluations (--eval-every) and between two
        # biomass probes of this viewer (training_progress.probe_due; the
        # first update of a session always probes). Training only,
        # row 4 of the status bar, linear 1-50, locked by default like
        # the other training sliders. The default is the CLI value as
        # given (it may lie outside the range; the knob is then clamped);
        # the progress evaluator reads the override before each update.
        self._evalint_min: int = 1
        self._evalint_max: int = 50
        self._evalint_default: int = 1
        self._evalint_override: Optional[int] = None
        self._evalint_track_rect: Optional[tuple] = None
        self._dragging_evalint: bool = False
        self._evalint_locked: bool = True
        self._evalint_lock_rect: Optional[tuple] = None
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
        # "equilibrium" (section 207, the benthos' depth-habitat start)
        # has no parameters and no templates; it is listed so a group
        # that spawns with it shows its mode instead of "(default)".
        self._spawn_modes: tuple = (
            "uniform", "perlin", "colony", "env_driven", "equilibrium")
        self._spawn_mode_labels: Dict[str, str] = {
            "uniform": "Uniform", "perlin": "Perlin",
            "colony": "Colony", "env_driven": "Env-driven",
            "equilibrium": "Equilibrium",
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
        # Daylight calendar at the current tick (month, day length), or
        # None when the project runs without one (section 137).
        self._calendar: Optional[dict] = None
        # The calendar of the frame on screen: the live one, or the
        # replayed frame's during playback (set in ``_render_full``).
        self._shown_calendar: Optional[dict] = None
        # The loader's ``params['daylight']`` of the live rollout: the
        # inference plot's calendar band maps every x in its window to a
        # month and a clock time with it, not only the current tick.
        self._daylight_cfg: Optional[dict] = None
        self._fps_clock_t = time.monotonic()
        self._fps_frames = 0
        self._fps_value = 0.0

        # ---- Layout -------------------------------------------------------
        # Grid of heatmaps: choose columns so cells stay readable. With 8 FGs
        # we get 4 cols x 2 rows; with 1 FG, 1x1; etc.
        n = len(self.fg_ids)
        self._cols = min(4, n)
        # Heatmaps sorted under "Decision makers" and "Non-decision
        # makers" (section 198), each group under a header row and
        # starting on a row of its own. Without NDM information (or with
        # only one kind present) there is one untitled group, the old
        # layout.
        ndms = [f for f in self.fg_ids if f in self._ndm_ids]
        dms = [f for f in self.fg_ids if f not in self._ndm_ids]
        if dms and ndms:
            self._hm_groups = [("Decision makers", dms),
                               ("Non-decision makers", ndms)]
            self._hm_header_h = 22
        else:
            self._hm_groups = [(None, list(self.fg_ids))]
            self._hm_header_h = 0
        self._rows = sum((len(g) + self._cols - 1) // self._cols
                         for _t, g in self._hm_groups)
        hm_w = self.grid_w * self.cell_px
        hm_h = self.grid_h * self.cell_px
        # Five info lines above the heatmap: FG id, 'B0 = ... B = ...',
        # 'mv/rs/et = ...' (DM only), 'pr/st = ...' (DM only), and the
        # diet breakdown '<abbr>/<abbr>/... = X/Y/...%' (DM only), followed
        # by a horizontal b0 slider between the info block and heatmap.
        # NDM panels leave the three action/loss/diet lines blank to keep
        # heatmap origin aligned across panels.
        # title_h = 5 textrader (~14 px var) + slider (~22 px) + luft.
        title_h = 104
        self._slider_h = 16
        cbar_h = 16  # colorbar strip (gradient + 0/max labels)
        # Två extra rader under colorbaren: "Mode: …" och "Tpl: …".
        # Varje rad ~14 px text + 2 px padding ⇒ 32 px för båda.
        spawn_dd_h = 34
        pad = 8
        # The natural layout: ``self._cols`` panels of the reference cell
        # size per row. It sets the initial window; after that
        # ``_compute_layout`` derives everything from the window size and
        # the splitter position.
        nat_panel_w = hm_w + 2 * pad
        nat_block_w = self._cols * nat_panel_w
        nat_block_h = (self._rows * (hm_h + title_h + cbar_h + spawn_dd_h + 2 * pad)
                       + self._hm_header_h * sum(
                           1 for t, _g in self._hm_groups if t is not None))
        # Reserve enough horizontal room in the plot panel for the legend:
        # roughly 7 px per character of the longest FG id, plus the swatch
        # and gutter. This avoids the legend overflowing the window with
        # long names like ``pelagic_fish_rnd``.
        all_legend_ids = list(self.plot_fg_ids) or list(self.fg_ids)
        # Longest name in any display language, so the layout does not
        # change with the language.
        max_name_len = max((len(self._display_name(fid, code))
                            for fid in all_legend_ids
                            for code in i18n.CODES), default=8)
        self._legend_w = 38 + 7 * max_name_len + 8  # checkbox + swatch + text + pad
        # Bredd för tab-strippen: approximera font-bredden till ~7 px per
        # tecken och lägg på 10 px padding + 4 px gutter per flik, plus
        # en startmarginal. Denna bredd måste rymmas inom plot-panelens
        # legend-fria område, annars wrappar/försvinner flikar. Vi
        # garanterar därför att plot_w är minst så stor att hela
        # tab-strippen (+ legend + några pixlar marginal) ryms på en rad.
        tab_strip_w = 6 + max(
            sum(7 * len(i18n.translate(code, _TAB_NAMES.get(t, t))) + 10 + 4
                for t in self._tabs)
            for code in i18n.CODES) + 8
        plot_w = max(420, nat_block_w // 2 + self._legend_w,
                     tab_strip_w + self._legend_w + 4)
        # Status-baren rymmer textraden och två slider-rader. I training är
        # rad 3 n_eval_ticks; i inference är rad 3 biomass display scale.
        # In training a fourth row holds the evaluation probe interval.
        if str(mode).lower().startswith("infer"):
            status_h = 62
        else:
            status_h = 80
        # Playback bar (under heatmaps and plot) for replaying the most
        # recent probe/inference rollout. Holds the buttons << / >> / play
        # / pause / speed +/- and a frame indicator. Always allocated, even
        # before a film exists.
        playback_h = 32
        self._status_h = status_h
        self._playback_h = playback_h
        self._pad = pad
        # Panel height on top of the heatmap itself.
        self._panel_extra_h = title_h + cbar_h + spawn_dd_h + 2 * pad
        # Heatmaps never get smaller than the reference cell size: the
        # five info lines need a panel about that wide. A narrower
        # heatmap area gets fewer columns instead.
        self._min_cell_px = self.cell_px
        # Heatmap area = viewport + a scrollbar column (always reserved,
        # drawn only when the grid is taller than the viewport). The
        # splitter between it and the plot sets the area's share of the
        # window width (``_split_frac``) and is dragged with the mouse.
        self._hm_sb_w = 12
        self._splitter_w = 8
        self._min_hm_area_w = nat_panel_w + self._hm_sb_w
        self._min_plot_w = max(320, self._legend_w + 160)
        nat_hm_area_w = nat_block_w + self._hm_sb_w
        nat_win_w = nat_hm_area_w + self._splitter_w + plot_w + pad // 2
        # The status-bar sliders end where the plot starts, but never
        # left of where they end in the natural layout.
        self._status_track_right_min = nat_hm_area_w + self._splitter_w
        self._hm_scroll_y = 0
        self._hm_scroll_step = 48  # px per wheel notch
        # Folding the heatmap groups (section 214): how open each titled
        # group is (1 open, 0 folded under its header; in between while it
        # glides), the block-coordinate span its panels may paint in, and
        # the header rows' click targets (rebuilt every frame).
        self._group_open: Dict[str, float] = {
            t: 1.0 for t, _g in self._hm_groups if t is not None}
        # Where each group is heading: 1 open, 0 folded. Sets the sign in
        # its box (minus = a click folds in, plus = a click folds out).
        self._group_target: Dict[str, float] = dict(self._group_open)
        # Groups folded in an earlier run start folded (section 215).
        saved_folded = _load_settings().get("folded_groups")
        if isinstance(saved_folded, list):
            for title in saved_folded:
                if title in self._group_open:
                    self._group_open[title] = 0.0
                    self._group_target[title] = 0.0
        self._hm_group_spans: list = []
        self._hm_slot_group: Dict[str, int] = {}
        self._hm_header_rects: list = []
        # Blank room below the block that lets the scroll position stay
        # put when a fold shrinks the block, so the header just clicked
        # does not slide away from the pointer. It only shrinks: scrolling
        # up (or the block growing back) gives it up.
        self._hm_scroll_slack = 0
        self._dragging_hm_scroll = False
        self._hm_scroll_drag_dy = 0
        self._hm_thumb_rect: Optional[tuple] = None
        self._hm_scrollbar_rect: Optional[tuple] = None
        self._dragging_splitter = False
        self._splitter_drag_dx = 0
        self._splitter_hover = False
        # The plot panel folds in under the heatmaps (section 190): a
        # button at the top of the splitter (or the P key) rolls it in
        # under the heatmaps - they keep their size and place, the button
        # stays under the pointer, the window edge glides in to the strip
        # - and the same button rolls it out again (section 194). Folded,
        # the window's own edges and corners size the heatmap area
        # (section 195).
        self._plot_collapsed = False
        self._plot_toggle_h = 36
        self._plot_toggle_rect: Optional[tuple] = None
        self._plot_toggle_hover = False
        self._collapsed_strip_w = 16
        # Window width and plot width at the moment the plot folded; the
        # window gets the former back on unfold.
        self._expanded_win_w = 0
        self._expanded_plot_w = 0
        # Heatmap-area width pinned while the plot is folded or rolling
        # (section 194): the splitter - and the fold button under the
        # pointer - stays where it was. None = the splitter share rules.
        self._hm_area_fixed: Optional[int] = None
        # Fold/unfold roll: the window's right edge glides over this many
        # seconds (ease-out) with a captured image of the plot in the
        # widening/narrowing plot area. 0 = one step (tests).
        self._fold_anim_s = 0.25
        self._plot_anim_img = None
        self._sdl_window = None
        # Set when the user resizes the folded window (section 195):
        # unfolding then puts the plot back beside the new heatmap area
        # instead of restoring the old window width.
        self._collapsed_resized = False
        # The status and playback bars need about this much; with one or
        # two heatmap columns the folded window stops here.
        self._min_collapsed_win_w = 600
        # The window starts at its natural size, capped to the screen so
        # the playback bar stays reachable; it is resizable (and can be
        # maximised), and every resize goes through ``_compute_layout``.
        win_w = nat_win_w
        win_h = status_h + nat_block_h + playback_h + pad
        usable = self._usable_screen_size()
        if usable is not None:
            win_w = min(win_w, max(640, usable[0]))
            win_h = min(win_h, max(200, usable[1]))
        self._win_w = int(win_w)
        self._win_h = int(win_h)
        # A width-capped window takes the cut from the plot first.
        self._split_frac = nat_hm_area_w / float(self._win_w)
        self._compute_layout()
        self._split_frac = self._splitter_rect[0] / float(self._win_w)
        # ---- Playback state ----------------------------------------------
        # Den senast färdigställda rollouten lagras som en lista av frame-
        # snapshots (dicts med biomass/totals/b0/loss_breakdown/diet_breakdown/
        # action_fracs/tick/status). Under live-träningens probe-rollout
        # ackumuleras frames i ``_pending_rollout``; när ``end_rollout_recording``
        # kallas flyttas listan över till ``_current_rollout`` och knapparna
        # blinkar tills användaren klickar (eller en ny film tar över).
        self._recording = False
        # ``capture=False`` (inference --no-capture): rollouts are never
        # recorded, so no film exists and the playback bar stays disabled.
        self._capture = bool(capture)
        self._pending_rollout: list = []
        self._current_rollout: list = []
        # Capture every ``_capture_stride``-th update_biomass call while
        # recording; ``_capture_calls`` counts the calls (see
        # ``_MAX_ROLLOUT_FRAMES``).
        self._capture_stride = 1
        self._capture_calls = 0
        self._playback_mode = "live"  # "live" | "paused" | "playing"
        self._playback_idx = 0
        self._playback_fps = 60.0  # justerbar via knappar (1..60)
        self._playback_last_advance = 0.0
        self._playback_blink_until = 0.0
        self._playback_rects: list = []  # [(rect, action_name)]

        # ---- Plot horizontal scroll --------------------------------------
        # I inference-läget: när rolloutens x-axel är längre än ett fast
        # fönster (``_plot_window_width`` ticks) ritas en horisontell
        # scrollbar under plot-arean; ``_plot_scroll_offset`` är vänsterkant
        # (i tick-koordinater) för det synliga fönstret. ``None`` = auto
        # (följ senaste data-änden, dvs. samma beteende som tidigare).
        self._plot_window_width: int = 100
        # Panorerings-steg (andel av synligt fönster per ``event.x``-enhet)
        # när användaren scrollar horisontellt via touchpad-gest eller
        # horisontellt scrollhjul. 0.15 = ~15 % av fönstret per "notch",
        # vilket motsvarar ett par pixlar smidig touchpad-panorering.
        self._plot_pan_step_frac: float = 0.15
        # Grafens plot-area (px0, py0, pw, ph) — sätts av ``_draw_plot``
        # varje frame och används av MOUSEWHEEL-hanteringen för att
        # avgöra om användaren scrollar över graf-fönstret (i så fall
        # tolkas hjulet som zoom, inte som fönster-scroll).
        self._plot_area_rect: Optional[tuple] = None
        # Min/max på graf-fönstrets bredd i tick-koordinater. Min
        # sätts så att grafen inte kan bli trivialt smal; max lyfts
        # dynamiskt till full data-span vid zoom-ut så användaren kan
        # se hela historiken. Se ``_handle_plot_wheel_zoom``.
        self._plot_window_width_min: int = 10
        self._plot_scroll_offset: Optional[float] = None
        self._plot_scroll_track_rect: Optional[tuple] = None
        self._plot_scroll_thumb_rect: Optional[tuple] = None
        self._dragging_plot_scroll: bool = False
        self._plot_scroll_drag_dx: float = 0.0
        # "Live"-knapp bredvid plot-scrollbaren (endast inference).
        # Klick sätter ``_plot_scroll_offset = None`` så plotten återgår
        # att följa senaste tick automatiskt. Sätts av ``_draw_plot``
        # varje frame; None när scroll inte är aktiv.
        self._live_button_rect: Optional[tuple] = None
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
            # RESIZABLE: the window can be resized and maximised; the
            # surface stays a software surface.
            self._screen = pygame.display.set_mode(
                (self._win_w, self._win_h),
                pygame.SWSURFACE | pygame.RESIZABLE)
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
            self._calendar = self._calendar_from(fgs, self._tick)
            for fid in self.fg_ids:
                if fid not in fgs:
                    continue
                obj = fgs[fid]
                arr = obj.biomass if hasattr(obj, "biomass") else obj
                arr = np.asarray(arr, dtype=np.float32)
                # Hide only what the model itself treats as extinct: the
                # per-cell sweep threshold extinction_threshold_factor *
                # min_split_biomass (section 61). A fixed 0.5 hid live
                # porpoise cells of 5-125 kg (factor 0.02, section 146).
                # Continuous biomass, or a factor of 0, keeps a numerical
                # floor that only suppresses subnormal noise.
                msb = float(getattr(obj, 'min_split_biomass', 0.0)) \
                    if hasattr(obj, 'biomass') else 0.0
                factor = float(getattr(obj, 'extinction_threshold_factor', 0.5))
                dead_eps = factor * msb if msb > 0.0 and factor > 0.0 else 1e-20
                if dead_eps > 0.0:
                    # Kopiera först för att inte mutera env:s underliggande
                    # array; nolla sedan alla celler under tröskeln.
                    arr = np.where(arr <= dead_eps,
                                   np.float32(0.0), arr).astype(np.float32,
                                                                copy=False)
                if hasattr(obj, "params"):
                    members = (obj.params or {}).get("member_species")
                    if self._member_lines(members):
                        self._members[fid] = list(members)
                    else:
                        self._members.pop(fid, None)
                total = float(arr.sum())
                self._biomass[fid] = arr
                self._totals[fid] = total
                if new_rollout or self._heatmap_vmax0.get(fid, 0.0) <= 0.0:
                    self._b0[fid] = total
                    try:
                        self._heatmap_vmax0[fid] = max(float(arr.max()), 1e-12)
                    except Exception:
                        self._heatmap_vmax0[fid] = 1e-12
            if extra:
                self._status.update(extra)
            # Spela in en frame om vi är mitt i en rollout-inspelning.
            # Capture tas EFTER att all state uppdaterats men FÖRE
            # _maybe_render så snapshotten reflekterar samma frame som
            # live-renderingen visar.
            if self._recording:
                try:
                    if self._capture_calls % self._capture_stride == 0:
                        self._pending_rollout.append(self._capture_frame())
                    self._capture_calls += 1
                    # Frame cap: 10000 frames ≈ 1.1 GB per film på 8 FG /
                    # 60×60 okomprimerat, acceptabelt så länge bara en
                    # film lever åt gången (den gamla frigörs i
                    # ``end_rollout_recording``). Past the cap, drop every
                    # other frame in place and double the stride, so the
                    # film stays evenly spaced in ticks however long the
                    # rollout runs.
                    if len(self._pending_rollout) > self._MAX_ROLLOUT_FRAMES:
                        del self._pending_rollout[1::2]
                        self._capture_stride *= 2
                except Exception as e:
                    self._log_once(f"frame capture failed: {e!r}")
            self._maybe_render()
        except Exception as e:
            self._log_once(f"update_biomass failed: {e!r}")

    def _calendar_from(self, fgs: Mapping[str, object],
                       tick: int) -> Optional[dict]:
        """The daylight calendar at ``tick``, read off the FG objects.

        The loader puts the same ``params['daylight']`` (latitude, tick
        length, the world's start tick) on every FG, and the environment
        builds its light tables from exactly that, so every caller gets
        the month and day length without passing anything new. Raw
        arrays (tests) and projects without the calendar give None.

        The day is the one at the START of ``tick`` - the instant the
        time label describes - i.e. the day of the tick about to run,
        not of the one just completed.
        """
        self._daylight_cfg = None
        for obj in fgs.values():
            params = getattr(obj, "params", None)
            cfg = params.get("daylight") if isinstance(params, dict) else None
            if cfg:
                self._daylight_cfg = cfg
                try:
                    return daylight.calendar_at(cfg, tick)
                except Exception as e:
                    self._log_once(f"daylight calendar failed: {e!r}")
                    return None
        return None

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
            self._dirty = True
        except Exception as e:
            self._log_once(f"update_series failed: {e!r}")

    def _viability_line(self):
        """The progress tab's viability line and whether every group
        passes (section 223.18); None without a scored probe. Built at
        draw time so it follows the display language."""
        v = self._probe_viability
        if not v or v.get("worst") not in v.get("groups", {}):
            return None
        years = "-".join(str(y) for y in v.get("years", ()))
        score = "-" if v.get("score") is None else f"{v['score']:+.2f}"
        # Most telling first: the panel clips a long line at its edge.
        text = self._t("Viable {viable}/{present}, V {score}",
                       viable=v["viable"], present=v["present"], score=score)
        worst = v["groups"][v["worst"]]
        why = self._t(_VIABILITY_BINDING.get(worst.get("binding"), "extinct"))
        name = self._display_name(v["worst"])
        if worst.get("margin") is None:
            text += " | " + self._t("worst {name}: {why}", name=name, why=why)
        else:
            text += " | " + self._t("worst {name}: {margin} to {why}", name=name, why=why,
                                    margin=f"{worst['margin']:+.2f}")
        if (worst.get("years_to_floor") is not None
                and worst["years_to_floor"] <= _FLOOR_NOTE_MAX_YEARS):
            text += ", " + self._t("floor in ~{years} yr",
                                   years=f"{worst['years_to_floor']:.0f}")
        text += " | " + self._t("long-run years {years}", years=years)
        return text, v["viable"] == v["present"]

    def _set_viability_history(self, records) -> None:
        """Fill the viability tab from the probe history (section 223.20).

        One point per probe with a viability block, at its training
        update: each group's margin, V (the worst group) and an extinct
        mark, clipped to the tab's fixed axis. Probes without a block
        (before section 223.18, or without a complete year) are skipped.
        """
        if "viability" not in self._series:
            return
        lo, hi = _VIABILITY_AXIS
        series = {fid: [] for fid in self._series["viability"]}
        self._viability_v, self._viability_extinct, self._viability_scores = [], [], {}
        for record in records or ():
            v = record.get("viability")
            if not v:
                continue
            step = int(record.get("step", 0))
            for fid, g in v.get("groups", {}).items():
                margin = g.get("margin")
                if margin is None:
                    self._viability_extinct.append((step, fid))
                if fid in series:
                    series[fid].append(
                        (step, lo if margin is None else min(hi, max(lo, float(margin)))))
            score = v.get("score")
            self._viability_scores[step] = score
            self._viability_v.append(
                (step, lo if score is None else min(hi, max(lo, float(score)))))
        self._series["viability"] = series

    def set_progress_message(self, message: str) -> None:
        self._progress_message = message
        self._progress_fmt = (message, {})
        self._last_frame_ts = 0.0
        self._maybe_render()

    def set_training_progress(self, records, config, latest=None) -> None:
        """Show the latest probe (section 223.12): every group's biomass as
        a % of its start, once a day over the probe's years, x in days.

        ``latest`` is ``TrainingProgress.latest`` (step, seed, every_ticks,
        ticks_per_year, series); None keeps the waiting message. A group
        without starting biomass has no curve.
        """
        if not self.enabled or "progress" not in self._series:
            return
        self._set_viability_history(records)
        try:
            self._long_run_ticks = int((config or {}).get("ticks") or 0) or None
        except (TypeError, ValueError):
            self._long_run_ticks = None
        if not latest:
            self._last_frame_ts = 0.0
            self._maybe_render()
            return
        every = max(1, int(latest.get("every_ticks", 1)))
        per_year = max(1, int(latest.get("ticks_per_year", 1460)))
        self._probe_days_per_year = per_year / float(every)
        series = latest.get("series", {})
        for fid in self._series["progress"]:
            values = series.get(fid)
            self._series["progress"][fid] = [] if values is None else [
                (day, 100.0 * float(v)) for day, v in enumerate(values)
                if v is not None]
        self._probe_viability = (records[-1].get("viability")
                                 if records else None)
        missing = [self._display_name(fid) for fid in self.fg_ids
                   if fid not in self._ndm_ids and records
                   and records[-1]["survival_ticks"].get(fid) is None]
        if missing:
            self._progress_fmt = ("No starting biomass: {names}",
                                  {"names": ", ".join(missing)})
        else:
            self._progress_fmt = ("Update {step}: frozen policies, seed {seed} | {n} probes",
                                  {"step": int(latest.get("step", 0)),
                                   "seed": latest.get("seed", ""), "n": len(records)})
        self._progress_message = self._progress_fmt[0].format(
            **self._progress_fmt[1])
        self._last_frame_ts = 0.0
        self._maybe_render()

    def update_loss_breakdown(
        self,
        breakdown: Mapping[str, Mapping[str, float]],
    ) -> None:
        """Record per-FG biomass-loss fractions.

        ``breakdown`` maps fg_id -> {'predation': p, 'starvation': s,
        'impact': i, 'natural': n} where each value is a fraction in
        [0, 1] summing to 1.0 when there was any loss at all. Unknown keys
        are tolerated. Used to draw the ``pr/st/im/nm=...%`` line above
        each heatmap.
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
                    'natural':    float(lb.get('natural', 0.0)),
                }
            self._dirty = True
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
            self._dirty = True
        except Exception as e:
            self._log_once(f"update_diet_breakdown failed: {e!r}")

    def update_predation_breakdown(
        self,
        breakdown: Mapping[str, Mapping[str, float]],
    ) -> None:
        """Record per-prey predation by predator.

        ``breakdown`` maps prey_fg_id -> {predator_fg_id: fraction in [0, 1]}
        of the prey's predation loss (normalised over its predators). The
        hover tooltip of the prey's heatmap panel lists it. A prey that was
        not eaten is simply absent.
        """
        if not self.enabled:
            return
        try:
            for prey_id, pred_map in breakdown.items():
                if not isinstance(pred_map, Mapping):
                    continue
                self._predation_breakdown[prey_id] = {
                    str(k): float(v) for k, v in pred_map.items()
                }
            self._dirty = True
        except Exception as e:
            self._log_once(f"update_predation_breakdown failed: {e!r}")

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
            self._dirty = True
        except Exception as e:
            self._log_once(f"update_action_fracs failed: {e!r}")

    def update_status(self, **kw) -> None:
        if not self.enabled:
            return
        self._status.update(kw)
        self._dirty = True

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

        Klampas till ``[_ticks_min, _ticks_max]`` = [3, 25000] (train)
        eller [3, 100000] (inference). Anropas
        en gång av train.py (``--n_eval_ticks``) och inference.py
        (``--ticks``) när viz initieras.

        In inference only the lower bound applies: a CLI rollout longer
        than the slider's max runs to its full length, and the slider
        handle sits at the right end until it is dragged.
        """
        if not self.enabled:
            return
        try:
            v = int(ticks)
        except (TypeError, ValueError):
            return
        if not str(self.mode).lower().startswith("infer"):
            v = min(self._ticks_max, v)
        self._ticks_default = max(self._ticks_min, v)

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

    # ---- Evaluation probe interval slider (--eval-every) -------------
    def set_eval_interval_default(self, every: int) -> None:
        """Register the CLI --eval-every as the slider's default."""
        if not self.enabled:
            return
        try:
            self._evalint_default = max(1, int(every))
        except (TypeError, ValueError):
            return

    def get_eval_interval_override(self) -> Optional[int]:
        """The interval chosen on the slider, or None (use the CLI value)."""
        return self._evalint_override

    def get_eval_interval(self) -> int:
        """Effective interval: the override if set, else the default."""
        if self._evalint_override is not None:
            return int(self._evalint_override)
        return int(self._evalint_default)

    def _evalint_pos_from_value(self, v: int) -> float:
        lo, hi = float(self._evalint_min), float(self._evalint_max)
        return (max(lo, min(hi, float(v))) - lo) / (hi - lo)

    def _evalint_value_from_pos(self, frac: float) -> int:
        lo, hi = float(self._evalint_min), float(self._evalint_max)
        frac = max(0.0, min(1.0, float(frac)))
        return int(round(lo + frac * (hi - lo)))

    def _evalint_value_from_x(self, mx: int) -> Optional[int]:
        rect = self._evalint_track_rect
        if rect is None:
            return None
        rx, _ry, rw, _rh = rect
        return self._evalint_value_from_pos((mx - (rx + 2)) / float(max(1, rw - 4)))

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
        # Without capture the resets below still run; only the frame
        # recording is skipped.
        self._recording = self._capture
        self._pending_rollout = []
        self._capture_stride = 1
        self._capture_calls = 0
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
            self._predation_breakdown.clear()
            self._action_fracs.clear()
        except Exception as e:
            self._log_once(
                f"begin_rollout_recording breakdown-reset failed: {e!r}")

    def end_rollout_recording(self) -> None:
        """Avsluta inspelning; den nya filmen tar över ``_current_rollout``."""
        if not self.enabled:
            return
        self._recording = False
        # With a stride > 1 the last tick may have fallen between two
        # captures; add it so the film ends on the final state.
        if (self._pending_rollout and self._capture_calls > 0
                and (self._capture_calls - 1) % self._capture_stride != 0):
            try:
                self._pending_rollout.append(self._capture_frame())
            except Exception as e:
                self._log_once(f"final frame capture failed: {e!r}")
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
        """Bygg en snapshot av all state som ``_draw_one_heatmap`` läser.

        RAM-optimering (steg 1 + 2):
          1. Biomass-arrays castas till ``float32`` (halverar minnet mot
             float64; visualiseringen kvantiserar ändå till 256 färg-nivåer
             så precisionsförlusten är osynlig).
          2. Om en lossless kompressor finns (``ECO_ROLLOUT_COMPRESS``)
             wrappas varje array i en ``_CompressedArray`` som håller
             komprimerade bytes i minnet och dekomprimerar först när
             frame:t swappas in för visning. Glesa/spatialt korrelerade
             biomassa-fält komprimerar typiskt 5–20× med zstd-3.
        """
        # Kopior/casts är essentiella: live-state muteras efter capture.
        comp = _ROLLOUT_COMPRESSOR
        biomass: Dict[str, object] = {}
        for fid, arr in self._biomass.items():
            # Downcast float64 → float32 (lossless i praktiken för viz).
            # Andra dtypes lämnas orörda (int-masker etc. förekommer inte
            # här idag, men vi vill inte tvinga fram cast om de dyker upp).
            if arr.dtype == np.float64:
                a32 = arr.astype(np.float32, copy=True)
            elif arr.dtype == np.float32:
                a32 = arr.copy()
            else:
                a32 = arr.copy()
            if comp is not None and a32.dtype in (np.float32, np.float64):
                try:
                    biomass[fid] = _CompressedArray(a32, comp)
                    continue
                except Exception:
                    # Om kompressorn strular av någon anledning: fallback
                    # till rå array, aldrig krascha rollout-inspelningen.
                    pass
            biomass[fid] = a32
        totals = dict(self._totals)
        b0 = dict(self._b0)
        heatmap_vmax0 = dict(self._heatmap_vmax0)
        loss = {fid: dict(v) for fid, v in self._loss_breakdown.items()}
        diet = {fid: dict(v) for fid, v in self._diet_breakdown.items()}
        predation = {fid: dict(v) for fid, v in self._predation_breakdown.items()}
        actf = {fid: dict(v) for fid, v in self._action_fracs.items()}
        return {
            "biomass": biomass,
            "totals": totals,
            "b0": b0,
            "heatmap_vmax0": heatmap_vmax0,
            "loss_breakdown": loss,
            "diet_breakdown": diet,
            "predation_breakdown": predation,
            "action_fracs": actf,
            "tick": int(self._tick),
            "calendar": self._calendar,
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
            # Plain pointer movement is not an "interaction" - it changes
            # no state - but the hover tooltip is drawn from the live
            # mouse position inside ``_draw_plot``, so it only exists
            # while something repaints. During a PROBE rollout
            # ``update_biomass`` repaints every tick and the tooltip
            # works; during the perturbation rollouts nothing calls it
            # and this loop is the only thing running, so the tooltip
            # never appeared. Tracked separately from ``interacted`` so
            # it goes through the frame cap rather than forcing a redraw
            # per motion event.
            hover = False
            for event in pg.event.get():
                if self._handle_layout_event(event):
                    interacted = True
                elif event.type == pg.QUIT:
                    self._quit = True
                elif event.type == pg.KEYDOWN:
                    self._handle_key(event.key)
                    interacted = True
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                    interacted = True
                elif (event.type == pg.MOUSEBUTTONDOWN
                      and event.button == 3
                      and self._hit_plot_scroll(event.pos)):
                    # Högerklick på plot-scrollbaren → gå live.
                    self._plot_scroll_offset = None
                    self._dragging_plot_scroll = False
                    interacted = True
                elif event.type == pg.MOUSEWHEEL:
                    # Scrollhjul över graf-arean: vertikalt (event.y) →
                    # zoom in/ut, horisontellt (event.x, touchpad-gest
                    # eller horisontellt hjul) → panorera i tid.
                    try:
                        mpos = pg.mouse.get_pos()
                    except Exception:
                        mpos = (0, 0)
                    if self._handle_plot_wheel_zoom(mpos, int(event.y)):
                        interacted = True
                    if self._handle_plot_wheel_pan(mpos, int(getattr(event, "x", 0))):
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
                elif event.type == pg.MOUSEMOTION and self._dragging_evalint:
                    v = self._evalint_value_from_x(event.pos[0])
                    if v is not None:
                        self._evalint_override = int(v)
                        interacted = True
                elif event.type == pg.MOUSEMOTION and self._dragging_biomass_scale:
                    v = self._biomass_scale_value_from_x(event.pos[0])
                    if v is not None:
                        self._set_biomass_display_scale(v)
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
                    # Om thumb släpptes vid trackens högerkant → återgå
                    # till live-läge så plotten följer senaste tick igen.
                    self._plot_scroll_snap_to_live_if_at_end()
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
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1 \
                        and self._dragging_evalint:
                    self._dragging_evalint = False
                    interacted = True
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1 \
                        and self._dragging_biomass_scale:
                    self._dragging_biomass_scale = False
                    interacted = True
                elif event.type == pg.MOUSEMOTION:
                    # Reached only when no drag claimed the motion.
                    inside = self._hit_rect(self._plot_area_rect, event.pos)
                    if inside or self._plot_hover:
                        hover = True
                    self._plot_hover = bool(inside)
                    # Same for the species tooltip over a panel's info
                    # lines: repaint while inside and on the way out.
                    on_members = any(self._hit_rect(rect, event.pos)
                                     for rect, _fid in self._member_rects)
                    if on_members or self._member_hover:
                        hover = True
                    self._member_hover = on_members
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
            elif hover or self._dirty:
                # Frame-capped: a hover repaint is worth ~5 ms and the
                # pointer can produce hundreds of motion events a second.
                # New data (``_dirty``) that the cap holds back stays
                # flagged and is drawn by the next pump.
                self._maybe_render()
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
                    if self._handle_layout_event(event):
                        pass
                    elif event.type == pg.QUIT:
                        self._quit = True
                    elif event.type == pg.KEYDOWN:
                        self._handle_key(event.key)
                    elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)
                    elif (event.type == pg.MOUSEBUTTONDOWN
                          and event.button == 3
                          and self._hit_plot_scroll(event.pos)):
                        # Högerklick på plot-scrollbaren → gå live.
                        self._plot_scroll_offset = None
                        self._dragging_plot_scroll = False
                    elif event.type == pg.MOUSEWHEEL:
                        # Vertikalt hjul → zoom, horisontellt (touchpad
                        # eller horisontellt hjul) → panorera i tid.
                        try:
                            mpos = pg.mouse.get_pos()
                        except Exception:
                            mpos = (0, 0)
                        self._handle_plot_wheel_zoom(mpos, int(event.y))
                        self._handle_plot_wheel_pan(mpos, int(getattr(event, "x", 0)))
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
                          and self._dragging_evalint):
                        v = self._evalint_value_from_x(event.pos[0])
                        if v is not None:
                            self._evalint_override = int(v)
                    elif (event.type == pg.MOUSEMOTION
                          and self._dragging_biomass_scale):
                        v = self._biomass_scale_value_from_x(event.pos[0])
                        if v is not None:
                            self._set_biomass_display_scale(v)
                    elif (event.type == pg.MOUSEMOTION
                          and self._dragging_playback):
                        idx = self._playback_idx_from_x(event.pos[0])
                        if idx is not None:
                            self._playback_idx = idx
                            self._playback_mode = "paused"
                    elif (event.type == pg.MOUSEMOTION
                          and self._dragging_plot_scroll):
                        new_off = self._plot_scroll_offset_from_x(event.pos[0])
                        if new_off is not None:
                            self._plot_scroll_offset = float(new_off)
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_playback):
                        self._dragging_playback = False
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_plot_scroll):
                        self._dragging_plot_scroll = False
                        self._plot_scroll_snap_to_live_if_at_end()
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_slider is not None):
                        self._dragging_slider = None
                        self._b0_dirty = True
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_ticks):
                        self._dragging_ticks = False
                        self._ticks_dirty = True
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_neval):
                        self._dragging_neval = False
                        self._neval_dirty = True
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_evalint):
                        self._dragging_evalint = False
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_biomass_scale):
                        self._dragging_biomass_scale = False
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
            # Render one last full frame so the banner is visible.
            try:
                self._render_full()
            except Exception:
                pass
            while not self._quit:
                for event in pg.event.get():
                    if self._handle_layout_event(event):
                        pass
                    elif event.type == pg.QUIT:
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
                    elif (event.type == pg.MOUSEBUTTONDOWN
                          and event.button == 3
                          and self._hit_plot_scroll(event.pos)):
                        # Högerklick på plot-scrollbaren → gå live.
                        self._plot_scroll_offset = None
                        self._dragging_plot_scroll = False
                    elif event.type == pg.MOUSEWHEEL:
                        # Vertikalt hjul → zoom, horisontellt (touchpad
                        # eller horisontellt hjul) → panorera i tid.
                        try:
                            mpos = pg.mouse.get_pos()
                        except Exception:
                            mpos = (0, 0)
                        self._handle_plot_wheel_zoom(mpos, int(event.y))
                        self._handle_plot_wheel_pan(mpos, int(getattr(event, "x", 0)))
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
                          and self._dragging_evalint):
                        v = self._evalint_value_from_x(event.pos[0])
                        if v is not None:
                            self._evalint_override = int(v)
                    elif (event.type == pg.MOUSEMOTION
                          and self._dragging_biomass_scale):
                        v = self._biomass_scale_value_from_x(event.pos[0])
                        if v is not None:
                            self._set_biomass_display_scale(v)
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
                        self._plot_scroll_snap_to_live_if_at_end()
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
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_evalint):
                        self._dragging_evalint = False
                    elif (event.type == pg.MOUSEBUTTONUP and event.button == 1
                          and self._dragging_biomass_scale):
                        self._dragging_biomass_scale = False
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
    def _display_name(self, fid: str, lang: Optional[str] = None) -> str:
        """An FG id as a label in the display language.

        English replaces underscores with spaces and capitalises the
        first letter, e.g. ``pelagic_fish`` -> ``Pelagic fish``; the other
        languages take the name from ``i18n.FG_NAMES``. A random-action
        baseline series (``<fid>_rnd``) is its FG's name plus " rnd".
        """
        if not fid:
            return fid
        lang = self._lang if lang is None else lang
        base, suffix = str(fid), ""
        if base.endswith("_rnd"):
            base, suffix = base[:-4], " rnd"
        name = i18n.fg_name(lang, base)
        if name is None:
            s = base.replace("_", " ").strip()
            name = s[:1].upper() + s[1:] if s else s
        return name + suffix

    def _t(self, text: str, **fields) -> str:
        """``text`` (an English template) in the display language."""
        return i18n.translate(self._lang, text, **fields)

    def _t_width(self, font, text: str, **fields) -> int:
        """Widest rendering of ``text`` over every language.

        Layout that follows a label (slider tracks, buttons) is sized by
        this, so switching language never moves a control.
        """
        return max(font.size(i18n.translate(code, text, **fields))[0]
                   for code in i18n.CODES)

    @staticmethod
    def _member_lines(entries, lang: str = i18n.DEFAULT) -> list:
        """Tooltip lines for a library ``member_species`` list.

        Each entry is ``{name, share}`` (share optional, a fraction of the
        group's biomass) or a bare name; ``{name: others, count: n}`` is
        the remainder of n species. Names are shown in ``lang``. A
        single-species group gets no lines, so its panel has no tooltip.
        """
        lines = []
        for entry in entries or ():
            if isinstance(entry, Mapping):
                name, share = entry.get("name"), entry.get("share")
                count = entry.get("count")
            else:
                name, share, count = entry, None, None
            name = str(name or "").strip()
            if not name:
                continue
            if name.lower() == "others" and count is not None:
                name = i18n.translate(lang, "others ({count} species)",
                                      count=count)
            else:
                name = i18n.species_name(lang, name)
            name = name[:1].upper() + name[1:]
            try:
                lines.append(f"{name}  {float(share) * 100.0:.0f} %"
                             if share is not None else name)
            except (TypeError, ValueError):
                lines.append(name)
        return lines if len(lines) > 1 else []

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
        elif key == pg.K_n:
            self._rolling_heatmap = not self._rolling_heatmap
        elif key == pg.K_r:
            self._log_plot = not self._log_plot
        elif key == pg.K_p:
            self._toggle_plot()
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

    def _biomass_scale_pos_from_value(self, value: float) -> float:
        import math
        lo = max(1e-6, float(self._biomass_scale_min))
        hi = max(lo * 1.0001, float(self._biomass_scale_max))
        v = max(lo, min(hi, float(value)))
        return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))

    def _biomass_scale_value_from_pos(self, frac: float) -> float:
        import math
        lo = max(1e-6, float(self._biomass_scale_min))
        hi = max(lo * 1.0001, float(self._biomass_scale_max))
        f = max(0.0, min(1.0, float(frac)))
        return math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo)))

    def _hit_biomass_scale_slider(self, pos) -> bool:
        return self._hit_rect(self._biomass_scale_track_rect, pos)

    def _set_biomass_display_scale(self, value: float) -> None:
        """Pin the biomass display scale to ``value`` (manual override).

        Heatmaps always use their rollout-start reference. Adjusting the
        slider also pins the biomass plot axis to a multiple of 100%.
        """
        lo = float(self._biomass_scale_min)
        hi = float(self._biomass_scale_max)
        self._biomass_display_scale = max(lo, min(hi, float(value)))
        self._biomass_scale_manual = True

    def _reset_biomass_display_scale(self) -> None:
        """Restore start-value heatmap colours and an automatic plot y-axis."""
        self._biomass_scale_manual = False
        self._biomass_display_scale = 1.0

    def _biomass_scale_label(self) -> str:
        return self._t("Biomass scale = {value}x",
                       value=f"{self._biomass_display_scale:.2f}")

    def _biomass_scale_value_from_x(self, mx: int) -> Optional[float]:
        rect = self._biomass_scale_track_rect
        if rect is None:
            return None
        rx, _ry, rw, _rh = rect
        track_x = rx + 2
        track_w = max(1, rw - 4)
        frac = (mx - track_x) / float(track_w)
        return self._biomass_scale_value_from_pos(frac)

    def _handle_click(self, pos) -> None:
        try:
            mx, my = pos
        except Exception:
            return
        # Open language menu: a flag picks that language, and any click
        # closes the menu without reaching what lies under it.
        if self._lang_menu_open:
            for rect, code in self._lang_item_rects:
                if self._hit_rect(rect, pos):
                    if code != self._lang:
                        self._lang = code
                        _save_settings(language=code)
                    break
            self._lang_menu_open = False
            self._lang_item_rects = []
            return
        if self._hit_rect(self._lang_button_rect, pos):
            self._lang_menu_open = True
            self._spawn_popup = None
            self._spawn_popup_item_rects = []
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
        # A heatmap group's header row folds the group in or out.
        for rect, title in self._hm_header_rects:
            if self._hit_rect(rect, pos):
                self._toggle_group(title)
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
        if self._hit_rect(self._evalint_lock_rect, pos):
            self._evalint_locked = not self._evalint_locked
            return
        # Heatmap normalisation button: sits in the status bar, so it must
        # be claimed before anything routes the click on to a heatmap.
        if self._hit_rect(self._hm_norm_button_rect, pos):
            self._rolling_heatmap = not self._rolling_heatmap
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
        # Evaluation probe interval slider (row 4, training only).
        if self._hit_rect(self._evalint_track_rect, pos):
            if self._evalint_locked:
                return
            v = self._evalint_value_from_x(mx)
            if v is not None:
                self._evalint_override = int(v)
                self._dragging_evalint = True
            return
        # Klick på etiketten "Biomass scale = …" släpper override:en och
        # återgår till autoskalning (kollas före slider-tracken).
        if self._biomass_scale_label_rect is not None \
                and self._hit_rect(self._biomass_scale_label_rect, pos):
            self._reset_biomass_display_scale()
            return
        if self._hit_biomass_scale_slider(pos):
            v = self._biomass_scale_value_from_x(mx)
            if v is not None:
                self._set_biomass_display_scale(v)
                self._dragging_biomass_scale = True
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
        # Live-knapp bredvid plot-scrollbaren: nollställer scroll-offset
        # så plotten återgår till att följa senaste tick automatiskt.
        if self._live_button_rect is not None \
                and self._hit_rect(self._live_button_rect, pos):
            self._plot_scroll_offset = None
            self._dragging_plot_scroll = False
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
        # Select all / deselect all over this tab's legend entries:
        # all on -> all off, otherwise all on (and no solo series).
        all_rect = getattr(self, "_legend_all_rect", None)
        if all_rect is not None:
            rx, ry, rw, rh = all_rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                ids = self._active_plot_ids()
                on = not all(self._plot_enabled.get(fid, True) for fid in ids)
                for fid in ids:
                    self._plot_enabled[fid] = on
                self._solo = None
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

    _FLAG_W, _FLAG_H = 18, 11

    def _draw_lang_button(self) -> None:
        """The status bar's language button: the current flag and a caret."""
        pg = self._pg
        x, y, w, h = self._lang_button_rect
        active = self._lang_menu_open
        pg.draw.rect(self._screen, (70, 70, 88) if active else (55, 55, 65),
                     (x, y, w, h))
        pg.draw.rect(self._screen,
                     (160, 160, 180) if active else (120, 130, 145),
                     (x, y, w, h), 1)
        fw, fh = self._FLAG_W, self._FLAG_H
        fy = y + (h - fh) // 2
        i18n.draw_flag(pg, self._screen, (x + 4, fy, fw, fh), self._lang)
        cx, cy = x + w - 7, y + h // 2
        pg.draw.polygon(self._screen, (210, 210, 220),
                        [(cx - 3, cy - 1), (cx + 3, cy - 1), (cx, cy + 2)])

    def _draw_lang_menu(self) -> None:
        """The open language menu: one row per flag, under the button.

        Right-aligned with the button and held inside the window; the
        current language is highlighted. Rows are click targets for
        ``_handle_click``.
        """
        self._lang_item_rects = []
        if not self._lang_menu_open or self._lang_button_rect is None:
            return
        pg = self._pg
        font = self._font
        bx, by, bw, bh = self._lang_button_rect
        fw, fh = self._FLAG_W, self._FLAG_H
        item_h, pad = 20, 6
        text_w = max(font.size(name)[0] for _code, name in i18n.LANGUAGES)
        menu_w = pad + fw + 8 + text_w + pad
        menu_h = item_h * len(i18n.LANGUAGES) + 2
        win_w, win_h = self._screen.get_size()
        mx0 = max(0, min(bx + bw - menu_w, win_w - menu_w))
        my0 = max(0, min(by + bh + 2, win_h - menu_h))
        pg.draw.rect(self._screen, (40, 40, 50), (mx0, my0, menu_w, menu_h))
        pg.draw.rect(self._screen, (160, 160, 180),
                     (mx0, my0, menu_w, menu_h), 1)
        for i, (code, name) in enumerate(i18n.LANGUAGES):
            ry = my0 + 1 + i * item_h
            item = (mx0 + 1, ry, menu_w - 2, item_h)
            if code == self._lang:
                pg.draw.rect(self._screen, (75, 85, 110), item)
            i18n.draw_flag(pg, self._screen,
                           (mx0 + pad, ry + (item_h - fh) // 2, fw, fh), code)
            surf = font.render(name, True, (230, 230, 240))
            self._screen.blit(surf, (mx0 + pad + fw + 8,
                                     ry + (item_h - surf.get_height()) // 2))
            self._lang_item_rects.append((item, code))

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
        label_widths = [font.size(self._t(lbl))[0] for lbl, _ in options]
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
            txt = font.render(self._t(label), True, (230, 230, 240))
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

    def _plot_scroll_snap_to_live_if_at_end(self) -> None:
        """Om nuvarande ``_plot_scroll_offset`` ligger på (eller mycket nära)
        det maxvärde där thumb sitter i trackens högerkant, nolla offset så
        plotten återgår till att följa senaste tick automatiskt ("live").

        Utan den här snap-mekaniken hamnar användaren i ett fruset visnings-
        fönster så snart hen dragit i scrollbaren en enda gång — även om
        thumb dras hela vägen till högerkanten fortsätter offset att vara
        satt till ``max_off`` och sluter sig från live-uppdateringar
        eftersom nya data (större xmax) fortsätter komma in. Att sätta
        ``_plot_scroll_offset = None`` ger tillbaka "follow-latest"-läget.
        """
        if self._plot_scroll_offset is None:
            return
        span = getattr(self, "_plot_scroll_span_cache", None)
        if not span:
            return
        data_xmin, data_xmax = span
        win_w_x = float(self._plot_window_width)
        full_span = float(data_xmax - data_xmin)
        if full_span <= win_w_x:
            # Scroll är inte aktiv → återgå direkt till live.
            self._plot_scroll_offset = None
            return
        max_off = float(data_xmax) - win_w_x
        # Snap-tröskel: 2% av synligt fönster (min 1 tick). Ger användaren
        # ett generöst mål-område i trackens högerkant för att gå live.
        snap_eps = max(1.0, 0.02 * win_w_x)
        if float(self._plot_scroll_offset) >= max_off - snap_eps:
            self._plot_scroll_offset = None

    def _hit_plot_area(self, pos) -> bool:
        """Retur True om ``pos`` ligger inom plot-arean (linjerna, ej
        tab-strip/scrollbar/legend). Används av MOUSEWHEEL för att
        avgöra om hjulet ska tolkas som zoom av graf-fönstret."""
        rect = self._plot_area_rect
        if not rect:
            return False
        mx, my = pos
        px0, py0, pw, ph = rect
        return (px0 <= mx <= px0 + pw) and (py0 <= my <= py0 + ph)

    def _handle_plot_wheel_zoom(self, pos, dy: int) -> bool:
        """Zooma graf-fönstret runt musens x-position.

        ``dy`` > 0 = zoom in (mindre fönster), ``dy`` < 0 = zoom ut.
        Behåller den datapunkt musen pekar på under muspekaren så
        zoom känns naturlig. Returnerar True om något ändrades.
        """
        if dy == 0 or not self._hit_plot_area(pos):
            return False
        span = getattr(self, "_plot_scroll_span_cache", None)
        rect = self._plot_area_rect
        if not span or not rect:
            return False
        data_xmin, data_xmax = span
        full_span = float(data_xmax) - float(data_xmin)
        if full_span <= 0.0:
            return False
        px0, _py0, pw, _ph = rect
        if pw <= 4:
            return False
        # Nuvarande synligt x-fönster.
        old_win = float(self._plot_window_width)
        # Om vi för närvarande visar hela data-spannet (scroll ej aktiv),
        # använd det som utgångspunkt så zoom-in från "hela historiken"
        # känns naturligt.
        if full_span <= old_win:
            old_win = full_span
        if self._plot_scroll_offset is None:
            # Live-läge: högerkanten är vid data_xmax.
            old_xmin = float(data_xmax) - old_win
        else:
            old_xmin = float(self._plot_scroll_offset)
        old_xmax = old_xmin + old_win
        # Muspos i tick-koordinater (var vi vill hålla kvar under pekaren).
        mx = float(pos[0])
        frac = (mx - float(px0)) / float(pw)
        frac = max(0.0, min(1.0, frac))
        anchor_x = old_xmin + frac * old_win
        # Ny fönsterbredd — 1.2x per tick, klampad.
        factor = 1.0 / 1.2 if dy > 0 else 1.2
        new_win = old_win * factor
        min_win = float(self._plot_window_width_min)
        max_win = max(min_win, full_span)
        new_win = max(min_win, min(max_win, new_win))
        if abs(new_win - old_win) < 1e-6:
            return False
        new_xmin = anchor_x - frac * new_win
        new_xmax = new_xmin + new_win
        # Klampa mot data-spannet.
        if new_xmin < float(data_xmin):
            new_xmin = float(data_xmin)
            new_xmax = new_xmin + new_win
        if new_xmax > float(data_xmax):
            new_xmax = float(data_xmax)
            new_xmin = new_xmax - new_win
        self._plot_window_width = max(1, int(round(new_win)))
        # Om nya fönstret täcker hela data → gå live (ingen scroll behövs).
        if new_win >= full_span - 1e-6:
            self._plot_scroll_offset = None
        else:
            # Om högerkanten sammanfaller med data_xmax → gå live.
            if new_xmax >= float(data_xmax) - 1e-6:
                self._plot_scroll_offset = None
            else:
                self._plot_scroll_offset = float(new_xmin)
        return True

    def _handle_plot_wheel_pan(self, pos, dx: int) -> bool:
        """Panorera graf-fönstret horisontellt via touchpad-gest eller
        horisontellt scrollhjul.

        ``dx`` > 0 = scroll åt höger → framåt i tid (mot data-slutet).
        ``dx`` < 0 = scroll åt vänster → bakåt i tid (mot data-början).
        Returnerar True om något ändrades.
        """
        if dx == 0 or not self._hit_plot_area(pos):
            return False
        span = getattr(self, "_plot_scroll_span_cache", None)
        rect = self._plot_area_rect
        if not span or not rect:
            return False
        data_xmin, data_xmax = span
        full_span = float(data_xmax) - float(data_xmin)
        if full_span <= 0.0:
            return False
        win_w = float(self._plot_window_width)
        if full_span <= win_w:
            # Hela historiken syns redan → inget att panorera.
            return False
        # Nuvarande vänsterkant.
        if self._plot_scroll_offset is None:
            # Live-läge: vänsterkanten ligger vid data_xmax - win_w.
            old_xmin = float(data_xmax) - win_w
        else:
            old_xmin = float(self._plot_scroll_offset)
        # Flippat tecken: användaren förväntar sig att två-finger-drag åt
        # höger (positiv ``dx`` i pygame på deras system) ska panorera
        # bakåt i tid (dokumentliknande gest — innehållet flyttas åt
        # höger, dvs. äldre data kommer in från vänster). Tidigare
        # gjorde vi tvärtom.
        step = -float(self._plot_pan_step_frac) * win_w * float(dx)
        new_xmin = old_xmin + step
        # Klampa till giltigt intervall.
        max_off = float(data_xmax) - win_w
        min_off = float(data_xmin)
        if new_xmin < min_off:
            new_xmin = min_off
        if new_xmin > max_off:
            new_xmin = max_off
        if abs(new_xmin - old_xmin) < 1e-6 and self._plot_scroll_offset is not None:
            return False
        # Om högerkanten sammanfaller med data-slutet → gå live.
        if new_xmin >= max_off - 1e-6:
            if self._plot_scroll_offset is None:
                return False
            self._plot_scroll_offset = None
        else:
            self._plot_scroll_offset = float(new_xmin)
        return True

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
        self._dirty = False
        screen = self._screen
        screen.fill((18, 18, 22))
        replaying = (not self._recording
                     and self._playback_mode != "live"
                     and self._current_rollout
                     and 0 <= self._playback_idx < len(self._current_rollout))
        # The status bar stays live during replay, except for the month
        # and day length: those belong to the frame whose heatmaps (and
        # daylight frames) are on screen, or the two would disagree.
        self._shown_calendar = (
            self._current_rollout[self._playback_idx].get("calendar")
            if replaying else self._calendar)
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
        if replaying:
            swap = self._swap_in_frame(self._current_rollout[self._playback_idx])
        try:
            self._draw_heatmaps()
        finally:
            if swap is not None:
                self._swap_out_frame(swap)
        if self._plot_collapsed:
            self._clear_plot_hit_rects()
        elif self._plot_anim_img is not None:
            # Rolling: the captured plot, cut off by the moving edge.
            self._clear_plot_hit_rects()
            self._screen.blit(self._plot_anim_img, self._plot_rect[:2])
        else:
            self._draw_plot()
        self._draw_splitter()
        self._draw_playback_bar()
        self._draw_member_tooltip()
        # Spawn-popup ritas sist så den ligger ovanpå allt annat.
        self._draw_spawn_popup()
        self._draw_lang_menu()
        self._pg.display.flip()

    def _swap_in_frame(self, frame: dict) -> dict:
        """Tillfälligt ersätt live-state med en snapshot. Returnerar
        ett ``saved``-dict som ``_swap_out_frame`` använder för att
        återställa."""
        saved = {
            "biomass": self._biomass,
            "totals": self._totals,
            "b0": self._b0,
            "heatmap_vmax0": self._heatmap_vmax0,
            "loss_breakdown": self._loss_breakdown,
            "diet_breakdown": self._diet_breakdown,
            "predation_breakdown": self._predation_breakdown,
            "action_fracs": self._action_fracs,
            "tick": self._tick,
            "calendar": self._calendar,
        }
        # Dekomprimera biomass-fält on demand: om ``_capture_frame``
        # lagrat ``_CompressedArray``-wrappers packar vi upp dem här så
        # ritkoden nedströms alltid ser vanliga ndarrays.
        raw_biomass = frame.get("biomass", {})
        self._biomass = {
            fid: _maybe_unwrap_array(v) for fid, v in raw_biomass.items()
        }
        self._totals = frame.get("totals", {})
        self._b0 = frame.get("b0", {})
        self._heatmap_vmax0 = frame.get("heatmap_vmax0", self._heatmap_vmax0)
        self._loss_breakdown = frame.get("loss_breakdown", {})
        self._diet_breakdown = frame.get("diet_breakdown", {})
        self._predation_breakdown = frame.get("predation_breakdown", {})
        self._action_fracs = frame.get("action_fracs", {})
        self._tick = int(frame.get("tick", 0))
        self._calendar = frame.get("calendar")
        return saved

    def _swap_out_frame(self, saved: dict) -> None:
        self._biomass = saved["biomass"]
        self._totals = saved["totals"]
        self._b0 = saved["b0"]
        self._heatmap_vmax0 = saved["heatmap_vmax0"]
        self._loss_breakdown = saved["loss_breakdown"]
        self._diet_breakdown = saved["diet_breakdown"]
        self._predation_breakdown = saved["predation_breakdown"]
        self._action_fracs = saved["action_fracs"]
        self._tick = saved["tick"]
        self._calendar = saved["calendar"]

    def _draw_playback_bar(self) -> None:
        """Rita uppspelningskontroller under heatmap-griden.

        Layout: [<<] [play/pause] [>>] [-]  speed=X fps  [+]   pause train
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
        # användaren ännu inte interagerat. Toggle 2 Hz. In any playback
        # mode: a viewer that has scrubbed is paused on the new film's
        # last frame, which looks the same as the old one's.
        blinking = (not disabled
                    and self._playback_blink_until > time.monotonic())
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
            surf = self._font.render(self._t(label), True, fg)
            self._screen.blit(surf,
                              (bx + (bw - surf.get_width()) // 2,
                               by + (22 - surf.get_height()) // 2))
            self._playback_rects.append((rect, action))
            bx += bw + 4

        # Frame indicator on the right, in a zone of stable width (the
        # widest variant at the current counts) so the scrub slider does
        # not change size when the text switches between live / frame /
        # recording. Widths are measured with font.size: a rendered
        # surface that is never blitted leaks (sections 60 and 189).
        # When the full texts would leave the scrub slider less than
        # ``_MIN_SCRUB_W`` (a narrow window, e.g. with the plot folded
        # in, section 191), the compact texts are used instead, so the
        # slider stays visible and draggable.
        n = len(self._current_rollout)
        n_pending = len(self._pending_rollout)
        _max_n = max(n, 1)
        track_x = bx + 8
        # Templates (English source) and their fields; the reserved width
        # is the widest variant in any language, so the scrub slider does
        # not change size with the language either.
        for compact in (False, True):
            if compact:
                live_t, frame_t = "live ({n})", "{i}/{n}"
                rec_t, off_t = "rec ({n})", "no capture"
            else:
                live_t, frame_t = "live  (saved frames: {n})", "frame = {i}/{n}"
                rec_t, off_t = "recording... ({n} frames)", "capture off"
            live_txt = self._t(live_t, n=n)
            frame_txt = self._t(frame_t, i=self._playback_idx + 1, n=n)
            rec_txt = self._t(rec_t, n=n_pending)
            off_txt = self._t(off_t)
            _ind_reserve_w = max(
                self._t_width(self._font, live_t, n=_max_n),
                self._t_width(self._font, frame_t, i=_max_n, n=_max_n),
                self._t_width(self._font, rec_t, n=n_pending),
                self._t_width(self._font, off_t))
            ind_zone_left = x + w - _ind_reserve_w - 10
            if ind_zone_left - 10 - track_x >= self._MIN_SCRUB_W:
                break
        self._playback_compact = compact
        if self._playback_mode == "live":
            ind = live_txt
        else:
            ind = frame_txt
        if self._recording:
            ind = rec_txt
        if not self._capture:
            ind = off_txt
        ind_surf = self._font.render(ind, True, (200, 200, 215))
        # Höger-justera den faktiska texten inom den reserverade zonen
        # så att den alltid hamnar vid samma högerkant.
        ind_x = x + w - ind_surf.get_width() - 10
        self._screen.blit(ind_surf,
                          (ind_x,
                           y + (h - ind_surf.get_height()) // 2))

        # Scrub-slider (timeline) mellan knapparna och frame-indikatorn.
        # Disabled samma villkor som transport-knapparna (recording eller
        # ingen film). Klick/drag sätter ``_playback_idx`` och pausar.
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

    def _sim_time_label(self) -> str:
        """Simulated time at the current tick, as ``"2 y 40 d"``.

        Integer arithmetic on hours, so a whole year lands on exactly
        "1 y 0 d" instead of "0 y 364 d" through float error. The part
        of a day a sub-daily tick leaves over is dropped rather than
        rounded: the tick counter beside it already carries the finer
        resolution, and a day that ticks over early reads as wrong.
        """
        total_hours = int(self._tick) * int(self.tick_hours)
        days = total_hours // HOURS_PER_DAY
        years, rest_days = divmod(days, DAYS_PER_YEAR)
        return self._t("{years} y {days} d", years=years, days=rest_days)

    def _draw_status_bar(self) -> None:
        pg = self._pg
        x, y, w, h = self._status_rect
        parts = [self._t("mode = {mode}", mode=self._t(str(self.mode))),
                 self._t("tick = {tick}", tick=self._tick),
                 self._t("time = {time}", time=self._sim_time_label())]
        # Daylight calendar: the month is kept with mode/tick/time when
        # the line is too long. The day length (what the frame around
        # every heatmap shows, in hours) goes after gen/iter/T, so it is
        # the first of them to be dropped - the frame still shows it.
        calendar = self._shown_calendar
        n_fixed = 3
        if calendar:
            parts.append(self._t("month = {month}", month=i18n.month_name(
                self._lang, calendar["month"])[:3]))
            n_fixed = 4
        for k in ("gen", "iter", "T"):
            if k in self._status:
                v = self._status[k]
                if isinstance(v, float):
                    parts.append(f"{k} = {v:.3f}")
                else:
                    parts.append(f"{k} = {v}")
        if calendar:
            parts.append(self._t("daylight = {hours} h",
                                 hours=f"{calendar['day_length_h']:.1f}"))
        if self._paused:
            parts.append(self._t("[PAUSED]"))
        if self._log_heatmap:
            parts.append("hm:log")
        if self._log_plot:
            parts.append(self._t("plot:log"))
        if self._solo:
            parts.append(f"solo = {self._display_name(self._solo)}")
        parts.append(self._t("grid = {w}x{h}", w=self.grid_w, h=self.grid_h))
        parts.append(f"fps = {self._fps_value:4.1f}")
        # ---- Heatmap normalisation button (row 1, right aligned) -----
        # The ONLY always-visible control for the reference: 'n' is not
        # discoverable on its own, and the biomass-scale slider that
        # would have been its natural neighbour exists in inference mode
        # only - so during training there was nothing to see at all.
        # Styled like the plot's Live button: green while the non-default
        # mode is active, so the state reads at a glance.
        #
        # The button owns a fixed slot at the right end of the row and the
        # TEXT is what yields: a narrow window with a long status line
        # would otherwise either push the button off the edge or slide it
        # under the text. ``parts`` is already ordered by importance, so
        # trailing fields (fps, then grid, ...) are dropped until the rest
        # fits. The first three - mode, tick and simulated time - are
        # never dropped, nor is the month when there is a calendar.
        rolling = self._rolling_heatmap
        norm_key = "norm: rolling [n]" if rolling else "norm: fixed [n]"
        norm_surf = self._font.render(
            self._t(norm_key), True,
            (220, 240, 220) if rolling else (210, 220, 230))
        btn_h = self._font.get_height() + 4
        btn_y = y + 2
        # The language flag owns the right end; the norm button sits left
        # of it, as wide as its label in the widest language.
        lang_w, lang_h = 34, btn_h
        lang_x = x + w - lang_w - 8
        self._lang_button_rect = (lang_x, btn_y, lang_w, lang_h)
        self._draw_lang_button()
        btn_w = self._t_width(self._font, norm_key) + 12
        btn_x = lang_x - btn_w - 6

        available = btn_x - (x + 6) - 12
        while len(parts) > n_fixed and self._font_big.size("  ".join(parts))[0] > available:
            parts.pop()
        text = "  ".join(parts)
        surf = self._font_big.render(text, True, (230, 230, 235))
        self._screen.blit(surf, (x + 6, y + 3))

        pg.draw.rect(self._screen,
                     (55, 90, 60) if rolling else (55, 55, 65),
                     (btn_x, btn_y, btn_w, btn_h))
        pg.draw.rect(self._screen,
                     (110, 180, 130) if rolling else (120, 130, 145),
                     (btn_x, btn_y, btn_w, btn_h), 1)
        self._screen.blit(norm_surf, (btn_x + (btn_w - norm_surf.get_width()) // 2,
                                      btn_y + 2))
        self._hm_norm_button_rect = (btn_x, btn_y, btn_w, btn_h)

        # ---- Rollout-längd-slider (rad 2 i status-baren) -------------
        # Layout: vänster label "rollout ticks = N", track, höger gränser.
        # Range = [_ticks_min, _ticks_max] (log-skala). Vid drag visas
        # det realtidsuppdaterade värdet inline.
        cur = self.get_ticks()
        marker = "*" if self._ticks_override is not None else ""
        label = self._t("Probe ticks{marker} = {value}", marker=marker, value=cur)
        lbl_surf = self._font.render(label, True, (210, 210, 220))
        row2_y = y + 22
        row2_h = 18
        self._screen.blit(lbl_surf, (x + 6, row2_y + 2))
        lo_surf = self._font.render(str(self._ticks_min), True, (150, 150, 160))
        hi_surf = self._font.render(str(self._ticks_max), True, (150, 150, 160))
        _is_inference = str(self.mode).lower().startswith("infer")
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
        # Every label is measured in its widest language, so switching
        # language never moves the tracks.
        _lbl2_preview_w = self._t_width(
            self._font, "Perturbation ticks{marker} = {value}",
            marker=marker2_preview, value=cur2_preview)
        _bio_label_w = (self._t_width(
            self._font, "Biomass scale = {value}x",
            value=f"{self._biomass_display_scale:.2f}")
            if _is_inference else 0)
        _lbl4_preview_w = (0 if _is_inference else self._t_width(
            self._font, "Evaluation probe interval{marker} = {value}",
            marker="*" if self._evalint_override is not None else "",
            value=self.get_eval_interval()))
        _lbl1_w = self._t_width(self._font, "Probe ticks{marker} = {value}",
                                marker=marker, value=cur)
        _label_w = max(_lbl1_w, _lbl2_preview_w, _bio_label_w,
                       _lbl4_preview_w)
        # Track-rect: börjar efter label (+ liten gutter), slutar före
        # hi-label (+ gutter). Mappar mot fönsterbredden.
        try:
            _lo2_preview_w = self._font.size(str(self._neval_min))[0]
        except Exception:
            _lo2_preview_w = lo_surf.get_width()
        _bio_lo_txt = f"{self._biomass_scale_min:.2g}x"
        _bio_hi_txt = f"{self._biomass_scale_max:.0f}x"
        _bio_lo_w = self._font.size(_bio_lo_txt)[0] if _is_inference else 0
        _lo_w = max(lo_surf.get_width(), _lo2_preview_w, _bio_lo_w)
        track_x = x + 6 + _label_w + 12 + _lo_w + 6
        # Begränsa sliderns högerkant så att den slutar där plot-rutan
        # börjar (annars sträcker den sig över hela fönsterbredden och
        # täcker grafritarrutan, vilket ser fult ut och är onödigt långt).
        try:
            plot_left = min(x + w - 8, max(int(self._expanded_plot_x),
                                           int(self._status_track_right_min)))
        except Exception:
            plot_left = x + w
        # Lämna plats för låsikon (16 px + 6 px gutter) mellan track och
        # hi-label. I inference-läget döljs låsikonerna och
        # n_eval_ticks-slidern helt, så ingen plats reserveras.
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
        _bio_hi_w = self._font.size(_bio_hi_txt)[0] if _is_inference else 0
        _hi_w = max(hi_surf.get_width(), _hi2_preview_w, _bio_hi_w)
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
            self._evalint_track_rect = None
            self._evalint_lock_rect = None
            row3_y = y + 40
            row3_h = 18
            bio_lbl = self._font.render(
                self._biomass_scale_label(),
                True,
                (210, 210, 220) if self._biomass_scale_manual
                else (150, 150, 160))
            lo3_surf = self._font.render(_bio_lo_txt, True, (150, 150, 160))
            hi3_surf = self._font.render(_bio_hi_txt, True, (150, 150, 160))
            self._screen.blit(bio_lbl, (x + 6, row3_y + 2))
            # Clicking restores 1x start colours and the automatic plot axis.
            self._biomass_scale_label_rect = (
                x + 4, row3_y, bio_lbl.get_width() + 4,
                self._font.get_height() + 4)
            track3_y = row3_y + (row3_h - 6) // 2
            track3_h = 6
            self._screen.blit(lo3_surf,
                              (track_x - 6 - lo3_surf.get_width(),
                               row3_y + 2))
            self._screen.blit(hi3_surf,
                              (track_x + track_w + 6, row3_y + 2))
            pg.draw.rect(self._screen, (60, 60, 70),
                         (track_x, track3_y, track_w, track3_h))
            pg.draw.rect(self._screen, (110, 110, 125),
                         (track_x, track3_y, track_w, track3_h), 1)
            try:
                df3 = self._biomass_scale_pos_from_value(1.0)
                dx3 = int(track_x + df3 * (track_w - 1))
                pg.draw.line(self._screen, (140, 140, 150),
                             (dx3, track3_y - 2),
                             (dx3, track3_y + track3_h + 2), 1)
            except Exception:
                pass
            try:
                frac3 = self._biomass_scale_pos_from_value(
                    self._biomass_display_scale)
            except Exception:
                frac3 = self._biomass_scale_pos_from_value(1.0)
            hx3 = int(track_x + frac3 * (track_w - 1))
            knob3_col = ((120, 205, 170) if self._biomass_scale_manual
                         else (110, 110, 125))
            pg.draw.circle(self._screen, knob3_col,
                           (hx3, track3_y + track3_h // 2), 6)
            pg.draw.circle(self._screen, (40, 40, 50),
                           (hx3, track3_y + track3_h // 2), 6, 1)
            self._biomass_scale_track_rect = (
                track_x - 2, track3_y - 6, track_w + 4, track3_h + 12)
            return
        self._biomass_scale_track_rect = None
        self._biomass_scale_label_rect = None
        # Identisk layout som ticks-slidern men med "n_eval_ticks" som
        # label. Aktiv främst under träning; värdet konsumeras av
        # train.py mellan ARS-iterationer.
        cur2 = self.get_neval_ticks()
        marker2 = "*" if self._neval_override is not None else ""
        label2 = self._t("Perturbation ticks{marker} = {value}",
                         marker=marker2, value=cur2)
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
        _label2_w = max(lbl2_surf.get_width(), _label_w)
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

        # ---- Evaluation probe interval (row 4, training only) --------
        # Same track geometry as row 3; linear 1-50 updates.
        lbl4_surf = self._font.render(self._evalint_label(), True, (210, 210, 220))
        row4_y = y + 58
        row4_h = 18
        self._screen.blit(lbl4_surf, (x + 6, row4_y + 2))
        lo4_surf = self._font.render(str(self._evalint_min), True, (150, 150, 160))
        hi4_surf = self._font.render(str(self._evalint_max), True, (150, 150, 160))
        track4_y = row4_y + (row4_h - 6) // 2
        track4_h = 6
        self._screen.blit(lo4_surf, (track2_x - 6 - lo4_surf.get_width(), row4_y + 2))
        self._screen.blit(
            hi4_surf,
            (track2_x + track2_w + lock_gutter + lock_size + 6, row4_y + 2))
        pg.draw.rect(self._screen, (60, 60, 70),
                     (track2_x, track4_y, track2_w, track4_h))
        pg.draw.rect(self._screen, (110, 110, 125),
                     (track2_x, track4_y, track2_w, track4_h), 1)
        dx4 = int(track2_x + self._evalint_pos_from_value(self._evalint_default)
                  * (track2_w - 1))
        pg.draw.line(self._screen, (140, 140, 150),
                     (dx4, track4_y - 2), (dx4, track4_y + track4_h + 2), 1)
        hx4 = int(track2_x + self._evalint_pos_from_value(self.get_eval_interval())
                  * (track2_w - 1))
        hcol4 = ((220, 220, 90) if self._evalint_override is not None
                 else (200, 200, 210))
        pg.draw.circle(self._screen, hcol4, (hx4, track4_y + track4_h // 2), 6)
        pg.draw.circle(self._screen, (40, 40, 50), (hx4, track4_y + track4_h // 2), 6, 1)
        self._evalint_track_rect = (track2_x - 2, track4_y - 6,
                                    track2_w + 4, track4_h + 12)
        self._evalint_lock_rect = self._draw_lock_icon(
            lock2_x, row4_y + (row4_h - lock_size) // 2, lock_size,
            self._evalint_locked)

    def _evalint_label(self) -> str:
        marker = "*" if self._evalint_override is not None else ""
        return self._t("Evaluation probe interval{marker} = {value}",
                       marker=marker, value=self.get_eval_interval())

    def _draw_heatmaps(self) -> None:
        ox, oy = self._heatmap_origin
        oy -= self._hm_scroll_y
        vx, vy, vw, vh = self._hm_view_rect
        # Slider-rektangellistan byggs om varje frame så stale entries
        # från föregående layout aldrig kan trigga drag i fel panel.
        self._slider_rects = []
        # Samma sak för spawn-dropdown-rektanglarna.
        self._spawn_dropdown_rects = []
        self._member_rects = []
        self._hm_header_rects = []
        view = self._pg.Rect(vx, vy, vw, vh)
        # Everything a panel draws is clipped to the viewport, so a panel
        # scrolled half out of view cannot paint over the status bar or
        # the playback bar - and to its group's span, so a folding group
        # slides in under its header (section 214).
        try:
            self._screen.set_clip(view)
            for title, y_off in self._hm_headers:
                self._draw_group_header(title, ox, oy + y_off, vw)
            for fid, x_off, y_off in self._hm_slots:
                g0, g1 = self._hm_group_spans[self._hm_slot_group[fid]]
                span = view.clip(self._pg.Rect(vx, oy + g0, vw, g1 - g0))
                px = ox + x_off
                py = oy + y_off
                if (span.height <= 0 or py >= span.bottom
                        or py + self._panel_h <= span.top):
                    continue
                self._screen.set_clip(span)
                n0 = (len(self._slider_rects), len(self._spawn_dropdown_rects),
                      len(self._member_rects))
                self._draw_one_heatmap(fid, px, py)
                # A control slid under the header takes no clicks.
                for name, start in zip(("_slider_rects",
                                        "_spawn_dropdown_rects",
                                        "_member_rects"), n0):
                    rects = getattr(self, name)
                    rects[start:] = self._clip_hit_rects(
                        rects[start:], (span.top, span.height))
        finally:
            self._screen.set_clip(None)
        # Hit rects are clipped to the viewport in y (x and width are kept:
        # ``_slider_value_from_x`` reads the track from them), so a control
        # scrolled out of view never claims a click meant for the status
        # bar or the playback bar.
        self._slider_rects = self._clip_hit_rects(self._slider_rects)
        self._spawn_dropdown_rects = self._clip_hit_rects(
            self._spawn_dropdown_rects)
        self._member_rects = self._clip_hit_rects(self._member_rects)
        self._hm_header_rects = self._clip_hit_rects(self._hm_header_rects)
        self._draw_hm_scrollbar()

    _FOLD_BOX = 11

    def _draw_group_header(self, title: str, x: int, y: int, w: int) -> None:
        """A group's header row: a fold box, its name, a rule (198, 214).

        The box holds a minus while the group is open (a click folds it
        in) and a plus while it is folded or folding (a click folds it
        out). The whole row is the click target.
        """
        pg = self._pg
        h = self._hm_header_h
        vy, vh = self._hm_view_rect[1], self._hm_view_rect[3]
        if y >= vy + vh or y + h <= vy:
            return
        pg.draw.rect(self._screen, (28, 28, 36), (x, y, w, h))
        b = self._FOLD_BOX
        bx, by = x + 6, y + (h - b) // 2
        pg.draw.rect(self._screen, (36, 36, 46), (bx, by, b, b))
        pg.draw.rect(self._screen, (150, 150, 170), (bx, by, b, b), 1)
        mid_x, mid_y = bx + b // 2, by + b // 2
        sign = (215, 215, 230)
        pg.draw.line(self._screen, sign, (bx + 3, mid_y), (bx + b - 4, mid_y), 1)
        if self._group_target.get(title, 1.0) < 0.5:
            pg.draw.line(self._screen, sign, (mid_x, by + 3),
                         (mid_x, by + b - 4), 1)
        surf = self._font_big.render(self._t(title), True, (225, 225, 235))
        self._screen.blit(surf, (bx + b + 7, y + (h - surf.get_height()) // 2))
        pg.draw.line(self._screen, (70, 70, 85),
                     (x, y + h - 1), (x + w - 1, y + h - 1), 1)
        self._hm_header_rects.append(((x, y, w, h), title))

    def _clip_hit_rects(self, entries: list, span=None) -> list:
        """Clip hit rects in y to the viewport, or to ``span`` (y, height)."""
        if span is None:
            _vx, vy, _vw, vh = self._hm_view_rect
        else:
            vy, vh = span
        out = []
        for entry in entries:
            rx, ry, rw, rh = entry[0]
            y0 = max(ry, vy)
            y1 = min(ry + rh, vy + vh)
            if y1 > y0:
                out.append(((rx, y0, rw, y1 - y0),) + tuple(entry[1:]))
        return out

    def _usable_screen_size(self) -> Optional[tuple]:
        """(width, height) the window may start at, or None if unknown.

        The first desktop's size minus room for the title bar and panels
        in height; ``MARELD_VIZ_MAX_HEIGHT`` (px) overrides the height.
        """
        try:
            self._pg.display.init()
            sizes = self._pg.display.get_desktop_sizes()
            if sizes:
                width, height = int(sizes[0][0]), int(sizes[0][1])
            else:
                info = self._pg.display.Info()
                width, height = int(info.current_w), int(info.current_h)
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        height -= 100
        env = os.environ.get("MARELD_VIZ_MAX_HEIGHT")
        if env:
            try:
                height = max(200, int(env))
            except ValueError:
                pass
        return width, height

    def _compute_layout(self) -> None:
        """Derive every layout rect from the window size and the splitter.

        The heatmap area takes ``_split_frac`` of the window width (at
        least one reference panel, leaving the plot ``_min_plot_w``). Its
        panels keep ``min(4, n)`` columns and grow with the area; when
        the area is too narrow for the reference cell size, columns are
        dropped instead. Rows that do not fit in the viewport scroll.
        With the plot collapsed the splitter becomes a strip right at
        the heatmap area's edge and the plot gets no width; the heatmaps
        keep the area they had when it folded (``_hm_area_fixed``) and
        ``_toggle_plot`` rolls the window edge in to the strip.
        """
        W, H = int(self._win_w), int(self._win_h)
        status_h, playback_h, pad = self._status_h, self._playback_h, self._pad
        sb_w = self._hm_sb_w
        sp_w = (self._collapsed_strip_w if self._plot_collapsed
                else self._splitter_w)
        if self._hm_area_fixed is not None:
            # Folded or rolling: the heatmaps stay as they were arranged;
            # only a narrower window can squeeze them.
            hm_area_w = min(int(self._hm_area_fixed), W - sp_w)
        else:
            hm_area_w = int(round(self._split_frac * W))
            hm_area_w = min(hm_area_w, W - sp_w - self._min_plot_w)
        hm_area_w = max(hm_area_w, self._min_hm_area_w)
        # Where the plot starts when expanded; the status-bar sliders end
        # there in both states, so they do not jump on a toggle.
        self._expanded_plot_x = hm_area_w + self._splitter_w
        view_h = max(60, H - status_h - playback_h - pad)
        avail_w = hm_area_w - sb_w
        n = len(self.fg_ids)
        cols, self.cell_px = self._heatmap_columns(avail_w)
        self._cols = cols
        self._panel_w = self.grid_w * self.cell_px + 2 * pad
        self._panel_h = self.grid_h * self.cell_px + self._panel_extra_h
        self._heatmap_origin = (0, status_h)
        self._hm_view_rect = (0, status_h, avail_w, view_h)
        self._layout_hm_block()
        self._splitter_rect = (hm_area_w, status_h, sp_w, view_h)
        plot_x = hm_area_w + sp_w
        if self._plot_collapsed:
            self._plot_rect = (plot_x, status_h, 0, view_h)
        else:
            self._plot_rect = (plot_x, status_h,
                               max(1, W - plot_x - pad // 2), view_h)
        self._plot_toggle_rect = (hm_area_w, status_h, sp_w,
                                  min(self._plot_toggle_h, view_h))
        self._status_rect = (0, 0, W, status_h)
        self._playback_rect = (0, status_h + view_h, W, playback_h)

    def _heatmap_columns(self, avail_w: int) -> tuple:
        """``(columns, cell_px)`` for a heatmap viewport ``avail_w`` wide:
        the most columns (at most ``min(4, n)``) whose cells reach the
        reference size, one column otherwise."""
        pad = self._pad

        def cell_for(cols: int) -> int:
            return (avail_w // cols - 2 * pad) // max(1, self.grid_w)

        cols = 1
        for c in range(min(4, len(self.fg_ids)), 0, -1):
            if cell_for(c) >= self._min_cell_px:
                cols = c
                break
        return cols, max(1, cell_for(cols))

    def _layout_hm_block(self) -> None:
        """Panel and group-header offsets inside the scrolled block.

        A group whose header is folded (``_group_open`` below 1) keeps
        that share of its rows' height: its panels slide up under the
        header and paint only inside its span (``_hm_group_spans``), and
        the groups below follow. Fully open it is the layout of section
        198. Also sets the scrollbar and clamps the scroll position.
        """
        cols = self._cols
        self._hm_slots = []
        self._hm_headers = []
        self._hm_group_spans = []
        self._hm_slot_group = {}
        y_off = 0
        rows = 0
        for g, (title, fids) in enumerate(self._hm_groups):
            if title is not None:
                self._hm_headers.append((title, y_off))
                y_off += self._hm_header_h
            group_rows = (len(fids) + cols - 1) // cols
            full_h = group_rows * self._panel_h
            frac = (1.0 if title is None
                    else max(0.0, min(1.0, self._group_open.get(title, 1.0))))
            shown_h = int(round(full_h * frac))
            shift = shown_h - full_h
            for i, fid in enumerate(fids):
                self._hm_slots.append((fid, (i % cols) * self._panel_w,
                                       y_off + shift
                                       + (i // cols) * self._panel_h))
                self._hm_slot_group[fid] = g
            self._hm_group_spans.append((y_off, y_off + shown_h))
            rows += group_rows
            y_off += shown_h
        self._rows = rows
        self._hm_block_h = y_off
        _vx, vy, avail_w, view_h = self._hm_view_rect
        # The slack never exceeds what keeps the current position valid.
        self._hm_scroll_slack = max(0, min(
            self._hm_scroll_slack,
            self._hm_scroll_y + view_h - self._hm_block_h))
        self._hm_scrollbar_rect = (
            (avail_w, vy, self._hm_sb_w, view_h)
            if self._hm_block_h + self._hm_scroll_slack > view_h else None)
        if self._hm_scrollbar_rect is None:
            self._dragging_hm_scroll = False
        self._hm_scroll_y = max(0, min(self._hm_max_scroll(), self._hm_scroll_y))

    def _toggle_group(self, title: str) -> None:
        """Fold a heatmap group in under its header, or out again.

        The group's panels glide (``_fold_anim_s``, ease-out, as the plot
        fold of section 194) and the groups below follow; every step is
        painted before the next. The scroll position is held through the
        glide (``_hm_scroll_slack``), so the clicked header stays under
        the pointer even when the fold shrinks the block below the view.
        """
        if title not in self._group_open:
            return
        target = 0.0 if self._group_target.get(title, 1.0) >= 0.5 else 1.0
        self._group_target[title] = target
        # Remember it; titles of groups this viewer does not show keep
        # their saved state.
        saved = _load_settings().get("folded_groups")
        folded = [t for t in (saved if isinstance(saved, list) else [])
                  if isinstance(t, str) and t != title]
        if target == 0.0:
            folded.append(title)
        _save_settings(folded_groups=folded)
        self._spawn_popup = None
        self._spawn_popup_item_rects = []
        start = float(self._group_open[title])
        scroll0 = self._hm_scroll_y
        dur = float(self._fold_anim_s)
        t0 = time.monotonic()
        while True:
            f = 1.0 if dur <= 0 else min(1.0, (time.monotonic() - t0) / dur)
            e = 1.0 - (1.0 - f) ** 3
            self._group_open[title] = start + (target - start) * e
            # Room for exactly the held position; _layout_hm_block trims it.
            self._hm_scroll_slack = 1 << 30
            self._hm_scroll_y = scroll0
            self._layout_hm_block()
            if f >= 1.0:
                return
            self._render_full()
            self._pg.event.pump()

    def _hm_max_scroll(self) -> int:
        return max(0, self._hm_block_h + self._hm_scroll_slack
                   - self._hm_view_rect[3])

    def _set_hm_scroll(self, y: float) -> None:
        self._hm_scroll_y = int(max(0, min(self._hm_max_scroll(), y)))
        # Scrolling up gives up the blank room a fold left below.
        self._hm_scroll_slack = max(0, min(
            self._hm_scroll_slack,
            self._hm_scroll_y + self._hm_view_rect[3] - self._hm_block_h))
        # The spawn popup is anchored to a dropdown that has just moved.
        self._spawn_popup = None
        self._spawn_popup_item_rects = []

    def _draw_hm_scrollbar(self) -> None:
        self._hm_thumb_rect = None
        if self._hm_scrollbar_rect is None:
            return
        pg = self._pg
        sx, sy, sw, sh = self._hm_scrollbar_rect
        pg.draw.rect(self._screen, (30, 30, 38), (sx, sy, sw, sh))
        pg.draw.rect(self._screen, (60, 60, 70), (sx, sy, sw, sh), 1)
        max_scroll = self._hm_max_scroll()
        th = max(24, int(sh * sh / max(
            1, self._hm_block_h + self._hm_scroll_slack)))
        ty = sy + int((sh - th) * self._hm_scroll_y / max(1, max_scroll))
        thumb = (sx + 2, ty, sw - 4, th)
        col = (150, 150, 170) if self._dragging_hm_scroll else (110, 110, 130)
        pg.draw.rect(self._screen, col, thumb)
        pg.draw.rect(self._screen, (170, 170, 190), thumb, 1)
        self._hm_thumb_rect = (sx, ty, sw, th)

    def _draw_splitter(self) -> None:
        pg = self._pg
        x, y, w, h = self._splitter_rect
        active = self._dragging_splitter or self._splitter_hover
        pg.draw.rect(self._screen, (44, 44, 54) if active else (30, 30, 38),
                     (x, y, w, h))
        # The drag grip; folded, the window's own edges set the heatmap
        # area instead (section 195).
        if not self._plot_collapsed:
            grip = (170, 170, 190) if active else (90, 90, 105)
            cx = x + w // 2
            cy = y + h // 2
            for dy in (-8, -4, 0, 4, 8):
                pg.draw.circle(self._screen, grip, (cx, cy + dy), 1)
        # Fold button at the top: the chevron points the way the plot
        # moves (left = fold it in under the heatmaps, right = out again).
        bh = min(self._plot_toggle_h, h)
        hover = self._plot_toggle_hover
        pg.draw.rect(self._screen, (70, 70, 88) if hover else (52, 52, 64),
                     (x, y, w, bh))
        pg.draw.rect(self._screen, (110, 110, 130), (x, y, w, bh), 1)
        cx, cy = x + w // 2, y + bh // 2
        d = -1 if self._plot_collapsed else 1
        col = (230, 230, 240) if hover else (190, 190, 205)
        pg.draw.polygon(self._screen, col, [(cx + 2 * d, cy - 4),
                                            (cx - 2 * d, cy),
                                            (cx + 2 * d, cy + 4)])
        if self._plot_collapsed:
            # The folded panel's name, one letter per line.
            ly = y + bh + 8
            for ch in self._t("PLOT"):
                surf = self._font.render(ch, True, col)
                self._screen.blit(surf, (cx - surf.get_width() // 2, ly))
                ly += 13

    def _hit_plot_toggle(self, pos) -> bool:
        x, y, w, h = self._plot_toggle_rect
        # No tolerance to the left: the heatmap scrollbar ends there.
        return self._hit_rect((x, y, w + (0 if self._plot_collapsed else 3),
                               h), pos)

    def _toggle_plot(self) -> None:
        """Fold the plot panel in under the heatmaps, or roll it out.

        The heatmap area - and with it the splitter and the fold button
        under the pointer - stays exactly where it is; only the window's
        right edge moves. It glides (``_fold_anim_s``, ease-out) with a
        captured image of the plot in the plot area, so the plot rolls
        in under the heatmaps or out from under them. Folded, the window
        is the heatmap area plus the strip (never below
        ``_min_collapsed_win_w``); unfolded, it gets its earlier width
        back (section 194).
        """
        pg = self._pg
        self._dragging_splitter = False
        self._dragging_plot_scroll = False
        self._splitter_hover = False
        self._set_cursor(False)
        self._spawn_popup = None
        self._spawn_popup_item_rects = []
        hm_w = self._splitter_rect[0]
        if not self._plot_collapsed:
            self._expanded_win_w = self._win_w
            self._expanded_plot_w = self._win_w - self._expanded_plot_x
            self._collapsed_resized = False
            self._hm_area_fixed = hm_w
            self._render_full()
            self._plot_anim_img = self._capture_plot_image(self._screen)
            target = max(self._min_collapsed_win_w,
                         hm_w + self._collapsed_strip_w)
            try:
                self._roll_window(self._win_w, target)
            finally:
                self._plot_anim_img = None
            self._plot_collapsed = True
            self._clear_plot_hit_rects()
            self._set_window_width(target)
            return
        if not self._collapsed_resized:
            target = self._expanded_win_w
        else:
            # The heatmaps were resized while folded: the plot comes
            # back beside them at the width it had.
            target = hm_w + self._splitter_w + self._expanded_plot_w
        target = max(target, hm_w + self._splitter_w + self._min_plot_w)
        max_w = self._max_window_w()
        if max_w is not None:
            target = min(target, max_w)
        self._plot_collapsed = False
        # The plot as it will look at the target width, drawn off screen.
        start_w, screen = self._win_w, self._screen
        try:
            self._win_w = target
            self._compute_layout()
            self._screen = pg.Surface((target, self._win_h))
            self._screen.fill((18, 18, 22))
            self._draw_plot()
            self._plot_anim_img = self._capture_plot_image(self._screen)
        finally:
            self._screen, self._win_w = screen, start_w
            self._compute_layout()
        try:
            self._roll_window(start_w, target)
        finally:
            self._plot_anim_img = None
        self._set_window_width(target, render=False)
        # Back to the splitter share, at exactly the pinned position -
        # unless the screen left the plot under its minimum width; then
        # the area stays pinned (the button must not move) until the
        # splitter is dragged.
        self._split_frac = self._splitter_rect[0] / float(max(1, self._win_w))
        if self._plot_rect[2] >= self._min_plot_w:
            self._hm_area_fixed = None
            self._compute_layout()
        self._render_full()

    def _capture_plot_image(self, surface):
        x, y, w, h = self._plot_rect
        w = min(w, surface.get_width() - x)
        if w <= 0 or h <= 0:
            return None
        return surface.subsurface((x, y, w, h)).copy()

    def _roll_window(self, w0: int, w1: int) -> None:
        """Glide the window's right edge from ``w0`` to ``w1`` (ease-out),
        painting every step before the next."""
        dur = float(self._fold_anim_s)
        if dur <= 0 or w0 == w1:
            return
        t0 = time.monotonic()
        while True:
            f = min(1.0, (time.monotonic() - t0) / dur)
            e = 1.0 - (1.0 - f) ** 3
            w = int(round(w0 + (w1 - w0) * e))
            if w != self._win_w:
                t_step = time.monotonic()
                self._set_window_width(w)
                slow = time.monotonic() - t_step
                if slow > 0.04 and f < 0.5:
                    # A slow step (un-maximising) is not part of the
                    # roll: do not let it eat the glide.
                    t0 += slow
            self._pg.event.pump()
            if f >= 1.0:
                return

    def _set_window_width(self, width: int, render: bool = True) -> None:
        """Resize the window in x (height kept), re-run the layout and
        paint the new frame at once.

        Through SDL_SetWindowSize (``pygame._sdl2``): the same X window,
        ~20 ms. ``display.set_mode`` destroys and re-creates the window
        on every call here (pygame 2.6 / sdl2-compat, a new X window id
        each time, ~120 ms) - that was the flicker of sections 190-193;
        it is only the fallback. The window manager may refuse a size
        (e.g. a maximised window); the layout follows the surface the
        window actually has.
        """
        pg = self._pg
        width, height = int(width), int(self._win_h)
        surf = None
        try:
            if self._sdl_window is None:
                from pygame._sdl2.video import Window
                self._sdl_window = Window.from_display_module()
            win = self._sdl_window
            win.size = (width, height)
            surf = pg.display.get_surface()
            if surf.get_size() != (width, height):
                # A maximised window refuses a new size - and the window
                # manager maximises this one by itself when it opens near
                # screen size. Un-maximise and set the geometry in the
                # same breath, at the same origin, so the restored size
                # never shows (measured on X11: no intermediate geometry).
                pos = win.position
                win.restore()
                win.position = pos
                win.size = (width, height)
                end = time.monotonic() + 0.3
                while time.monotonic() < end:
                    pg.event.pump()
                    surf = pg.display.get_surface()
                    if surf.get_size() == (width, height):
                        break
                    time.sleep(0.005)
        except Exception as e:
            self._log_once(f"SDL window resize failed, using set_mode: {e!r}")
            self._sdl_window = None
        if surf is None or surf.get_size() != (width, height):
            try:
                surf = pg.display.set_mode((width, height),
                                           pg.SWSURFACE | pg.RESIZABLE)
            except Exception as e:
                self._log_once(f"window resize failed: {e!r}")
                surf = pg.display.get_surface()
        if surf is not None:
            self._screen = surf
            self._win_w, self._win_h = (int(v) for v in surf.get_size())
        self._compute_layout()
        if render:
            self._render_full()

    def _max_window_w(self) -> Optional[int]:
        usable = self._usable_screen_size()
        return None if usable is None else int(usable[0])

    def _clear_plot_hit_rects(self) -> None:
        """Forget the plot's click targets; ``_draw_plot`` sets them each
        frame, so a collapsed plot would otherwise keep claiming clicks
        and wheel zoom over the heatmaps."""
        self._tab_rects = []
        self._legend_rects = []
        self._legend_all_rect = None
        self._plot_area_rect = None
        self._live_button_rect = None
        self._save_plot_button_rect = None
        self._plot_scroll_track_rect = None
        self._plot_scroll_thumb_rect = None
        self._plot_hover = False

    def _set_cursor(self, resize: bool) -> None:
        pg = self._pg
        try:
            pg.mouse.set_cursor(pg.SYSTEM_CURSOR_SIZEWE if resize
                                else pg.SYSTEM_CURSOR_ARROW)
        except Exception:
            pass

    def _hit_splitter(self, pos) -> bool:
        # The fold button never drags; folded, the strip does not either
        # (the window's edges set the heatmap area, section 195).
        if self._plot_collapsed or self._hit_plot_toggle(pos):
            return False
        x, y, w, h = self._splitter_rect
        # A few px of grab tolerance into the plot (not to the left: the
        # heatmap scrollbar ends where the splitter starts).
        return self._hit_rect((x, y, w + 3, h), pos)

    def _handle_layout_event(self, event) -> bool:
        """Window resize, the splitter, and the heatmap scroll view.

        Returns True when the event was consumed (the caller redraws).
        Called first in every event loop (``pump_events``, its pause
        loop, ``wait_for_close``).
        """
        pg = self._pg
        if event.type in (pg.VIDEORESIZE,
                          getattr(pg, "WINDOWSIZECHANGED", pg.VIDEORESIZE)):
            surf = pg.display.get_surface()
            if surf is None:
                return False
            self._screen = surf
            size = surf.get_size()
            if size != (self._win_w, self._win_h):
                width_changed = int(size[0]) != self._win_w
                self._win_w, self._win_h = int(size[0]), int(size[1])
                self._spawn_popup = None
                self._spawn_popup_item_rects = []
                if self._plot_collapsed and width_changed:
                    # Folded, the window's own edges and corners set the
                    # heatmap area (section 195): it fills the window up
                    # to the strip, and unfolding puts the plot beside it.
                    self._hm_area_fixed = max(
                        self._min_hm_area_w,
                        self._win_w - self._collapsed_strip_w)
                    self._collapsed_resized = True
                self._compute_layout()
            return True
        if (event.type == pg.MOUSEBUTTONDOWN and event.button == 1
                and self._spawn_popup is None
                and self._hit_plot_toggle(event.pos)):
            self._toggle_plot()
            return True
        if (event.type == pg.MOUSEBUTTONDOWN and event.button == 1
                and self._spawn_popup is None
                and self._hit_splitter(event.pos)):
            self._dragging_splitter = True
            self._splitter_drag_dx = event.pos[0] - self._splitter_rect[0]
            self._hm_area_fixed = None
            return True
        if event.type == pg.MOUSEMOTION and self._dragging_splitter:
            hm_area_w = event.pos[0] - self._splitter_drag_dx
            self._split_frac = hm_area_w / float(max(1, self._win_w))
            self._compute_layout()
            # Keep the stored share inside the clamped range, so a later
            # resize starts from where the splitter actually is.
            self._split_frac = self._splitter_rect[0] / float(max(1, self._win_w))
            return True
        if (event.type == pg.MOUSEBUTTONUP and event.button == 1
                and self._dragging_splitter):
            self._dragging_splitter = False
            if not self._hit_splitter(event.pos):
                self._set_cursor(False)
                self._splitter_hover = False
            return True
        if event.type == pg.MOUSEMOTION:
            self._plot_toggle_hover = self._hit_plot_toggle(event.pos)
            hover = self._hit_splitter(event.pos)
            if hover != self._splitter_hover:
                self._splitter_hover = hover
                self._set_cursor(hover)
            # Not consumed: the plot hover tooltip also reads motion.
        if self._hm_scrollbar_rect is None:
            return False
        if event.type == pg.MOUSEWHEEL:
            dy = int(getattr(event, "y", 0))
            if dy == 0:
                return False
            try:
                mpos = pg.mouse.get_pos()
            except Exception:
                return False
            if not (self._hit_rect(self._hm_view_rect, mpos)
                    or self._hit_rect(self._hm_scrollbar_rect, mpos)):
                return False
            self._set_hm_scroll(self._hm_scroll_y - dy * self._hm_scroll_step)
            return True
        if (event.type == pg.MOUSEBUTTONDOWN and event.button == 1
                and self._spawn_popup is None
                and self._hit_rect(self._hm_scrollbar_rect, event.pos)):
            my = event.pos[1]
            thumb = self._hm_thumb_rect
            if thumb is not None and thumb[1] <= my < thumb[1] + thumb[3]:
                self._dragging_hm_scroll = True
                self._hm_scroll_drag_dy = my - thumb[1]
            else:
                # Click beside the thumb pages one viewport up or down.
                page = self._hm_view_rect[3]
                if thumb is not None and my < thumb[1]:
                    page = -page
                self._set_hm_scroll(self._hm_scroll_y + page)
            return True
        if event.type == pg.MOUSEMOTION and self._dragging_hm_scroll:
            _sx, sy, _sw, sh = self._hm_scrollbar_rect
            th = self._hm_thumb_rect[3] if self._hm_thumb_rect else 24
            frac = (event.pos[1] - self._hm_scroll_drag_dy - sy) / max(1, sh - th)
            self._set_hm_scroll(frac * self._hm_max_scroll())
            return True
        if (event.type == pg.MOUSEBUTTONUP and event.button == 1
                and self._dragging_hm_scroll):
            self._dragging_hm_scroll = False
            return True
        return False

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

    @staticmethod
    def _fit_diet_text(codes, pcts, fits) -> str:
        """The diet line for prey ``codes`` with shares ``pcts`` (largest
        first), in the first form ``fits(text)`` accepts.

        ``re/ia/pf = 38/31/21%`` as before; when that is wider than the
        panel (the benthivores have five prey since section 223.15), the
        pairs ``re38 ia31 pf21%``; when even that is too wide, the
        smallest shares are dropped and a trailing ``+`` says so.
        """
        nums = [f"{p:.0f}" for p in pcts]
        text = "/".join(codes) + " = " + "/".join(nums) + "%"
        if fits(text):
            return text
        pairs = [c + n for c, n in zip(codes, nums)]
        text = " ".join(pairs) + "%"
        for k in range(len(pairs) - 1, 0, -1):
            if fits(text):
                return text
            text = " ".join(pairs[:k]) + "% +"
        return text

    @classmethod
    def _unique_abbrevs(cls, names) -> dict:
        """``_abbrev_fg`` per name, lengthened where two names collide.

        Colliding multi-word names whose second words differ keep the
        first letter of the first word and take as many letters of the
        second word as it takes to tell them apart: Italian ``Pesci
        planctivori`` / ``Pesci predatori`` give ``ppl`` / ``ppr``
        (section 216; the first word alone gave ``pescip`` for both).
        Otherwise - one-word names, or second words that are the same -
        they take one more letter of their first word at a time:
        ``planktivorous_fish`` / ``predatory_fish`` give ``plf`` /
        ``prf``. Words are split on spaces, underscores and hyphens.
        """
        def words(n):
            return [q for q in n.lower().replace('-', '_')
                    .replace(' ', '_').split('_') if q]

        names = list(dict.fromkeys(str(n) for n in names if n))
        codes = {n: cls._abbrev_fg(n) for n in names}
        for k in range(2, 12):
            seen: dict = {}
            for n, c in codes.items():
                seen.setdefault(c, []).append(n)
            clashes = [ns for ns in seen.values() if len(ns) > 1]
            if not clashes:
                break
            for ns in clashes:
                split = [words(n) for n in ns]
                by_second = (all(len(p) > 1 for p in split)
                             and len({p[1] for p in split}) == len(split))
                for n, p in zip(ns, split):
                    if by_second:
                        codes[n] = (p[0][0] + p[1][:k]
                                    + ''.join(q[0] for q in p[2:]))
                    else:
                        codes[n] = p[0][:k] + ''.join(q[0] for q in p[1:])
        return codes

    def _fg_abbrev(self, fid: str) -> str:
        """Unique short code of ``fid`` among the viewer's groups.

        English codes come from the ids (as before section 212); the other
        languages take them from the translated names, one cache each.
        """
        caches = self.__dict__.setdefault("_abbrev_caches", {})
        cache = caches.get(self._lang)
        if cache is None or fid not in cache:
            fids = list(dict.fromkeys(list(self.fg_ids) + [fid]))
            if self._lang == i18n.DEFAULT:
                cache = self._unique_abbrevs(fids)
            else:
                names = {f: self._display_name(f) for f in fids}
                codes = self._unique_abbrevs(names.values())
                cache = {f: codes[n] for f, n in names.items()}
            caches[self._lang] = cache
        return cache[fid]

    def _draw_one_heatmap(self, fid: str, px: int, py: int) -> None:
        pg = self._pg
        pad = 4
        # Five info lines above the heatmap: FG name, 'B0 = ... B = ...',
        # action distribution 'mv/rs/et = ...', biomass-loss breakdown
        # 'pr/st/im/nm = ...', and diet breakdown '<abbr>/... = X/...%',
        # followed by a b0 slider. An NDM has no actions, diet, M1 or
        # starvation: its third line is its loss breakdown 'pr/im = ...'
        # (eaten / impacts such as bottom trawling) and the rest is blank.
        title_h = 104
        arr = self._biomass.get(fid)
        legend: list = []
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

        # Third info line: action distribution mv/rs/et for DMs.
        act_y = info_y + info_surf.get_height()
        is_ndm = fid in self._ndm_ids
        if is_ndm:
            lb = self._loss_breakdown.get(fid)
            if lb is not None:
                pr = float(lb.get('predation', 0.0)) * 100.0
                im = float(lb.get('impact', 0.0)) * 100.0
                ndm_surf = self._font.render(self._t(
                    "pr/im = {pr}/{im}%", pr=f"{pr:.0f}", im=f"{im:.0f}"),
                    True, info_col)
                self._screen.blit(ndm_surf, (px + pad, act_y))
        else:
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
                act_txt = self._t("mv/rs/et = {mv}/{rs}/{et}%",
                                  mv=f"{mv:.1f}", rs=f"{rs:.1f}",
                                  et=f"{et:.1f}")
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
                    nm = float(lb.get('natural', 0.0)) * 100.0
                    loss_txt = self._t(
                        "pr/st/im/nm = {pr}/{st}/{im}/{nm}%",
                        pr=f"{pr:.0f}", st=f"{st:.0f}", im=f"{im:.0f}",
                        nm=f"{nm:.0f}")
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
                                abbrs = [self._fg_abbrev(pid)
                                         for pid, _ in items]
                                pcts = [v / tot * 100.0 for _, v in items]
                                avail = self._panel_w - 2 * pad
                                diet_txt = self._fit_diet_text(
                                    abbrs, pcts,
                                    lambda t: self._font.size(t)[0] <= avail)
                                diet_surf = self._font.render(
                                    diet_txt, True, info_col)
                                diet_y = loss_y + loss_surf.get_height()
                                self._screen.blit(
                                    diet_surf, (px + pad, diet_y))
                                # Every prey, also those a narrow panel
                                # dropped behind the trailing ``+``.
                                legend += [(a, self._display_name(pid))
                                           for a, (pid, _v)
                                           in zip(abbrs, items)]

        # ---- b0-slider --------------------------------------------------
        # Horisontell slider placerad mellan info-blocket ovanför och själva
        # heatmapen nedanför. Range = [0, 4 * b0_default]; mittposition
        # motsvarar projektets gridskalade default. När ingen default är
        # satt (FG saknar inference_initial_biomass) ritas en disabled
        # placeholder så heatmap-origin förblir aligned med andra paneler.
        slider_h = int(getattr(self, '_slider_h', 16))
        slider_y = py + title_h - slider_h - 4
        # Hovering the info lines above the slider lists the group's
        # species (multi-species groups) and which prey the codes on its
        # diet line stand for (``_draw_member_tooltip``).
        self._panel_legend[fid] = legend
        # Who eats this group: its predation loss by predator (largest
        # first), for the same tooltip.
        eaten = self._predation_breakdown.get(fid) or {}
        tot = sum(float(v) for v in eaten.values() if float(v) > 0.0)
        predators = (sorted(((pid, float(v) / tot) for pid, v in eaten.items()
                             if float(v) > 0.0), key=lambda kv: -kv[1])
                     if tot > 0.0 else [])
        self._panel_predators[fid] = predators
        if fid in self._members or legend or predators:
            self._member_rects.append(
                ((px, py, self._panel_w, slider_y - 2 - py), fid))
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
        # Per-FG colour reference. Fixed (default): the rollout-start
        # maximum, also during replay and in log mode, never falling back
        # to the frame's own max even for a zero start. Rolling ('n'): this
        # frame's own maximum, which is what the fixed reference replaced
        # in section 94 and what the toggle brings back on demand. The
        # display-scale slider multiplies whichever one is in use.
        if self._rolling_heatmap:
            try:
                reference = float(np.max(v))
            except Exception:
                reference = 0.0
        else:
            reference = float(self._heatmap_vmax0.get(fid, 0.0))
        raw_vmax = max(reference * float(self._biomass_display_scale), 1e-12)
        if self._log_heatmap:
            v = np.log1p(np.maximum(v, 0.0))
            vmax = float(np.log1p(raw_vmax))
        else:
            vmax = raw_vmax
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
        # Daylight: a thin frame just outside the heatmap whose grey is
        # the day's sun hours on the year's scale - black on the shortest
        # day, white on the longest, linear between, constant within a
        # day (it changes at midnight, not over the day).
        # 2 px is what fits: the colorbar starts 3 px below the heatmap.
        if self._calendar:
            grey = int(round(255.0 * min(1.0, max(0.0, float(
                self._calendar.get("day_length_frac", 0.0))))))
            pg.draw.rect(self._screen, (grey, grey, grey),
                         (hm_x - _LIGHT_FRAME_PX, hm_y - _LIGHT_FRAME_PX,
                          W * self.cell_px + 2 * _LIGHT_FRAME_PX,
                          H * self.cell_px + 2 * _LIGHT_FRAME_PX),
                         _LIGHT_FRAME_PX)

        # ---- Colorbar legend under the heatmap ----------------------------
        # Per-FG normalisation: shows what the colour gradient maps to,
        # from 0 (left, dark) to the reference in use times the display
        # scale (right, bright), also when log mode is enabled. The number
        # is the whole point of the rolling toggle being legible: it is the
        # rollout-start maximum when fixed and this frame's when rolling.
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
        max_txt = self._fmt_compact(raw_vmax)
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
            mode_label = self._t("(default)")
            tpl_label = "—"
        elif cur_mode_raw is None:
            default_mode = self._spawn_defaults.get(fid)
            if default_mode:
                mode_label = self._t(self._spawn_mode_labels.get(
                    default_mode, default_mode))
            else:
                mode_label = self._t("(default)")
            tpl_label = "—"
        else:
            mode_label = self._t(
                self._spawn_mode_labels.get(cur_mode_raw, cur_mode_raw))
            tpl_label = cur_tpl if cur_tpl else "—"
        dd_bg = (38, 38, 48) if not dim else (28, 28, 35)
        dd_border = (90, 90, 105) if not dim else (50, 50, 60)
        dd_text = (210, 210, 220) if not dim else (110, 110, 120)
        for i, (label_tpl, value_text, which) in enumerate(
                (("Mode: {value}", mode_label, "mode"),
                 ("Tpl:  {value}", tpl_label, "template"))):
            rx, ry = hm_x, dd_y0 + i * (dd_row_h + 2)
            rect = (rx, ry, dd_w, dd_row_h)
            pg.draw.rect(self._screen, dd_bg, rect)
            pg.draw.rect(self._screen, dd_border, rect, 1)
            txt = self._t(label_tpl, value=value_text)
            # Truncate to fit. Measured with font.size: a rendered
            # surface that is never blitted leaks (section 189).
            max_w = dd_w - 14
            if self._font.size(txt)[0] > max_w:
                # Crude truncation.
                while len(txt) > 4 and self._font.size(txt + "…")[0] > max_w:
                    txt = txt[:-2]
                txt = txt + "…"
            txt_surf = self._font.render(txt, True, dd_text)
            self._screen.blit(txt_surf, (rx + 4, ry + 1))
            # Triangle indicator on the right.
            tri_x = rx + dd_w - 10
            tri_y = ry + dd_row_h // 2
            pg.draw.polygon(self._screen, dd_text,
                            [(tri_x, tri_y - 2),
                             (tri_x + 6, tri_y - 2),
                             (tri_x + 3, tri_y + 2)])
            self._spawn_dropdown_rects.append((rect, fid, which))

    _REWARD_TITLE_PREFIX = "ARS reward per FG"

    def _duration(self, days: float) -> str:
        """``3 years``, ``1 year``, ``3.8 days`` in the display language."""
        if days >= 0.95 * DAYS_PER_YEAR:
            n = days / DAYS_PER_YEAR
            text = f"{n:.2g}" if abs(n - round(n)) > 0.05 else str(int(round(n)))
            return self._t("{n} year" if text == "1" else "{n} years", n=text)
        text = f"{days:.1f}".rstrip("0").rstrip(".") if days < 10 else str(int(round(days)))
        return self._t("{n} day" if text == "1" else "{n} days", n=text)

    def _tab_title(self, tab: str) -> str:
        """The plot title of ``tab`` in the display language.

        A title set with fields (the progress tab) is formatted from its
        template; the reward title from ``set_reward_label`` has only its
        leading words translated (the formula after them is notation).
        """
        fmt = self._tab_label_fmt.get(tab)
        if fmt is not None:
            return self._t(fmt[0], **fmt[1])
        if tab == "progress" and self._long_run_ticks:
            # The horizon next to the one the policies were trained on
            # (section 223.21); the rollout slider can change the latter.
            template = "Long run of the current policies, {probe}; trained on {train} rollouts (biomass, % of start)"
            return self._t(
                template,
                probe=self._duration(self._long_run_ticks * LIBRARY_TICK_HOURS / HOURS_PER_DAY),
                train=self._duration(self.get_neval_ticks() * self.tick_hours / HOURS_PER_DAY))
        label = self._tab_labels.get(tab, tab)
        prefix = self._REWARD_TITLE_PREFIX
        if label.startswith(prefix) and label != prefix:
            return self._t(prefix) + label[len(prefix):]
        return self._t(label)

    def _active_plot_ids(self) -> list:
        """Return ``plot_fg_ids`` filtered for the active tab.

        Non-decision-makers (``self._ndm_ids``) only appear on the
        ``biomass`` tab; on every other tab they are filtered out. The
        same filter applies to their ``<id>_rnd`` baseline counterparts.
        On the action tabs (move/rest/eat) only DMs have meaningful data,
        so NDMs are likewise filtered out there.
        """
        active = self._tabs[self._active_tab]
        if active in _PROBE_TABS:
            # The probe (section 223.12) and its viability history
            # (223.20) show every group.
            return [fid for fid in self.plot_fg_ids if not fid.endswith("_rnd")]
        # NDMs are included on the biomass tab and loss tabs because they
        # can lose biomass to predation and impacts without making
        # decisions.
        if (active in ("biomass", "predation", "starvation", "impacts",
                       "natural")
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
          * ``predation`` / ``starvation`` / ``impacts``: viz stores %;
            written as ``loss_breakdown[fid][<cause>]`` in fractions
            0..1.
        """
        # Bygg tick-indexerad union av alla stegkoordinater som finns i
        # någon serie. Varje record motsvarar en tick.
        series = self._series
        all_steps: set = set()
        for tab in ("biomass", "energy", "move", "rest", "eat",
                    "predation", "starvation", "impacts", "natural"):
            if tab not in series:
                continue
            for fid, buf in series[tab].items():
                for step, _v in buf:
                    all_steps.add(int(step))
        if not all_steps:
            self._save_plot_button_flash_msg = self._t("No data to save")
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
        nm = _index("natural")

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
            all_fids = set(pr) | set(st) | set(im) | set(nm)
            for fid in all_fids:
                lp = pr.get(fid, {}).get(step)
                ls = st.get(fid, {}).get(step)
                li = im.get(fid, {}).get(step)
                ln = nm.get(fid, {}).get(step)
                if lp is None and ls is None and li is None and ln is None:
                    continue
                loss_breakdown[fid] = {
                    "predation":  (lp / 100.0) if lp is not None else 0.0,
                    "starvation": (ls / 100.0) if ls is not None else 0.0,
                    "impact":     (li / 100.0) if li is not None else 0.0,
                    "natural":    (ln / 100.0) if ln is not None else 0.0,
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
        self._save_plot_button_flash_msg = self._t(
            "Saved: {name}", name=os.path.basename(out_path))
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
                title=self._t("Save inference plot as HTML"),
                initialfile=default_name,
                defaultextension=".html",
                filetypes=[(self._t("HTML files"), "*.html"),
                           (self._t("All files"), "*.*")],
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

    def _draw_viability_overlay(self, xmin, xmax, ymin, ymax, px0, py0, pw, ph,
                                under=False) -> None:
        """The viability tab's bound, V and extinct marks (section 223.20).

        ``under`` (before the groups' margins): the bound (margin 0) as a
        brighter line and V - the worst group, i.e. the lower envelope -
        as a thick band, so the worst group's own line stays visible on
        it. Otherwise (after them): the bound's label and an extinct group
        as a cross at the axis floor.
        """
        pg = self._pg
        span_x, span_y = float(xmax - xmin), float(ymax - ymin)
        if span_x <= 0 or span_y <= 0:
            return

        def to_px(step, value):
            fx = (float(step) - xmin) / span_x
            fy = (ymax - float(value)) / span_y
            return (int(np.clip(px0 + fx * pw, px0, px0 + pw)),
                    int(np.clip(py0 + fy * ph, py0, py0 + ph)))

        zero_y = to_px(xmin, 0.0)[1]
        if under:
            pg.draw.line(self._screen, (150, 150, 170), (px0, zero_y), (px0 + pw, zero_y), 1)
            pts = [to_px(step, value) for step, value in self._viability_v
                   if xmin <= step <= xmax]
            if len(pts) > 1:
                pg.draw.lines(self._screen, (215, 215, 225), False, pts, 5)
            for pt in pts:
                pg.draw.circle(self._screen, (215, 215, 225), pt, 4)
            return
        label = self._font.render(self._t("bound"), True, (150, 150, 170))
        self._screen.blit(label, (px0 + pw - label.get_width() - 2, zero_y - 14))
        for step, fid in self._viability_extinct:
            if not (xmin <= step <= xmax) or not self._plot_enabled.get(fid, True):
                continue
            if self._solo is not None and self._solo != fid:
                continue
            cx, cy = to_px(step, ymin)
            col = self._fg_colour.get(fid, (220, 150, 130))
            cy -= 4
            pg.draw.line(self._screen, col, (cx - 4, cy - 4), (cx + 4, cy + 4), 2)
            pg.draw.line(self._screen, col, (cx - 4, cy + 4), (cx + 4, cy - 4), 2)

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
        progress = active == "progress"
        viability = active == "viability"
        probe_tab = progress or viability
        base_label = self._tab_title(active)
        # The biomass tab only leaves auto-scaling when the user pins the
        # display scale; log scaling is never applied to it.
        biomass_fixed_axis = (active == "biomass"
                              and self._biomass_scale_manual)
        use_log_plot = bool(self._log_plot and active not in _LINEAR_TABS)
        if biomass_fixed_axis:
            ylabel = self._t(
                "{label}  display max = {pct}%", label=base_label,
                pct=f"{100.0 * float(self._biomass_display_scale):.0f}")
        elif active == "reward" and use_log_plot:
            ylabel = self._t("reward (log10 ratio)")
        elif progress and use_log_plot:
            # R toggles a log axis on the probe too (section 223.12): the
            # plankton's seasonal peaks span three decades.
            ylabel = self._t("long-run biomass (log10 of % of start)")
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
        # A plot narrowed by the splitter cuts the strip at its edge.
        self._screen.set_clip(pg.Rect(x, y, w, strip_h))
        for i, tab in enumerate(self._tabs):
            is_active = (i == self._active_tab)
            col = (240, 240, 250) if is_active else (140, 140, 150)
            surf = self._font.render(self._t(_TAB_NAMES.get(tab, tab)), True, col)
            tw = surf.get_width() + 10
            rect = (tab_x, tab_y, tw, 14)
            if is_active:
                pg.draw.rect(self._screen, (45, 45, 60), rect)
            pg.draw.rect(self._screen, (60, 60, 70), rect, 1)
            self._screen.blit(surf, (tab_x + 5, tab_y + 1))
            self._tab_rects.append((rect, i))
            tab_x += tw + 4
        self._screen.set_clip(None)

        # Title (under tab strip).
        title_y = y + strip_h + 2
        tsurf = self._font.render(ylabel, True, (200, 200, 210))
        self._screen.blit(tsurf, (x + 6, title_y))
        if probe_tab:
            detail = self._font.render(
                self._t(self._progress_fmt[0], **self._progress_fmt[1]),
                True, (160, 160, 170))
            self._screen.blit(detail, (x + 6, title_y + 14))
            line = self._viability_line()
            if line is not None:
                text, ok = line
                surf = self._font.render(
                    text, True, (160, 160, 170) if ok else (220, 150, 130))
                self._screen.blit(surf, (x + 6, title_y + 28))

        active_ids = self._active_plot_ids()

        # Plot area geometry — computed up-front so the legend is always
        # drawn (and clickable) even when there is no data to plot, e.g.
        # when every FG has been toggled off. Otherwise the legend would
        # disappear together with the lines and the user couldn't re-
        # enable any series without blindly guessing checkbox positions.
        plot_pad_l = 36
        plot_pad_r = getattr(self, "_legend_w", 110)  # room for legend
        plot_pad_t = strip_h + 20  # tab strip (variable rows) + ylabel line
        if probe_tab:
            plot_pad_t += 16
            if self._probe_viability:
                plot_pad_t += 14
        # Reservera ~14 px längst ned för horisontell scrollbar. Används
        # i både inference- och train-läget så att användaren kan
        # scrolla tillbaka i graferna även under träning (och gå live
        # igen via Live-knappen eller genom att dra thumb till högerkant).
        scroll_bar_h = 0 if probe_tab else 12
        # Save-knappen (inference) ritas under scrollbaren; reservera 22 px.
        inference = str(self.mode).lower().startswith("infer")
        save_btn_h = 22 if inference else 0
        # Calendar band (section 155): inference x is the tick, so with
        # the daylight calendar every x has a month and a clock time.
        cal_cfg = self._daylight_cfg if inference and not progress else None
        band_h = _CAL_BAND_H if cal_cfg else 0
        plot_pad_b = 16 + band_h + (scroll_bar_h + 4 if scroll_bar_h else 0) \
            + (save_btn_h + 4 if save_btn_h else 0)
        if probe_tab:
            plot_pad_b = 40
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

            if biomass_fixed_axis:
                ymin = 0.0
                ymax = max(1.0, 100.0 * float(self._biomass_display_scale))
            elif viability:
                # Fixed (section 223.20): the bound stays where it is.
                ymin, ymax = _VIABILITY_AXIS
            else:
                # Y-range: bara från punkter inom det synliga x-intervallet så
                # skalan följer det som faktiskt syns i fönstret.
                visible: list = []
                for fid in active_ids:
                    if self._solo is not None and self._solo != fid:
                        continue
                    if not self._plot_enabled.get(fid, True):
                        continue
                    steps, vals, _sorted = _series_arrays(buffers[fid])
                    if _sorted:
                        lo = np.searchsorted(steps, xmin, side="left")
                        hi = np.searchsorted(steps, xmax, side="right")
                        visible.append(vals[lo:hi])
                    else:
                        visible.append(vals[(steps >= xmin) & (steps <= xmax)])
                arr = (np.concatenate(visible) if visible
                       else np.empty(0, dtype=np.float64))
                if arr.size == 0:
                    have_data = False
                else:
                    if use_log_plot:
                        arr = np.sign(arr) * np.log10(np.abs(arr) + 1e-12)
                    ymin = float(arr.min())
                    ymax = float(arr.max())
                    if not np.isfinite(ymin) or not np.isfinite(ymax):
                        have_data = False
                    elif ymax - ymin < 1e-9:
                        ymax = ymin + 1.0

        if have_data:
            # Y-axis tick labels (max, 3/4, mid, 1/4, min) in plain
            # decimals - no scientific notation, no plus sign. A long
            # label widens the left margin instead of running into the
            # plot (px0/pw are not used before this point).
            fracs = (0.0, 0.25, 0.5, 0.75, 1.0)
            vals = [ymax - f * (ymax - ymin) for f in fracs]
            labels = _axis_labels(vals, (ymax - ymin) / 4.0)
            lab_w = max(self._font.size(t)[0] for t in labels)
            if lab_w + 6 > plot_pad_l and w - (lab_w + 6) - plot_pad_r > 4:
                plot_pad_l = lab_w + 6
                px0 = x + plot_pad_l
                pw = w - plot_pad_l - plot_pad_r
            for frac, lab_txt in zip(fracs, labels):
                yy = int(py0 + frac * ph)
                pg.draw.line(self._screen, (50, 50, 60),
                             (px0, yy), (px0 + pw, yy), 1)
                lab = self._font.render(lab_txt, True, (160, 160, 170))
                self._screen.blit(lab, (x + 2, yy - 7))

            # Zero line if in range.
            if ymin < 0.0 < ymax:
                yy = int(py0 + (ymax - 0.0) / (ymax - ymin) * ph)
                pg.draw.line(self._screen, (90, 90, 110),
                             (px0, yy), (px0 + pw, yy), 1)

            # Before the series, so its month guides lie under them.
            if cal_cfg:
                self._draw_calendar_band(cal_cfg, xmin, xmax,
                                         px0, py0, pw, ph, py0 + ph + 3)

            if viability:
                self._draw_viability_overlay(xmin, xmax, ymin, ymax, px0, py0, pw, ph,
                                             under=True)

            # Plot lines. Klipp till synligt x-fönster med en padding-punkt
            # på varje sida så linjesegment som skär fönstrets kanter
            # fortfarande når hela vägen ut.
            for fid in active_ids:
                if self._solo is not None and self._solo != fid:
                    continue
                if not self._plot_enabled.get(fid, True):
                    continue
                buf = buffers[fid]
                if not buf or (len(buf) < 2 and not probe_tab):
                    continue
                steps, vals, _sorted = _series_arrays(buf)
                n_buf = len(steps)
                # Bygg indexlista av synliga punkter + en granne på varje
                # sida (för att linjen ska nå fönsterkanten).
                if _sorted:
                    i = int(np.searchsorted(steps, xmin, side="left"))
                    j = int(np.searchsorted(steps, xmax, side="right")) - 1
                else:
                    ge = np.flatnonzero(steps >= xmin)
                    le = np.flatnonzero(steps <= xmax)
                    i = int(ge[0]) if ge.size else n_buf
                    j = int(le[-1]) if le.size else -1
                start_i = max(0, i - 1) if i < n_buf else 0
                end_i = min(n_buf - 1, j + 1) if j >= 0 else n_buf - 1
                if end_i < start_i:
                    continue
                v = vals[start_i:end_i + 1]
                if use_log_plot:
                    v = np.sign(v) * np.log10(np.abs(v) + 1e-12)
                fx = (steps[start_i:end_i + 1] - xmin) / (xmax - xmin)
                fy = (ymax - v) / (ymax - ymin)
                # Klipp x till plot-arean så linjer inte spiller över.
                # Klipp även y: annars kan padding-grannar (som ligger
                # utanför synligt x-fönster men vars värden är utanför
                # det synliga y-intervallet) rita linjer som spiller
                # över tab-strippen ovanför eller scrollbaren nedanför.
                # (int() truncation, as before; NaN rows are dropped.)
                xf = px0 + fx * pw
                yf = py0 + fy * ph
                ok = np.isfinite(xf) & np.isfinite(yf)
                if not ok.all():
                    xf, yf = xf[ok], yf[ok]
                xs = np.clip(np.trunc(xf), px0, px0 + pw).astype(np.int64)
                ys = np.clip(np.trunc(yf), py0, py0 + ph).astype(np.int64)
                # A long rollout has far more points than the plot has
                # pixel columns; collapse each column to first/min/max/
                # last so the cost of a frame is bounded by the plot
                # width, not the rollout length.
                if _sorted and len(xs) > 2 * pw:
                    xs, ys = _decimate_polyline(xs, ys)
                pts = list(zip(xs.tolist(), ys.tolist()))
                if viability:
                    # Probes are sparse: a dot on each one.
                    for pt in pts:
                        pg.draw.circle(self._screen, self._fg_colour[fid], pt, 2)
                if len(pts) < 2:
                    continue
                try:
                    pg.draw.aalines(self._screen, self._fg_colour[fid], False, pts)
                except Exception:
                    pg.draw.lines(self._screen, self._fg_colour[fid], False, pts, 1)

            if viability:
                self._draw_viability_overlay(xmin, xmax, ymin, ymax, px0, py0, pw, ph)
                # The training updates at the ends of the axis.
                for value, anchor in ((xmin, 0.0), (xmax, 1.0)):
                    label = self._font.render(str(int(round(value))), True, (160, 160, 170))
                    lx = px0 + int(anchor * pw) - int(anchor * label.get_width())
                    self._screen.blit(label, (lx, py0 + ph + 5))
                label = self._font.render(self._t("Training update"), True, (200, 200, 210))
                self._screen.blit(label, (px0 + (pw - label.get_width()) // 2, py0 + ph + 21))

            if progress:
                # Year marks under the probe (x is the probe day).
                per_year = float(getattr(self, "_probe_days_per_year", 365.0) or 365.0)
                year = 0
                while year * per_year <= xmax + 1e-9:
                    value = year * per_year
                    if value >= xmin - 1e-9:
                        frac = (value - xmin) / (xmax - xmin)
                        mx = px0 + int(frac * pw)
                        pg.draw.line(self._screen, (60, 60, 70), (mx, py0), (mx, py0 + ph), 1)
                        label = self._font.render(str(year), True, (160, 160, 170))
                        self._screen.blit(label, (mx - label.get_width() // 2, py0 + ph + 5))
                    year += 1
                label = self._font.render(self._t("Long-run year (inference start = 0)"),
                                          True, (200, 200, 210))
                self._screen.blit(label, (px0 + (pw - label.get_width()) // 2, py0 + ph + 21))

        # ---- Horisontell scrollbar (inference) ---------------------------
        # Ritas alltid när scroll_bar_h > 0 (reserverar utrymme), men blir
        # bara interaktiv/synlig thumb när data-intervallet överstiger
        # fönstret (scroll_active). ``_plot_scroll_track_rect`` sätts alltid
        # så hit-tests kan avgöra klick även vid mindre data.
        self._plot_scroll_track_rect = None
        self._plot_scroll_thumb_rect = None
        # Cacha data-span för scrollbar-drag (används av
        # ``_plot_scroll_offset_from_x`` mellan render-anrop).
        if have_data:
            # Cacha alltid full data-span (även när scroll inte är aktiv)
            # så att MOUSEWHEEL-zoom kan använda den för att beräkna
            # ny fönsterbredd och ev. aktivera scrollbaren vid inzoomning.
            self._plot_scroll_span_cache = (float(data_xmin), float(data_xmax))
        else:
            self._plot_scroll_span_cache = None
        # Spara plot-arean för MOUSEWHEEL hit-test.
        self._plot_area_rect = (int(px0), int(py0), int(pw), int(ph))
        self._live_button_rect = None
        if scroll_bar_h > 0 and pw > 4:
            sb_y = py0 + ph + 4 + band_h
            sb_h = scroll_bar_h
            # Reservera plats till höger för "Live"-knappen så den ligger
            # bredvid scrollbaren (bara när scroll faktiskt är aktiv och
            # utrymmet räcker; annars använder tracken hela bredden).
            live_btn_w = 42
            live_gap = 4
            if scroll_active and pw > live_btn_w + live_gap + 20:
                track_w = pw - live_btn_w - live_gap
            else:
                track_w = pw
            track = (px0, sb_y, track_w, sb_h)
            pg.draw.rect(self._screen, (30, 30, 38), track)
            pg.draw.rect(self._screen, (60, 60, 70), track, 1)
            if scroll_active:
                full_span = float(data_xmax - data_xmin)
                win_w_x = float(self._plot_window_width)
                thumb_w = max(20, int(track_w * (win_w_x / full_span)))
                # Position i track baserat på offset.
                off_frac = (float(xmin) - float(data_xmin)) / max(
                    1e-9, full_span - win_w_x)
                off_frac = max(0.0, min(1.0, off_frac))
                thumb_x = int(px0 + off_frac * (track_w - thumb_w))
                thumb_rect = (thumb_x, sb_y + 1, thumb_w, sb_h - 2)
                pg.draw.rect(self._screen, (110, 110, 130), thumb_rect)
                pg.draw.rect(self._screen, (170, 170, 190), thumb_rect, 1)
                self._plot_scroll_thumb_rect = thumb_rect
                self._plot_scroll_track_rect = track
                # Live-knapp: highlightas grönt när plotten redan följer
                # senaste tick (``_plot_scroll_offset is None``); annars
                # dämpad så användaren ser att ett klick kommer att
                # återta live-läget. Placeras direkt till höger om track.
                lb_x = px0 + track_w + live_gap
                lb_rect = (lb_x, sb_y, live_btn_w, sb_h)
                is_live = self._plot_scroll_offset is None
                if is_live:
                    bg = (55, 90, 60)
                    border = (110, 180, 130)
                    txt_col = (220, 240, 220)
                else:
                    bg = (55, 55, 65)
                    border = (120, 130, 145)
                    txt_col = (210, 220, 230)
                pg.draw.rect(self._screen, bg, lb_rect)
                pg.draw.rect(self._screen, border, lb_rect, 1)
                label = "● Live" if is_live else "Live"
                surf = self._font.render(label, True, txt_col)
                tx = lb_x + max(2, (live_btn_w - surf.get_width()) // 2)
                ty = sb_y + (sb_h - surf.get_height()) // 2
                self._screen.blit(surf, (tx, ty))
                self._live_button_rect = lb_rect
            else:
                # Ingen scrollning möjlig — rita en tunn markör som fyller
                # hela tracken (visar att allt data är synligt).
                inner = (px0 + 1, sb_y + 1, track_w - 2, sb_h - 2)
                pg.draw.rect(self._screen, (55, 55, 65), inner)

        # ---- Save-plot-HTML-knapp (inference) ---------------------------
        # Ritas under scrollbaren. Enabled endast när inspelning INTE
        # pågår OCH det finns data att spara. Vid klick öppnas en
        # Tk-fildialog med default-filnamn ``inferenceplot.html``.
        self._save_plot_button_rect = None
        if save_btn_h > 0 and pw > 4:
            btn_y = py0 + ph + 4 + band_h \
                + (scroll_bar_h + 4 if scroll_bar_h else 0)
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
                label = self._t("Save plot as HTML...")
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
        # Select all / deselect all: a master checkbox over the entries of
        # this tab (ticked = all on, dash = some, empty = none).
        self._legend_all_rect = None
        if active_ids:
            states = [self._plot_enabled.get(fid, True) for fid in active_ids]
            cb_rect = (lx, ly + 3, 12, 12)
            pg.draw.rect(self._screen, (200, 200, 210), cb_rect, 1)
            if all(states):
                pg.draw.line(self._screen, (220, 220, 230),
                             (lx + 2, ly + 9), (lx + 5, ly + 12), 2)
                pg.draw.line(self._screen, (220, 220, 230),
                             (lx + 5, ly + 12), (lx + 11, ly + 4), 2)
            elif any(states):
                pg.draw.line(self._screen, (220, 220, 230),
                             (lx + 3, ly + 9), (lx + 9, ly + 9), 2)
            lab = self._font.render(self._t("all"), True, (200, 200, 210))
            self._screen.blit(lab, (lx + 16, ly))
            self._legend_all_rect = (lx, ly + 3, 16 + lab.get_width(), 12)
            ly += 16
            pg.draw.line(self._screen, (60, 60, 70), (lx, ly - 1),
                         (lx + self._legend_w - 12, ly - 1), 1)
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

        # ---- Hover-tooltip ----------------------------------------------
        # Ritas sist så den ligger ovanpå linjer, gridlines och legend.
        # Läser musposition direkt via ``pg.mouse.get_pos()`` istället för
        # att haka in i MOUSEMOTION-hanteringen (som är utspridd över
        # flera event-loopar). Kräver att plot-arean har data och att
        # muspekaren ligger inom ``_plot_area_rect``.
        if have_data and self._plot_area_rect is not None:
            try:
                mx, my = pg.mouse.get_pos()
            except Exception:
                mx = my = -1
            px, py, pwv, phv = self._plot_area_rect
            if px <= mx <= px + pwv and py <= my <= py + phv:
                self._draw_plot_tooltip(
                    mx, my,
                    active_ids=active_ids,
                    buffers=buffers,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                    px0=px, py0=py, pw=pwv, ph=phv,
                    plot_rect=(x, y, w, h),
                )

    def _draw_calendar_band(self, cfg: Mapping, xmin: float, xmax: float,
                            px0: int, py0: int, pw: int, ph: int,
                            band_y: int) -> None:
        """Month and clock rows under the inference plot (section 155).

        Profiler-style blocks: each month spans its ticks, and the clock
        row under it is cut at the finest clock step that leaves room
        for a label (``06:00``; the day of the month once only whole
        days fit; no labels when even a day is too narrow). The clock
        row is shaded by the light level of its ticks, night to day. A
        label is centred in the block's visible part, or left out when
        it does not fit. Month starts are faint guides through the
        plot, New Year a brighter one.
        """
        pg = self._pg
        screen = self._screen
        if xmax <= xmin or pw <= 4:
            return
        tick_h = resolve_tick_hours(cfg["tick_hours"])
        h0 = float(_plot_x_hours(cfg, xmin))
        h1 = float(_plot_x_hours(cfg, xmax))
        px_per_h = pw / (h1 - h0)
        month_y = band_y
        clock_y = band_y + _CAL_ROW_H + 1

        def span(start, end):
            a = max(px0, int(round(px0 + (start - h0) * px_per_h)))
            b = min(px0 + pw, int(round(px0 + (end - h0) * px_per_h)))
            return a, b

        # Clock row shading, one pixel column at a time.
        edges = h0 + np.arange(pw + 1, dtype=np.float64) * (h1 - h0) / pw
        light = np.clip(_column_light(cfg, edges), 0.0, 1.0)
        rgb = _CAL_NIGHT_RGB + light[:, None] * (_CAL_DAY_RGB - _CAL_NIGHT_RGB)
        cols = np.repeat(rgb[:, None, :], _CAL_ROW_H, axis=1)
        screen.blit(pg.surfarray.make_surface(cols.astype(np.uint8)),
                    (px0, clock_y))

        for start, end, m in _month_blocks(h0, h1):
            a, b = span(start, end)
            if b <= a:
                continue
            base = _CAL_SEASON_RGB[((m + 1) % 12) // 3]
            shade = 1.0 if m % 2 == 0 else 0.8
            pg.draw.rect(screen, tuple(int(c * shade) for c in base),
                         (a, month_y, b - a, _CAL_ROW_H))
            if start > h0:
                guide = (95, 95, 115) if m == 0 else (52, 52, 64)
                pg.draw.line(screen, guide, (a, py0), (a, py0 + ph), 1)
                pg.draw.line(screen, (18, 18, 22),
                             (a, month_y), (a, clock_y + _CAL_ROW_H - 1), 1)
            name = i18n.month_name(self._lang, m)
            self._blit_block_label((name, name[:3], name[0]), a, b, month_y)

        step = _clock_step_hours(tick_h, px_per_h)
        labelled = step is not None
        if step is None and HOURS_PER_DAY * px_per_h >= _CAL_MIN_DAY_PX:
            step = HOURS_PER_DAY
        if step is not None:
            for k in range(int(np.floor(h0 / step)), int(np.ceil(h1 / step))):
                start = k * step
                a, b = span(start, start + step)
                if b <= a:
                    continue
                if start > h0:
                    pg.draw.line(screen, (18, 18, 22), (a, clock_y),
                                 (a, clock_y + _CAL_ROW_H - 1), 1)
                if not labelled:
                    continue
                _month, dom, hour = _date_parts(start)
                if step < HOURS_PER_DAY:
                    labels = (f"{hour:02d}:00", f"{hour:02d}")
                else:
                    labels = (str(dom),)
                self._blit_block_label(labels, a, b, clock_y)
        pg.draw.rect(screen, (60, 60, 70),
                     (px0, band_y - 1, pw + 1, 2 * _CAL_ROW_H + 3), 1)

    def _blit_block_label(self, labels: Sequence[str], a: int, b: int,
                          row_y: int) -> None:
        """The first of ``labels`` that fits in [a, b), centred, shadowed.

        Widths are measured with ``font.size``: an antialiased
        ``font.render`` surface that is never blitted leaks its pixel
        buffer (section 60, section 189), and zoomed out over a long
        rollout nearly every block is too narrow for any variant.
        """
        if self._font.size(labels[-1])[0] + 4 > b - a:
            return
        for text in labels:
            if self._font.size(text)[0] + 4 > b - a:
                continue
            surf = self._font.render(text, True, (235, 235, 240))
            tx = a + (b - a - surf.get_width()) // 2
            ty = row_y + (_CAL_ROW_H - surf.get_height()) // 2
            shadow = self._font.render(text, True, (10, 10, 14))
            self._screen.blit(shadow, (tx + 1, ty + 1))
            self._screen.blit(surf, (tx, ty))
            return

    def _draw_plot_tooltip(self, mx: int, my: int, *,
                           active_ids: list,
                           buffers: dict,
                           xmin: float, xmax: float,
                           ymin: float, ymax: float,
                           px0: int, py0: int, pw: int, ph: int,
                           plot_rect: tuple) -> None:
        """Rita hover-tooltip vid muspekaren med y-värden per synlig serie.

        Interpolerar linjärt mellan de två närmaste samplepunkterna per
        serie vid den x-tick som motsvarar musens x-pixel. Serier utan
        data i synligt fönster eller som är avstängda via legend/solo
        hoppas över. Respekterar ``_log_plot``-läget: värdena presenteras
        alltid i sin råa (linjära) form oavsett skalning, så det som
        visas är det som ligger i bufferten.
        """
        pg = self._pg
        if xmax <= xmin or ph <= 0 or pw <= 0:
            return
        # Ticknummer under muspekaren (linjärt mellan xmin och xmax i pixelrymden).
        frac_x = (mx - px0) / float(pw)
        frac_x = max(0.0, min(1.0, frac_x))
        tick_at_mouse = xmin + frac_x * (xmax - xmin)
        viability = self._tabs[self._active_tab] == "viability"
        if viability and self._viability_v:
            # Probes are discrete: read the nearest one, not a blend.
            tick_at_mouse = min((step for step, _v in self._viability_v),
                                key=lambda step: abs(step - tick_at_mouse))
            mx = px0 + int((tick_at_mouse - xmin) / (xmax - xmin) * pw)

        # Samla (fid, colour, value) för alla aktiva serier med data.
        entries: list = []
        for fid in active_ids:
            if self._solo is not None and self._solo != fid:
                continue
            if not self._plot_enabled.get(fid, True):
                continue
            buf = buffers.get(fid)
            if not buf or len(buf) == 0:
                continue
            prev = None
            nxt = None
            steps, vals, _sorted = _series_arrays(buf)
            if _sorted:
                i = int(np.searchsorted(steps, tick_at_mouse, side="right"))
                j = int(np.searchsorted(steps, tick_at_mouse, side="left"))
                if i > 0:
                    prev = (steps[i - 1], vals[i - 1])
                if j < len(steps):
                    nxt = (steps[j], vals[j])
            else:
                for step, val in buf:
                    if step <= tick_at_mouse:
                        prev = (step, val)
                    if step >= tick_at_mouse and nxt is None:
                        nxt = (step, val)
                        break
            if prev is None and nxt is None:
                continue
            if prev is None:
                v = float(nxt[1])
            elif nxt is None:
                v = float(prev[1])
            elif prev[0] == nxt[0]:
                v = float(prev[1])
            else:
                # Linjär interpolation.
                t = (tick_at_mouse - prev[0]) / float(nxt[0] - prev[0])
                v = float(prev[1]) + t * (float(nxt[1]) - float(prev[1]))
            entries.append((fid, self._fg_colour[fid], v))

        if not entries:
            return

        # Vertikal spårlinje.
        pg.draw.line(self._screen, (140, 140, 160),
                     (mx, py0), (mx, py0 + ph), 1)

        # Rita en liten cirkelmarkör per serie vid dess (interpolerade) y.
        for fid, col, v in entries:
            v_plot = v
            if self._log_plot and self._tabs[self._active_tab] not in _LINEAR_TABS:
                v_plot = float(np.sign(v_plot) * np.log10(abs(v_plot) + 1e-12))
            if ymax - ymin < 1e-12:
                continue
            fy = (ymax - v_plot) / (ymax - ymin)
            yi = int(py0 + fy * ph)
            if py0 <= yi <= py0 + ph:
                try:
                    pg.draw.circle(self._screen, col, (mx, yi), 3)
                    pg.draw.circle(self._screen, (240, 240, 250), (mx, yi), 3, 1)
                except Exception:
                    pass

        # Bygg tooltip-text: rad 1 = tick, sedan en rad per serie.
        progress = self._tabs[self._active_tab] == "progress"
        head = self._t("day {n}" if progress else "update {n}" if viability else "tick {n}",
                       n=int(round(tick_at_mouse)))
        inference = (str(self.mode).lower().startswith("infer")
                     and not progress and not viability)
        if inference:
            # Simulation year from the start, counted like the status
            # bar's time (the point at step s is the state at tick s + 1).
            hours = (int(round(tick_at_mouse)) + 1) * int(self.tick_hours)
            head += "  " + self._t(
                "year {n}",
                n=hours // (DAYS_PER_YEAR * HOURS_PER_DAY) + 1)
        # Same instant as the calendar band under the plot (section 155).
        if self._daylight_cfg and inference:
            head += "  " + _calendar_label(float(_plot_x_hours(
                self._daylight_cfg, round(tick_at_mouse))), self._lang)
        lines: list = [(head, (220, 220, 230))]
        # Sortera efter y-värde (fallande) för läsbarhet.
        entries_sorted = sorted(entries, key=lambda e: -e[2])
        # Plain decimals like the y axis (section 196): no scientific
        # notation, no plus sign, the axis's number of decimals so the
        # values read against its labels. On a log-scaled axis (whose
        # labels are log10) the raw values get three significant digits
        # of the largest one instead.
        if (self._log_plot
                and self._tabs[self._active_tab] not in _LINEAR_TABS):
            big = max(abs(e[2]) for e in entries_sorted)
            step = 10.0 ** (np.floor(np.log10(big)) - 1) if big > 0 else 0.0
        else:
            step = (ymax - ymin) / 4.0
        texts = _axis_labels([e[2] for e in entries_sorted], step)
        if viability:
            # Margins read to two decimals; the axis clips below -1.
            texts = ["<= -1" if e[2] <= _VIABILITY_AXIS[0] else f"{e[2]:+.2f}"
                     for e in entries_sorted]
        if viability and any(step == tick_at_mouse for step, _v in self._viability_v):
            score = self._viability_scores.get(tick_at_mouse)
            text = (self._t("extinct") if score is None
                    else "<= -1" if score <= _VIABILITY_AXIS[0] else f"{score:+.2f}")
            lines.append((f"V: {text}", (235, 235, 245)))
            extinct = {fid for step, fid in self._viability_extinct if step == tick_at_mouse}
        for (fid, col, _v), txt in zip(entries_sorted, texts):
            name = self._display_name(fid)
            if viability and fid in extinct:
                txt = self._t("extinct")
            lines.append((f"{name}: {txt}", col))
        # A box wider than the room on either side (long group names in a
        # narrow plot) is held inside the plot area instead of spilling
        # over its edge (and out of what the next repaint redraws).
        self._draw_tooltip_box(lines, mx, my, flip_rect=plot_rect,
                               clamp_rect=(px0, py0, pw, ph))

    def _draw_member_tooltip(self) -> None:
        """The species of the FG whose info lines are under the pointer,
        then - below a rule - the prey behind the codes on its diet line.

        Drawn from ``_render_full`` after every panel, so it lies on top
        of the neighbouring heatmaps and the plot, and is held inside the
        window. Not while a spawn popup is open (that owns the pointer).
        """
        if not self._member_rects or self._spawn_popup or self._lang_menu_open:
            return
        try:
            mx, my = self._pg.mouse.get_pos()
        except Exception:
            return
        fid = next((f for rect, f in self._member_rects
                    if self._hit_rect(rect, (mx, my))), None)
        if fid is None:
            return
        members = self._member_lines(self._members.get(fid), self._lang)
        legend = self._panel_legend.get(fid) or []
        predators = self._panel_predators.get(fid) or []
        if not members and not legend and not predators:
            return
        lines = [(self._display_name(fid), self._fg_colour[fid])]
        lines += [(txt, (210, 210, 220)) for txt in members]
        if members and legend:
            lines.append(None)
        lines += [(f"{code} = {meaning}", (170, 170, 185))
                  for code, meaning in legend]
        if predators:
            # Share of this group's predation loss by predator.
            if members or legend:
                lines.append(None)
            lines.append((self._t("Eaten by"), (210, 210, 220)))
            lines += [(f"  {self._display_name(pid)} {share * 100.0:.0f}%",
                       self._fg_colour.get(pid, (170, 170, 185)))
                      for pid, share in predators]
        win = (0, 0) + tuple(self._screen.get_size())
        self._draw_tooltip_box(lines, mx, my, flip_rect=win, clamp_rect=win)

    def _draw_tooltip_box(self, lines: list, mx: int, my: int, *,
                          flip_rect: tuple, clamp_rect: tuple) -> None:
        """Draw ``lines`` ((text, colour) pairs) in a box by the pointer.

        A ``None`` entry is a horizontal rule between two blocks of lines.

        The box goes right of and below the pointer, flips to the other
        side where it would cross ``flip_rect``, and is finally held
        inside ``clamp_rect``.
        """
        pg = self._pg
        # Mät textstorlek.
        pad = 4
        line_h = 12
        rule_h = 7
        max_w = 0
        surfs = []
        for entry in lines:
            if entry is None:
                surfs.append(None)
                continue
            txt, col = entry
            s = self._font.render(txt, True, col)
            surfs.append(s)
            if s.get_width() > max_w:
                max_w = s.get_width()
        box_w = max_w + 2 * pad
        box_h = (sum(rule_h if s is None else line_h for s in surfs)
                 + 2 * pad)

        # Placera boxen — helst till höger om muspekaren, men flippa till
        # vänster om den skulle sticka utanför ``flip_rect``.
        prx, pry, prw, prh = flip_rect
        bx = mx + 12
        if bx + box_w > prx + prw - 2:
            bx = mx - 12 - box_w
        by = my + 12
        if by + box_h > pry + prh - 2:
            by = my - 12 - box_h
        cx, cy, cw, ch = clamp_rect
        bx = max(cx, min(bx, cx + cw - box_w))
        by = max(cy, min(by, cy + ch - box_h))

        # Halvtransparent bakgrund via Surface med per-pixel alpha.
        try:
            box_surf = pg.Surface((box_w, box_h), pg.SRCALPHA)
            box_surf.fill((20, 22, 30, 230))
            self._screen.blit(box_surf, (bx, by))
        except Exception:
            pg.draw.rect(self._screen, (20, 22, 30), (bx, by, box_w, box_h))
        pg.draw.rect(self._screen, (140, 140, 160), (bx, by, box_w, box_h), 1)
        y = by + pad
        for s in surfs:
            if s is None:
                ry = y + rule_h // 2 - 1
                pg.draw.line(self._screen, (110, 110, 130),
                             (bx + pad, ry), (bx + box_w - pad - 1, ry))
                y += rule_h
                continue
            self._screen.blit(s, (bx + pad, y))
            y += line_h


def _fg_colour_for(idx: int, n: int) -> tuple:
    """Distinct-ish RGB colour per FG, deterministic by index."""
    # Golden-ratio hue spacing for visually well-separated colours.
    import colorsys
    hue = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)
