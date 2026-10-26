"""Generate ``Strategi_okt2.pdf`` - the second October 2026 edition of
the strategy deck, written against the mechanics the code runs after
section 185 of mareld_resume.txt.

A copy of ``tools/make_strategi_deck.py`` (the 12 October edition, kept
so ``Strategi_okt.pdf`` can still be rebuilt): the FG set grew from 8 to
10 groups (microzooplankton, subsurface phytoplankton), the seals are
active, the water column has two layers and a nutrient pool, fishing is
in today's world, and a new part explains the biology behind every
group's growth rate and carrying capacity.

Everything stated on a slide is read from, or checked against:
  lib/environments/ecosystem_env/*.py   the tick
  lib/runners/trainer.py, parallel_worker.py, policy.py   ARS
  lib/world/tick_time.py, functional_group.py, energy_balance.py
  lib/world/growth_budget.py, daylight.py, temperature.py, nutrients.py
  lib/gpu/   the tensor engine and trainer behind train_gpu.py
  fgconfig/fg_library.yaml, mareld2.yaml   the calibration
  VIABILITY.md, mareld_resume.txt sections 69-185

The value tables, the growth-budget table, the pair table, the
Observability matrix, the food web and the fishing tables are read from
fgconfig/fg_library.yaml, mareld2.yaml and lib/world/growth_budget.py at
build time, so they cannot drift from the calibration.

Slide text is Swedish; parameter names keep their English library
spelling so a slide can be traced to the YAML key it describes. The
English edition is tools/make_strategy_deck2_en.py.

Usage:
    python3 tools/make_strategi_deck2.py [-o Strategi_okt2.pdf] [--png DIR]
"""
import argparse
import os
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath

# ----------------------------------------------------------------------
# Page geometry and palette
# ----------------------------------------------------------------------
PAGE_W, PAGE_H = 960.0, 540.0          # points, 16:9 - as the original
DPI = 72.0                             # 1 axis unit == 1 pt

INK = "#14243A"          # body text
NAVY = "#12355B"         # titles
ACCENT = "#1F7A8C"       # rules, bullets
ACCENT_2 = "#BF5B04"     # highlight / "new since May"
MUTED = "#6B7A8C"        # footnotes
PAPER = "#FFFFFF"
BAND = "#EEF3F7"         # table/figure background
BAND_2 = "#F7FAFC"

SANS = ["DejaVu Sans"]
MONO = ["DejaVu Sans Mono"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": SANS,
    "pdf.fonttype": 42,
    "text.usetex": False,
    # Formulas are matplotlib mathtext ($...$) in Computer Modern, so they
    # read like LaTeX without a TeX install; å/ä/ö fall back to DejaVu.
    "mathtext.fontset": "cm",
})

TITLE_SIZE = 27
SUB_SIZE = 15
BODY_SIZE = 16.5
BODY_SIZE_2 = 14
NOTE_SIZE = 11.5

LEFT = 62.0
RIGHT = 62.0
BODY_TOP = 400.0        # first body line baseline
BODY_BOTTOM = 52.0


class Deck:
    """A 16:9 slide deck written straight to a PdfPages file."""

    #: Footer and section-number label; the English build
    #: (tools/make_strategy_deck2_en.py) overrides both.
    FOOTER = "Strategi - Mareld / Poseidon Nord - oktober 2026, utgåva 2"
    PART = "Del"

    def __init__(self, pdf, png_dir=None):
        self.pdf = pdf
        self.png_dir = png_dir
        self.n = 0
        self.total = None

    # -- page scaffolding ------------------------------------------------
    def _page(self):
        fig = plt.figure(figsize=(PAGE_W / DPI, PAGE_H / DPI), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, PAGE_W)
        ax.set_ylim(0, PAGE_H)
        ax.set_axis_off()
        ax.add_patch(Rectangle((0, 0), PAGE_W, PAGE_H, color=PAPER, zorder=-10))
        return fig, ax

    def _close(self, fig, name=None):
        self.pdf.savefig(fig)
        if self.png_dir:
            tag = name or f"slide_{self.n:02d}"
            fig.savefig(os.path.join(self.png_dir, f"{tag}.png"), dpi=DPI)
        plt.close(fig)

    def _chrome(self, ax, title, kicker=None, note=None):
        """Title bar, accent rule, footer. Returns the y of the first line."""
        self.n += 1
        y = 486.0
        if kicker:
            ax.text(LEFT, 508.0, kicker.upper(), color=ACCENT, size=11.5,
                    weight="bold", va="center", ha="left", family="sans-serif")
        lines = wrap(title, TITLE_SIZE, PAGE_W - LEFT - RIGHT, weight="bold")
        for i, line in enumerate(lines):
            ax.text(LEFT, y - i * (TITLE_SIZE + 6), line, color=NAVY,
                    size=TITLE_SIZE, weight="bold", va="center", ha="left")
        y -= (len(lines) - 1) * (TITLE_SIZE + 6)
        rule_y = y - 26
        ax.plot([LEFT, PAGE_W - RIGHT], [rule_y, rule_y], color=ACCENT,
                lw=1.6, solid_capstyle="butt")
        if note:
            ax.text(PAGE_W - RIGHT, rule_y + 10, note, color=ACCENT_2,
                    size=11.5, weight="bold", va="center", ha="right")
        # footer
        ax.text(LEFT, 26, self.FOOTER,
                color=MUTED, size=9.5, va="center", ha="left")
        ax.text(PAGE_W - RIGHT, 26, str(self.n), color=MUTED, size=9.5,
                va="center", ha="right")
        return rule_y - 30

    # -- slide kinds -----------------------------------------------------
    def title_slide(self, title, subtitle, lines):
        fig, ax = self._page()
        self.n += 1
        ax.add_patch(Rectangle((0, 0), PAGE_W, PAGE_H, color=NAVY, zorder=-5))
        ax.add_patch(Rectangle((0, 0), PAGE_W, 8, color=ACCENT, zorder=-4))
        ax.text(LEFT, 352, title, color="white", size=52, weight="bold",
                va="center", ha="left")
        ax.text(LEFT, 300, subtitle, color="#BFD7E6", size=21,
                va="center", ha="left")
        ax.plot([LEFT, LEFT + 150], [270, 270], color=ACCENT, lw=3)
        y = 232
        for line in lines:
            ax.text(LEFT, y, line, color="#D9E6EF", size=14, va="center",
                    ha="left")
            y -= 26
        self._close(fig, "slide_01_title")

    def section(self, number, title, blurb=None):
        fig, ax = self._page()
        self.n += 1
        ax.add_patch(Rectangle((0, 0), PAGE_W, PAGE_H, color=BAND, zorder=-5))
        ax.add_patch(Rectangle((0, 0), 14, PAGE_H, color=ACCENT, zorder=-4))
        if number:
            ax.text(LEFT, 320, f"{self.PART} {number}", color=ACCENT, size=16,
                    weight="bold", va="center", ha="left")
        lines = wrap(title, 40, PAGE_W - LEFT - RIGHT - 60, weight="bold")
        for i, line in enumerate(lines):
            ax.text(LEFT, 272 - i * 48, line, color=NAVY, size=40,
                    weight="bold", va="center", ha="left")
        if blurb:
            yb = 210 - (len(lines) - 1) * 48
            for line in wrap(blurb, 15, PAGE_W - LEFT - RIGHT - 160):
                ax.text(LEFT, yb, line, color=INK, size=15, va="center",
                        ha="left")
                yb -= 22
        ax.text(PAGE_W - RIGHT, 26, str(self.n), color=MUTED, size=9.5,
                va="center", ha="right")
        self._close(fig, f"slide_{self.n:02d}_section")

    def bullets(self, title, items, kicker=None, note=None, footnote=None,
                size=BODY_SIZE):
        fig, ax = self._page()
        y = self._chrome(ax, title, kicker, note)
        draw_bullets(ax, items, LEFT, y, PAGE_W - LEFT - RIGHT, size=size)
        if footnote:
            draw_footnote(ax, footnote)
        self._close(fig)

    def table(self, title, headers, rows, widths, kicker=None, note=None,
              footnote=None, size=12.5, lead=None, align=None):
        fig, ax = self._page()
        y = self._chrome(ax, title, kicker, note)
        if lead:
            y = draw_bullets(ax, [lead], LEFT, y, PAGE_W - LEFT - RIGHT,
                             size=BODY_SIZE_2) - 6
        draw_table(ax, headers, rows, widths, LEFT, y, size=size, align=align)
        if footnote:
            draw_footnote(ax, footnote)
        self._close(fig)

    def figure(self, title, draw, kicker=None, note=None, footnote=None,
               lead=None, items=None):
        fig, ax = self._page()
        y = self._chrome(ax, title, kicker, note)
        if lead:
            y = draw_bullets(ax, [lead], LEFT, y, PAGE_W - LEFT - RIGHT,
                             size=BODY_SIZE_2) - 4
        draw(fig, ax, y)
        if items:
            draw_bullets(ax, items, LEFT, 150, PAGE_W - LEFT - RIGHT,
                         size=BODY_SIZE_2)
        if footnote:
            draw_footnote(ax, footnote)
        self._close(fig)

    def formula(self, title, blocks, kicker=None, note=None, footnote=None,
                lead=None, tail=None):
        fig, ax = self._page()
        y = self._chrome(ax, title, kicker, note)
        if lead:
            y = draw_bullets(ax, lead, LEFT, y, PAGE_W - LEFT - RIGHT,
                             size=BODY_SIZE_2) - 10
        for caption, code in blocks:
            draw = draw_math if isinstance(code, list) else draw_code
            y = draw(ax, caption, code, LEFT, y, PAGE_W - LEFT - RIGHT)
            y -= 14
        if tail:
            draw_bullets(ax, tail, LEFT, y - 2, PAGE_W - LEFT - RIGHT,
                         size=BODY_SIZE_2)
        if footnote:
            draw_footnote(ax, footnote)
        self._close(fig)


# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------
_WIDTH_CACHE = {}


def _glyph_width(ch, weight, mono):
    """Advance of one character at size 1, measured once per glyph."""
    key = (ch, weight, mono)
    hit = _WIDTH_CACHE.get(key)
    if hit is None:
        prop = FontProperties(family=MONO[0] if mono else SANS[0],
                              weight=weight, size=100)
        # Measure between two reference glyphs so side bearings and the
        # width of a space are included.
        ref = TextPath((0, 0), "HH", size=100, prop=prop).get_extents().x1
        both = TextPath((0, 0), "H" + ch + "H", size=100,
                        prop=prop).get_extents().x1
        hit = max(0.0, (both - ref) / 100.0)
        _WIDTH_CACHE[key] = hit
    return hit


_MATH_CACHE = {}


def math_extent(tex, size):
    """(width, height, descent) in points of mathtext ``tex`` (no $)."""
    key = (tex, size)
    hit = _MATH_CACHE.get(key)
    if hit is None:
        ext = TextPath((0, 0), "$" + tex + "$", size=size,
                       prop=FontProperties(family=SANS[0])).get_extents()
        hit = (ext.x1, ext.height, -min(0.0, ext.y0))
        _MATH_CACHE[key] = hit
    return hit


def text_width(text, size, weight="normal", mono=False):
    """Rendered width of ``text`` in points (per-glyph sum, no kerning).
    ``$...$`` segments are measured as mathtext."""
    parts = str(text).split("$")
    if len(parts) % 2 == 0:            # unpaired $: plain text
        parts = [str(text)]
    total = 0.0
    for i, part in enumerate(parts):
        if i % 2:
            total += math_extent(part, size)[0] if part else 0.0
        else:
            total += size * sum(_glyph_width(ch, weight, mono) for ch in part)
    return total


def char_width(size, weight="normal", mono=False):
    """Average glyph advance in points for the deck's fonts."""
    if mono:
        return 0.602 * size
    return (0.545 if weight == "bold" else 0.515) * size


def wrap(text, size, width, weight="normal", mono=False):
    """Greedy word wrap measured against the actual glyph widths."""
    words = str(text).split()
    if not words:
        return [""]
    lines, line = [], words[0]
    for word in words[1:]:
        trial = line + " " + word
        if text_width(trial, size, weight, mono) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def draw_bullets(ax, items, x, y, width, size=BODY_SIZE):
    """items: str (level 0), ('-', str) level 1, ('--', str) level 2,
    ('>', str) for a highlighted line, or ('', '') for a blank line."""
    lead = size * 1.42
    for item in items:
        if isinstance(item, tuple):
            kind, text = item
        else:
            kind, text = "", item
        if not text:
            y -= lead * 0.5
            continue
        if kind == "-":
            indent, marker, color, s = 26, "–", INK, size - 2.0
        elif kind == "--":
            indent, marker, color, s = 50, "·", MUTED, size - 3.0
        elif kind == ">":
            indent, marker, color, s = 0, "▸", ACCENT_2, size
        elif kind == "p":          # plain paragraph, no marker
            indent, marker, color, s = 0, "", INK, size
        else:
            indent, marker, color, s = 0, "▪", ACCENT, size
        lines = wrap(text, s, width - indent - 18)
        if marker:
            ax.text(x + indent, y, marker, color=color, size=s * 0.8,
                    va="center", ha="left")
        for i, line in enumerate(lines):
            ax.text(x + indent + 18, y - i * (s * 1.3), line,
                    color=INK if kind != ">" else ACCENT_2, size=s,
                    va="center", ha="left",
                    weight="bold" if kind == ">" else "normal")
        y -= (len(lines) - 1) * (s * 1.3) + lead
    return y


def draw_footnote(ax, text):
    for i, line in enumerate(wrap(text, NOTE_SIZE, PAGE_W - LEFT - RIGHT)):
        ax.text(LEFT, 46 + 13 * (len(wrap(text, NOTE_SIZE,
                                          PAGE_W - LEFT - RIGHT)) - 1 - i),
                line, color=MUTED, size=NOTE_SIZE, va="center", ha="left",
                style="italic")


def draw_code(ax, caption, code, x, y, width, size=13.5):
    """A light box with monospaced lines. Returns the y below the box."""
    lines = code.strip("\n").split("\n")
    lh = size * 1.45
    pad = 12
    head = 0
    if caption:
        head = 20
    h = pad * 2 + head + lh * len(lines) - (lh - size)
    top = y + size * 0.8
    ax.add_patch(FancyBboxPatch(
        (x, top - h), width, h, boxstyle="round,pad=0,rounding_size=6",
        facecolor=BAND, edgecolor="#D3DEE8", lw=1.0, zorder=0))
    ax.add_patch(Rectangle((x, top - h), 4, h, color=ACCENT, zorder=1))
    yy = top - pad - size * 0.5
    if caption:
        ax.text(x + 18, yy, caption, color=NAVY, size=12.5, weight="bold",
                va="center", ha="left")
        yy -= head
    for line in lines:
        ax.text(x + 18, yy, line, color=INK, size=size, va="center",
                ha="left", family="monospace")
        yy -= lh
    return top - h - 6


def draw_math(ax, caption, lines, x, y, width, size=14.0):
    """A light box of display formulas, LaTeX align style.

    ``lines`` entries: a mathtext string without $ (``&`` marks the
    alignment point, as in align), ``""`` for a half-line gap, or
    ``("t", text)`` for a plain line that may carry inline ``$...$``.
    Returns the y below the box, like ``draw_code``."""
    pad, head = 12, (20 if caption else 0)
    gap = size * 0.55
    rows, col = [], 0.0
    for line in lines:
        if line == "":
            rows.append(("gap", None, gap))
        elif isinstance(line, tuple):
            rows.append(("text", line[1], size * 1.45))
        else:
            lhs, _, rhs = line.partition("&")
            if not _:
                lhs, rhs = "", line
            h = max(math_extent(part, size)[1] for part in (lhs, rhs) if part)
            if lhs:
                col = max(col, math_extent(lhs, size)[0])
            rows.append(("math", (lhs, rhs), max(size * 1.5, h + size * 0.5)))
    h = pad * 2 + head + sum(r[2] for r in rows)
    top = y + size * 0.8
    ax.add_patch(FancyBboxPatch(
        (x, top - h), width, h, boxstyle="round,pad=0,rounding_size=6",
        facecolor=BAND, edgecolor="#D3DEE8", lw=1.0, zorder=0))
    ax.add_patch(Rectangle((x, top - h), 4, h, color=ACCENT, zorder=1))
    yy = top - pad
    if caption:
        ax.text(x + 18, yy - size * 0.5, caption, color=NAVY, size=12.5,
                weight="bold", va="center", ha="left")
        yy -= head
    x_eq = x + 26 + col
    for kind, body, rh in rows:
        mid = yy - rh / 2
        if kind == "text":
            ax.text(x + 26, mid, body, color=INK, size=size - 2.5,
                    va="center", ha="left")
        elif kind == "math":
            lhs, rhs = body
            if lhs:
                ax.text(x_eq, mid, "$" + lhs + "$", color=INK, size=size,
                        va="center", ha="right")
                ax.text(x_eq, mid, "$" + rhs + "$", color=INK, size=size,
                        va="center", ha="left")
            else:
                ax.text(x + 26, mid, "$" + rhs + "$", color=INK, size=size,
                        va="center", ha="left")
        yy -= rh
    return top - h - 6


def _cell_anchor(cx, width, how):
    if how == "right":
        return cx + width - 8, "right"
    if how == "center":
        return cx + width / 2, "center"
    return cx + 8, "left"


def draw_table(ax, headers, rows, widths, x, y, size=12.5, align=None):
    total = sum(widths)
    scale = (PAGE_W - LEFT - RIGHT) / total
    widths = [w * scale for w in widths]
    align = align or ["left"] * len(headers)
    head_h = size * 2.1
    row_h = size * 1.95
    top = y + size

    # wrap cells first so a tall row gets the height it needs
    wrapped, heights = [], []
    for row in rows:
        cells = []
        for j, cell in enumerate(row):
            cells.append(wrap(str(cell), size, widths[j] - 16))
        wrapped.append(cells)
        heights.append(max(row_h, size * 1.5 * max(len(c) for c in cells) + 8))

    ax.add_patch(Rectangle((x, top - head_h), sum(widths), head_h,
                           facecolor=NAVY, edgecolor="none", zorder=1))
    cx = x
    for j, head in enumerate(headers):
        hx, ha = _cell_anchor(cx, widths[j], align[j])
        ax.text(hx, top - head_h / 2, head, color="white", size=size,
                weight="bold", va="center", ha=ha, zorder=2)
        cx += widths[j]

    yy = top - head_h
    for i, cells in enumerate(wrapped):
        h = heights[i]
        if i % 2 == 0:
            ax.add_patch(Rectangle((x, yy - h), sum(widths), h,
                                   facecolor=BAND_2, edgecolor="none",
                                   zorder=0))
        ax.plot([x, x + sum(widths)], [yy - h, yy - h], color="#DCE5EC",
                lw=0.8, zorder=1)
        cx = x
        for j, cell_lines in enumerate(cells):
            tx, ha = _cell_anchor(cx, widths[j], align[j])
            for k, line in enumerate(cell_lines):
                ax.text(tx, yy - h / 2 + (len(cell_lines) - 1 - 2 * k)
                        * size * 0.75, line, color=INK, size=size,
                        va="center", ha=ha, zorder=2)
            cx += widths[j]
        yy -= h
    return yy


# ----------------------------------------------------------------------
# Diagram labels
# ----------------------------------------------------------------------
#: Every piece of text the diagram functions draw. The English build
#: replaces the entries with ``LABELS.update(...)`` before it draws and
#: asserts that it supplies exactly these keys.
LABELS = {
    "pipeline": [
        ("1  Beslut", "observation → policy → maskad softmax → hela enheter"),
        ("2  Påverkansmortalitet", "impact_table × karta, alla FG (fisket)"),
        ("3  Predation", "Holling II/III, interferens, ljus, Q10 på intaget"),
        ("4  Energikostnader", "vila / äta / röra sig × Q10(T); sill växer här"),
        ("5  Rörelse", "N/E/S/W × hastighet, energi följer med"),
        ("6  Strömmar", "current_response; näringspoolen följer vattnet"),
        ("7  Vattenpelaren", "lagerutbyte, tillförsel, betning av icke beslutande"),
        ("8  Tillväxt och förluster", "M1, regionalt K, massbalanserad tillväxt"),
        ("9  Masker", "accessibility, utdöendetröskel"),
    ],
    "web_nodes": {
        "phytoplankton": "Växtplankton",
        "subsurface_phytoplankton": "Djupväxtplankton",
        "microzooplankton": "Mikrozooplankton",
        "benthic_infauna": "Bottensamhälle",
        "zooplankton": "Djurplankton", "pelagic_fish": "Pelagisk fisk",
        "gadoids": "Torskfiskar", "porpoises": "Tumlare", "seals": "Sälar",
        "seabirds": "Sjöfåglar",
    },
    "web_legend": "grönt = icke beslutande (näringspool, betning, logistik)\n"
                  "blått = beslutande (policynätverk och energibudget)\n"
                  "streckat = muted i mareld2.yaml",
    "web_self": "kannibalism (urkryssad)",
    "fr_herr": "pelagisk fisk  typ III, h = {h}",
    "fr_herr_w": "pelagisk fisk  w = 1, B_pred = 0,5",
    "fr_benth": "bottensamhälle  typ II, h = {h}",
    "fr_ceiling_e": "energitak 4 500/130",
    "fr_title_gad": "Torskfiskar: a = {a}, h ∝ bytets energi",
    "fr_title_zoo": "Djurplankton: halvmättnad i koncentration",
    "fr_zoo_m6": "växtpl., typ II, M = 6 m (sommar)",
    "fr_zoo_m17": "växtpl., typ II, M = 17 m (november)",
    "fr_zoo_micro": "mikrozoopl., typ III, M = 10 m",
    "fr_ceiling": "tak 1/h",
    "fr_xlabel": "synlig bytesbiomassa i cellen  [ton]",
    "fr_ylabel": "intag  [ton byte / ton predator / tick]",
    "fr_ylabel_e": "energiintag  [MJ / ton torskfisk / tick]",
    "hg_u5": "u = 0,5  (djurplankton, torskfisk, säl, tumlare)",
    "hg_u2": "u = 0,2  (pelagisk fisk)",
    "hg_xlabel1": "mättnadsgrad  s_X = E_X / ME_X",
    "hg_ylabel1": "hungerfaktor  h_X",
    "hg_zoo": "djurplankton  g = {g} / SR = {sr}",
    "hg_gad": "torskfisk  g = {g} / SR = {sr}  (×10)",
    "hg_sstar": "s*",
    "hg_xlabel2": "mättnadsgrad  s_X  (q_X = s_X − u_X)",
    "hg_ylabel2": "relativ biomassaändring / tick",
    "ir_title": "{fg}: s* = {s}",
    "ir_income": "intag h(s) · C",
    "ir_need": "kostnad + P · (ec + s · ME)",
    "ir_xlabel": "mättnadsgrad s",
    "ir_ylabel": "MJ / ton / tick",
    "ir_fg": {"zooplankton": "Djurplankton", "gadoids": "Torskfiskar"},
    "cap_title": "{fg}: r = {r}, K = {K} t/km², F = {F}",
    "cap_withk": "med K:  r (1 − B/K) − F",
    "cap_nok": "utan K:  r − F",
    "cap_today": "dagens B",
    "cap_eq": "jämvikt",
    "cap_xlabel": "regional täthet B  [t/km²]",
    "cap_ylabel": "nettotillväxt  [1/år]",
    "cap_fg": {"gadoids": "Torskfiskar", "pelagic_fish": "Pelagisk fisk"},
    "fm_title_b": "Bottentrål, medel {m} MW-h/km²/år",
    "fm_title_p": "Pelagisk trål, medel {m} MW-h/km²/år",
    "wc_light": "ljus  I0(t) · e^(−K_d z)",
    "wc_mixed": "Blandat lager  0 – M(t)   (6-17 m)",
    "wc_mixed_items": "växtplankton · näringspool N_m · mikro- och djurplankton",
    "wc_sub": "Djuplager  M(t) – 30 m",
    "wc_sub_items": "djupväxtplankton (I_k 30) · näringspool N_d",
    "wc_res": "Reservoar 50 m  (SHARK A17, nitrat per månad)",
    "wc_e": "utbyte e(t)\n0,05-0,3 m/d",
    "wc_eb": "utbyte e_b(t)\n0,2-1,0 m/d",
    "wc_ent": "medryckning när M(t)\nväxer, kvarlämning\nnär det krymper",
    "wc_regen": "betning ger tillbaka\n0,3 (djurpl.) / 0,4 (mikro)",
    "wc_export": "M1 (sjunkning)\nlämnar kretsen",
    "obs_centre": "egen cell",
    "obs_shape": "plusform: egen cell + fyra grannar",
    "obs_caption": "Kanaler per cell (kompakt layout)",
    "obs_code": """
egen cell  [ B_egen, s_egen, B_obs_1..k, imp_1..m ]
granne d   [ B_egen, B_obs_1..k, imp_1..m ]
sist       [ ljus F(t) ]       (med dagsljuskalendern)

in_dim_X = (2 + k_X + m) + 4 · (1 + k_X + m) + 1
""",
    "obs_bullets": [
        ("-", "k_X = antal ANDRA aktiva FG som X observerar enligt "
              "Observability-matrisen"),
        ("-", "m = antal aktiva impaktkartor märkta observable - i dag "
              "0: fisket är inte observerbart, vindkraften muted"),
        ("-", "B_obs är SYNLIG biomassa: den gömda (vilande) andelen "
              "dras bort"),
        ("-", "Mareld i dag: k = 3-5 och en ljuskanal, dvs 22-32 kanaler"),
    ],
    "net_in": "observation\n(in_dim)",
    "net_h1": "dolt lager\n16 noder, tanh",
    "net_out": "logits → mask → softmax",
    "net_actions": ["Flytta N", "Flytta E/S/W", "Vila (= gömma sig)",
                    "Äta Y_1 ... Y_n", ""],
    "ef_stages": ["Intag  intag_XY", "Assimilerat  · assimilation",
                  "Reserv R_X  (+ intag − kostnad)", "Ny biomassa  dB ton"],
    "ef_spill": "spill",
    "ef_costs": "vila / rörelse / födosök\n+ påverkan",
    "ef_price": "priset dB · energy_content debiteras reserven",
    "spawn_panels": ["uniform", "perlin  (växtplankton)",
                     "colony  (pelagisk fisk)",
                     "env_driven  (följer växtplankton)"],
}


# ----------------------------------------------------------------------
# Diagram helpers
# ----------------------------------------------------------------------
def axes_pt(fig, x, y, w, h):
    """Add an axes positioned in page points."""
    return fig.add_axes([x / PAGE_W, y / PAGE_H, w / PAGE_W, h / PAGE_H])


def box(ax, x, y, w, h, text, face=BAND, edge=ACCENT, color=INK, size=11.5,
        weight="normal", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=5",
                                facecolor=face, edgecolor=edge, lw=lw,
                                zorder=2))
    lines = wrap(text, size, w - 10, weight=weight)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + (len(lines) - 1 - 2 * i) * size * 0.62,
                line, color=color, size=size, weight=weight, va="center",
                ha="center", zorder=3)


def arrow(ax, x0, y0, x1, y1, color=ACCENT, lw=1.4, head=7.0, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls,
                                shrinkA=0, shrinkB=0,
                                mutation_scale=head * 1.6), zorder=2)


def draw_tick_pipeline(fig, ax, y):
    steps = LABELS["pipeline"]
    top = y - 4
    bottom = 78.0
    gap = 6.0
    h = (top - bottom - gap * (len(steps) - 1)) / len(steps)
    w = PAGE_W - LEFT - RIGHT
    for i, (head, sub) in enumerate(steps):
        yy = top - i * (h + gap) - h
        face = BAND if i % 2 == 0 else BAND_2
        ax.add_patch(FancyBboxPatch((LEFT, yy), w, h,
                                    boxstyle="round,pad=0,rounding_size=5",
                                    facecolor=face, edgecolor="#D3DEE8",
                                    lw=1.0, zorder=1))
        ax.add_patch(Rectangle((LEFT, yy), 5, h, color=ACCENT, zorder=2))
        ax.text(LEFT + 20, yy + h / 2, head, color=NAVY, size=13.5,
                weight="bold", va="center", ha="left", zorder=3)
        ax.text(LEFT + 270, yy + h / 2, sub, color=INK, size=12.5,
                va="center", ha="left", zorder=3)
        if i < len(steps) - 1:
            arrow(ax, LEFT + 120, yy, LEFT + 120, yy - gap, lw=1.0, head=4)


def _rect_edge(cx, cy, w, h, tx, ty, pad=4.0):
    """Point where the segment (cx,cy)->(tx,ty) leaves the box."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (w / 2 + pad) / abs(dx) if dx else float("inf")
    sy = (h / 2 + pad) / abs(dy) if dy else float("inf")
    t = min(sx, sy)
    return cx + dx * t, cy + dy * t


# ----------------------------------------------------------------------
# Library values, read at build time so the tables cannot drift
# ----------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
_YAML_CACHE = {}

DM_ORDER = ["zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals",
            "seabirds"]
NDM_ORDER = ["phytoplankton", "subsurface_phytoplankton", "microzooplankton",
             "benthic_infauna"]
OBS_ORDER = ["phytoplankton", "subsurface_phytoplankton", "microzooplankton",
             "zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals",
             "seabirds", "benthic_infauna"]
PAIR_ORDER = [
    ("microzooplankton", "phytoplankton"),
    ("microzooplankton", "subsurface_phytoplankton"),
    ("zooplankton", "phytoplankton"),
    ("zooplankton", "subsurface_phytoplankton"),
    ("zooplankton", "microzooplankton"),
    ("pelagic_fish", "zooplankton"),
    ("gadoids", "pelagic_fish"), ("gadoids", "zooplankton"),
    ("gadoids", "benthic_infauna"), ("gadoids", "gadoids"),
    ("porpoises", "pelagic_fish"), ("porpoises", "gadoids"),
    ("seals", "pelagic_fish"), ("seals", "gadoids"),
    ("seabirds", "pelagic_fish"),
]
#: Groups in the order the growth-budget table lists them.
BUDGET_ORDER = ["phytoplankton", "subsurface_phytoplankton", "zooplankton",
                "pelagic_fish", "gadoids", "porpoises", "seals"]


def _yaml(rel):
    hit = _YAML_CACHE.get(rel)
    if hit is None:
        import yaml
        with open(os.path.join(REPO, rel), encoding="utf-8-sig") as fh:
            hit = yaml.safe_load(fh)
        _YAML_CACHE[rel] = hit
    return hit


def library():
    return _yaml(os.path.join("fgconfig", "fg_library.yaml"))


def manifest():
    return _yaml("mareld2.yaml")


def species(name):
    return library()["species_definitions"][name]


def pair(pred, prey, rel="preys_on"):
    return library()["interaction_definitions"].get(
        f"{pred}_{rel}_{prey}") or {}


def muted_groups():
    """FG ids muted in mareld2.yaml."""
    m = manifest()
    out = set()
    for key in ("decision_makers", "non_decision_makers"):
        for entry in m.get(key) or []:
            if entry.get("muted"):
                out.add(entry["group_id"])
    return out


def active_groups():
    m = manifest()
    return [e["group_id"] for key in ("decision_makers", "non_decision_makers")
            for e in m.get(key) or [] if not e.get("muted")]


def budget(fg):
    """The live growth budget of ``fg`` for mareld2.yaml (mortality on)."""
    from lib.world.growth_budget import growth_budget
    lib_ = library()
    return growth_budget(fg, lib_["species_definitions"],
                         lib_["interaction_definitions"], active_groups())


def capacity_rate(fg):
    from lib.world.growth_budget import capacity_loss_rate
    return capacity_loss_rate(species(fg))


#: Decimal mark and thousands separator for numbers read from the
#: library; the English build sets "." and ",".
NUM = {"dec": ",", "sep": " "}


def fmt(x, dec=None, sep=None, sig=3):
    """A library number for a slide: three significant digits, trailing
    zeros dropped only when the value is exact, thousands grouped, and
    values below 0.001 as mantissa e exponent."""
    dec = NUM["dec"] if dec is None else dec
    sep = NUM["sep"] if sep is None else sep
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, str):
        try:                       # PyYAML reads 1e-06 as a string
            x = float(x)
        except ValueError:
            return x
    x = float(x)
    if x == 0:
        return "0"
    ax_ = abs(x)
    if ax_ >= 1000:
        s = f"{x:,.0f}".replace(",", sep)
    elif ax_ < 1e-3:
        mant, exp = f"{x:.2e}".split("e")
        mant = mant.rstrip("0").rstrip(".")
        s = f"{mant}e{int(exp)}"
    else:
        d = max(0, sig - 1 - int(np.floor(np.log10(ax_))))
        s = f"{x:.{d}f}"
        if abs(x - round(x, d)) < 1e-12 and "." in s:
            s = s.rstrip("0").rstrip(".")
    return s.replace(".", dec) if dec != "." else s


def pair_attack(pred, prey):
    return pair(pred, prey).get("max_intake_rate",
                                species(pred).get("max_intake_rate"))


def response_type(pred, prey):
    """Holling type the engine uses: the pair's functional_response, or
    for a decision maker III as a specialist (exactly one active prey)
    and II otherwise. A grazing NDM is Holling II (with an optional
    food-perception threshold)."""
    forced = pair(pred, prey).get("functional_response")
    if forced:
        return int(forced), True
    if not species(pred).get("is_decision_maker"):
        return 2, False
    muted = muted_groups()
    n = sum(1 for (p, y) in PAIR_ORDER
            if p == pred and y not in muted and pair(p, y).get("preys_on"))
    return (3 if n == 1 else 2), False


def dm_energy_rows(dec=None, sep=None, annual="årsmedel"):
    f = lambda x: fmt(x, dec, sep)
    rows = []

    def row(label, get):
        rows.append([label] + [get(species(n)) for n in DM_ORDER])

    row("maintenance_level  u", lambda s: f(s.get("maintenance_level")))
    row("max_energy_reserve  ME", lambda s: f(s.get("max_energy_reserve")))
    row("energy_content  ec", lambda s: f(s.get("energy_content")))
    row("resting_metabolism", lambda s: f(s.get("resting_metabolism")))
    row("feeding / movement_cost",
        lambda s: f"{f(s.get('feeding_cost'))} / {f(s.get('movement_cost'))}")
    row("metabolism_q10 / t_ref",
        lambda s: "-" if not s.get("metabolism_q10") else
        f"{f(s['metabolism_q10'])} / " + (
            annual if s.get("metabolism_t_ref") == "annual_mean"
            else f"{f(s.get('metabolism_t_ref'))} °C"))
    row("ingestion / growth_q10",
        lambda s: "-" if not (s.get("ingestion_q10") or s.get("growth_q10"))
        else f"{f(s.get('ingestion_q10'))} / {f(s.get('growth_q10'))}")
    row("max_intake_rate  a", lambda s: f(s.get("max_intake_rate")))
    row("movement_speed  v", lambda s: f(s.get("movement_speed")))
    row("interference w / vis.floor",
        lambda s: f"{f(s.get('interference', 0))} / "
                  f"{f(s.get('visibility_floor', 0))}")
    row("min_split_biomass  (kg)", lambda s: f(s.get("min_split_biomass")))
    return rows


def dm_growth_rows(dec=None, sep=None, year="år", derived="härledd",
                   months="mån."):
    f = lambda x: fmt(x, dec, sep)
    rows = []

    def row(label, get):
        rows.append([label] + [get(n, species(n)) for n in DM_ORDER])

    row(f"r_max  (1/{year})", lambda n, s: f(s.get("r_max")))
    row("natural_mortality  M1", lambda n, s: f(s.get("natural_mortality")))
    row(f"growth_rate  ({derived})", lambda n, s: f(s.get("growth_rate")))
    row("starve_rate", lambda n, s: f(s.get("starve_rate")))
    row(f"income_growth_kappa / {months}",
        lambda n, s: "-" if not s.get("income_growth_kappa") else
        f"{f(s['income_growth_kappa'])} / {f(s.get('feeding_season_months'))}")
    row("regional_carrying_capacity K",
        lambda n, s: f(s.get("regional_carrying_capacity")))
    row(f"capacity_loss_rate c  ({derived})",
        lambda n, s: f(capacity_rate(n)))
    return rows


def budget_rows(dec=None, sep=None, words=None):
    """One row per group with a derived growth rate: r_max, M1, the
    checked M2 by predator, the gross budget, s*, g and Pmax, K."""
    words = words or {"none": "-", "full": "–"}
    f = lambda x: fmt(x, dec, sep)
    rows = []
    for fg in BUDGET_ORDER:
        b = budget(fg)
        if b is None or b.error:
            continue
        m2 = " + ".join(f"{f(v)}" for k, v in sorted(
            b.predation.items(), key=lambda kv: -kv[1])) or words["none"]
        s_star = f(b.reference_fill) if b.reference_fill is not None else \
            words["full"]
        season = "" if b.season_factor == 1.0 else \
            f" ×12/{f(12.0 / b.season_factor)}"
        k = species(fg).get("regional_carrying_capacity") or \
            species(fg).get("max_carrying_capacity")
        k_txt = f(k) + ("" if species(fg).get("regional_carrying_capacity")
                        else " /cell") if k else "-"
        rows.append([fg, f(b.r_max), f(b.m1_annual), m2, f(b.gross_annual),
                     s_star, f(b.growth_rate) + season,
                     f(b.pmax_annual) if b.pmax_annual else "-", k_txt])
    return rows


def pair_value_rows(dec=None, sep=None, prey_word="byte", type_word="typ",
                    forced_word="satt", graze_word="betning"):
    f = lambda x: fmt(x, dec, sep)
    muted = muted_groups()
    rows = []
    for pred, prey in PAIR_ORDER:
        p = pair(pred, prey)
        if not p.get("preys_on"):
            continue
        prey_dm = species(prey).get("is_decision_maker")
        if "visibility_floor" in p:
            floor = f(p["visibility_floor"])
        elif prey_dm:
            floor = f"{prey_word} {f(species(prey).get('visibility_floor', 0))}"
        else:
            floor = "-"
        kind, forced = response_type(pred, prey)
        resp = f"{type_word} {'III' if kind == 3 else 'II'}" + (
            f" ({forced_word})" if forced else "")
        if not species(pred).get("is_decision_maker"):
            resp += f", {graze_word}"
        extra = []
        if p.get("dark_ratio") is not None and p["dark_ratio"] < 1:
            extra.append(f"ρ {f(p['dark_ratio'])} / "
                         f"{f(p.get('light_threshold_deg', -6))}°")
        if p.get("mixed_layer_reference_m"):
            extra.append(f"M_ref {f(p['mixed_layer_reference_m'])} m")
        if p.get("food_threshold"):
            extra.append(f"F0 {f(p['food_threshold'])}")
        if p.get("edibility_floor") is not None:
            extra.append(f"e {f(p['edibility_floor'])}/"
                         f"{f(p.get('edibility_half_saturation_umol_l'))}")
        name = f"{pred} → {prey}" + (" *" if pred in muted else "")
        rows.append([name, f(pair_attack(pred, prey)),
                     f(p.get("handling_time")),
                     f(p.get("assimilation_factor")) if
                     "assimilation_factor" in p else "-",
                     floor, resp, ", ".join(extra) or "-",
                     f(p.get("predation_mortality"))])
    return rows


def observability_rows(yes="ja"):
    muted = muted_groups()
    rows = []
    for obs in DM_ORDER:
        name = obs + (" *" if obs in muted else "")
        rows.append([name] + [
            yes if (pair(obs, tgt, "observes").get("observes")) else "-"
            for tgt in OBS_ORDER])
    return rows


def observation_width(fg):
    """(k, in_dim) for a DM in mareld2.yaml: observed other active FGs
    and the compact observation width with the light channel."""
    active = set(active_groups())
    k = sum(1 for tgt in OBS_ORDER if tgt != fg and tgt in active
            and pair(fg, tgt, "observes").get("observes"))
    return k, (2 + k) + 4 * (1 + k) + 1


def _decimal_ticks(ax_):
    """Tick labels with the deck's decimal mark (comma in Swedish)."""
    from matplotlib.ticker import ScalarFormatter

    class DecimalFormatter(ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace(".", NUM["dec"])
    ax_.xaxis.set_major_formatter(DecimalFormatter())
    ax_.yaxis.set_major_formatter(DecimalFormatter())


def _style(*axes):
    for a_ in axes:
        a_.tick_params(labelsize=8.5)
        _decimal_ticks(a_)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.grid(alpha=0.25, lw=0.6)


def draw_food_web(fig, ax, y):
    names = LABELS["web_nodes"]
    nodes = {
        "phytoplankton": (150, 92),
        "subsurface_phytoplankton": (345, 92),
        "benthic_infauna": (775, 92),
        "microzooplankton": (150, 182),
        "zooplankton": (345, 182),
        "pelagic_fish": (345, 272),
        "gadoids": (650, 222),
        "seabirds": (150, 362),
        "seals": (560, 362),
        "porpoises": (790, 312),
    }
    muted = muted_groups()
    bw, bh = 150, 38
    for pred, prey in PAIR_ORDER:
        if pred == prey or not pair(pred, prey).get("preys_on"):
            continue
        xa, ya = nodes[prey]
        xb, yb = nodes[pred]
        x0, y0 = _rect_edge(xa, ya, bw, bh, xb, yb)
        x1, y1 = _rect_edge(xb, yb, bw, bh, xa, ya)
        arrow(ax, x0, y0, x1, y1, color="#8FA9BC", lw=1.3, head=6.5,
              ls="--" if pred in muted else "-")
    gx, gy = nodes["gadoids"]
    if pair("gadoids", "gadoids"):
        ax.annotate("", xy=(gx + bw / 2 + 2, gy - 8),
                    xytext=(gx + bw / 2 + 2, gy + 8),
                    arrowprops=dict(arrowstyle="-|>", color="#C3D0DA", lw=1.2,
                                    linestyle=":" if not pair(
                                        "gadoids", "gadoids").get("preys_on")
                                    else "-",
                                    connectionstyle="arc3,rad=-1.6",
                                    shrinkA=0, shrinkB=0,
                                    mutation_scale=10), zorder=2)
        ax.text(gx + bw / 2 + 30, gy, LABELS["web_self"], color=MUTED,
                size=9.5, va="center", ha="left")
    for key, (x, yy) in nodes.items():
        ndm = not species(key).get("is_decision_maker")
        grazer = key == "microzooplankton"
        face = "#E4EEF3" if not ndm else ("#EEF5E2" if grazer else "#E9F2E4")
        edge = ACCENT if not ndm else ("#7FA35A" if grazer else "#5C8A3C")
        box(ax, x - bw / 2, yy - bh / 2, bw, bh, names[key], face=face,
            edge=edge, size=11.5, weight="bold")
        if key in muted:
            ax.patches[-1].set_linestyle("--")
            ax.patches[-1].set_facecolor("#F4F7F9")
    ax.text(560, 140, LABELS["web_legend"], color=MUTED, size=10.5,
            va="center", ha="center")


def draw_functional_response(fig, ax, y):
    # Left: the gadoid ceilings as an energy limit (section 148).
    a_g = pair_attack("gadoids", "pelagic_fish")
    h_herr = pair("gadoids", "pelagic_fish")["handling_time"]
    h_ben = pair("gadoids", "benthic_infauna")["handling_time"]
    herr = species("pelagic_fish")
    e_herr = herr["energy_content"] + herr.get("reserve_reference_fill", 0.5) \
        * herr["max_energy_reserve"]
    e_ben = species("benthic_infauna")["energy_content"]
    w = species("gadoids").get("interference", 0.0)
    B = np.linspace(0, 3.0, 400)
    f_herr = e_herr * a_g * B ** 2 / (1 + a_g * h_herr * B ** 2)
    f_herr_w = e_herr * a_g * B ** 2 / (1 + a_g * h_herr * B ** 2 + w * 0.5)
    f_ben = e_ben * a_g * B / (1 + a_g * h_ben * B)
    ax1 = axes_pt(fig, 92, 200, 360, 165)
    ax1.plot(B, f_herr, color=ACCENT, lw=2.2,
             label=LABELS["fr_herr"].format(h=fmt(h_herr)))
    ax1.plot(B, f_herr_w, color=ACCENT, lw=1.6, ls="--",
             label=LABELS["fr_herr_w"])
    ax1.plot(B, f_ben, color=ACCENT_2, lw=2.2,
             label=LABELS["fr_benth"].format(h=fmt(h_ben)))
    ceil_e = e_herr / h_herr
    ax1.axhline(ceil_e, color=MUTED, lw=1.0, ls=":")
    ax1.text(2.98, ceil_e + 0.6, LABELS["fr_ceiling_e"], color=MUTED,
             size=9.5, ha="right")
    ax1.set_ylim(0, ceil_e * 1.25)
    ax1.set_title(LABELS["fr_title_gad"].format(a=fmt(a_g)), size=11,
                  color=NAVY)
    ax1.legend(fontsize=8.0, loc="lower right", frameon=False)
    ax1.set_ylabel(LABELS["fr_ylabel_e"], size=9)

    # Right: zooplankton grazing on concentration (section 177): a is
    # scaled by M_ref / M(t), so the half-saturation in t/km2 grows with
    # the mixed layer; the microzooplankton pair is Type III (178).
    p_ph = pair("zooplankton", "phytoplankton")
    p_mi = pair("zooplankton", "microzooplankton")
    a_ph, h_ph = p_ph["max_intake_rate"], p_ph["handling_time"]
    a_mi, h_mi = p_mi["max_intake_rate"], p_mi["handling_time"]
    m_ref = p_ph.get("mixed_layer_reference_m", 10.0)
    Bz = np.linspace(0, 30, 400)
    ax2 = axes_pt(fig, 520, 200, 360, 165)
    for depth, key, ls in ((6.0, "fr_zoo_m6", "-"), (17.0, "fr_zoo_m17", "--")):
        a_ = a_ph * m_ref / depth
        ax2.plot(Bz, a_ * Bz / (1 + a_ * h_ph * Bz), color=ACCENT, lw=2.0,
                 ls=ls, label=LABELS[key])
    a3 = a_mi * p_mi.get("mixed_layer_reference_m", 10.0) / 10.0
    ax2.plot(Bz, a3 * Bz ** 2 / (1 + a3 * h_mi * Bz ** 2), color=ACCENT_2,
             lw=2.0, label=LABELS["fr_zoo_micro"])
    ax2.axhline(1 / h_ph, color=MUTED, lw=1.0, ls=":")
    ax2.text(29.5, 1 / h_ph + 0.006, LABELS["fr_ceiling"], color=MUTED,
             size=9.5, ha="right")
    ax2.set_ylim(0, 1 / h_ph * 1.2)
    ax2.set_title(LABELS["fr_title_zoo"], size=11, color=NAVY)
    ax2.legend(fontsize=8.0, loc="lower right", frameon=False)
    ax2.set_ylabel(LABELS["fr_ylabel"], size=9)
    for a_ in (ax1, ax2):
        a_.set_xlabel(LABELS["fr_xlabel"], size=9.5)
    _style(ax1, ax2)


def draw_hunger_gate(fig, ax, y):
    s = np.linspace(0, 1, 300)
    ax1 = axes_pt(fig, 92, 178, 360, 190)
    for u, label, color in ((0.5, LABELS["hg_u5"], ACCENT),
                            (0.2, LABELS["hg_u2"], ACCENT_2)):
        ax1.plot(s, np.clip((1 - s) / (1 - u), 0.0, 1.0), color=color,
                 lw=2.2, label=label)
    ax1.set_ylim(0, 1.12)
    ax1.set_xlabel(LABELS["hg_xlabel1"], size=9.5)
    ax1.set_ylabel(LABELS["hg_ylabel1"], size=9.5)
    ax1.legend(fontsize=8.5, frameon=False, loc="lower left")

    ax2 = axes_pt(fig, 520, 178, 360, 190)
    for fg, k, label, color in (("zooplankton", 1.0, LABELS["hg_zoo"], ACCENT),
                                ("gadoids", 10.0, LABELS["hg_gad"], ACCENT_2)):
        sp_ = species(fg)
        u, g, sr = sp_["maintenance_level"], sp_["growth_rate"], \
            sp_["starve_rate"]
        q = s - u
        ax2.plot(s, k * np.where(q >= 0, g * q, sr * q), color=color, lw=2.2,
                 label=label.format(g=fmt(g), sr=fmt(sr)))
        ax2.axvline(u, color=color, lw=0.8, ls=":")
        b = budget(fg)
        if b is not None and b.reference_fill:
            ss = b.reference_fill
            ax2.plot([ss], [k * g * (ss - u)], "o", color=color, ms=5)
            ax2.text(ss, k * g * (ss - u) + 0.006, LABELS["hg_sstar"],
                     color=color, size=9, ha="center")
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_ylim(-0.11, 0.07)
    ax2.set_xlabel(LABELS["hg_xlabel2"], size=9.5)
    ax2.set_ylabel(LABELS["hg_ylabel2"], size=9.5)
    ax2.legend(fontsize=8.5, frameon=False, loc="lower right")
    _style(ax1, ax2)


def draw_ideal_reserve(fig, ax, y):
    """Section 160: the reserve a DM settles on at the Holling ceiling,
    where the income h(s) C meets cost + P (ec + s ME)."""
    for i, fg in enumerate(("zooplankton", "gadoids")):
        b = budget(fg)
        sp_ = species(fg)
        u, ec, me = b.maintenance, sp_["energy_content"], \
            sp_["max_energy_reserve"]
        from lib.world.growth_budget import TICKS_PER_YEAR
        P = b.gross_annual / TICKS_PER_YEAR
        s = np.linspace(u, 1.0, 300)
        income = b.ideal_intake * np.clip((1 - s) / (1 - u), 0, 1)
        need = b.ideal_cost + P * (ec + s * me)
        a_ = axes_pt(fig, 92 + i * 428, 200, 360, 165)
        a_.plot(s, income, color=ACCENT, lw=2.2, label=LABELS["ir_income"])
        a_.plot(s, need, color=ACCENT_2, lw=2.2, label=LABELS["ir_need"])
        ss = b.reference_fill
        a_.axvline(ss, color=MUTED, lw=0.9, ls=":")
        a_.plot([ss], [b.ideal_cost + P * (ec + ss * me)], "o", color=NAVY,
                ms=5, zorder=4)
        a_.set_title(LABELS["ir_title"].format(
            fg=LABELS["ir_fg"][fg], s=fmt(ss)), size=11, color=NAVY)
        a_.set_xlabel(LABELS["ir_xlabel"], size=9.5)
        a_.set_ylabel(LABELS["ir_ylabel"], size=9.5)
        a_.set_ylim(0, b.ideal_intake * 1.08)
        a_.legend(fontsize=8.5, frameon=False, loc="upper right")
        _style(a_)


def _fishing_f(fg, impact):
    """Annual F per unit of effort from the library's linear table."""
    rows = pair(fg, impact, "impacted_by").get("impact_table") or []
    if len(rows) < 2:
        return 0.0
    top = rows[-1]
    return float(top["biomass_factor"]) * 1460.0 / float(top["value"])


def map_mean(name, key):
    path = os.path.join(REPO, "maps", name)
    try:
        return float(np.load(path)[key].mean())
    except (OSError, KeyError):
        return None


def regional_f(fg):
    """Today's grid-mean F of ``fg`` from the two trawl maps."""
    out = 0.0
    for impact, name in (("bottom_trawling", "fishing_bottom_otter_trawls.npz"),
                         ("pelagic_trawling", "fishing_pelagic.npz")):
        m = map_mean(name, impact)
        if m is not None:
            out += _fishing_f(fg, impact) * m
    return round(out, 4)


def draw_capacity(fig, ax, y):
    """Section 185: net growth of a pool with and without its regional
    carrying capacity, at today's fishing mortality."""
    start = {e["group_id"]: e.get("inference_initial_biomass")
             for k in ("decision_makers",) for e in manifest()[k]}
    n_cells = (manifest()["project_metadata"]["reference_grid_width"]
               * manifest()["project_metadata"]["reference_grid_height"])
    for i, fg in enumerate(("gadoids", "pelagic_fish")):
        sp_ = species(fg)
        r, K = sp_["r_max"], sp_["regional_carrying_capacity"]
        F = regional_f(fg)
        B = np.linspace(0, K * 1.15, 300)
        a_ = axes_pt(fig, 92 + i * 428, 200, 360, 165)
        a_.plot(B, r * (1 - B / K) - F, color=ACCENT, lw=2.2,
                label=LABELS["cap_withk"])
        a_.plot(B, np.full_like(B, r - F), color=ACCENT_2, lw=2.0, ls="--",
                label=LABELS["cap_nok"])
        a_.axhline(0, color=MUTED, lw=0.8)
        today = start[fg] / n_cells
        eq = K * (1 - F / r)
        a_.axvline(today, color=NAVY, lw=0.9, ls=":")
        a_.text(today, r * 0.92, " " + LABELS["cap_today"], color=NAVY,
                size=9, ha="left" if i == 0 else "right")
        a_.plot([eq], [0], "o", color=NAVY, ms=5, zorder=4)
        a_.text(eq, -0.06, LABELS["cap_eq"], color=NAVY, size=9,
                ha="center")
        a_.set_title(LABELS["cap_title"].format(
            fg=LABELS["cap_fg"][fg], r=fmt(r), K=fmt(K), F=fmt(F, sig=2)),
            size=11, color=NAVY)
        a_.set_xlabel(LABELS["cap_xlabel"], size=9.5)
        a_.set_ylabel(LABELS["cap_ylabel"], size=9.5)
        a_.set_ylim(-0.25, r * 1.05)
        a_.legend(fontsize=8.5, frameon=False, loc="lower left")
        _style(a_)


def draw_fishing_maps(fig, ax, y):
    for i, (name, key, title) in enumerate((
            ("fishing_bottom_otter_trawls.npz", "bottom_trawling",
             "fm_title_b"),
            ("fishing_pelagic.npz", "pelagic_trawling", "fm_title_p"))):
        path = os.path.join(REPO, "maps", name)
        a_ = axes_pt(fig, 150 + i * 360, 178, 190, 190)
        try:
            field = np.load(path)[key]
        except (OSError, KeyError):
            field = np.zeros((60, 60))
        im = a_.imshow(field, cmap="YlOrBr", interpolation="nearest")
        a_.set_xticks([]); a_.set_yticks([])
        a_.set_title(LABELS[title].format(m=fmt(float(field.mean()))),
                     size=10.5, color=NAVY)
        cb = fig.colorbar(im, ax=a_, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
        _decimal_ticks(cb.ax)


def draw_water_column(fig, ax, y):
    L = LABELS
    x0, w = LEFT + 40, 560
    # light
    ax.text(x0 + 10, y - 6, L["wc_light"], color=ACCENT_2, size=11.5,
            weight="bold", va="center", ha="left")
    arrow(ax, x0 + 250, y - 6, x0 + 250, y - 30, color=ACCENT_2, lw=1.6)
    boxes = [
        (y - 112, 80, L["wc_mixed"], L["wc_mixed_items"], "#E4EEF3"),
        (y - 222, 80, L["wc_sub"], L["wc_sub_items"], "#E9F2E4"),
        (y - 300, 46, L["wc_res"], "", BAND),
    ]
    for top, h, head, items, face in boxes:
        ax.add_patch(FancyBboxPatch((x0, top), w, h,
                                    boxstyle="round,pad=0,rounding_size=5",
                                    facecolor=face, edgecolor="#9FB6C6",
                                    lw=1.1, zorder=1))
        ax.text(x0 + 16, top + h - 18, head, color=NAVY, size=12.5,
                weight="bold", va="center", ha="left", zorder=3)
        if items:
            ax.text(x0 + 16, top + h - 46, items, color=INK, size=11.5,
                    va="center", ha="left", zorder=3)
    xm = x0 + w * 0.42
    for (ya, yb, label) in ((y - 112, y - 142, L["wc_e"]),
                            (y - 222, y - 254, L["wc_eb"])):
        arrow(ax, xm, ya + 2, xm, yb - 2, color=ACCENT, lw=1.4)
        arrow(ax, xm + 14, yb - 2, xm + 14, ya + 2, color=ACCENT, lw=1.4)
        lines = label.split("\n")
        for k, line in enumerate(lines):
            ax.text(xm + 30, (ya + yb) / 2 + (len(lines) - 1 - 2 * k) * 7,
                    line, color=INK, size=10, va="center", ha="left")
    xr = x0 + w + 24
    for k, (label, yy, color) in enumerate((
            (L["wc_ent"], y - 120, ACCENT),
            (L["wc_regen"], y - 46, "#5C8A3C"),
            (L["wc_export"], y - 200, MUTED))):
        lines = label.split("\n")
        for j, line in enumerate(lines):
            ax.text(xr, yy - j * 14, line, color=color, size=10.5,
                    va="center", ha="left")


def draw_observation(fig, ax, y):
    cx, cy, c = 300, 250, 74
    cells = {"C": (cx, cy), "N": (cx, cy + c + 6), "S": (cx, cy - c - 6),
             "E": (cx + c + 6, cy), "W": (cx - c - 6, cy)}
    for key, (x, yy) in cells.items():
        centre = key == "C"
        box(ax, x - c / 2, yy - c / 2, c, c,
            LABELS["obs_centre"] if centre else key,
            face="#E4EEF3" if centre else BAND,
            edge=ACCENT if centre else "#9FB6C6", size=12,
            weight="bold" if centre else "normal")
    ax.text(cx, cy - 2 * c - 6, LABELS["obs_shape"],
            color=MUTED, size=11.5, ha="center", va="center")
    xb = 468
    draw_code(ax, LABELS["obs_caption"], LABELS["obs_code"],
              xb, y - 6, PAGE_W - RIGHT - xb, size=12.5)
    draw_bullets(ax, LABELS["obs_bullets"], xb, y - 128,
                 PAGE_W - RIGHT - xb, size=13)


def draw_policy_net(fig, ax, y):
    x0, x1, x3 = 190, 470, 750

    def column(x, n, label, color, size=7.0):
        ys = np.linspace(200, 350, n)
        for yy in ys:
            ax.add_patch(plt.Circle((x, yy), size, facecolor=color,
                                    edgecolor="none", zorder=3))
        ax.text(x, 176, label, color=NAVY, size=12, weight="bold",
                ha="center", va="center")
        return ys
    a = column(x0, 7, LABELS["net_in"], ACCENT)
    b = column(x1, 8, LABELS["net_h1"], "#7FB2C0")
    d = column(x3, 5, LABELS["net_out"], ACCENT_2)
    for u in a:
        for v in b:
            ax.plot([x0, x1], [u, v], color="#C9D8E2", lw=0.4, zorder=1)
    for u in b:
        for v in d:
            ax.plot([x1, x3], [u, v], color="#C9D8E2", lw=0.4, zorder=1)
    labels = LABELS["net_actions"]
    for yy, lab in zip(d[::-1], labels):
        if lab:
            ax.text(x3 + 16, yy, lab, color=INK, size=11.5, va="center",
                    ha="left")


def draw_energy_flow(fig, ax, y):
    yb = y - 128
    w, h = 176, 62
    names = LABELS["ef_stages"]
    stages = [
        (LEFT, names[0], BAND),
        (LEFT + 210, names[1], "#E4EEF3"),
        (LEFT + 420, names[2], "#E4EEF3"),
        (LEFT + 630, names[3], "#E9F2E4"),
    ]
    for x, text, face in stages:
        box(ax, x, yb, w, h, text, face=face, size=12, weight="bold")
    for i in range(3):
        arrow(ax, stages[i][0] + w, yb + h / 2, stages[i + 1][0],
              yb + h / 2, lw=1.6)
    for x, label in ((LEFT + 210, LABELS["ef_spill"]),
                     (LEFT + 420, LABELS["ef_costs"])):
        arrow(ax, x + w / 2, yb, x + w / 2, yb - 44, color=MUTED, lw=1.2)
        ax.text(x + w / 2, yb - 56, label.replace("\n", ", "), color=MUTED,
                size=11, ha="center", va="center")
    ax.text(LEFT + 630 + w / 2, yb + h + 26,
            LABELS["ef_price"],
            color=ACCENT_2, size=12, weight="bold", ha="center", va="center")
    arrow(ax, LEFT + 630 + w / 2, yb + h + 16, LEFT + 630 + w / 2, yb + h + 2,
          color=ACCENT_2, lw=1.4)


def draw_spawn_examples(fig, ax, y):
    from lib.spawn.strategies import StrategySpec, make_weights
    grid = (60, 60)
    perlin = make_weights(StrategySpec.from_dict(
        {"mode": "perlin", "scale": 12.0, "octaves": 4, "persistence": 0.5,
         "lacunarity": 2.0, "min_frac_of_max": 0.1}), grid, 7)
    colony = make_weights(StrategySpec.from_dict(
        {"mode": "colony", "n_colonies": 25, "sigma_cells": 1.0,
         "anchor": "free", "amplitude_mode": "jitter", "amplitude_min": 0.0,
         "amplitude_max": 1.0}), grid, 7)
    env = make_weights(StrategySpec.from_dict(
        {"mode": "env_driven", "floor": 0.0, "noise_amp": 0.0,
         "min_frac_of_max": 0.1,
         "refs": [{"name": "phytoplankton", "weight": 1.0,
                   "transform": "linear"}]}), grid, 7,
        context={"env_fields": {"phytoplankton": perlin}})
    uniform = make_weights(StrategySpec.from_dict({"mode": "uniform"}),
                           grid, 7)
    panels = list(zip(LABELS["spawn_panels"],
                      (uniform, perlin, colony, env)))
    for i, (label, field) in enumerate(panels):
        a = axes_pt(fig, 96 + i * 196, 190, 160, 160)
        a.imshow(field, cmap="YlGnBu", interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(label, size=10.5, color=NAVY)


# ----------------------------------------------------------------------
# The deck
# ----------------------------------------------------------------------
def build(d):
    f = fmt
    phy, sub = species("phytoplankton"), species("subsurface_phytoplankton")
    ben, mic = species("benthic_infauna"), species("microzooplankton")
    zoo, pel = species("zooplankton"), species("pelagic_fish")
    gad = species("gadoids")
    b_zoo, b_pel, b_gad = budget("zooplankton"), budget("pelagic_fish"), \
        budget("gadoids")
    b_phy = budget("phytoplankton")
    f_gad, f_pel = regional_f("gadoids"), regional_f("pelagic_fish")
    k_gad, k_pel = gad["regional_carrying_capacity"], \
        pel["regional_carrying_capacity"]
    eq_gad = k_gad * (1 - f_gad / gad["r_max"])
    eq_pel = k_pel * (1 - f_pel / pel["r_max"])
    # ---------------------------------------------------------------- 0
    d.title_slide(
        "Strategi",
        "För att bygga ekosystemsimulatorer",
        ["Utgåva 2 - oktober 2026",
         "Mareld / Poseidon Nord",
         "Beskriver den mekanik som simulatorn kör efter avsnitt 185"])

    d.bullets(
        "Om den här utgåvan",
        [("p", "Den här versionen ersätter utgåvan från 12 oktober "
               "(Strategi_okt.pdf). Sedan dess har modellen fått två nya "
               "funktionsgrupper, en vattenpelare med två lager och en "
               "näringspool, dagens fiske, aktiva sälar och ett regionalt "
               "bärkraftstak för fisken."),
         "",
         "Koden är modellen. Dokumentet beskriver den - inte tvärtom. "
         "Varje formel och siffra är hämtad ur tickpipelinen, ur "
         "fg_library.yaml, ur mareld2.yaml eller ur "
         "lib/world/growth_budget.py; tabellerna läses ur filerna när "
         "dokumentet byggs.",
         "Nytt är Del 4, som i detalj förklarar den biologiska motiveringen "
         "för varje grupps tillväxttakt och för det logistiska taket: "
         "varifrån r_max kommer, vilka förluster tillväxten ska betala, "
         "varför budgeten betalas vid s* och varför torskfisk och pelagisk "
         "fisk har ett regionalt K.",
         "Del 9 väger modellens styrkor mot dess svagheter och listar "
         "förbättringsförslag i prioritetsordning.",
         ("-", "Strategin är ett utvecklingsdokument, inte en "
               "kravspecifikation. Att koden stämmer med dokumentet är "
               "inget bevis för att en term är ekologiskt rätt."),
         ],
        kicker="Läsanvisning",
        footnote="Genereras av tools/make_strategi_deck2.py; den engelska "
                 "utgåvan av tools/make_strategy_deck2_en.py. Avsnittsnummer "
                 "hänvisar till mareld_resume.txt.")

    d.table(
        "Vad som har ändrats sedan 12 oktober",
        ["Område", "12 oktober", "I dag", "Avsnitt"],
        [["Grupper", "8 FG, sälar muted",
          "10 FG: mikrozooplankton och djupväxtplankton nya; sälar aktiva",
          "166, 177, 180"],
         ["Växtplankton", "Ljusbegränsad logistik mot K = 24 t/km²",
          "Näringspool, sjunkning som M1, två lager, betas av "
          "mikrozooplankton", "172-182"],
         ["Tillväxtbudget", "Betald vid full reserv, s = 1",
          "Betald vid s*, reserven under ideala förhållanden; kannibalism "
          "urkryssad", "160"],
         ["Logistiskt tak", "Bara för icke beslutande grupper",
          "Regionalt K för torskfisk och pelagisk fisk; r_max = dagens "
          "produktivitet", "184-185"],
         ["Sill", "Tillväxt ur fettreserven året runt",
          "Tillväxt ur inkomsten (κ 0,75, 5,5 månader); w 0,05; typ II på "
          "djurplankton", "162, 184"],
         ["Djurplankton", "Q10 bara på kostnaderna",
          "Q10 på intag (2,8) och tillväxt (3,0); betar i koncentration",
          "177-183"],
         ["Fiske", "Alla impakter muted",
          "Bottentrål och pelagisk trål ur EMODnet-kartor, F 0,12 / 0,10; "
          "tumlarbifångst", "161, 165"],
         ["Delning", "Kontinuerlig från en enhet",
          "Bara hela enheter flyttar (regel D)", "159"]],
        [12, 23, 52, 11], kicker="Översikt", size=10.8,
        footnote="Alla policyer som tränats före ändringarna måste tränas om "
                 "innan siffror citeras; FG-uppsättningen ändrades i "
                 "avsnitt 177 och 180, vilket gör äldre checkpoints "
                 "ogiltiga.")

    # ---------------------------------------------------------------- 1
    d.section("1", "Strategi och beståndsdelar",
              "Vad modellen är, vad den ska användas till och vilka delar "
              "den består av.")

    d.bullets(
        "Strategi",
        ["Ekosystem modelleras som en karta med flera kartlager som "
         "utvecklas över tid.",
         "Kartlagren beskriver miljöförhållanden, utbredning av "
         "populationer tillhörande olika funktionsgrupper, näringsämnen och "
         "olika sorters miljöpåverkan. En kalender för ljus, temperatur och "
         "skiktning driver säsongen.",
         "Populationernas beteende - särskilt rörelse och födosök - styrs "
         "av beteendemodeller som tränas med ARS, men kan också vara "
         "handkodade (viabilitetsriggen kör helt utan nätverk).",
         "Biologin driver, inte action-hackar. Parametrar och mekanismer "
         "justeras först; formande termer i belöningen är sista utvägen "
         "och är avstängda som standard.",
         "Modellen är Eulerisk: vi följer värden i fasta celler, inte "
         "individer. Biomassan kan delas, flytta åt olika håll och blandas "
         "med annan biomassa i samma cell."],
        kicker="Grundidéer")

    d.bullets(
        "Målsättning",
        ["Strategin används för att bygga simuleringsmodeller som fungerar "
         "som scenario- och hypotesverktyg.",
         "Modellerna ger inte absoluta populationsprognoser.",
         "Resultaten ska tolkas relativt, jämförande och under antaganden.",
         "",
         ("p", "Innan ett scenario jämförs ska världen vara självbärande "
               "med de policyer som faktiskt används - ingen "
               "funktionsgrupp får dö ut eller glida mot noll under "
               "horisonten. Startbiomassorna beskriver området som det är "
               "i dag, under dagens fiske; ett nollscenario ska därför "
               "ligga nära startläget."),
         ("p", "Viabilitetsriggen, som kör med fryst beteende, är ett "
               "diagnosverktyg för mekaniken, inte domaren."),
         ],
        kicker="Vad modellen svarar på")

    d.bullets(
        "Beståndsdelar",
        ["En ekosystemmodell består av:",
         ("-", "en grid med längdskala och ticklängd"),
         ("-", "en uppsättning funktionsgrupper (FG), beslutande och icke "
               "beslutande"),
         ("-", "en uppsättning kartor - dagens fiske som baskartor, ett "
               "scenario som ersättande kartor"),
         ("-", "en kalender: dagsljus, vattentemperatur och blandningsdjup "
               "för platsen"),
         ("-", "en vattenpelare: blandat lager, djuplager och en "
               "näringsreservoar"),
         ("-", "egenskaper för varje funktionsgrupp och varje "
               "predationspar (fg_library.yaml)"),
         ("-", "en dynamisk modell med uppdateringsregler - ticken"),
         ("-", "en beteendemodell för varje beslutande funktionsgrupp"),
         ("-", "spawnstrategier som bestämmer hur startbiomassan läggs ut"),
         ("-", "diagnostik: energibudgetgrind, fryst viabilitetsrigg och "
               "probeverktyg för års- och månadsbudgetar"),
         "Biblioteket är gemensamt för alla projekt. Projektfilen "
         "(mareld2.yaml) väljer FG, startbiomassor, impaktkartor och "
         "platsens ljus-, temperatur- och näringsklimat."],
        kicker="Modellens delar", size=15)

    d.bullets(
        "Grid",
        ["Varje ekosystemmodell har en grid, dvs en matris av celler med:",
         ("-", "en dimension - Mareld kör 60 x 60 celler"),
         ("-", "en längdskala - en cellsida = 1 km, så ton/cell = t/km²"),
         ("-", "en tidskala - ett tick = 1-6 timmar, normalt 6"),
         ("-", "en position - 58,15° N, som sätter dagsljuset"),
         "Griden är plan och fyrgrannskopplad: all rörelse sker till N, E, "
         "S eller W, aldrig diagonalt.",
         "En valfri karta accessibility styr vilka celler som är "
         "tillgängliga. Rörelse in i en otillgänglig cell maskas bort, och "
         "biomassa som ändå hamnar där nollställs i slutet av ticket.",
         "Utan migration är gridens kant sluten. Med --migration on lämnar "
         "biomassan griden och återinförs koncentrerat längs kanten.",
         ("p", "Griden täcker 3 600 km² av Skagerrak. Bestånden den "
               "rymmer är andelar av mycket större bestånd - en skillnad i "
               "skala som återkommer i Del 4 och Del 10.")],
        kicker="Geografi")

    d.table(
        "Ticklängd är en körningsparameter",
        ["Parametertyp", "Regel vid faktor k = ny/gammal", "Exempel"],
        [["flux - mängd som överförs per tick", "x · k",
          "max_intake_rate (även per par), resting_metabolism, "
          "movement_speed, seed_rate, capacity_loss_rate"],
         ["growth - andel som LÄGGS TILL ett lager", "(1 + x)ᵏ − 1",
          "growth_rate"],
         ["loss - andel som TAS BORT ur ett lager", "1 − (1 − x)ᵏ",
          "natural_mortality, starve_rate, biomass_factor i impacttabeller"],
         ["inverse - storhet vars invers är takten", "x / k",
          "handling_time (Hollingtaket 1/h ska hållas fast i realtid)"],
         ["tick-oberoende", "oförändrad",
          "max_energy_reserve, energy_content, maintenance_level, r_max, "
          "regional_carrying_capacity, Q10-värden, näringsnycklar"]],
        [22, 24, 40], kicker="--tick-length", size=11.5,
        lead="Motorn är ticklängdsagnostisk: varje bibliotekstal är uttryckt "
             "PER TICK vid 6 h. Ticklängden räknar om per-tick-takter till "
             "realtid och skalar, när den inte är 6 h, om varje tickberoende "
             "parameter i minnet när FG:erna byggs. Ljus-, temperatur-, "
             "närings- och lagertabellerna byggs direkt för körningens "
             "ticklängd.",
        footnote="fg_library.yaml har EN kalibrering (6 h) och skrivs aldrig "
                 "om av en körning. lib/world/tick_time.py är enda "
                 "definitionspunkten.")

    d.table(
        "Funktionsgrupper",
        ["Funktionsgrupp", "Typ", "Meny", "Roll i modellen"],
        [["phytoplankton", "icke beslutande", "-",
          "Primärproducent i blandade lagret; ljus- och näringsbegränsad, "
          "sjunker (M1), driver med strömmen"],
         ["subsurface_phytoplankton", "icke beslutande", "-",
          "Samma alger i lagret M(t)-30 m: djupklorofyllmaximum, "
          "skuggadapterad"],
         ["microzooplankton", "icke beslutande, betar",
          "phyto, subsurface",
          "Ciliater och heterotrofa dinoflagellater; växer bara genom "
          "betning"],
         ["benthic_infauna", "icke beslutande", "-",
          "Bottenföda för torskfisk; logistisk, utarmas av bottentrål"],
         ["zooplankton", "beslutande", "phyto, subsurface, micro",
          "Copepoder; betar i koncentration, Q10 på intag och tillväxt"],
         ["pelagic_fish", "beslutande", "zooplankton",
          "Sill, makrill, skarpsill; fettreserv, tillväxt ur inkomsten, "
          "regionalt K"],
         ["gadoids", "beslutande", "pelagic, benthic, zooplankton",
          "Torsk, kolja, vitling, gråsej, kummel; energitak, regionalt K"],
         ["porpoises", "beslutande", "pelagic, gadoids",
          "Tumlare; endoterm, biosonar, bifångst på trålkartan"],
         ["seals", "beslutande", "pelagic, gadoids",
          "Knubbsäl; ~95 % torskfisk, jagar med morrhåren"],
         ["seabirds", "beslutande", "pelagic_fish",
          "Muted; enda FG med rotormortalitet"]],
        [21, 15, 21, 43], kicker="Mareld: tio grupper, sex med policy",
        size=10.5,
        footnote="Engelska snake_case-id:n är kanoniska. Bara seabirds är "
                 "muted i mareld2.yaml.")

    d.figure("Näringsväv", draw_food_web, kicker="Meny enligt biblioteket",
             footnote="Pilen går från byte till konsument och ritas ur "
                      "biblioteket. Varje par har egna värden för "
                      "handling_time, assimilation, predation_mortality och "
                      "- för vissa par - attackhastighet, synlighetsgolv, "
                      "ljusberoende och koncentrationsskala.")

    # ---------------------------------------------------------------- 2
    d.section("2", "Kartor, kalender och vattenpelare",
              "Allt tillstånd är kartor över griden. Ljus, temperatur och "
              "skiktning kommer från en kalender för platsen.")

    d.bullets(
        "Kartor",
        ["En karta är en funktion från griden till de reella talen.",
         "Modellen håller två tillståndskartor per funktionsgrupp: "
         "biomassa B_X(c) i ton och en reservkarta R_X(c).",
         "För en beslutande FG är R_X reservenergi i MJ. Energikartan "
         "E_X = R_X / B_X är härledd, vilket gör blandning vid rörelse "
         "trivial: två inflöden adderar sina R och sina B.",
         "Mättnadsgraden s_X = E_X / ME_X i [0, 1] styr hunger, tillväxt "
         "och svält.",
         "För en primärproducent bär samma plats näringspoolen N(c), i "
         "producentens egna enheter (t/km² den skulle kunna bli). "
         "Strömmarna och den öppna kanten flyttar den då med vattnet "
         "utan ny kod - och inget som läser reserven som energi läser en "
         "icke beslutande grupps reserv."],
        kicker="Definition")

    d.table(
        "Karttyper och kalendertabeller",
        ["Typ", "Symbol", "Enhet", "Tolkning"],
        [["Biomassakartor", "B_X(c)", "ton/cell",
          "Biomassa av funktionsgrupp X i cell c."],
         ["Reservenergikartor", "R_X(c)", "MJ/cell",
          "Lagrad energi hos beslutande FG; klipps till B_X · ME_X."],
         ["Näringspool", "N(c)", "t/km²",
          "Löst kväve per producent och lager, i producentens enheter."],
         ["Bärkraftskartor", "CC_X(c)", "ton/cell",
          "Tak per cell för icke beslutande FG (max_carrying_capacity)."],
         ["Tillgänglighet", "accessibility(c)", "[0,1]",
          "Gemensam habitatmask."],
         ["Fiskekartor", "Impact_p(c)", "MW-h/km²/år",
          "EMODnet-ansträngning för bottentrål och pelagisk trål."],
         ["Övriga påverkanskartor", "Impact_p(c)", "godtycklig",
          "Buller i dB, rotortäthet; muted tills ett scenario använder dem."],
         ["Strömfält", "-", "andel/tick",
          "Genereras proceduralt per tick ur brus, lagras inte."],
         ["Kalendertabeller", "F, m, P, T, M, N0", "per tick",
          "Ljus, attackfaktor, produktionsfaktor, temperatur, "
          "blandningsdjup, nitrat under lagret."]],
        [18, 16, 13, 43], kicker="Vad som finns i en körning", size=11,
        footnote="Kalendertabellerna är samma för alla celler och byggs ur "
                 "manifestet. Temperatur och nitrat är tidsserier per "
                 "vattenlager, inte kartor.")

    d.bullets(
        "Dagsljuskalendern",
        ["Kalendern ger varje tick ett datum och en tid: årstick k börjar "
         "k · ticklängd timmar efter 1 januari 00:00 soltid. 8 760 h är "
         "delbart med varje ticklängd 1-6 h.",
         "Solhöjden räknas ur latitud (58,15° N), deklination och soltid. "
         "Ljusnivån F(t) är medelvärdet av en logistisk funktion av "
         "solhöjden över tickets 10-minuterssteg.",
         "Startdagen lottas per värld och delas av alla delta-par och båda "
         "tecknen i den världen (CRN).",
         "Kalendern används på fem ställen:",
         ("-", "ljusberoende attackhastighet för visuella predatorer"),
         ("-", "ljusbegränsad tillväxt i båda producentlagren"),
         ("-", "månadstemperaturen som styr metabolism, intag och "
               "tillväxt"),
         ("-", "blandningsdjupet M(t), som späder bytet för betare och "
               "styr näringstillförseln"),
         ("-", "den tolv månader långa nitratserien under lagret"),
         "Varje beslutande FG ser F(t) som sista observationskanal. Att slå "
         "av kalendern ändrar in_dim och gör checkpoints ogiltiga."],
        kicker="simulation_settings.daylight", size=15)

    d.table(
        "Kalendern månad för månad",
        ["", "jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep",
         "okt", "nov", "dec"],
        [["ljus F", "0,37", "0,44", "0,54", "0,66", "0,79", "0,87", "0,83",
          "0,70", "0,58", "0,47", "0,38", "0,34"],
         ["blandningsdjup M (m)", "13", "9", "6", "6", "6", "6", "6", "7",
          "10", "10", "17", "15"],
         ["m sill → djurpl.", "0,70", "0,80", "0,94", "1,10", "1,30", "1,42",
          "1,36", "1,17", "0,99", "0,84", "0,72", "0,66"],
         ["m torsk → sill", "0,90", "0,93", "0,98", "1,04", "1,10", "1,13",
          "1,11", "1,06", "1,00", "0,95", "0,91", "0,88"],
         ["P/P_ref växtpl.", "0,16", "0,38", "0,72", "0,99", "1,12", "1,14",
          "1,03", "0,84", "0,58", "0,37", "0,15", "0,10"],
         ["P/P_ref djupväxtpl.", "0,27", "0,54", "0,80", "1,00", "1,17",
          "1,25", "1,21", "1,05", "0,82", "0,56", "0,25", "0,18"],
         ["T 0-50 m (°C)", "6,4", "5,5", "5,3", "6,2", "7,9", "9,9", "12,0",
          "13,3", "13,6", "12,6", "10,5", "8,4"],
         ["T 50 m (°C)", "7,8", "7,2", "6,7", "6,6", "6,8", "7,2", "8,0",
          "8,7", "10,2", "11,4", "10,6", "9,3"],
         ["Q10 sill, kostnad", "0,55", "0,51", "0,51", "0,54", "0,62", "0,73",
          "0,85", "0,95", "0,97", "0,90", "0,76", "0,64"],
         ["Q10 torsk, kostnad", "0,95", "0,90", "0,87", "0,86", "0,88",
          "0,91", "0,96", "1,01", "1,14", "1,26", "1,18", "1,06"],
         ["Q10 djurpl., kostnad", "0,80", "0,75", "0,74", "0,79", "0,88",
          "1,02", "1,18", "1,29", "1,32", "1,23", "1,06", "0,92"],
         ["Q10 djurpl., intag", "0,71", "0,64", "0,63", "0,69", "0,82",
          "1,01", "1,25", "1,43", "1,48", "1,34", "1,08", "0,87"],
         ["Q10 djurpl., tillväxt", "0,69", "0,62", "0,61", "0,67", "0,81",
          "1,01", "1,27", "1,46", "1,52", "1,36", "1,08", "0,86"]],
        [21] + [6] * 12, size=10.2, align=["left"] + ["right"] * 12,
        kicker="58,15° N, 6 h/tick, månadsmedel",
        footnote="Ur tools/probes/daylight_table.py och lib/world/"
                 "temperature.py. Attackfaktorerna och alla Q10-faktorer "
                 "utom sillens har årsmedel exakt 1; producentfaktorerna är "
                 "1 den 15 april. Temperatur: SMHI SHARK A16 + A17, "
                 "1991-2020; blandningsdjup: A17.")

    d.bullets(
        "Vattentemperatur",
        ["Med simulation_settings.temperature följer en växelvarm grupp "
         "temperaturen i sitt vattenlager (0-50 m för sill, djurplankton "
         "och mikrozooplankton; 50 m för torskfisk). Tre skilda effekter:",
         ("-", "metabolism_q10 på viloförbränningen och därmed alla "
               "actionkostnader: sill 2,2 (t_ref 14 °C, en "
               "sommarkostnad), torskfisk 2,2 och djurplankton 2,0 "
               "(årsmedel 1)"),
         ("-", "ingestion_q10 på djurplanktonets hela funktionella svar, "
               "a och 1/h tillsammans: 2,8 (Hansen m fl 1997)"),
         ("-", "growth_q10 på djurplanktonets tillväxtgren: 3,0 (Huntley "
               "& Lopez 1992), svälten oskalad"),
         "Mikrozooplanktonets betning har grazing_q10 2,6 vid 10 °C "
         "(Nielsen & Kiørboe 1994).",
         "Tumlare och sälar har ingen Q10: endoterma, och fältmetabolismen "
         "är platt i 0-20 °C vatten.",
         "Intag och tillväxt är normerade till årsmedel 1, så "
         "bibliotekets a, h och growth_rate förblir årsmedel och "
         "tillväxtbudgetens s* ändras inte (Del 4).",
         ("p", "Effekt: djurplanktonets vår startar i mars som vid A17, i "
               "stället för en månad för tidigt; sillens vintersvält "
               "försvann när kostnaden följde temperaturen.")],
        kicker="Q10 på kalendern", size=14.5)

    d.figure(
        "Vattenpelaren: två lager och en reservoar", draw_water_column,
        kicker="simulation_settings.nutrients",
        footnote="Linjär 2 × 2-avbildning per tick som flyttar vatten OCH "
                 "alger mellan lagren; varje producent har sin egen pool. "
                 "Nytt kväve kommer bara via utbytet med reservoaren.")

    d.formula(
        "Näringspoolen",
        [("Tillväxt betald ur poolen (Evans & Parslow 1985)", [
            r"\Delta B &= \min\left(r\,L(t)\,B\,\dfrac{N}{N + k_N\,Y\,H(t)},\ N,\ K - B\right) \qquad N \leftarrow N - \Delta B",
            r"\dfrac{dN}{dt} &= Y\left[e\,(N_0(t) - c) + \max(\dot M, 0)\,N_0 - \max(-\dot M, 0)\,c\right] \qquad c = \dfrac{N}{Y\,H}",
            ("t", "r = härledd growth_rate, L(t) = ljusfaktorn, H = lagrets "
                  "tjocklek (M(t) i blandade lagret), Y = 0,795 t/km² per "
                  "µmol/l och meter."),
        ])],
        tail=[("-", "k_N 0,5 µmol/l: en pool för nitrat och återbildat "
                    "ammonium. Y = 14 mg N/m² × Redfield 5,68 / C:WW 0,10. "
                    "N0 = A17 nitrat under lagret per månad."),
              ("-", "Betning återför grazing_regeneration 0,3 av den betade "
                    "biomassan (djurplankton) och 0,4 av intaget "
                    "(mikrozooplankton). Sjunkning (M1) och fekalier lämnar "
                    "kretsen: budgeten är öppen."),
              ("-", "Det logistiska faktorn är borta; K = 24 t/km² finns kvar "
                    "bara som tak (Del 4)."),
              ("-", "Utbytet e(t) 0,05-0,3 m/d och reservoarutbytet "
                    "0,2-1,0 m/d är anpassade till A17-nitratet, inte "
                    "uppmätta.")],
        kicker="lib/world/nutrients.py")

    # ---------------------------------------------------------------- 3
    d.section("3", "Egenskaper hos funktionsgrupper",
              "Parametrarna i fg_library.yaml, deras roll i ticken och "
              "deras kalibrerade värden.")

    d.table(
        "Egenskaper: energi och ämnesomsättning",
        ["Egenskap", "Nyckel", "Enhet", "Roll"],
        [["Beslutsmodell", "is_decision_maker", "ja/nej",
          "Avgör om FG:n har en policy och en energibudget."],
         ["Maximal reservenergi", "max_energy_reserve (ME_X)", "MJ/ton",
          "Tak för E_X. Reserven klipps till B_X · ME_X efter rörelse."],
         ["Viloförbränning", "resting_metabolism", "MJ/ton/tick",
          "Basförbrukning. Multipliceras med actionens kostnadsfaktor."],
         ["Kostnadsfaktorer", "feeding_cost, movement_cost",
          "faktor", "Multiplikator på viloförbränningen per action."],
         ["Temperatur", "metabolism_q10, metabolism_t_ref", "-, °C",
          "Q10 på viloförbränningen; t_ref i °C eller annual_mean."],
         ["Intag och tillväxt", "ingestion_q10, growth_q10", "-",
          "Q10 på hela funktionella svaret resp. tillväxtgrenen (årsmedel 1)."],
         ["Energiinnehåll", "energy_content", "MJ/ton",
          "Värdet som föda - och priset på ett ton egen ny biomassa."],
         ["Reserven som föda", "prey_includes_reserve, "
          "reserve_reference_fill", "ja/nej, andel",
          "Bytets reserv äts med (sill)."],
         ["Underhållsnivå", "maintenance_level (u_X)", "andel av ME",
          "Nollpunkt q = s − u och knät i hungerkurvan."]],
        [17, 26, 12, 40], size=10.8, kicker="Per funktionsgrupp")

    d.table(
        "Egenskaper: tillväxt, förluster och rörelse",
        ["Egenskap", "Nyckel", "Enhet", "Roll"],
        [["Nettotillväxt vid låg täthet", "r_max", "1/år",
          "Litteraturens r; grund för den härledda growth_rate (Del 4)."],
         ["Tillväxttakt", "growth_rate (MG_X)", "andel/tick",
          "HÄRLEDD av laddaren - redigeras aldrig direkt."],
         ["Naturlig mortalitet", "natural_mortality (M1)", "andel/tick",
          "Förluster som inte är predation från en FG."],
         ["Inkomstfinansierad tillväxt", "income_growth_kappa, "
          "feeding_season_months", "andel, mån",
          "Tillväxt ur tickets intag; budgeten betalas över säsongen."],
         ["Regionalt bärkraftstak", "regional_carrying_capacity (K)",
          "t/km²", "Förlust r_max/1460 · B_reg/K utanför budgeten."],
         ["Svälttakt", "starve_rate", "andel/tick",
          "Katabolism när q < 0."],
         ["Attackhastighet", "max_intake_rate (a)", "1/(ton·tick)",
          "Lutningen i det funktionella svaret; kan sättas per par."],
         ["Interferens", "interference (w)", "1/ton",
          "Beddington-DeAngelis: egen täthet sänker intaget."],
         ["Hastighet", "movement_speed (v)", "cellsidor/tick",
          "Sträcka per tick för den rörliga kohorten."],
         ["Minsta enhet", "min_split_biomass", "kg",
          "En individ: bara hela enheter delas (regel D)."],
         ["Utdöendetröskel", "extinction_threshold_factor", "faktor",
          "Cell nollställs under faktor · min_split_biomass."]],
        [19, 27, 12, 37], size=10.4, kicker="Per funktionsgrupp")

    d.table(
        "Egenskaper: predationspar",
        ["Egenskap", "Nyckel", "Enhet", "Roll"],
        [["Predationsmortalitet", "predation_mortality (M2)", "1/år",
          "Litteraturens förlust av bytet till predatorn; ingår i bytets "
          "tillväxtbudget."],
         ["Attackhastighet per par", "max_intake_rate", "1/(ton·tick)",
          "Flyttar halvmättnaden 1/(a·h) utan att flytta taket."],
         ["Hanteringstid", "handling_time (h)", "tick",
          "Sätter taket 1/h ton byte per ton predator och tick."],
         ["Assimilation", "assimilation_factor", "andel",
          "Andel av bytets energi som predatorn behåller."],
         ["Synlighetsgolv", "visibility_floor", "[0,1]",
          "Andel som förblir synlig trots vila; per par eller bytets."],
         ["Funktionellt svar", "functional_response", "2 | 3",
          "Tvingar Holling typ II eller III för paret."],
         ["Ljusberoende", "dark_ratio, light_threshold_deg", "andel, °",
          "Attack i mörker relativt dagsljus; solhöjd för halv detektion."],
         ["Koncentration", "mixed_layer_reference_m", "m",
          "a skalas med M_ref / M(t): betare äter på koncentration."]],
        [19, 27, 12, 37], size=10.8, kicker="Per par")

    d.table(
        "Egenskaper: producenter och betare",
        ["Egenskap", "Nyckel", "Enhet", "Roll"],
        [["Bärkraft per cell", "max_carrying_capacity", "ton/cell",
          "Logistikens K; med näringspool bara ett tak."],
         ["Ljusmättnad", "light_saturation (I_k)", "µmol/m²/s",
          "P-I-kurvans knä; tolv månadsvärden eller ett."],
         ["Vattenlager", "water_layer", "mixed | subsurface",
          "Vilket lager producenten lever i."],
         ["Näringspool", "nutrient_half_saturation, nutrient_yield",
          "µmol/l, t/km² per µmol/l·m", "Monod-begränsning och "
          "omräkning N → biomassa."],
         ["Återbildning", "grazing_regeneration, excretion_regeneration",
          "andel", "Del av betningen som går tillbaka till poolen."],
         ["Betarens effektivitet", "growth_efficiency", "andel",
          "Bruttotillväxteffektivitet (GGE) hos en betande NDM."],
         ["Betarens temperatur", "grazing_q10, grazing_t_ref", "-, °C",
          "Q10 på betningen i betarens lager."],
         ["Födotröskel", "food_threshold (F0)", "t/km²",
          "ERSEM:s perceptionsterm: refug vid låg bytestäthet."],
         ["Ätbar andel", "edibility_floor, edibility_half_saturation_umol_l",
          "andel, µmol/l", "Betaren ser e · B, e faller med nitratet."],
         ["Strömrespons", "current_response", "[0,1]",
          "Andel av strömfältet som FG:n följer."]],
        [17, 32, 15, 31], size=10.4, kicker="Icke beslutande grupper")

    d.table(
        "Bibliotekets värden - energi och rörelse",
        ["", "zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals",
         "seabirds *"],
        dm_energy_rows(),
        [24, 13, 13, 13, 13, 13, 13], size=10.4,
        align=["left", "right", "right", "right", "right", "right", "right"],
        kicker="fg_library.yaml, kalibrerat vid 6 h/tick",
        footnote="Läst ur fg_library.yaml när dokumentet byggs. Energier i "
                 "MJ/ton, takter per tick. * = muted i mareld2.yaml. "
                 "Sillens energy_content är den magra kroppen; fettreserven "
                 "kommer ovanpå.")

    d.table(
        "Bibliotekets värden - tillväxt och tak",
        ["", "zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals",
         "seabirds *"],
        dm_growth_rows(),
        [26, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5], size=11,
        align=["left", "right", "right", "right", "right", "right", "right"],
        kicker="Del 4 förklarar talen",
        lead="growth_rate och capacity_loss_rate härleds av laddaren ur "
             "r_max, M1, predationsmatrisen och K. Talen nedan är det "
             "laddaren räknar fram för mareld2.yaml med --mortality on.",
        footnote="Producenterna: growth_rate "
                 f"{f(phy['growth_rate'])} per tick (brutto), M1 "
                 f"{f(phy['natural_mortality'])}, K "
                 f"{f(phy['max_carrying_capacity'])} t/km². Bottensamhället: "
                 f"handsatt {f(ben['growth_rate'])} (0,82/år), K "
                 f"{f(ben['max_carrying_capacity'])}. Mikrozooplankton: 0 "
                 "- växer bara genom betning.")

    d.table(
        "Bibliotekets värden - predationspar",
        ["Predator → byte", "a", "h", "assim.", "golv", "svar",
         "övrigt", "M2 (1/år)"],
        pair_value_rows(),
        [33, 7, 7, 7, 9, 15, 19, 9], size=9.6,
        align=["left", "right", "right", "right", "right", "left", "left",
               "right"],
        kicker="Läst ur fg_library.yaml",
        footnote="\"byte 0,35\" = paret ärver bytets visibility_floor; "
                 "\"satt\" = functional_response i biblioteket. ρ = "
                 "dark_ratio / tröskel; F0 = food_threshold; e = "
                 "ätbarhetsgolv / halvmättnad (µmol/l). * = predatorn är muted.")

    d.bullets(
        "Plankton: tre grupper i två lager",
        ["Mikrozooplankton (ciliater och heterotrofa dinoflagellater) var "
         "tidigare ett avdrag på växtplanktonets takt, \"60 % "
         "mikrobetning\". Nu är det en egen icke beslutande grupp som betar "
         "båda producentlagren och själv äts av djurplanktonet:",
         ("-", "Imax 1,7/d vid 10 °C (Hansen m fl 1997), halvmättnad 70 µg "
               "C/l, födotröskel 12 mg C/m³ (ERSEM), GGE 0,3 (Straile 1997), "
               "M1 0,05/d, Q10 2,6"),
         "Djupväxtplanktonet är samma alger i lagret mellan blandningsdjupet "
         "och 30 m: vid A17 ligger april-augusti ~90 % av klorofyllet under "
         "det 6 m djupa sommarlagret, på nitraklinen.",
         "Djurplanktonet betar på KONCENTRATION: a skalas med 10 m / M(t) "
         "(djuplagret: 10 m / H(t)). Halvmättnad 50 µg C/l = 5 t/km² i "
         "ett 10 m lager.",
         ("-", "Djurplankton → mikrozooplankton är typ III (bytesväxling, "
               "Kiørboe m fl 1996): med typ II betades det ned till 1e-18 "
               "på två veckor."),
         ("-", "Mikrozooplanktonet ser bara en ätbar andel av djuplagret, "
               "e = 0,1 + 0,9 N/(N + 2): sensommarens Ceratium-skikt är för "
               "stort för ciliater (Nielsen & Kiørboe 1994)."),
         ("p", "Effekt: bruttoprimärproduktionen steg från 13 till ~40 g "
               "C/m²/år och djurplanktonets sommarnivå från 0,4 till 3-4 "
               "t/km² (A17 4,6-6,9).")],
        kicker="Avsnitt 177-182", size=14.5)

    d.bullets(
        "Sill: fettreserv, vinterföda och tillväxt ur inkomsten",
        ["Kroppen skiljs från reserven: energy_content 4 000 MJ/ton = mager "
         "kropp, max_energy_reserve 7 000 ≈ 18 % fett, maintenance_level "
         "0,2. Predatorer äter reserven med (prey_includes_reserve).",
         "Vinterfödan (avsnitt 162): ingen interferens är uppmätt hos sill "
         "och varje anpassning av sillens funktionella svar är typ II. "
         "Därför w 0,7 → 0,05, typ II på djurplankton och synlighetsgolv "
         "0,2 → 0,5 (grundbank, copepoder vandrar lite).",
         "Tillväxten byggs ur tickets intag, inte ur fettet (avsnitt 184): "
         "med fettfinansierad tillväxt omvandlade sillen fett till biomassa "
         "okt-dec, gick mager in i vintern och förlorade 25-30 %/år.",
         "Biomassan hålls av ett regionalt K = 5,0 t/km² och r_max 0,34 "
         "(Del 4).",
         ("p", "I fix68: ingen svält någon månad, s 0,35-0,41 i februari "
               "och 0,80 i maj-juni - fettet sparas till vintern. "
               "Kvarvarande vinterförlust är M1 och predation.")],
        kicker="Avsnitt 140, 162, 184, 185", size=15)

    d.bullets(
        "Torskfiskar: tre byten, energitak och regionalt K",
        ["Menyn är sill (typ III, ljusberoende), djurplankton (krillproxy) "
         "och bottendjur. Kannibalismen är urkryssad (avsnitt 160): en "
         "gruppbelöning lär sig aldrig att äta den egna gruppen, så "
         "budgeten betalade 0,22/år för en förlust som aldrig togs.",
         "Taket är en ENERGIgräns per par, h = 130 · E / 4 500: magsäckens "
         "tömningstid skalar med bytets energitäthet, så alla par mättar "
         "vid samma ~35 MJ per ton torskfisk och tick (~3 % av "
         "kroppsvikten per dygn).",
         "M1 0,415/år = residual M0 0,20 (ICES SMS) + predation från "
         "grupper som inte är FG (0,214).",
         "Fiske F 0,12/år på kartan, 98 % bottentrål.",
         "r_max 0,37 (dagens produktivitet) och K 1,02 t/km² - förklarat "
         "i Del 4.",
         ("p", "Utan K växte torskfisken med r − F ≈ 0,35/år, åt ned sillen "
               "och föll sedan i svält på vintern; med K ligger den på 0,63-0,66 "
               "t/km² och sillpoolen är stabil.")],
        kicker="Avsnitt 141, 147-148, 157, 160, 185", size=15)

    d.bullets(
        "Tumlare och sälar",
        ["Tumlare: kostnader mot uppmätt metabolism (Rojano-Donate m fl "
         "2018), vila 60 MJ/t/tick, äta = fältmetabolismen 90. Dieten är "
         "anpassad till Ecopath (attackhastighet 0,015 på sill, 0,2 på "
         "torskfisk), biosonar ger sillen golv 0,95. En enhet = 50 kg.",
         ("-", "Bifångst 0,017/år i garn läggs på bottentrålkartan: "
               "garnansträngningen i griden är ~0."),
         "Sälar (avsnitt 166-168, 185): bara knubbsäl, ~55 kg per enhet. "
         "Vila 50, äta 1,5×, M1 0,10 och r_max 0,11 (Carroll m fl 2025). "
         "Spawn på torskfisk.",
         ("-", "Torskfisk bär ~95 % av dieten (plattfisk, kolja, gråsej "
               "lumpas in): M2 0,12/år, a 0,2, golv 0,85 - säler hittar "
               "vilande bottenfisk med morrhåren."),
         ("-", "Pelagisk fisk är typ III med a 3e-4 per (t/cell)²: en låg "
               "attackhastighet står för den vertikala separation som en "
               "enlagersmodell saknar. Mål ~10 % av dieten."),
         ("p", "Öppet: den tränade sälpolicyn äter ~0 % pelagisk fisk. "
               "Ätkanalen har mättat (logit ~−8) och ARS-gradienten genom "
               "en softmaxandel är proportionell mot andelen (Del 9).")],
        kicker="Toppredatorer", size=14.5)

    # ---------------------------------------------------------------- 4
    d.section("4", "Tillväxttakt och logistiskt tak",
              "Den biologiska motiveringen: varifrån r_max kommer, vilka "
              "förluster tillväxten betalar, varför budgeten betalas vid "
              "s* och varför fisken har ett regionalt K.")

    d.bullets(
        "Vad tillväxttakten ska betyda",
        ["I maj-utgåvan var growth_rate (MG_X) ett fritt tal per grupp. Ett "
         "fritt tal går inte att pröva mot data: samma värde ger helt olika "
         "nettotillväxt beroende på vilka predatorer, vilken mortalitet och "
         "vilken energi modellen har.",
         "Det som går att mäta i naturen är r_max: hur snabbt ett glest "
         "bestånd växer när födan inte begränsar - rekrytering och "
         "individtillväxt minus all naturlig dödlighet. Det är parametern "
         "i den logistiska modellen dB/dt = r B (1 − B/K).",
         "Modellen tar själv ut förluster: M1 (naturlig mortalitet som inte "
         "är en FG) och M2 (predation från modellens FG). Tillväxttermen "
         "måste därför vara BRUTTO, r_max + M1 + ΣM2, för att nettot vid "
         "låg täthet ska bli r_max.",
         "Kryssas en predator in eller ur ändras bytets takt med just "
         "predatorns M2 - nettot är detsamma. Modellen tål en ändrad "
         "FG-uppsättning utan omkalibrering.",
         "Fiske, bifångst och det regionala taket ligger UTANFÖR budgeten: "
         "r_max är ofiskad tillväxt vid låg täthet, och fisket ska kunna "
         "sänka nettot.",
         ("p", "Budgeten är en förutsättning, inte ett utfall: det "
               "realiserade beror på föda och på vad policyerna lär sig.")],
        kicker="Princip (avsnitt 129)", size=15)

    d.formula(
        "Från litteraturbudget till tillväxttakt",
        [("lib/world/growth_budget.py - laddaren, FG-editorn och testerna "
          "räknar likadant", [
            r"g_X &= \dfrac{r_{\max,X} + M1_X + \sum_j M2_{Xj}}{(s^{*}_X - u_X)\cdot 1460}\cdot\dfrac{12}{n_X} \qquad \mathrm{beslutande\ FG}",
            r"r_X &= \dfrac{r_{\max,X} + M1_X + \sum_j M2_{Xj}}{1460} \qquad \mathrm{icke\ beslutande\ FG}",
            "",
            r"\Delta B_X &= B_X\,g_X\,\theta_X(t)\,(s_X - u_X) \qquad s_X \geq u_X,\ \mathrm{debiteras\ reserven}",
            ("t", "1 460 = antal 6-timmarstick per år; θ_X(t) = growth_q10-"
                  "faktorn (årsmedel 1); n_X = feeding_season_months "
                  "(12 om inte satt)."),
        ])],
        tail=[("-", "M2_Xj räknas när paret är ikryssat och predatorn är "
                    "aktiv - en betande icke beslutande predator bara om "
                    "paret har en attackhastighet."),
              ("-", "Med --mortality off lämnar laddaren M1 utanför g: "
                    "tas en förlust bort tas också dess kompensation bort."),
              ("-", "Grupper utan r_max (bottensamhälle, mikrozooplankton) "
                    "behåller sitt handsatta growth_rate.")],
        kicker="r_max, M1, M2, s*")

    d.table(
        "Budgeten per funktionsgrupp",
        ["FG", "r_max", "M1", "ΣM2 per predator", "brutto", "s*",
         "g / tick", "Pmax", "K"],
        budget_rows(),
        [22, 7, 7, 17, 7, 6, 15, 7, 9], size=10.4,
        align=["left", "right", "right", "left", "right", "right", "right",
               "right", "right"],
        kicker="Läst ur growth_budget när dokumentet byggs",
        lead="Alla takter per år utom g. Bruttot är vad tillväxten ska "
             "producera per år vid s*; Pmax är det mesta energin vid "
             "Hollingtaket tillåter. K är regionalt (t/km²) för fisken och "
             "per cell för producenterna.",
        footnote="Producenternas \"–\": icke beslutande grupper har ingen "
                 "reserv, så nämnaren är 1. Sillens g är ×12/5,5 eftersom "
                 "den bara växer när den äter. Tumlare och sälar har inga "
                 "predatorer i modellen.")

    d.table(
        "r_max: biologin bakom talen",
        ["FG", "r_max/år", "Vad talet är", "Källa"],
        [["phytoplankton", f(phy["r_max"]),
          "Utspädningsexperimentens tillväxt i april, 0,67/d brutto (netto "
          "av respiration och lys), minus det modellen tar ut explicit: "
          "mikrobetning 147, djurplankton 22,6, sjunkning 7,3",
          "Stelfox-Widdicombe m fl 2004; Calbet & Landry 2004"],
         ["subsurface_phyto.", f(sub["r_max"]),
          "Samma alger; bara ljus och näring skiljer", "som ovan"],
         ["zooplankton", f(zoo["r_max"]),
          "Copepodproduktion utan födobegränsning, 0,16/d vid 11,5 °C = "
          "58,4/år, minus total mortalitet 25/år",
          "Huntley & Lopez 1992"],
         ["pelagic_fish", f(pel["r_max"]),
          "Dagens tillväxt vid låg täthet för platsens blandning (vikter: sill "
          "0,73, makrill 0,19, skarpsill 0,06, taggmakrill 0,02)",
          "FishBase CMSY; ICES SAG; Euler-Lotka"],
         ["gadoids", f(gad["r_max"]),
          "Samma metod för torsk, kolja, vitling, gråsej och kummel",
          "Froese m fl 2017; ICES SAG"],
         ["porpoises", f(species("porpoises")["r_max"]),
          "Observerad tillväxt efter att bifångsten upphörde, 9,6 %/år",
          "Lockyer 2003; Forney m fl 2021"],
         ["seals", f(species("seals")["r_max"]),
          "Knubbsäl i Kattegatt-Skagerrak efter 1988, 10,2-13,6 %/år",
          "Carroll m fl 2025"],
         ["benthic_comm.", "(0,82)",
          "Bottensamhällets biomassåterhämtning efter trålning; handsatt g",
          "Hiddink m fl 2017; Pitcher m fl 2022"]],
        [15, 9, 50, 26], size=10.2, kicker="Ett tal per grupp, med källa",
        footnote="Mikrozooplankton har inget r_max: det växer med GGE 0,3 "
                 "av det det betar (Straile 1997). Sjöfåglar (muted) 0,10/år "
                 "(Niel & Lebreton).")

    d.formula(
        "Från historisk till dagens produktivitet",
        [("Euler-Lotka: ett fall i rekrytering per lekbiomassa med "
          "faktorn d", [
            r"r_{\mathrm{nu}} &= r_{\mathrm{prior}} - \dfrac{\ln d}{T} \qquad T = \dfrac{3}{r_{\mathrm{prior}}}\ \ \mathrm{(generationstid)}",
            ("t", "Torskfiskar: torsk 0,35, kolja 0,24, vitling 0,49, gråsej "
                  "0,44, kummel 0,26 → poolen 0,34-0,40, vald 0,37."),
            ("t", "Pelagisk fisk: NSAS-sill d 1,7, WBSS 2,25, makrill 3,6, "
                  "taggmakrill 1,6, skarpsill 1 → 0,34 (0,27-0,40)."),
        ])],
        lead=[("p", "CMSY-priors (r 0,50-0,51 för torskfiskarna, 0,44 för "
                    "platsens pelagiska blandning) är 1960-90-talets "
                    "tillväxt vid låg täthet. Rekryteringen per lekbiomassa "
                    "har sedan fallit: torsk 2,6×, kolja 4,7×, kummel 3,9× "
                    "(ICES SAG), kopplat till uppvärmning och hysteres "
                    "(Sguotti m fl 2019).")],
        tail=[("-", "Ett fall med faktorn d sänker r med ln(d)/T - inte med "
                    "faktorn d. Försöket med 0,51/2,5 = 0,20 var därför för "
                    "lågt (avsnitt 184.17-18)."),
              ("-", "Den gamla pelagiska 0,648 vägde in 36 % tobis, som "
                    "saknar bottnar i griden."),
              ("-", "Dagens r är alltså låg produktivitet vid låg täthet - "
                    "inte trängsel nära ett tak.")],
        kicker="Avsnitt 184.18, 185.18")

    d.bullets(
        "M1 och M2: förlusterna tillväxten betalar",
        ["M2 per par kommer från Nordsjöns Ecopath-modell för 1991 "
         "(Mackinson & Daskalov 2007), med alla arter under paraplytermen. "
         "Sill: torskfisk 0,30, tumlare 0,0117, säl 0,0015 per år.",
         "M1 = residual naturlig dödlighet M0 + predation från grupper som "
         "modellen saknar:",
         ("-", "torskfisk 0,20 (ICES SMS) + 0,214 (bläckfisk, hälleflundra, "
               "knot, rockor, marulk) = 0,415"),
         ("-", "pelagisk fisk 0,391 + 0,189 (hajar, rockor, annan fisk) = "
               "0,580"),
         ("-", "djurplankton: total mortalitet 25/år minus de 21 % fisken tar "
               "= 19,7; tumlare och sälar 0,10 ur Leslie-modeller"),
         ("-", "växtplankton: bara sjunkningen, 0,02/d - respiration och lys "
               "ingår redan i utspädningstakten och får inte räknas två "
               "gånger"),
         "Mikrozooplanktonets M2 147/år är en KVOT, 60 % av bruttotakten "
         "0,67/d: den gör växtplanktonets g till bruttotakten.",
         "Inte i budgeten: fiske (F 0,12 / 0,10), tumlarbifångst 0,017, det "
         "regionala taket och kannibalism (urkryssad)."],
        kicker="Ecopath 1991, ICES SMS", size=14.5)

    d.formula(
        "Varför budgeten betalas vid s*",
        [("Reserven under ideala förhållanden (avsnitt 160)", [
            r"h(s)\,C &= \mathrm{kostnad} + P\,(ec + s\,\mathrm{ME}) \qquad h(s) = \dfrac{1 - s}{1 - u}",
            r"s^{*} &= \dfrac{C/(1-u) - \mathrm{kostnad} - P \cdot ec}{C/(1-u) + P \cdot \mathrm{ME}} \qquad P = \dfrac{\mathrm{brutto}}{1460}",
            ("t", "C = taket 1/h × assimilation × energi för det bästa "
                  "ikryssade bytet; kostnad = resting_metabolism × "
                  "feeding_cost."),
        ])],
        lead=[("p", "Tillväxten per tick är g (s − u). Före avsnitt 160 "
                    "dimensionerades g vid full reserv, s = 1. Men "
                    "hungerfaktorn stänger intaget just där, så ingen grupp "
                    "producerade sin budget: produktion/budget = (medel-s − "
                    "u)/(1 − u), dvs sill 60 %, torskfisk 48 %, tumlare 39 %, "
                    "djurplankton 38 %.")],
        tail=[("-", "s* är där reserven stannar när gruppen äter bästa bytet "
                    "vid taket varje tick: inkomsten möter kostnad, "
                    "tillväxtens pris och utspädningen. Sluten form - ingen "
                    "policy, ingen rollout (att kalibrera mot en tränad "
                    "policys s vore cirkulärt)."),
              ("-", "Budgeten underkänns om P ≥ Pmax = (C − kostnad)/(ec + "
                    "u ME), eller om g > ME/ec (massbalansens tak).")],
        kicker="Hungern stänger vid full reserv")

    d.figure(
        "s* för djurplankton och torskfisk", draw_ideal_reserve,
        kicker="Där inkomst möter behov",
        items=[("-", f"Djurplankton: bästa byte mikrozooplankton, C = "
                     f"{f(b_zoo.ideal_intake)} MJ/t/tick mot kostnad "
                     f"{f(b_zoo.ideal_cost)}; s* {f(b_zoo.reference_fill)}, "
                     f"g {f(b_zoo.growth_rate)} - nära massbalanstaket "
                     f"ME/ec = {f(zoo['max_energy_reserve'] / zoo['energy_content'])}."),
               ("-", f"Torskfisk: C = {f(b_gad.ideal_intake)} mot kostnad "
                     f"{f(b_gad.ideal_cost)}; s* {f(b_gad.reference_fill)}. "
                     "Ett lågt s* ger ett högt g för samma budget.")],
        footnote="s* räknas vid bibliotekets kostnad (referenstemperaturen); "
                 "Q10-faktorerna har årsmedel 1 och lämnas utanför, så "
                 "budgeten förblir en egenskap hos biblioteket.")

    d.formula(
        "Sill: tillväxt ur inkomsten",
        [("Inkomstfinansierad tillväxt (avsnitt 184)", [
            r"G &= \min\left(\kappa\,I - C,\ \ B\,g\,(s - u)\,ec\right) \qquad \Delta B = G / ec",
            r"g &= \dfrac{r_{\max} + M1 + \sum M2}{(s^{*} - u)\cdot 1460}\cdot\dfrac{12}{5{,}5}",
            ("t", "I = tickets assimilerade intag, C = tickets "
                  "actionkostnader (MJ); resten av inkomsten går till "
                  "reserven. Inget intag - ingen tillväxt."),
        ])],
        lead=[("p", "Med fettfinansierad tillväxt fortsatte sillen att "
                    "omvandla fett till biomassa i oktober-december "
                    "(0,17-0,27 %/d) efter att födan tagit slut. Den gick "
                    "mager in i vintern och förlorade 25-30 % per år.")],
        tail=[("-", "Biologi: struktur byggs av föda - fett innehåller inget "
                    "kväve, och fastande fisk tappar GH → IGF-1-signalen. "
                    "Sillens somatiska tillväxt sker april - mitten av "
                    "september (Bachiller m fl 2018); 65-93 % av årets "
                    "längdtillväxt vid platsen sker före augusti (DATRAS)."),
              ("-", "Samma form som Wisconsin-bioenergetik och DEBlipid "
                    "(Martin m fl 2017). κ 0,75 är där budgeten går ihop "
                    "vid dagens s*."),
              ("-", "Månaderna dimensionerar bara takten: NÄR sillen växer "
                    "avgörs av födan. En kalenderbrytare prövades och togs "
                    "bort.")],
        kicker="income_growth_kappa, feeding_season_months")

    d.bullets(
        "Temperatur och tillväxt",
        ["Djurplanktonets tillväxt har en egen Q10 (growth_q10 3,0): "
         "Huntley & Lopez (1992) ger den födomättade copepodtillväxten "
         "0,0445 e^(0,111 T), ~0,08/d vid 5 °C. Den platta takten lät "
         "modellens djurplankton växa 13 %/d i februari.",
         ("-", "Effekt: våren startar på A17:s residenta bestånd (mars 1,7 "
               "mot 1,6 t/km²) och toppen hamnar i maj."),
         "Intaget har ingestion_q10 2,8 på hela funktionella svaret, så "
         "halvmättnaden står still medan taket följer temperaturen.",
         "Båda är normerade till årsmedel 1: bibliotekets growth_rate är "
         "årsmedlet, och s* och budgeten ändras inte.",
         "En Q10 på sillens tillväxt prövades och förkastades: 0-50 m-vattnet "
         "är varmast i augusti-oktober, så tillväxten flyttade IN i hösten - "
         "fel fas. Temperaturen är inte sillens tillväxtklocka; födan är "
         "det.",
         ("p", "Torskfiskar, sälar och tumlare har ingen tillväxt-Q10. "
               "Torskfiskarnas kostnad följer 50 m-temperaturen.")],
        kicker="Avsnitt 181, 183", size=15)

    d.formula(
        "Primärproducenterna: brutto, ljus och näring",
        [("Växtplankton och djupväxtplankton", [
            r"g &= \dfrac{r_{\max} + M1 + M2_{\mathrm{mikro}} + M2_{\mathrm{djur}}}{1460} = \dfrac{0{,}67\,\mathrm{d}^{-1}\cdot 365}{1460}",
            r"r(t) &= g\,L(t) \qquad L(t) = \dfrac{\mathrm{medel}_{z}\,\tanh\left(I_0(t)\,e^{-K_d z}/I_k\right)}{\mathrm{samma\ den\ 15\ april}}",
            r"\Delta B &= \min\left(r(t)\,B\,\dfrac{N}{N + k_N Y H},\ N,\ K - B\right)",
        ])],
        lead=[("p", "Utspädningstekniken mäter algernas egen tillväxt i "
                    "slutna flaskor under 24 h: netto av respiration och "
                    "lys (de verkar i varje flaska), brutto av betning och "
                    "sjunkning. Därför är g brutto, och betningen och "
                    "sjunkningen tas ut explicit.")],
        tail=[("-", "0,67/d är den högsta in situ-takten i april vid 7-9,5 "
                    "°C i södra Nordsjön; L(t) = 1 den 15 april och 0 på "
                    "natten."),
              ("-", "I_k 111-298 µmol/m²/s per månad (SMHI 14C, Anholt E); "
                    "djuplagret 30 (skuggadapterat, under regionala mätningar "
                    "63-117)."),
              ("-", "Logistikfaktorn är borta: det som stoppar en blomning "
                    "är att nitratet tar slut, inte trängsel.")],
        kicker="Avsnitt 131, 138, 172-173")

    d.bullets(
        "Det logistiska taket: varför ett K behövs",
        ["r_max-priorerna är r i en Schaefer-modell, dB/dt = r B (1 − B/K) − "
         "fångst. Modellen behöll r och släppte K: vid god föda växte "
         "poolen med r − F vid VARJE täthet.",
         "Mätt (avsnitt 184.13): torskfisken växte +0,35/år, 2,7× på fyra "
         "år, åt ned sillen och föll sedan i svält vintern år 3 - en "
         "predator-bytescykel som bara föda reglerade. Någon sådan cykel "
         "mellan torskfisk och sill finns inte dokumenterad; SMS ger "
         "torskfiskens M2 på sill 0,15/år utan trend.",
         "Interferens kan inte vara bromsen: w = 1 ger redan 2-3 % lägre "
         "intag per fördubbling (empiriskt 3-6 %), och nollväxt skulle "
         "kräva w ≈ 11 (avsnitt 161.1).",
         "Pelagiska poolen utan K växte 15-25 %/år och betade djurplanktonet "
         "10-14 gånger dess stående stock per år (Ecopath 1,9×). Litteraturen "
         "finner ingen top-down-kontroll av sill på Nordsjöns djurplankton.",
         "Täthetsberoendet hos torskfiskar är i huvudsak "
         "rekryteringskompensation (68-78 % av NO-atlantiska bestånd) plus "
         "kannibalism."],
        kicker="Avsnitt 184.13-185", size=14.5)

    d.formula(
        "Regionalt bärkraftstak",
        [("Förlusten efter M1, i varje cell (avsnitt 185)", [
            r"c &= \dfrac{r_{\max}}{1460\,K} \qquad B(c) \leftarrow B(c)\,\left(1 - c\,\bar B\right) \qquad R\ \mathrm{skalas\ lika}",
            r"\dfrac{1}{B}\dfrac{dB}{dt} &= r_{\max}\left(1 - \dfrac{\bar B}{K}\right) - F \qquad \mathrm{när\ M1\ och\ M2\ tas\ som\ budgeterat}",
            r"K &= \dfrac{B_{\mathrm{idag}}}{1 - F/r_{\max}} \qquad \mathrm{torskfisk:}\ \dfrac{0{,}69}{1 - 0{,}12/0{,}37} = 1{,}02",
            r"K &= \dfrac{B_{\mathrm{idag}}}{1 - P/r_{\max}} \qquad \mathrm{pelagisk:}\ \dfrac{3{,}33}{1 - 0{,}115/0{,}34} = 5{,}0",
            ("t", "B_reg = gridmedeltätheten (t/km²); P = bestånden "
                  "överskottsproduktion vid platsens vikter."),
        ])],
        tail=[("-", "Förlusten ligger utanför den härledda g, så nettot vid "
                    "god föda blir den logistiska modellen som r_max-priorn "
                    "kom ifrån."),
              ("-", "Den verkar med eller utan --mortality (den stänger "
                    "r_max, som g alltid betalar) och skalas inte av "
                    "--mortality_multiplier."),
              ("-", f"capacity_loss_rate: torskfisk {f(capacity_rate('gadoids'))}, "
                    f"pelagisk fisk {f(capacity_rate('pelagic_fish'))} per "
                    "tick och t/km².")],
        kicker="regional_carrying_capacity")

    d.figure(
        "Nettotillväxt med och utan tak", draw_capacity,
        kicker="F ur dagens fiskekartor",
        items=[("-", f"Jämvikten K (1 − F/r): torskfisk {f(eq_gad)} t/km² "
                     f"(= dagens 0,69), pelagisk fisk {f(eq_pel)} mot "
                     "startens 3,33 - K sattes med överskottsproduktionen "
                     "0,115, inte med F 0,10."),
               ("-", "Utan K är nettot konstant positivt: inget stoppar "
                     "tillväxten förrän födan tar slut."),
               ("-", "Realiserat (frysta policyer): torskfisk 0,63-0,66 - "
                     "sälarna tar 0,17-0,19 mot budgetens 0,12; pelagisk "
                     "fisk 2,75 - ~0,05/år mer förlust än budgeten.")],
        footnote="Kurvorna antar att M1 och M2 tas exakt som budgeterat. "
                 "Varje avvikelse i realiserad predation flyttar jämvikten.")

    d.bullets(
        "Effektiv kapacitet, inte historiskt tak",
        ["Torskfisk: dagens 0,69 t/km² är ett utarmat bestånd - decennier "
         "av överfiske, trålade uppväxtområden och uppvärmning. Även "
         "2010-19 var utarmat, och det historiska beståndet var mycket "
         "större. K är därför dagens EFFEKTIVA kapacitet, skild från det "
         "historiska taket, och r är dagens låga produktivitet.",
         ("-", "Prövat: K 0,9 och 1,4 med r 0,511; r 0,20 med K 3,0; r 0,37 "
               "med K 1,02 (avsnitt 184.15-19). Det sista håller beståndet "
               "stationärt och sillpoolen stabil."),
         "Pelagisk fisk: K står för kompensation på beståndsskala "
         "(täthetsberoende rekrytering) plus nettoemigration. Griden rymmer "
         "~0,4 % av den höstlekande sillen, de vuxna övervintrar i "
         "Öresund/Køge bukt och vinterbeståndet är ~90 % 1-2-åringar.",
         "Varför regionalt och inte per cell: kompensationen verkar på "
         "beståndet i uppväxtområdena, inte i en 1 km-cell. En term per cell "
         "skulle ge policyn en gradient att sprida ut sig som naturen inte "
         "har. B_reg ger samma förlusttakt i varje cell.",
         "K som funktion av bottentrålningen (habitat) prövades i "
         "litteraturen: ingen publicerad funktion, flytande turbiner, effekt "
         "0,03-0,3 %. Lämnat utanför baslinjen (avsnitt 185.9)."],
        kicker="Avsnitt 185.1, 185.9, 185.18", size=14.5)

    d.table(
        "Tak för icke beslutande grupper",
        ["FG", "K per cell", "Hur taket verkar", "Biologisk läsning"],
        [["phytoplankton", f(phy["max_carrying_capacity"]),
          "Bara ett tak när näringspoolen är på; tillväxten är Monod",
          "Det verkliga taket är nitrattillförseln; K kalibreras inte. "
          "Obetad stannar stocken på K."],
         ["subsurface_phyto.", f(sub["max_carrying_capacity"]),
          "Som ovan; lagret har egen pool och reservoar",
          "Tränat djurplankton betar ned det från K i april-juli"],
         ["microzooplankton", f(mic["max_carrying_capacity"]),
          "Inaktivt: gruppen växer bara genom betning",
          "Fältbiomassa 0,3-5 t/km², toppar ~10"],
         ["benthic_infauna", f(ben["max_carrying_capacity"]),
          "Logistiskt, r 0,82/år handsatt",
          "Användarens värde; litteraturen ~45-150 t/km². Modellens bestånd "
          "är 10-35× mindre, så torskfiskens M2 0,0048 är verkningslös."]],
        [17, 9, 34, 40], size=10.8, kicker="max_carrying_capacity",
        lead="För en icke beslutande grupp utan näringspool gäller den "
             "klassiska logistiken ΔB = r B (1 − B/K) per cell, plus "
             "seed_rate som återkoloniseringsgolv (bottensamhället 1e-6, "
             "växtplankton 0).",
        footnote="Bottentrålning utarmar bottensamhället med d × SAR "
                 "(d 0,10 per pass, Rijnsdorp m fl 2020), ~0,2/år vid "
                 "dagens ansträngning.")

    d.table(
        "Vad takt och tak ger i dag",
        ["Försök", "Torskfisk", "Pelagisk pool", "Djurplankton", "Avsnitt"],
        [["Torskfisk utan K, r 0,511",
          "0,92 → 1,94 → 1,24, svält vintern år 3",
          "jan 4,36 / 3,87 / 2,83, faller", "-", "184.13"],
         ["Torskfisk K 1,02, r 0,37", "0,63-0,66, stationär",
          "jan 5,35 → 7,06, stabil från år 3", "-", "184.19"],
         ["Pelagisk utan K, r 0,648", "0,66 / 0,66 / 0,68",
          "4,07 / 5,54 / 6,77, +15-25 %/år",
          "5,88 / 3,59 / 2,94; betas 14×/år", "185.17"],
         ["Pelagisk K 5,0, r 0,34", "0,66 / 0,62 / 0,58",
          "3,09 / 2,75 / 2,73, stationär",
          "6,44 / 5,66 / 5,50; betas 3,6×/år", "185.19"]],
        [22, 22, 23, 22, 9], size=11,
        kicker="Årsmedel t/km², år 1 / 2 / 3",
        lead="Frysta policyer i sandlådan, deterministisk probevärld med "
             "dagens fiske. Med K stannar beståndet nära dagens nivå i "
             "stället för att skjuta över; där tar taket ungefär "
             "överskottet r − F (torskfisk 0,23-0,24/år).",
        footnote="Policyerna var tränade UTAN det tak de testades med; "
                 "jämvikten (2,75 mot 3,33 för pelagisk fisk, torskfiskens "
                 "nedgång 9 %/år i sista raden) ska läsas först efter "
                 "omträning.")

    d.bullets(
        "Öppna frågor om takt och tak",
        ["s* antar bästa bytet vid taket varje tick. Den realiserade "
         "mättnaden är lägre, så budgeten är en övre gräns för den "
         "produktion gruppen faktiskt når.",
         "K är fenomenologiskt: det står för rekrytering och migration och "
         "härleds ur dagens biomassa, så jämvikten är dagens läge per "
         "konstruktion. Modellen kan inte förutsäga en återhämtning bortom "
         "K - en återhämtning kräver att K själv ändras.",
         "Skala: K och F gäller griden, inte beståndet. Stängs parkernas "
         "trålning sjunker torskfiskens F i griden från 0,12 till ~0,088 "
         "och jämvikten stiger ~13 % - men beståndets F ändras knappt.",
         "Realiserad predation avviker från budgeten: sälar tar 0,17-0,19 "
         "av torskfisken (0,12), tumlare 0,003-0,009 (0,056); torskfiskens "
         "diet är 62-76 % pelagisk mot Ecopath 26-39 %.",
         "Sälar och tumlare saknar K. Knubbsälen ligger nära sin bärkraft "
         "(Carroll m fl 2025) och borde växa 0-0,03/år, inte r_max.",
         "Bottensamhällets takt och K är handsatta; mikrozooplanktonets M2 "
         "är en kvot, inte en mätning."],
        kicker="Vad som inte är löst", size=14.5)

    # ---------------------------------------------------------------- 5
    d.section("5", "Beteendemodeller",
              "Policynätverket, observationen och maskningen som gör "
              "utdatan ekologiskt meningsfull.")

    d.figure(
        "Policynätverk", draw_policy_net, kicker="Arkitektur",
        lead="En beteendemodell är ett litet fullt kopplat nätverk. "
             "Standard är två dolda lager om 30 noder med sigmoid; de "
             "aktuella körningarna använder --policynetwork 1 16 tanh, ett "
             "lager om 16 noder.",
        items=[("-", "Alla beslutande FG utvärderas i EN batchad "
                     "torch.bmm över hela griden - en policy ser alla sina "
                     "celler samtidigt."),
               ("-", "Utdatan är 5 + N logits, N = alla FG i projektet. "
                     "Maskning sker FÖRE softmax, så en omöjlig action får "
                     "noll sannolikhetsmassa."),
               ("-", "Actionvektorn [N, E, S, W, Vila, Ät Y_1 ... Y_n] är en "
                     "fördelning: varje cell delar sin biomassa mellan "
                     "alla actions.")])

    d.figure(
        "Observation", draw_observation, kicker="Vad en population ser",
        footnote="Ändras Observability-matrisen, de observerbara "
                 "impaktkartorna, FG-uppsättningen, gridlayouten eller "
                 "dagsljuskalendern blir befintliga checkpoints ogiltiga.")

    widths = {fg: observation_width(fg) for fg in DM_ORDER}
    d.table(
        "Observability-matrisen",
        ["Observatör", "phyto", "subs.", "mikro", "zoo", "pel.", "torsk",
         "tuml.", "säl", "fågel", "botten"],
        observability_rows("ja"),
        [17] + [8.3] * 10, size=11,
        align=["left"] + ["center"] * 10,
        lead="Matrisen bestämmer observationens bredd: k_X = antal ja på "
             "raden utöver FG:n själv, räknat bara på aktiva FG. I dag "
             + ", ".join(f"{fg} k = {widths[fg][0]} ({widths[fg][1]} kanaler)"
                         for fg in DM_ORDER if fg not in muted_groups())
             + ".",
        footnote="Läst ur fg_library.yaml. Djurplanktonet ser sina tre byten "
                 "men inte sillen - inget predatorberoende beteende är "
                 "möjligt för det. * = muted.")

    d.bullets(
        "Maskning och hela enheter",
        ["Predation maskas bort när bytet inte finns i cellen eller inte "
         "ligger på menyn. Hungern är ingen mask utan en kontinuerlig "
         "faktor h = min(1, max(0, (1 − s)/(1 − u))).",
         "Rörelse maskas bort när grannen är otillgänglig, när riktningen "
         "lämnar griden (utan --migration on) eller när movement_speed är 0. "
         "Om varje action är maskad öppnas Vila.",
         "min_split_biomass är en odelbar enhet - en individ (tumlare 50 kg, "
         "säl 55 kg, fisk 0,5 kg):",
         ("-", "under en enhet kollapsar fördelningen till argmax och "
               "cellen flyttar hel"),
         ("-", "från en enhet (regel D, avsnitt 159) delas cellen på fem "
               "rumsliga andelar - stanna och N/E/S/W - och en andel under "
               "en enhet lämnas proportionellt till de andelar som håller "
               "en; ät/vila-mixen inom stanna behålls"),
         "Före regel D höll 88 % av tumlarcellerna mindre än ett djur, 95 % "
         "av den massan skapad av delningar. Ingen massa läckte, men "
         "\"21 kg-tumlare\" var meningslösa.",
         ("p", "Maskningen gör att modellen undviker orimliga val och låter "
               "utdatan följa modellens ekologiska regler.")],
        kicker="Före och efter softmax", size=15)

    d.figure(
        "Hunger, underhåll och tillväxt", draw_hunger_gate,
        kicker="Två kurvor som styr allt",
        lead="Hungerfaktorn h_X är 1 upp till underhållsnivån u_X och faller "
             "sedan linjärt till 0 vid full reserv. Överskottet "
             "q_X = s_X − u_X sätter tecknet på tillväxten; s* är punkten "
             "där tillväxtbudgeten betalas.",
        footnote="Under u_X äter en population med full aptit. Sillen följer "
                 "samma hungerkurva men växer ur inkomsten (Del 4). "
                 "Torskfiskens tillväxtkurva är förstorad tio gånger.")

    d.bullets(
        "Vila är också att gömma sig",
        ["Den andel av en beslutande FG som väljer Vila i ett tick är "
         "skyddad från predation samma tick - och osynlig i nästa ticks "
         "observation.",
         ("-", "synlig andel = 1 − π_vila · (1 − golv)"),
         "Golvet är bytets eget visibility_floor eller sätts per PAR: "
         "detektionsförmåga tillhör paret. Tumlarens biosonar ger golv 0,95 "
         "på sill, sälens morrhår 0,85 på torskfisk, sillens syn 0,5 på "
         "djurplankton.",
         "Ljuset och gömslet är olika saker: golvet säger hur väl gömslet "
         "fungerar, ljuset hur farligt det är att vara synlig. Vila stoppar "
         "också betningen - den verkliga kostnaden för dygnsvandring.",
         "Taket på uttaget ur en cell sätts av den BÄST detekterande "
         "predatorn, medan varje predator är begränsad av vad den själv "
         "ser.",
         ("-", "En exponeringsviktad M1 (avsnitt 139) finns i motorn men är "
               "inte satt i biblioteket.")],
        kicker="visibility_floor")

    # ---------------------------------------------------------------- 6
    d.section("6", "Uppdateringsregler",
              "Ticken, steg för steg, med de formler motorn kör.")

    d.figure("Pseudokod för ett tick", draw_tick_pipeline,
             kicker="Nio steg, i denna ordning",
             footnote="FG-ordningen slumpas varje tick, men inget steg beror "
                      "på den: predation och rörelse är vektoriserade, och i "
                      "steg 7-8 går vattenpelaren och betarna först i fast "
                      "ordning. GPU-motorn speglar varje steg.")

    d.formula(
        "Steg 2: påverkansmortalitet",
        [("Uppslagning i FG:ns egen tabell", [
            r"(\mathrm{biomass\_factor}_p,\ \mathrm{energy\_factor}_p) = \mathrm{interp}\left(\mathrm{impact\_table},\ \mathrm{Impact}_p(c)\right)",
            "",
            r"B_X(c) &\leftarrow B_X(c)\,\left(1 - \sum_p \mathrm{biomass\_factor}_p(c)\right) \qquad R_X\ \mathrm{skalas\ lika}",
        ])],
        lead=[("p", "Påverkan tas ut FÖRE predationen, så att den "
                    "bytesbiomassa predatorerna ser redan är reducerad. Den "
                    "verkar på alla grupper med en tabell - icke beslutande "
                    "grupper bara med biomassfaktorn.")],
        tail=[("-", "Tabellen interpoleras linjärt och klipps vid "
                    "ändpunkterna; ingen extrapolation."),
              ("-", "energy_factor höjer i steg 4 varje actionkostnad med "
                    "(1 + Σ energy_factor)."),
              ("-", "Träning: en impakt med baskarta får den kartan i varje "
                    "värld; en observerbar impakt utan karta spawnas per "
                    "värld; övriga saknas. Inferens läser bara "
                    "inference.impact_maps - en karta per impakt, aldrig "
                    "staplade."),
              ("-", "Påverkan ligger utanför tillväxtbudgeten.")],
        kicker="impact_table")

    d.figure(
        "Dagens fiske", draw_fishing_maps,
        kicker="EMODnet, fartyg > 12 m, medel 2015-2024",
        items=[("-", f"Torskfisk F {f(f_gad, sig=2)}/år (98 % bottentrål), "
                     f"pelagisk fisk {f(f_pel, sig=2)} (98 % pelagisk trål): "
                     "biomass_factor = q × ansträngning, q = F / 1460 / "
                     "kartans medel."),
               ("-", f"Bottensamhälle d × SAR, ~{f(regional_f('benthic_infauna'), sig=2)}/år; "
                     f"tumlarbifångst {f(regional_f('porpoises'), sig=2)}/år "
                     "på bottentrålkartan (garnen syns inte i VMS)."),
               ("-", "Vindkraftens buller och rotor ligger muted tills ett "
                     "scenario använder dem.")],
        footnote="tools/make_fishing_maps.py bygger kartorna. Samma kartor "
                 "ligger som baskartor (träning) och inferenskartor, så "
                 "träning, probe och inferens fiskar dagens ansträngning.")

    d.formula(
        "Steg 3: predation",
        [("Önskat intag, effektiv attackhastighet och uttag", [
            r"B_Y^{\mathrm{synlig}} &= B_Y\,\left(1 - \pi_{Y,\mathrm{vila}}\,(1 - \mathrm{golv}_{XY})\right) \qquad a_{XY}(t) = a_{XY}\,m_{XY}(t)\,\dfrac{M_{\mathrm{ref}}}{M(t)}",
            r"f_{XY} &= \theta_X(t)\,\dfrac{a_{XY}(t)\,B_Y^{\mathrm{synlig}}}{1 + a_{XY}(t)\,h_{XY}\,B_Y^{\mathrm{synlig}} + w_X\,B_X} \qquad (\mathrm{typ\ II})",
            r"D_{XY} &= B_X\,\pi_{X,\mathrm{ät}\,Y}\,f_{XY}\,h_X \qquad \mathrm{intag}_{XY} = D_{XY}\,\min\left(1,\ \dfrac{0{,}999\,B_Y^{\mathrm{synlig}}}{\Sigma_X D_{XY}}\right)",
        ]),
         ("Bokföring", [
            r"R_X &\leftarrow R_X + \sum_Y \mathrm{intag}_{XY}\,\mathrm{assimilation}_{XY}\,\left(\mathrm{energy\_content}_Y + [R_Y/B_Y]\right) \qquad B_Y \leftarrow B_Y - \sum_X \mathrm{intag}_{XY}",
        ])],
        tail=[("-", "Typ III ersätter B med B² i täljare och nämnare "
                    "(specialister och par med functional_response 3)."),
              ("-", "m_XY(t) = dagsljusfaktorn; M_ref/M(t) bara för par med "
                    "mixed_layer_reference_m; θ_X(t) = ingestion_q10-faktorn "
                    "(a och 1/h tillsammans); [R_Y/B_Y] bara för sill."),
              ("-", "w_X B_X är Beddington-DeAngelis-interferens. 0,999 "
                    "håller en överbetad cell borta från exakt noll.")],
        kicker="Holling + interferens + ljus + koncentration")

    d.figure(
        "Funktionellt svar", draw_functional_response,
        kicker="Vad formeln gör",
        items=[("-", "Vänster: torskfiskarnas tak är en energigräns - sill "
                     "(typ III, h 217) och bottendjur (typ II, h 87) mättar "
                     "vid samma MJ per ton och tick."),
               ("-", "Höger: djurplanktonets halvmättnad är 5 t/km² i ett "
                     "10 m lager. I sommarens 6 m lager räcker 3 t/km², i "
                     "novembers 17 m krävs 8,5 - samma koncentration."),
               ("-", "Typ III på mikrozooplankton ger bytet en refug vid låg "
                     "täthet: djurplanktonet växlar bort.")],
        footnote="Ingestion_q10 skalar hela kurvan med årstiden (0,63 i mars, "
                 "1,48 i september) utan att flytta halvmättnaden.")

    d.formula(
        "Ljusberoende attackhastighet",
        [("Per par med dark_ratio ρ < 1", [
            r"f(\mathrm{elev}) &= \mathrm{logistisk}\left(\dfrac{\mathrm{elev} - \theta}{2^\circ}\right) \qquad F(t) = \mathrm{medel\ över\ tickets\ 10\ min\ steg}",
            r"m_{XY}(t) &= \dfrac{\rho + (1 - \rho)\,F(t)}{\mathrm{årsmedel}\left(\rho + (1 - \rho)\,F\right)} \qquad a_{XY}(t) = a_{XY}\,m_{XY}(t)",
        ])],
        lead=[("p", "Visuella predatorer ser bytet mycket bättre i ljus. "
                    "ORSAKEN läggs i modellen - attackhastigheten - medan "
                    "bytets svar, dygnsvandring, lämnas åt policyerna.")],
        tail=[("-", "Normeringen gör bibliotekets a till årsmedlet, så "
                    "tillväxtbudgeten och energibudgetgrinden håller i "
                    "medel."),
              ("-", "Sill → djurplankton: ρ = 0,1, θ = −8° (bett ned till "
                    "0,001 lx, Batty m fl 1990)."),
              ("-", "Torskfisk → sill: ρ = 0,6, θ = −5° (magsäcksdata)."),
              ("-", "Sälar: inget ljusberoende - de jagar med morrhåren, och "
                    "dygnsrytmen lämnas åt Vila och ljuskanalen (avsnitt "
                    "167).")],
        kicker="dark_ratio, light_threshold_deg")

    d.formula(
        "Betande icke beslutande grupper",
        [("Mikrozooplankton, i början av populationssteget", [
            r"F &= e\,B_p \quad \mathrm{eller} \quad \dfrac{(e\,B_p)^2}{e\,B_p + F_0} \qquad e = e_0 + (1 - e_0)\,\dfrac{N}{N + K_e}",
            r"G &= B_g\,\theta_g(t)\,\dfrac{a'\,F}{1 + a'\,h\,F} \qquad a' = a\,\dfrac{M_{\mathrm{ref}}}{M(t)}",
            r"\Delta B_g &= \mathrm{gge}\cdot G \qquad \Delta N_p = \mathrm{excretion\_regeneration}\,(1 - \mathrm{gge})\,G",
        ])],
        lead=[("p", "En icke beslutande grupp med ett ikryssat par som har "
                    "en attackhastighet betar utan policy, med fast "
                    "Holling II och en födotröskel F0 (ERSEM:s "
                    "perceptionsterm).")],
        tail=[("-", "θ_g(t) = grazing_q10^((T − 10)/10) i betarens lager; "
                    "uttaget klipps vid MAX_HARVEST_FRAC."),
              ("-", "Bokförs som predation, och betaren räknas som predator "
                    "i bytets tillväxtbudget."),
              ("-", "Ätbarheten e finns bara på djuplagerparet: e0 0,1, K_e "
                    "2 µmol/l.")],
        kicker="Avsnitt 177, 182")

    d.formula(
        "Steg 4: energi efter action",
        [("Kostnad per action, före rörelse", [
            r"\mathrm{kostnad}_a &= B_X\,\pi_{X,a}\,\mathrm{resting\_metabolism}_X\,Q_{10}^{(T(t) - t_{\mathrm{ref}})/10}\,\mathrm{cost}_a\,\left(1 + \sum_p \mathrm{energy\_factor}_p\right)",
            r"R_{a} &= \max\left(0,\ R_X\,\pi_{a} - \mathrm{kostnad}_{a}\right) \quad (+\ \mathrm{energivinst\ för\ ät})",
            "",
            r"G &= \min\left(\kappa\,I - C,\ B\,g\,(s - u)\,ec\right) \qquad \mathrm{bara\ inkomstfinansierad\ FG\ (sill)}",
        ])],
        lead=[("p", "Reserven delas upp på actiongrupperna i proportion "
                    "till hur biomassan valde, varje grupp betalar sin "
                    "kostnad, och ätgruppen får predationens energivinst.")],
        tail=[("-", "Alla actions kostar. Vila är den billigaste, inte "
                    "gratis."),
              ("-", "Påverkan verkar multiplikativt: buller gör varje action "
                    "dyrare i stället för att döda."),
              ("-", "Sillens tillväxt G tas här, ur den stillastående "
                    "reserven, före rörelsen; tillväxtgrenen i steg 8 är "
                    "då noll för sillen."),
              ("-", "Efter rörelsen klipps reserven till [0, B · ME].")],
        kicker="Metabolism")

    d.formula(
        "Steg 5: rörelse",
        [("Utflöde, kvarvarande massa och blandning", [
            r"\mathrm{ut}_d(c) &= B_X(c)\,\pi_{X,d}(c)\,v_X\,\mathrm{mask}_d(c)",
            r"B_X^{+}(c) &= B_X(c) - \sum_d \mathrm{ut}_d(c) + \sum_d \mathrm{ut}_d\left(\mathrm{granne}_d^{-1}(c)\right)",
            "",
            ("t", "R följer med sin biomassa: flyttas 20 % av massan följer "
                  "20 % av den actiongruppens reserv med."),
        ])],
        lead=[("p", "Rörelse skapar och förstör ingen biomassa. Den delar "
                    "bara upp cellens massa i utflöden och kvarvarande "
                    "massa.")],
        tail=[("-", "v_X är en sträcka per tick, inte en sannolikhet; den "
                    "skalas linjärt med ticklängden."),
              ("-", "Blockerat flöde stannar i källcellen. Ett utflöde vars "
                    "målcell skulle hamna under utdöendetröskeln återförs; "
                    "en cell under tröskeln som ändå skickar lämnar ALL sin "
                    "massa till grannarna (avsnitt 156)."),
              ("-", "Med --migration on lämnar kantflödet griden och "
                    "återinförs koncentrerat längs kanten.")],
        kicker="Slice-assign, fyra riktningar")

    d.bullets(
        "Strömmar",
        ["Med --currents on adderas en advektion efter rörelsen: ett "
         "molnlikt brusfält som skrollar mot öst och syd flyttar en andel "
         "av biomassan varje tick.",
         ("-", "Fältet samplas proceduralt ur tickräknaren och en "
               "världsseed - det lagras inte och förbrukar inte ekologins "
               "slumptal."),
         ("-", "--current-strength sätter den maximala andelen per tick "
               "(standard 0,1)."),
         "Vilka grupper som följer med styrs av current_response: "
         "växtplankton, djupväxtplankton, mikrozooplankton och djurplankton "
         "1,0, bottensamhälle och fisk 0.",
         "Näringspoolen ligger i producentens reservplats och flyttas därför "
         "med vattnet, konservativt.",
         "Strömmen delar rörelsens masker - blockerat flöde stannar kvar - "
         "och skapar eller förstör ingen massa."],
        kicker="current_response")

    d.formula(
        "Steg 7: vattenpelaren",
        [("Lagerutbyte per tick, före all tillväxt", [
            r"(X_m,\ X_d)' &= A(t)\,(X_m,\ X_d) \qquad X = \mathrm{vatten,\ näring,\ alger}",
            r"f_{\mathrm{upp}} &= \dfrac{\max(\Delta M, 0)}{H} \qquad f_{\mathrm{ned}} = \dfrac{\max(-\Delta M, 0)}{M} \qquad + \ e(t),\ e_b(t)",
            ("t", "M = blandningsdjupet, H = 30 m − M; e mellan lagren, e_b "
                  "mellan djuplagret och reservoaren vid 50 m."),
        ])],
        lead=[("p", "När blandningsdjupet växer på hösten rycks "
                    "djuplagrets vatten, näring och alger med in i det "
                    "blandade lagret; när det krymper på våren lämnas de "
                    "kvar. Utbytet sker innan någon grupp växer.")],
        tail=[("-", "Nytt kväve kommer bara genom e_b (N_res − c_d): "
                    "reservoaren är A17:s nitrat vid 50 m per månad."),
              ("-", "Djuplagrets ljus: ytljuset × e^(−0,29 M) × e^(−0,17 z) "
                    "över lagret (Kd 0,17 = 1,7 / siktdjup)."),
              ("-", "Djuplagrets I_k 30 och sommarutbytet 0,2 m/d är "
                    "generösa mot litteraturen (63-117; 0,02-0,05) - "
                    "anpassade till A17:s nitrat och kolonnklorofyll.")],
        kicker="Avsnitt 180")

    d.formula(
        "Steg 8: tillväxt för beslutande FG",
        [("M1, regionalt tak, sedan tillväxt eller svält", [
            r"B, R &\leftarrow (B, R)\,(1 - M1) \qquad B, R \leftarrow (B, R)\,(1 - c\,\bar B)",
            r"\mathrm{önskan} &= B\,\mathrm{MG}_X\,\theta_X(t)\,q \qquad q = s - u \geq 0",
            r"\Delta B &= \min\left(\mathrm{önskan},\ \dfrac{\max(0,\ R - u\,\mathrm{ME}\,B)}{ec}\right) \qquad R \leftarrow R - \Delta B\,ec",
        ]),
         ("Svält (q < 0) är en förlust, inte ett köp", [
            r"\Delta B &= B\,\mathrm{starve\_rate}\,q \qquad R\ \mathrm{skalas\ med}\ B",
        ])],
        lead=[("p", "Tillväxten är GRINDAD av mättnaden, men begränsad av "
                    "den reserv som står över underhållsnivån - och den "
                    "reserven DEBITERAS. MG_X är den härledda growth_rate.")],
        tail=[("-", "Utan debiteringen bär ny biomassa ett energiinnehåll "
                    "som aldrig tagits ur något: mätt till 198-344 % av den "
                    "massa som åts, mot ett termodynamiskt tak på 28,9 %."),
              ("-", "ME/ec blir ett hårt tak på growth_rate: djurplankton "
                    f"{f(zoo['max_energy_reserve'] / zoo['energy_content'])} "
                    f"mot g {f(zoo['growth_rate'])}."),
              ("-", "Sillens tillväxtgren är noll här (den växte i steg 4); "
                    "svälten gäller alla.")],
        kicker="Massbalanserad sedan september")

    d.figure(
        "Energibokföringen i ett tick", draw_energy_flow,
        kicker="Var massan kommer ifrån",
        items=[("-", "Spillet är (1 − assimilation) av intaget: 0,35 för "
                     "djurplankton på växtplankton, 0,10 för tumlare på "
                     "sill."),
               ("-", "Reserven är den enda bufferten. Alla actions tar ur "
                     "den; temperatur och påverkan höjer kostnaden."),
               ("-", "Reserv blir biomassa till priset energy_content per "
                     "ton. En population på underhållsnivån kan inte växa "
                     "hur mycket föda som än finns.")],
        footnote="tools/probes/fg_year_budget.py och hloss-proberna läser ut "
                 "fördelningen mellan andning, tillväxt och spill per FG.")

    d.bullets(
        "Förluster: fyra kanaler",
        ["Ticken bokför fyra förlustkanaler per FG: predation, svält, "
         "påverkan och naturlig mortalitet. lib/environments/ecosystem_env/"
         "loss_accounting.py är enda stället som gör andelar av dem.",
         "Naturlig mortalitet (M1) kräver --mortality on och skalas av "
         "--mortality_multiplier. Den tas ut på biomassa och reserv, före "
         "tillväxten - för icke beslutande grupper bara på biomassan.",
         "Det regionala taket bokförs också som naturlig förlust men verkar "
         "alltid och skalas inte.",
         "Svält är katabolism: under underhållsnivån krymper biomassan med "
         "starve_rate · |q|, och reserven skalas med samma kvot.",
         "Massa som nollställs av utdöendetröskeln bokförs som svält.",
         "Mortalitet genom påverkan - fiske, bifångst, buller, rotorer - "
         "kommer från påverkanskartorna."],
        kicker="Predation, svält, påverkan, naturlig")

    d.formula(
        "Steg 8: tillväxt för icke beslutande FG",
        [("Producenter med näringspool; övriga logistiskt", [
            r"B &\leftarrow B\,(1 - M1) \qquad N \leftarrow N\,\mathrm{keep}(t) + \mathrm{tillförsel}(t)",
            r"\Delta B &= \min\left(r\,L(t)\,B\,\dfrac{N}{N + k_N Y H},\ N,\ \mathrm{CC} - B\right) \qquad \mathrm{producent}",
            r"\Delta B &= r\,B\,\left(1 - \dfrac{B}{\mathrm{CC}}\right) + \mathrm{seed\_rate}\cdot\mathrm{CC}\cdot 10^{U(-1,1)} \qquad \mathrm{övriga}",
        ])],
        lead=[("p", "Betningen har redan tagits ut i steg 3 (djurplankton) och "
                    "i början av steg 7 (mikrozooplankton). Den gamla "
                    "sinustermen är borttagen; säsongen kommer från "
                    "kalendern.")],
        tail=[("-", "I_0: klarhimmelsinstrålning × 0,45 PAR × 4,57 µmol/J × "
                    "molntransmission (SMHI). K_d 0,29/m; blandningsdjup "
                    "6-17 m (A17)."),
              ("-", "En näringsstressförlust (stress_loss_max, ERSEM-form) "
                    "finns i motorn men är avstängd: sommarminimum kommer "
                    "från betningen."),
              ("-", "Bottensamhället är logistiskt med handsatt r 0,82/år.")],
        kicker="Avsnitt 138, 172-177")

    d.bullets(
        "Sammanfattning av bokföringen",
        ["Ny biomassa = biomassa efter påverkan, predation, förflyttning och "
         "lagerutbyte + tillväxt − svält − naturlig mortalitet − regionalt "
         "tak.",
         "Biomassa kan bara öka när energin tillåter det: för beslutande FG "
         "reserven över underhållsnivån (sill: tickets inkomst), för "
         "producenter näringspoolen, för mikrozooplankton betningen.",
         "Reservenergin följer alltid sin biomassa; hos sill följer den "
         "med in i predatorn.",
         "Kvävet har en öppen budget: tillförsel från reservoaren, "
         "återbildning från betning, export genom sjunkning och fekalier.",
         "Det som INTE bevaras: mortalitet, påverkan, taket och "
         "utdöendetröskeln tar massa ur systemet; migration flyttar massa "
         "över randen. Predation, rörelse, strömmar och lagerutbyte bevarar "
         "den."],
        kicker="Vad som bevaras och inte")

    # ---------------------------------------------------------------- 7
    d.section("7", "Träning med ARS",
              "Hur beteendemodellerna tas fram, vad de optimerar och vad "
              "de inte ska förväxlas med.")

    d.bullets(
        "Träning: mål och perspektiv",
        ["Målet är inte ekologiskt optimala policyer, utan plausibla, "
         "stabila och testbara policynätverk för scenarioanalys.",
         "Modellen är Eulerisk. Det finns ingen individ att följa i flera "
         "steg - biomassan delas upp, flyttar och blandas. Därför "
         "optimeras hur väl ett FÄLT klarar sig över en rollout.",
         "Det som körs är ARS: gradientfritt och parallelliserat, på CPU "
         "(train.py) eller i tensorform på GPU (train_gpu.py).",
         "Med kalendern tränas på ettårsrollouts, 1 460 tick, så att varje "
         "rollout innehåller alla årstider. Fleråriga effekter läses i "
         "separata inferenskörningar.",
         "En gruppbelöning belönar återhållsamhet: djurplanktonet vilar "
         "62-83 % fast det enskilt lönar sig att äta (avsnitt 170) - en "
         "allmänningsdisciplin som verkliga copepoder inte visar."],
        kicker="Vad ARS ska åstadkomma")

    d.formula(
        "Fitness",
        [("Standard: total energi, integrerad över rollouten", [
            r"E_{\mathrm{tot}}(t) &= \sum_c \left(B_X(c,t)\,\mathrm{energy\_content}_X + R_X(c,t)\right)",
            r"\mathrm{Fitness}_X &= \dfrac{1}{H}\sum_{t=1}^{H}\,\log\dfrac{E_{\mathrm{tot}}(t) + \varepsilon}{E_{\mathrm{tot}}(0) + \varepsilon}",
            ("t", "H = --n_eval_ticks, ε = max(1e-6 E_tot(0), 1e-9)"),
        ])],
        tail=[("-", "Biomassa och reserv i SAMMA valuta: ett ton biomassa "
                    "är värt exakt sitt energiinnehåll."),
              ("-", "Integralformen ger lägre betyg åt en policy som "
                    "kraschar bytet mitt i rollouten och återhämtar sig på "
                    "slutet."),
              ("-", "Logaritmen gör belöningen skalinvariant: en halvering "
                    "kostar exakt vad en fördubbling är värd."),
              ("-", "--legacyreward ger maj-utgåvans α log(B) + β log(R) "
                    "+ κ t/H.")],
        kicker="Vad en rollout mäter")

    d.formula(
        "Lokal belöning per cell",
        [("--local_reward: samma fråga, ställd per cell", [
            r"A(c,t) &= B(c,t)\,\mathrm{energy\_content} + R(c,t)",
            r"\mathrm{belöning}(c) &= \mathrm{klipp}\left(\dfrac{B(c,t+1)}{A(c,t)},\ 0{,}2,\ 5{,}0\right) \qquad B(c,t+1) = \sum_d \mathrm{andel}_d(c)\,Q(\mathrm{mål}(c,d),\ t+1)",
            r"\mathrm{Fitness}_X &= \mathrm{medel}_t\ \dfrac{1}{N_{\mathrm{celler}}}\sum_c w_c\,\log\,\mathrm{belöning}(c)",
        ])],
        lead=[("p", "Den globala belöningen domineras av de stora cellerna: "
                    "en policy som sköter en tusentonscell bra och ett kilo "
                    "dåligt får samma betyg som tvärtom.")],
        tail=[("-", "andel_d(c) är en identitet, inte en approximation: "
                    "allt efter rörelsen är cellvis multiplikativt."),
              ("-", "Använd --local_reward_norm grid med metriken log: sum "
                    "belönar tunn spridning och mean belönar att döda de "
                    "sämsta cellerna."),
              ("-", "w_c = A_cᶿ; θ = 1 återger den globala belöningen.")],
        kicker="Skalinvariant per cell")

    d.formula(
        "ARS-uppdateringen",
        [("Per iteration, per beslutande FG med vikter θ", [
            r"\delta_1, \ldots, \delta_N &\sim \mathcal{N}(0,\ I) \qquad F_i^{\pm} = \mathrm{fitness}(\theta \pm \sigma\,\delta_i)",
            ("t", "Samma värld, seed och kalenderstart för + och − (CRN)."),
            r"\theta &\leftarrow \theta + \dfrac{\eta}{b\,\sigma_F}\sum_{i \in \mathrm{topp}\,b}\left(F_i^{+} - F_i^{-}\right)\delta_i \qquad b = N/2",
        ])],
        tail=[("-", "ARS-V2: observationerna normaliseras med löpande medel "
                    "och varians per FG och klipps till ±10."),
              ("-", "Topp-b väljer paren med högst max(F⁺, F⁻); σ_F är "
                    "deras standardavvikelse, så steglängden är oberoende "
                    "av belöningens skala."),
              ("-", "Gradienten genom en softmaxandel är proportionell mot "
                    "andelen: en action som tryckts ned nära noll kan inte "
                    "återhämta sig (sälens pelagiska ätkanal, avsnitt "
                    "185.14).")],
        kicker="Augmented Random Search")

    d.bullets(
        "Co-evolution",
        ["Standardläget är co-evolution: ALLA beslutande FG perturberas "
         "samtidigt inom samma delta-par och utvärderas i EN gemensam "
         "rollout. Varje FG får sin egen fitness ur samma simulering.",
         ("-", "En bytespolicy och en predatorpolicy är bara meningsfulla "
               "mot varandra."),
         "--no_coevolution ger det alternerande schemat: en FG i taget, "
         "övriga frysta.",
         "En generation = --iter-per-gen iterationer. Efter varje "
         "generation sparas checkpoints, och en deterministisk probe "
         "loggar biomassan per FG.",
         "Utvärderingsproben följer rolloutlängden och körs efter varje "
         "uppdatering som standard (--eval-every 1; reglaget i visningen "
         "styr också biomassproben)."],
        kicker="Alla arter samtidigt")

    d.bullets(
        "GPU-träning: train_gpu.py",
        ["lib/gpu/ är en tensorversion av samma tick: alla världar, delta-par "
         "och tecken körs samtidigt, i block av tick som spelas upp som "
         "CUDA-grafer. Paritetstester jämför varje steg - också "
         "näringspool, lagerutbyte, betning, Q10, inkomsttillväxt och "
         "regionalt K - mot CPU-motorn.",
         "Aktuella körningar (fix64-fix69): 32 delta-par × 1 värld × "
         "1 460 tick, 1 × 16 tanh, η 0,06, σ 0,1; mortalitet, migration, "
         "strömmar, dagsljus, temperatur, näringspool och fiske på; "
         "~85 s per uppdatering.",
         "Kalenderstarten dras per VÄRLD och delas av alla par (avsnitt "
         "150).",
         "Håll η/σ ≲ 0,6: vid η 0,1 och σ 0,05 växte vikternas norm som en "
         "slumpvandring.",
         "--init-from startar från en annan körnings policyer när "
         "observationen och utdatan är oförändrade.",
         ("-", "Visningens probe läser om fg_library.yaml vid varje probe, "
               "men tränaren behåller biblioteket den startade med.")],
        kicker="lib/gpu/", size=14.5)

    d.table(
        "Profiler och formande termer",
        ["Inställning", "Standard på train.py", "sanity", "info", "deep"],
        [["generations / iter-per-gen", "inf / 20", "10 / 15", "10 / 20",
          "80 / 20"],
         ["n_deltas", "10", "16", "16", "20"],
         ["n_eval_ticks", "15", "100", "150", "200"],
         ["rollouts_per_delta", "1", "3", "3", "3"],
         ["entropy_coef", "0", "0", "0", "0"],
         ["argmax_penalty", "0", "0", "0", "0"],
         ["softmax-temperatur", "1,0", "1,0", "1,0", "1,0"]],
        [26, 22, 14, 14, 14], size=12,
        align=["left", "right", "right", "right", "right"],
        lead="Entropibonus, argmax-straff och temperaturhärdning verkar på "
             "actionfördelningen i stället för på biologin. De finns kvar "
             "som flaggor men är avstängda som standard, med eller utan "
             "profil, på båda tränarna.",
        footnote="--profile sanity | info | deep. Explicita flaggor vinner "
                 "alltid över profilens värden.")

    d.figure(
        "Träningsvärldar", draw_spawn_examples,
        kicker="Fyra spawnstrategier",
        lead="För att undvika överanpassning till en enskild karta tränas "
             "policyerna i många slumpade världar. Varje FG har en egen "
             "spawnstrategi och ett intervall för startbiomassa.",
        items=[("-", "uniform: oberoende slumpvikt per cell. perlin: "
                     "naturlikt brus (växtplankton, bottensamhälle)."),
               ("-", "colony: gaussiska kolonier med jittrad amplitud "
                     "(sill, torskfisk)."),
               ("-", "env_driven: vikten byggs ur andra fält - djurplankton "
                     "och mikrozooplankton följer växtplankton, tumlare "
                     "sill, sälar torskfisk.")],
        footnote="Startbiomassan lottas per värld ur intervallet i "
                 "mareld2.yaml, startdagen ur kalendern, näringspoolen ur "
                 "ytnitratet vid startdagen.")

    d.bullets(
        "Checkpoints",
        ["En checkpoint är results/<körning>/policy_<fg>.pth med "
         "{'state_dict': ..., 'obs_stats': {...}} - vikterna OCH "
         "observationsnormaliseringens statistik.",
         "--resume fortsätter en körning; --init-from startar en ny från "
         "en annan körnings policyer; inference.py kör dem mot en "
         "deterministisk värld.",
         "Följande gör befintliga checkpoints ogiltiga:",
         ("-", "FG-uppsättningen: utdatan har 5 + N kolumner i alfabetisk "
               "ordning, så en ny FG flyttar varje ätkolumn (avsnitt 177, "
               "180)"),
         ("-", "Observability-matrisen, observerbara impakter, gridlayouten "
               "och dagsljuskalendern"),
         "Biblioteksändringar (K, r_max, attackhastigheter) gör dem inte "
         "formellt ogiltiga, men en policy som tränats i en annan värld ska "
         "tränas om innan den citeras.",
         "Kopiera en checkpoint innan den jämförs - en pågående körning "
         "skriver över policy_*.pth."],
        kicker="Vad som sparas och vad som bryter det")

    # ---------------------------------------------------------------- 8
    d.section("8", "Diagnostik och utvärdering",
              "Verktyg som testar mekaniken utan ARS, och vad modellen gör "
              "mot data.")

    d.table(
        "Viabilitetsriggen",
        ["#", "Villkor", "Formel", "Standard"],
        [["1", "Överlevnad", "B_f(t) > 0 för alla t ≤ T", "-"],
         ["2", "Golv", "eq_f ≥ golv · B_f(0)", "golv = 0,10"],
         ["3", "Tak", "eq_f ≤ tak · B_f(0)", "tak = 10"],
         ["4", "Stationaritet",
          "max(eq_f/föreg_f, föreg_f/eq_f) ≤ max_drift", "max_drift = 2,0"]],
        [6, 26, 44, 24], size=12.5, kicker="Fryst beteende",
        lead="Riggen kör världen med FRYST, handkodat beteende: snabb och "
             "deterministisk, och därmed bra för att mäta en parameter- "
             "eller mekanikändring isolerat. eq_f är medelbiomassan i "
             "horisontens sista fönster.",
        footnote="Armar: greedy, greedy_hide, greedy_flee, eat, eat_pays, "
                 "_stay-varianter och blandade armar (policy för alla utom "
                 "en grupp). Utslaget gäller DET beteendet, inte världen.")

    d.table(
        "Handkodat mot tränat",
        ["FG", "greedy", "greedy_stay", "eat", "tränat (fix58)"],
        [["phytoplankton", "0,12", "0,12", "0,14", "0,59-0,62"],
         ["zooplankton", "0,16", "0,16", "0,16", "0,72"],
         ["benthic_infauna", "0,13-0,27", "0,45-0,62", "1,24-1,32",
          "0,55-0,62"],
         ["pelagic_fish", "0,001", "0,01", "0,25", "0,55-0,59"],
         ["gadoids", "0,92-1,03", "0,53-0,64", "0,20-0,24", "3,2"],
         ["porpoises", "utdöd t 43-49", "0,34-0,40", "utdöd t 1 755+",
          "1,00-1,18"]],
        [22, 18, 18, 18, 18], size=11.5,
        align=["left", "right", "right", "right", "right"],
        kicker="Sista årets medel / spawn, 5 år, 3 seeds (avsnitt 169)",
        lead="Är beteendet bara \"gå mot föda, fly predatorer, ät när "
             "hungrig\"? De tränade policyerna slår varje handkodad arm för "
             "varje grupp.",
        footnote="Det är TRÖSKLARNA som avgör: när man slutar flytta och hur "
                 "hårt man betar. Bara tumlaren kan ersättas av en enkel "
                 "regel; djurplankton måste vila ~80 %, sill sluta resa "
                 "tidigare och torskfisk fortsätta söka efter sill.")

    d.bullets(
        "Energibudgetgrinden",
        ["Kan varje beslutande FG gå ihop energimässigt över sin RATION? "
         "Frågan besvaras utan rollout, på sekunder: "
         "python3 tools/probes/budget_gate.py",
         ("p", "ration  = taket 1/h vid full aptit   [ton byte / ton "
               "predator / tick]"),
         ("p", "krav    = resting_metabolism · feeding_cost / ration ≤ "
               "ransonens energikvalitet"),
         "Grinden utvärderas över hela menyn: motorn summerar intaget över "
         "alla byten och ger predatorn EN energivinst.",
         ("-", "I dag går alla fem aktiva predatorer ihop på vilken diet som "
               "helst. Djurplankton break-even vid 0,65 t/km² synligt "
               "växtplankton eller 0,28 mikrozooplankton; säl 0,14 t synlig "
               "torskfisk per cell."),
         ("-", "Sälens pelagiska par går inte ihop ensamt (62,9 t/cell): "
               "lågkvalitativ föda, avsiktligt."),
         ("-", "Grinden räknar vid t_ref - för sill en sommarkostnad.")],
        kicker="Går budgeten ihop?", size=15)

    d.table(
        "Planktonet mot A17",
        ["", "jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep",
         "okt", "nov", "dec"],
        [["djurplankton, modell", "0,27", "0,72", "2,04", "6,74", "18,3",
          "10,2", "6,25", "3,33", "2,08", "1,02", "0,30", "0,39"],
         ["djurplankton, A17-mål", "0,21", "0,18", "1,56", "19,3", "26,0",
          "11,4", "6,9", "4,6", "3,7", "2,9", "3,5", "0,92"],
         ["nitrat 0-10 m, A17", "5,1", "4,8", "2,3", "0,3", "0,1", "0,1",
          "0,1", "0,1", "0,1", "0,3", "1,3", "3,6"],
         ["klorofyll 0-10 m, A17", "1,21", "3,88", "3,28", "1,48", "0,94",
          "0,80", "0,46", "0,65", "0,79", "1,81", "1,54", "0,77"]],
        [24] + [6.3] * 12, size=10.5, align=["left"] + ["right"] * 12,
        kicker="t/km², µmol/l, µg/l",
        lead="fix68 generation 3, tre probeår (avsnitt 184.11). Målet är det "
             "residenta beståndet dec-mar (A17 minus Calanus och Metridia "
             "som stiger ur Norska rännan) och A17:s totala apr-nov.",
        footnote="Nitrat-logRMSE 0,54, r(P, klorofyll) +0,81, "
                 "mikrozooplankton 0,85 t/km². Maj-september stämmer; februari "
                 "är 4× hög, april 3× låg (A17:s april är Calanus) och "
                 "okt-dec 2-12× låg. Bruttoprimärproduktionen ~40 g C/m²/år mot "
                 "190-290 i Kattegatt.")

    d.table(
        "Realiserad mot budgeterad dödlighet",
        ["Storhet", "Budget / mål", "Realiserat", "Läsning"],
        [["Torskfisk: sälar", "0,12 /år", "0,17-0,19",
          "Sälarna tar mer torskfisk än budgeterat"],
         ["Torskfisk: tumlare", "0,056", "0,003-0,009",
          "Tumlarna tar nästan ingen torskfisk"],
         ["Torskfisk: fiske", "F 0,12", "0,10-0,115", "Som avsett"],
         ["Sill: torskfisk", "0,30", "0,26-0,31", "Som Ecopath (0,30)"],
         ["Torskfiskens diet, pelagisk andel", "26-39 % (Ecopath)",
          "62-76 %", "1,6-2,5× pelagisk ranson"],
         ["Sälens diet, pelagisk andel", "~10 %", "0,2-0,3 %",
          "Policyns ätkanal är mättad (Del 9)"],
         ["Djurplankton betat av sill", "1,9× stock/år (Ecopath)",
          "3,6× (med K), 14× (utan)", "K tog bort överbetningen"]],
        [28, 20, 20, 32], size=11,
        kicker="Avsnitt 184.15-19, 185.10-19",
        lead="Tillväxtbudgeten antar att varje predator tar sin M2. Sandlådan "
             "med frysta policyer och det regionala taket visar var det "
             "håller och var det inte gör det.",
        footnote="Realiserad M2 = årligt uttag / bytets medelbiomassa. "
                 "Policyerna var tränade före 185.20; läs om efter "
                 "omträning.")

    d.bullets(
        "Vad riggen och proberna är till för",
        ["Riggen är ett mätinstrument för mekaniken, inte en dom över "
         "världen. Med fryst beteende och fast seed är den deterministisk, "
         "så en biblioteksändring kan mätas isolerat utan träningsbrus.",
         "Den fångar fel som ingen policy kan rädda. Den säger inte att en "
         "värld är obrukbar: de tränade policyerna betar inte ned "
         "växtplanktonet som det giriga beteendet gör.",
         "Sandlådeproberna (phyto_bloom, pair_intake, hloss, gloss, "
         "sealscan) kör tränade policyer med krokar i motorn, så en "
         "mekanism kan prövas innan den byggs in - varje inbyggd mekanism "
         "har reproducerat sin probe inom ~1 %.",
         "Livskraft bör bedömas med de policyer som faktiskt används, på "
         "flera seeds och med ett stationaritetskrav som tål naturliga "
         "cykler. Det är inte byggt än."],
        kicker="Diagnos, inte dom")

    # ---------------------------------------------------------------- 9
    d.section("9", "Styrkor, svagheter och förbättringsförslag",
              "En samlad bedömning av modellen efter avsnitt 185: vad den "
              "gör bra, var den brister och vad som bör göras härnäst.")

    d.bullets(
        "Styrkor: mekanik och metod",
        ["Massa, energi och kväve bokförs slutet. Predation, rörelse, "
         "strömmar och lagerutbyte bevarar massa; tillväxten betalas ur "
         "reserven, inkomsten eller näringspoolen; varje förlust har en "
         "orsak.",
         "Tillväxten är spårbar: varje g härleds ur r_max, M1 och Ecopath-M2 "
         "vid s*, och varje r_max har en källa. Ett tak läggs bara där en "
         "mätt överskjutning visar att det behövs, och härleds ur dagens "
         "biomassa och fiske.",
         "Platsdata driver säsongen: ljusklimat, temperatur, blandningsdjup, "
         "nitrat och P-I-kurvor ur SMHI och SHARK A17; fisket ur EMODnet.",
         "Beteendet lärs, det hackas inte. Tränade policyer slår varje "
         "handkodad regel för varje grupp.",
         "Reproducerbart och testat: CRN, deterministiska probevärldar, "
         "CPU/GPU-paritet per steg och över 1 300 tester.",
         "Snabb diagnostik i tre steg: energibudgetgrinden på sekunder, "
         "fryst rigg och sandlådeprober på minuter."],
        kicker="Vad som bär", size=15)

    d.table(
        "Styrkor: beteende som liknar naturen",
        ["Fenomen", "I modellen", "Jämförelse"],
        [["Vårblomning och sommarminimum",
          "Växtplankton blommar i februari-mars, minimum på sommaren; "
          "nitratet tas upp under våren", "A17 klorofyll och nitrat, "
          "r(P, klorofyll) +0,8"],
         ["Djurplanktonets vår", "Startar i mars, topp i maj",
          "A17 residenta bestånd: mars 1,6, topp maj"],
         ["Sillens fettcykel", "s 0,35-0,41 i februari, 0,80 i maj-juni, "
          "ingen svält", "Mager Q1, fet Q3; ingen vintersvält hos vuxen sill"],
         ["Tillväxt när födan finns", "Sillen växer bara när den äter",
          "Somatisk tillväxt april - september"],
         ["Dygnsvandring", "Djurplankton vilar mer i ljus",
          "DVM där riktig natt finns"],
         ["Visuell planktonätare", "Sill äter i ljus, vilar på natten",
          "Sill äter vid gryning och skymning"],
         ["Säsongsbyte i dieten", "Torskfisk: sommar sill och djurplankton, "
          "vinter bottendjur", "Riktning rimlig, andel förhöjd"],
         ["Kontinuerligt födosök", "Tumlare äter 97-100 % av tiden",
          "Wisniewska m fl 2016"],
         ["Stabila bestånd med tak", "Torskfisk 0,63-0,66, sill stabil",
          "Bestånden ungefär stationära 2010-2020"]],
        [22, 42, 32], size=10.6, kicker="Emergent, inte inprogrammerat",
        footnote="Ur avsnitt 139, 156, 169, 177, 181-185. Inget av detta är "
                 "inskrivet i reglerna - det följer av kalendern, "
                 "energibudgeten och vad policyerna har lärt sig.")

    d.bullets(
        "Svagheter: biologi och struktur",
        ["Primärproduktionen är ~40 g C/m²/år mot 190-290 i Kattegatt; "
         "djuplagret som bär den är generöst kalibrerat (I_k 30 mot 63-117, "
         "sommarutbyte 0,2 mot 0,02-0,05 m/d) och blommar två månader för "
         "tidigt.",
         "Djurplanktonet är 2-12× för lågt i oktober-december och april är "
         "3× låg: modellen har ingen import av Calanus från Norska rännan.",
         "Sillens nedgång har följt torskfiskens uppbyggnad; med K har den "
         "flyttats till en jämvikt under dagens nivå (2,75 mot 3,33).",
         "Toppredatorernas dieter är anpassade, inte emergenta: tumlare tar "
         "nästan ingen torskfisk, sälar för mycket, och torskfiskens diet är "
         "2× för pelagisk.",
         "Sälar och tumlare saknar täthetsberoende. Modellens torskfisk "
         "(0,69 t/km²) är ~4× under IBTS-tätheten.",
         "Djupet är två lager för plankton men ett för fisk och "
         "däggdjur: säl och sill delar samma cell, så en låg "
         "attackhastighet får bära den vertikala separationen.",
         "Flera värden vilar på en källa eller är anpassade [D]: mörkerkvoter, "
         "ätbarhet, utbyten, κ."],
        kicker="Var modellen brister", size=14)

    d.bullets(
        "Svagheter: tillväxtbudget och tak",
        ["K är fenomenologiskt och härlett ur dagens biomassa: jämvikten är "
         "dagens läge per konstruktion, och en återhämtning bortom K kan "
         "inte uppstå i modellen.",
         "K och F gäller griden men står för processer på beståndsskala "
         "(rekrytering, migration). Ett lokalt scenario som stänger fiske "
         "kan överskatta effekten.",
         "s* är en övre gräns: den realiserade mättnaden är lägre, så "
         "grupperna producerar under budget när födan begränsar.",
         "Budgeten antar att varje M2 tas. När realiserad predation avviker "
         "(sälar, tumlare) flyttas bytets jämvikt, och ingen mekanism "
         "korrigerar det.",
         "Bottensamhället är handsatt (r, K 10 mot ~100), och "
         "mikrozooplanktonets M2 147 är en kvot - båda ligger utanför den "
         "härledda budgeten.",
         "r_max har ett brett intervall: torskfisk 0,29-0,42, pelagisk "
         "0,27-0,40. Jämvikten är känslig för r nära F."],
        kicker="Del 4 i kritisk läsning", size=14.5)

    d.bullets(
        "Svagheter: träning och utvärdering",
        ["Policyerna tränas på ettårsrollouts. Fleråriga trender ligger "
         "utanför träningsfördelningen.",
         "En gruppbelöning lär ut allmänningsdisciplin: djurplanktonet håller "
         "växtplanktonet nära K/2, vilket inga enskilda copepoder gör.",
         "ARS kan inte återuppliva en action som tryckts ned: sälens "
         "pelagiska ätkanal sitter på logit ~−8 och tränas inte tillbaka, "
         "även efter att attackhastigheten höjts.",
         "Generation-till-generation-spridningen med en utvärderingsseed och "
         "en värld är lika stor som många effekter som mäts.",
         "Varje biblioteksändring kräver omträning innan siffror citeras. "
         "Det pelagiska K (185.20) har ännu inte tränats in.",
         "Nätverket är litet (1 × 16) och utan minne; policyerna ser ljuset "
         "men inte datumet."],
        kicker="Var metoden brister", size=15)

    d.table(
        "Förbättringsförslag: tillväxt och tak",
        ["#", "Förslag", "Motiv", "Avsnitt"],
        [["1", "Träna om med båda K och läs jämvikten på flera seeds",
          "Inget K är ännu inträngt i policyerna", "185.19-20"],
         ["2", "Säsongsvis in- och utvandring för pelagisk fisk i stället "
          "för K", "Mekanistisk version av nettoemigrationen", "183.10, 185.18"],
         ["3", "Täthetsberoende tillväxt för säl (ungar, fruktsamhet)",
          "Knubbsälen ligger nära sin bärkraft", "168.5"],
         ["4", "K som funktion av habitat - bara som känslighetsfall",
          "Ingen publicerad funktion; dubbelräknar bentosvägen", "185.9"],
         ["5", "Bottensamhälle med r_max och litteraturens K",
          "Handsatt; torskfiskens M2 är verkningslös", "136, 168.1"],
         ["6", "Realiserad mättnad som referens i budgeten",
          "s* överskattar produktionen när födan begränsar", "160.3"]],
        [4, 44, 38, 14], size=11, kicker="Del 4",
        footnote="Varje förslag mäts först i sandlådan med tränade policyer "
                 "innan biblioteket ändras.")

    d.table(
        "Förbättringsförslag: modellen",
        ["#", "Förslag", "Motiv", "Avsnitt"],
        [["1", "Calanus-import som säsongskälla för djurplankton",
          "April och höst för låga; A17 får djupvattensdjur", "183.5-6"],
         ["2", "Kalibrera djuplagret mot litteraturen med återbildning i "
          "lagret", "I_k och utbyte generösa; våren för tidig", "181.8, 182.10"],
         ["3", "Remineraliseringspool (detritus/DOM)",
          "M1 hos plankton försvinner ur kvävekretsen", "179"],
         ["4", "Sälens och tumlarens dieter som mål vid omträning",
          "Realiserad M2 avviker från budgeten", "185.10-17"],
         ["5", "Lekkostnad för sill, diapaus för djurplankton",
          "Fettminimum och vinteröverlevnad", "143.6, 183"],
         ["6", "Temperaturskalad attackhastighet för fisk",
          "Nästa steg efter Q10 på metabolismen", "143.2"]],
        [4, 44, 38, 14], size=11, kicker="Biologi och mekanik",
        footnote="Ordningen är en bedömning av nytta mot kostnad.")

    d.table(
        "Förbättringsförslag: träning och utvärdering",
        ["#", "Förslag", "Motiv", "Avsnitt"],
        [["1", "Återställ mättade ätkanaler vid omträning (logitkopiering)",
          "ARS-gradienten kan inte lyfta en andel nära noll", "185.15"],
         ["2", "Flera utvärderingsseeds och världar per uppdatering",
          "Spridningen mellan generationer är stor", "184.11"],
         ["3", "Cykeltolerant livskraftskriterium på inferensdata",
          "Två-fönstertestet underkänner naturliga cykler", "126"],
         ["4", "Normkontroll i ARS; σ som utforskningsratt",
          "Slumpvandring i vikterna vid högt η/σ", "154.8-10"],
         ["5", "Flerårig utvärdering vid sidan av ettårsträningen",
          "Fleråriga trender optimeras aldrig", "156.2, 163"],
         ["6", "Aktivera vindkraftsscenarier först när nollscenariot är "
          "livskraftigt efter omträning", "En jämförelse kräver en "
          "självbärande bas", "-"]],
        [4, 44, 38, 14], size=11, kicker="Metod",
        footnote="Inget av förslagen kräver formande termer i belöningen.")

    # --------------------------------------------------------------- 10
    d.section("10", "Scenarier och körningar",
              "Vad ett scenario är, hur det körs och vad som skiljer den "
              "här utgåvan från maj-utgåvan.")

    d.bullets(
        "Scenario",
        ["Ett scenario är en uppsättning kartor för en viss uppsättning "
         "funktionsgrupper och en viss grid.",
         "Scenarier analyseras med uppdateringsreglerna och de tränade "
         "beteendemodellerna, och tolkas alltid som en JÄMFÖRELSE: "
         "nollalternativ mot projektalternativ, med samma seeds och samma "
         "policyer.",
         "Dagens fiske ingår i båda: bottentrål- och pelagiska kartor "
         "ligger som baskartor i träningen och som inferenskartor. Ett "
         "projektalternativ ersätter en karta - stängd park, ändrad "
         "ansträngning - aldrig staplar en andra impakt.",
         "Vindkraftens buller (observerbart) och rotor ligger muted. "
         "vindparker.npz bygger rotorkartan ur parkernas verkliga geometri.",
         "Projektkartan måste ange om parkernas trålansträngning tas bort "
         "eller flyttas inom griden: torskfiskens F i griden faller från "
         "0,12 till ~0,088 om den tas bort.",
         "Skillnaden ska rapporteras som relativ förändring per "
         "funktionsgrupp, under uttalade antaganden."],
        kicker="Definition och praktik", size=15)

    d.formula(
        "Körningar",
        [("Träning, inferens och grindar", """
python3 train.py --project mareld2.yaml --run-name <namn> < /dev/null
python3 train_gpu.py --project mareld2.yaml --run-name <namn> \\
        --policynetwork 1 16 tanh --mortality on --migration on ...
python3 inference.py --project mareld2.yaml --run-name <namn>
python3 tools/biomass_html.py results/<namn>                # plots.html
python3 tools/probes/budget_gate.py        # sekunder, ingen rollout
python3 tools/probes/daylight_table.py     # kalendern månad för månad
python3 tools/make_fishing_maps.py         # EMODnet -> maps/*.npz
python3 tools/viability.py                 # ~4 min, fryst beteende
python3 -m pytest tests/ -q
""")],
        tail=[("-", "Viktiga flaggor: --tick-length 1-6, --mortality on, "
                    "--migration on, --currents on, --local_reward, "
                    "--init-from, --resume, --visual, --rnd_baseline, "
                    "--eval-every."),
              ("-", "Dokumentet: python3 tools/make_strategi_deck2.py och "
                    "tools/make_strategy_deck2_en.py.")],
        kicker="Vad som körs")

    d.table(
        "Avsiktliga avvikelser från maj-utgåvan",
        ["Vad", "Varför"],
        [["Tillväxten är massbalanserad och growth_rate härleds vid s*",
          "Termen B·(1 + MG·q) skapade massa utan att debitera energi; "
          "budgeten vid full reserv nåddes aldrig. Nettot vid låg täthet "
          "blir litteraturens r_max."],
         ["Torskfisk och pelagisk fisk har ett regionalt K",
          "r_max är en Schaefer-r; utan K växte poolerna med r − F tills "
          "födan tog slut och kraschade."],
         ["Sillen växer ur inkomsten",
          "Fett blev biomassa efter att födan tagit slut; sillen svalt varje "
          "vinter."],
         ["Växtplanktonet har en näringspool och två lager",
          "Logistik mot ett konstant K gav ingen vårblomning och sommarens "
          "maximum i stället för minimum."],
         ["Hungern är kontinuerlig, inte en mask",
          "Full aptit upp till underhållsnivån; ätandet styrs av föda och "
          "magkapacitet."],
         ["Fitness räknar biomassa och reserv i samma valuta",
          "α och β var godtyckliga vikter mellan storheter som redan har en "
          "växelkurs."],
         ["Rollouterna är ett år och säsongen kommer från en kalender",
          "Kalendern kräver att en rollout rymmer årstiderna."]],
        [32, 68], size=10.8, kicker="Dokumentet mot koden")

    d.bullets(
        "Öppna frågor",
        ["Hur ser jämvikten ut när policyerna har tränats med båda K? "
         "Ligger pelagisk fisk nära 3,33 och torskfisk nära 0,69?",
         "Ska djurplanktonet få en Calanus-källa, och hur stor är den på en "
         "100 m bank utan diapaushabitat?",
         "Ska djuplagret kalibreras mot litteraturen, och vad håller då "
         "höstens djurplankton och sillen uppe?",
         "Hur får sälpolicyn tillbaka sin pelagiska ätkanal - "
         "logitåterställning, eller en annan parametrisering av ätandet?",
         "Behöver sälar och tumlare ett täthetsberoende, och ska det ligga "
         "på tillväxten (ungar) snarare än på vuxendödligheten?",
         "Vilken skala ska ett fiskescenario läsas i - grid eller bestånd?",
         ("p", "Hela historiken ligger i mareld_resume.txt, avsnitt 0-186.")],
        kicker="Vad som återstår", size=15)

    d.section("", "Tack",
              "Frågor, invändningar och rättelser hör hemma i "
              "mareld_resume.txt lika mycket som i koden.")


# ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default="Strategi_okt2.pdf")
    parser.add_argument("--png", default=None,
                        help="Also write one PNG per slide into this "
                             "directory (for proofreading).")
    args = parser.parse_args(argv)

    if args.png:
        os.makedirs(args.png, exist_ok=True)
    with PdfPages(args.out) as pdf:
        deck = Deck(pdf, args.png)
        build(deck)
        info = pdf.infodict()
        info["Title"] = "Strategi - för att bygga ekosystemsimulatorer"
        info["Author"] = "Mareld / Poseidon Nord"
        info["Subject"] = ("Strategi, oktober 2026, utgåva 2 - mekaniken "
                           "efter avsnitt 185 och den biologiska motiveringen "
                           "för tillväxttakt och bärkraftstak")
    print(f"{args.out}: {deck.n} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
