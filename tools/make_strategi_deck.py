"""Generate ``Strategi_okt.pdf`` - the October 2026 edition of the
strategy deck, written against the mechanics the code actually runs.

The original ``Strategi.pdf`` (May 2026) is a 37-slide 16:9 PowerPoint
export. This generator reproduces that format (960 x 540 pt pages) with
matplotlib so the deck can be regenerated whenever the model moves.

Everything stated on a slide is read from, or checked against:
  lib/environments/ecosystem_env/*.py   the tick
  lib/runners/trainer.py, parallel_worker.py, policy.py   ARS
  lib/world/tick_time.py, functional_group.py, energy_balance.py
  lib/world/growth_budget.py, daylight.py, temperature.py
  lib/gpu/   the tensor engine and trainer behind train_gpu.py
  fgconfig/fg_library.yaml, mareld2.yaml   the calibration
  VIABILITY.md, mareld_resume.txt sections 69-157

The two library tables, the pair table and the Observability matrix are
read from fgconfig/fg_library.yaml and mareld2.yaml at build time, so
they cannot drift from the calibration.

Slide text is Swedish (the original deck's language); parameter names
are kept in their English library spelling so a slide can be traced to
the YAML key it describes.

Usage:
    python3 tools/make_strategi_deck.py [-o Strategi_okt.pdf] [--png DIR]
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
    #: (tools/make_strategy_deck_en.py) overrides both.
    FOOTER = "Strategi - Mareld / Poseidon Nord - oktober 2026"
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
#: replaces the entries with ``LABELS.update(...)`` before it draws.
LABELS = {
    "pipeline": [
        ("1  Beslut", "observation → policy → maskad softmax"),
        ("2  Påverkansmortalitet", "impact_table → biomassa"),
        ("3  Predation", "Holling II/III + interferens"),
        ("4  Energikostnader", "vila / äta / röra sig, × Q10(T)"),
        ("5  Rörelse", "N/E/S/W × hastighet, energi följer med"),
        ("6  Strömmar", "valfritt: current_response"),
        ("7  Tillväxt / svält", "massbalanserad, + mortalitet, ljus för NDM"),
        ("8  Masker", "accessibility, utdöendetröskel"),
    ],
    "web_nodes": {
        "phytoplankton": "Växtplankton", "benthic_infauna": "Bottensamhälle",
        "zooplankton": "Djurplankton", "pelagic_fish": "Pelagisk fisk",
        "gadoids": "Torskfiskar", "porpoises": "Tumlare", "seals": "Sälar",
        "seabirds": "Sjöfåglar",
    },
    "web_legend": "grönt = icke beslutande (logistisk tillväxt)\n"
                  "blått = beslutande (policynätverk och energibudget)\n"
                  "streckat = muted i mareld2.yaml",
    "web_self": "kannibalism",
    "fr_herr": "pelagisk fisk  typ III, h = {h}",
    "fr_herr_w": "pelagisk fisk  w = 1, B_pred = 0,5",
    "fr_benth": "bottensamhälle  typ II, h = {h}",
    "fr_typeII_plain": "typ II",
    "fr_typeIII": "typ III (specialist)",
    "fr_ceiling": "tak 1/h",
    "fr_ceiling_e": "energitak 4 500/130",
    "fr_title_gad": "Torskfiskar: a = {a}, h ∝ bytets energi",
    "fr_title_zoo": "Djurplankton → växtplankton: a = {a}  h = {h}",
    "fr_xlabel": "synlig bytesbiomassa i cellen  [ton]",
    "fr_ylabel": "intag  [ton byte / ton predator / tick]",
    "fr_ylabel_e": "energiintag  [MJ / ton torskfisk / tick]",
    "hg_u5": "u = 0,5  (djurplankton, torskfisk, tumlare)",
    "hg_u2": "u = 0,2  (pelagisk fisk)",
    "hg_xlabel1": "mättnadsgrad  s_X = E_X / ME_X",
    "hg_ylabel1": "hungerfaktor  h_X",
    "hg_zoo": "djurplankton  g = {g} / SR = {sr}",
    "hg_pel": "pelagisk fisk  g = {g} / SR = {sr}  (×10)",
    "hg_xlabel2": "mättnadsgrad  s_X  (q_X = s_X − u_X)",
    "hg_ylabel2": "relativ biomassaändring / tick",
    "ic_title1": "windfarm_noise → tumlare: energy_factor",
    "ic_xlabel1": "kartvärde  (dB)",
    "ic_ylabel1": "extra metabolisk kostnad",
    "ic_title2": "rotor → sjöfåglar: biomass_factor",
    "ic_xlabel2": "kartvärde  (rotortäthet 0-1)",
    "ic_ylabel2": "biomassaförlust / tick",
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
        ("-", "m = antal aktiva impaktkartor märkta observable"),
        ("-", "B_obs är SYNLIG biomassa: den gömda (vilande) andelen "
              "dras bort"),
        ("-", "Mareld i dag: k = 1-4, m = 0 (windfarm_noise är muted) "
              "och en ljuskanal, dvs 12-27 kanaler"),
    ],
    "net_in": "observation\n(in_dim)",
    "net_h1": "dolt lager 1\n30 noder, sigmoid",
    "net_h2": "dolt lager 2\n30 noder, sigmoid",
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
    top = y - 8
    h = 34.0
    gap = 8.0
    w = PAGE_W - LEFT - RIGHT
    for i, (head, sub) in enumerate(steps):
        yy = top - i * (h + gap) - h
        face = BAND if i % 2 == 0 else BAND_2
        ax.add_patch(FancyBboxPatch((LEFT, yy), w, h,
                                    boxstyle="round,pad=0,rounding_size=5",
                                    facecolor=face, edgecolor="#D3DEE8",
                                    lw=1.0, zorder=1))
        ax.add_patch(Rectangle((LEFT, yy), 5, h, color=ACCENT, zorder=2))
        ax.text(LEFT + 20, yy + h / 2, head, color=NAVY, size=14,
                weight="bold", va="center", ha="left", zorder=3)
        ax.text(LEFT + 250, yy + h / 2, sub, color=INK, size=13,
                va="center", ha="left", zorder=3)
        if i < len(steps) - 1:
            arrow(ax, LEFT + 120, yy, LEFT + 120, yy - gap, lw=1.0, head=5)


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
_YAML_CACHE = {}

DM_ORDER = ["zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals",
            "seabirds"]
OBS_ORDER = ["phytoplankton", "zooplankton", "pelagic_fish", "gadoids",
             "porpoises", "seals", "seabirds", "benthic_infauna"]
PAIR_ORDER = [
    ("zooplankton", "phytoplankton"), ("pelagic_fish", "zooplankton"),
    ("gadoids", "pelagic_fish"), ("gadoids", "zooplankton"),
    ("gadoids", "benthic_infauna"), ("gadoids", "gadoids"),
    ("porpoises", "pelagic_fish"), ("porpoises", "gadoids"),
    ("seals", "pelagic_fish"), ("seals", "gadoids"),
    ("seabirds", "pelagic_fish"),
]


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


def species(name):
    return library()["species_definitions"][name]


def pair(pred, prey, rel="preys_on"):
    return library()["interaction_definitions"].get(
        f"{pred}_{rel}_{prey}") or {}


def muted_groups():
    """FG ids muted in mareld2.yaml."""
    m = _yaml("mareld2.yaml")
    out = set()
    for key in ("decision_makers", "non_decision_makers"):
        for entry in m.get(key) or []:
            if entry.get("muted"):
                out.add(entry["group_id"])
    return out


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
    III for a specialist (exactly one active prey) and II otherwise."""
    forced = pair(pred, prey).get("functional_response")
    if forced:
        return int(forced), True
    muted = muted_groups()
    n = sum(1 for (p, y) in PAIR_ORDER
            if p == pred and y not in muted and pair(p, y).get("preys_on"))
    return (3 if n == 1 else 2), False


def dm_value_rows(dec=None, sep=None, year="år", annual="årsmedel",
                  derived="härledd"):
    f = lambda x: fmt(x, dec, sep)
    rows = []

    def row(label, get):
        rows.append([label] + [get(species(n)) for n in DM_ORDER])

    row(f"r_max  (1/{year})", lambda s: f(s.get("r_max")))
    row(f"growth_rate  ({derived})", lambda s: f(s.get("growth_rate")))
    row("natural_mortality  M1", lambda s: f(s.get("natural_mortality")))
    row("maintenance_level  u", lambda s: f(s.get("maintenance_level")))
    row("max_energy_reserve", lambda s: f(s.get("max_energy_reserve")))
    row("energy_content", lambda s: f(s.get("energy_content")))
    row("resting_metabolism", lambda s: f(s.get("resting_metabolism")))
    row("feeding / movement_cost",
        lambda s: f"{f(s.get('feeding_cost'))} / {f(s.get('movement_cost'))}")
    row("metabolism_q10 / t_ref",
        lambda s: "-" if not s.get("metabolism_q10") else
        f"{f(s['metabolism_q10'])} / " + (
            annual if s.get("metabolism_t_ref") == "annual_mean"
            else f"{f(s.get('metabolism_t_ref'))} °C"))
    row("starve_rate", lambda s: f(s.get("starve_rate")))
    row("max_intake_rate  a", lambda s: f(s.get("max_intake_rate")))
    row("movement_speed  v", lambda s: f(s.get("movement_speed")))
    row("interference w / vis.floor",
        lambda s: f"{f(s.get('interference', 0))} / "
                  f"{f(s.get('visibility_floor', 0))}")
    row("min_split_biomass  (kg)", lambda s: f(s.get("min_split_biomass")))
    return rows


def pair_value_rows(dec=None, sep=None, prey_word="byte", type_word="typ",
                    forced_word="satt"):
    f = lambda x: fmt(x, dec, sep)
    muted = muted_groups()
    rows = []
    for pred, prey in PAIR_ORDER:
        p = pair(pred, prey)
        if not p.get("preys_on"):
            continue
        if "visibility_floor" in p:
            floor = f(p["visibility_floor"])
        else:
            floor = f"{prey_word} {f(species(prey).get('visibility_floor', 0))}"
        kind, forced = response_type(pred, prey)
        resp = f"{type_word} {'III' if kind == 3 else 'II'}" + (
            f" ({forced_word})" if forced else "")
        light = "-"
        if p.get("dark_ratio") is not None and p["dark_ratio"] < 1:
            light = (f"{f(p['dark_ratio'])} / "
                     f"{f(p.get('light_threshold_deg', -6))}°")
        name = f"{pred} → {prey}" + (" *" if pred in muted else "")
        rows.append([name, f(pair_attack(pred, prey)),
                     f(p.get("handling_time")), f(p.get("assimilation_factor")),
                     floor, resp, light, f(p.get("predation_mortality"))])
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


def _decimal_ticks(ax_):
    """Tick labels with the deck's decimal mark (comma in Swedish)."""
    from matplotlib.ticker import ScalarFormatter

    class DecimalFormatter(ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace(".", NUM["dec"])
    ax_.xaxis.set_major_formatter(DecimalFormatter())
    ax_.yaxis.set_major_formatter(DecimalFormatter())


def draw_food_web(fig, ax, y):
    names = LABELS["web_nodes"]
    nodes = {
        "phytoplankton": (210, 118, names["phytoplankton"]),
        "benthic_infauna": (720, 118, names["benthic_infauna"]),
        "zooplankton": (210, 205, names["zooplankton"]),
        "pelagic_fish": (300, 292, names["pelagic_fish"]),
        "gadoids": (650, 205, names["gadoids"]),
        "porpoises": (790, 292, names["porpoises"]),
        "seals": (520, 370, names["seals"]),
        "seabirds": (200, 370, names["seabirds"]),
    }
    muted = muted_groups()
    bw, bh = 142, 40
    for pred, prey in PAIR_ORDER:
        if pred == prey or not pair(pred, prey).get("preys_on"):
            continue
        xa, ya, _ = nodes[prey]
        xb, yb, _ = nodes[pred]
        x0, y0 = _rect_edge(xa, ya, bw, bh, xb, yb)
        x1, y1 = _rect_edge(xb, yb, bw, bh, xa, ya)
        arrow(ax, x0, y0, x1, y1, color="#8FA9BC", lw=1.4, head=7,
              ls="--" if pred in muted else "-")
    if pair("gadoids", "gadoids").get("preys_on"):
        gx, gy, _ = nodes["gadoids"]
        ax.annotate("", xy=(gx + bw / 2 + 2, gy - 8),
                    xytext=(gx + bw / 2 + 2, gy + 8),
                    arrowprops=dict(arrowstyle="-|>", color="#8FA9BC", lw=1.4,
                                    connectionstyle="arc3,rad=-1.6",
                                    shrinkA=0, shrinkB=0,
                                    mutation_scale=11), zorder=2)
        ax.text(gx + bw / 2 + 34, gy, LABELS["web_self"], color=MUTED,
                size=10.5, va="center", ha="left")
    for key, (x, yy, label) in nodes.items():
        ndm = key in ("phytoplankton", "benthic_infauna")
        box(ax, x - bw / 2, yy - bh / 2, bw, bh, label,
            face="#E4EEF3" if not ndm else "#E9F2E4",
            edge=ACCENT if not ndm else "#5C8A3C", size=12, weight="bold")
        if key in muted:
            ax.patches[-1].set_linestyle("--")
            ax.patches[-1].set_facecolor("#F4F7F9")
    # Centred in the free strip between the two NDM boxes, which sit on
    # the same row; at LEFT it overlapped the phytoplankton box.
    ax.text(465, 112, LABELS["web_legend"], color=MUTED, size=11, va="center",
            ha="center")


def draw_functional_response(fig, ax, y):
    # Left: the gadoid ceilings as an energy limit (section 148): h is
    # proportional to the prey's energy as eaten, so every pair saturates
    # at the same MJ per tonne of gadoid and tick.
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
    # Axes bottom at 200 pt: the x label hangs ~28 pt below the axes and
    # the bullet list below starts at 150 pt (Deck.figure items).
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

    a3 = pair_attack("zooplankton", "phytoplankton")
    h3 = pair("zooplankton", "phytoplankton")["handling_time"]
    Bz = np.linspace(0, 40, 400)
    f3 = a3 * Bz ** 2 / (1 + a3 * h3 * Bz ** 2)
    f2z = a3 * Bz / (1 + a3 * h3 * Bz)
    ax2 = axes_pt(fig, 520, 200, 360, 165)
    ax2.plot(Bz, f2z, color="#9FB6C6", lw=2.0, ls="--",
             label=LABELS["fr_typeII_plain"])
    ax2.plot(Bz, f3, color=ACCENT, lw=2.2, label=LABELS["fr_typeIII"])
    ax2.axhline(1 / h3, color=MUTED, lw=1.0, ls=":")
    ax2.text(39.5, 1 / h3 + 0.005, LABELS["fr_ceiling"], color=MUTED,
             size=9.5, ha="right")
    ax2.set_ylim(0, 1 / h3 * 1.2)
    ax2.set_title(LABELS["fr_title_zoo"].format(a=fmt(a3), h=fmt(h3)),
                  size=11, color=NAVY)
    ax2.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax2.set_ylabel(LABELS["fr_ylabel"], size=9)

    for a_ in (ax1, ax2):
        a_.set_xlabel(LABELS["fr_xlabel"], size=9.5)
        a_.tick_params(labelsize=8.5)
        _decimal_ticks(a_)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        a_.grid(alpha=0.25, lw=0.6)


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

    zoo, pel = species("zooplankton"), species("pelagic_fish")
    ax2 = axes_pt(fig, 520, 178, 360, 190)
    for sp_, k, label, color in (
            (zoo, 1.0, LABELS["hg_zoo"], ACCENT),
            (pel, 10.0, LABELS["hg_pel"], ACCENT_2)):
        u, g, sr = sp_["maintenance_level"], sp_["growth_rate"], \
            sp_["starve_rate"]
        q = s - u
        ax2.plot(s, k * np.where(q >= 0, g * q, sr * q), color=color, lw=2.2,
                 label=label.format(g=fmt(g), sr=fmt(sr)))
        ax2.axvline(u, color=color, lw=0.8, ls=":")
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_ylim(-0.11, 0.05)
    ax2.set_xlabel(LABELS["hg_xlabel2"], size=9.5)
    ax2.set_ylabel(LABELS["hg_ylabel2"], size=9.5)
    ax2.legend(fontsize=8.5, frameon=False, loc="lower right")

    for a_ in (ax1, ax2):
        a_.tick_params(labelsize=8.5)
        _decimal_ticks(a_)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.grid(alpha=0.25, lw=0.6)


def draw_impact_curves(fig, ax, y):
    # Axes bottom at 200 pt: the x label hangs ~28 pt below the axes and
    # the bullet list below starts at 150 pt (Deck.figure items).
    ax1 = axes_pt(fig, 92, 200, 360, 165)
    xs = np.array([60.0, 80.0, 100.0, 120.0])
    ef = np.array([0.0, 0.15, 0.40, 0.80])
    q = np.linspace(40, 140, 400)
    ax1.plot(q, np.interp(q, xs, ef, left=ef[0], right=ef[-1]),
             color=ACCENT, lw=2.2)
    ax1.plot(xs, ef, "o", color=ACCENT_2, ms=5.5, zorder=3)
    ax1.set_title(LABELS["ic_title1"], size=11,
                  color=NAVY)
    ax1.set_xlabel(LABELS["ic_xlabel1"], size=9.5)
    ax1.set_ylabel(LABELS["ic_ylabel1"], size=9.5)

    ax2 = axes_pt(fig, 520, 200, 360, 165)
    xs2 = np.array([0.0, 0.25, 0.5, 1.0])
    bf2 = np.array([0.0, 0.005, 0.015, 0.03])
    r = np.linspace(-0.1, 1.2, 400)
    ax2.plot(r, np.interp(r, xs2, bf2, left=bf2[0], right=bf2[-1]),
             color=ACCENT, lw=2.2)
    ax2.plot(xs2, bf2, "o", color=ACCENT_2, ms=5.5, zorder=3)
    ax2.set_title(LABELS["ic_title2"], size=11, color=NAVY)
    ax2.set_xlabel(LABELS["ic_xlabel2"], size=9.5)
    ax2.set_ylabel(LABELS["ic_ylabel2"], size=9.5)

    for a_ in (ax1, ax2):
        a_.tick_params(labelsize=8.5)
        _decimal_ticks(a_)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.grid(alpha=0.25, lw=0.6)


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
    x0, x1, x2, x3 = 150, 380, 560, 790
    def column(x, n, label, color, size=7.0):
        ys = np.linspace(200, 350, n)
        for yy in ys:
            ax.add_patch(plt.Circle((x, yy), size, facecolor=color,
                                    edgecolor="none", zorder=3))
        ax.text(x, 176, label, color=NAVY, size=12, weight="bold",
                ha="center", va="center")
        return ys
    a = column(x0, 7, LABELS["net_in"], ACCENT)
    b = column(x1, 6, LABELS["net_h1"], "#7FB2C0")
    c = column(x2, 6, LABELS["net_h2"], "#7FB2C0")
    d = column(x3, 5, LABELS["net_out"], ACCENT_2)
    for u in a:
        for v in b:
            ax.plot([x0, x1], [u, v], color="#C9D8E2", lw=0.4, zorder=1)
    for u in b:
        for v in c:
            ax.plot([x1, x2], [u, v], color="#C9D8E2", lw=0.4, zorder=1)
    for u in c:
        for v in d:
            ax.plot([x2, x3], [u, v], color="#C9D8E2", lw=0.4, zorder=1)
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
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    phy, ben = species("phytoplankton"), species("benthic_infauna")
    # ---------------------------------------------------------------- 1
    d.title_slide(
        "Strategi",
        "För att bygga ekosystemsimulatorer",
        ["Uppdaterad utgåva - oktober 2026",
         "Mareld / Poseidon Nord",
         "Beskriver den mekanik som simulatorn faktiskt kör i dag"])

    d.bullets(
        "Om den här utgåvan",
        [("p", "Den här versionen ersätter september-utgåvan, som i sin tur "
               "ersatte maj-utgåvan av Strategi.pdf. Strukturen är "
               "densamma; innehållet är skrivet mot koden som den ser ut i "
               "dag, och en ny del väger modellens styrkor mot dess "
               "svagheter."),
         "",
         "Koden är modellen. Dokumentet beskriver den - inte tvärtom. "
         "Där tidigare utgåvor och koden skiljer sig är det koden som "
         "gäller, och skillnaden redovisas i klartext.",
         "Varje formel och siffra är hämtad ur tickpipelinen i "
         "lib/environments/ecosystem_env/, ur ARS-tränarna i lib/runners/ "
         "och lib/gpu/, ur fg_library.yaml eller ur mareld2.yaml. "
         "Bibliotekstabellerna läses direkt ur YAML-filerna när "
         "dokumentet byggs.",
         "Parameternamn står kvar på engelska (growth_rate, handling_time, "
         "max_intake_rate) så att en slide kan spåras till den YAML-nyckel "
         "den beskriver.",
         ("-", "Strategin är fortfarande ett utvecklingsdokument, inte en "
               "kravspecifikation. Att koden stämmer med dokumentet är "
               "inget bevis för att en term är ekologiskt rätt."),
         ],
        kicker="Läsanvisning",
        footnote="Genereras av tools/make_strategi_deck.py. "
                 "Kör om skriptet när mekaniken ändras.")

    d.table(
        "Vad som har ändrats sedan maj 2026",
        ["Område", "Maj-utgåvan", "I dag"],
        [["Tillväxt", "B·(1 + MG·q) - ny massa var gratis",
          "Massbalanserad: tillväxten betalas ur energireserven; "
          "growth_rate härleds ur en litteraturbudget"],
         ["Predation", "Önskat intag = B·π·I·h, proportionell nedskalning",
          "Holling typ II/III per par med Beddington-DeAngelis-interferens "
          "och ljusberoende attackhastighet"],
         ["Hunger", "All predation maskades bort över s_X = 0,8",
          "Ingen mask - full aptit upp till underhållsnivån, sedan linjärt "
          "mot noll vid full reserv"],
         ["Vila", "Vila = låg energikostnad",
          "Vila är också en gömma-sig-action med per-par synlighetsgolv"],
         ["Fitness", "α·ΔB + β·ΔR över två transitioner",
          "Medelvärde över rollouten av log(total energi / startenergi)"],
         ["Träning", "Alternerande: en FG i taget",
          "Co-evolution på CPU (train.py) eller GPU (train_gpu.py), "
          "ettårsrollouts"],
         ["Ticklängd", "Fast 6 h", "--tick-length 1-6 h, biblioteket "
          "räknas om i minnet vid inläsning"],
         ["Nytt", "-", "Dagsljuskalender, vattentemperatur, strömmar, "
          "migration, utdöendetröskel, lokal belöning per cell, "
          "diagnostikverktyg"]],
        [10, 30, 46], kicker="Översikt", size=11.5,
        footnote="Del 9 listar de avvikelser från maj-utgåvan som är "
                 "avsiktliga och varför.")

    d.table(
        "Vad som har ändrats sedan september 2026",
        ["Område", "September", "Oktober", "Avsnitt"],
        [["growth_rate", "Fritt tal per FG",
          "Härleds ur r_max, M1 och predationsmatrisen (M2)", "129-131"],
         ["Hunger", "h = max(0, 1 − s/scale)",
          "h = min(1, (1 − s)/(1 − u)); satiation_scale borttagen",
          "130, 133"],
         ["Säsong", "Sinus för växtplankton (avstängd)",
          "Dagsljuskalender: ljusberoende attack och ljusbegränsad "
          "primärproduktion", "137-138"],
         ["Temperatur", "Ingen",
          "Q10 på viloförbränningen per FG och vattenlager", "143"],
         ["Sill", "En energipott, 7 500 MJ/ton",
          "Mager kropp 4 000 + fettreserv upp till 7 000; reserven äts med",
          "140"],
         ["Torskfiskar", "Två byten, typ II, tak 16 %/dygn",
          "Fyra byten, typ III på sill, tak som energigräns, M1 0,415/år",
          "141, 147-148, 157"],
         ["Tumlare", "Vila 90, odelbar enhet 250 kg",
          "Kostnader mot uppmätt FMR, diet anpassad till Ecopath, 50 kg",
          "145, 151-152"],
         ["Plankton", "Fri halvmättnad, 2 000 MJ/ton",
          "Halvmättnad 20 t/km², 4 800 MJ/ton, feeding_cost 4,0",
          "133-135"],
         ["Rörelse", "Rester under tröskeln sveptes bort",
          "Rester följer utflödet; små emigrationer stannar", "156"]],
        [11, 25, 50, 12], kicker="Översikt", size=10.8,
        footnote="Avsnittsnumren hänvisar till mareld_resume.txt, där "
                 "varje ändring är motiverad och mätt.")

    # ---------------------------------------------------------------- 2
    d.section("1", "Strategi och beståndsdelar",
              "Vad modellen är, vad den ska användas till och vilka delar "
              "den består av.")

    d.bullets(
        "Strategi",
        ["Ekosystem modelleras som en karta med flera kartlager som "
         "utvecklas över tid.",
         "Kartlagren beskriver miljöförhållanden, utbredning av "
         "populationer tillhörande olika funktionsgrupper, samt olika "
         "sorters miljöpåverkan. En kalender för ljus och temperatur "
         "driver säsongen.",
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
               "horisonten. Viabilitetsriggen, som kör med fryst "
               "beteende, är ett diagnosverktyg för mekaniken, inte "
               "domaren."),
         ],
        kicker="Vad modellen svarar på")

    d.bullets(
        "Beståndsdelar",
        ["En ekosystemmodell består av:",
         ("-", "en grid med längdskala och ticklängd"),
         ("-", "en uppsättning funktionsgrupper (FG), beslutande och icke "
               "beslutande"),
         ("-", "en uppsättning kartor - ett scenario"),
         ("-", "en kalender: dagsljus och vattentemperatur för platsen"),
         ("-", "en uppsättning egenskaper för varje funktionsgrupp och "
               "varje predationspar (fg_library.yaml)"),
         ("-", "en dynamisk modell med uppdateringsregler för kartorna - "
               "ticken"),
         ("-", "en beteendemodell för varje beslutande funktionsgrupp"),
         ("-", "spawnstrategier som bestämmer hur startbiomassan läggs ut"),
         ("-", "diagnostik: energibudgetgrind, fryst viabilitetsrigg och "
               "probeverktyg för årsbudgetar"),
         "Biblioteket är gemensamt för alla projekt. Projektfilen "
         "(mareld2.yaml) väljer FG, startbiomassor, impaktvariabler, "
         "gridstorlek och platsens ljus- och temperaturklimat."],
        kicker="Modellens delar", size=15.5)

    d.bullets(
        "Grid",
        ["Varje ekosystemmodell har en grid, dvs en matris av celler med:",
         ("-", "en dimension - Mareld kör 60 x 60 celler"),
         ("-", "en längdskala - en cellsida = 1 km"),
         ("-", "en tidskala - ett tick = 1-6 timmar, normalt 6"),
         ("-", "en position - 58,15° N, som sätter dagsljuset"),
         "Griden är plan och fyrgrannskopplad: all rörelse sker till N, E, "
         "S eller W, aldrig diagonalt.",
         "En valfri karta accessibility styr vilka celler som är "
         "tillgängliga. Den är gemensam för alla FG: rörelse in i en "
         "otillgänglig cell maskas bort, och biomassa som ändå hamnar där "
         "nollställs i slutet av ticket.",
         "Utan migration är gridens kant sluten. Med --migration on lämnar "
         "biomassan griden och återinförs koncentrerat längs kanten."],
        kicker="Geografi")

    d.table(
        "Ticklängd är en körningsparameter",
        ["Parametertyp", "Regel vid faktor k = ny/gammal", "Exempel"],
        [["flux - mängd som överförs per tick", "x · k",
          "max_intake_rate (även per par), resting_metabolism, "
          "movement_speed, seed_rate"],
         ["growth - andel som LÄGGS TILL ett lager", "(1 + x)ᵏ − 1",
          "growth_rate"],
         ["loss - andel som TAS BORT ur ett lager", "1 − (1 − x)ᵏ",
          "natural_mortality, starve_rate, biomass_factor i impacttabeller"],
         ["inverse - storhet vars invers är takten", "x / k",
          "handling_time (Hollingtaket 1/h ska hållas fast i realtid)"],
         ["tick-oberoende", "oförändrad",
          "max_energy_reserve, energy_content, maintenance_level, "
          "interference, min_split_biomass, kostnadsmultiplikatorerna, "
          "dark_ratio, light_threshold_deg, metabolism_q10"]],
        [22, 24, 40], kicker="--tick-length", size=11.5,
        lead="Motorn är ticklängdsagnostisk: varje bibliotekstal är uttryckt "
             "PER TICK. Ticklängden räknar om per-tick-takter till realtid "
             "- och skalar, när den inte är 6 h, om varje tickberoende "
             "parameter i minnet när FG:erna byggs. Ljus- och "
             "temperaturtabellerna byggs direkt för körningens ticklängd.",
        footnote="fg_library.yaml har EN kalibrering (6 h) och skrivs aldrig "
                 "om av en körning. lib/world/tick_time.py är enda "
                 "definitionspunkten.")

    d.table(
        "Funktionsgrupper",
        ["Funktionsgrupp", "Typ", "Meny", "Roll i modellen"],
        [["phytoplankton", "icke beslutande", "-",
          "Primärproducent, ljusbegränsad logistisk tillväxt mot bärkraft, "
          "driver med strömmen"],
         ["benthic_infauna", "icke beslutande", "-",
          "Bottenföda för torskfisk, mycket långsam tillväxt, "
          "förflyttas inte"],
         ["zooplankton", "beslutande", "phytoplankton",
          "Betare, specialist (typ III), rör sig 0,02 cell/tick"],
         ["pelagic_fish", "beslutande", "zooplankton",
          "Sill/skarpsill, specialist (typ III), mager kropp + fettreserv"],
         ["gadoids", "beslutande",
          "pelagic_fish, benthic_infauna, zooplankton, gadoids",
          "Generalist; typ III på sill, energitak, interferens 1,0"],
         ["porpoises", "beslutande", "pelagic_fish, gadoids",
          "Toppredator, endoterm, känslig för undervattensbuller"],
         ["seals", "beslutande", "pelagic_fish, gadoids",
          "Toppredator, koloni-spawn (muted)"],
         ["seabirds", "beslutande", "pelagic_fish",
          "Toppredator, enda FG med rotormortalitet (muted)"]],
        [16, 12, 26, 38], kicker="Mareld: åtta grupper, sex med policy",
        size=11.5,
        footnote="Engelska snake_case-id:n är kanoniska. seabirds och seals "
                 "är muted i mareld2.yaml - definierade men inte utlagda i "
                 "det aktuella scenariot.")

    d.figure("Näringsväv", draw_food_web, kicker="Meny enligt biblioteket",
             footnote="Pilen går från byte till konsument. Varje par har "
                      "egna värden för handling_time, assimilation_factor, "
                      "predation_mortality och - för vissa par - "
                      "attackhastighet, synlighetsgolv och ljusberoende.")

    # ---------------------------------------------------------------- 3
    d.section("2", "Kartor och kalender",
              "Allt tillstånd i modellen är kartor över griden. Ljus och "
              "temperatur kommer från en kalender för platsen.")

    d.bullets(
        "Kartor",
        ["En karta är en funktion från griden till de reella talen.",
         "En karta med värden i [0, 1] kan betraktas som en gråskalebild. "
         "Andra kartor - djup, biomassa, buller - visas som heatmaps.",
         "Modellen håller två tillståndskartor per funktionsgrupp: "
         "biomassa B_X(c) i ton och reservenergi R_X(c) i MJ.",
         "Energikartan E_X(c) = R_X(c) / B_X(c) är härledd, inte lagrad. "
         "Det är den formen som gör blandning vid rörelse trivial: "
         "två inflöden adderar sina R och sina B, och kvoten räknas om "
         "efteråt.",
         "Mättnadsgraden s_X(c) = E_X(c) / ME_X är den storhet som styr "
         "hunger, tillväxt och svält. Den ligger i [0, 1]."],
        kicker="Definition")

    d.table(
        "Karttyper och kalendertabeller",
        ["Typ", "Symbol", "Enhet", "Tolkning"],
        [["Biomassakartor", "B_X(c)", "ton/cell",
          "Biomassa av funktionsgrupp X i cell c."],
         ["Reservenergikartor", "R_X(c)", "MJ/cell",
          "Lagrad energi. Primärt tillstånd; klipps till B_X · ME_X."],
         ["Energikartor", "E_X(c)", "MJ/ton",
          "Härledd: R_X / B_X. Noll där biomassan är noll."],
         ["Bärkraftskartor", "CC_X(c)", "ton/cell",
          "Endast icke beslutande FG (max_carrying_capacity)."],
         ["Tillgänglighet", "accessibility(c)", "[0,1]",
          "Gemensam habitatmask. > 0 = cellen får bebos och nås."],
         ["Påverkanskartor", "Impact_p(c)", "godtycklig skala",
          "Buller i dB, trålningsintensitet, rotortäthet. Slås upp i "
          "FG:ns impact_table."],
         ["Strömfält", "-", "andel/tick",
          "Genereras proceduralt per tick ur brus, lagras inte."],
         ["Kalendertabeller", "F(t), m(t), P(t), T(t)", "per tick",
          "Ljus, attackmultiplikator, primärproduktionsfaktor och "
          "vattentemperatur. Samma för alla celler; byggs ur manifestet."]],
        [17, 16, 12, 45], kicker="Vad som finns i en körning", size=11.5,
        footnote="Geografiska kartor som djup och temperatur per cell och "
                 "administrativa kartor finns i strategin men läses inte av "
                 "motorn i dag; temperaturen är en tidsserie per vattenlager, "
                 "inte en karta.")

    d.bullets(
        "Dagsljuskalendern",
        ["Kalendern ger varje tick ett datum och en tid: årstick k börjar "
         "k · ticklängd timmar efter 1 januari 00:00 soltid. 8 760 h är "
         "delbart med varje ticklängd 1-6 h, så året går exakt jämnt ut.",
         "Solhöjden räknas ur latitud (58,15° N), deklination och soltid. "
         "Ljusnivån F(t) är medelvärdet av en logistisk funktion av "
         "solhöjden över tickets 10-minuterssteg - medel av f, inte f av "
         "medlet, eftersom f är starkt olinjär.",
         "Startdagen lottas per värld (start_day_of_year: random) och delas "
         "av alla delta-par och båda tecknen i den världen (CRN).",
         "Kalendern används på tre ställen:",
         ("-", "ljusberoende attackhastighet för visuella predatorer"),
         ("-", "ljusbegränsad tillväxt för växtplankton"),
         ("-", "månadstemperaturen som styr ämnesomsättningen"),
         "Varje beslutande FG ser F(t) som sista observationskanal. Att slå "
         "på eller av kalendern ändrar in_dim med 1 och gör checkpoints "
         "ogiltiga."],
        kicker="simulation_settings.daylight", size=15.5)

    d.table(
        "Kalendern månad för månad",
        ["", "jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep",
         "okt", "nov", "dec"],
        [["ljus F", "0,37", "0,44", "0,54", "0,66", "0,79", "0,87", "0,83",
          "0,70", "0,58", "0,47", "0,38", "0,34"],
         ["m sill → djurpl.", "0,70", "0,80", "0,94", "1,10", "1,30", "1,42",
          "1,36", "1,17", "0,99", "0,84", "0,72", "0,66"],
         ["m torsk → sill", "0,90", "0,93", "0,98", "1,04", "1,10", "1,13",
          "1,11", "1,06", "1,00", "0,95", "0,91", "0,88"],
         ["P/P_ref växtpl.", "0,16", "0,38", "0,72", "0,99", "1,12", "1,14",
          "1,03", "0,84", "0,58", "0,37", "0,15", "0,10"],
         ["T 0-50 m (°C)", "6,4", "5,5", "5,3", "6,2", "7,9", "9,9", "12,0",
          "13,3", "13,6", "12,6", "10,5", "8,4"],
         ["T 50 m (°C)", "7,8", "7,2", "6,7", "6,6", "6,8", "7,2", "8,0",
          "8,7", "10,2", "11,4", "10,6", "9,3"],
         ["Q10-faktor sill", "0,55", "0,51", "0,51", "0,54", "0,62", "0,73",
          "0,85", "0,95", "0,97", "0,90", "0,76", "0,64"],
         ["Q10-faktor torsk", "0,95", "0,90", "0,87", "0,86", "0,88", "0,91",
          "0,96", "1,01", "1,14", "1,26", "1,18", "1,06"],
         ["Q10-faktor djurpl.", "0,80", "0,75", "0,74", "0,79", "0,88",
          "1,02", "1,18", "1,29", "1,32", "1,23", "1,06", "0,92"]],
        [20] + [6] * 12, size=11.5, align=["left"] + ["right"] * 12,
        kicker="58,15° N, 6 h/tick, månadsmedel",
        footnote="Ur tools/probes/daylight_table.py. Attackmultiplikatorn m "
                 "har årsmedel exakt 1; växtplanktonfaktorn är 1 den 15 "
                 "april (dag 105) och har årsmedel 0,63; sillens Q10-faktor "
                 "är relativ en sommarkostnad vid 14 °C. Temperatur: SMHI "
                 "SHARK A16 + A17, 1991-2020.")

    d.bullets(
        "Vattentemperatur",
        ["Utan temperatur betalade varje växelvarm FG sin sommarkostnad "
         "hela året. Med simulation_settings.temperature betalar en "
         "beslutande FG med metabolism_q10:",
         ("p", "resting_metabolism · Q10^((T(t) − t_ref) / 10)"),
         "Alla actionkostnader är multiplar av resting_metabolism, så vila, "
         "äta och röra sig följer med. T(t) interpoleras cirkulärt ur tolv "
         "månadsmedel för FG:ns vattenlager: 0-50 m för sill och "
         "djurplankton, 50 m för torskfisk.",
         ("-", "Sill: Q10 2,2, t_ref 14 °C - bibliotekets värde är "
               "Rudstam/Rose-kostnaden vid sommartemperatur."),
         ("-", "Torskfisk 2,2 och djurplankton 2,0 med t_ref = årsmedel: "
               "faktorn har årsmedel 1."),
         ("-", "Tumlare: ingen - endoterm, fältmetabolismen är platt över "
               "0-20 °C."),
         "Attackhastigheterna skalas inte med temperaturen, och ingen "
         "observationskanal tillkommer: checkpoints förblir giltiga.",
         ("p", "Effekt i en probe med tränade policyer: sillens vintersvält försvann; "
               "reservfyllnaden i januari-februari hamnade på 0,31-0,32, "
               "inom litteraturens 0,28-0,45.")],
        kicker="Q10 på kalendern", size=15)

    # ---------------------------------------------------------------- 4
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
         ["Kostnadsfaktorer", "resting_cost, feeding_cost, movement_cost",
          "faktor", "Multiplikator på viloförbränningen per action."],
         ["Temperatur", "metabolism_q10, metabolism_t_ref", "-, °C",
          "Q10-faktor på viloförbränningen; t_ref i °C eller annual_mean."],
         ["Energiinnehåll", "energy_content", "MJ/ton",
          "Vad ett ton av FG:n är värt som föda - och priset på ett ton "
          "egen ny biomassa."],
         ["Reserven som föda", "prey_includes_reserve, "
          "reserve_reference_fill", "ja/nej, andel",
          "Bytets reserv äts med. Statisk kvalitet för grindarna vid "
          "referensfyllnaden."],
         ["Underhållsnivå", "maintenance_level (u_X)", "andel av ME",
          "Nollpunkt: q_X = s_X - u_X, och knät i hungerkurvan. 0,5 för "
          "alla beslutande FG utom sill (0,2)."],
         ["Energikänslighet", "impact_table.energy_factor", "faktor",
          "Extra metabolisk kostnad från påverkan p."]],
        [17, 25, 12, 41], size=11.0, kicker="Per funktionsgrupp")

    d.table(
        "Egenskaper: tillväxt, förluster och rörelse",
        ["Egenskap", "Nyckel", "Enhet", "Roll"],
        [["Maximal nettotillväxt", "r_max", "1/år",
          "Litteraturens nettotillväxt vid låg täthet. Grund för den "
          "härledda growth_rate."],
         ["Tillväxttakt", "growth_rate (MG_X)", "andel/tick",
          "HÄRLEDD av laddaren - redigeras aldrig direkt (utom "
          "bottensamhället)."],
         ["Naturlig mortalitet", "natural_mortality (M1)", "andel/tick",
          "Endast förluster som inte är predation från en FG. Kräver "
          "--mortality on."],
         ["Svälttakt", "starve_rate", "andel/tick",
          "Katabolism när q < 0."],
         ["Maximal bärkraft", "max_carrying_capacity", "ton/cell",
          "Endast icke beslutande FG."],
         ["Återkolonisering", "seed_rate", "andel av CC/tick",
          "Golv för icke beslutande FG; 0 för växtplankton."],
         ["Attackhastighet", "max_intake_rate (a)", "1/(ton·tick)",
          "Lutningen i det funktionella svaret; kan sättas per par."],
         ["Interferens", "interference (w)", "1/ton",
          "Beddington-DeAngelis: egen täthet sänker intaget."],
         ["Hastighet", "movement_speed (v)", "cellsidor/tick",
          "Sträcka per tick för den rörliga kohorten. Klipps till [0, 1]."],
         ["Minsta delbara massa", "min_split_biomass", "kg",
          "Under detta kollapsar actionfördelningen till argmax."],
         ["Utdöendetröskel", "extinction_threshold_factor", "faktor",
          "Cell nollställs under faktor · min_split_biomass."]],
        [17, 24, 14, 40], size=10.8, kicker="Per funktionsgrupp")

    d.table(
        "Egenskaper: predationspar och primärproducent",
        ["Egenskap", "Nyckel", "Enhet", "Roll"],
        [["Predationsmortalitet", "predation_mortality (M2)", "1/år",
          "Litteraturens förlust av bytet till predatorn. Ingår i bytets "
          "härledda tillväxt."],
         ["Attackhastighet per par", "max_intake_rate", "1/(ton·tick)",
          "Ersätter artens värde. Flyttar halvmättnaden 1/(a·h) utan att "
          "flytta taket."],
         ["Hanteringstid", "handling_time (h)", "tick",
          "Sätter taket 1/h ton byte per ton predator och tick."],
         ["Assimilation", "assimilation_factor", "andel",
          "Andel av bytets energi som predatorn behåller."],
         ["Synlighetsgolv", "visibility_floor", "[0,1]",
          "Andel som förblir synlig trots vila; annars bytets eget."],
         ["Funktionellt svar", "functional_response", "2 | 3",
          "Tvingar Holling typ II eller III för paret."],
         ["Ljusberoende", "dark_ratio, light_threshold_deg", "andel, °",
          "Attack i mörker relativt dagsljus, och solhöjden för halv "
          "detektion."],
         ["Ljusmättnad", "light_saturation (I_k)", "µmol/m²/s",
          "P-I-kurvans knä för växtplankton; ett värde eller tolv."],
         ["Referensdag", "light_reference_day", "dag",
          "Dagen då ljusfaktorn är exakt 1 (105 = mitten av april)."],
         ["Strömrespons", "current_response", "[0,1]",
          "Andel av strömfältet som FG:n följer."]],
        [19, 25, 12, 40], size=10.8, kicker="Per par och för producenter")

    d.formula(
        "Tillväxttakten härleds ur en litteraturbudget",
        [("lib/world/growth_budget.py - laddaren, FG-editorn och testerna "
          "räknar likadant", [
            r"g_X &= \dfrac{r_{\max,X} + M1_X + \sum_j M2_{Xj}}{(1 - u_X)\cdot 1460} \qquad \mathrm{beslutande\ FG}",
            r"r_X &= \dfrac{r_{\max,X} + \sum_j M2_{Xj}}{1460} \qquad \mathrm{icke\ beslutande\ FG}",
            ("t", "1 460 = antal 6-timmarstick per år. M2_Xj räknas när "
                  "paret är ikryssat, predatorn är beslutande och aktiv i "
                  "projektet."),
        ])],
        lead=[("p", "growth_rate är inte längre ett fritt tal. Tillväxten "
                    "måste betala för de förluster modellen själv tar ut, "
                    "så att nettotillväxten vid låg täthet blir "
                    "litteraturens r_max.")],
        tail=[("-", "M1 och M2 kommer från Nordsjöns Ecopath-modell för 1991 "
                    "(Mackinson & Daskalov 2007); r_max från FishBase/CMSY, "
                    "observerad återhämtning (säl, tumlare), Huntley & Lopez "
                    "(copepoder) och utspädningsexperiment (växtplankton)."),
              ("-", "Kryssas en predator ur i FG-editorn sjunker bytets g "
                    "med predatorns andel. Med --mortality off lämnar "
                    "laddaren M1 utanför g."),
              ("-", "Budgeten betalas fullt först vid s = 1, där "
                    "hungern stänger intaget. Det är en känd svaghet "
                    "(del 8). Bottensamhället är fortfarande handsatt.")],
        kicker="r_max, M1, M2")

    d.table(
        "Bibliotekets värden - beslutande funktionsgrupper",
        ["", "zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals *",
         "seabirds *"],
        dm_value_rows(),
        [22, 13, 13, 13, 13, 13, 13], size=10.2,
        align=["left", "right", "right", "right", "right", "right", "right"],
        kicker="fg_library.yaml, kalibrerat vid 6 h/tick",
        footnote="Läst ur fg_library.yaml när dokumentet byggs. Energier i "
                 "MJ/ton, takter per tick. * = muted i mareld2.yaml. "
                 "Sillens energy_content är den magra kroppen; reserven "
                 "kommer ovanpå.")

    d.table(
        "Bibliotekets värden - predationspar",
        ["Predator → byte", "a", "h", "assim.", "golv", "svar",
         "mörker / tröskel", "M2 (1/år)"],
        pair_value_rows(),
        [28, 8, 8, 9, 11, 13, 14, 11], size=10.8,
        align=["left", "right", "right", "right", "right", "left", "left",
               "right"],
        lead=f"Icke beslutande: phytoplankton växer med growth_rate "
             f"{f(phy['growth_rate'])} (aprilvärdet, ljusbegränsat) mot "
             f"max_carrying_capacity {f(phy['max_carrying_capacity'])} "
             f"ton/cell, {f(phy['energy_content'])} MJ/ton; "
             f"benthic_infauna med {f(ben['growth_rate'])} mot "
             f"{f(ben['max_carrying_capacity'])} ton/cell, "
             f"{f(ben['energy_content'])} MJ/ton, plus seed_rate "
             f"{f(ben['seed_rate'])}.",
        footnote="\"byte 0,2\" = paret ärver bytets eget visibility_floor. "
                 "\"satt\" = functional_response i biblioteket; annars får "
                 "en specialist typ III och en generalist typ II. "
                 "* = predatorn är muted.")

    d.bullets(
        "Sill: mager kropp och fettreserv",
        ["Sillen bar tidigare en enda energipott (7 500 MJ/ton). I "
         "tränade körningar svalt den ihjäl varje vinter, men ingen källa "
         "rapporterar vintersvält hos vuxen sill: verklig sill går ned "
         "till 4,4-6,5 kJ/g i första kvartalet och lever.",
         "Nu skiljs kroppen från reserven:",
         ("-", "energy_content 4 000 MJ/ton = mager kropp (≈ 4,1 kJ/g)"),
         ("-", "max_energy_reserve 7 000 MJ/ton ≈ 18 % fett"),
         ("-", "maintenance_level 0,2 - svält först när fettet nästan är slut"),
         ("-", "prey_includes_reserve: en predator får "
               "intag · assimilation · (4 000 + R/B i cellen)"),
         "Grindarna och FG-editorn räknar med reserven vid "
         "referensfyllnaden 0,5, dvs samma kvalitet 7 500 MJ/ton som förut, "
         "så tumlarens energibudget är oförändrad.",
         ("p", "Med delningen lönar sig vila vid låg föda - en sill kan lära "
               "sig fasta. I tioårskörningen följer sillen en fettcykel: "
               "s 0,23 i februari, 0,87 i juni.")],
        kicker="prey_includes_reserve", size=15)

    d.bullets(
        "Torskfiskar: fyra byten och ett energitak",
        ["Ecopath 1991 ger torskfiskarna 45 % av födan från byten modellen "
         "saknar. Med bara sill och bottendjur lade modellens torskfiskar "
         "dubbla Ecopath-trycket på sillen. Djurplankton (krillproxy) och "
         "kannibalism lades till; sillparet fick typ III, som ger glest "
         "byte en refug.",
         "Taket var 1/h = 16 % av kroppsvikten per dygn - mot 2-3,5 % för "
         "vuxen torsk. Hungern, inte magen, satte intaget. Magsäckens "
         "tömningstid skalar med bytets energitäthet, så taket sätts nu "
         "som en ENERGIgräns per par:",
         ("p", "h = 130 · E / 4 500   →   bottendjur 87, djurplankton 130, "
               "torskfisk 159, sill 217"),
         "Alla par mättar då vid samma MJ per ton torskfisk och tick.",
         "M1 sänktes 12 oktober från 0,683 till 0,415 per år: residual M0 "
         "0,20 enligt ICES SMS plus predation från grupper som inte är FG "
         "(0,214). Den gamla bottendelen kom av dubblerade rader i "
         "Ecopath-tabellen."],
        kicker="Avsnitt 141, 147-148, 157", size=15)

    d.bullets(
        "Tumlare: kostnader, diet och odelbar enhet",
        ["Kostnader mot uppmätt metabolism (Rojano-Donate m fl 2018: vilda "
         "tumlare 0,35 MJ/kg/dygn, fältmetabolism inklusive födosök):",
         ("-", "resting_metabolism 60 MJ/ton/tick = 0,24 MJ/kg/dygn"),
         ("-", "feeding_cost 1,5 → äta = 90 = den uppmätta "
               "fältmetabolismen; movement_cost 1,3"),
         "Diet: Ecopath ger ~49 % torskfisk i massa. Sillen mättade "
         "tumlaren, så attackhastigheten per par sattes till 0,015 på sill "
         "och 0,2 på torskfisk. Kvoten 13 är en anpassning till "
         "dietdata, inte en mekanism.",
         "Odelbar enhet: min_split_biomass 250 → 50 kg (ett djur), "
         "utdöendesvep vid 5 kg. Med 250 kg var varje tumlarcell argmax, "
         "vilket dolde policyns verkliga fördelning.",
         ("p", "Tränat på 50 kg: 44 % torskfisk i dieten, tumlarna sprids "
               "över 264 celler, flyttsannolikheten tränades ned från "
               "22 % till 7 %.")],
        kicker="Avsnitt 145, 151-152, 154", size=15)

    # ---------------------------------------------------------------- 5
    d.section("4", "Beteendemodeller",
              "Policynätverket, observationen och maskningen som gör "
              "utdatan ekologiskt meningsfull.")

    d.figure(
        "Policynätverk", draw_policy_net, kicker="Arkitektur",
        lead="En beteendemodell är ett litet fullt kopplat nätverk. "
             "Standard är två dolda lager om 30 noder med sigmoid och ett "
             "linjärt utsteg; --policynetwork LAGER NODER [AKTIVERING] "
             "ändrar det.",
        items=[("-", "Alla beslutande FG utvärderas i EN batchad "
                     "torch.bmm över hela griden - en policy ser alla sina "
                     "celler samtidigt."),
               ("-", "Utdatan är logits. Maskning sker FÖRE softmax, så en "
                     "omöjlig action får noll sannolikhetsmassa."),
               ("-", "Actionvektorn är [N, E, S, W, Vila, Ät Y_1 ... Y_n] "
                     "- en fördelning, inte ett val: varje cell delar sin "
                     "biomassa mellan alla actions.")])

    d.figure(
        "Observation", draw_observation, kicker="Vad en population ser",
        footnote="Ändras Observability-matrisen, de observerbara "
                 "impaktkartorna, FG-uppsättningen, gridlayouten eller "
                 "dagsljuskalendern blir befintliga checkpoints ogiltiga - "
                 "observationens bredd ändras.")

    d.table(
        "Observability-matrisen",
        ["Observatör", "phyto", "zoo", "pel.fisk", "torskfisk", "tumlare",
         "säl", "sjöfågel", "botten"],
        observability_rows("ja"),
        [20, 10, 10, 11, 12, 11, 9, 11, 10], size=11.5,
        align=["left"] + ["center"] * 8,
        lead="Matrisen bestämmer observationens bredd per FG: k_X = "
             "antal ja på raden utöver FG:n själv, räknat bara på aktiva "
             "FG (den egna biomassan och mättnadsgraden är alltid med). "
             "I dag k = 1 för djurplankton, 2 för tumlare, 3 för pelagisk "
             "fisk och 4 för torskfisk.",
        footnote="Läst ur fg_library.yaml. Torskfiskarna ser djurplankton "
                 "sedan avsnitt 141 (in_dim 22 → 27). * = muted. Matrisen "
                 "redigeras i FG-editorn.")

    d.bullets(
        "Maskning av actions",
        ["Predation maskas bort när bytet inte finns i cellen eller inte "
         "ligger på menyn. Ingenting annat maskar ätandet.",
         ("-", "Maj-utgåvan maskade bort all predation över s_X = 0,8. "
               "Den regeln finns inte i koden. Hungern är i stället en "
               "KONTINUERLIG faktor "
               "h_X = min(1, max(0, (1 − s_X)/(1 − u_X))) som skalar det "
               "önskade intaget."),
         "Rörelse i en riktning maskas bort när grannen är otillgänglig, "
         "när riktningen lämnar griden (utan --migration on) eller när "
         "FG:ns movement_speed är 0.",
         "Om varje action skulle vara maskad öppnas Vila, så att "
         "fördelningen alltid är definierad.",
         "Under min_split_biomass kollapsar fördelningen i cellen till "
         "argmax - en population som inte kan delas väljer en action i "
         "stället för att spridas ut i godtyckligt små delar.",
         "",
         ("p", "Tolkning: maskningen gör att modellen undviker orimliga "
               "val, förenklar träningen och låter utdatan följa modellens "
               "ekologiska regler.")],
        kicker="Före softmax")

    d.figure(
        "Hunger, underhåll och tillväxt", draw_hunger_gate,
        kicker="Två kurvor som styr allt",
        lead="Hungerfaktorn h_X är 1 upp till underhållsnivån u_X och faller "
             "sedan linjärt till 0 vid full reserv. Överskottsenergin "
             "q_X = s_X − u_X sätter tecknet på tillväxten. Båda kurvorna "
             "har sitt knä i u_X.",
        footnote="Under u_X äter en population med full aptit - ätandet styrs "
                 "av föda och magkapacitet, inte av reserven. "
                 "Energibudgetgrinden räknar därför på h(u_X) = 1. "
                 "Sillens tillväxtkurva är förstorad tio gånger.")

    d.bullets(
        "Vila är också att gömma sig",
        ["Den andel av en beslutande FG som väljer Vila i ett tick är "
         "skyddad från predation samma tick - och osynlig i nästa ticks "
         "observation.",
         ("-", "synlig andel = 1 - π_vila · (1 - golv)"),
         "Golvet är i första hand bytets eget visibility_floor, men kan "
         "sättas per PAR: detektionsförmåga tillhör paret, inte bytet. "
         "Tumlaren använder biosonar - därför golv 0,95 för tumlare → "
         "pelagisk fisk mot 0,25 för torskfisk → pelagisk fisk.",
         "Ljuset och gömslet är två olika saker: golvet säger hur väl "
         "gömslet fungerar, ljuset hur farligt det är att vara synlig. "
         "Vila stoppar också betningen - den verkliga kostnaden för "
         "dygnsvandring.",
         "Taket på uttaget ur en cell sätts av den BÄST detekterande "
         "predatorn, medan varje enskild predator är begränsad av vad den "
         "själv ser.",
         ("-", "En exponeringsviktad M1 (avsnitt 139: visuell del följer "
               "ljuset, taktil del drabbar de gömda) finns i motorn men är "
               "inte satt i biblioteket.")],
        kicker="visibility_floor")

    # ---------------------------------------------------------------- 6
    d.section("5", "Uppdateringsregler",
              "Ticken, steg för steg, med de formler motorn kör.")

    d.figure("Pseudokod för ett tick", draw_tick_pipeline,
             kicker="Åtta steg, i denna ordning",
             footnote="FG-ordningen slumpas varje tick, men inget steg beror "
                      "på den: i de loopade stegen rör varje FG bara sitt "
                      "eget tillstånd, och predation och rörelse är "
                      "vektoriserade - alla predatorer äter samtidigt ur "
                      "tillståndet vid t.")

    d.formula(
        "Steg 2: påverkansmortalitet",
        [("Uppslagning i FG:ns egen tabell", [
            r"(\mathrm{biomass\_factor}_p,\ \mathrm{energy\_factor}_p) = \mathrm{interp}\left(\mathrm{impact\_table},\ \mathrm{Impact}_p(c)\right)",
            "",
            r"m_X^{\mathrm{Impact}}(c) &= B_X(c)\,\sum_p \mathrm{biomass\_factor}_p(c)",
            r"B_X(c) &\leftarrow \max\left(0,\ B_X(c) - m_X^{\mathrm{Impact}}(c)\right)",
            r"R_X(c) &\leftarrow R_X(c)\,\dfrac{B_X^{\mathrm{ny}}(c)}{B_X^{\mathrm{gammal}}(c)}",
        ])],
        lead=[("p", "Påverkan tas ut FÖRE predationen, så att den "
                    "bytesbiomassa predatorerna ser redan är reducerad.")],
        tail=[("-", "Tabellen är en lista av (value, biomass_factor, "
                    "energy_factor). Mellan punkterna interpoleras linjärt; "
                    "utanför stödet används närmaste ändpunkt - ingen "
                    "extrapolation."),
              ("-", "energy_factor används inte här utan i steg 4, där den "
                    "höjer den metaboliska kostnaden med faktorn "
                    "(1 + Σ energy_factor)."),
              ("-", "Endast beslutande FG bär impacttabeller i den "
                    "nuvarande modellen.")],
        kicker="impact_table")

    d.figure(
        "Påverkanskurvor i biblioteket", draw_impact_curves,
        kicker="De tre aktiva tabellerna",
        items=[("-", "windfarm_noise → porpoises: ren energikostnad, "
                     "0 vid 60 dB, +80 % metabolism vid 120 dB."),
               ("-", "rotor → seabirds: ren mortalitet, upp till 3 % av "
                     "biomassan per tick vid full rotortäthet."),
               ("-", "pelagic_trawling → pelagic_fish: linjär från 0 till "
                     "100 % uttag vid kartvärde 1 000.")],
        footnote="Alla tre impaktvariabler ligger muted i mareld2.yaml - "
                 "de aktiveras när ett scenario ska köras.")

    d.formula(
        "Steg 3: predation",
        [("Önskat intag, effektiv attackhastighet och uttag", [
            r"B_Y^{\mathrm{synlig}}(c) &= B_Y(c)\,\left(1 - \pi_{Y,\mathrm{vila}}(c)\,(1 - \mathrm{golv}_{XY})\right) \qquad a_{XY}(t) = a_{XY}\,m_{XY}(t)",
            r"f_{XY}(c) &= \dfrac{a_{XY}(t)\,B_Y^{\mathrm{synlig}}}{1 + a_{XY}(t)\,h_{XY}\,B_Y^{\mathrm{synlig}} + w_X\,B_X(c)} \qquad (\mathrm{typ\ II})",
            r"D_{XY}(c) &= B_X(c)\,\pi_{X,\mathrm{ät}\,Y}(c)\,f_{XY}(c)\,h_X(c)",
            r"\mathrm{skala}(c) &= \min\left(1,\ 0{,}999\,B_Y^{\mathrm{synlig}}(c)\ /\ \Sigma_X\,D_{XY}(c)\right) \qquad \mathrm{intag}_{XY} = D_{XY}(c)\,\mathrm{skala}(c)",
        ]),
         ("Bokföring", [
            r"R_X(c) &\leftarrow R_X(c) + \sum_Y \mathrm{intag}_{XY}(c)\,\mathrm{assimilation}_{XY}\,\left(\mathrm{energy\_content}_Y + [R_Y/B_Y]\right)",
            r"B_Y(c) &\leftarrow B_Y(c) - \sum_X \mathrm{intag}_{XY}(c) \qquad R_Y\ \mathrm{skalas\ med\ samma\ kvot}",
        ])],
        tail=[("-", "Typ III ersätter B med B² i täljare och nämnare. Den "
                    "används av specialister och av par med "
                    "functional_response: 3."),
              ("-", "m_XY(t) är dagsljusmultiplikatorn (1 utan kalender); "
                    "[R_Y/B_Y] ingår bara för byten med "
                    "prey_includes_reserve, i dag sill."),
              ("-", "w_X · B_X(c) är Beddington-DeAngelis-interferens. "
                    "Faktorn 0,999 håller en överbetad cell borta från "
                    "exakt noll, som är absorberande.")],
        kicker="Holling + interferens + ljus")

    d.figure(
        "Funktionellt svar", draw_functional_response,
        kicker="Vad formeln gör",
        items=[("-", "Vänster: torskfiskarnas tak är en energigräns. "
                     "Sill (typ III, h 217) och bottendjur (typ II, h 87) "
                     "mättar vid samma MJ per ton torskfisk och tick."),
               ("-", "Höger: djurplankton på växtplankton, tak 1/h = 0,25. "
                     "Attackhastigheten per par sätter 1/(a·h) = 20 t/km²; "
                     "med typ III nås halva taket vid √20 ≈ 4,5 ton/cell."),
               ("-", "Interferensen w sänker hela kurvan när predatorerna "
                     "packar ihop sig - det tog bort de vertikala banden i "
                     "rörelsemönstret.")],
        footnote="Energin räknas som den äts: sillen med reserven vid "
                 "referensfyllnaden (4 000 + 0,5 · 7 000 MJ/ton).")

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
                    "0,001 lx, Batty m fl 1990). m: 0,66 i december, 1,42 i "
                    "juni - midnattssolen går bara till −8,5°."),
              ("-", "Torskfisk → sill: ρ = 0,6, θ = −5° (magsäcksdata från "
                    "fält: sill i morgon- och kvällsmagar). Övriga par "
                    "saknar mätningar och är ljusoberoende."),
              ("-", "Elevation, inte lux: synavståndet växer med log(ljus) "
                    "och mättar i skymningen. Moln ignoreras.")],
        kicker="dark_ratio, light_threshold_deg")

    d.formula(
        "Steg 4: energi efter action",
        [("Kostnad per action, före rörelse", [
            r"\mathrm{kostnad}_a(c) &= B_X(c)\,\pi_{X,a}(c)\,\mathrm{resting\_metabolism}_X\,Q_{10}^{(T(t) - t_{\mathrm{ref}})/10}\,\mathrm{cost}_a\,\left(1 + \sum_p \mathrm{energy\_factor}_p(c)\right)",
            r"\mathrm{cost}_a &\in \{\mathrm{resting\_cost},\ \mathrm{feeding\_cost},\ \mathrm{movement\_cost}\}",
            "",
            r"R_{\mathrm{vila}}(c) &= \max\left(0,\ R_X(c)\,\pi_{\mathrm{vila}} - \mathrm{kostnad}_{\mathrm{vila}}\right)",
            r"R_{\mathrm{ät}}(c) &= \max\left(0,\ R_X(c)\,\pi_{\mathrm{ät}} - \mathrm{kostnad}_{\mathrm{ät}}\right) + \mathrm{energivinst}(c)",
            r"R_{\mathrm{rör}}(c) &= \max\left(0,\ R_X(c)\,\pi_{\mathrm{rör}} - \mathrm{kostnad}_{\mathrm{rör}}\right)",
        ])],
        lead=[("p", "Reserven delas upp på de tre actiongrupperna i "
                    "proportion till hur biomassan valde, varje grupp "
                    "betalar sin kostnad, och ätgruppen får predationens "
                    "energivinst.")],
        tail=[("-", "Alla actions kostar. Vila är den billigaste, inte "
                    "gratis."),
              ("-", "Q10-faktorn är 1 för FG utan metabolism_q10 och utan "
                    "temperaturblock. Påverkan verkar multiplikativt: "
                    "buller gör varje action dyrare i stället för att döda."),
              ("-", "Efter rörelsen klipps reserven till [0, B_X · ME_X] "
                    "i varje cell.")],
        kicker="Metabolism")

    d.formula(
        "Steg 5: rörelse",
        [("Utflöde, kvarvarande massa och blandning", [
            r"\mathrm{ut}_d(c) &= B_X(c)\,\pi_{X,d}(c)\,v_X\,\mathrm{mask}_d(c)",
            r"\mathrm{kvar}(c) &= B_X(c) - \sum_d \mathrm{ut}_d(c)",
            r"B_X^{+}(c) &= \mathrm{kvar}(c) + \sum_d \mathrm{ut}_d\left(\mathrm{granne}_d^{-1}(c)\right)",
            "",
            ("t", "R följer med sin biomassa: flyttas 20 % av massan följer "
                  "20 % av den actiongruppens reserv med."),
            r"E_X &= \dfrac{R_X}{B_X} \qquad \mathrm{räknas\ om\ efter\ blandning;}\ B = 0\ \Rightarrow\ E = 0",
        ])],
        lead=[("p", "Rörelse skapar och förstör ingen biomassa. Den delar "
                    "bara upp cellens massa i utflöden och kvarvarande "
                    "massa.")],
        tail=[("-", "v_X är en sträcka per tick, inte en sannolikhet: den "
                    "flyttande kohortens tyngdpunkt förflyttas v celler på "
                    "ett tick. Därför skalas den linjärt med ticklängden."),
              ("-", "mask_d är accessibility och gridkanten. Blockerat "
                    "flöde stannar i källcellen; sannolikheterna "
                    "normaliseras aldrig om i efterhand."),
              ("-", "Rörelse är en ren omfördelning - all mortalitet har "
                    "redan tagits ut i steg 2 och 3.")],
        kicker="Slice-assign, fyra riktningar")

    d.bullets(
        "Rörelse: trösklar, migration och utdöende",
        ["Ett utflöde vars MÅLCELL ändå skulle hamna under "
         "utdöendetröskeln ställs in och återförs till källcellen.",
         "Nytt i oktober (avsnitt 156): en cell som efter detta själv ligger "
         "under tröskeln men fortfarande skickar biomassa till en granne "
         "lämnar ALL sin biomassa och reserv till grannarna, i proportion "
         "till utflödena. Och en emigration över kanten som är mindre än "
         "tröskeln stannar hemma.",
         ("-", "Motivet: rester och emigrationsdamm sveptes bort och "
               "bokfördes som svält - för tumlare hälften av den "
               "\"svältförlust\" som syntes."),
         "Med --migration on lämnar kantflödet griden och återinförs "
         "koncentrerat till de tyngst viktade kantcellerna, så att "
         "invandringen inte sprids ut under tröskeln.",
         "Sista steget i ticket nollställer celler där "
         "0 < B < extinction_threshold_factor · min_split_biomass och "
         "bokför massan som svältförlust. Motivet är numeriskt: under "
         "tröskeln närmar sig värdena float32:s subnormaler.",
         "Kontinuerliga FG - plankton, med min_split_biomass 0 - berörs "
         "inte av vare sig argmax-kollaps eller utdöendetröskel."],
        kicker="Diskreta populationer i ett kontinuerligt fält", size=15.5)

    d.bullets(
        "Strömmar",
        ["Med --currents on adderas en advektion efter rörelsen: ett "
         "molnlikt brusfält som skrollar mot öst och syd flyttar en andel "
         "av biomassan varje tick.",
         ("-", "Fältet samplas proceduralt ur tickräknaren och en "
               "världsseed - det lagras inte och förbrukar inte ekologins "
               "slumptal."),
         ("-", "--current-strength sätter den maximala andelen per tick "
               "(standard 0,1), --current-period hur snabbt fältet "
               "skrollar, --current-scale brusets skala i celler."),
         "Vilka grupper som följer med styrs av current_response i "
         "biblioteket, inte av om de är beslutande: en simmare som "
         "svarar advekteras ovanpå den förflyttning policyn redan valt.",
         ("-", "I dag: phytoplankton 1,0 och zooplankton 1,0, "
               "benthic_infauna 0,0, övriga 0,0 som standard."),
         "Strömmen delar rörelsens masker - blockerat flöde stannar kvar - "
         "och är konservativ: ingen massa skapas eller försvinner."],
        kicker="current_response")

    d.formula(
        "Steg 7: tillväxt för beslutande FG - massbalanserad",
        [("Önskan, tak och debitering", [
            r"s_X(c) &= \dfrac{E_X(c)}{\mathrm{ME}_X} \qquad q_X(c) = s_X(c) - u_X",
            r"\mathrm{önskan} &= B_X(c)\,\mathrm{MG}_X\,q_X(c) \qquad (q \geq 0)",
            r"E_{\mathrm{ledig}} &= \max\left(0,\ R_X(c) - u_X\,\mathrm{ME}_X\,B_X(c)\right)",
            r"\Delta B &= \min\left(\mathrm{önskan},\ \dfrac{E_{\mathrm{ledig}}}{\mathrm{energy\_content}_X}\right)",
            r"R_X(c) &\leftarrow R_X(c) - \Delta B\,\mathrm{energy\_content}_X \qquad B_X(c) \leftarrow B_X(c) + \Delta B",
        ]),
         ("Svält (q < 0) är en förlust, inte ett köp", [
            r"\Delta B &= B_X(c)\,\mathrm{starve\_rate}_X\,q_X(c) \qquad R\ \mathrm{skalas\ med}\ B",
        ])],
        lead=[("p", "Tillväxten är GRINDAD av mättnaden - B · MG · q är "
                    "önskan - men önskan är begränsad av den reserv som "
                    "står över underhållsnivån, och den reserven "
                    "DEBITERAS. MG_X är den härledda growth_rate.")],
        tail=[("-", "Utan debiteringen bär ny biomassa ett energiinnehåll "
                    "som aldrig tagits ur något: mätt till 198-344 % av "
                    "den massa som faktiskt åts, mot ett termodynamiskt "
                    "tak på 28,9 %."),
              ("-", "Konsekvens: ME_X / energy_content_X blir ett hårt tak "
                    "på den uppnåbara growth_rate. För djurplankton är "
                    f"kvoten {f(species('zooplankton')['max_energy_reserve'] / species('zooplankton')['energy_content'])} "
                    f"mot growth_rate {f(species('zooplankton')['growth_rate'])}."),
              ("-", "--no-mass-balance återställer maj-utgåvans term. "
                    "Biblioteket är INTE kalibrerat för den.")],
        kicker="--mass-balance, på sedan september")

    d.figure(
        "Energibokföringen i ett tick", draw_energy_flow,
        kicker="Var massan kommer ifrån",
        items=[("-", "Spillet är (1 − assimilation_factor) av intaget: "
                     "0,35 för djurplankton på växtplankton, 0,10 för "
                     "tumlare på pelagisk fisk."),
               ("-", "Reserven är den enda bufferten. Alla actions tar ur "
                     "den; temperatur och påverkan höjer kostnaden "
                     "multiplikativt."),
               ("-", "Först i tillväxtsteget blir reserv till biomassa, "
                     "till priset energy_content MJ per ton. En population "
                     "på eller under underhållsnivån kan därför inte växa "
                     "hur mycket föda som än finns.")],
        footnote="Fördelningen mellan andning, tillväxt och spill per FG går "
                 "att läsa ut tick för tick ur förlustredovisningen och "
                 "probeverktyget tools/probes/fg_year_budget.py.")

    d.bullets(
        "Förluster: fyra kanaler",
        ["Ticken bokför fyra förlustkanaler separat per FG: predation, "
         "svält, påverkan och naturlig mortalitet. "
         "lib/environments/ecosystem_env/loss_accounting.py är enda stället "
         "som gör andelar av dem; träningsloggen, inferensen, den levande "
         "visningen och plots.html använder samma funktion.",
         "Naturlig mortalitet (M1) är täthetsoberoende och valfri: den "
         "kräver --mortality on och skalas av --mortality_multiplier "
         "(keep = 1 - FAKTOR · natural_mortality). Den tas ut på både "
         "biomassa och reserv, före tillväxttermen. Före avsnitt 132 syntes "
         "den inte alls i förlustredovisningen.",
         "Svält är katabolism: under underhållsnivån krymper biomassan med "
         "starve_rate · |q|, och reserven skalas med samma kvot.",
         "Massa som nollställs av utdöendetröskeln bokförs som svält, och "
         "antalet händelser räknas separat.",
         "Mortalitet genom påverkan modelleras med påverkanskartor: fiske, "
         "jakt, kemikalier eller mekanisk dödlighet."],
        kicker="Predation, svält, påverkan, naturlig", size=15.5)

    d.formula(
        "Steg 7: tillväxt för icke beslutande FG",
        [("Ljusbegränsad logistisk tillväxt mot bärkraft", [
            r"r(t) &= \mathrm{growth\_rate}\,\dfrac{P(t)}{P_{\mathrm{ref}}} \qquad P(t) = \mathrm{medel}_{\mathrm{tick},\,z}\ \tanh\left(\dfrac{I_0(t)\,e^{-K_d z}}{I_k}\right)",
            r"\Delta B(c) &= r(t)\,B(c)\,\left(1 - \dfrac{B(c)}{\mathrm{CC}(c)}\right) + \mathrm{seed\_rate}\,\mathrm{CC}(c)\,u \qquad u = 10^{U(-1,\,1)}",
            r"B(c,t+1) &= \mathrm{klipp}\left(B(c) + \Delta B(c),\ 0,\ \mathrm{CC}(c)\right)",
        ])],
        lead=[("p", "Gäller funktionsgrupper vars energi inte modelleras. "
                    "Betet har redan tagits ut i steg 3. Den gamla "
                    "sinustermen är borttagen; säsongen kommer från "
                    "kalendern.")],
        tail=[("-", "I_0: klarhimmelsinstrålning (Haurwitz) × 0,45 PAR × "
                    "4,57 µmol/J × månatlig molntransmission (SMHI "
                    "Nordkoster). K_d 0,29/m; blandningsdjup 6-17 m (SHARK "
                    "A17); I_k 111-298 µmol/m²/s per månad (SMHI 14C, "
                    "Anholt E). P = 0 på natten."),
              ("-", f"growth_rate {f(phy['growth_rate'])} per 6 h är den "
                    "högsta in situ-takten i utspädningsexperiment i april "
                    "minus mikrozooplanktonbetning; P_ref är dag 105, så "
                    "faktorn är 1 i mitten av april och 0,63 i årsmedel."),
              ("-", "Bottensamhället är inte ljusbegränsat: growth_rate "
                    f"{f(ben['growth_rate'])}, plus seed_rate som "
                    "återkoloniseringsgolv.")],
        kicker="light_saturation, light_reference_day")

    d.bullets(
        "Sammanfattning av bokföringen",
        ["Ny biomassa = biomassa efter påverkan, predation och förflyttning "
         "+ tillväxt - svält - naturlig mortalitet.",
         "Biomassa kan bara öka när energireserven tillåter det. För icke "
         "beslutande FG är bärkraften taket och ljuset takten; för "
         "beslutande FG är reserven över underhållsnivån taket.",
         "Reservenergin följer alltid sin biomassa: varje gång biomassan "
         "skalas ned skalas reserven med samma kvot, och varje gång massa "
         "flyttas följer motsvarande andel av reserven med. Hos sill följer "
         "reserven dessutom med in i predatorn.",
         "Det som INTE bevaras: naturlig mortalitet, påverkan och "
         "utdöendetröskeln tar massa ur systemet, och migration flyttar "
         "massa över gridens rand. Predation och rörelse bevarar den.",
         ("-", "Med --migration off och utan mortalitet är "
               "källspårningens massbalans en identitet, vilket är vad som "
               "gör den lokala belöningen exakt.")],
        kicker="Vad som bevaras och inte")

    # ---------------------------------------------------------------- 7
    d.section("6", "Träning med ARS",
              "Hur beteendemodellerna tas fram, vad de optimerar och vad "
              "de inte ska förväxlas med.")

    d.bullets(
        "Träning: mål och perspektiv",
        ["Målet med träningen är inte att hitta ekologiskt optimala "
         "policyer, utan plausibla, stabila och testbara policynätverk som "
         "kan användas i scenarioanalys.",
         "Modellen är Eulerisk. Det finns ingen individ eller tydlig "
         "population att följa i flera steg - biomassan delas upp, flyttar "
         "åt olika håll och blandas med annan biomassa. Jämför människor "
         "på ett torg.",
         "Därför optimeras ingen långsiktig individuell avkastning, utan "
         "hur väl ett FÄLT klarar sig över en rollout.",
         "Det som körs är ARS: gradientfritt och parallelliserat, på CPU "
         "(train.py) eller i tensorform på GPU (train_gpu.py).",
         "Rollouten är 15 tick som standard och 100-200 tick i profilerna. "
         "Med kalendern på tränas i praktiken på ettårsrollouts, 1 460 "
         "tick, så att varje rollout innehåller alla årstider."],
        kicker="Vad ARS ska åstadkomma")

    d.formula(
        "Fitness",
        [("Standard: total energi, integrerad över rollouten", [
            r"E_{\mathrm{tot}}(t) &= \sum_c \left(B_X(c,t)\,\mathrm{energy\_content}_X + R_X(c,t)\right)",
            r"\mathrm{Fitness}_X &= \dfrac{1}{H}\sum_{t=1}^{H}\,\log\dfrac{E_{\mathrm{tot}}(t) + \varepsilon}{E_{\mathrm{tot}}(0) + \varepsilon}",
            r"\varepsilon &= \max\left(10^{-6}\,E_{\mathrm{tot}}(0),\ 10^{-9}\right)",
            ("t", "H = --n_eval_ticks"),
        ]),
         ("Kvar som --legacyreward: maj-utgåvans form", [
            r"\mathrm{Fitness}_X &= \alpha\,\log\dfrac{B_H}{B_0} + \beta\,\log\dfrac{R_H}{R_0} + \kappa\,\dfrac{t_{\mathrm{överlevd}}}{H}",
        ])],
        tail=[("-", "Att räkna biomassa och reserv i SAMMA valuta tar bort "
                    "viktningen mellan α och β: ett ton biomassa är värt "
                    "exakt sitt energiinnehåll."),
              ("-", "Integralformen (--integral_reward, på som standard) "
                    "ger lägre betyg åt en policy som kraschar bytet mitt "
                    "i rollouten och återhämtar sig på slutet."),
              ("-", "Logaritmen gör belöningen skalinvariant och "
                    "symmetrisk: en halvering kostar exakt vad en "
                    "fördubbling är värd.")],
        kicker="Vad en rollout mäter")

    d.formula(
        "Lokal belöning per cell",
        [("--local_reward: samma fråga, ställd per cell", [
            r"A(c,t) &= B(c,t)\,\mathrm{energy\_content} + R(c,t)",
            r"B(c,t+1) &= \sum_d \mathrm{andel}_d(c)\,Q\left(\mathrm{mål}(c,d),\ t+1\right)",
            r"\mathrm{belöning}(c) &= \mathrm{klipp}\left(\dfrac{B(c,t+1)}{A(c,t)},\ 0{,}2,\ 5{,}0\right)",
            r"\mathrm{Fitness}_X &= \mathrm{medel}_t\ \dfrac{1}{N_{\mathrm{celler}}}\sum_c w_c\,\log\,\mathrm{belöning}(c)",
        ])],
        lead=[("p", "Den globala belöningen domineras av de stora "
                    "cellerna: en policy som sköter en tusentonscell bra "
                    "och ett kilo dåligt får samma betyg som tvärtom.")],
        tail=[("-", "andel_d(c) är den andel av målcellens flyttade "
                    "biomassa som kom från c - en identitet, inte en "
                    "approximation, eftersom allt efter rörelsen är "
                    "cellvis multiplikativt."),
              ("-", "Använd --local_reward_norm grid med metriken log: "
                    "sum belönar tunn spridning och mean belönar att döda "
                    "de sämsta cellerna."),
              ("-", "w_c = A_cᶿ. θ = 0 väger varje bebodd cell lika, "
                    "θ = 1 återger den globala energibelöningen.")],
        kicker="Skalinvariant per cell")

    d.formula(
        "ARS-uppdateringen",
        [("Per iteration, per beslutande FG med vikter θ", [
            r"\delta_1, \ldots, \delta_N &\sim \mathcal{N}(0,\ I)",
            r"F_i^{+} &= \mathrm{fitness}(\theta + \sigma\,\delta_i) \qquad F_i^{-} = \mathrm{fitness}(\theta - \sigma\,\delta_i)",
            ("t", "N = --n_deltas. Samma värld, seed och kalenderstart för + "
                  "och − (CRN)."),
            r"\mathrm{topp}\ b &= \mathrm{de}\ b\ \mathrm{par\ med\ högst}\ \max(F_i^{+},\ F_i^{-}) \qquad b = N/2",
            r"\sigma_F &= \mathrm{standardavvikelsen\ över\ de\ utvalda\ parens\ fitness}",
            r"\theta &\leftarrow \theta + \dfrac{\eta}{b\,\sigma_F}\sum_{i \in \mathrm{topp}\,b}\left(F_i^{+} - F_i^{-}\right)\delta_i",
        ])],
        tail=[("-", "ARS-V2: observationerna normaliseras med ett löpande "
                    "medel och en löpande varians per FG och klipps till "
                    "±10."),
              ("-", "CRN - gemensamma slumptal - gör skillnaden F⁺ − F⁻ "
                    "meningsfull vid få rollouts: båda halvorna möter "
                    "identiska startfält och samma årstid."),
              ("-", "Standard: η = 0,03, σ = 0,1, N = 10, en värld per "
                    "delta. Normeringen med σ_F gör steglängden oberoende "
                    "av belöningens skala: |steg| / probradie ≈ "
                    "(η/σ) · k/√b.")],
        kicker="Augmented Random Search")

    d.bullets(
        "Co-evolution",
        ["Standardläget är co-evolution: ALLA beslutande FG perturberas "
         "samtidigt inom samma delta-par och utvärderas i EN gemensam "
         "rollout. Varje FG får sin egen fitness ur samma simulering, och "
         "ARS-uppdateringen görs sedan oberoende per FG.",
         ("-", "Motivet är att en bytespolicy och en predatorpolicy bara "
               "är meningsfulla mot varandra. Tränas de var för sig "
               "optimerar var och en mot en motpart som inte längre finns "
               "när båda är klara."),
         "--no_coevolution ger det alternerande schemat från maj-utgåvan: "
         "en FG i taget, övriga frysta.",
         "Policyer som inte tränas i iterationen - och icke beslutande FG "
         "- körs med sina senaste vikter som fast bakgrund.",
         "En generation = --iter-per-gen iterationer. Efter varje "
         "generation sparas checkpoints och en deterministisk probe-rollout "
         "loggar biomassan per FG till results/<run>/biomass.jsonl."],
        kicker="Alla arter samtidigt")

    d.bullets(
        "GPU-träning: train_gpu.py",
        ["lib/gpu/ är en tensorversion av samma tick: alla världar, delta-par "
         "och tecken körs samtidigt på grafikkortet, i block av tick som "
         "spelas upp som CUDA-grafer. Paritetstester jämför varje steg mot "
         "CPU-motorn.",
         "Aktuella körningar (fix53): 16 delta-par × 2 världar × 1 460 "
         "tick, 60 × 60, mortalitet, migration, strömmar, dagsljus och "
         "temperatur på.",
         "Kalenderstarten dras per VÄRLD och delas av alla par (avsnitt "
         "150). När varje par hade sin egen årstid sattes parens "
         "belöningsnivå av säsongen: topp-b-urvalet blev nästan slumpmässigt "
         "och steget krympte till en tredjedel.",
         "Steglängd: vid η = 0,1 och σ = 0,05 blev uppdateringen "
         "brusdominerad - vikternas norm växte som en slumpvandring "
         "(29-35 → 64-75) och policyerna skärptes mer än belöningen "
         "motiverar. Håll η/σ ≲ 0,6; σ är ratten för mer utforskning.",
         ("-", "Visningens probe läser om fg_library.yaml vid varje probe, "
               "men tränaren behåller biblioteket den startade med.")],
        kicker="lib/gpu/", size=15)

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
        lead="Entropibonus, argmax-straff och temperaturhärdning är "
             "formande termer som verkar på actionfördelningen i stället "
             "för på biologin. De finns kvar som flaggor men är avstängda "
             "som standard, med eller utan profil, på båda tränarna.",
        footnote="--profile sanity | info | deep. Explicita flaggor på "
                 "kommandoraden vinner alltid över profilens värden.")

    d.figure(
        "Träningsvärldar", draw_spawn_examples,
        kicker="Fyra spawnstrategier",
        lead="För att undvika överanpassning till en enskild karta tränas "
             "policyerna i många slumpade världar. Varje FG har en egen "
             "spawnstrategi i biblioteket och ett intervall för "
             "startbiomassa i projektfilen.",
        items=[("-", "uniform: oberoende slumpvikt per cell, ingen "
                     "rumslig struktur. perlin: naturlikt brus "
                     "(växtplankton, trålningsintensitet)."),
               ("-", "colony: n gaussiska kolonier med jittrad amplitud "
                     "(fisk, sälar, sjöfåglar, bullerkällor)."),
               ("-", "env_driven: vikten byggs ur andra fält - "
                     "djurplankton följer växtplankton, tumlare följer "
                     "pelagisk fisk.")],
        footnote="Startbiomassan lottas per värld ur "
                 "[initial_biomass_min, initial_biomass_max] i "
                 "mareld2.yaml, startdagen ur kalendern. I TRÄNINGSläge "
                 "lottas energinivån per cell kring underhållsnivån; "
                 "inferensvärlden startar deterministiskt på 0,7 · ME_X.")

    d.bullets(
        "Checkpoints",
        ["En checkpoint är results/<körning>/policy_<fg>.pth och innehåller "
         "{'state_dict': ..., 'obs_stats': {...}} - vikterna OCH "
         "observationsnormaliseringens löpande statistik.",
         "--resume läser in dem och fortsätter; inference.py kör dem mot "
         "en deterministisk värld.",
         "Följande ändringar gör befintliga checkpoints ogiltiga - "
         "observationens form eller betydelse ändras:",
         ("-", "Observability-matrisen (t ex torskfisk ser djurplankton, "
               "avsnitt 141)"),
         ("-", "vilka impaktkartor som är observable och aktiva"),
         ("-", "FG-uppsättningen i projektet och gridlayouten"),
         ("-", "dagsljuskalendern på eller av (ljuskanalen)"),
         "Temperaturen och en ändrad kalibrering gör dem inte formellt "
         "ogiltiga, men en policy som tränats i en väsentligt annan värld "
         "bör tränas om innan den citeras. Kopiera en checkpoint innan den "
         "jämförs - en pågående körning skriver över policy_*.pth."],
        kicker="Vad som sparas och vad som bryter det", size=15.5)

    # ---------------------------------------------------------------- 8
    d.section("7", "Diagnostik och livskraft",
              "Verktyg som testar mekaniken utan ARS, och vad tränade "
              "policyer gör över tio år.")

    d.table(
        "Viabilitetsriggen",
        ["#", "Villkor", "Formel", "Standard"],
        [["1", "Överlevnad", "B_f(t) > 0 för alla t ≤ T", "-"],
         ["2", "Golv", "eq_f ≥ golv · B_f(0)", "golv = 0,10"],
         ["3", "Tak", "eq_f ≤ tak · B_f(0)", "tak = 10"],
         ["4", "Stationaritet",
          "max(eq_f/föreg_f, föreg_f/eq_f) ≤ max_drift", "max_drift = 2,0"]],
        [6, 26, 44, 24], size=12.5, kicker="Fryst beteende",
        lead="Riggen kör världen med FRYST, handkodat beteende. Det gör den "
             "snabb och deterministisk, och därmed bra för att mäta en "
             "parameter- eller mekanikändring isolerat. eq_f är "
             "medelbiomassan i horisontens sista fönster.",
        footnote="Horisont 5 år (1 460 tick vid 6 h), 3 seeds, sista "
                 "fönstret 10 % av horisonten; standardhörnet är --spawn "
                 "colocated --behaviour greedy. Utslaget gäller DET "
                 "beteendet: alla äter maximalt, vilket betar ned "
                 "växtplanktonet och drar med sig kedjan uppåt. Tränade "
                 "policyer gör inte så. python3 tools/viability.py, ~4 min.")

    d.bullets(
        "Energibudgetgrinden",
        ["Kan varje beslutande FG gå ihop energimässigt över sin RATION? "
         "Frågan besvaras utan rollout, på sekunder: "
         "python3 tools/probes/budget_gate.py",
         ("p", "ration  = taket 1/h vid full aptit h(u_X) = 1   [ton byte / "
               "ton predator / tick]"),
         ("p", "kostnad = resting_metabolism · feeding_cost   [MJ / ton / "
               "tick]"),
         ("p", "krav    = kostnad / ration ≤ ransonens viktade "
               "energikvalitet"),
         "Grinden utvärderas över hela menyn, inte per par: motorn summerar "
         "intaget över alla byten och ger predatorn EN energivinst. Den "
         "rapporterar \"min share\" - minsta andel av det bästa bytet i "
         "dieten.",
         ("-", "I dag går alla fyra aktiva predatorer ihop på vilken diet "
               "som helst. Tumlare: break-even-ration på sill 5,3 % av "
               "kroppsvikten per dygn; mitt i Kastelein-fönstret (6,75 %) "
               "krävs minst 33 % sill - junk food-hypotesen."),
         ("-", "Grinden räknar på bibliotekets resting_metabolism, dvs vid "
               "t_ref - för sill en sommarkostnad.")],
        kicker="Går budgeten ihop?", size=15)

    d.table(
        "Tio år med tränade policyer",
        ["Funktionsgrupp", "År 1", "År 10, M1 0,683", "År 10, M1 0,415",
         "Förlopp med nuvarande bibliotek"],
        [["pelagic_fish (sill)", "12 565", "15 938", "6 232",
          "Halveras; biomassatopp okt-nov"],
         ["gadoids", "2 519", "1 430", "3 292",
          "+3 till +9 %/år, topp år 8, sedan nedgång"],
         ["porpoises", "11,5", "3,99", "5,9",
          "−6 %/år (förut −12 %/år)"],
         ["benthic_infauna", "18 603", "-", "15 856",
          "Stiger till år 3-4, betas sedan ned"],
         ["zooplankton", "~24 000", "~22 800", "-",
          "Stationär med årscykel ~9×"],
         ["phytoplankton", "~43 400", "~43 400", "-",
          "Stationär, vårblomning i mars"]],
        [20, 11, 15, 15, 39], size=11.5,
        align=["left", "right", "right", "right", "left"],
        kicker="Nollscenariot, 14 600 tick",
        lead="Fryst checkpoint från fix53 (tränad på M1 0,683), "
             "deterministisk probevärld 60 × 60 med mortalitet, migration, "
             "strömmar, dagsljus och temperatur, alla impakter muted. "
             "Årsmedel i ton.",
        footnote="Kolumnen M1 0,683 är slutet av år 10 (avsnitt 156.7), "
                 "M1 0,415 årsmedel (157.3); plankton ur avsnitt 156.2 "
                 "(närliggande checkpoint). En andra global seed höll sig "
                 "inom 2 %: cykeln drivs av kalendern, inte av kaos. "
                 "Policyerna bör tränas om på M1 0,415 innan den nya "
                 "balansen läses.")

    d.table(
        "Realiserad mot budgeterad predation",
        ["Par", "Bibliotekets M2 (1/år)", "Realiserad, år 1 → 10",
         "Läsning"],
        [["gadoids → gadoids", "0,217", "~0,001",
          "Tillväxt betald för en förlust som aldrig tas"],
         ["gadoids → pelagic_fish", "0,30", "0,19 → 0,14",
          "Sillen underbeskattas och växer tills födan bromsar"],
         ["gadoids → benthic", "0,0048", "0,33 → 0,17",
          "Annan bas (K 10 mot ~100 t/km²); dieten ~2× Ecopath"],
         ["zooplankton → phyto", "22,6", "30,5", "Något högre"],
         ["pelagic_fish → zoo", "5,24", "3,2 → 5,2", "Närmar sig budgeten"],
         ["porpoises → gadoids", "0,056", "0,087 → 0,020",
          "Följer tumlarnas nedgång"],
         ["porpoises → pelagic", "0,0117", "0,006 → 0,003",
          "Följer tumlarnas nedgång"]],
        [24, 18, 20, 38], size=11.5,
        align=["left", "right", "right", "left"],
        kicker="Avsnitt 156",
        lead="Tillväxtbudgeten antar att varje predator tar sin M2. I "
             "tioårskörningen gör de flesta par det ungefär - men inte "
             "alla.",
        footnote="Realiserad M2 = årligt uttag / bytets medelbiomassa. "
                 "Siffrorna gäller M1 0,683 för torskfisk (före avsnitt 157).")

    d.bullets(
        "Vad riggen är till för",
        ["Riggen är ett mätinstrument för mekaniken, inte en dom över "
         "världen. Med fryst beteende och fast seed är den deterministisk, "
         "så en biblioteksändring kan mätas isolerat utan träningsbrus.",
         "Den fångar fel som ingen policy kan rädda: en grupp som krymper "
         "även med tränat beteende - i dag tumlaren - pekar på ett "
         "strukturproblem. Där pekar riggen och energibudgetgrinden åt "
         "samma håll.",
         "Den säger inte att en värld är obrukbar. Premissen att en värld "
         "som underkänns inte kan räddas av träning håller inte: de "
         "tränade policyerna betar inte ned växtplanktonet som det giriga "
         "beteendet gör, och djurplanktonet lär sig hålla igen.",
         "Livskraft bör därför bedömas med de policyer som faktiskt "
         "används: samma kriterier på inferensdata från flera seeds, och "
         "ett stationaritetskrav som tål naturliga cykler i stället för att "
         "jämföra två fönster. Det är inte byggt än."],
        kicker="Diagnos, inte dom")

    # ---------------------------------------------------------------- 9
    d.section("8", "Styrkor, svagheter och förbättringsförslag",
              "En samlad bedömning av modellen som den står i oktober "
              "2026: vad den gör bra, var den brister och vad som bör "
              "göras härnäst.")

    d.bullets(
        "Styrkor: mekanik och metod",
        ["Massa och energi bokförs slutet. Predation, rörelse och strömmar "
         "bevarar massa; tillväxten betalas ur reserven; varje förlust "
         "bokförs under en av fyra orsaker. Källspårningen per cell är en "
         "identitet.",
         "Parametrarna är spårbara. growth_rate härleds ur r_max, M1 och "
         "Ecopath-M2; ljusklimat, temperatur och P-I-kurvor kommer ur "
         "SMHI-data för platsen; varje biblioteksvärde har en kommentar "
         "och ett avsnitt i resumén.",
         "Ett bibliotek, en kalibrering: ticklängden 1-6 h räknas om i "
         "minnet och skrivs aldrig tillbaka.",
         "Beteendet lärs, det hackas inte. Formande termer är avstängda; "
         "problem löses i biologin eller i perceptionen.",
         "Reproducerbart och testat: CRN, deterministiska probevärldar, "
         "CPU/GPU-paritet per steg och en testsvit på över 1 000 tester.",
         "Snabb diagnostik i tre steg: energibudgetgrinden på sekunder, "
         "fryst rigg på minuter, probeverktyg för års- och "
         "månadsbudgetar med tränade policyer."],
        kicker="Vad som bär", size=15)

    d.table(
        "Styrkor: beteende som liknar naturen",
        ["Fenomen", "I modellen (tränade policyer, år 2-10)",
         "Jämförelse"],
        [["Vårblomning", "Växtplankton 0,75 i januari → 1,12 i mars",
          "Kiselalgsblomning feb-mar i Skagerrak"],
         ["Djurplanktonets årscykel",
          "Minimum januari, maximum maj-juni (~9×)",
          "Calanus-likt, 2-3 månader efter blomningen"],
         ["Vintervila", "Djurplankton vilar 0,78-0,85 på vintern, äter "
          "mest i feb-mar", "Diapaus-likt (Calanus på djupet)"],
         ["Dygnsvandring", "Djurplankton vilar dag/natt 0,93/0,60 (vinter)",
          "DVM där riktig natt finns"],
         ["Visuell planktonätare", "Sill vilar 0,11 dag / 0,75 natt "
          "(vinter)", "Sill äter i ljus, filtrerar på natten"],
         ["Fettcykel", "Sillens s 0,23 i februari → 0,87 i juni",
          "Mager Q1, fet Q3"],
         ["Säsongsbyte i dieten", "Torskfisk: sommar sill + djurplankton, "
          "vinter bottendjur", "Riktning rimlig, andel förhöjd"],
         ["Kontinuerligt födosök", "Tumlare äter 97-100 % av tiden, Q/B "
          "22/år", "Nära kontinuerligt (Wisniewska m fl 2016)"],
         ["Återhållsam betning", "Djurplankton vilar ~0,8 och betar inte "
          "ned växtplanktonet", "Emergent på populationsnivå"]],
        [20, 44, 32], size=10.8, kicker="Emergent, inte inprogrammerat",
        footnote="Ur avsnitt 139 och 156, detrenderade månadsmönster. Inget "
                 "av detta är inskrivet i reglerna - det följer av kalendern, "
                 "energibudgeten och vad policyerna har lärt sig.")

    d.bullets(
        "Svagheter: biologi och struktur",
        ["Tillväxtbudgeten betalas fullt först vid s = 1, där hungern "
         "stänger intaget. Verkliga s ligger på 0,6-0,75, så torskfisk och "
         "tumlare realiserar 40-50 % av den designade produktionen - för "
         "tumlare täcker det inte ens M1.",
         "Inga näringsämnen: växtplanktonet ligger på en platå april-oktober "
         "(1,6× årsvariation mot 5-10× i klorofyll i naturen). Ingen "
         "sommarsvacka, ingen höstblomning.",
         "Toppredatorerna följer inte bytet: tumlare överlappar sitt byte "
         "till 0,15-0,22 och flyttar 0-3 % av tiden; torskfisk flyttar 1 %. "
         "Med fyllda tak ger en tätare cell bara ~2 % mer intag.",
         "Tumlarna minskar 6-12 % per år över tio år. Dieten är anpassad "
         "(attackkvot 13), inte emergent. Kannibalismen är budgeterad men "
         "inträffar inte.",
         "Djupet är kollapsat till 2D; ingen lekkostnad, ingen diapaus som "
         "tillstånd. Flera värden vilar på en enda källa (växtplanktonets "
         "takt, I_k från Kattegatt, mörkerkvoterna är skattningar).",
         "Sälar, sjöfåglar och alla impakter är muted: inget scenario är "
         "ännu kört med det nuvarande biblioteket."],
        kicker="Var modellen brister", size=14.5)

    d.bullets(
        "Svagheter: träning och utvärdering",
        ["Policyerna tränas på ettårsrollouts. Allt efter år 1 i en "
         "tioårskörning ligger utanför träningsfördelningen, och världen "
         "är inte i jämvikt efter tio år.",
         "ARS-steget kan bli brusdominerat: vid för högt η/σ växer vikternas "
         "norm som en slumpvandring och policyerna skärps mer än belöningen "
         "motiverar. Ingen normkontroll finns.",
         "Små gruppers belöning är brusig: tumlarnas σ_F låg 4-10× över "
         "övriga DM med 250 kg-enheten och ligger fortfarande ~4× över "
         "torskfisk och djurplankton. Under min_split_biomass döljer argmax den "
         "verkliga fördelningen, och fragment under ett djur dör genom "
         "utdöendesvepet.",
         "Livskraft bedöms med en eller få seeds, och det finns inget "
         "cykeltolerant stationaritetskrav; riggen som har ett kriterium "
         "kör fryst beteende.",
         "Varje biblioteksändring kräver omträning innan siffror citeras. "
         "Torskfiskens M1 ändrades 12 oktober - efter att fix53 tränats.",
         "Nätverket är litet (2 × 30) och utan minne; policyerna ser ljuset "
         "men inte datumet."],
        kicker="Var metoden brister", size=14.5)

    d.table(
        "Förbättringsförslag: modellen",
        ["#", "Förslag", "Motiv", "Avsnitt"],
        [["1", "Referensfyllnad < 1 i tillväxtbudgeten (t ex s ≈ 0,7)",
          "Budgeten betalas i dag bara vid s = 1, som aldrig nås",
          "156.4"],
         ["2", "Näringsämnespool för växtplankton",
          "Sommarsvacka och höstblomning saknas", "138.7, 156.4"],
         ["3", "Funktionellt svar för toppredatorer (a eller h per par) så "
          "att täthet lönar sig", "Torskfisk och tumlare står still",
          "154.6, 156.4"],
         ["4", "Kannibalism: sänk M2 eller ge den en mekanism",
          "Tillväxt betald för en förlust som inte tas", "156.4, 157.2"],
         ["5", "Lekkostnad för sill, vinterdiapaus för djurplankton",
          "Fettminimum i fel månad; vinteröverlevnad", "143.6, 138.7"],
         ["6", "SDA i assimilation_factor i stället för per-tick-avgift",
          "Kostnaden skalar med intaget, inte med tiden", "145.5"],
         ["7", "Temperaturskalad attackhastighet (E ~ 0,4 eV)",
          "Steg 2 efter Q10 på metabolismen", "143.2"],
         ["8", "Bottensamhälle med K ~100 t/km² enligt litteraturen",
          "Bottendominans i torskdieten", "136, 156.10"]],
        [4, 42, 38, 14], size=11, kicker="Biologi och mekanik",
        footnote="Ordningen är en bedömning av nytta mot kostnad. Varje "
                 "förslag mäts först med probeverktygen och tränade "
                 "policyer innan biblioteket ändras.")

    d.table(
        "Förbättringsförslag: träning och utvärdering",
        ["#", "Förslag", "Motiv", "Avsnitt"],
        [["1", "Träna om på det aktuella biblioteket och läs balansen med "
          "flera seeds", "Alla checkpoints är tränade före M1-ändringen",
          "157"],
         ["2", "Cykeltolerant livskraftskriterium på inferensdata",
          "Två-fönstertestet underkänner vanliga predator-bytescykler",
          "126"],
         ["3", "Normkontroll i ARS (viktavklingning eller fast |θ|); σ som "
          "utforskningsratt", "Slumpvandring i vikterna vid högt η/σ",
          "154.8-10"],
         ["4", "Sampling i stället för argmax under min_split_biomass; "
          "hantera fragment under ett djur", "Dold fördelning; tumlare dör "
          "genom svepet", "152.6, 156.9"],
         ["5", "Längre utvärdering än träningsepisoden: flerårsrollouts i "
          "probe och belöning", "Tioårsbeteendet ligger utanför "
          "träningsfördelningen", "156.2"],
         ["6", "Aktivera sälar, sjöfåglar och impaktscenarier först när "
          "nollscenariot är livskraftigt", "En jämförelse kräver en "
          "självbärande bas", "-"]],
        [4, 44, 38, 12], size=11, kicker="Metod",
        footnote="Inget av förslagen kräver formande termer i belöningen; "
                 "de rör biologin, optimeraren eller utvärderingen.")

    # --------------------------------------------------------------- 10
    d.section("9", "Scenarier och körningar",
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
         "I Mareld definieras påverkan av tre impaktvariabler i "
         "projektfilen - pelagic_trawling (0-1 000), windfarm_noise "
         "(0-140 dB, observerbar för policyerna) och rotor (0-1). "
         "Alla tre ligger muted tills ett scenario aktiverar dem.",
         "Kartor kan antingen spawnas proceduralt (perlin, colony) eller "
         "läsas in: inference-blocket i mareld2.yaml pekar rotor mot "
         "vindparker.npz, byggd ur parkernas verkliga geometri.",
         "Skillnaden mellan två scenarier ska rapporteras som relativ "
         "förändring per funktionsgrupp, under uttalade antaganden - "
         "aldrig som en absolut prognos."],
        kicker="Definition och praktik")

    d.formula(
        "Körningar",
        [("Träning, inferens och grindar", """
python3 train.py --project mareld2.yaml --run-name <namn> --profile info
python3 train_gpu.py --project mareld2.yaml --run-name <namn> ...  # GPU
python3 train.py --project mareld2.yaml --species pelagic_fish \\
        --generations 1 --iter-per-gen 1 --workers 1        # röktest
python3 inference.py --project mareld2.yaml --run-name <namn>
python3 tools/biomass_html.py results/<namn>                # plots.html
python3 tools/probes/budget_gate.py        # sekunder, ingen rollout
python3 tools/probes/daylight_table.py     # kalendern månad för månad
python3 tools/viability.py                 # ~4 min, fryst beteende
python3 -m pytest tests/ -q
""")],
        tail=[("-", "Viktiga flaggor: --tick-length 1-6, --mortality on, "
                    "--mortality_multiplier, --migration on, --currents on, "
                    "--local_reward, --no-mass-balance, --policynetwork, "
                    "--resume, --visual, --rnd_baseline."),
              ("-", "Den levande visningen visar månad, dagslängd och en "
                    "kalenderremsa under inferensplotten; --rnd_baseline "
                    "ritar slumpmässiga referenskurvor.")],
        kicker="Vad som körs")

    d.table(
        "Avsiktliga avvikelser från maj-utgåvan",
        ["Vad", "Varför"],
        [["Tillväxten är massbalanserad och growth_rate härleds",
          "Den dokumenterade termen B·(1 + MG·q) skapar massa utan att "
          "debitera energi - mätt till 198-344 % av vad som åts, mot ett "
          "termodynamiskt tak på 28,9 %. Takten härleds så att "
          "nettotillväxten blir litteraturens."],
         ["Hungern är kontinuerlig, inte en mask",
          "Maj-utgåvan hade både masken vid s = 0,8 och faktorn "
          "h = max(0, 1 − s/0,8). I dag gäller full aptit upp till "
          "underhållsnivån och linjär nedgång därefter: ätandet styrs av "
          "föda och magkapacitet, inte av reserven."],
         ["Fitness räknar biomassa och reserv i samma valuta",
          "α och β blir godtyckliga vikter mellan två storheter som redan "
          "har en växelkurs: energy_content."],
         ["Rollouterna är 15 tick till ett år, inte 2",
          "Integralbelöningen läser varje tick, och kalendern kräver att "
          "en rollout rymmer årstiderna."],
         ["Säsongen kommer från en kalender",
          "Maj-utgåvan hade ingen säsong. Ljus och temperatur för platsen "
          "driver nu attack, primärproduktion och metabolism."],
         ["Minnesutvidgningen N(c,t-1) och storleksstegen 3×3 → 60×60 "
          "ingår inte", "Enda tillståndet mellan tick är förra tickets "
          "gömda andel; variationen i träningen kommer från spawn, "
          "startbiomassa och kalenderstart."]],
        [27, 73], size=11, kicker="Dokumentet mot koden")

    d.bullets(
        "Öppna frågor",
        ["Tumlarna minskar med 6-12 % per år även med tränade policyer. De "
         "största kvarvarande orsakerna är tillväxtbudgeten vid s = 1, "
         "vinterkatabolism och fragment under ett djur.",
         "Med torskfiskens nya M1 (0,415) överskjuter torskfisken och "
         "halverar sillen inom tio år med de gamla policyerna. Är det en "
         "långsam cykel eller en ny nivå? Det kräver omträning.",
         "Djurplanktonet faller till ~13 % av startbiomassan i januari "
         "och återhämtar sig till maj. Diapaus som tillstånd är en "
         "kandidat om det inte håller med tränade policyer.",
         "Ska tillväxtbudgeten referera till en realistisk mättnad (s ≈ "
         "0,7) i stället för s = 1? Det ändrar varje DM:s growth_rate.",
         "Växtplanktonet saknar näringsbegränsning; sommarnivån ligger "
         "kvar nära aprilnivån.",
         ("p", "Hela historiken - varje kalibrering, A/B-test och öppet "
               "problem - ligger i mareld_resume.txt, avsnitt 0-158.")],
        kicker="Vad som återstår", size=15.5)

    d.section("", "Tack",
              "Frågor, invändningar och rättelser hör hemma i "
              "mareld_resume.txt lika mycket som i koden.")


# ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default="Strategi_okt.pdf")
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
        info["Subject"] = ("Uppdaterad strategi, oktober 2026 - beskriver "
                           "den mekanik simulatorn kör i dag")
    print(f"{args.out}: {deck.n} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
