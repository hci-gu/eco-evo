"""Generate ``Strategi_sep.pdf`` - the September 2026 edition of the
strategy deck, rewritten against the mechanics the code actually runs.

The original ``Strategi.pdf`` (May 2026) is a 37-slide 16:9 PowerPoint
export. This generator reproduces that format (960 x 540 pt pages) with
matplotlib so the deck can be regenerated whenever the model moves.

Everything stated on a slide is read from, or checked against:
  lib/environments/ecosystem_env/*.py   the tick
  lib/runners/trainer.py, parallel_worker.py, policy.py   ARS
  lib/world/tick_time.py, functional_group.py, energy_balance.py
  fgconfig/fg_library.yaml, mareld2.yaml   the calibration
  VIABILITY.md, mareld_resume.txt sections 69-121

Slide text is Swedish (the original deck's language); parameter names
are kept in their English library spelling so a slide can be traced to
the YAML key it describes.

Usage:
    python3 tools/make_strategi_deck.py [-o Strategi_sep.pdf] [--png DIR]
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
        ax.text(LEFT, 26, "Strategi - Mareld / Poseidon Nord - september 2026",
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
            ax.text(LEFT, 320, f"Del {number}", color=ACCENT, size=16,
                    weight="bold", va="center", ha="left")
        for i, line in enumerate(wrap(title, 40, PAGE_W - LEFT - RIGHT - 60,
                                      weight="bold")):
            ax.text(LEFT, 272 - i * 48, line, color=NAVY, size=40,
                    weight="bold", va="center", ha="left")
        if blurb:
            yb = 210
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
            y = draw_code(ax, caption, code, LEFT, y, PAGE_W - LEFT - RIGHT)
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


def text_width(text, size, weight="normal", mono=False):
    """Rendered width of ``text`` in points (per-glyph sum, no kerning)."""
    return size * sum(_glyph_width(ch, weight, mono) for ch in str(text))


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
    steps = [
        ("1  Beslut", "observation → policy → maskad softmax"),
        ("2  Påverkansmortalitet", "impact_table → biomassa"),
        ("3  Predation", "Holling II/III + interferens"),
        ("4  Energikostnader", "vila / äta / röra sig"),
        ("5  Rörelse", "N/E/S/W × hastighet, energi följer med"),
        ("6  Strömmar", "valfritt: current_response"),
        ("7  Tillväxt / svält", "massbalanserad, + mortalitet"),
        ("8  Masker", "accessibility, utdöendetröskel"),
    ]
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


def draw_food_web(fig, ax, y):
    nodes = {
        "phytoplankton": (210, 118, "Växtplankton"),
        "benthic_community": (720, 118, "Bottensamhälle"),
        "zooplankton": (210, 205, "Djurplankton"),
        "pelagic_fish": (300, 292, "Pelagisk fisk"),
        "gadoids": (650, 205, "Torskfiskar"),
        "porpoises": (790, 292, "Tumlare"),
        "seals": (520, 370, "Sälar"),
        "seabirds": (200, 370, "Sjöfåglar"),
    }
    edges = [
        ("phytoplankton", "zooplankton"),
        ("zooplankton", "pelagic_fish"),
        ("pelagic_fish", "gadoids"),
        ("benthic_community", "gadoids"),
        ("pelagic_fish", "porpoises"),
        ("gadoids", "porpoises"),
        ("pelagic_fish", "seals"),
        ("gadoids", "seals"),
        ("pelagic_fish", "seabirds"),
    ]
    bw, bh = 142, 40
    for a, b in edges:
        xa, ya, _ = nodes[a]
        xb, yb, _ = nodes[b]
        x0, y0 = _rect_edge(xa, ya, bw, bh, xb, yb)
        x1, y1 = _rect_edge(xb, yb, bw, bh, xa, ya)
        arrow(ax, x0, y0, x1, y1, color="#8FA9BC", lw=1.4, head=7)
    for key, (x, yy, label) in nodes.items():
        ndm = key in ("phytoplankton", "benthic_community")
        box(ax, x - bw / 2, yy - bh / 2, bw, bh, label,
            face="#E4EEF3" if not ndm else "#E9F2E4",
            edge=ACCENT if not ndm else "#5C8A3C", size=12, weight="bold")
    ax.text(LEFT, 112, "grönt = icke beslutande (logistisk tillväxt)\n"
                       "blått = beslutande (policynätverk och energibudget)",
            color=MUTED, size=11.5, va="center", ha="left")


def draw_functional_response(fig, ax, y):
    a2, h2, w2 = 0.04, 25.0, 1.0        # gadoids (typ II, interferens 1.0)
    a3, h3 = 0.25, 4.0                  # zooplankton (typ III)
    B = np.linspace(0, 1.2, 400)
    f2 = a2 * B / (1 + a2 * h2 * B)
    f2i = a2 * B / (1 + a2 * h2 * B + w2 * 0.5)
    ax1 = axes_pt(fig, 92, 175, 360, 190)
    ax1.plot(B, f2, color=ACCENT, lw=2.2, label="typ II  (w = 0)")
    ax1.plot(B, f2i, color=ACCENT_2, lw=2.2, ls="--",
             label="typ II  (w = 1, B_pred = 0,5)")
    ax1.axhline(1 / h2, color=MUTED, lw=1.0, ls=":")
    ax1.text(1.19, 1 / h2 + 0.0012, "tak 1/h", color=MUTED, size=9.5,
             ha="right")
    ax1.set_title("Torskfiskar: a = 0,04  h = 25  w = 1,0", size=11,
                  color=NAVY)
    ax1.legend(fontsize=8.5, loc="lower right", frameon=False)

    Bz = np.linspace(0, 2.5, 400)
    f3 = a3 * Bz ** 2 / (1 + a3 * h3 * Bz ** 2)
    f2z = a3 * Bz / (1 + a3 * h3 * Bz)
    ax2 = axes_pt(fig, 520, 175, 360, 190)
    ax2.plot(Bz, f2z, color="#9FB6C6", lw=2.0, ls="--", label="typ II")
    ax2.plot(Bz, f3, color=ACCENT, lw=2.2, label="typ III (specialist)")
    ax2.axhline(1 / h3, color=MUTED, lw=1.0, ls=":")
    ax2.text(2.48, 1 / h3 + 0.004, "tak 1/h", color=MUTED, size=9.5,
             ha="right")
    ax2.set_title("Djurplankton: a = 0,25  h = 4", size=11, color=NAVY)
    ax2.legend(fontsize=8.5, loc="lower right", frameon=False)

    for a_ in (ax1, ax2):
        a_.set_xlabel("synlig bytesbiomassa i cellen  [ton]", size=9.5)
        a_.set_ylabel("intag  [ton byte / ton predator / tick]", size=9)
        a_.tick_params(labelsize=8.5)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        a_.grid(alpha=0.25, lw=0.6)


def draw_hunger_gate(fig, ax, y):
    s = np.linspace(0, 1, 300)
    ax1 = axes_pt(fig, 92, 178, 360, 190)
    for scale, label, color in ((0.8, "standard  scale = 0,8", ACCENT),
                                (1.37, "tumlare  scale = 1,37", ACCENT_2)):
        ax1.plot(s, np.maximum(0.0, 1 - s / scale), color=color, lw=2.2,
                 label=label)
    ax1.axvline(0.5, color=MUTED, lw=1.0, ls=":")
    ax1.text(0.52, 0.92, "u_X = 0,5", color=MUTED, size=9.5)
    ax1.set_xlabel("mättnadsgrad  s_X = E_X / ME_X", size=9.5)
    ax1.set_ylabel("hungerfaktor  h_X", size=9.5)
    ax1.legend(fontsize=8.5, frameon=False)

    q = np.linspace(-0.5, 0.5, 300)
    ax2 = axes_pt(fig, 520, 178, 360, 190)
    ax2.plot(q, np.where(q >= 0, 0.0913 * q, 0.2 * q), color=ACCENT, lw=2.2,
             label="djurplankton  MG = 0,0913 / SR = 0,2")
    ax2.plot(q, np.where(q >= 0, 0.004 * q, 0.025 * q), color=ACCENT_2,
             lw=2.2, label="pelagisk fisk  MG = 0,004 / SR = 0,025")
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.axvline(0, color=MUTED, lw=0.8)
    ax2.set_xlabel("överskottsenergi  q_X = s_X - u_X", size=9.5)
    ax2.set_ylabel("relativ biomassaändring / tick", size=9.5)
    ax2.legend(fontsize=8.5, frameon=False, loc="upper left")

    for a_ in (ax1, ax2):
        a_.tick_params(labelsize=8.5)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.grid(alpha=0.25, lw=0.6)


def draw_impact_curves(fig, ax, y):
    ax1 = axes_pt(fig, 92, 178, 360, 190)
    xs = np.array([60.0, 80.0, 100.0, 120.0])
    ef = np.array([0.0, 0.15, 0.40, 0.80])
    q = np.linspace(40, 140, 400)
    ax1.plot(q, np.interp(q, xs, ef, left=ef[0], right=ef[-1]),
             color=ACCENT, lw=2.2)
    ax1.plot(xs, ef, "o", color=ACCENT_2, ms=5.5, zorder=3)
    ax1.set_title("windfarm_noise → tumlare: energy_factor", size=11,
                  color=NAVY)
    ax1.set_xlabel("kartvärde  (dB)", size=9.5)
    ax1.set_ylabel("extra metabolisk kostnad", size=9.5)

    ax2 = axes_pt(fig, 520, 178, 360, 190)
    xs2 = np.array([0.0, 0.25, 0.5, 1.0])
    bf2 = np.array([0.0, 0.005, 0.015, 0.03])
    r = np.linspace(-0.1, 1.2, 400)
    ax2.plot(r, np.interp(r, xs2, bf2, left=bf2[0], right=bf2[-1]),
             color=ACCENT, lw=2.2)
    ax2.plot(xs2, bf2, "o", color=ACCENT_2, ms=5.5, zorder=3)
    ax2.set_title("rotor → sjöfåglar: biomass_factor", size=11, color=NAVY)
    ax2.set_xlabel("kartvärde  (rotortäthet 0-1)", size=9.5)
    ax2.set_ylabel("biomassaforlust / tick", size=9.5)

    for a_ in (ax1, ax2):
        a_.tick_params(labelsize=8.5)
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
            "egen cell" if centre else key,
            face="#E4EEF3" if centre else BAND,
            edge=ACCENT if centre else "#9FB6C6", size=12,
            weight="bold" if centre else "normal")
    ax.text(cx, cy - 2 * c - 6, "plusform: egen cell + fyra grannar",
            color=MUTED, size=11.5, ha="center", va="center")
    xb = 468
    draw_code(ax, "Kanaler per cell (kompakt layout)", """
egen cell  [ B_egen, s_egen, B_obs_1..k, imp_1..m ]
granne d   [ B_egen, B_obs_1..k, imp_1..m ]

in_dim_X = (2 + k_X + m) + 4 · (1 + k_X + m)
""", xb, y - 6, PAGE_W - RIGHT - xb, size=12.5)
    draw_bullets(ax, [
        ("-", "k_X = antal ANDRA FG som X observerar enligt "
              "Observability-matrisen"),
        ("-", "m = antal impaktkartor märkta observable i projektfilen"),
        ("-", "B_obs är SYNLIG biomassa: den gömda (vilande) andelen "
              "dras bort"),
        ("-", "Mareld i dag: k = 1-5 och m = 1 (windfarm_noise), "
              "dvs 16-36 kanaler per beslutande FG"),
    ], xb, y - 128, PAGE_W - RIGHT - xb, size=13)


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
    a = column(x0, 7, "observation\n(in_dim)", ACCENT)
    b = column(x1, 6, "dolt lager 1\n30 noder, sigmoid", "#7FB2C0")
    c = column(x2, 6, "dolt lager 2\n30 noder, sigmoid", "#7FB2C0")
    d = column(x3, 5, "logits → mask → softmax", ACCENT_2)
    for u in a:
        for v in b:
            ax.plot([x0, x1], [u, v], color="#C9D8E2", lw=0.4, zorder=1)
    for u in b:
        for v in c:
            ax.plot([x1, x2], [u, v], color="#C9D8E2", lw=0.4, zorder=1)
    for u in c:
        for v in d:
            ax.plot([x2, x3], [u, v], color="#C9D8E2", lw=0.4, zorder=1)
    labels = ["Flytta N", "Flytta E/S/W", "Vila (= gömma sig)",
              "Äta Y_1 ... Y_n", ""]
    for yy, lab in zip(d[::-1], labels):
        if lab:
            ax.text(x3 + 16, yy, lab, color=INK, size=11.5, va="center",
                    ha="left")


def draw_energy_flow(fig, ax, y):
    yb = y - 128
    w, h = 176, 62
    stages = [
        (LEFT, "Intag  intag_XY", BAND),
        (LEFT + 210, "Assimilerat  · assimilation", "#E4EEF3"),
        (LEFT + 420, "Reserv R_X  (+ intag − kostnad)", "#E4EEF3"),
        (LEFT + 630, "Ny biomassa  dB ton", "#E9F2E4"),
    ]
    for x, text, face in stages:
        box(ax, x, yb, w, h, text, face=face, size=12, weight="bold")
    for i in range(3):
        arrow(ax, stages[i][0] + w, yb + h / 2, stages[i + 1][0],
              yb + h / 2, lw=1.6)
    for x, label in ((LEFT + 210, "spill"),
                     (LEFT + 420, "vila / rörelse / födosök\n+ påverkan")):
        arrow(ax, x + w / 2, yb, x + w / 2, yb - 44, color=MUTED, lw=1.2)
        ax.text(x + w / 2, yb - 56, label.replace("\n", ", "), color=MUTED,
                size=11, ha="center", va="center")
    ax.text(LEFT + 630 + w / 2, yb + h + 26,
            "priset dB · energy_content debiteras reserven",
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
    panels = [("uniform", uniform), ("perlin  (växtplankton)", perlin),
              ("colony  (pelagisk fisk)", colony),
              ("env_driven  (följer växtplankton)", env)]
    for i, (label, field) in enumerate(panels):
        a = axes_pt(fig, 96 + i * 196, 190, 160, 160)
        a.imshow(field, cmap="YlGnBu", interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(label, size=10.5, color=NAVY)


# ----------------------------------------------------------------------
# The deck
# ----------------------------------------------------------------------
def build(d):
    # ---------------------------------------------------------------- 1
    d.title_slide(
        "Strategi",
        "För att bygga ekosystemsimulatorer",
        ["Uppdaterad utgåva - september 2026",
         "Mareld / Poseidon Nord",
         "Beskriver den mekanik som simulatorn faktiskt kör i dag"])

    d.bullets(
        "Om den här utgåvan",
        [("p", "Den här versionen ersätter maj-utgåvan av Strategi.pdf. "
               "Strukturen är densamma; innehållet är skrivet mot koden "
               "som den ser ut i dag."),
         "",
         "Koden är modellen. Dokumentet beskriver den - inte tvärtom. "
         "Där maj-utgåvan och koden skiljer sig är det koden som gäller, "
         "och skillnaden redovisas i klartext.",
         "Varje formel och siffra i dokumentet är hämtad ur tickpipelinen "
         "i lib/environments/ecosystem_env/, ur ARS-tränaren i "
         "lib/runners/, ur fg_library.yaml eller ur mareld2.yaml.",
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
          "Massbalanserad: tillväxten betalas ur energireserven "
          "(--no-mass-balance ger gamla termen)"],
         ["Predation", "Önskat intag = B·π·I·h, proportionell nedskalning",
          "Holling typ II/III med Beddington-DeAngelis-interferens; "
          "nedskalningen finns kvar"],
         ["Hunger", "All predation maskades bort över s_X = 0,8",
          "Ingen mask - hungern är en kontinuerlig faktor "
          "h_X = max(0, 1 - s_X/scale)"],
         ["Vila", "Vila = låg energikostnad",
          "Vila är också en gömma-sig-action med per-par synlighetsgolv"],
         ["Fitness", "α·ΔB + β·ΔR över två transitioner",
          "Medelvärde över rollouten av log(total energi / startenergi); "
          "15-200 tick"],
         ["Träning", "Alternerande: en FG i taget",
          "Co-evolution: alla beslutande FG perturberas samtidigt i samma "
          "rollout"],
         ["Ticklängd", "Fast 6 h", "--tick-length 1-6 h, biblioteket "
          "räknas om i minnet vid inläsning"],
         ["Nytt", "-", "Strömmar, migration, utdöendetröskel, "
          "säsongsvariation, lokal belöning per cell, viabilitetsrigg"]],
        [10, 30, 46], kicker="Översikt", size=11.5,
        footnote="Avsnitt 8 i det här dokumentet listar de avvikelser som "
                 "är avsiktliga och varför.")

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
         "sorters miljöpåverkan.",
         "Populationernas beteende - särskilt rörelse och födosök - styrs "
         "av beteendemodeller som tränas med ARS, men kan också vara "
         "handkodade (viabilitetsriggen kör helt utan nätverk).",
         "Biologin driver, inte action-hackar. Parametrar och mekanismer "
         "justeras först; formande termer i belöningen är sista utvägen "
         "och är avstängda i alla träningsprofiler.",
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
         ("p", "Ett tillägg sedan maj: innan ett scenario jämförs måste "
               "världen klara viabilitetskriteriet i VIABILITY.md. En "
               "konfiguration som inte är självbärande gör varje jämförelse "
               "otolkbar, oavsett hur väl policyerna är tränade."),
         ],
        kicker="Vad modellen svarar på")

    d.bullets(
        "Beståndsdelar",
        ["En ekosystemmodell består av:",
         ("-", "en grid med längdskala och ticklängd"),
         ("-", "en uppsättning funktionsgrupper (FG), beslutande och icke "
               "beslutande"),
         ("-", "en uppsättning kartor - ett scenario"),
         ("-", "en uppsättning egenskaper för varje funktionsgrupp "
               "(fg_library.yaml)"),
         ("-", "en dynamisk modell med uppdateringsregler för kartorna - "
               "ticken"),
         ("-", "en beteendemodell för varje beslutande funktionsgrupp"),
         ("-", "spawnstrategier som bestämmer hur startbiomassan läggs ut"),
         ("-", "acceptanskriterier: viabilitetsrigg och energibudgetgrind"),
         "",
         "Biblioteket (fg_library.yaml) är gemensamt för alla projekt. "
         "Projektfilen (mareld2.yaml) väljer vilka FG som ingår, "
         "startbiomassor, impaktvariabler och gridstorlek."],
        kicker="Modellens delar")

    d.bullets(
        "Grid",
        ["Varje ekosystemmodell har en grid, dvs en matris av celler med:",
         ("-", "en dimension - Mareld kör 60 x 60 celler"),
         ("-", "en längdskala - en cellsida = 1 km"),
         ("-", "en tidskala - ett tick = 1-6 timmar, normalt 6"),
         "Griden är plan och fyrgrannskopplad: all rörelse sker till N, E, "
         "S eller W, aldrig diagonalt.",
         "En valfri karta accessibility styr vilka celler som är "
         "tillgängliga. Den är gemensam för alla FG, inte per FG: "
         "rörelse in i en otillgänglig cell maskas bort, och biomassa som "
         "ändå hamnar där nollställs i slutet av ticket.",
         "Utan migration är gridens kant sluten: rörelse ut över kanten "
         "maskas bort. Med --migration on lämnar biomassan griden och "
         "återinförs koncentrerat längs kanten."],
        kicker="Geografi")

    d.table(
        "Ticklängd är en körningsparameter",
        ["Parametertyp", "Regel vid faktor k = ny/gammal", "Exempel"],
        [["flux - mängd som överförs per tick", "x · k",
          "max_intake_rate, resting_metabolism, movement_speed, seed_rate"],
         ["growth - andel som LÄGGS TILL ett lager", "(1 + x)^k - 1",
          "growth_rate"],
         ["loss - andel som TAS BORT ur ett lager", "1 - (1 - x)^k",
          "natural_mortality, starve_rate, biomass_factor i impacttabeller"],
         ["inverse - storhet vars invers är takten", "x / k",
          "handling_time (Hollingtaket 1/h ska hållas fast i realtid)"],
         ["period - varaktighet mätt i tick", "p / k, lägst 1",
          "seasonal_period"],
         ["tick-oberoende", "oförändrad",
          "max_energy_reserve, energy_content, maintenance_level, "
          "satiation_scale, interference, min_split_biomass, "
          "kostnadsmultiplikatorerna"]],
        [22, 24, 40], kicker="--tick-length", size=11.5,
        lead="Motorn är ticklängdsagnostisk: varje bibliotekstal är uttryckt "
             "PER TICK. Ticklängden är en tolkningskonstant som räknar om "
             "per-tick-takter till realtid - och som, när den inte är 6 h, "
             "skalar om varje tickberoende parameter i minnet när FG:erna "
             "byggs.",
        footnote="fg_library.yaml har EN kalibrering (6 h) och skrivs aldrig "
                 "om av en körning. lib/world/tick_time.py är enda "
                 "definitionspunkten.")

    d.table(
        "Funktionsgrupper",
        ["Funktionsgrupp", "Typ", "Meny", "Roll i modellen"],
        [["phytoplankton", "icke beslutande", "-",
          "Primärproducent, logistisk tillväxt mot bärkraft, driver med "
          "strömmen"],
         ["benthic_community", "icke beslutande", "-",
          "Bottenföda för torskfisk, mycket långsam tillväxt, "
          "förflyttas inte"],
         ["zooplankton", "beslutande", "phytoplankton",
          "Betare, specialist (typ III), rör sig 0,02 cell/tick"],
         ["pelagic_fish", "beslutande", "zooplankton",
          "Stimfisk, specialist (typ III), interferens 0,7"],
         ["gadoids", "beslutande", "pelagic_fish, benthic_community",
          "Generalist (typ II), interferens 1,0"],
         ["porpoises", "beslutande", "pelagic_fish, gadoids",
          "Toppredator, hög viloförbränning, känslig för undervattensbuller"],
         ["seals", "beslutande", "pelagic_fish, gadoids",
          "Toppredator, koloni-spawn"],
         ["seabirds", "beslutande", "pelagic_fish",
          "Toppredator, enda FG med rotormortalitet"]],
        [16, 12, 22, 40], kicker="Mareld: åtta grupper, sex med policy",
        size=11.5,
        footnote="Engelska snake_case-id:n är kanoniska. seabirds och seals "
                 "är muted i mareld2.yaml - definierade men inte utlagda i "
                 "det aktuella scenariot.")

    d.figure("Näringsväv", draw_food_web, kicker="Meny enligt biblioteket",
             footnote="Pilen går från byte till konsument. Varje par har "
                      "egna värden för handling_time, assimilation_factor "
                      "och - för vissa par - ett eget synlighetsgolv.")

    # ---------------------------------------------------------------- 3
    d.section("2", "Kartor",
              "Allt tillstånd i modellen är kartor över griden.")

    d.bullets(
        "Kartor",
        ["En karta är en funktion från griden till de reella talen.",
         "En karta med värden i [0, 1] kan betraktas som en gråskalebild. "
         "Andra kartor - djup, temperatur, biomassa - visas som heatmaps.",
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
        "Karttyper",
        ["Karttyp", "Symbol", "Enhet", "Tolkning"],
        [["Geografiska kartor", "Geo_k(c)", "[0,1], m, °C",
          "Abiotisk miljö och habitat. accessibility är den enda som "
          "motorn läser i dag."],
         ["Biomassakartor", "B_X(c)", "ton/cell",
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
          "Genereras proceduralt per tick ur brus, lagras inte."]],
        [17, 13, 12, 48], kicker="Vad som finns i en körning", size=11.5,
        footnote="Administrativa kartor (andel av cell inom ett område) "
                 "finns i strategin men har ingen motsvarighet i motorn i "
                 "dag; de hör till efterbearbetningen.")

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
         ["Energiinnehåll", "energy_content", "MJ/ton",
          "Vad ett ton av FG:n är värt som föda - och priset på ett ton "
          "egen ny biomassa."],
         ["Underhållsnivå", "maintenance_level (u_X)", "andel av ME",
          "Nollpunkt: q_X = s_X - u_X. 0,5 för samtliga beslutande FG i "
          "biblioteket."],
         ["Mättnadsskala", "satiation_scale", "andel av ME",
          "Hungerfaktorns nollpunkt: h_X = max(0, 1 - s_X/scale). "
          "Standard 0,8; tumlare 1,37."],
         ["Energikänslighet", "impact_table.energy_factor", "faktor",
          "Extra metabolisk kostnad från påverkan p."]],
        [17, 24, 12, 42], size=11.5, kicker="Per funktionsgrupp")

    d.table(
        "Egenskaper: tillväxt, förluster och rörelse",
        ["Egenskap", "Nyckel", "Enhet", "Roll"],
        [["Maximal tillväxt", "growth_rate (MG_X)", "andel/tick",
          "Tillväxtönskan dB = B · MG · q när q >= 0."],
         ["Svälttakt", "starve_rate", "andel/tick",
          "Katabolism när q < 0. Egen takt: 2 ggr growth_rate för djurplankton, 500 ggr för tumlare."],
         ["Naturlig mortalitet", "natural_mortality", "andel/tick",
          "Täthetsoberoende. Kräver --mortality on."],
         ["Maximal bärkraft", "max_carrying_capacity", "ton/cell",
          "Endast icke beslutande FG."],
         ["Maximalt intag", "max_intake_rate (a)", "ton byte/ton X/tick",
          "Attackhastighet i det funktionella svaret."],
         ["Hanteringstid", "handling_time (h)", "tick/ton",
          "Sätter det fysiologiska taket 1/h."],
         ["Interferens", "interference (w)", "1/ton",
          "Beddington-DeAngelis: egen täthet sänker intaget."],
         ["Hastighet", "movement_speed (v)", "cellsidor/tick",
          "Andel av den rörliga biomassan som når grannen. Klipps till "
          "[0, 1]."],
         ["Minsta delbara massa", "min_split_biomass", "kg",
          "Under detta kollapsar actionfördelningen till argmax."],
         ["Utdöendetröskel", "extinction_threshold_factor", "faktor",
          "Cell nollställs under faktor · min_split_biomass. Standard 0,5."],
         ["Synlighetsgolv", "visibility_floor", "[0,1]",
          "Andel som förblir synlig trots vila."],
         ["Strömrespons", "current_response", "[0,1]",
          "Andel av strömfältet som FG:n följer."]],
        [17, 24, 14, 40], size=10.8, kicker="Per funktionsgrupp")

    d.table(
        "Bibliotekets värden - beslutande funktionsgrupper",
        ["", "zooplankton", "pelagic_fish", "gadoids", "porpoises", "seals",
         "seabirds"],
        [["maintenance_level u", "0,5", "0,5", "0,5", "0,5", "0,5", "0,5"],
         ["max_energy_reserve", "675", "2 400", "2 500", "2 800", "5 200",
          "3 400"],
         ["resting_metabolism", "22,5", "8,0", "6,0", "90", "45", "135"],
         ["energy_content", "4 500", "7 500", "5 500", "8 500", "10 000",
          "7 000"],
         ["growth_rate", "0,0913", "0,004", "0,0025", "0,001", "0,001",
          "0,001"],
         ["starve_rate", "0,2", "0,025", "0,012", "0,5", "0,03", "0,2"],
         ["natural_mortality", "0,0025", "1,0e-4", "7,0e-5", "6,0e-5",
          "6,0e-5", "8,0e-5"],
         ["max_intake_rate a", "0,25", "0,04", "0,04", "0,035", "0,05",
          "0,15"],
         ["movement_speed v", "0,02", "1,0", "0,25", "1,0", "1,0", "1,0"],
         ["interference w", "0", "0,7", "1,0", "0", "0", "0"],
         ["visibility_floor", "0,2", "0,35", "0,6", "0", "0", "0"],
         ["min_split_biomass (kg)", "0", "0,5", "0,5", "250", "80", "1,0"],
         ["feeding / movement_cost", "2,0 / 1,2", "1,9 / 1,5", "1,75 / 2,0",
          "1,4 / 1,6", "1,3 / 1,5", "2,2 / 2,8"]],
        [22, 13, 13, 13, 13, 13, 13], size=11.0,
        align=["left", "right", "right", "right", "right", "right", "right"],
        kicker="fg_library.yaml, kalibrerat vid 6 h/tick",
        footnote="Energier i MJ/ton, takter per tick. Tumlare har dessutom "
                 "satiation_scale 1,37 och extinction_threshold_factor 0,02.")

    d.table(
        "Bibliotekets värden - icke beslutande grupper och predationspar",
        ["Predator → byte", "a", "h", "assimilation", "vis.golv", "svar"],
        [["zooplankton → phytoplankton", "0,25", "4,0", "0,65", "prey 0",
          "typ III"],
         ["pelagic_fish → zooplankton", "0,04", "25,0", "0,80", "prey 0,2",
          "typ III"],
         ["gadoids → pelagic_fish", "0,04", "25,0", "0,80", "0,25",
          "typ II"],
         ["gadoids → benthic_community", "0,04", "25,0", "0,70", "prey 0",
          "typ II"],
         ["porpoises → pelagic_fish", "0,035", "28,6", "0,90", "0,95",
          "typ II"],
         ["porpoises → gadoids", "0,035", "28,6", "0,84", "0,75", "typ II"],
         ["seals → pelagic_fish", "0,05", "20,0", "0,92", "prey 0,35",
          "typ II"],
         ["seals → gadoids", "0,05", "20,0", "0,86", "prey 0,6", "typ II"],
         ["seabirds → pelagic_fish", "0,15", "0,0", "1,00", "prey 0,35",
          "typ III"]],
        [30, 10, 10, 14, 14, 12], size=11.5,
        align=["left", "right", "right", "right", "right", "left"],
        lead="Icke beslutande: phytoplankton växer med growth_rate 0,125 mot "
             "max_carrying_capacity 24 ton/cell och energy_content 2 000 "
             "MJ/ton; benthic_community med 0,001 mot 10 ton/cell och "
             "3 000 MJ/ton, plus seed_rate 1e-6.",
        footnote="\"prey 0,2\" = paret ärver bytets eget visibility_floor. "
                 "Typ III används av specialister - predatorer med exakt en "
                 "aktiv bytesgrupp - och väljs automatiskt av motorn.")

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
                 "impaktkartorna, FG-uppsättningen eller gridlayouten blir "
                 "befintliga checkpoints ogiltiga - observationens bredd "
                 "ändras.")

    d.table(
        "Observability-matrisen",
        ["Observatör", "phyto", "zoo", "pel.fisk", "torskfisk", "tumlare",
         "säl", "sjöfågel", "botten"],
        [["zooplankton", "ja", "ja", "-", "-", "-", "-", "-", "-"],
         ["pelagic_fish", "-", "ja", "ja", "ja", "ja", "ja", "ja", "-"],
         ["gadoids", "-", "-", "ja", "ja", "ja", "ja", "-", "ja"],
         ["porpoises", "-", "-", "ja", "ja", "ja", "ja", "-", "-"],
         ["seals", "-", "-", "ja", "ja", "ja", "ja", "ja", "-"],
         ["seabirds", "-", "-", "ja", "-", "-", "-", "ja", "-"]],
        [20, 10, 10, 11, 12, 11, 9, 11, 10], size=11.5,
        align=["left"] + ["center"] * 8,
        lead="Matrisen bestämmer observationens bredd per FG: k_X = "
             "antal ja på raden utöver FG:n själv (den egna biomassan och "
             "mättnadsgraden är alltid med). k = 1 för djurplankton och "
             "sjöfågel, 5 för pelagisk fisk.",
        footnote="Raderna är inte symmetriska: torskfisk ser tumlare "
                 "(predatorn) men sjöfågel ser bara pelagisk fisk och sig "
                 "själv. Matrisen redigeras i FG-editorn.")

    d.bullets(
        "Maskning av actions",
        ["Predation maskas bort när bytet inte finns i cellen eller inte "
         "ligger på menyn. Ingenting annat maskar ätandet.",
         ("-", "Maj-utgåvan maskade bort all predation över s_X = 0,8. "
               "Den regeln finns inte i koden: hungern är i stället en "
               "KONTINUERLIG faktor h_X = max(0, 1 - s_X/satiation_scale) "
               "som skalar det önskade intaget mjukt mot noll."),
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
        lead="Hungerfaktorn h_X skalar intaget; överskottsenergin q_X = "
             "s_X - u_X sätter tecknet på tillväxten. De är olika trösklar "
             "med olika nollpunkter, och avståndet mellan dem är den "
             "marginal en FG har att leva på.",
        footnote="Med u_X = 0,5 och satiation_scale 0,8 äter en mätt "
                 "population fortfarande med h = 0,375 vid sin nollpunkt. "
                 "Det är den siffran energibudgetgrinden räknar på.")

    d.bullets(
        "Vila är också att gömma sig",
        ["Den andel av en beslutande FG som väljer Vila i ett tick är "
         "skyddad från predation samma tick - och osynlig i nästa ticks "
         "observation.",
         ("-", "synlig andel = 1 - π_vila · (1 - golv)"),
         ("-", "Golvet verkar på själva gömda andelen, så det är aktivt "
               "för varje π_vila > 0, inte bara vid mättnad."),
         "Golvet är i första hand bytets eget visibility_floor, men kan "
         "sättas per PAR: detektionsförmåga tillhör paret, inte bytet. "
         "Tumlaren använder biosonar och påverkas inte av den visuella "
         "krypsis som skyddar sill mot sjöfågel - därför golv 0,95 för "
         "tumlare → pelagisk fisk mot 0,25 för torskfisk → pelagisk fisk.",
         "Den gömda andelen räknas bort både i predationen (bytet kan inte "
         "tas) och i observationen (bytet syns inte), med samma formel på "
         "båda ställena.",
         "Taket på uttaget ur en cell sätts av den BÄST detekterande "
         "predatorn, medan varje enskild predator är begränsad av vad den "
         "själv ser."],
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
        [("Uppslagning i FG:ns egen tabell", """
(biomass_factor, energy_factor) = interp( impact_table, Impact_p(c) )

m_X^Impact(c) = B_X(c) · Σ_p biomass_factor_p(c)
B_X(c)  ← max(0, B_X(c) − m_X^Impact(c))
R_X(c)  ← R_X(c) · B_X_ny(c) / B_X_gammal(c)
""")],
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
        [("Önskat intag, effektiv attackhastighet och uttag", """
B_synlig,Y(c)  = B_Y(c) · ( 1 − π_Y,vila(c) · (1 − golv_XY) )

                      a_XY · B_synlig,Y                (typ II)
f_XY(c)  =  ──────────────────────────────────────────────
            1 + a_XY · h_XY · B_synlig,Y + w_X · B_X(c)

D_XY(c)  = B_X(c) · π_X,ätY(c) · f_XY(c) · h_X(c)
skala(c) = min( 1 , 0.999 · B_synlig,Y(c) / Σ_X D_XY(c) )
intag_XY = D_XY(c) · skala(c)
"""),
         ("Bokföring", """
R_X(c) ← R_X(c) + Σ_Y intag_XY(c) · energy_content_Y · assimilation_XY
B_Y(c) ← B_Y(c) − Σ_X intag_XY(c)      R_Y skalas med samma kvot
""")],
        tail=[("-", "Typ III (specialister) ersätter B med B² i både "
                    "täljare och nämnare, vilket ger glest byte en refug."),
              ("-", "w_X · B_X(c) är Beddington-DeAngelis-interferens: "
                    "predatorns EGEN täthet i cellen sänker intaget per ton "
                    "predator. Det är den term som löste de vertikala "
                    "banden i rörelsemönstret."),
              ("-", "Faktorn 0,999 är ett hårt tak på uttaget ur en cell. "
                    "Noll är ett absorberande tillstånd - varje term i "
                    "ticken är multiplikativ i B - så en överbetad cell får "
                    "aldrig nå exakt noll.")],
        kicker="Holling + interferens")

    d.figure(
        "Funktionellt svar", draw_functional_response,
        kicker="Vad formeln gör",
        items=[("-", "a sätter lutningen vid låg täthet, h sätter taket: "
                     "en torskfisk kan aldrig äta mer än 1/25 = 0,04 ton "
                     "byte per ton torskfisk och tick."),
               ("-", "Interferensen w är vad som tar bort de vertikala "
                     "banden i rörelsemönstret: att packa ihop sig i en "
                     "cell lönar sig inte längre."),
               ("-", "Typ III väljs automatiskt för predatorer med exakt "
                     "en aktiv bytesgrupp - en generalist antas klara "
                     "bytesväxling via sin policy i stället.")],
        footnote="För h > 0 mättar intaget vid det fysiologiska taket "
                 "1/h oavsett bytestäthet; sjöfågel har h = 0 och saknar "
                 "därför tak utöver den synliga bytesbiomassan. Interferens "
                 "sänker hela kurvan men ändrar inte dess form; typ III tar "
                 "bort intaget vid låg täthet i stället.")

    d.formula(
        "Steg 4: energi efter action",
        [("Kostnad per action, före rörelse", """
kostnad_a(c) = B_X(c) · π_X,a(c) · resting_metabolism_X
               · cost_a · ( 1 + Σ_p energy_factor_p(c) )

     cost_a  =  resting_cost | feeding_cost | movement_cost

R_vila(c) = max(0, R_X(c)·π_vila − kostnad_vila)
R_ät(c)   = max(0, R_X(c)·π_ät   − kostnad_ät ) + energivinst(c)
R_rör(c)  = max(0, R_X(c)·π_rör  − kostnad_rör)
""")],
        lead=[("p", "Reserven delas upp på de tre actiongrupperna i "
                    "proportion till hur biomassan valde, varje grupp "
                    "betalar sin kostnad, och ätgruppen får predationens "
                    "energivinst.")],
        tail=[("-", "Alla actions kostar. Vila är den billigaste, inte "
                    "gratis."),
              ("-", "Påverkan verkar multiplikativt på kostnaden: buller "
                    "gör varje action dyrare i stället för att döda direkt."),
              ("-", "Efter rörelsen klipps reserven till [0, B_X · ME_X] "
                    "i varje cell.")],
        kicker="Metabolism")

    d.formula(
        "Steg 5: rörelse",
        [("Utflöde, kvarvarande massa och blandning", """
ut_d(c)   = B_X(c) · π_X,d(c) · v_X · mask_d(c)
kvar(c)   = B_X(c) − Σ_d ut_d(c)
B_X(c,+)  = kvar(c) + Σ_d ut_d( granne_d^-1(c) )

R följer med sin biomassa:  flyttas 20 % av massan följer
20 % av den actiongruppens reserv med.
E_X = R_X / B_X räknas om efter blandning; B = 0 ger E = 0.
""")],
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
         "utdöendetröskeln ställs in och återförs till källcellen. Utan den "
         "regeln blöder en diffunderande population biomassa genom "
         "utdöendesvepet snabbare än den svälter.",
         "Med --migration on lämnar kantflödet griden och återinförs "
         "fördelat längs kanten. Fördelningen koncentreras till de "
         "tyngst viktade kantcellerna (top-k) så att invandringen inte "
         "sprids ut under tröskeln och nollställs direkt.",
         "Sista steget i ticket nollställer celler där "
         "0 < B < extinction_threshold_factor · min_split_biomass och "
         "bokför massan som svältförlust. Motivet är numeriskt: under "
         "tröskeln närmar sig värdena float32:s subnormaler och bär ingen "
         "biologisk information, men stör observation, belöning och "
         "förlustredovisning.",
         "Kontinuerliga FG - plankton, med min_split_biomass 0 - berörs "
         "inte av vare sig argmax-kollaps eller utdöendetröskel."],
        kicker="Diskreta populationer i ett kontinuerligt fält")

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
               "benthic_community 0,0, övriga 0,0 som standard."),
         "Strömmen delar rörelsens masker - blockerat flöde stannar kvar - "
         "och är konservativ: ingen massa skapas eller försvinner."],
        kicker="current_response")

    d.formula(
        "Steg 7: tillväxt för beslutande FG - massbalanserad",
        [("Önskan, tak och debitering", """
s_X(c) = E_X(c) / ME_X            q_X(c) = s_X(c) − u_X

önskan   = B_X(c) · MG_X · q_X(c)                   ( q ≥ 0 )
E_ledig  = max( 0 , R_X(c) − u_X · ME_X · B_X(c) )
dB       = min( önskan , E_ledig / energy_content_X )
R_X(c)  ← R_X(c) − dB · energy_content_X
B_X(c)  ← B_X(c) + dB
"""),
         ("Svält (q < 0) är en förlust, inte ett köp", """
dB = B_X(c) · starve_rate_X · q_X(c)        R skalas med B
""")],
        lead=[("p", "Tillväxten är fortfarande GRINDAD av mättnaden precis "
                    "som i maj-utgåvan - B · MG · q är önskan - men önskan "
                    "är nu begränsad av den reserv som står över "
                    "underhållsnivån, och den reserven DEBITERAS.")],
        tail=[("-", "Utan debiteringen bär ny biomassa ett energiinnehåll "
                    "som aldrig tagits ur något: mätt till 198-344 % av "
                    "den massa som faktiskt åts, mot ett termodynamiskt "
                    "tak på 28,9 %."),
              ("-", "Konsekvens: ME_X / energy_content_X blir ett hårt tak "
                    "på den uppnåbara growth_rate. För djurplankton är "
                    "kvoten 0,150 mot growth_rate 0,0913 (0,100 och "
                    "91 % av gränsen före avsnitt 124)."),
              ("-", "--no-mass-balance återställer maj-utgåvans term. "
                    "Biblioteket är INTE kalibrerat för den.")],
        kicker="--mass-balance, på sedan september")

    d.figure(
        "Energibokföringen i ett tick", draw_energy_flow,
        kicker="Var massan kommer ifrån",
        items=[("-", "Spillet är (1 − assimilation_factor) av intaget: "
                     "0,65 för djurplankton på växtplankton, 0,90 för "
                     "tumlare på pelagisk fisk."),
               ("-", "Reserven är den enda bufferten. Alla actions tar ur "
                     "den, och påverkan höjer kostnaden multiplikativt."),
               ("-", "Först i tillväxtsteget blir reserv till biomassa, "
                     "till priset energy_content MJ per ton. En population "
                     "på eller under underhållsnivån kan därför inte växa "
                     "hur mycket föda som än finns.")],
        footnote="Mätt vid bibliotekets värden: av det djurplankton äter "
                 "andas 40,3 % bort som metabolism, 25,0 % byggs in som ny "
                 "biomassa och slutningsgapet är -0,3 %.")

    d.bullets(
        "Förluster: svält, mortalitet och påverkan",
        ["Ticken bokför tre förlustkanaler separat per FG, och de går att "
         "läsa ut i diagnostiken: predation, svält och påverkan.",
         "Naturlig mortalitet är täthetsoberoende och valfri: den kräver "
         "--mortality on och skalas av --mortality_multiplier "
         "(keep = 1 - FAKTOR · natural_mortality). Den tas ut på både "
         "biomassa och reserv, före tillväxttermen.",
         "Svält är katabolism: under underhållsnivån krymper biomassan med "
         "starve_rate · |q|, och reserven skalas med samma kvot.",
         "Massa som nollställs av utdöendetröskeln bokförs som svält - "
         "semantiskt närmast, och antalet händelser räknas separat.",
         "Mortalitet genom påverkan modelleras med påverkanskartor: fiske, "
         "jakt, kemikalier eller mekanisk dödlighet."],
        kicker="Tre kanaler")

    d.formula(
        "Steg 7: tillväxt för icke beslutande FG",
        [("Logistisk tillväxt mot bärkraft", """
r_eff(t)   = growth_rate · ( 1 + amplitud · sin(2π (t + fas) / period) )

dB(c)      = r_eff · B(c) · ( 1 − B(c) / CC(c) )  +  seed_rate · CC(c) · u
B(c,t+1)   = klipp( B(c) + dB(c) , 0 , CC(c) )
""")],
        lead=[("p", "Gäller funktionsgrupper vars energi inte modelleras "
                    "men som ingår i näringsväven. Betet har redan tagits "
                    "ut i steg 3, så B(c) är biomassan efter predation.")],
        tail=[("-", "seed_rate är ett återkoloniseringsgolv: en liten "
                    "andel av bärkraften läggs till i varje cell varje "
                    "tick, med en slumpfaktor 10^U(-1,1). Det gör noll "
                    "icke-absorberande för bottensamhället."),
              ("-", "Säsongsvariationen är avstängd i dag "
                    "(seasonal_amplitude 0): den mättes och sänkte "
                    "medelförsörjningen."),
              ("-", "Växtplankton: growth_rate 0,125 per 6 h mot bärkraft "
                    "24 ton/cell - en Eppley-takt vid 10-12 °C minus "
                    "obetad mikrozooplanktonbetning.")],
        kicker="Logistisk tillväxt")

    d.bullets(
        "Sammanfattning av bokföringen",
        ["Ny biomassa = biomassa efter påverkan, predation och förflyttning "
         "+ tillväxt - svält - naturlig mortalitet.",
         "Biomassa kan bara öka när energireserven tillåter det. För icke "
         "beslutande FG är bärkraften taket; för beslutande FG är "
         "reserven över underhållsnivån taket.",
         "Reservenergin följer alltid sin biomassa: varje gång biomassan "
         "skalas ned skalas reserven med samma kvot, och varje gång massa "
         "flyttas följer motsvarande andel av reserven med.",
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
         "Policynätverk kan konstrueras med flera metoder, t ex RL och "
         "ARS. Det som körs i dag är ARS: gradientfritt och "
         "parallelliserat över processer.",
         "Rollouten är 15 tick som standard och 100-200 tick i "
         "profilerna. Det är en förändring: maj-utgåvans motiv för "
         "tvåstegsrollouts gäller inte den nuvarande belöningen, som "
         "integrerar över hela rollouten."],
        kicker="Vad ARS ska åstadkomma")

    d.formula(
        "Fitness",
        [("Standard: total energi, integrerad över rollouten", """
E_tot(t) = Σ_c ( B_X(c,t) · energy_content_X + R_X(c,t) )

              1    H
Fitness_X = ───  Σ   log( ( E_tot(t) + ε ) / ( E_tot(0) + ε ) )
              H   t=1

ε = max( 1e-6 · E_tot(0) , 1e-9 )      H = --n_eval_ticks
"""),
         ("Kvar som --legacyreward: maj-utgåvans form", """
Fitness_X = α · log(B_H/B_0) + β · log(R_H/R_0) + κ · (t_överlevd / H)
""")],
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
        [("--local_reward: samma fråga, ställd per cell", """
A(c,t)   = B(c,t) · energy_content + R(c,t)
B(c,t+1) = Σ_d andel_d(c) · Q( mål(c,d) , t+1 )

belöning(c) = klipp( B(c,t+1) / A(c,t) , 0.2 , 5.0 )

Fitness_X = medel_t  Σ_c  w_c · log( belöning(c) )  / antal celler
""")],
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
              ("-", "w_c = A_c^θ. θ = 0 väger varje bebodd cell lika, "
                    "θ = 1 återger den globala energibelöningen.")],
        kicker="Skalinvariant per cell")

    d.formula(
        "ARS-uppdateringen",
        [("Per iteration, per beslutande FG med vikter θ", """
δ_1 ... δ_N ~ N(0, I)                        N = --n_deltas

F_i^+ = fitness( θ + σδ_i )    F_i^- = fitness( θ − σδ_i )
        - samma värld och samma seed för + och − (CRN)

topp b = de b par som har högst max(F_i^+, F_i^-)     b = N/2
σ_F    = standardavvikelsen över de utvalda parens fitness

                  η
θ  ←  θ  +  ─────────────  Σ_{i ∈ topp b} ( F_i^+ − F_i^- ) · δ_i
              b · σ_F
""")],
        tail=[("-", "ARS-V2: observationerna normaliseras med ett löpande "
                    "medel och en löpande varians per FG, delad mellan "
                    "processerna, och klipps till ±10."),
              ("-", "CRN - gemensamma slumptal - är det som gör "
                    "skillnaden F^+ - F^- meningsfull vid så få rollouts: "
                    "båda halvorna möter identiska startfält."),
              ("-", "Standard: η = 0,03, σ = 0,1, N = 10, "
                    "--rollouts_per_delta 1 (profilerna kör 3 världar per "
                    "delta och medelvärdesbildar).")],
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
             "som standard, med eller utan profil.",
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
                 "mareld2.yaml. I TRÄNINGSläge lottas energinivån dessutom "
                 "per cell kring underhållsnivån så att E[q_X(0)] ≈ 0; "
                 "inferensvärlden startar deterministiskt på 0,7 · ME_X.")

    d.bullets(
        "Checkpoints",
        ["En checkpoint är results/<körning>/policy_<fg>.pth och innehåller "
         "{'state_dict': ..., 'obs_stats': {...}} - vikterna OCH "
         "observationsnormaliseringens löpande statistik.",
         "--resume läser in dem och fortsätter; inference.py kör dem mot "
         "en deterministisk värld.",
         "",
         "Följande ändringar gör befintliga checkpoints ogiltiga - "
         "observationens form eller betydelse ändras och nätverket måste "
         "tränas om:",
         ("-", "Observability-matrisen"),
         ("-", "vilka impaktkartor som är observable"),
         ("-", "FG-uppsättningen i projektet"),
         ("-", "gridlayouten"),
         "En ändrad kalibrering i biblioteket gör dem inte formellt "
         "ogiltiga, men en policy som tränats i en väsentligt annan värld "
         "bör tränas om innan den citeras."],
        kicker="Vad som sparas och vad som bryter det")

    # ---------------------------------------------------------------- 8
    d.section("7", "Acceptanskriterier",
              "Två grindar som måste passeras innan ett resultat betyder "
              "något. Ingen av dem involverar ARS.")

    d.table(
        "Viabilitetsriggen",
        ["#", "Villkor", "Formel", "Standard"],
        [["1", "Överlevnad", "B_f(t) > 0 för alla t ≤ T", "-"],
         ["2", "Golv", "eq_f ≥ golv · B_f(0)", "golv = 0,10"],
         ["3", "Tak", "eq_f ≤ tak · B_f(0)", "tak = 10"],
         ["4", "Stationaritet",
          "max(eq_f/föreg_f, föreg_f/eq_f) ≤ max_drift", "max_drift = 2,0"]],
        [6, 26, 44, 24], size=12.5,
        lead="Är den konfigurerade världen självbärande över huvud taget? "
             "Frågan är en egenskap hos parametrarna och mekanismerna, inte "
             "hos policynätverken, så den mäts med FRYST beteende. eq_f är "
             "medelbiomassan i horisontens sista fönster.",
        footnote="Horisont 5 år (1 460 tick vid 6 h), 3 seeds, sämsta seed "
                 "avgör, sista fönstret 10 % av horisonten. Riggen har två "
                 "faktorer - beteende × spawngeometri - och utslaget kommer "
                 "från det normativa hörnet --spawn colocated "
                 "--behaviour greedy. python3 tools/viability.py, ~4 min.")

    d.bullets(
        "Energibudgetgrinden",
        ["Kan varje beslutande FG gå ihop energimässigt över sin RATION? "
         "Frågan besvaras utan rollout, på sekunder: "
         "python3 tools/probes/budget_gate.py",
         ("p", "ration  = a · h(u_X)          [ton byte / ton predator / "
               "tick]"),
         ("p", "kostnad = resting_metabolism · feeding_cost   [MJ / ton / "
               "tick]"),
         ("p", "krav    = kostnad / ration ≤ ransonens viktade "
               "energikvalitet"),
         "Grinden utvärderas över hela menyn, inte per par: motorn summerar "
         "intaget över alla byten och ger predatorn EN energivinst, så "
         "hungergrinden och födosökskostnaden tas ut en gång på totalen.",
         "Därför rapporterar grinden \"min share\" - minsta andel av det "
         "bästa bytet i dieten. Ett par som inte kan betala för sig på "
         "egen hand är lågkvalitativ föda, inte ren förlust.",
         ("-", "Tumlare kan inte leva på enbart torskfisk - och "
               "litteraturen säger detsamma (junk food-hypotesen). "
               "Magsäcksdata från Kattegatt/Skagerrak, 50-70 % sill och "
               "skarpsill i massa, klarar kravet.")],
        kicker="Går budgeten ihop?")

    d.table(
        "Status i dag",
        ["Funktionsgrupp", "--mortality on", "--mortality off"],
        [["gadoids", "UNDERKÄND - under golvet", "UNDERKÄND - utdöd @3924"],
         ["pelagic_fish", "UNDERKÄND - under golvet",
          "UNDERKÄND - under golvet"],
         ["porpoises", "UNDERKÄND - utdöd @71", "UNDERKÄND - utdöd @73"],
         ["phytoplankton", "UNDERKÄND - 0,054", "UNDERKÄND - 0,065"],
         ["benthic_community", "GODKÄND - 0,555", "GODKÄND - 0,533"],
         ["zooplankton", "GODKÄND - 0,270", "GODKÄND - 0,303"],
         ["samlat omdöme", "INTE VIABEL", "INTE VIABEL"]],
        [26, 28, 28], size=12.5,
        lead="Det normativa hörnet (--spawn colocated --behaviour greedy), "
             "20 × 20, 7 300 tick, 3 seeds, vid bibliotekets nuvarande "
             "värden. Siffran är eq_f / spawn mot ett golv på 0,10.",
        footnote="Utslaget drivs av tre sedan länge kända brister - "
                 "torskfisk och pelagisk fisk under golvet, tumlare utdöd - "
                 "och inget av det här arbetet har satt ut att åtgärda dem. "
                 "Det som ändrats i september är att djurplankton nu klarar "
                 "sig med marginal och att båda planktongrupperna är "
                 "stationära.")

    # ---------------------------------------------------------------- 9
    d.section("8", "Scenarier och körningar",
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
python3 train.py --project mareld2.yaml --species pelagic_fish \\
        --generations 1 --iter-per-gen 1 --workers 1        # röktest

python3 inference.py --project mareld2.yaml --run-name <namn>
python3 tools/biomass_html.py results/<namn>                # plots.html

python3 tools/viability.py                 # ~4 min, avslut 1 = inte viabel
python3 tools/probes/budget_gate.py        # sekunder, ingen rollout
python3 -m pytest tests/ -q
""")],
        tail=[("-", "Viktiga flaggor: --tick-length 1-6, --mortality on, "
                    "--mortality_multiplier, --migration on, --currents on, "
                    "--local_reward, --no-mass-balance, --policynetwork, "
                    "--resume, --visual, --rnd_baseline."),
              ("-", "--rnd_baseline ritar slumpmässiga referenskurvor i "
                    "den levande plotten: all = alla beslutande FG "
                    "slumpar, solo = en i taget. Utan en sådan referens "
                    "går det inte att säga om en policy gör något alls.")],
        kicker="Vad som körs")

    d.table(
        "Avsiktliga avvikelser från maj-utgåvan",
        ["Vad", "Varför"],
        [["Tillväxten är massbalanserad",
          "Den dokumenterade termen B·(1 + MG·q) skapar massa utan att "
          "debitera någon energi - mätt till 198-344 % av vad som åts, mot "
          "ett termodynamiskt tak på 28,9 %. Balansen slöts annars av "
          "svält, dvs en meningslös cykel."],
         ["Hungergrinden är kontinuerlig, inte en mask",
          "Maj-utgåvan hade BÅDE masken vid s = 0,8 och faktorn "
          "h = max(0, 1 - s/0,8), som redan är noll där masken slog till. "
          "Koden behöll faktorn - numera per FG via satiation_scale - och "
          "tog bort masken. Kvarvarande skillnad: massa som ändå väljer att "
          "äta betalar feeding_cost."],
         ["Fitness räknar biomassa och reserv i samma valuta",
          "α och β blir godtyckliga vikter mellan två storheter som redan "
          "har en växelkurs: energy_content. Med det gemensamma måttet "
          "försvinner viktningsfrågan."],
         ["Rollouterna är 15-200 tick, inte 2",
          "Motivet för två transitioner var en belöning som bara läste "
          "start och slut. Integralbelöningen läser varje tick, och en "
          "längre rollout blir informativ i stället för brusig."],
         ["Minnesutvidgningen N(c,t-1) är inte byggd",
          "Det enda tillstånd som förs mellan tick i dag är vilken andel "
          "som gömde sig förra ticket, och det används för synlighet - "
          "inte som policyinput."],
         ["Storleksstegen 3×3 → 60×60 ingår inte i det rekommenderade "
          "flödet",
          "--grid finns kvar och viabilitetsriggen kör 20×20, men "
          "variationen i träningen kommer från spawnstrategierna, "
          "startbiomassaintervallen och flera världar per delta."]],
        [26, 74], size=11.5, kicker="Dokumentet mot koden")

    d.bullets(
        "Öppna frågor",
        ["Tre funktionsgrupper klarar inte viabilitetskriteriet i det "
         "normativa hörnet: torskfisk och pelagisk fisk hamnar under "
         "golvet och tumlaren dör ut tidigt. Det är den viktigaste "
         "öppna punkten.",
         "Med tränade policyer är planktonnivåerna stationära över "
         "10 000 tick (växtplankton 0,56×, djurplankton 2,3× spawn), men "
         "kvoten djurplankton:växtplankton landar på 1,8:1 mot 0,44:1 vid "
         "spawn. Med fruset girigt beteende ligger nivåerna däremot en "
         "faktor 4-20 under spawn.",
         "Djurplanktonets resting_metabolism (2 %/dygn) motsvarar fastande "
         "djur och ligger i nederkanten av litteraturbandet; Ikedas "
         "regression ger mer för små copepoder (avsnitt 124).",
         "Policyerna i results/ är tränade i en väsentligt annan värld än "
         "den biblioteket beskriver efter september och bör tränas om "
         "innan de citeras.",
         ("p", "Hela historiken - varje kalibrering, A/B-test och öppet "
               "problem - ligger i mareld_resume.txt, avsnitt 0-124.")],
        kicker="Vad som återstår")

    d.section("", "Tack",
              "Frågor, invändningar och rättelser hör hemma i "
              "mareld_resume.txt lika mycket som i koden.")


# ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default="Strategi_sep.pdf")
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
        info["Subject"] = ("Uppdaterad strategi, september 2026 - beskriver "
                           "den mekanik simulatorn kör i dag")
    print(f"{args.out}: {deck.n} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
