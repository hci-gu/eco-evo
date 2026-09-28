"""Shared CLI and legend conventions for the random-action baseline.

``--rnd_baseline [all|solo|none]`` overlays ``_rnd`` curves from parallel
worlds in which decision makers act uniformly at random (section 62 of
``mareld_resume.txt``). ``train.py`` defines the semantics; ``inference.py``
and ``train_gpu.py`` use these helpers so the three entry points agree.
"""

RND_BASELINE_MODES = ("all", "solo", "none")


def add_rnd_baseline_argument(parser, *, extra_help=""):
    """Register ``--rnd_baseline``: a bare flag means ``all``, absent ``none``."""
    parser.add_argument(
        "--rnd-baseline", "--rnd_baseline", dest="rnd_baseline",
        nargs="?", const="all", default="none", choices=RND_BASELINE_MODES,
        help="Overlay a random-action baseline in the live plot; legend "
             "entries are suffixed with '_rnd'. 'all' (default when the "
             "flag is given alone): one parallel world where EVERY "
             "decision maker acts uniformly at random (mask-respecting); "
             "every FG, NDMs included, gets an _rnd curve. 'solo': "
             "leave-one-out, one parallel world per decision maker k in "
             "which only k is random and the other DMs keep their trained "
             "policies; only DMs get _rnd curves, each read from its own "
             "world. Costs one extra world per DM. 'none': off."
             + extra_help)


def rnd_baseline_plot_ids(mode, fg_ids, dm_ids):
    """Legend ids for the ``_rnd`` series of a baseline mode, or None.

    ``all`` reports every FG, because with all DMs random each NDM has one
    well-defined cascade signal. ``solo`` reports DMs only: an NDM differs
    between the N leave-one-out worlds, so a single curve for it would be
    ambiguous (section 62.5).
    """
    mode = str(mode or "none").lower()
    if mode == "all":
        return [fid + "_rnd" for fid in fg_ids]
    if mode == "solo":
        return [fid + "_rnd" for fid in dm_ids]
    return None
