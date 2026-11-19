"""Shared helpers for the section 280 drivers (research_notes/"Djupljus i synparen").

Every older note script is read from its own file and executed in memory; the only changes are exact string
replacements, each anchor asserted to occur the stated number of times. Nothing in an existing note is written.
"""
import contextlib
import io
import os
import sys

RN_REL = "research_notes"


def patched_source(path, subs, upto=None):
    """source of `path` (cut before `upto` when given) with each (old, new, count) replaced; anchors asserted."""
    src = open(path, encoding="utf-8").read()
    if upto is not None:
        assert src.count(upto) == 1, (path, upto)
        src = src[:src.index(upto)]
    for old, new, n in subs:
        c = src.count(old)
        assert c == n, (os.path.basename(path), old, c, n)
        src = src.replace(old, new)
    return src


def run_patched(path, subs, argv, name="patched", upto=None, quiet=False, extra=None, file_as=None):
    """exec the patched source with sys.argv = argv; returns the namespace. file_as: the __file__ the code sees."""
    src = patched_source(path, subs, upto)
    ns = {"__name__": name, "__file__": file_as or path}
    if extra:
        ns.update(extra)
    save = list(sys.argv)
    sys.argv = list(argv)
    try:
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, path, "exec"), ns)
        else:
            exec(compile(src, path, "exec"), ns)
    finally:
        sys.argv = save
    return ns
