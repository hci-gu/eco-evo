"""Minimal parser for OPeNDAP .ascii responses of THREDDS Grid variables.

Returns the ARRAY part of a Grid as a numpy array (the MAPS are ignored).
Only standard library + numpy.
"""
import re
import numpy as np


def parse_grid(text, var):
    """Parse '<var>.<var>[n1][n2]...' block from an OPeNDAP ascii response."""
    m = re.search(r"^%s\.%s((?:\[\d+\])+)\s*$" % (re.escape(var), re.escape(var)), text, re.M)
    if m is None:
        # plain array (not a Grid)
        m = re.search(r"^%s((?:\[\d+\])+)\s*$" % re.escape(var), text, re.M)
    if m is None:
        raise ValueError("variable %s not found in response" % var)
    shape = [int(s) for s in re.findall(r"\[(\d+)\]", m.group(1))]
    body = text[m.end():]
    # stop at the next variable header (a line that starts with a letter)
    stop = re.search(r"^[A-Za-z_]", body, re.M)
    if stop is not None:
        body = body[:stop.start()]
    vals = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        # drop leading index tokens "[i][j]," then parse the values
        line = re.sub(r"^(\[\d+\])+,\s*", "", line)
        for tok in line.split(","):
            tok = tok.strip()
            if tok:
                vals.append(float(tok))
    arr = np.array(vals, dtype=float)
    if arr.size != int(np.prod(shape)):
        raise ValueError("size mismatch for %s: %d vs %s" % (var, arr.size, shape))
    return arr.reshape(shape)
