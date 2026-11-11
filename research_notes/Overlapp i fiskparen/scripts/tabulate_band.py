"""Compact gate tables for gate_band.json: research_notes/"En synregel for fiskparen"/scripts/tabulate.py exec'd
unchanged except for the input file name (gate_variants.json -> gate_band.json).

Usage: python3 -I tabulate_band.py <repo_root> <tables_dir>
"""
import os
import sys

ROOT = os.path.abspath(sys.argv[1])
TB = os.path.join(ROOT, "research_notes", "En synregel for fiskparen", "scripts", "tabulate.py")
src = open(TB).read()
assert src.count('"gate_variants.json"') == 1
src = src.replace('"gate_variants.json"', '"gate_band.json"')
exec(compile(src, TB, "exec"), {"__name__": "__main__", "__file__": TB})
