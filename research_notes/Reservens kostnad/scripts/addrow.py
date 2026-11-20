"""Append one row to tables/cost_raw.csv (creates header if missing).
Usage: python3 -I addrow.py source doi species proxy mechanism method value unit tag note"""
import csv, os, sys
here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, '..', 'tables', 'cost_raw.csv')
cols = ['source','doi','species','proxy','mechanism','method','value','unit','tag','note']
args = sys.argv[1:]
if len(args) != len(cols):
    sys.exit('need %d args, got %d' % (len(cols), len(args)))
new = not os.path.exists(path)
with open(path, 'a', newline='') as f:
    w = csv.writer(f)
    if new:
        w.writerow(cols)
    w.writerow(args)
