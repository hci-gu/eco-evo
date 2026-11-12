"""budget_gate's field-spread diet gate with the benthivores also spread as a PREDATOR.

FIELD_BED_SHARES (lib/diagnostics/field_diet.py) spreads only the water column and whiting
as predators; the benthivores are spread only as prey (by stage). Here the module dict gets
a benthivore entry before tools/probes/budget_gate.py runs, so the gate prints the
benthivores' encounter-implied diet at a given bed spread. SPREAD selects it:
  gov   - GOV adults in the box (FIELD_STAGE_BED_SHARES), the field reading
  fix98 - fix98's realised year-3 biomass share (bed_share_fix98.txt), for comparison
  fix97 - fix97's realised year-3 biomass share (bed_share_fix97.txt)
Nothing in the repository is changed."""
import os
import runpy
import sys

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from lib.diagnostics import field_diet  # noqa: E402

SPREADS = {
    'gov': ((0.308, 0.441, 0.191, 0.060), 'GOV adults in the box (predator side, scratch)'),
    'fix98': ((0.335, 0.472, 0.106, 0.087), 'fix98 realised year 3 (scratch)'),
    'fix97': ((0.020, 0.303, 0.520, 0.156), 'fix97 realised year 3 (scratch)'),
}
field_diet.FIELD_BED_SHARES['benthivorous_fish'] = SPREADS[os.environ.get('SPREAD', 'gov')]
sys.argv = ['budget_gate.py']
runpy.run_path(os.path.join(ROOT, 'tools/probes/budget_gate.py'), run_name='__main__')
