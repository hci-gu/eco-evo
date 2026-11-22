"""Section 298: field_diet.potential_ration and budget_gate's potential-ration rows."""
import os
import subprocess
import sys

import numpy as np
import pytest

from lib.diagnostics.field_diet import potential_ration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_single_prey_is_holling_type_ii():
    a, h, b = 0.02, 3.0, 5.0
    total, per_prey, ceil = potential_ration([[a]], [h], [[b]], [1.0], ticks_per_year=100)
    assert total == pytest.approx(a * b / (1 + a * h * b) * 100)
    assert per_prey[0] == pytest.approx(total)
    assert ceil == pytest.approx(a * h * b / (1 + a * h * b))


def test_shared_handling_and_ceiling_bound():
    attack = np.array([[0.5, 0.01], [0.3, 0.02]])
    prey = np.array([[40.0, 2.0], [10.0, 1.0]])
    handling = np.array([2.0, 4.0])
    total, per_prey, ceil = potential_ration(attack, handling, prey, [0.5, 0.5], ticks_per_year=1)
    lam = attack * prey
    denom = 1 + (handling[:, None] * lam).sum(0)
    assert per_prey == pytest.approx((lam / denom * 0.5).sum(1))
    # handling time spent never exceeds the tick
    assert (per_prey * handling).sum() <= 1.0
    assert 0.0 < ceil < 1.0


def test_monotone_in_attack_and_interference_lowers():
    prey, h, w = [[3.0, 8.0]], [2.0], [0.3, 0.7]
    lo = potential_ration([[0.01, 0.01]], h, prey, w)[0]
    hi = potential_ration([[0.05, 0.05]], h, prey, w)[0]
    assert hi > lo
    crowded = potential_ration([[0.05, 0.05]], h, prey, w, interference=0.1, predator=[2.0, 2.0])[0]
    assert crowded < hi


def test_type3_and_weights():
    a, h, b = 0.01, 1.0, 4.0
    t3 = potential_ration([[a]], [h], [[b]], [1.0], type3=[True], ticks_per_year=1)[0]
    assert t3 == pytest.approx(a * b * b / (1 + a * h * b * b))
    # weights pick the cell
    only_second = potential_ration([[a, a]], [h], [[1.0, 9.0]], [0.0, 1.0], ticks_per_year=1)[0]
    assert only_second == pytest.approx(a * 9.0 / (1 + a * h * 9.0))


def test_budget_gate_prints_the_potential_rows():
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'probes', 'budget_gate.py')],
                         capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    text = out.stdout
    assert "potential ration at the field spread vs field Q/B (section 298)" in text
    tail = text.split("(section 298)")[1]
    assert "predatory_fish (saithe, GOV in the box)" in tail
    assert "Potential ration" in tail
