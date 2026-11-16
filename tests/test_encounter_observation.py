"""Section 275: the encounter observation (design note research_notes/
Observabilitetsparen section 5).

Per tick, observer, target, region and cell one uniform picks: met (the
density), entered but met nothing (0) or, in a neighbour, never there (the
own-cell reading after its own draw). Diffuse components (kappa inf) are met
wherever the target is present. The uniforms are a counter-based hash shared
bit for bit by both engines; the loader enforces the rule on every
observation pair once one carries it.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import yaml
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.config import config_loader  # noqa: E402
from lib.environments.ecosystem_env import encounter, observations  # noqa: E402
from lib.environments.ecosystem_env.constants import DIRECTIONS  # noqa: E402
from lib.gpu import random as gpu_random  # noqa: E402
from test_edge_closure import DAYLIGHT, STAGES  # noqa: E402
from test_gpu_ecosystem import _assert_parity, device, make_env  # noqa: E402,F401

PROJECT = ROOT / "mareld2.yaml"
INF = float("inf")
ALWAYS = {"own": ((1.0,), (INF,)), "neighbour": ((1.0,), (INF,))}


def _env(table, **kw):
    """Fixture: ``table`` maps observer -> {target: components}."""
    kw.setdefault("min_split", 0.0)
    kw.setdefault("extinction_factor", 0.0)
    env = make_env(regen={fid: {"encounter": t} for fid, t in table.items()}, **kw)
    env.current_world_seed = 12345
    env.build_static_caches()
    return env


def _full(a=None, b=None):
    """Every pair of the fixture (a observes a, b, c; b observes b, a)."""
    ta = {"a": ALWAYS, "b": ALWAYS, "c": ALWAYS}
    tb = {"b": ALWAYS, "a": ALWAYS}
    ta.update(a or {})
    tb.update(b or {})
    return {"a": ta, "b": tb}


def _batch(env):
    return observations.build_observation_batch(env).reshape(env.N_dm, env.H, env.W, -1)


def _slots(env, i):
    n_other = len(env.obs_others_idx[i]) + len(env.obs_juvenile_idx[i])
    centre_dim, nbr_dim = 2 + n_other, 1 + n_other
    return n_other, centre_dim, nbr_dim


# ---- the random numbers --------------------------------------------------------

def test_hash_twin_is_bit_identical_to_the_tensor_engine():
    rng = np.random.default_rng(3)
    x = rng.integers(-2 ** 62, 2 ** 62, 4000)
    np.testing.assert_array_equal(encounter.hash32(x).astype(np.int64),
                                  gpu_random.hash32(torch.tensor(x)).numpy())
    for world, tick in ((0, 0), (12345, 77), (2 ** 33 + 5, 1459)):
        key = encounter.tick_key(world, tick)
        tkey = gpu_random.fold_in(gpu_random.fold_in(torch.tensor([world]), tick),
                                  encounter.STREAM)
        assert int(key) == int(tkey[0])
        counters = np.arange(0, 50000, 3)
        u = encounter.uniform_at(key, counters)
        np.testing.assert_array_equal(u, gpu_random.uniform_at(tkey, torch.tensor(counters))[0].numpy())
        assert 0.0 < u.min() and u.max() < 1.0


def test_int32_hash_matches_the_reference_bits(device):
    rng = np.random.default_rng(5)
    key = encounter.tick_key(987654321, 42)
    counters = rng.integers(0, 2 ** 40, 20000)
    want = encounter.uniform_at(key, counters)
    cell = encounter.hash32(counters + 1).astype(np.uint32).view(np.int32)
    k32 = torch.tensor([int(key)], dtype=torch.int64).to(torch.int32)
    x = k32.to(device)[:, None] ^ torch.tensor(cell, device=device)[None]
    np.testing.assert_array_equal(gpu_random.uniform_bits_i32(x)[0].cpu().numpy(), want)


# ---- the rule on the reference engine -----------------------------------------

def test_always_met_is_todays_observation():
    plain = _env({})
    env = _env(_full())
    assert env._has_encounter and not plain._has_encounter
    np.testing.assert_array_equal(_batch(plain), _batch(env))


def test_never_entered_neighbours_read_the_own_cell():
    stay = {"own": ((1.0,), (INF,)), "neighbour": ((0.0,), (INF,))}
    env = _env(_full(a={"c": stay, "a": stay}))
    plain = _env({})
    out, base = _batch(env), _batch(plain)
    i = env.dm_ids.index("a")
    others = [env.global_fg_order[int(j)] for j in env.obs_others_idx[i]]
    k_c = 1 + others.index("c")
    n_other, centre_dim, nbr_dim = _slots(env, i)
    inside = {d: observations.shift_field(np.ones((env.H, env.W), np.float32), d)
              for d in DIRECTIONS}
    for q, d in enumerate(DIRECTIONS):
        for k, centre in ((0, 0), (k_c, 1 + k_c)):
            np.testing.assert_array_equal(out[i, :, :, centre_dim + q * nbr_dim + k],
                                          inside[d] * base[i, :, :, centre])


def test_a_pair_never_met_leaks_nothing():
    """Own cell 0, and the neighbours fall back to that 0 - never to c."""
    blind = {"own": ((1.0,), (0.0,)), "neighbour": ((0.0,), (INF,))}
    env = _env(_full(a={"b": blind}))
    out = _batch(env)
    i = env.dm_ids.index("a")
    others = [env.global_fg_order[int(j)] for j in env.obs_others_idx[i]]
    k_b = 1 + others.index("b")
    n_other, centre_dim, nbr_dim = _slots(env, i)
    assert _batch(_env({}))[i, :, :, 1 + k_b].max() > 0
    assert out[i, :, :, 1 + k_b].max() == 0.0
    for q in range(4):
        assert out[i, :, :, centre_dim + q * nbr_dim + k_b].max() == 0.0


def test_entered_but_not_met_reads_zero():
    empty = {"own": ((1.0,), (INF,)), "neighbour": ((1.0,), (0.0,))}
    env = _env(_full(a={"b": empty}))
    out = _batch(env)
    i = env.dm_ids.index("a")
    others = [env.global_fg_order[int(j)] for j in env.obs_others_idx[i]]
    k_b = 1 + others.index("b")
    n_other, centre_dim, nbr_dim = _slots(env, i)
    for q in range(4):
        assert out[i, :, :, centre_dim + q * nbr_dim + k_b].max() == 0.0


def test_the_met_frequency_follows_the_encounter_count():
    b = make_env().fgs["b"].biomass
    density = float(np.median(b[b > 0]))
    kappa = np.log(2.0) / density               # p = 1/2 at the median cell
    half = {"own": ((1.0,), (kappa,)), "neighbour": ((1.0,), (INF,))}
    env = _env(_full(a={"b": half}))
    i = env.dm_ids.index("a")
    others = [env.global_fg_order[int(j)] for j in env.obs_others_idx[i]]
    k_b = 1 + others.index("b")
    true = _batch(_env({}))[i, :, :, 1 + k_b]
    want = 1.0 - np.exp(-kappa * true)
    hits = np.zeros_like(true)
    ticks = 400
    for t in range(ticks):
        env.tick_count = t
        hits += _batch(env)[i, :, :, 1 + k_b] > 0
    np.testing.assert_allclose(hits / ticks, want, atol=0.08)
    assert abs(hits.sum() / ticks - want.sum()) < 0.02 * true.size


def test_juvenile_channels_follow_the_group_draw():
    blind = {"own": ((1.0,), (0.0,)), "neighbour": ((0.0,), (INF,))}
    env = _env(_full(b={"a": blind}), stages={"a": dict(STAGES)}, daylight=dict(DAYLIGHT))
    i = env.dm_ids.index("b")
    enc = env.obs_encounter[i]
    j_a = env.global_fg_order.index("a")
    k_j = enc["targets"].index(j_a, 2)          # a's J after the biomasses
    assert enc["source"][k_j] == enc["targets"].index(j_a)
    out = _batch(env)
    n_other, centre_dim, nbr_dim = _slots(env, i)
    assert out[i, :, :, 1 + k_j].max() == 0.0
    for q in range(4):
        assert out[i, :, :, centre_dim + q * nbr_dim + k_j].max() == 0.0


# ---- the tensor engine -----------------------------------------------------------

def _random_table():
    half = {"own": ((0.6, 0.4), (40.0, INF)), "neighbour": ((0.3, 0.2), (15.0, INF))}
    rare = {"own": ((1.0,), (2.0,)), "neighbour": ((0.4,), (1.0,))}
    return _full(a={"b": half, "c": rare, "a": {"own": ((1.0,), (INF,)),
                                               "neighbour": ((0.35,), (INF,))}},
                 b={"a": half})


@pytest.mark.parametrize("staged", [False, True])
def test_tensor_engine_matches_reference(device, staged):
    kw = dict(stages={"a": dict(STAGES)}, daylight=dict(DAYLIGHT)) if staged else {}
    env = _env(_random_table(), migration=True, **kw)
    model = _assert_parity(env, device)
    assert model.encounter_on


def test_tensor_observation_is_exact_on_the_same_state(device):
    from lib.gpu.ecosystem import TensorEcosystem
    env = _env(_random_table(), stages={"a": dict(STAGES)}, daylight=dict(DAYLIGHT))
    model = TensorEcosystem(env, device)
    b, r, hidden = model.import_state([env])
    starts = model.light_starts([env])
    juv = model.import_juvenile([env])
    for tick in (0, 5, 333):
        env.tick_count = tick
        t = torch.tensor(tick, device=model.device)
        obs = model.observations(b, r, hidden, model.light_index(starts, t), None, juv, tick=t)
        ref = observations.build_observation_batch(env)
        np.testing.assert_allclose(obs[0].cpu().numpy(), ref, rtol=1e-6, atol=1e-7)


# ---- the loader ------------------------------------------------------------------

def _library():
    return config_loader.load_config(ROOT / "fgconfig" / "fg_library.yaml")


def _load(lib, tick_hours=6):
    result = config_loader.load_project_config(
        str(PROJECT), grid_size=(8, 8), seed=1, spawn_seed=1,
        library_config=copy.deepcopy(lib), tick_hours=tick_hours)
    return result[0] if isinstance(result, tuple) else result


def _with_encounter(lib, components=None):
    comp = components or {"encounter_own": [[0.5, 4.0], [0.5, INF]],
                          "encounter_neighbour": [[0.3, 2.0]]}
    for key, entry in lib["interaction_definitions"].items():
        if "_observes_" in key and (entry or {}).get("observes"):
            entry.pop("neighbour_weight", None)
            entry.update(copy.deepcopy(comp))
    return lib


def test_loader_reads_and_rescales_the_components():
    lib = _with_encounter(_library())
    fgs = _load(lib)
    enc = fgs["planktivorous_fish"].params["encounter"]["porpoises"]
    assert enc["own"] == ((0.5, 0.5), (4.0, INF))
    assert enc["neighbour"] == ((0.3,), (2.0,))
    half = _load(lib, tick_hours=3)["planktivorous_fish"].params["encounter"]["porpoises"]
    assert half["own"] == ((0.5, 0.5), (2.0, INF))
    assert half["neighbour"] == ((0.3,), (1.0,))


def test_loader_refuses_a_partial_or_mixed_rule():
    lib = _with_encounter(_library())
    inter = lib["interaction_definitions"]
    key = "planktivorous_fish_observes_porpoises"
    bad = copy.deepcopy(lib)
    for k in config_loader.ENCOUNTER_KEYS:
        bad["interaction_definitions"][key].pop(k)
    with pytest.raises(ValueError, match=key):
        _load(bad)
    bad = copy.deepcopy(lib)
    bad["interaction_definitions"][key]["neighbour_weight"] = 0.1
    with pytest.raises(ValueError, match="exclude each other"):
        _load(bad)
    bad = copy.deepcopy(lib)
    bad["interaction_definitions"][key].pop("encounter_neighbour")
    with pytest.raises(ValueError, match="go together"):
        _load(bad)
    for own, nbr, match in (([[0.5, 1.0]], [[0.3, 2.0]], "sum to 1"),
                            ([[1.0, -1.0]], [[0.3, 2.0]], "kappa"),
                            ([[1.0, 1.0]], [[0.7, 1.0], [0.6, 1.0]], "sum to")):
        bad = copy.deepcopy(lib)
        bad["interaction_definitions"][key].update(encounter_own=own, encounter_neighbour=nbr)
        with pytest.raises(ValueError, match=match):
            _load(bad)
    assert inter[key]["encounter_own"]       # the good library still loads
    _load(lib)


# ---- the library -------------------------------------------------------------------

NOTES = ROOT / "research_notes" / "Observabilitetsparen" / "tables"


def _listing():
    import re
    out = {}
    for line in open(NOTES / "encounter_components_library.txt"):
        m = re.match(r"(\S+) encounter_own: (\[.*\]) encounter_neighbour: (\[.*\])$", line.strip())
        if m:
            import yaml
            out[m.group(1)] = (yaml.safe_load(m.group(2)), yaml.safe_load(m.group(3)))
    return out


def test_the_library_carries_the_derived_components():
    inter = _library()["interaction_definitions"]
    listing = _listing()
    observed = [k for k, e in inter.items() if "_observes_" in k and (e or {}).get("observes")]
    assert set(listing) <= set(observed)
    for key, (own, nbr) in listing.items():
        assert inter[key]["encounter_own"] == own, key
        assert inter[key]["encounter_neighbour"] == nbr, key
        assert "neighbour_weight" not in inter[key], key
    # Every observation pair of the manifest's active groups is covered
    # (the loader would refuse the project otherwise).
    fgs = _load(_library())
    assert fgs["porpoises"].params["encounter"]["planktivorous_fish"]["own"] == (
        (0.5, 0.5), (1.12, 56000.0))


def test_neighbour_components_reproduce_the_section_273_weights():
    """P(met in a neighbour) at the box mean against the exact CRW w of 1.3
    (tables/neighbour_weight_library.txt): the path is binned, within 5 %."""
    import yaml
    weights = {}
    for line in open(NOTES / "neighbour_weight_library.txt"):
        key, value = line.split()
        if value != "-":
            weights[key] = float(value)
    project = yaml.safe_load(open(PROJECT, encoding="utf-8-sig"))
    start = {}
    for cat in ("decision_makers", "non_decision_makers"):
        for fg in project.get(cat) or []:
            start[fg["group_id"]] = fg
    inter = _library()["interaction_definitions"]
    density = {"planktivorous_fish": 4.72, "benthivorous_fish": 1.53, "whiting": 0.446,
               "predatory_fish": 0.617, "mackerel": 0.85, "porpoises": 84 / 3600,
               "seals": 15 / 3600}
    checked = 0
    for key, w in weights.items():
        obs, tgt = key.split("_observes_", 1)
        if tgt not in density or obs == tgt or key not in inter:
            continue
        comps = inter[key]["encounter_neighbour"]
        p = sum(s * (1.0 if np.isinf(k) else 1.0 - np.exp(-k * density[tgt])) for s, k in comps)
        assert p == pytest.approx(w, rel=0.05, abs=0.002), key
        checked += 1
    assert checked > 20


def test_the_editor_keeps_the_components_on_save():
    import io
    import yaml
    sys.path.insert(0, str(ROOT / "fgconfig"))
    fgconfig = pytest.importorskip("fgconfig")
    path = ROOT / "fgconfig" / "fg_library.yaml"
    data = fgconfig.yaml.load(open(path, encoding="utf-8-sig"))
    buf = io.BytesIO()
    fgconfig.yaml.dump(data, buf)
    saved = yaml.safe_load(buf.getvalue().decode())["interaction_definitions"]
    for key, (own, nbr) in _listing().items():
        assert saved[key]["encounter_own"] == own and saved[key]["encounter_neighbour"] == nbr


def test_the_editor_cells_round_trip_and_check_like_the_loader():
    sys.path.insert(0, str(ROOT / "fgconfig"))
    fgconfig = pytest.importorskip("fgconfig")
    App = fgconfig.FGConfigApp
    for key, (own, nbr) in _listing().items():
        for data_key, value in (("encounter_own", own), ("encounter_neighbour", nbr)):
            rows, error = App._parse_encounter(data_key, App._encounter_text(value))
            assert error is None and App._same_components(value, rows), key
    assert App._parse_encounter("encounter_own", "") == (None, None)
    assert App._encounter_text([[1, INF]]) == "1.0:inf"
    for data_key, text, match in (("encounter_own", "0.5:1.0", "sum to 1"),
                                  ("encounter_own", "1:-1", "kappa"),
                                  ("encounter_neighbour", "0.7:1, 0.6:1", "sum to"),
                                  ("encounter_own", "1", "share:kappa")):
        rows, error = App._parse_encounter(data_key, text)
        assert rows is None and match in error, text


def _active_ids():
    project = yaml.safe_load(open(PROJECT, encoding="utf-8-sig"))
    def ids(cat):
        return [fg["group_id"] for fg in project.get(cat) or [] if not fg.get("muted")]
    dms = ids("decision_makers")
    return dms, dms + ids("non_decision_makers")


def test_the_editor_check_agrees_with_the_loader():
    lib = _library()
    dms, active = _active_ids()
    inter = lib["interaction_definitions"]
    assert config_loader.missing_encounter_pairs(inter, dms, active) == []
    _load(lib)
    for key in ("planktivorous_fish_observes_porpoises", "seals_observes_seals"):
        bad = copy.deepcopy(lib)
        for k in config_loader.ENCOUNTER_KEYS:
            bad["interaction_definitions"][key].pop(k)
        assert config_loader.missing_encounter_pairs(
            bad["interaction_definitions"], dms, active) == [key]
        with pytest.raises(ValueError, match=key):
            _load(bad)
    # A newly observed pair without the rule is refused by both.
    new = next(k for k, d in inter.items() if "_observes_" in k
               and k.split("_observes_")[0] in dms and k.split("_observes_")[1] in active
               and not (d or {}).get("observes"))
    bad = copy.deepcopy(lib)
    bad["interaction_definitions"][new]["observes"] = True
    assert config_loader.missing_encounter_pairs(
        bad["interaction_definitions"], dms, active) == [new]
    with pytest.raises(ValueError, match=new):
        _load(bad)
    # Without the rule anywhere nothing is missing.
    plain = copy.deepcopy(lib)
    for d in plain["interaction_definitions"].values():
        for k in config_loader.ENCOUNTER_KEYS:
            (d or {}).pop(k, None)
    assert config_loader.missing_encounter_pairs(
        plain["interaction_definitions"], dms, active) == []


def test_the_editor_apply_refuses_an_incomplete_rule(tmp_path):
    tk = pytest.importorskip("tkinter")
    sys.path.insert(0, str(ROOT / "fgconfig"))
    fgconfig = pytest.importorskip("fgconfig")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    shown = []
    saved = []
    try:
        app = fgconfig.FGConfigApp(root)
        app.load_project_from_path(str(PROJECT))
        app.library_path = str(tmp_path / "lib.yaml")
        app.save_yaml = lambda data, path: saved.append(path)
        fgconfig.messagebox.showinfo = lambda *a, **k: shown.append(("info",) + a)
        fgconfig.messagebox.showerror = lambda *a, **k: shown.append(("error",) + a)
        app.refresh_matrix()
        key = "planktivorous_fish_observes_porpoises"
        inter = app.global_library["interaction_definitions"]
        before = list(inter[key]["encounter_own"])
        # Untouched: saved, nothing rewritten.
        app.apply_matrix_changes()
        assert saved and shown[-1][0] == "info"
        assert app.global_library["interaction_definitions"][key]["encounter_own"] == before
        # An emptied pair: refused, the library restored, nothing saved.
        saved.clear()
        for k in fgconfig.ENCOUNTER_KEYS:
            app.matrix_entries[key][k].set("")
        app.apply_matrix_changes()
        assert not saved and shown[-1][0] == "error" and key in shown[-1][2]
        assert app.global_library["interaction_definitions"][key]["encounter_own"] == before
        # Only one region: a cell error; the pair keeps its values.
        app.matrix_entries[key]["encounter_own"].set("1:2.5")
        app.apply_matrix_changes()
        assert saved and "go together" in shown[-1][2]
        assert app.global_library["interaction_definitions"][key]["encounter_own"] == before
        # Both regions edited: written.
        app.matrix_entries[key]["encounter_neighbour"].set("0.4:inf")
        app.apply_matrix_changes()
        entry = app.global_library["interaction_definitions"][key]
        assert [list(r) for r in entry["encounter_own"]] == [[1.0, 2.5]]
        assert [list(r) for r in entry["encounter_neighbour"]] == [[0.4, INF]]
    finally:
        root.destroy()
