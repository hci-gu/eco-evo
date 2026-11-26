"""Section 308.9: random spawn choice of the GPU training worlds
(train_gpu --spawn-mix, the viewer's "Random choice" boxes)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

FISH = {"benthivorous_fish", "mackerel", "planktivorous_fish", "predatory_fish", "whiting"}


def _trainer(mix, groups=None, worlds=8):
    from lib.gpu.config import EnvironmentBuilder, ProjectSpec
    from lib.gpu.trainer import TensorARSTrainer
    torch.set_num_threads(1)
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(12, 12),
                                          migration=True))
    return TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=2, hidden_dim=7,
                            worlds=worlds, spawn_mix=mix, spawn_mix_groups=groups)


def test_the_menu_is_the_projects_templates_without_env_driven():
    from lib.gpu.spawn import load_spawn_templates, random_spawn_menu
    menu = random_spawn_menu(load_spawn_templates("mareld2.yaml"))
    modes = {cfg["mode"] for _n, cfg in menu}
    assert modes == {"uniform", "perlin", "colony"}
    names = [n for n, _c in menu]
    assert {"PerlinPatchy", "Colonies25Wide", "Colonies10Wide", "25Colonies"} <= set(names)
    assert not {"FollowPhyto", "Followfish", "FollowDemersalFish"} & set(names)


def test_the_project_sets_share_and_groups_and_a_given_share_overrides():
    from lib.config.config_loader import load_config, spawn_mix_settings
    share, groups = spawn_mix_settings(load_config("mareld2.yaml"))
    assert share == 0.5 and set(groups) == FISH
    assert _trainer(None).spawn_mix == 0.5            # no flag: the project's share
    assert _trainer(0.2).spawn_mix == 0.2             # --spawn-mix overrides it
    assert _trainer(None, groups=["seals"]).spawn_random_groups == ("seals",)


def test_the_loader_refuses_a_bad_block():
    from lib.config.config_loader import check_spawn_mix, spawn_mix_settings
    assert spawn_mix_settings({}) == (0.0, [])
    with pytest.raises(ValueError):
        spawn_mix_settings({"simulation_settings": {"spawn_mix": {"share": 2}}})
    with pytest.raises(ValueError):
        spawn_mix_settings({"simulation_settings": {"spawn_mix": {"groups": "whiting"}}})
    with pytest.raises(ValueError):
        check_spawn_mix({"simulation_settings": {"spawn_mix": {"groups": ["nobody"]}}},
                        {"whiting": {}})


def test_the_cli_passes_none_without_the_flag():
    from lib.gpu.cli import trainer_options
    from train_gpu import build_parser
    parser = build_parser()
    args = parser.parse_args(["--project", "mareld2.yaml"])
    assert trainer_options(args)["spawn_mix"] is None
    args = parser.parse_args(["--project", "mareld2.yaml", "--spawn-mix", "0"])
    assert trainer_options(args)["spawn_mix"] == 0.0


def test_default_random_set_is_the_fish_and_own_spawn_is_not_repeated():
    t = _trainer(0.5)
    assert set(t.spawn_random_groups) == FISH
    options = t.spawner.random_options("benthivorous_fish")
    assert options[0][0] == "own"
    # The fish spawn with Perlin12: the menu's copy of it is left out.
    assert "Perlin12" not in [n for n, _c in options]
    assert not _trainer(0.0).spawn_random_groups


def test_share_zero_draws_the_default_worlds():
    a, b = _trainer(0.0), _trainer(0.0, groups=[])
    a.refresh_worlds()
    b.refresh_worlds()
    torch.testing.assert_close(a.world_biomass, b.world_biomass, rtol=0, atol=0)
    assert a.spawn_choice_record() == {}


def test_drawn_patterns_keep_totals_and_touch_only_the_random_groups():
    on, off = _trainer(1.0, worlds=24), _trainer(0.0, worlds=24)
    on.refresh_worlds()
    off.refresh_worlds()
    m = on.model
    torch.testing.assert_close(on.world_biomass.double().sum(-1), off.world_biomass.double().sum(-1),
                               rtol=1e-5, atol=1e-3)
    rows = [m.ids.index(f) for f in FISH]
    others = [g for g in range(m.G) if g not in rows]
    torch.testing.assert_close(on.world_biomass[:, others], off.world_biomass[:, others])
    record = on.spawn_choice_record()
    assert set(record) == FISH
    names = {n for row in record.values() for n in row}
    assert None not in names and len(names) >= 4
    # A world that drew "own" has the default field.
    g = m.ids.index("whiting")
    for w, name in enumerate(record["whiting"]):
        same = torch.equal(on.world_biomass[w, g], off.world_biomass[w, g])
        assert same == (name == "own")


def test_clumped_choices_are_clumped():
    t = _trainer(1.0, worlds=48)
    t.refresh_worlds()
    g = t.model.ids.index("benthivorous_fish")
    b = t.world_biomass[:, g].double()
    lloyd = (b.shape[1] * (b ** 2).sum(-1) / b.sum(-1) ** 2).numpy()
    names = np.array(t.spawn_choice_record()["benthivorous_fish"])
    assert lloyd[names == "25Colonies"].min() > lloyd[names == "own"].max()


def test_same_epoch_same_choice_and_a_changed_set_redraws():
    a, b = _trainer(0.5), _trainer(0.5)
    a.refresh_worlds(epoch=3)
    b.refresh_worlds(epoch=3)
    torch.testing.assert_close(a.world_biomass, b.world_biomass, rtol=0, atol=0)
    assert a.spawn_choice_record() == b.spawn_choice_record()
    assert a.last_world_epoch == 3
    assert a.set_spawn_random(["whiting", "porpoises"])
    assert a.last_world_epoch is None
    assert a.spawn_random_groups == ("porpoises", "whiting")
    assert not a.set_spawn_random(["porpoises", "whiting"])
    with pytest.raises(ValueError):
        a.set_spawn_random(["no_such_group"])


def test_a_dropdown_override_is_the_own_option():
    from inference import _load_spawn_templates, spawn_override_configs
    t = _trainer(1.0)
    configs = spawn_override_configs({"whiting": {"mode": "colony", "template": "25Colonies"}},
                                     _load_spawn_templates("mareld2.yaml"))
    t.set_spawn_overrides(configs)
    options = t.spawner.random_options("whiting")
    assert options[0][1]["mode"] == "colony"
    assert "25Colonies" not in [n for n, _c in options[1:]]
    assert "Perlin12" in [n for n, _c in options]


def test_viewer_boxes_reach_the_trainer(capsys):  # noqa: D103
    from types import SimpleNamespace
    from lib.gpu.visual import GPUTrainingVisualizer
    visual = GPUTrainingVisualizer.__new__(GPUTrainingVisualizer)
    visual.spawn_templates, visual.training_spawn_overrides = {}, {}
    visual.pump = lambda: None
    state = {"dirty": True}

    def consume():
        dirty, state["dirty"] = state["dirty"], False
        return dirty
    visual.viz = SimpleNamespace(get_spawn_overrides=lambda: {},
                                 consume_spawn_random_change=consume,
                                 get_spawn_random=lambda: ["whiting"])
    t = _trainer(0.5)
    visual.training_spawn(t)
    assert t.spawn_random_groups == ("whiting",)
    assert "Random spawn choice now on: whiting" in capsys.readouterr().out
    visual.training_spawn(t)
    assert capsys.readouterr().out == ""


@pytest.fixture
def viz(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.delenv("MARELD_VIZ_MAX_HEIGHT", raising=False)
    from lib.viz.pygame_viz import LiveVisualizer
    fids = ["planktivorous_fish", "whiting", "phytoplankton"]
    v = LiveVisualizer(fg_ids=fids, grid_shape=(12, 12), mode="train",
                       ndm_ids=["phytoplankton"], spawn_random=True)
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    v.set_b0_defaults({f: 10.0 for f in fids})
    v.update_biomass({f: np.ones((12, 12), dtype=np.float32) for f in fids}, tick=1)
    v._render_full()
    yield v
    v.close()


def test_every_panel_has_a_random_choice_box_above_mode(viz):
    viz.set_spawn_random(["whiting"])
    viz._render_full()
    boxes = {fid: rect for rect, fid in viz._spawn_random_rects}
    modes = {fid: rect for rect, fid, which in viz._spawn_dropdown_rects if which == "mode"}
    # Every panel in view has its box, right above its Mode row.
    assert {"planktivorous_fish", "whiting"} <= set(boxes) and set(boxes) == set(modes)
    for fid, rect in modes.items():
        assert boxes[fid][1] + boxes[fid][3] <= rect[1] + 1
    assert viz.get_spawn_random() == ["whiting"]
    x, y, w, h = boxes["planktivorous_fish"]
    viz._handle_click((x + 4, y + h // 2))
    assert viz.consume_spawn_random_change()
    assert viz.get_spawn_random() == ["planktivorous_fish", "whiting"]
    assert not viz.consume_spawn_random_change()


def test_no_box_without_the_spawn_mix(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from lib.viz.pygame_viz import LiveVisualizer
    v = LiveVisualizer(fg_ids=["whiting"], grid_shape=(12, 12), mode="train")
    if not v.enabled:
        pytest.skip("pygame display unavailable")
    v.set_b0_defaults({"whiting": 10.0})
    v.update_biomass({"whiting": np.ones((12, 12), dtype=np.float32)}, tick=1)
    v._render_full()
    assert v._spawn_random_rects == []
    v.close()
