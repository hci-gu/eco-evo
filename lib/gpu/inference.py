"""Inference rollouts on the tensor engine (sections 234, 235).

``inference.py --engine gpu`` builds its worlds exactly as the CPU path
does (``build_env``: spawn, b0 and spawn overrides, the inference impact
maps, the start day) and uploads that state once. The frozen policies
and frozen observation statistics are packed into a :class:`RolloutRunner`
with one world and one candidate per world the CPU path would step: the
trained world first, then each random-action baseline (``--rnd_baseline``)
as a candidate whose random decision makers have all-zero weights. Zero
weights give zero logits, a uniform softmax over the allowed actions -
exactly ``inference.RandomPolicy`` - so the baselines ride in the same
batch instead of being stepped one after another.

Two ways to run it:

* :func:`run_gpu_inference` - headless: every group's total biomass is
  written inside the tick, so the rollout replays as CUDA graphs and the
  host reads it only at the progress lines.
* :class:`GpuStepper` - the viewer: one tick at a time, after which the
  state and the diagnostic books are copied into the CPU envs, so
  ``inference.run_inference`` and the viewer read them as if the CPU
  engine had stepped those envs.

The tensor engine matches the CPU engine tick by tick to float32
rounding (tests/test_gpu_ecosystem.py), not bit for bit; over a long
rollout the two drift apart the way any float change does. Producer
seeding draws from the tensor engine's own counter-based RNG, so a
project with seeding gives a different (equally valid) world.
"""

import numpy as np
import torch

from lib.gpu.ecosystem import TensorEcosystem
from lib.gpu.policy import PolicyBank
from lib.gpu.rollout import RolloutRunner
from lib.runners.policy import PolicyNetwork


class InferenceRunner(RolloutRunner):
    """A :class:`RolloutRunner` that can also record the totals of every tick.

    ``max_ticks`` None records nothing (the viewer reads the state itself
    and may run past any length fixed up front).
    """

    def __init__(self, model, bank, candidates, max_ticks=None, **options):
        super().__init__(model, bank, candidates, 1, **options)
        self.biomass_history = None
        if max_ticks is not None:
            # [tick, world, group] totals, written by ``_tick``.
            self.biomass_history = torch.zeros(
                (max(1, int(max_ticks)), self.E, model.G),
                dtype=torch.float64, device=model.device)
            self.state_buffers.append(self.biomass_history)

    def _tick(self):
        super()._tick()
        if self.biomass_history is None:
            return
        # ``tick`` has advanced, so this tick's row is tick - 1. A tensor
        # index keeps the write inside a captured graph.
        row = (self.tick - 1).clamp(0, self.biomass_history.shape[0] - 1).reshape(1)
        self.biomass_history.index_copy_(0, row, self.biomass.sum(-1).double()[None])


def _architecture(policy):
    layers = [m for m in policy.net if isinstance(m, torch.nn.Linear)]
    return layers[0].out_features, len(layers) - 1, getattr(policy, "activation", "sig")


def build_runner(env, policies, obs_mean, obs_var, n_ticks=None, device="cuda",
                 execution=None, graph_ticks=32, random_dms=(), diagnostics=False):
    """Upload ``env``'s state and the frozen policies; returns ``(model, runner)``.

    ``random_dms`` has one entry per extra candidate: the decision makers
    that act at random in it (zero weights). Candidate 0 is the trained
    world. Every candidate starts from ``env``'s state.
    """
    model = TensorEcosystem(env, device)
    if execution is None:
        execution = "cuda-graph" if model.device.type == "cuda" else "eager"
    missing = [fid for fid in model.dm_ids if not isinstance(policies.get(fid), PolicyNetwork)]
    if missing:
        raise ValueError(f"the tensor engine needs a policy network for every decision "
                         f"maker; missing: {missing}")
    shapes = {_architecture(policies[fid]) for fid in model.dm_ids}
    if len(shapes) != 1:
        raise ValueError(f"the tensor engine packs one architecture for every decision "
                         f"maker; the checkpoints have {sorted(shapes)}")
    hidden_dim, hidden_layers, activation = shapes.pop()
    bank = PolicyBank(model, hidden_dim=hidden_dim, hidden_layers=hidden_layers,
                      activation=activation)
    trained = [torch.cat([p.detach().reshape(-1) for p in policies[fid].parameters()])
               .to(model.device) for fid in model.dm_ids]
    bank.install(trained)
    rows = [set()] + [set(r) for r in random_dms]
    unknown = set().union(*rows) - set(model.dm_ids)
    if unknown:
        raise ValueError(f"random_dms names unknown decision makers: {sorted(unknown)}")

    normalize = obs_mean is not None and obs_var is not None
    if normalize and np.shape(obs_mean) != (model.D, model.F):
        raise ValueError(f"obs stats shape {np.shape(obs_mean)} != {(model.D, model.F)}")
    runner = InferenceRunner(model, bank, len(rows), max_ticks=n_ticks,
                             obs_normalize=normalize, execution=execution,
                             graph_ticks=graph_ticks, diagnostics=diagnostics)
    if len(rows) > 1:
        runner.set_weights(bank.pack([
            torch.stack([torch.zeros_like(flat) if fid in row else flat for row in rows])
            for fid, flat in zip(model.dm_ids, trained)]))
    order = [list(env.dm_ids).index(fid) for fid in model.dm_ids]
    if normalize:
        mean = model.tensor(np.asarray(obs_mean, dtype=np.float32)[order])
        var = model.tensor(np.asarray(obs_var, dtype=np.float32)[order])
    else:
        mean = torch.zeros((model.D, model.F), device=model.device)
        var = torch.ones((model.D, model.F), device=model.device)

    # One world's state, broadcast over the candidates by ``reset``.
    biomass, reserve, hidden = model.import_state([env])
    extra = impact_state(model, [env])
    if model.stages_on:
        extra["juvenile"] = model.import_juvenile([env])
    keys = torch.full((1,), int(getattr(env, "current_world_seed", 0)),
                      dtype=torch.int64, device=model.device)
    temperature = float(getattr(env, "softmax_temperature", 1.0) or 1.0)
    runner.reset(biomass, reserve, keys, int(n_ticks or 1), mean, var,
                 torch.tensor(temperature, device=model.device),
                 light_start=model.light_starts([env]), hidden=hidden, **extra)
    return model, runner


def impact_state(model, envs):
    """The impact buffers of ``reset`` for ``envs``' current maps."""
    extra = {}
    if model.has_impacts:
        extra["impacts"] = model.import_impacts(envs)
    if model.n_obs_imp:
        extra["impact_obs"] = model.import_impact_observations(envs)
    return extra


def _dead_threshold(env):
    """``inference.run_inference``'s collapse cutoff for the same world."""
    units = [float(getattr(fg, "min_split_biomass", 0.0)) for fg in env.fgs.values()]
    units = [u for u in units if u > 0.0]
    return 0.5 * min(units) if units else 1e-9


def run_gpu_inference(env, policies, obs_mean, obs_var, n_ticks, verbose=True,
                      progress_ticks=(), device="cuda", execution=None):
    """Step ``env``'s world ``n_ticks`` on the tensor engine, headless.

    Returns ``{fid: [total biomass after tick 1, 2, ...]}`` like
    ``inference.run_inference``, stopping after the first tick at which
    every group has collapsed. ``progress_ticks`` are the 0-based ticks
    after which the ``tick x/y`` and B/B0 lines are printed (the CPU
    loop's schedule, ``inference.progress_due``). ``env`` itself is not
    stepped.
    """
    n_ticks = int(n_ticks)
    if n_ticks < 1:
        return {fid: [] for fid in env.fgs}
    model, runner = build_runner(env, policies, obs_mean, obs_var, n_ticks,
                                 device=device, execution=execution)
    start = {fid: float(fg.biomass.sum()) for fid, fg in env.fgs.items()}
    column = {fid: model.ids.index(fid) for fid in env.fgs}
    dead_eps = _dead_threshold(env)
    stops = sorted({int(t) + 1 for t in progress_ticks if 0 <= int(t) < n_ticks} | {n_ticks})
    done, end = 0, n_ticks
    try:
        for stop in stops:
            runner.run(stop - done)
            done = stop
            totals = runner.biomass_history[:done, 0].cpu().numpy()
            dead = np.flatnonzero(totals.sum(axis=1) <= dead_eps)
            if dead.size:
                end = int(dead[0]) + 1
            if verbose:
                shown = min(done, end)
                print(f"    tick {shown}/{n_ticks}")
                ratios = []
                for fid in env.fgs:
                    b0 = start[fid]
                    cur = float(totals[shown - 1, column[fid]])
                    ratios.append(f"{fid} {cur / b0:.3f}" if b0 > 0.0 else f"{fid} -")
                print(f"      B/B0: {'  '.join(ratios)}")
            if dead.size:
                if verbose:
                    print(f"    [inference] all biomass has collapsed to 0 "
                          f"at tick {end}/{n_ticks}; stopping early.")
                break
        totals = runner.biomass_history[:min(done, end), 0].cpu().numpy()
    except KeyboardInterrupt:
        # The tick counter, not ``done``: the interrupted block may have
        # run part of its ticks.
        done = min(int(runner.tick.item()), n_ticks, end)
        if verbose:
            print(f"\n    interrupted at tick {done}/{n_ticks}; "
                  f"returning partial history.")
        totals = runner.biomass_history[:done, 0].cpu().numpy()
    finally:
        runner.close()
    return {fid: [float(v) for v in totals[:, column[fid]]] for fid in env.fgs}


class GpuStepper:
    """Step ``envs`` on the tensor engine and mirror them back each tick.

    ``envs[0]`` is the trained world, the others the random baselines;
    ``random_dms[k]`` names the decision makers acting at random in
    ``envs[k + 1]``. All envs must hold the same world (``build_env`` with
    the same seed and overrides, as ``inference.py`` builds them); only
    ``envs[0]``'s state is uploaded. After :meth:`step` each env carries
    what the CPU engine would have left in it for the viewer: biomass,
    reserve, juveniles and hide fractions per group, and the cumulative
    ``loss_*``, ``intake_by_pred_prey`` and ``_action_*`` books.
    """

    def __init__(self, envs, policies, obs_mean, obs_var, random_dms=(),
                 device="cuda", execution=None, graph_ticks=1):
        self.envs = list(envs)
        if len(random_dms) != len(self.envs) - 1:
            raise ValueError("one random_dms entry per baseline env")
        self.model, self.runner = build_runner(
            self.envs[0], policies, obs_mean, obs_var, None, device=device,
            execution=execution, graph_ticks=graph_ticks, random_dms=random_dms,
            diagnostics=True)
        m = self.model
        self._staged = [m.ids[g] for g in range(m.G)
                        if getattr(self.envs[0].fgs[m.ids[g]], "juvenile", None) is not None]

    def step(self):
        self.runner.run(1)
        self.mirror()

    def refresh_impacts(self):
        """Upload the envs' impact maps again (a live intensity change)."""
        extra = impact_state(self.model, self.envs)
        if "impacts" in extra:
            self.runner.impact_loss.copy_(extra["impacts"][0])
            self.runner.impact_cost.copy_(extra["impacts"][1])
        if "impact_obs" in extra:
            self.runner.impact_obs.copy_(extra["impact_obs"])

    def mirror(self):
        m, r = self.model, self.runner
        shape = (len(self.envs), m.G, m.H, m.W)
        biomass = r.biomass.reshape(shape).cpu().numpy()
        reserve = r.reserve.reshape(shape).cpu().numpy()
        hidden = r.hidden.reshape(shape).cpu().numpy()
        juvenile = (r.juvenile.reshape(shape).cpu().numpy()
                    if m.stages_on else None)
        losses = r.loss_book.cpu().numpy()
        diet = r.diet_book.cpu().numpy()
        actions = r.action_sum.cpu().numpy()
        active = r.active_ticks.cpu().numpy()
        tick = int(r.tick.item())
        ids = m.ids
        for k, env in enumerate(self.envs):
            for g, fid in enumerate(ids):
                fg = env.fgs[fid]
                fg.biomass = biomass[k, g].copy()
                fg.energy_reserve = reserve[k, g].copy()
                if juvenile is not None and fid in self._staged:
                    fg.juvenile = juvenile[k, g].copy()
            env.prev_hidden_frac = hidden[k].copy()
            for row, name in enumerate(r.LOSS_ORDER):
                setattr(env, "loss_" + name,
                        {fid: float(losses[k, row, g]) for g, fid in enumerate(ids)})
            env.intake_by_pred_prey = {
                ids[p]: {ids[q]: float(diet[k, p, q]) for q in range(m.G) if diet[k, p, q] != 0.0}
                for p in range(m.G) if diet[k, p].any()}
            # ``decisions.update_action_diagnostics``' books; the engine
            # keeps [entropy, move, rest, eat] in DM order (env.dm_ids).
            env._action_entropy_sum = actions[k, :, 0].copy()
            env._action_move_frac = actions[k, :, 1].copy()
            env._action_rest_frac = actions[k, :, 2].copy()
            env._action_eat_frac = actions[k, :, 3].copy()
            env._action_active_ticks = active[k].astype(np.int64)
            env._action_entropy_count = tick
            env.tick_count = tick

    def close(self):
        self.runner.close()
