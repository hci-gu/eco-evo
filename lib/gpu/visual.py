"""Bridge the existing CPU probe viewer to completed tensor training updates."""

import copy
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from lib.runners.training_progress import probe_randomness, make_visual_progress


@torch.no_grad()
def policy_snapshot(trainer):
    """Copy authoritative weights/statistics without altering the training bank."""
    policies = {fid: copy.deepcopy(p).cpu().eval()
                for fid, p in trainer.bank.policies.items()}
    for fid, flat in zip(trainer.model.dm_ids, trainer.theta):
        torch.nn.utils.vector_to_parameters(flat.detach().cpu().clone(), policies[fid].parameters())
    stats = {}
    if trainer.obs_normalize:
        mean = trainer.obs_mean.float().cpu().numpy().copy()
        var = trainer.obs_var.float().cpu().numpy().copy()
        count = trainer.obs_count.cpu().tolist()
        stats = {fid: dict(mean=mean[d], var=var[d], count=count[d])
                 for d, fid in enumerate(trainer.model.dm_ids)}
    return SimpleNamespace(policies=policies, obs_stats=stats,
                           survival_reward=trainer.runner_options.get("survival_reward"),
                           obs_normalize=trainer.obs_normalize,
                           softmax_temperature=float(trainer.temperature.cpu()))


class GPUTrainingVisualizer:
    """SDL stays on the main thread; one worker executes each complete update.

    The main thread only reads trainer tensors after the worker finishes. The
    worker uses the caller's CUDA stream and synchronizes at that boundary, so
    event pumping continues during compilation, capture, and GPU execution.
    """

    def __init__(self, trainer, args, directory):
        self.viz = None
        self.progress = None
        self.executor = None
        self.directory = directory
        try:
            # Reuse the CPU CLI's complete probe (maps, controls, playback and
            # plots). Guard its legacy module-level no-grad setting on import.
            with torch.no_grad():
                from train import _ProbeEnvBuilder, _probe_biomass
            from lib.viz import LiveVisualizer
            from inference import _load_spawn_defaults, _load_spawn_templates
            from tools.biomass_html import _reward_title_from_meta

            self.probe = _probe_biomass
            builder = trainer.spec.builder
            self.builder = _ProbeEnvBuilder(
                builder.project_path, builder.grid, builder.mortality,
                builder.migration, currents=builder.currents, library_path=builder.library_path,
                food_blobs=builder.food_blobs,
                boundary=builder.boundary,
                # Without this the probe falls back to train.py's module
                # default (1.0) and runs with the unscaled library rates,
                # so --mortality_multiplier would not reach the viewer.
                mortality_multiplier=builder.mortality_multiplier,
            )
            with probe_randomness() as seed:
                env = self.builder(seed=seed)
            dm_ids = list(env.dm_ids)
            ndm_ids = [fid for fid in env.fgs if fid not in dm_ids]
            self.viz = LiveVisualizer(
                fg_ids=list(env.fgs), grid_shape=builder.grid, mode="train",
                plot_fg_ids=dm_ids + ndm_ids, ndm_ids=ndm_ids,
                title="Mareld GPU training" + (" [TORUS]" if builder.boundary == "torus" else "")
                      + (" [DEBUG FOOD BLOBS]" if builder.food_blobs else ""),
            )
            if not self.viz.enabled:
                self._disable()
                return
            self.viz.set_reward_label(_reward_title_from_meta(vars(args)))
            self.viz.set_b0_defaults({fid: float(fg.biomass.sum()) for fid, fg in env.fgs.items()})
            self.viz.set_ticks_default(args.ticks)
            self.viz.set_neval_ticks_default(args.ticks)
            self.viz.set_spawn_templates(_load_spawn_templates(builder.project_path))
            self.viz.set_spawn_defaults(_load_spawn_defaults(builder.project_path))
            self.viz.set_save_dir(str(directory))
            self.viz.update_biomass(env.fgs, tick=0)
            self.pump()
            if self.viz is None:
                return
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-training")
            self.progress = make_visual_progress("gpu", args, self.viz)
            print("[viz] GPU training viewer enabled; CPU inference probes run between updates.", flush=True)
        except Exception as error:
            print(f"[viz] Could not start viewer: {error}; continuing training.", flush=True)
            self.close()

    def _disable(self):
        viz, self.viz = self.viz, None
        if viz is not None:
            try:
                viz.close()
            except Exception:
                pass

    def update_progress(self, trainer, args):
        if self.viz is not None and self.progress is not None:
            self.progress(trainer, trainer.iterations_completed, self.directory, args)
            self.pump()

    def pump(self):
        if self.viz is None:
            return
        try:
            if not self.viz.enabled or not self.viz.pump_events():
                self._disable()
        except Exception as error:
            print(f"[viz] Viewer disabled: {error}", flush=True)
            self._disable()

    def training_ticks(self, default):
        self.pump()
        if self.viz is None:
            return default
        try:
            override = self.viz.get_neval_ticks_override()
            return default if override is None else max(1, int(override))
        except Exception as error:
            print(f"[viz] Viewer disabled: {error}", flush=True)
            self._disable()
            return default

    def train_step(self, trainer, targets, ticks, epoch):
        if self.executor is None:
            return trainer.train_step(targets, ticks, epoch)
        device = trainer.model.device
        stream = torch.cuda.current_stream(device) if device.type == "cuda" else None

        def update():
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                result = trainer.train_step(targets, ticks, epoch)
                if stream is not None:
                    stream.synchronize()
                return result

        future = self.executor.submit(update)
        while not future.done():
            self.pump()
            wait([future], timeout=0.05)
        return future.result()  # Training failures must propagate to the caller.

    def update(self, trainer, targets, generation, iteration, ticks):
        self.pump()
        if self.viz is None:
            return
        try:
            snapshot = policy_snapshot(trainer)
            metrics = trainer.metrics()
            rewards = {fid: value for fid, value in zip(trainer.model.dm_ids, metrics["reward_mean"])
                       if fid in targets}
            step = trainer.iterations_completed
            for fid, reward in rewards.items():
                self.viz.update_reward(fid, reward, step=step)
            self.probe(
                snapshot, self.builder, n_ticks=ticks, gen=generation,
                it=iteration, jsonl_path=self.directory / "biomass.jsonl",
                compact=False, viz=self.viz, viz_step=step,
                viz_extra={"gen": generation + 1, "iter": iteration + 1,
                           "T": snapshot.softmax_temperature},
                reward_by_fid=rewards,
            )
            self.pump()
        except Exception as error:
            print(f"[viz] Probe failed: {error}; viewer disabled, training continues.", flush=True)
            self._disable()

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        self._disable()
