"""Opt-in biomass-flow diagnostics for inference runs.

The ledger is deliberately external to the ecosystem model. The environment
only calls methods on ``env.biomass_ledger`` when one has been attached, so the
normal training/inference path stays unchanged and this module can be removed
without changing the ecological equations.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np


SPECIAL_INITIAL = "initial_reserve"
SPECIAL_UNATTRIBUTED = "unattributed"
SPECIAL_NDM_GROWTH = "intrinsic_ndm_growth"
SPECIAL_SEED = "seed_recolonisation"
SPECIAL_IMPACT = "impact"
SPECIAL_NATURAL = "natural_mortality"
SPECIAL_STARVATION = "energy_deficit"


@dataclass
class BiomassLedger:
    """Collect aggregate biomass and energy flow diagnostics.

    ``source_energy`` tracks the provenance of decision-maker energy reserves.
    Positive DM biomass growth is then allocated across these reserve sources.
    This is an attribution layer only; it does not feed back into the model.
    """

    fgs: dict
    dm_ids: list[str]
    global_fg_order: list[str]
    grid_shape: tuple[int, int]
    dtype: object = np.float32

    source_labels: list[str] = field(init=False)
    dm_index: dict[str, int] = field(init=False)
    fg_index: dict[str, int] = field(init=False)
    source_index: dict[str, int] = field(init=False)
    source_energy: np.ndarray = field(init=False)
    pending_source_gains: np.ndarray = field(init=False)

    predation_tonnes: list[np.ndarray] = field(default_factory=list)
    assimilation_energy: list[np.ndarray] = field(default_factory=list)
    biomass_gain_by_source: list[np.ndarray] = field(default_factory=list)
    biomass_loss_by_sink: list[np.ndarray] = field(default_factory=list)
    biomass_totals: list[np.ndarray] = field(default_factory=list)
    action_percentages: list[np.ndarray] = field(default_factory=list)
    action_category_percentages: list[np.ndarray] = field(default_factory=list)
    initial_biomass: np.ndarray | None = None
    events: list[dict] = field(default_factory=list)
    tick: int = 0

    def __post_init__(self):
        H, W = self.grid_shape
        self.dm_ids = list(self.dm_ids)
        self.global_fg_order = list(self.global_fg_order)
        self.source_labels = (
            list(self.global_fg_order)
            + [
                SPECIAL_INITIAL,
                SPECIAL_UNATTRIBUTED,
                SPECIAL_NDM_GROWTH,
                SPECIAL_SEED,
                SPECIAL_IMPACT,
                SPECIAL_NATURAL,
                SPECIAL_STARVATION,
            ]
        )
        self.action_labels = (
            ["move_north", "move_east", "move_south", "move_west", "rest"]
            + [f"eat_{fid}" for fid in self.global_fg_order]
        )
        self.action_category_labels = ["move", "rest", "eat"]
        self.dm_index = {fid: i for i, fid in enumerate(self.dm_ids)}
        self.fg_index = {fid: i for i, fid in enumerate(self.global_fg_order)}
        self.source_index = {label: i for i, label in enumerate(self.source_labels)}
        self.source_energy = np.zeros(
            (len(self.dm_ids), len(self.source_labels), H, W),
            dtype=self.dtype,
        )
        self.pending_source_gains = np.zeros(
            (len(self.dm_ids), len(self.global_fg_order), H, W),
            dtype=self.dtype,
        )
        initial_idx = self.source_index[SPECIAL_INITIAL]
        for fid, i in self.dm_index.items():
            reserve = getattr(self.fgs[fid], "energy_reserve", None)
            if reserve is not None:
                self.source_energy[i, initial_idx] = reserve.astype(self.dtype, copy=False)

    @classmethod
    def from_env(cls, env):
        if not getattr(env, "_static_built", False):
            env._build_static_caches()
        return cls(
            fgs=env.fgs,
            dm_ids=list(env.dm_ids),
            global_fg_order=list(env.global_fg_order),
            grid_shape=(env.H, env.W),
            dtype=env.dtype,
        )

    def begin_tick(self):
        n_all = len(self.global_fg_order)
        n_dm = len(self.dm_ids)
        n_src = len(self.source_labels)
        if self.initial_biomass is None:
            self.initial_biomass = self._current_biomass_totals()
        self._tick_predation = np.zeros((n_dm, n_all), dtype=np.float64)
        self._tick_assimilation = np.zeros((n_dm, n_all), dtype=np.float64)
        self._tick_gain = np.zeros((n_all, n_src), dtype=np.float64)
        self._tick_loss = np.zeros((n_all, n_src), dtype=np.float64)
        self._tick_action_percentages = np.zeros(
            (n_dm, len(self.action_labels)), dtype=np.float64
        )
        self._tick_action_category_percentages = np.zeros((n_dm, 3), dtype=np.float64)
        self.pending_source_gains.fill(0.0)

    def end_tick(self):
        self.predation_tonnes.append(self._tick_predation.copy())
        self.assimilation_energy.append(self._tick_assimilation.copy())
        self.biomass_gain_by_source.append(self._tick_gain.copy())
        self.biomass_loss_by_sink.append(self._tick_loss.copy())
        self.biomass_totals.append(self._current_biomass_totals())
        self.action_percentages.append(self._tick_action_percentages.copy())
        self.action_category_percentages.append(
            self._tick_action_category_percentages.copy()
        )
        self.tick += 1

    def record_actions(self, B_dm, pi_move, pi_rest, pi_eat):
        """Record biomass-weighted action percentages for each DM this tick."""
        if B_dm.size == 0:
            return
        weights = B_dm.astype(np.float64, copy=False)
        denom = weights.sum(axis=(1, 2))

        move_vals = (weights[:, None, :, :] * pi_move).sum(axis=(2, 3))
        rest_vals = (weights * pi_rest).sum(axis=(1, 2))[:, None]
        eat_vals = (weights[:, None, :, :] * pi_eat).sum(axis=(2, 3))
        detailed = np.concatenate([move_vals, rest_vals, eat_vals], axis=1)
        category = np.stack(
            [move_vals.sum(axis=1), rest_vals[:, 0], eat_vals.sum(axis=1)],
            axis=1,
        )

        self._tick_action_percentages = np.divide(
            detailed,
            denom[:, None],
            out=np.zeros_like(detailed, dtype=np.float64),
            where=denom[:, None] > 0.0,
        ) * 100.0
        self._tick_action_category_percentages = np.divide(
            category,
            denom[:, None],
            out=np.zeros_like(category, dtype=np.float64),
            where=denom[:, None] > 0.0,
        ) * 100.0

    def record_predation(self, actual, energy_gain_mat):
        """Record predator-prey intake and pending energy provenance."""
        if actual.size == 0:
            return
        pred_by_prey = actual.sum(axis=(2, 3), dtype=np.float64)
        energy_by_prey_cell = actual * energy_gain_mat[:, :, None, None]
        energy_by_prey = energy_by_prey_cell.sum(axis=(2, 3), dtype=np.float64)

        self._tick_predation += pred_by_prey
        self._tick_assimilation += energy_by_prey
        self.pending_source_gains += energy_by_prey_cell.astype(self.dtype, copy=False)

        for i, predator in enumerate(self.dm_ids):
            for j, prey in enumerate(self.global_fg_order):
                tonnes = float(pred_by_prey[i, j])
                energy = float(energy_by_prey[i, j])
                if tonnes <= 0.0 and energy <= 0.0:
                    continue
                self._add_loss(prey, predator, tonnes)
                self._add_event(prey, "loss", "predation", predator, tonnes, 0.0)
                self._add_event(predator, "gain", "assimilation", prey, 0.0, energy)

    def reduce_dm_energy_sources(self, fg_id, reduction):
        """Apply the same proportional reserve reduction used by the model."""
        i = self.dm_index.get(fg_id)
        if i is None:
            return
        self.source_energy[i] = (
            self.source_energy[i] * reduction[None, :, :]
        ).astype(self.dtype, copy=False)

    def apply_energy_movement(self, R, pi_rest, pi_eat_sum, pi_move,
                              r_rest, r_eat_base, r_after_move_meta,
                              dm_v, new_R):
        """Move source-reserve layers through the same action flow as energy."""
        if self.source_energy.size == 0:
            return
        eps = np.float32(1e-9)
        src = self.source_energy

        rest_before = src * pi_rest[:, None, :, :]
        rest_total_before = R * pi_rest
        rest_factor = np.divide(
            r_rest,
            rest_total_before,
            out=np.zeros_like(R, dtype=self.dtype),
            where=rest_total_before > eps,
        )
        rest_after = rest_before * rest_factor[:, None, :, :]

        eat_before = src * pi_eat_sum[:, None, :, :]
        eat_total_before = R * pi_eat_sum
        eat_factor = np.divide(
            r_eat_base,
            eat_total_before,
            out=np.zeros_like(R, dtype=self.dtype),
            where=eat_total_before > eps,
        )
        eat_after = eat_before * eat_factor[:, None, :, :]
        eat_after[:, :len(self.global_fg_order)] += self.pending_source_gains

        move_before = src[:, :, None, :, :] * pi_move[:, None, :, :, :]
        move_factor = np.divide(
            r_after_move_meta,
            R,
            out=np.zeros_like(R, dtype=self.dtype),
            where=R > eps,
        )
        move_after = move_before * move_factor[:, None, None, :, :]
        flux = dm_v[:, None, None, None, None]
        src_out = move_after * flux
        src_keep = move_after - src_out

        src_total = rest_after + eat_after + src_keep.sum(axis=2)
        # Direction order: 0=N, 1=E, 2=S, 3=W.
        src_total[:, :, :-1, :] += src_out[:, :, 0, 1:, :]
        src_total[:, :, :, 1:] += src_out[:, :, 1, :, :-1]
        src_total[:, :, 1:, :] += src_out[:, :, 2, :-1, :]
        src_total[:, :, :, :-1] += src_out[:, :, 3, :, 1:]

        src_sum = src_total.sum(axis=1)
        scale = np.divide(
            new_R,
            src_sum,
            out=np.zeros_like(new_R, dtype=self.dtype),
            where=src_sum > eps,
        )
        src_total = src_total * scale[:, None, :, :]

        missing = (new_R > eps) & (src_sum <= eps)
        if np.any(missing):
            unattributed = self.source_index[SPECIAL_UNATTRIBUTED]
            src_total[:, unattributed] += np.where(missing, new_R, 0.0)

        self.source_energy = src_total.astype(self.dtype, copy=False)
        self.pending_source_gains.fill(0.0)

    def record_impact_loss(self, fg_id, impact_id, loss):
        tonnes = float(np.maximum(loss, 0.0).sum())
        if tonnes <= 0.0:
            return
        self._add_loss(fg_id, SPECIAL_IMPACT, tonnes)
        self._add_event(fg_id, "loss", "impact", impact_id, tonnes, 0.0)

    def record_natural_mortality(self, fg_id, loss):
        tonnes = float(np.maximum(loss, 0.0).sum())
        if tonnes <= 0.0:
            return
        self._add_loss(fg_id, SPECIAL_NATURAL, tonnes)
        self._add_event(fg_id, "loss", "natural_mortality", SPECIAL_NATURAL, tonnes, 0.0)

    def record_dm_growth(self, fg_id, growth):
        i_dm = self.dm_index.get(fg_id)
        if i_dm is None:
            return
        positive = np.maximum(growth, 0.0)
        negative = -np.minimum(growth, 0.0)
        if float(positive.sum()) > 0.0:
            reserve = self.source_energy[i_dm]
            reserve_sum = reserve.sum(axis=0)
            share = np.divide(
                reserve,
                reserve_sum[None, :, :],
                out=np.zeros_like(reserve, dtype=self.dtype),
                where=reserve_sum[None, :, :] > np.float32(1e-9),
            )
            attributed = share * positive[None, :, :]
            missing = positive - attributed.sum(axis=0)
            unattributed = self.source_index[SPECIAL_UNATTRIBUTED]
            attributed[unattributed] += np.maximum(missing, 0.0)
            totals = attributed.sum(axis=(1, 2), dtype=np.float64)
            for src_idx, tonnes in enumerate(totals):
                if tonnes <= 0.0:
                    continue
                src = self.source_labels[src_idx]
                self._add_gain(fg_id, src, float(tonnes))
                self._add_event(fg_id, "gain", "growth_allocated", src, float(tonnes), 0.0)
        loss = float(negative.sum())
        if loss > 0.0:
            self._add_loss(fg_id, SPECIAL_STARVATION, loss)
            self._add_event(fg_id, "loss", "energy_deficit", SPECIAL_STARVATION, loss, 0.0)

    def record_ndm_growth(self, fg_id, intrinsic_growth, seed_growth=None,
                          effective_delta=None):
        pos_intrinsic = np.maximum(intrinsic_growth, 0.0)
        neg_intrinsic = -np.minimum(intrinsic_growth, 0.0)
        pos_seed = np.maximum(seed_growth, 0.0) if seed_growth is not None else 0.0

        if effective_delta is not None:
            effective_gain = np.maximum(effective_delta, 0.0)
            positive_raw = pos_intrinsic + pos_seed
            scale = np.divide(
                effective_gain,
                positive_raw,
                out=np.zeros_like(effective_gain, dtype=self.dtype),
                where=positive_raw > np.float32(1e-9),
            )
            pos_intrinsic = pos_intrinsic * scale
            pos_seed = pos_seed * scale
            effective_loss = -np.minimum(effective_delta, 0.0)
            raw_loss = neg_intrinsic
            loss_scale = np.divide(
                effective_loss,
                raw_loss,
                out=np.zeros_like(effective_loss, dtype=self.dtype),
                where=raw_loss > np.float32(1e-9),
            )
            neg_intrinsic = raw_loss * loss_scale

        gain_intrinsic = float(np.asarray(pos_intrinsic).sum())
        loss_intrinsic = float(np.asarray(neg_intrinsic).sum())
        gain_seed = float(np.asarray(pos_seed).sum())
        if gain_intrinsic > 0.0:
            self._add_gain(fg_id, SPECIAL_NDM_GROWTH, gain_intrinsic)
            self._add_event(fg_id, "gain", "ndm_growth", SPECIAL_NDM_GROWTH, gain_intrinsic, 0.0)
        if gain_seed > 0.0:
            self._add_gain(fg_id, SPECIAL_SEED, gain_seed)
            self._add_event(fg_id, "gain", "seed_recolonisation", SPECIAL_SEED, gain_seed, 0.0)
        if loss_intrinsic > 0.0:
            self._add_loss(fg_id, SPECIAL_NDM_GROWTH, loss_intrinsic)
            self._add_event(fg_id, "loss", "logistic_decline", SPECIAL_NDM_GROWTH, loss_intrinsic, 0.0)

    def save(self, path):
        """Save aggregate arrays to ``path`` and event rows beside it as CSV."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        predation = np.asarray(self.predation_tonnes, dtype=np.float64)
        assimilation = np.asarray(self.assimilation_energy, dtype=np.float64)
        gains = np.asarray(self.biomass_gain_by_source, dtype=np.float64)
        losses = np.asarray(self.biomass_loss_by_sink, dtype=np.float64)
        np.savez(
            path,
            predation_tonnes=predation,
            assimilation_energy=assimilation,
            biomass_gain_by_source=gains,
            biomass_loss_by_sink=losses,
            initial_biomass=np.asarray(
                self.initial_biomass if self.initial_biomass is not None else [],
                dtype=np.float64,
            ),
            biomass_totals=np.asarray(self.biomass_totals, dtype=np.float64),
            action_percentages=np.asarray(self.action_percentages, dtype=np.float64),
            action_category_percentages=np.asarray(
                self.action_category_percentages, dtype=np.float64,
            ),
            dm_ids=np.asarray(self.dm_ids),
            fg_ids=np.asarray(self.global_fg_order),
            source_labels=np.asarray(self.source_labels),
            action_labels=np.asarray(self.action_labels),
            action_category_labels=np.asarray(self.action_category_labels),
            columns=np.asarray([
                "predation_tonnes[tick, predator_dm, prey]",
                "assimilation_energy[tick, predator_dm, prey]",
                "biomass_gain_by_source[tick, species, source]",
                "biomass_loss_by_sink[tick, species, sink]",
                "initial_biomass[species]",
                "biomass_totals[tick, species]",
                "action_percentages[tick, predator_dm, action]",
                "action_category_percentages[tick, predator_dm, category]",
            ]),
        )
        csv_path = os.path.splitext(path)[0] + ".csv"
        self.save_events_csv(csv_path)
        return csv_path

    def save_events_csv(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = ["tick", "species", "direction", "mechanism", "counterparty", "tonnes", "energy"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)

    def _add_gain(self, species, source, tonnes):
        i = self.fg_index.get(species)
        j = self.source_index.get(source, self.source_index[SPECIAL_UNATTRIBUTED])
        if i is not None:
            self._tick_gain[i, j] += tonnes

    def _add_loss(self, species, sink, tonnes):
        i = self.fg_index.get(species)
        j = self.source_index.get(sink, self.source_index[SPECIAL_UNATTRIBUTED])
        if i is not None:
            self._tick_loss[i, j] += tonnes

    def _add_event(self, species, direction, mechanism, counterparty, tonnes, energy):
        if tonnes <= 0.0 and energy <= 0.0:
            return
        self.events.append({
            "tick": self.tick,
            "species": species,
            "direction": direction,
            "mechanism": mechanism,
            "counterparty": counterparty,
            "tonnes": f"{float(tonnes):.10g}",
            "energy": f"{float(energy):.10g}",
        })

    def _current_biomass_totals(self):
        return np.asarray(
            [
                float(np.asarray(self.fgs[fid].biomass, dtype=np.float64).sum())
                for fid in self.global_fg_order
            ],
            dtype=np.float64,
        )
