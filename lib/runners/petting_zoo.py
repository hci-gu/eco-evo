# import lib.constants as const
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from lib.config import const
from lib.config.settings import Settings
from lib.config.species import build_species_map
from lib.model import Model
import lib.model as model
import lib.evolution as evolution  # Your NumPy-based evolution functions
from lib.visualize import plot_generations, plot_biomass
from lib.data_manager import update_generations_data, process_data
from lib.behaviour import run_all_scenarios
from lib.evaluate import predict_years
import numpy as np
import random
import copy
import time
from lib.environments.petting_zoo import env
import optuna

# ---- Parallel worker plumbing ----
_WORKER_RUNNER = None  # one runner per proces
_WORKER_POPULATION_STATE = None
_WORKER_MODEL_CACHE = None

def noop(a, b, c):
    pass

class RandomBaselinePolicy:
    """Simple random policy adapter with a Model-like forward interface."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(int(seed))

    def forward(self, x):
        n = x.shape[0]
        logits = self.rng.random((n, model.OUTPUT_SIZE), dtype=np.float32) + 1e-6
        probs = logits / logits.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

class PettingZooRunner():
    def __init__(
        self,
        settings: Settings,
        render_mode: str = "none",
        map_folder: str = "maps/baltic",
        build_population: bool = True,
    ):
        # Create the environment (render_mode can be "none" if visualization is not needed)
        self.species_map = build_species_map(settings)
        self.env = env(
            settings=settings,
            species_map=self.species_map,
            render_mode=render_mode,
            map_folder=map_folder,
        )
        self.settings = settings
        self.empty_action = self.env.action_space("plankton").sample()
        self.env.reset()
        self.current_generation = 0
        self.rng = random.Random(None if self.settings.seed < 0 else self.settings.seed)

        # Use all species in the simulation (including plankton, if desired).
        self.species_list = [species for species in const.ACTING_SPECIES]
        # Create a population for each species (training only).
        self.population = {}
        if build_population:
            self.population = {
                species: [Model() for _ in range(self.settings.num_agents)]
                for species in self.species_list
            }
        # Track best fitness and best model for each species.
        self.best_fitness = {species: -float('inf') for species in self.species_list}
        self.best_agent = {species: None for species in self.species_list}
        self.agent_index = 0
        self.eval_index = 0
        self.champion_progress_history = []
        self.fixed_validation_history = []
        self.fixed_validation_metric = str(
            getattr(self.settings, "fixed_validation_metric", "fitness")
        ).strip().lower()
        if self.fixed_validation_metric not in {"fitness", "survival"}:
            raise ValueError(
                f"Unsupported fixed_validation_metric '{self.settings.fixed_validation_metric}'. "
                "Use 'fitness' or 'survival'."
            )
        if self.settings.seed >= 0:
            benchmark_rng = random.Random(self.settings.seed + 9_999_937)
            self.champion_progress_seed = benchmark_rng.randrange(100000)
        else:
            self.champion_progress_seed = 137
            benchmark_rng = random.Random(self.champion_progress_seed)

        configured_fixed_seed = int(getattr(self.settings, "fixed_validation_seed", 4242))
        if configured_fixed_seed >= 0:
            self.fixed_validation_seed = configured_fixed_seed
        elif self.settings.seed >= 0:
            self.fixed_validation_seed = benchmark_rng.randrange(2**31)
        else:
            self.fixed_validation_seed = 4242

    def _get_alive_base_species(self):
        threshold = float(self.settings.alive_species_biomass_threshold)
        totals = {base_species: 0.0 for base_species in const.ACTING_BASE_SPECIES}
        for species_name, props in self.species_map.items():
            base_species = props.base_species
            if base_species not in totals:
                continue
            biomass_offset = model.MODEL_OFFSETS[species_name]["biomass"]
            totals[base_species] += float(self.env.world[..., biomass_offset].sum())
        return [base_species for base_species, total in totals.items() if total > threshold]

    def _energy_channel_indices(self):
        indices = []
        for species_name in self.species_map.keys():
            offsets = model.MODEL_OFFSETS.get(species_name)
            if offsets is None:
                continue
            indices.append(offsets["energy"])
        return indices

    def _scale_all_species_energy(self, scale: float):
        scale = float(scale)
        if abs(scale - 1.0) < 1e-9:
            return
        energy_channels = self._energy_channel_indices()
        if not energy_channels:
            return
        world = self.env.world
        world[..., energy_channels] *= scale
        np.clip(world[..., energy_channels], 0.0, self.settings.max_energy, out=world[..., energy_channels])

    def _apply_energy_decay(self, decay_per_cycle: float):
        decay = float(decay_per_cycle)
        if decay <= 0.0:
            return
        keep = max(0.0, 1.0 - decay)
        energy_channels = self._energy_channel_indices()
        if not energy_channels:
            return
        world = self.env.world
        world[..., energy_channels] *= keep
        np.clip(world[..., energy_channels], 0.0, self.settings.max_energy, out=world[..., energy_channels])

    def run(
        self,
        candidates,
        species_being_evaluated = "cod",
        seed = None,
        is_evaluation = False,
        callback = noop,
        collect_plot_data = True,
        python_random_seed = None,
        simple_win_threshold = None,
        max_steps_override = None,
        initial_energy_scale = 1.0,
        energy_decay_per_cycle = 0.0,
    ):
        if python_random_seed is not None:
            random.seed(python_random_seed)
        self.env.reset(seed)
        fitness_method = str(self.settings.fitness_method).strip().lower()
        if fitness_method not in {"simple", "biomass_pct"}:
            raise ValueError(
                f"Unsupported fitness_method '{self.settings.fitness_method}'. "
                "Use 'simple' or 'biomass_pct'."
            )
        biomass_scope = str(getattr(self.settings, "biomass_fitness_scope", "agent")).strip().lower()
        if biomass_scope not in {"agent", "base_species"}:
            raise ValueError(
                f"Unsupported biomass_fitness_scope '{self.settings.biomass_fitness_scope}'. "
                "Use 'agent' or 'base_species'."
            )

        if max_steps_override is not None:
            eval_horizon = max(1, int(max_steps_override))
        else:
            eval_horizon = self.settings.max_steps
        if fitness_method == "biomass_pct" and max_steps_override is None:
            eval_horizon = max(1, min(self.settings.max_steps, int(self.settings.fitness_eval_steps)))

        self._scale_all_species_energy(initial_energy_scale)

        fitness_target_species = species_being_evaluated
        if fitness_method == "biomass_pct" and biomass_scope == "base_species":
            if species_being_evaluated in self.species_map:
                fitness_target_species = self.species_map[species_being_evaluated].base_species
            else:
                fitness_target_species = const.base_species_name(species_being_evaluated)

        initial_biomass = self.env.get_total_biomass(fitness_target_species)
        current_biomass = initial_biomass
        prev_cycle_count = self.env.cycle_count
        episode_length = 0
        fitness = 0.0
        callback(self.env.world, fitness, False)

        while not all(self.env.terminations.values()) and not all(self.env.truncations.values()) and episode_length < eval_horizon:
            agent = self.env.agent_selection
            if agent == "plankton":
                self.env.step(self.empty_action)
            else:
                obs, reward, termination, truncation, info = self.env.last()
                candidate = candidates[agent]
                action_values = candidate.forward(obs.reshape(-1, model.INPUT_SIZE))
                action_values = action_values.reshape(self.settings.world_size, self.settings.world_size, model.OUTPUT_SIZE)
                # mean_action_values = action_values.mean(axis=(0, 1))

                self.env.step(action_values)

            current_cycle_count = self.env.cycle_count
            if current_cycle_count > prev_cycle_count:
                episode_length = current_cycle_count
                prev_cycle_count = current_cycle_count
                self._apply_energy_decay(energy_decay_per_cycle)
                current_biomass = self.env.get_total_biomass(fitness_target_species)

                if fitness_method == "simple":
                    fitness = float(episode_length)
                else:
                    fitness = (
                        100.0
                        * (current_biomass - initial_biomass)
                        / max(initial_biomass, 1e-8)
                    )

                if collect_plot_data and self.settings.enable_plotting and (is_evaluation == False or self.env.render_mode != "none"):
                    process_data({
                        'species': species_being_evaluated if not is_evaluation else None,
                        'agent_index': self.agent_index,
                        'eval_index': self.eval_index,
                        'step': episode_length,
                        'fitness': fitness,
                        'world': self.env.world
                    }, self.env.plot_data)
                callback(self.env.world, fitness, False)

                if self.settings.stop_on_single_species_left:
                    alive_base_species = self._get_alive_base_species()
                    if len(alive_base_species) <= 1:
                        winner = alive_base_species[0] if alive_base_species else "none"
                        if not self.env.reason:
                            self.env.reason = f"single_species_left:{winner}"
                        break

                if fitness_method == "simple" and simple_win_threshold is not None and fitness > float(simple_win_threshold):
                    if not self.env.reason:
                        self.env.reason = "simple_win_threshold_reached"
                    break
        
        if fitness_method == "biomass_pct":
            fitness = (
                100.0
                * (current_biomass - initial_biomass)
                / max(initial_biomass, 1e-8)
            )

        if self.settings.enable_logging:
            print("end of simulation", fitness, episode_length)
        callback(self.env.world, fitness, True)
        return fitness, episode_length, self.env.reason

    def evaluate_population(self):
        """
        Evaluate each model in the population for every species.
        When evaluating a candidate for a species S, for every other species (not S),
        we use the best model from previous generations (if available) to decide their actions.
        Returns a dictionary mapping species to a list of (chromosome, fitness) tuples.
        """
        if self.settings.seed >= 0:
            # Deterministic per-generation RNG for eval setup.
            eval_rng = random.Random(self.settings.seed + self.current_generation * 1_000_003)
        else:
            eval_rng = self.rng

        # Same eval seeds for every agent in the generation.
        eval_seeds = [eval_rng.randrange(100000) for _ in range(self.settings.agent_evaluations)]
        fitnesses = {species: [] for species in self.species_list}
        fitness_method = str(self.settings.fitness_method).strip().lower()

        end_reasons = []
        tasks = []
        population_state = {
            species: [m.state_dict() for m in self.population[species]]
            for species in self.species_list
        }
        for idx in range(self.settings.num_agents):
            for species in self.species_list:
                other_species = [s for s in self.species_list if s != species]
                for eval_index in range(self.settings.agent_evaluations):
                    other_indices = {}
                    # pick random agent from other species
                    for other_species_name in other_species:
                        other_species_idx = eval_rng.randint(0, self.settings.num_agents - 1)
                        other_indices[other_species_name] = other_species_idx

                    tasks.append({
                        "agent_index": idx,
                        "species": species,
                        "eval_index": eval_index,
                        "other_indices": other_indices,
                        "map_seed": eval_seeds[eval_index],
                        "python_random_seed": eval_rng.randrange(2**32),
                    })

        local_model_cache = {species: {} for species in self.species_list}

        def _get_local_model(species, idx):
            cache = local_model_cache[species]
            model = cache.get(idx)
            if model is None:
                model = Model(chromosome=population_state[species][idx])
                cache[idx] = model
            return model

        def _run_eval_task_local(task, simple_win_threshold=None):
            override_policy = str(task.get("override_policy", "")).strip().lower()
            if override_policy == "random":
                self_model = RandomBaselinePolicy(seed=int(task.get("override_policy_seed", task["python_random_seed"])))
            else:
                self_model = _get_local_model(task["species"], task["agent_index"])
            eval_species = {task["species"]: self_model}
            for sp, idx in task["other_indices"].items():
                eval_species[sp] = _get_local_model(sp, idx)

            self.agent_index = task["agent_index"]
            self.eval_index = task["eval_index"]

            map_seed = task["map_seed"]
            python_random_seed = task["python_random_seed"]
            start_time = time.time()
            fitness, episode_length, end_reason = self.run(
                eval_species,
                task["species"],
                map_seed,
                is_evaluation=False,
                callback=noop,
                collect_plot_data=True,
                python_random_seed=python_random_seed,
                simple_win_threshold=simple_win_threshold,
                max_steps_override=task.get("max_steps_override"),
                initial_energy_scale=task.get("initial_energy_scale", 1.0),
                energy_decay_per_cycle=task.get("energy_decay_per_cycle", 0.0),
            )
            while episode_length == 0:
                if self.settings.enable_logging:
                    print("something went wrong update this seed")
                map_seed = random.randrange(100000)
                python_random_seed = random.randrange(2**32)
                fitness, episode_length, end_reason = self.run(
                    eval_species,
                    task["species"],
                    map_seed,
                    is_evaluation=False,
                    callback=noop,
                    collect_plot_data=True,
                    python_random_seed=python_random_seed,
                    simple_win_threshold=simple_win_threshold,
                    max_steps_override=task.get("max_steps_override"),
                    initial_energy_scale=task.get("initial_energy_scale", 1.0),
                    energy_decay_per_cycle=task.get("energy_decay_per_cycle", 0.0),
                )

            end_time = time.time()
            return {
                "agent_index": task["agent_index"],
                "species": task["species"],
                "eval_index": task["eval_index"],
                "fitness": fitness,
                "episode_length": episode_length,
                "end_reason": end_reason,
                "steps_per_sec": episode_length / max(end_time - start_time, 1e-9),
            }

        # Optional custom pipeline:
        # 1) short eval for all candidates
        # 2) long eval for top candidates
        # 3) blended score for selection
        two_stage_enabled = bool(self.settings.two_stage_eval_enabled) and fitness_method == "biomass_pct"
        if fitness_method == "biomass_pct":
            short_steps = int(self.settings.fitness_eval_steps)
        else:
            short_steps = int(self.settings.max_steps)
        if two_stage_enabled:
            short_steps = int(self.settings.short_eval_steps)
        short_steps = max(1, short_steps)
        long_steps = max(short_steps, int(self.settings.long_eval_steps))
        long_weight = float(np.clip(self.settings.long_eval_weight, 0.0, 1.0))
        top_fraction = float(np.clip(self.settings.long_eval_top_fraction, 0.0, 1.0))
        top_n = max(1, int(np.ceil(self.settings.num_agents * top_fraction))) if top_fraction > 0 else 0
        relative_baseline_enabled = bool(self.settings.relative_baseline_enabled) and fitness_method == "biomass_pct"
        relative_baseline_policy = str(getattr(self.settings, "relative_baseline_policy", "random")).strip().lower()
        if relative_baseline_enabled and relative_baseline_policy != "random":
            raise ValueError(
                f"Unsupported relative_baseline_policy '{self.settings.relative_baseline_policy}'. "
                "Use 'random'."
            )

        initial_energy_scale = max(0.0, float(self.settings.training_initial_energy_scale))
        energy_decay_per_cycle = float(np.clip(self.settings.training_energy_decay_per_cycle, 0.0, 0.95))

        use_custom_pipeline = (
            two_stage_enabled
            or relative_baseline_enabled
            or abs(initial_energy_scale - 1.0) > 1e-9
            or energy_decay_per_cycle > 0.0
        )

        if use_custom_pipeline:
            def _execute_tasks(task_list):
                if not task_list:
                    return []
                local_results = []
                if self.settings.num_workers <= 1:
                    for task in task_list:
                        local_results.append(_run_eval_task_local(task, simple_win_threshold=None))
                else:
                    with ProcessPoolExecutor(
                        max_workers=self.settings.num_workers,
                        initializer=_init_worker,
                        initargs=(self.settings, self.env.render_mode, self.env.map_folder, population_state),
                    ) as executor:
                        for res in executor.map(_run_eval_task_worker, task_list, chunksize=1):
                            local_results.append(res)
                return local_results

            def _baseline_seed(task):
                species_code = sum((i + 1) * ord(ch) for i, ch in enumerate(task["species"]))
                return int(
                    (
                        int(task["map_seed"]) * 1_000_003
                        + int(task["eval_index"]) * 97
                        + species_code
                        + 17
                    )
                    % (2**32)
                )

            short_tasks = []
            for task in tasks:
                t = dict(task)
                t["max_steps_override"] = short_steps
                t["initial_energy_scale"] = initial_energy_scale
                t["energy_decay_per_cycle"] = energy_decay_per_cycle
                short_tasks.append(t)

            short_results = _execute_tasks(short_tasks)
            short_by_key = {(r["agent_index"], r["species"], r["eval_index"]): r for r in short_results}
            end_reasons.extend([r["end_reason"] for r in short_results])
            short_scores_by_key = {
                (r["agent_index"], r["species"], r["eval_index"]): float(r["fitness"])
                for r in short_results
            }

            if relative_baseline_enabled:
                baseline_short_tasks = []
                for task in short_tasks:
                    t = dict(task)
                    t["override_policy"] = "random"
                    t["override_policy_seed"] = _baseline_seed(task)
                    baseline_short_tasks.append(t)
                baseline_short_results = _execute_tasks(baseline_short_tasks)
                baseline_short_by_key = {
                    (r["agent_index"], r["species"], r["eval_index"]): r
                    for r in baseline_short_results
                }
                for key, score in list(short_scores_by_key.items()):
                    baseline_result = baseline_short_by_key.get(key)
                    if baseline_result is None:
                        continue
                    short_scores_by_key[key] = score - float(baseline_result["fitness"])

            short_avg = {}
            for idx in range(self.settings.num_agents):
                for species in self.species_list:
                    vals = [
                        short_scores_by_key[(idx, species, eval_index)]
                        for eval_index in range(self.settings.agent_evaluations)
                    ]
                    short_avg[(idx, species)] = float(np.mean(vals))

            long_candidates = set()
            if two_stage_enabled and long_steps > short_steps and top_n > 0:
                for species in self.species_list:
                    scored = [(idx, short_avg[(idx, species)]) for idx in range(self.settings.num_agents)]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    for idx, _ in scored[:top_n]:
                        long_candidates.add((idx, species))

            long_by_key = {}
            long_scores_by_key = {}
            if long_candidates:
                long_tasks = []
                for task in tasks:
                    key = (task["agent_index"], task["species"])
                    if key not in long_candidates:
                        continue
                    t = dict(task)
                    t["max_steps_override"] = long_steps
                    t["initial_energy_scale"] = initial_energy_scale
                    t["energy_decay_per_cycle"] = energy_decay_per_cycle
                    long_tasks.append(t)
                long_results = _execute_tasks(long_tasks)
                long_by_key = {(r["agent_index"], r["species"], r["eval_index"]): r for r in long_results}
                end_reasons.extend([r["end_reason"] for r in long_results])
                long_scores_by_key = {
                    (r["agent_index"], r["species"], r["eval_index"]): float(r["fitness"])
                    for r in long_results
                }

                if relative_baseline_enabled:
                    baseline_long_tasks = []
                    for task in long_tasks:
                        t = dict(task)
                        t["override_policy"] = "random"
                        t["override_policy_seed"] = _baseline_seed(task)
                        baseline_long_tasks.append(t)
                    baseline_long_results = _execute_tasks(baseline_long_tasks)
                    baseline_long_by_key = {
                        (r["agent_index"], r["species"], r["eval_index"]): r
                        for r in baseline_long_results
                    }
                    for key, score in list(long_scores_by_key.items()):
                        baseline_result = baseline_long_by_key.get(key)
                        if baseline_result is None:
                            continue
                        long_scores_by_key[key] = score - float(baseline_result["fitness"])

            for idx in range(self.settings.num_agents):
                self.agent_index = idx
                for species in self.species_list:
                    score = short_avg[(idx, species)]
                    if (idx, species) in long_candidates:
                        long_vals = [
                            long_scores_by_key[(idx, species, eval_index)]
                            for eval_index in range(self.settings.agent_evaluations)
                            if (idx, species, eval_index) in long_scores_by_key
                        ]
                        if long_vals:
                            long_score = float(np.mean(long_vals))
                            score = (1.0 - long_weight) * score + long_weight * long_score

                    fitnesses[species].append((self.population[species][idx].state_dict(), float(score)))
                    if self.settings.enable_logging:
                        print(f'finished eval for species: {species}, idx {idx}, fitness: {score:.3f}')

                    # Ensure plot data reflects the final score used for selection.
                    if self.settings.enable_plotting:
                        for eval_index in range(self.settings.agent_evaluations):
                            key = (idx, species, eval_index)
                            base_res = long_by_key.get(key, short_by_key.get(key))
                            if base_res is None:
                                continue
                            process_data({
                                'species': species,
                                'agent_index': idx,
                                'eval_index': eval_index,
                                'step': base_res["episode_length"],
                                'fitness': float(score),
                            }, self.env.plot_data)

            if self.settings.enable_logging:
                print(
                    f"Custom eval: short_steps={short_steps}, long_steps={long_steps}, "
                    f"top_n={top_n}, long_weight={long_weight:.2f}, "
                    f"energy_scale={initial_energy_scale:.2f}, energy_decay={energy_decay_per_cycle:.3f}, "
                    f"relative_baseline={relative_baseline_enabled}"
                )
                print("End reasons:", {reason: end_reasons.count(reason) for reason in set(end_reasons)})
            return fitnesses

        results = []
        if self.settings.num_workers <= 1:
            tasks_by_key = {
                (task["agent_index"], task["species"], task["eval_index"]): task
                for task in tasks
            }
            best_avg_so_far = {species: -float("inf") for species in self.species_list}

            for idx in range(self.settings.num_agents):
                self.agent_index = idx
                for species in self.species_list:
                    evals_fitness = []
                    for eval_index in range(self.settings.agent_evaluations):
                        task = tasks_by_key[(idx, species, eval_index)]
                        self.eval_index = eval_index

                        simple_win_threshold = None
                        can_short_circuit_last_candidate = (
                            fitness_method == "simple"
                            and self.settings.stop_last_candidate_when_winner
                            and idx == self.settings.num_agents - 1
                            and np.isfinite(best_avg_so_far[species])
                        )
                        if can_short_circuit_last_candidate:
                            completed_sum = float(sum(evals_fitness))
                            guaranteed_min_avg = completed_sum / self.settings.agent_evaluations
                            if guaranteed_min_avg > best_avg_so_far[species]:
                                skipped = self.settings.agent_evaluations - len(evals_fitness)
                                if skipped > 0:
                                    evals_fitness.extend([0.0] * skipped)
                                    end_reasons.extend(["skipped_last_candidate_already_won"] * skipped)
                                    if self.settings.enable_logging:
                                        print(
                                            f"idx {idx}, species {species}: short-circuiting {skipped} "
                                            f"remaining eval(s); winner already guaranteed."
                                        )
                                break
                            simple_win_threshold = (
                                (best_avg_so_far[species] * self.settings.agent_evaluations)
                                - completed_sum
                            )

                        r = _run_eval_task_local(task, simple_win_threshold=simple_win_threshold)
                        end_reasons.append(r["end_reason"])
                        evals_fitness.append(r["fitness"])
                        if self.settings.enable_logging:
                            print(
                                f'idx {idx}, eval {eval_index}, fitness {r["fitness"]:.1f}, '
                                f'episode_length {r["episode_length"]}, steps/sec {r["steps_per_sec"]:.2f}'
                            )

                        if can_short_circuit_last_candidate:
                            guaranteed_min_avg = float(sum(evals_fitness)) / self.settings.agent_evaluations
                            if guaranteed_min_avg > best_avg_so_far[species]:
                                skipped = self.settings.agent_evaluations - len(evals_fitness)
                                if skipped > 0:
                                    evals_fitness.extend([0.0] * skipped)
                                    end_reasons.extend(["skipped_last_candidate_already_won"] * skipped)
                                    if self.settings.enable_logging:
                                        print(
                                            f"idx {idx}, species {species}: short-circuiting {skipped} "
                                            f"remaining eval(s); winner already guaranteed."
                                        )
                                break

                    avg_fitness = float(sum(evals_fitness)) / self.settings.agent_evaluations
                    fitnesses[species].append((self.population[species][idx].state_dict(), avg_fitness))
                    best_avg_so_far[species] = max(best_avg_so_far[species], avg_fitness)
                    if self.settings.enable_logging:
                        print(f'finished eval for species: {species}, fitness: {avg_fitness:.1f}')

            if self.settings.enable_logging:
                print("End reasons:", {reason: end_reasons.count(reason) for reason in set(end_reasons)})
            return fitnesses
        else:
            with ProcessPoolExecutor(
                max_workers=self.settings.num_workers,
                initializer=_init_worker,
                initargs=(self.settings, self.env.render_mode, self.env.map_folder, population_state),
            ) as executor:
                # Small tasks improve load-balancing across workers.
                for res in executor.map(_run_eval_task_worker, tasks, chunksize=1):
                    results.append(res)

        # Aggregate results in deterministic order.
        results_by_key = {(r["agent_index"], r["species"], r["eval_index"]): r for r in results}
        for idx in range(self.settings.num_agents):
            self.agent_index = idx
            for species in self.species_list:
                evals_fitness = []
                for eval_index in range(self.settings.agent_evaluations):
                    r = results_by_key[(idx, species, eval_index)]
                    self.eval_index = eval_index
                    end_reasons.append(r["end_reason"])
                    evals_fitness.append(r["fitness"])
                    if self.settings.enable_logging:
                        print(f'idx {idx}, eval {eval_index}, fitness {r["fitness"]:.1f}, episode_length {r["episode_length"]}, steps/sec {r["steps_per_sec"]:.2f}')

                    # Minimal plot data update for parallel runs (preserves generation stats).
                    if self.settings.num_workers > 1 and self.settings.enable_plotting:
                        process_data({
                            'species': species,
                            'agent_index': idx,
                            'eval_index': eval_index,
                            'step': r["episode_length"],
                            'fitness': r["fitness"],
                        }, self.env.plot_data)

                avg_fitness = sum(evals_fitness) / len(evals_fitness)
                fitnesses[species].append((self.population[species][idx].state_dict(), avg_fitness))
                if self.settings.enable_logging:
                    print(f'finished eval for species: {species}, fitness: {avg_fitness:.1f}')

        if self.settings.enable_logging:
            print("End reasons:", {reason: end_reasons.count(reason) for reason in set(end_reasons)})
        return fitnesses

    def evolve_population(self, fitnesses):
        """
        Given a dictionary of fitnesses (per species), evolve each species' population.
        """
        new_population = {}
        fittest_agents_for_generation = {}
        for species in self.species_list:
            fittest_agent = max(fitnesses[species], key=lambda x: x[1])
            fittest_agents_for_generation[species] = Model(chromosome=fittest_agent[0])

        for species in self.species_list:
            current_population = fitnesses[species]

            # Select elites.
            elites = evolution.elitism_selection(current_population, self.settings.elitism_selection)
            next_pop = []

            next_gen_agents = self.settings.num_agents - 1
            if (self.current_generation < 25):
                next_gen_agents = self.settings.num_agents - 1 - 2

            while len(next_pop) < next_gen_agents:
                (p1, _), (p2, _) = evolution.tournament_selection(elites, 2, self.settings.tournament_selection)
                current_eta = min(10.0, self.settings.sbx_eta * (self.settings.sbx_eta_decay ** self.current_generation))
                c1_weights, c2_weights = evolution.sbx_crossover(p1, p2, current_eta)
                current_mutation_rate = max(self.settings.mutation_rate_min, self.settings.mutation_rate * (self.settings.mutation_rate_decay ** self.current_generation))
                evolution.mutation(c1_weights, current_mutation_rate, current_mutation_rate)
                evolution.mutation(c2_weights, current_mutation_rate, current_mutation_rate)
                next_pop.append(c1_weights)
                next_pop.append(c2_weights)

            # Update best fitness/agent.
            best_for_species = max(current_population, key=lambda x: x[1])
            if best_for_species[1] > self.best_fitness[species] + 0.01:
                self.best_fitness[species] = best_for_species[1]
                self.best_agent[species] = best_for_species[0]
                model = Model(chromosome=self.best_agent[species])
                model.save(f'{self.settings.folder}/agents/{self.current_generation}_${species}_{self.best_fitness[species]}.npy')

            # add best agent to the population
            next_pop.append(self.best_agent[species])

            new_population[species] = [Model(chromosome=chrom) for chrom in next_pop]
        
            if (self.current_generation < 25):
                new_population[species].append(Model())
                new_population[species].append(Model())
                
        for species in self.species_list:
            # shuffle the population
            self.rng.shuffle(new_population[species])

        self.population = new_population

    def _build_generation_champion_models(self, fitnesses):
        generation_best_states = {}
        best_species = None
        best_score = -float("inf")
        for species in self.species_list:
            best_state, best_species_score = max(fitnesses[species], key=lambda x: x[1])
            generation_best_states[species] = best_state
            if best_species_score > best_score:
                best_score = best_species_score
                best_species = species

        champion_models = {}
        for species in self.species_list:
            if species == best_species:
                benchmark_state = generation_best_states[species]
            elif self.best_agent.get(species) is not None:
                benchmark_state = self.best_agent[species]
            else:
                benchmark_state = generation_best_states[species]
            champion_models[species] = Model(chromosome=benchmark_state)

        return champion_models, best_species, best_score

    def _evaluate_generation_champions(self, fitnesses):
        every = max(1, int(self.settings.champion_progress_every))
        if (
            not self.settings.champion_progress_enabled
            or (self.current_generation % every) != 0
        ):
            self.champion_progress_history.append(np.nan)
            return

        # Build a single "champion benchmark" policy set:
        # one globally best champion for this generation + strongest available opponents.
        champion_models, best_species, _ = self._build_generation_champion_models(fitnesses)

        horizon = max(1, int(self.settings.champion_progress_steps))
        episodes = max(1, int(getattr(self.settings, "champion_progress_episodes", 1)))
        lengths = []
        for episode_idx in range(episodes):
            seed = int(self.champion_progress_seed + self.current_generation * 9973 + episode_idx * 101)
            _, episode_length, _ = self.run(
                candidates=champion_models,
                species_being_evaluated=best_species if best_species is not None else self.species_list[0],
                seed=seed,
                is_evaluation=True,
                callback=noop,
                collect_plot_data=False,
                python_random_seed=seed,
                max_steps_override=horizon,
            )
            lengths.append(float(episode_length))

        self.champion_progress_history.append(float(np.mean(lengths)))

    def _evaluate_fixed_validation(self, fitnesses):
        every = max(1, int(getattr(self.settings, "fixed_validation_every", 1)))
        if (
            not bool(getattr(self.settings, "fixed_validation_enabled", False))
            or (self.current_generation % every) != 0
        ):
            self.fixed_validation_history.append(np.nan)
            return

        champion_models, best_species, _ = self._build_generation_champion_models(fitnesses)
        horizon = max(1, int(getattr(self.settings, "fixed_validation_steps", self.settings.fitness_eval_steps)))
        episodes = max(1, int(getattr(self.settings, "fixed_validation_episodes", 1)))
        metric_values = []
        for episode_idx in range(episodes):
            seed = int(self.fixed_validation_seed + episode_idx * 9973)
            fitness, episode_length, _ = self.run(
                candidates=champion_models,
                species_being_evaluated=best_species if best_species is not None else self.species_list[0],
                seed=seed,
                is_evaluation=True,
                callback=noop,
                collect_plot_data=False,
                python_random_seed=seed,
                max_steps_override=horizon,
            )
            if self.fixed_validation_metric == "survival":
                metric_values.append(float(episode_length))
            else:
                metric_values.append(float(fitness))

        self.fixed_validation_history.append(float(np.mean(metric_values)))

    def run_generation(self):
        # Evaluate the current population.
        fitnesses = self.evaluate_population()
        self.current_generation += 1
        self._evaluate_generation_champions(fitnesses)
        self._evaluate_fixed_validation(fitnesses)
        if self.settings.enable_logging:
            print(f"Generation {self.current_generation} complete. Fitnesses: { {sp: max(fit, key=lambda x: x[1])[1] for sp, fit in fitnesses.items()} }")
            if self.champion_progress_history and np.isfinite(self.champion_progress_history[-1]):
                print(f"Champion benchmark survival: {self.champion_progress_history[-1]:.1f}")
            if self.fixed_validation_history and np.isfinite(self.fixed_validation_history[-1]):
                fixed_label = "survival cycles" if self.fixed_validation_metric == "survival" else "fitness"
                print(f"Fixed validation ({fixed_label}): {self.fixed_validation_history[-1]:.3f}")
        if self.settings.enable_plotting:
            generations_data = update_generations_data(self.settings, self.current_generation, self.env.plot_data)
            plot_generations(
                self.settings,
                generations_data,
                champion_progress=self.champion_progress_history,
                fixed_validation=self.fixed_validation_history,
                fixed_validation_metric=self.fixed_validation_metric,
            )
        self.env.plot_data = {}
        # Evolve the population based on fitnesses.
        self.evolve_population(fitnesses)

    def train(self):
        for i in range(self.settings.generations_per_run):
            self.run_generation()
            # if i > 25 and i % 5 == 0:
            #     self.optimize_params()
            # Optionally, save best models or log additional statistics.

    def evaluate(self, model_paths = [], callback = noop):
        candidates = {}
        for model_path in model_paths:
            model = Model(
                chromosome=np.load(model_path['path'])
            )
            candidates[model_path['species']] = model

        return self.run(candidates, "cod", None, True, callback)


def _init_worker(settings: Settings, render_mode: str, map_folder: str, population_state):
    global _WORKER_RUNNER
    global _WORKER_POPULATION_STATE
    global _WORKER_MODEL_CACHE
    _WORKER_RUNNER = PettingZooRunner(
        settings=settings,
        render_mode=render_mode,
        map_folder=map_folder,
        build_population=False,
    )
    _WORKER_POPULATION_STATE = population_state
    _WORKER_MODEL_CACHE = {species: {} for species in population_state}


def _run_eval_task_worker(task):
    global _WORKER_RUNNER
    global _WORKER_POPULATION_STATE
    global _WORKER_MODEL_CACHE
    runner = _WORKER_RUNNER

    def _get_worker_model(species, idx):
        cache = _WORKER_MODEL_CACHE[species]
        model = cache.get(idx)
        if model is None:
            model = Model(chromosome=_WORKER_POPULATION_STATE[species][idx])
            cache[idx] = model
        return model

    override_policy = str(task.get("override_policy", "")).strip().lower()
    if override_policy == "random":
        self_model = RandomBaselinePolicy(seed=int(task.get("override_policy_seed", task["python_random_seed"])))
    else:
        self_model = _get_worker_model(task["species"], task["agent_index"])
    eval_species = {task["species"]: self_model}
    for sp, idx in task["other_indices"].items():
        eval_species[sp] = _get_worker_model(sp, idx)

    map_seed = task["map_seed"]
    python_random_seed = task["python_random_seed"]
    start_time = time.time()
    fitness, episode_length, end_reason = runner.run(
        eval_species,
        task["species"],
        map_seed,
        is_evaluation=False,
        callback=noop,
        collect_plot_data=False,
        python_random_seed=python_random_seed,
        max_steps_override=task.get("max_steps_override"),
        initial_energy_scale=task.get("initial_energy_scale", 1.0),
        energy_decay_per_cycle=task.get("energy_decay_per_cycle", 0.0),
    )
    while episode_length == 0:
        if runner.settings.enable_logging:
            print("something went wrong update this seed")
        map_seed = random.randrange(100000)
        python_random_seed = random.randrange(2**32)
        fitness, episode_length, end_reason = runner.run(
            eval_species,
            task["species"],
            map_seed,
            is_evaluation=False,
            callback=noop,
            collect_plot_data=False,
            python_random_seed=python_random_seed,
            max_steps_override=task.get("max_steps_override"),
            initial_energy_scale=task.get("initial_energy_scale", 1.0),
            energy_decay_per_cycle=task.get("energy_decay_per_cycle", 0.0),
        )

    end_time = time.time()
    return {
        "agent_index": task["agent_index"],
        "species": task["species"],
        "eval_index": task["eval_index"],
        "fitness": fitness,
        "episode_length": episode_length,
        "end_reason": end_reason,
        "steps_per_sec": episode_length / max(end_time - start_time, 1e-9),
    }
