from lib.config import const
from lib.config.settings import Settings
from lib.model import Model
import lib.evolution as evolution  # Your NumPy-based evolution functions
from lib.visualize import plot_generations
from lib.data_manager import update_generations_data, process_data
import numpy as np
import random
import copy
import time
from lib.environments.pbm_gym import PbmEnv, env_factory

def noop(a, b, c):
    pass

class PBMRunner():
    def __init__(self, settings: Settings, render_mode="none"):
        self.env = env_factory()
        self.settings = settings
        self._grid_shape = None
        self._grid_cell_count = None
        self._action_space_shape = None
        self._per_cell_action_dim = None
        self._action_batch_dim = None
        self._configure_action_geometry()
        initial_reset = self.env.reset()
        initial_obs = initial_reset[0] if isinstance(initial_reset, tuple) else initial_reset
        self._flat_obs_dim = self._flatten_obs(initial_obs).shape[1]
        self.current_generation = 0

        # Use all species in the simulation (including plankton, if desired).
        self.species_list = [species for species in const.ACTING_SPECIES]
        # Create a population for each species.
        self.population = [
            Model(input_size=self._flat_obs_dim, output_size=self._per_cell_action_dim)
            for _ in range(self.settings.num_agents)
        ]
        # Track best fitness and best model for each species.
        self.best_fitness = -float('inf')
        self.best_agent = None
        self.agent_index = 0
        self.eval_index = 0
        self.plot_data = {}

    def run(self, candidate, seed = None, is_evaluation = False, callback = noop):
        reset_output = self.env.reset(seed=seed)
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        flat_obs = self._flatten_obs(obs)
        steps = 0
        episode_length = 0
        fitness = 0
        # callback(self.env.world, fitness, False)

        env_terminated = False
        env_truncated = False

        while not env_terminated and not env_truncated and episode_length < self.settings.max_steps:
            steps += 1
            action_values = candidate.forward(flat_obs)
            action = self._reshape_action(action_values)
            obs, reward, terminated, truncated, infos = self.env.step(action)
            flat_obs = self._flatten_obs(obs)
            env_terminated = terminated
            env_truncated = truncated
            
            episode_length += 1
            fitness += reward
            if (is_evaluation == False or self.env.render_mode != "none"):
                process_data({
                    'species': None,
                    'agent_index': self.agent_index,
                    'eval_index': self.eval_index,
                    'step': episode_length,
                }, self.plot_data)
            callback(infos, fitness, False)
        
        print("end of simulation", fitness, episode_length)
        callback(None, fitness, True)
        return fitness, episode_length, None

    def _flatten_obs(self, obs):
        """
        Convert PBM observations (which may include channel-first tensors or nested tuples)
        into a 2D array shaped (num_cells, features) expected by the simple Model.
        """
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_array = np.asarray(obs, dtype=np.float32)
        grid_shape = tuple(self._grid_shape) if self._grid_shape is not None else None
        grid_cells = self._grid_cell_count

        if obs_array.ndim == 3:
            if grid_shape and obs_array.shape[1:] == grid_shape:
                # Channel-first (C, H, W) -> (H, W, C)
                obs_array = np.moveaxis(obs_array, 0, -1)
            elif grid_shape and obs_array.shape[:2] != grid_shape:
                raise ValueError(f"Unexpected observation shape {obs_array.shape} for grid {grid_shape}")

        if obs_array.ndim >= 4 and grid_shape and obs_array.shape[:2] != grid_shape:
            raise ValueError(f"Unexpected high-dimensional observation shape {obs_array.shape}")

        if grid_cells is not None:
            try:
                return obs_array.reshape(grid_cells, -1)
            except ValueError as exc:
                raise ValueError(
                    f"Failed to reshape observation of shape {obs_array.shape} into ({grid_cells}, -1)"
                ) from exc

        if obs_array.ndim == 2:
            return obs_array

        return obs_array.reshape(obs_array.shape[0], -1)

    def _configure_action_geometry(self):
        shape = getattr(self.env.action_space, "shape", None)
        if shape is None:
            raise ValueError("PBM environment must expose a Box action space.")

        self._action_space_shape = shape
        if len(shape) == 3:
            self._action_batch_dim = None
            self._grid_shape = tuple(shape[:2])
            self._per_cell_action_dim = shape[2]
        elif len(shape) == 4:
            self._action_batch_dim = shape[0]
            self._grid_shape = tuple(shape[1:3])
            self._per_cell_action_dim = shape[3]
        else:
            raise ValueError(f"Unsupported action space shape: {shape}")

        if self._action_batch_dim not in (None, 1):
            raise ValueError("PBMRunner currently supports only a single environment batch dimension.")

        self._grid_cell_count = int(np.prod(self._grid_shape))

    def _reshape_action(self, action_values):
        action_array = np.asarray(action_values, dtype=np.float32)
        try:
            action_array = action_array.reshape(self._grid_cell_count, self._per_cell_action_dim)
        except ValueError as exc:
            raise ValueError(
                f"Model output shape {action_array.shape} does not match expected "
                f"({self._grid_cell_count}, {self._per_cell_action_dim})"
            ) from exc

        reshaped = action_array.reshape(*self._grid_shape, self._per_cell_action_dim)
        if self._action_batch_dim:
            reshaped = reshaped[np.newaxis, ...]
        return reshaped

    def evaluate_population(self):
        """
        Evaluate each model in the population for every species.
        When evaluating a candidate for a species S, for every other species (not S),
        we use the best model from previous generations (if available) to decide their actions.
        Returns a dictionary mapping species to a list of (chromosome, fitness) tuples.
        """
        eval_seeds = [int(random.random() * 100000) for _ in range(self.settings.agent_evaluations)]
        fitnesses = []

        end_reasons = []
        for idx in range(self.settings.num_agents):
            self.agent_index = idx
            evaluation_candidate = self.population[idx]

            evals_fitness = []
            for eval_index in range(self.settings.agent_evaluations):
                start_time = time.time()
                self.eval_index = eval_index
                
                fitness, episode_length, end_reason = self.run(evaluation_candidate, eval_seeds[eval_index])
                end_reasons.append(end_reason)
                
                evals_fitness.append(fitness)
                end_time = time.time()
                print(f'idx {idx}, eval {eval_index}, fitness {fitness:.1f}, episode_length {episode_length}, steps/sec {episode_length/(end_time - start_time):.2f}')
            
            avg_fitness = sum(evals_fitness) / len(evals_fitness)

            fitnesses.append((evaluation_candidate.state_dict(), avg_fitness))
            print(f'finished eval, fitness: {avg_fitness:.1f}')

        return fitnesses

    def evolve_population(self, fitnesses):
        """
        Given a dictionary of fitnesses (per species), evolve each species' population.
        """
        new_population = []
        fittest_agent = max(fitnesses, key=lambda x: x[1])

        current_population = fitnesses

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
        if best_for_species[1] > self.best_fitness + 0.01:
            self.best_fitness = best_for_species[1]
            self.best_agent = best_for_species[0]
            model = Model(chromosome=self.best_agent)
            model.save(f'{self.settings.folder}/agents/{self.current_generation}_{self.best_fitness}.npy')

        # add best agent to the population
        next_pop.append(self.best_agent)

        new_population = [Model(chromosome=chrom) for chrom in next_pop]
    
        if (self.current_generation < 25):
            new_population.append(
                Model(input_size=self._flat_obs_dim, output_size=self._per_cell_action_dim)
            )
            new_population.append(
                Model(input_size=self._flat_obs_dim, output_size=self._per_cell_action_dim)
            )

        random.shuffle(new_population)

        self.population = new_population

    def run_generation(self):
        # Evaluate the current population.
        fitnesses = self.evaluate_population()
        self.current_generation += 1
        print(f"Generation {self.current_generation} complete")
        generations_data = update_generations_data(self.settings, self.current_generation, self.plot_data)
        plot_generations(self.settings, generations_data)
        self.env.plot_data = {}
        # Evolve the population based on fitnesses.
        self.evolve_population(fitnesses)

    def train(self):
        for i in range(self.settings.generations_per_run):
            self.run_generation()
            # if i > 25 and i % 5 == 0:
            #     self.optimize_params()
            # Optionally, save best models or log additional statistics.

    def evaluate(self, model_path = '', callback = noop):
        print(f"Evaluating PBM model from {model_path}")
        model = Model(
            chromosome=np.load(model_path)
        )

        return self.run(model, None, True, callback)
