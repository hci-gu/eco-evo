import lib.constants as const
from lib.model import Model
import lib.evolution as evolution
import gymnasium as gym
import torch
import copy
import random

class SingleAgentGymRunner():
    def __init__(self):
        self.env = gym.make('gymnasium_env/Ecotwin-v0', render_mode="none")

        self.current_generation = 0
        self.agents = [(Model().state_dict(), 0) for _ in range(const.NUM_AGENTS)]
        self.best_fitness = 0
        self.best_agent = None

    def run_generation(self):
        species_order = [species for species in const.SPECIES_MAP.keys()]

        agent_fitnesses = [0] * len(self.agents)
        for idx, (agent_dict, _) in enumerate(self.agents):
            agent = Model(chromosome=agent_dict)
            for _ in range(const.AGENT_EVALUATIONS):
                self.env.reset()
                episode_done = False

                while not episode_done:
                    species_order = sorted(species_order, key=lambda x: random.random())

                    for species in species_order:
                        if episode_done:
                            break
                        if species == "plankton":
                            self.env.step({
                                "species": species,
                                "action": None,
                                "positions": None
                            })
                            continue
                        # i need observation here, cant take previous one because its the observation of another species
                        obs, selection = self.env.unwrapped.get_obs(species)
                        
                        action = agent.forward(obs, species)
                        
                        _, reward, done, truncated, info = self.env.step({
                            "species": species,
                            "action": action,
                            "positions": selection
                        })
                        episode_done = done
                        # print(reward, done, truncated, info)
                    self.env.render()

                    agent_fitnesses[idx] += reward
                    self.env.unwrapped.step_count += 1

        for i, fitness in enumerate(agent_fitnesses):
            self.agents[i] = (self.agents[i][0], fitness / const.AGENT_EVALUATIONS)

        self.next_generation()

    def next_generation(self):
        self.current_generation += 1
        
        fittest_agent = max(self.agents, key=lambda x: x[1])

        average_fitness = sum([x[1] for x in self.agents]) / len(self.agents)
        if average_fitness > self.best_fitness:
            self.best_fitness = average_fitness
            print(f"New best fitness: {self.best_fitness}")
            self.best_agent = copy.deepcopy(fittest_agent[0])
            model = torch.jit.script(Model(chromosome=fittest_agent[0]))
            torch.jit.save(model, f'{const.CURRENT_FOLDER}/agents/{self.current_generation}_{self.best_fitness}.pt')

        elites = evolution.elitism_selection(self.agents, const.ELITISM_SELECTION)
        next_pop = []

        while len(next_pop) < (const.NUM_AGENTS - 2):
            (p1, _), (p2, _) = evolution.tournament_selection(elites, 2, const.TOURNAMENT_SELECTION)

            c1_weights, c2_weights = evolution.crossover(p1, p2)

            current_mutation_rate = max(const.MIN_MUTATION_RATE, const.INITIAL_MUTATION_RATE * (const.MUTATION_RATE_DECAY ** self.current_generation))
            evolution.mutation(c1_weights, current_mutation_rate, current_mutation_rate)
            evolution.mutation(c2_weights, current_mutation_rate, current_mutation_rate)

            next_pop.append((c1_weights, 0))
            next_pop.append((c2_weights, 0))
            
        self.agents = next_pop
        self.agents.append((fittest_agent[0], 0))
        self.agents.append((self.best_agent, 0))
        self.agents.append((Model().state_dict(), 0))

