"""
marine heatwave: metabolic rate increases
mortality rate increases
cost of growth increases

cyanobacteria

short-term (heatwave) vs long-term disturbance (climate change)
"""


class GenericFunctionalGroup(object):
    def __init__(
        self,
        properties=None,
        dispersal_ability=None,
        growth_rate=None,
        max_capacity=None,
        energy_table=None,
    ):
        self.properties = properties
        self.dispersal_ability = dispersal_ability
        self.growth_rate = growth_rate
        self.max_capacity = (
            max_capacity  # max theoretical capacity (eg if not predators)
        )
        self.energy_table = energy_table  # how much energy depending on what you eat


class AnimalFunctionalGroup(GenericFunctionalGroup):
    def __init__(
        self,
        properties=None,
        dispersal_ability=None,
        growth_rate=None,
        max_capacity=None,
        energy_table=None,
    ):
        super().__init__(
            properties=None,
            dispersal_ability=None,
            growth_rate=None,
            max_capacity=None,
            energy_table=None,
        )


class PlantFunctionalGroup(GenericFunctionalGroup):
    def __init__(self):
        self.properties = []
        self.growth_rate = None
        self.max_capacity = None  # max theoretical capacity (eg if not predators)
        self.env_suitability = None


class DetritusFunctionalGroup(GenericFunctionalGroup):
    pass


"""
animal functional group 

"""


class BioDivGrid(object):
    def __init__(self, grid_shape, temperature, salinity):
        self.grid_shape = grid_shape
        self.temperature = temperature
        self.salinity = salinity


# extract functions for testing
# MIGRATION STEP
def migrate(env, fg_list, vec):
    # current distribution of FGs
    current_h = env.h + 0
    new_h = np.zeros(current_h.shape)
    current_energy_tbl = env.energy_table + 0
    new_energy_tbl = env.energy_table * 0

    cell_counter = 0
    for x in range(0, grid_shape[0]):
        for y in range(0, grid_shape[1]):
            xy_mask = env.dispersal_masks[cell_counter]
            # loop over FGs
            for i, fg in enumerate(fg_list):
                h_i = current_h[i, x + env.padding, y + env.padding]
                m = (
                    fg.migration_action(
                        m=vec[cell_counter, i, :2],
                        s=float(vec[cell_counter, i, 2]),
                        mask=xy_mask,
                    )
                    * h_i
                )
                new_h[i, (x) : (x + 3), (y) : (y + 3)] += m
                # energy costs
                en_i = current_energy_tbl[i, x + env.padding, y + env.padding]
                new_energy_tbl[i, (x) : (x + 3), (y) : (y + 3)] += m * (
                    en_i + fg.migration_energy_cost
                )
            cell_counter += 1

    new_h = np.round(new_h)
    den = new_h + 0
    den[den == 0] = 1.0
    new_energy_tbl_r = new_energy_tbl / den
    return new_h, new_energy_tbl_r


# FEEDING STEP
def hide_n_feed(env, feed_fg, vec):
    current_h = env.h + 0
    new_h = np.zeros(current_h.shape)
    cell_counter = 0
    for x in range(grid_shape[0]):
        for y in range(grid_shape[0]):
            h_tmp = current_h[:, x + env.padding, y + env.padding]
            new_h[:, x + env.padding, y + env.padding] = feed_fg.feeding_hiding_action(
                h_tmp, vec[cell_counter, :, 3:]
            )
            cell_counter += 1

    energy_level = 0
    return new_h, energy_level


# biomass by product of energy level and n. individuals
# size could affect what you eat: keep track ofr average size (!= from health)
# size == age


# TEST FIX ROUNDING ERROR
import numpy as np

a = np.random.binomial(1, 0.4, (2, 5, 5))
# get indices of non zeros
x, y, z = np.where(a > 0)
N = 8
r = len(x)
rnd_i = np.random.choice(range(r), size=(3, 4), replace=True)

unique_indx, counts = np.unique(rnd_i, return_counts=True)

b = a + 0
b[x[unique_indx], y[unique_indx], z[unique_indx]] = (
    b[x[unique_indx], y[unique_indx], z[unique_indx]] + counts
)

b.sum() - a.sum() == N
