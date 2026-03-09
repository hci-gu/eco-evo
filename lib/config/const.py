AGE_GROUP_SEPARATOR = "__a"

BASE_SPECIES = ['plankton', 'sprat', 'herring', 'cod']
ACTING_BASE_SPECIES = ['sprat', 'herring', 'cod']

SPECIES = BASE_SPECIES.copy()
ACTING_SPECIES = ACTING_BASE_SPECIES.copy()

def make_age_group_name(base_species: str, age_index: int) -> str:
    if base_species == "plankton":
        return base_species
    return f"{base_species}{AGE_GROUP_SEPARATOR}{age_index}"

def is_age_group(species_name: str) -> bool:
    return AGE_GROUP_SEPARATOR in species_name

def base_species_name(species_name: str) -> str:
    return species_name.split(AGE_GROUP_SEPARATOR)[0]

def list_age_groups(base_species: str, age_groups: int) -> list[str]:
    if base_species == "plankton" or age_groups <= 1:
        return [base_species]
    return [make_age_group_name(base_species, idx) for idx in range(age_groups)]

def configure_age_groups(age_groups: int) -> tuple[list[str], list[str]]:
    """
    Expand SPECIES/ACTING_SPECIES in-place based on age_groups.
    Plankton is always single-age.
    """
    age_groups = max(1, int(age_groups))

    new_species: list[str] = []
    for base in BASE_SPECIES:
        new_species.extend(list_age_groups(base, age_groups))

    new_acting: list[str] = []
    for base in ACTING_BASE_SPECIES:
        new_acting.extend(list_age_groups(base, age_groups))

    SPECIES[:] = new_species
    ACTING_SPECIES[:] = new_acting
    return SPECIES, ACTING_SPECIES
