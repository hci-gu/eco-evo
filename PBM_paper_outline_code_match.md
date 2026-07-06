# PBM Paper Outline vs Current Code

## Summary

The current environment matches the broad PBM idea: functional groups live on a grid, carry biomass and reserve energy, make move/eat/rest decisions, interact through predation, and are trained against energy-based outcomes.

The main gap is that the paper describes an age-group and population-unit model, while the code currently models continuous biomass per functional group and cell.

## What We Have

- Functional groups on a spatial grid.
- Biomass and energy reserve state per functional group and cell.
- Decision-maker policy actions: move, eat, rest/hide.
- Explicit action energy costs through `apply_energy_costs`.
- Movement separated from energy-cost settlement through `apply_movement`.
- Food-web interactions through menus, predation matrices, assimilation, and energy gain.
- Rest/hide reducing prey visibility.
- Starvation and biomass change based on energy level versus maintenance level.
- Training reward based on total energy: `B * energy_content + R`.

## What Is Missing

- Age groups such as juvenile/adult as a first-class model concept.
- Yearly reproduction, such as adult biomass turning into juvenile biomass.
- Juvenile-to-adult transitions.
- Integer population units with rounding below one unit.
- Detritus as a dedicated energy-flow component, unless added manually as a normal functional group.
- Seasonal migration rules.
- Hunting success as a probability model near reserve thresholds.
- A fixed reserve biomass fraction unavailable for predation.

## Likely Mismatches

- The paper says eat has a success rate; the code uses deterministic intake scaled by hunger, prey visibility, prey availability, and optional Holling-II handling.
- The paper treats hide/rest as affecting hunting probability; the code uses `rest` as both the hiding fraction and the resting-cost action fraction.
- The paper describes population units; the code splits continuous biomass fractions.
- The paper scenarios rely on pressures like trawling, noise, and rotor effects. The current environment does not model those pressure effects.

## Overall

The current code aligns with the action-energy-foodweb core of the outline. It does not yet implement the paper's age-group, reproduction, and discrete population-unit layers.
