# Survival-first training reward

Enable with `--survival-reward --profile info` in `train.py` or `train_gpu.py`.
The existing total-energy reward remains the default when the flag is absent.
This changes training fitness only: biology, movement costs, starvation,
predation, food blobs and inference behaviour are unchanged.

For each decision-making species, in each rollout world:

```
fitness = (T + 0.5 * Q) / H
```

- `H`: configured rollout length (`--n_eval_ticks`, or GPU alias `--ticks`).
- `T`: consecutive post-tick states with total biomass at least 30% of that
  species' initial biomass. The first dip below the floor ends the count
  permanently, even if biomass later recovers. Configure with
  `--survival-reward-floor 0.3`. There is no upper biomass limit in this score.
- `Q`: average reserve fullness `clip(R / (B * max_energy_reserve), 0, 1)`
  over the final `ceil(T / 3)` viable ticks. This is the last third of the
  species' viable lifetime, not the last third of the rollout horizon.

Empty starts, zero biomass, invalid biomass/reserve values, negative reserves,
or failure on the first tick score zero. Species are scored independently;
the ecosystem continues to run for all `H` ticks so other species still receive
their own scores. A species that survives the entire rollout can score slightly
above 1 (at most `1 + 0.5/H`).

For the same initial world and horizon, one extra viable tick always outweighs
any reserve-quality difference. For example, at `H=300`, 20 viable ticks with
full reserves score `20.5/300`; 21 ticks with empty reserves score `21/300`.
Early death is not made attractive by dividing by a shorter lifetime.
Multiple worlds are averaged as before: this optimises mean performance,
not a worst-world survival guarantee.

The mode rejects local, legacy and population-stability reward combinations,
`--no-integral-reward`, and nonzero entropy bonuses or argmax penalties.
`--profile info` already disables those action-shaping terms.

## Progress and limitations

`--eval-ticks 5000` still controls the separate progress evaluation; it does
not change a `--n_eval_ticks 300` training horizon. Progress keeps its existing
biomass band (default 0.3 to 3 times initial biomass) and fixed evaluation seed.
Training survival uses only the lower bound. The two metrics are therefore
related but not identical, and neither is simply time to literal extinction.

Once policies reliably survive 300 ticks, the main term is saturated: increase
the training horizon to reward further improvements. Late-life reserve quality
alone cannot guarantee survival to tick 5000. Co-evolving prey and different
spawn worlds can still cause noisy progress. No training convergence or
ecological viability improvement is guaranteed by this reward change.

## Run

Start a fresh run to keep old and new reward histories separate. GPU `--resume`
checks that the reward mode and floor match its checkpoint. To reuse old
weights with the new objective, use a new run name plus
`--init-from results/<old-run>` instead of `--resume`. Inference uses exported
policies as before; no reward flag is needed for inference.

One-line GPU command for a 24x24 blob world, 300-tick training rollouts and
5000-tick progress evaluations:

```bash
uv run python train_gpu.py --device cuda --project mareld2.yaml --run-name food-blobs-survival --debug-food-blobs --survival-reward --migration on --mortality off --generations inf --profile info --iter-per-gen 50 --n_eval_ticks 300 --eval-ticks 5000 --n_deltas 16 --grid "24*24" --worlds_refresh iteration --policynetwork 2 48 tanh --lr 0.03 --sigma 0.05 --visual
```

GPU scoring stays on-device, including exact late-life tail averages. Its
prefix buffer scales with rollout ticks times candidate-worlds times species;
growing the horizon reallocates that buffer and recaptures CUDA graphs.
