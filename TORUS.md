# Periodic world for food-following experiments

Pass `--boundary torus` to `train.py`, `train_gpu.py`, and `inference.py`.
The default `bounded` mode preserves existing experiments. Torus takes
precedence over `--migration`: it replaces immigration/emigration, rather
than wrapping biomass and redistributing it a second time.

## Spatial contract

- North/south and east/west edges are adjacent, at the same column/row.
- Biomass and its reserves move directly to the adjacent wrapped cell.
  Movement costs, speeds and split suppression still apply. No edge pooling.
- Policy observations read the same wrapped neighbours, including hidden-prey
  visibility and observable impact fields in the CPU environment.
- Accessibility checks the wrapped destination. Land still blocks movement.
- Local source-tracked rewards attribute wrapped flows to their source cells;
  crossing the seam is not treated as leaving the simulated world.
- Passive currents use the same topology. Their scrolling noise is periodic
  too, with a whole number of coarse/fine noise cells per circumference, so
  there is no discontinuous noise seam. `--current-scale` is approximate in
  this mode. Flow hitting inaccessible habitat remains at its source.

The tensor engine uses wrapped neighbour indices, shared by observations,
movement, currents, split suppression and local reward. CPU transfer kernels
retain interior slice assignments and add the four seam transfers.
Existing tensor-engine restrictions on project impact layers are unchanged.
Fixed geographical maps are not smoothed across seams: a torus is a controlled
training topology, not necessarily a realistic boundary for a coastal domain.

## Randomly turning food blob

Combine with `--debug-food-blobs`. Torus food draws a random initial position,
uniform random heading, and integer segment duration (default 40-160 ticks,
inclusive). After exactly that many moves it draws a new heading and duration
for the next move. Control this with `--food-blob-segment-ticks MIN MAX`.
The centre speed stays `--food-blob-speed`; zero remains a stationary control.

The field uses shortest periodic distances, so it splits visually over a seam
and rejoins on the other side. Total food biomass and reserve stay constant.
Internal obstacles clip/renormalise it; an entirely blocked footprint falls
back to the nearest allowed cells, using periodic distance. It does not route
around obstacles and should be used on open habitat for a clean tracking test.

Every world has a small position, velocity, segment index and countdown state.
Random choices depend only on the world/blob seed and segment index. ARS plus
and minus candidates get identical trajectories; worlds and updates get
different keys. Reset and CUDA capture restore all motion buffers. Training
does not read GPU tensors back to the host to choose a direction or duration.
No growing trajectory table or finite repeated path is used.

## Training, progress and inference

Builders, multiprocessing workers, CPU visual probes and the separate progress
evaluation all inherit the topology and blob settings. The viewer title shows
`[TORUS]`. Live probes use fresh seeds between updates, logged as `probe_seed`
in `biomass.jsonl`. Progress evaluation separately keeps its fixed seed and
tick cap, with a default biomass band of 0.1 to 10 times initial biomass
(`--biomass-bounds 0.1 10`). This does not change the training reward. When
resuming an old progress history with different bounds, use a new `--plot-dir`.
Standalone inference needs the same ecology flags explicitly.

Start a new run/progress directory. Policy input dimensions are unchanged,
so old weights can technically be imported, but their neighbourhood semantics
and task have changed; do not interpret this as an exact training continuation.
GPU `--resume` rejects a different boundary. To deliberately reuse weights,
use a new run with `--init-from results/<old-run>`. Fresh training is recommended
for a clean comparison. Torus does not solve starvation or establish viability.

GPU, 24x24, survival-first reward, 300 training ticks and a 5000-tick progress cap:

```bash
uv run python train_gpu.py --device cuda --project mareld2.yaml --run-name food-blobs-torus --boundary torus --debug-food-blobs --food-blob-segment-ticks 40 160 --survival-reward --migration on --mortality off --generations inf --profile info --iter-per-gen 50 --n_eval_ticks 300 --eval-ticks 5000 --n_deltas 16 --grid "24*24" --worlds_refresh iteration --policynetwork 2 48 tanh --lr 0.03 --sigma 0.05 --visual
```

Standalone inference on the exported policies:

```bash
uv run python inference.py --project mareld2.yaml --run-name food-blobs-torus --boundary torus --debug-food-blobs --food-blob-segment-ticks 40 160 --grid "24*24" --mortality off --ticks 5000 --seed 17 --visual
```

The same commands are valid with `--currents on`; replenished debug food itself
ignores currents, as before. Other responding non-decision makers still drift.
