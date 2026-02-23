# API Docs

This API provides a two-step workflow:

1) Upload a map (texture + depth) to create a `simulation_id`.
2) Run the simulation and retrieve sampled per-cell biomass snapshots.

The server stores uploaded files under `maps/<simulation_id>/map.png` and `maps/<simulation_id>/depth.png`.

## 1) Create simulation

**Route**

`POST /simulate/upload`

**Form fields**

- `texture`: file (saved as `map.png`)
- `depth`: file (saved as `depth.png`)
- `options`: JSON string (see “Options” below)

**Response (JSON)**

- `id`: string UUID (`simulation_id`)

## 2) Run simulation + fetch biomass samples

**Route**

`GET /simulate/<simulation_id>`

This runs a simulation using the uploaded map and returns a sampled biomass time series.

### Sampling behavior

- The simulation runs for `settings.max_steps` environment steps.
- Biomass grids are sampled every `sampleEvery` steps (default `10`).
- The biomass snapshot is taken from the *unpadded* world grid: `world[1:-1, 1:-1, :]`.

### Query parameters

- `worldSize` / `world_size` (int): overrides `Settings.world_size` (default `50`).
- `maxSteps` / `max_steps` (int): overrides `Settings.max_steps` (default `3000`).
- `sampleEvery` / `sample_every` (int): sample interval (default `10`).
- `includeFinal` / `include_final` (bool): include final sample (default `true`).
- `format` (string): output format:
  - `base64` (default): JSON with base64-encoded raw bytes
  - `npz`: returns `application/octet-stream` containing a compressed NumPy `.npz`
- `agentSet` / `agent_set` / `agent` / `agents` (string): load models from `agents/<agentSet>/`:
  - if the folder contains 1 `.npy.npz` / `.npz`, it is used as a shared model for all acting species
  - if it contains multiple files, the filename must include the acting species name (e.g. `cod`) and each acting species must be present
- `modelPath` / `model_path` (string): path to a `.npy.npz` / `.npz` model file (used as a shared model when `agentSet` is not provided)

### Response format: `format=base64` (default)

**Response (JSON)**

- `simulation_id`: string
- `world_size`: int
- `species`: array of strings (order of the last axis in `biomass`)
- `sample_every`: int
- `include_final`: bool
- `dtype`: string (currently `float32`)
- `shape`: array `[N, H, W, S]` where:
  - `N` = number of snapshots
  - `H=W=world_size`
  - `S` = number of species (`len(species)`)
- `steps`: array of ints (the simulation step index for each snapshot, length `N`)
- `fitness`: float (accumulated fitness used by runner)
- `episode_length`: int (number of environment steps completed)
- `end_reason`: string (if the world terminated early)
- `biomass_b64`: base64 string of the raw `biomass` tensor bytes in C-order

**How to decode `biomass_b64` (Python)**

```python
import base64
import numpy as np

# response_json = requests.get(...).json()
shape = response_json["shape"]          # [N, H, W, S]
raw = base64.b64decode(response_json["biomass_b64"])
biomass = np.frombuffer(raw, dtype=np.float32).reshape(shape)
steps = np.asarray(response_json["steps"], dtype=np.int32)
species = response_json["species"]
```

### Response format: `format=npz`

Returns `application/octet-stream` containing a compressed NumPy archive with:

- `biomass`: `float32` array shaped `(N, H, W, S)`
- `steps`: `int32` array shaped `(N,)`
- `species`: array of strings
- `world_size`: `int32` array shaped `(1,)`
- `sample_every`: `int32` array shaped `(1,)`

**How to read (Python)**

```python
import io
import numpy as np

# content = requests.get(...).content
with np.load(io.BytesIO(content)) as z:
    biomass = z["biomass"]
    steps = z["steps"]
    species = z["species"].tolist()
```

## Options

`options` is a JSON object passed to `POST /simulate/upload`. The server maps keys onto `lib.config.settings.Settings`.

Common keys:

- `world_size` / `worldSize`
- `max_steps` / `maxSteps`

Unrecognized keys are ignored.

## List available agent sets

**Route**

`GET /simulate/agents`

Returns an array of agent sets found under `agents/` (each agent set is a subfolder).
