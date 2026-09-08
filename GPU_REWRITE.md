# GPU-resident ecosystem training proposal

Implementation follow-up: the tensor backend, training entry point, benchmark,
and validation suite now exist. See [GPU_TRAINING.md](GPU_TRAINING.md) for the
implemented workflow and current validation limits. The remainder records the
original design proposal.

Proposal based on the current repository, inspected 2026-09-08. This is a design sketch, not an implemented or benchmarked GPU backend. Target assumption: an NVIDIA GPU on Linux. The inspected local Python environment reports PyTorch 2.2.2, CUDA unavailable, and MPS unavailable.

The proposed rewrite keeps all numerical simulation and ARS training work on the GPU: world generation and reset, observations, policy inference, ecological dynamics, random numbers, fitness, observation statistics, perturbation selection, and weight updates. Python handles configuration, launching compiled work, and occasional output. “100% GPU” here means a GPU-resident numerical loop, with no host round trips inside a rollout; configuration, disk I/O, and job orchestration still require the CPU.

The main opportunity is to evaluate a batch of entire ecosystems together. The current model already uses continuous biomass fields and mostly local array operations. It does not require a variable population of individual animal objects. ARS is gradient-free, so the rewrite also needs no backward pass or saved trajectory for differentiation.

The following code establishes the current design:

| Current component | Rewrite responsibility |
| --- | --- |
| `lib/environments/ecosystem_env/observations.py` | Batched GPU neighbor gathers and per-species observation packing |
| `lib/environments/ecosystem_env/policies.py` | GPU policy weights, batched matrix multiplication, masked softmax |
| `lib/environments/ecosystem_env/predation.py` | Simultaneous predator demand and prey-limited intake reductions |
| `lib/environments/ecosystem_env/movement.py` | Action energy settlement, neighbor transfers, edge redistribution |
| `lib/environments/ecosystem_env/population_change.py` | Growth, mortality, stochastic reseeding, extinction thresholds |
| `lib/runners/trainer.py` | GPU perturbations, rollout batching, reward reductions, ARS update |
| `lib/runners/parallel_worker.py` | Replaced on the GPU path by a single process owning a device batch |
| `lib/config/config_loader.py`, `lib/spawn/` | Parse static configuration once; generate and reset worlds on device |

`PolicyController.forward()` currently wraps NumPy observations with `torch.from_numpy()` and returns logits through `.cpu().numpy()` every tick. These are CPU array/tensor boundaries today, not evidence of existing PCIe traffic. Moving only the network to CUDA would introduce repeated host/device transfers while leaving the ecosystem on the CPU.

A small cProfile diagnostic used `mareld2.yaml`, a 60 × 60 grid, 10 perturbation pairs, one spawn world, 15 ticks per rollout, co-evolution, one worker, one Torch thread, default 2 × 30 sigmoid networks, and observation normalization. One iteration took 9.732 profiled seconds. Environment configuration loading accumulated 4.232 seconds across 21 calls, including 42 YAML loads; batched policy inference accumulated 2.737 seconds across 300 calls. These are instrumented timings from one short serial run, not production throughput measurements. Longer rollouts amortize setup differently, and the parallel CPU baseline must be measured separately.

1. **Represent the ecosystem as fixed-shape device arrays.**

   Use separate contiguous arrays for biomass, reserves, and previous hiding fractions. Functional-group identifiers and dictionaries become static metadata outside the numerical loop. Extinct cells remain zero-valued slots. Masks express optional behavior without extracting variable-length lists of active cells.

   Let `K` be perturbation pairs, `M` spawn worlds, `E = 2*K*M` independent ecosystems, `G` all functional groups, `D` decision makers, `C = H*W` cells, `F` padded observation width, and `A = 5+G` action outputs.

   | Device array | Logical shape |
   | --- | --- |
   | Biomass, reserves, previous hiding fraction | `[E, G, H, W]` each |
   | Observations | `[E, D, C, F]` |
   | Actions | `[E, D, C, A]` |
   | Food-web coefficients | `[D, G]`, shared across ecosystems |
   | Fitness accumulators | `[E, D]` |
   | Perturbations | `[K, P_d]` per species with `P_d` parameters |
   | Candidate layer weights | `[2*K, D, input_width, output_width]`, padded if needed |

   The active Mareld2 configuration has six functional groups, four decision makers, 11 action outputs, and observation widths 21, 21, 16, and 11. Preserve each species' input ordering and mask padding. Share candidate weights across the M worlds; do not duplicate weights per cell. Grouped matrix multiplication or explicit batching evaluates all species, cells, and candidates together.

   The existing default `K=10, M=1` gives 20 ecosystems. Using `M=5` gives 100 ecosystems. This illustrates available parallelism, not a proposal to change the number of evaluations for a speed comparison. Time steps remain sequential because each depends on the preceding state.

2. **Keep world creation and the full ecological step on device.**

   Parse YAML and resolve species ordering, observability, food webs, masks, and spawn dependencies once. Upload immutable data once. Generate refreshed spawn maps, initial reserves, and runtime environmental noise with GPU operations. Keep reusable initial-state buffers and reset them with device copies.

   GPU spawning must cover the configured strategies and their allocation rules, not just uniform initialization. The current “Perlin” strategy uses filtered noise and FFT smoothing. Colony maps can use batched center sampling and Gaussian fields; environment-driven spawning follows a static dependency order. Preserve periodic smoothing for spawn fields separately from ecological movement boundaries.

   The floor allocator's decrementing Python loop can be reformulated using sorted positive weights and cumulative sums. For a prefix of length `n`, test whether its smallest allocation `total_b * weight[n-1] / cumulative_weight[n-1]` meets the floor; choose the largest valid prefix subject to the current count bound. Keep stable tie ordering, allowed-cell masks, zero-weight behavior, and the below-one-cell fallback. Verify equivalence with the existing allocator before using this replacement.

   Each tick retains the existing order:

   ```text
   observations using previous hiding fractions
       -> frozen observation normalization
       -> batched policy logits, action masking, softmax
       -> subthreshold action collapse and action statistics
       -> simultaneous predation using current hide/rest choices
       -> action energy costs
       -> movement of the settled biomass and reserves
       -> population change and extinction cleanup
       -> update hiding history and accumulate fitness
   ```

   Keep biomass fractions across movement/rest/eating actions, including the current subthreshold argmax rule. Do not replace these fractions with sampled individual actions. Preserve the specialist Type III and generalist Type II responses, shared-prey competition, and reserve losses with consumed biomass.

   Implement movement as destination gathers from four neighboring outgoing fluxes, using separate old/new buffers so cells never read partly updated neighbors. Preserve closed-edge masking and optional edge immigration; avoid introducing toroidal movement. Immigration concentration can use precomputed sorted edge indices plus device prefix tests and masks instead of Python scalar decisions.

3. **Move the entire ARS update to the same device.**

   Generate perturbations on the GPU, evaluate both signs against the same environmental random streams, average fitness over worlds, rank perturbation pairs, and update each species' parameters on device. Keep co-evolution and round-robin as explicit modes: co-evolution perturbs all target species together and produces a separate reward for each species from the same ecosystem.

   Preserve the current default reward:

   ```text
   total_energy[t] = sum_cells(biomass[t]) * energy_content
                     + sum_cells(reserves[t])
   fitness = mean_t(log((total_energy[t] + epsilon)
                       / (initial_total_energy + epsilon)))
   ```

   Preserve reward options, temperature, selected-pair reward standard deviation, and the current ARS step-size convention. In a Torch port use population standard deviation (`correction=0`) to match NumPy; Torch's default correction differs. [Torch standard deviation documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.std.html).

   Freeze observation normalization for the entire candidate batch, accumulate raw-observation moments during rollouts, and merge only after all candidates finish. Keep the existing inclusion of zero/empty cells unless deliberately changing the algorithm. Start with FP32 ecosystem fields and policy math, and FP64 reduction accumulators where the CPU code uses them; evaluate faster stable reductions only after checking reward rankings and normalization accuracy.

   Make randomness explicit. Use world-specific keys for shared spawn maps and pair/world-specific keys for initial reserve variation and runtime noise, with no sign component. Add species, tick, and cell counters as needed. This couples positive and negative evaluations while avoiding dependence on worker order or batch chunking. CPU NumPy seeds alone do not provide portable CPU/GPU identical streams; parity tests should inject the same initial arrays and noise arrays into both backends. Verify replay advances randomness across iterations while preserving coupling within each pair.

4. **Compile and replay substantial chunks of numerical work.**

   My first implementation choice is PyTorch CUDA because the project already has Torch policies and checkpoints. Write the new simulation as tensor functions, then apply `torch.compile` and CUDA Graph replay after correctness is established. Compilation can fuse operations; CUDA Graphs reduce the repeated host launch overhead. They are complementary, and a captured rollout remains multiple ordered GPU kernels. [PyTorch CUDA Graphs overview](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

   Start with a compiled tick and a captured fixed block of ticks, for example 32, with a separately handled remainder. Keep state, accumulators, tick counters, and random state at stable device addresses. Reset them on device. Reuse buffers with explicit lifetimes; do not accidentally overwrite outputs still needed by the next block. Larger captures, up to a training iteration where practical, follow profiling. Avoid unrolling thousands of ticks into an enormous compiler graph.

   No `.cpu()`, `.numpy()`, `.item()`, `float(device_tensor)`, or Python branches based on GPU values inside the numerical loop. Express conditions with masks and reductions. Static configuration branches are fine. Check for implicit synchronization and graph breaks with profiling. `torch.compile(mode="reduce-overhead")` is a candidate to benchmark, not a guarantee of successful capture, particularly with mutated inputs. [Torch compile documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

   JAX is a credible alternative for a more extensive rewrite: pure state transitions, `vmap` across worlds, and `lax.scan` across time. Its scan primitive lowers to a loop with fixed-shape carried state, avoiding a Python tick loop; it does not make dependent ticks parallel or guarantee a faster kernel schedule. It would also require policy/checkpoint conversion and a different runtime. [JAX scan documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html).

   Add custom Triton/CUDA kernels only where measured launch or memory traffic remains expensive, likely observation packing and ecological update fusion. Keep shared GPU buffers and avoid a multi-framework host transfer boundary. Backend choice and achievable performance must be revisited if the actual target is Apple Silicon or another GPU vendor.

5. **Budget memory around simultaneous worlds, not trajectory length.**

   With 100 Mareld2 ecosystems at 60 × 60, FP32 biomass plus reserves occupy about 16.5 MiB. One padded observation buffer is about 115.4 MiB, and one action buffer about 60.4 MiB. Hiding state, masks, predation intermediates, policy activations, next-state buffers, compiler workspace, and captured graph pools add to this. These are arithmetic estimates, not measured peak VRAM.

   Retain only current/next state and online fitness/statistics accumulators during training. No autograd tape or full per-tick trajectory is needed. If the batch exceeds VRAM, evaluate fixed-size chunks and retain rewards/statistics on GPU. Average all worlds and select top perturbations only after the complete batch; never update parameters between chunks. Preserve each environment's identity and random stream independently of chunk size.

   Periodically copy a small metrics packet for logging and a selected, downsampled ecosystem for visualization. Checkpoint only at configured boundaries. Live visualization must not require copying every candidate's grid after every tick. Asynchronous snapshots need owned staging buffers and events so training cannot overwrite data during a copy. Disk output and UI refreshes are intentional transfer boundaries, not part of the resident training loop.

6. **Deliver through small, verifiable stages.**

   First implement a separate `lib/gpu/` backend with static configuration packing, GPU state, observation/policy evaluation, the full tick, and a headless rollout benchmark. Keep the current environment as a numerical reference. Then add GPU resets and ARS, checkpoint conversion, compiled replay, and sampled telemetry. This is a replacement of the numerical core, not a requirement to rewrite the configuration editor, API, and visualization tools.

   Validate each ecological stage against fixed CPU fixtures: neighbor direction, current versus previous hiding, action masks, energy settlement, shared-prey demand scaling, mortality, reseeding, thresholds, and migration. Assert positivity and movement conservation before documented reserve caps and losses. Test batch isolation and equivalence across chunk sizes. Compare a complete ARS update with fixed perturbations and injected randomness, including observation moments and top-pair ranking.

   One compatibility decision needs attention: the current batched policy forward hardcodes sigmoid, while `PolicyNetwork` supports other activations. Establish the intended reference behavior for non-sigmoid checkpoints before porting them.

   Benchmark unprofiled headless wall time against the existing multi-worker CPU runner with identical grid, K, M, ticks, policies, and reward settings. Report world-ticks/second, complete iteration time including reset and ARS, peak VRAM, cold compile/capture cost, and warmed performance separately. Synchronize at timing boundaries so measurements include GPU execution. Verify the timeline has no host/device copies or scalar synchronizations in rollouts, and inspect launch gaps. Check long-run ecological outcomes statistically because small floating-point differences can alter extinction decisions and subsequent trajectories.

   Substantial acceleration is plausible from ecosystem batching, removing repeated setup, GPU policy inference, and fused ecological operations. A 10× end-to-end improvement can be an engineering target to evaluate, but there is no GPU measurement here to substantiate that number. Small batches, short runs, compilation, memory bandwidth, and output frequency can limit the gain. More simulation throughput also does not by itself improve ARS sample efficiency or ecological convergence.
