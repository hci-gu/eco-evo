"""Sequential GPU rollout-length experiment with resumable jobs and an HTML report."""

import argparse
import hashlib
import html
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.gpu.cli import grid_size, positive_int
from lib.runners.progress_plot import plot_series, running_average, save_progress_plot


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_records(path):
    if not path.exists():
        return []
    # A process interruption may leave an incomplete final JSONL record.
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/rollout-experiment"))
    p.add_argument("--project", type=Path, default=ROOT / "mareld2.yaml")
    p.add_argument("--rollouts", type=positive_int, nargs="+", default=[100, 300, 1000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--generations", type=positive_int, default=25)
    p.add_argument("--iter-per-gen", type=positive_int, default=50)
    p.add_argument("--eval-ticks", type=positive_int, default=5000)
    p.add_argument("--eval-every", type=positive_int, default=50)
    p.add_argument("--eval-seed", type=int, default=20260530)
    p.add_argument("--progress-window", type=positive_int, default=5)
    p.add_argument("--grid", type=grid_size, default=(24, 24))
    p.add_argument("--n-deltas", type=positive_int, default=16)
    p.add_argument("--worlds", type=positive_int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--execution", choices=("eager", "compile", "cuda-graph", "compile-graph"))
    p.add_argument("--pairs-per-batch", type=positive_int)
    p.add_argument("--resume", action="store_true", help="Skip completed jobs; resume partial jobs from checkpoints")
    p.add_argument("--report-only", action="store_true", help="Rebuild report from the saved experiment, without training")
    p.add_argument("--dry-run", action="store_true", help="Print the job matrix without creating files or training")
    return p


def experiment_config(args):
    if len(set(args.rollouts)) != len(args.rollouts) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Rollouts and seeds must not contain duplicates")
    if any(not 0 <= s < 2**32 for s in [*args.seeds, args.eval_seed]):
        raise ValueError("Seeds must be in [0, 2**32)")
    config = {k: v for k, v in vars(args).items()
              if k not in {"output", "resume", "report_only", "dry_run"}}
    config["project"] = str(args.project.resolve())
    config["grid"] = list(args.grid)
    config["execution"] = args.execution or ("eager" if args.device == "cpu" else "cuda-graph")
    # Prevent accidental pooling/resuming across changed biology or simulator code.
    paths = [args.project.resolve(), ROOT / "fgconfig/fg_library.yaml", ROOT / "train_gpu.py",
             Path(__file__), *sorted((ROOT / "lib").rglob("*.py"))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    config["source_sha256"] = digest.hexdigest()
    return config


def training_command(config, job, directory, *, generations=None, resume=False):
    c = config
    command = [sys.executable, "-u", str(ROOT / "train_gpu.py"),
               "--project", c["project"], "--output", str(directory),
               "--device", c["device"], "--execution", c["execution"],
               "--boundary", "torus", "--debug-food-blobs",
               "--food-blob-segment-ticks", "40", "160", "--survival-reward",
               "--migration", "on", "--mortality", "off", "--profile", "info",
               "--generations", str(generations if generations is not None else c["generations"]),
               "--iter-per-gen", str(c["iter_per_gen"]), "--ticks", str(job["rollout"]),
               "--seed", str(job["seed"]), "--n-deltas", str(c["n_deltas"]),
               "--worlds", str(c["worlds"]), "--grid", "x".join(map(str, c["grid"])),
               "--worlds-refresh", "iteration", "--policynetwork", "2", "48", "tanh",
               "--lr", "0.03", "--sigma", "0.05", "--progress",
               "--eval-ticks", str(c["eval_ticks"]), "--eval-every", str(c["eval_every"]),
               "--eval-seed", str(c["eval_seed"]), "--biomass-bounds", "0.1", "10",
               "--progress-window", str(c["progress_window"])]
    if c["pairs_per_batch"] is not None:
        command.extend(["--pairs-per-batch", str(c["pairs_per_batch"])])
    if resume:
        command.append("--resume")
    return command


def checkpoint_progress(directory, iterations):
    checkpoint = directory / "trainer.pth"
    if not checkpoint.exists():
        return 0, 0
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    generation = int(payload["generation"])
    if int(payload["iteration_in_generation"]) >= iterations:
        generation += 1
    return generation, int(payload["trainer"]["iterations_completed"])


def make_report(output, manifest):
    """Report observed outcomes only; unfinished runs never enter aggregate rankings."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np

    c = manifest["config"]
    window = c["progress_window"]
    histories, rows, images = {}, [], []
    for job in manifest["jobs"]:
        directory = output / job["name"]
        records = read_records(directory / "progress/survival.jsonl")
        histories[job["name"]] = records
        evaluation_config = directory / "progress/config.json"
        if records and evaluation_config.exists():
            save_progress_plot(records, json.loads(evaluation_config.read_text()),
                               directory / "progress/progress.png", window, c["iter_per_gen"])
            images.append(f'<figure><figcaption>{html.escape(job["name"])} '
                          f'({job["status"]})</figcaption><img loading="lazy" '
                          f'src="{job["name"]}/progress/progress.png"></figure>')
        for fid in sorted({f for r in records for f in r["survival_ticks"]}):
            values = [r["survival_ticks"].get(fid) for r in records]
            tail = [v for v in values[-window:] if v is not None]
            valid = [v for v in values if v is not None]
            rows.append(dict(run=job["name"], status=job["status"], rollout=job["rollout"],
                             seed=job["seed"], species=fid, final=values[-1],
                             last_window_mean=float(np.mean(tail)) if tail else None,
                             best=max(valid) if valid else None,
                             tail_cap_fraction=float(np.mean(np.array(tail) == c["eval_ticks"])) if tail else None,
                             updates=records[-1]["step"], seconds=job.get("seconds", 0),
                             candidate_world_ticks=2 * c["n_deltas"] * c["worlds"] *
                             job["rollout"] * job.get("updates", 0)))
    write_json(output / "summary.json", rows)
    import csv
    if rows:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    species = sorted({r["species"] for r in rows})
    comparison = ""
    if species:
        fig = Figure(figsize=(12, 3.6 * ((len(species) + 1) // 2)), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots((len(species) + 1) // 2, 2, squeeze=False).flat
        for ax, fid in zip(axes, species):
            for index, rollout in enumerate(c["rollouts"]):
                jobs = [j for j in manifest["jobs"] if j["rollout"] == rollout and histories[j["name"]]]
                color = f"C{index % 10}"
                samples = {}
                for job in jobs:
                    for record in histories[job["name"]]:
                        value = record["survival_ticks"].get(fid)
                        if value is not None:
                            ax.scatter(record["step"] / c["iter_per_gen"], value,
                                       color=color, alpha=.5, s=10)
                            samples.setdefault(record["step"], []).append(value)
                # Only average checkpoints observed in every participating seed.
                steps = sorted(s for s, values in samples.items() if len(values) == len(jobs))
                if steps:
                    means = [np.mean(samples[s]) for s in steps]
                    ax.plot(np.array(steps) / c["iter_per_gen"], running_average(means, window),
                            color=color, label=f"H={rollout}, n={len(jobs)} seeds")
            ax.set(title=fid, xlabel="Generation", ylabel="Viable ticks",
                   ylim=(0, c["eval_ticks"] * 1.02))
            ax.grid(alpha=.2)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
        for ax in list(fig.axes)[len(species):]:
            ax.set_visible(False)
        fig.savefig(output / "comparison.png", dpi=140)
        comparison = '<img src="comparison.png" alt="Rollout comparison">'
    aggregates = []
    for fid in species:
        for rollout in c["rollouts"]:
            group = [r for r in rows if r["species"] == fid and r["rollout"] == rollout
                     and r["status"] == "complete" and r["last_window_mean"] is not None]
            if group:
                values = [r["last_window_mean"] for r in group]
                aggregates.append(dict(species=fid, rollout=rollout, completed_seeds=len(values),
                                       mean=float(np.mean(values)), std=float(np.std(values))))
    write_json(output / "aggregate.json", aggregates)
    def table(data, columns):
        def cell(value):
            return html.escape(f"{value:.2f}" if isinstance(value, float) else str(value))
        return "<table><tr>" + "".join(f"<th>{k}</th>" for k in columns) + "</tr>" + "".join(
            "<tr>" + "".join(f"<td>{cell(row.get(k, ''))}</td>" for k in columns) + "</tr>"
            for row in data) + "</table>"
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Rollout experiment</title>
<style>body{{font:15px system-ui;margin:30px;background:#fafafa;color:#222}}img{{width:100%;max-width:1400px}}
table{{border-collapse:collapse;margin:20px 0}}td,th{{padding:7px;border:1px solid #ccc;text-align:left}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr))}}figure{{margin:10px}}</style>
<h1>Rollout-length experiment</h1><p>{c['generations']} generations x {c['iter_per_gen']} updates;
progress cap {c['eval_ticks']} ticks; biomass band 0.1-10 x start; eval seed {c['eval_seed']}.</p>
<p>Dots: individual evaluations at 50% opacity. Lines: trailing {window}-evaluation means.
Comparison lines first average matching checkpoints across participating training seeds.
Partial runs are shown but excluded from aggregate summaries. Startup windows use fewer samples.
Scores at the cap are censored, not proof of survival beyond it. This is one fixed evaluation world,
not a generalization test. Equal generations do not mean equal compute.</p>
<h2>Jobs</h2>{table(manifest['jobs'], ['name', 'status', 'updates', 'seconds', 'error'])}
<h2>Comparison</h2>{comparison}
<h2>Completed-run summary</h2><p>Mean and population standard deviation across seeds of each run's
last {window} evaluations; not a confidence interval. Best single scores are not used to rank runs.</p>
{table(aggregates, ['species', 'rollout', 'completed_seeds', 'mean', 'std'])}
<h2>Per-run measurements</h2>{table(rows, ['run', 'species', 'final', 'last_window_mean', 'best', 'tail_cap_fraction', 'candidate_world_ticks'])}
<p><a href="summary.csv">CSV</a> | <a href="manifest.json">Settings and commands</a></p>
<h2>Progress images</h2><div class="gallery">{''.join(images)}</div></html>'''
    (output / "report.html").write_text(page, encoding="utf-8")


def run_job(config, job, directory):
    generations, updates = checkpoint_progress(directory, config["iter_per_gen"])
    job["updates"] = updates
    remaining = config["generations"] - generations
    if remaining > 0:
        command = training_command(config, job, directory, generations=remaining,
                                   resume=(directory / "trainer.pth").exists())
        job["command"] = command
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "trainer.pth").exists() and (directory / "progress/config.json").exists():
            raise ValueError("Partial progress without a checkpoint; use a fresh experiment output")
        with (directory / "console.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
            try:
                code = child.wait()
            except KeyboardInterrupt:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        if code:
            raise RuntimeError(f"Training exited {code}; see {directory / 'console.log'}")
    _, updates = checkpoint_progress(directory, config["iter_per_gen"])
    job["updates"] = updates
    records = read_records(directory / "progress/survival.jsonl")
    if updates != config["generations"] * config["iter_per_gen"]:
        raise RuntimeError("Training stopped before the requested update count")
    if not records or records[-1]["step"] != updates:
        raise RuntimeError("Final progress evaluation is missing; inspect console.log")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    output = args.output.resolve()
    path = output / "manifest.json"
    if args.report_only:
        make_report(output, json.loads(path.read_text(encoding="utf-8")))
        print(f"Report: {output / 'report.html'}")
        return 0
    try:
        config = experiment_config(args)
        jobs = [dict(name=f"rollout-{h}_seed-{s}", rollout=h, seed=s, status="pending")
                for h in args.rollouts for s in args.seeds]
        if args.dry_run:
            for job in jobs:
                print(json.dumps(training_command(config, job, output / job["name"])))
            return 0
        if path.exists():
            if not args.resume:
                raise ValueError("Experiment already exists; use --resume, --report-only, or a new --output")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["config"] != config:
                raise ValueError("Experiment settings/source changed; use a new output directory")
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError("Output directory must be empty for a new experiment")
            output.mkdir(parents=True, exist_ok=True)
            manifest = dict(format_version=1, config=config, jobs=jobs)
    except (ValueError, OSError) as error:
        p.error(str(error))
    write_json(path, manifest)
    for job in manifest["jobs"]:
        if job["status"] == "complete":
            continue
        print(f"Starting {job['name']} ({config['generations']} generations). "
              f"Log: {output / job['name'] / 'console.log'}", flush=True)
        job["status"] = "running"
        job.pop("error", None)
        write_json(path, manifest)
        started = time.perf_counter()
        interrupted = False
        try:
            run_job(config, job, output / job["name"])
            job["status"] = "complete"
        except KeyboardInterrupt:
            job["status"] = "interrupted"
            interrupted = True
        except Exception as error:
            job["status"], job["error"] = "failed", str(error)
        job["seconds"] = job.get("seconds", 0) + time.perf_counter() - started
        write_json(path, manifest)
        make_report(output, manifest)
        print(f"{job['name']}: {job['status']}. Report: {output / 'report.html'}", flush=True)
        if interrupted:
            return 130
    make_report(output, manifest)
    return 0 if all(j["status"] == "complete" for j in manifest["jobs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
