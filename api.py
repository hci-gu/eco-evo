"""Pure inference HTTP API for Mareld.

Run locally:

    uv run api.py --host 127.0.0.1 --port 8000

This intentionally uses only the Python standard library for HTTP handling so
the API is runnable in the current project without adding a web framework.
The model execution path imports and reuses the existing inference helpers
(`build_env` and `load_policies_and_stats`).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import logging
import os
import re
import sys
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any

import numpy as np
from PIL import Image

# TODO: Replace cgi.FieldStorage before Python 3.13, either with
# python-multipart/FastAPI or a small maintained multipart parser.
warnings.filterwarnings(
    "ignore",
    message="'cgi' is deprecated.*",
    category=DeprecationWarning,
)
import cgi

from inference import (
    PolicyCheckpointMismatchError,
    build_env,
    load_policies_and_stats,
)
from lib.config.config_loader import load_project_config


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_PROJECT = os.path.join(ROOT_DIR, "mareld2.yaml")
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DEFAULT_DEBUG_MAP_DIR = os.path.join(ROOT_DIR, "debug_maps")

RUNS: dict[str, dict[str, Any]] = {}
LOGGER = logging.getLogger("mareld.api")

INTERNAL_TO_API_SPECIES = {
    "gadoids": "codfish",
    "pelagic_fish": "pelagicFish",
}
API_TO_INTERNAL_SPECIES = {
    **{v: k for k, v in INTERNAL_TO_API_SPECIES.items()},
    "phytoplankton": "phytoplankton",
    "zooplankton": "zooplankton",
    "porpoises": "porpoises",
    "seabirds": "seabirds",
    "seals": "seals",
    "benthic_community": "benthic_community",
}

UNSUPPORTED_TODOS = [
    {
        "feature": "asynchronous execution queue",
        "todo": "Add a durable job queue, background workers, progress callbacks, and persisted result storage.",
    },
    {
        "feature": "species_effort_multiplier pressure",
        "todo": "Add fishing/effort controls to EcosystemEnvironment so policy/action or mortality effects can be species-scoped.",
    },
    {
        "feature": "time-varying pressure windows",
        "todo": "Add pressure effects to the environment transition before accepting pressure inputs.",
    },
    {
        "feature": "texture/depth ecological semantics",
        "todo": "Map uploaded texture/depth rasters to model variables once the trained policies and FG config define how those layers are observed or consumed.",
    },
    {
        "feature": "age-group outputs",
        "todo": "Split FunctionalGroup state by age cohort and extend checkpoint/output schemas.",
    },
]


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    checkpoint_dir: str
    project_path: str
    files: list[str]


def _set_run_state(
    run_id: str,
    status: str,
    *,
    progress: float | None = None,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {"run_id": run_id, "status": status}
    if progress is not None:
        record["progress"] = float(max(0.0, min(1.0, progress)))
    if result is not None:
        record["result"] = result
    if error is not None:
        record["error"] = error
    RUNS[run_id] = record


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error_response(handler: BaseHTTPRequestHandler, err: ApiError) -> None:
    _json_response(
        handler,
        err.status,
        {"error": err.code, "message": err.message, "details": err.details},
    )


def _safe_path(path: str, base: str = ROOT_DIR) -> str:
    resolved = os.path.abspath(path if os.path.isabs(path) else os.path.join(ROOT_DIR, path))
    if os.path.commonpath([resolved, base]) != base:
        raise ApiError(422, "UNSUPPORTED_MODEL_INPUT", "Paths must stay inside the project directory.")
    return resolved


def _api_species(internal_ids: list[str]) -> list[str]:
    return [INTERNAL_TO_API_SPECIES.get(s, s) for s in internal_ids]


def _internal_species(api_ids: list[str]) -> list[str]:
    out = []
    for sid in api_ids:
        out.append(API_TO_INTERNAL_SPECIES.get(sid, sid))
    return out


def discover_models() -> list[ModelConfig]:
    if not os.path.isdir(DEFAULT_RESULTS_DIR):
        return []

    models: list[ModelConfig] = []
    for name in sorted(os.listdir(DEFAULT_RESULTS_DIR)):
        checkpoint_dir = os.path.join(DEFAULT_RESULTS_DIR, name)
        if not os.path.isdir(checkpoint_dir):
            continue
        files = sorted(f for f in os.listdir(checkpoint_dir) if re.fullmatch(r"policy_.+\.pth", f))
        if not files:
            continue
        models.append(
            ModelConfig(
                id=name,
                name=name.replace("_", " ").replace("-", " ").title(),
                checkpoint_dir=checkpoint_dir,
                project_path=DEFAULT_PROJECT,
                files=files,
            )
        )
    return models


def model_payload(model: ModelConfig) -> dict[str, Any]:
    species: list[str] = []
    try:
        fgs, _, _, _ = load_project_config(
            model.project_path, grid_size=(60, 60), mode="inference"
        )
        species = _api_species(list(fgs.keys()))
    except Exception:
        # Model discovery must stay non-fatal; POST /v1/runs performs the
        # strict project/checkpoint validation before executing.
        species = []

    return {
        "id": model.id,
        "name": model.name,
        "kind": "agent_set",
        "species": species,
        "supports_age_groups": False,
        "files": model.files,
        "description": "Mareld policy checkpoint directory discovered under results/.",
    }


def resolve_model(model_request: dict[str, Any]) -> ModelConfig:
    model_id = model_request.get("id")
    models = {m.id: m for m in discover_models()}
    if model_id not in models:
        raise ApiError(404, "MODEL_NOT_FOUND", f"Unknown model id: {model_id!r}")

    model = models[model_id]
    if model_request.get("model_path"):
        # TODO: Separate project path from checkpoint path in the request
        # schema. For now model_path is treated as a checkpoint-dir override.
        checkpoint_dir = _safe_path(str(model_request["model_path"]))
        if not os.path.isdir(checkpoint_dir):
            raise ApiError(404, "MODEL_NOT_FOUND", f"model_path does not exist: {checkpoint_dir}")
        model = ModelConfig(
            id=model.id,
            name=model.name,
            checkpoint_dir=checkpoint_dir,
            project_path=model.project_path,
            files=sorted(f for f in os.listdir(checkpoint_dir) if re.fullmatch(r"policy_.+\.pth", f)),
        )
    return model


def _read_png_field(field: cgi.FieldStorage, expected_w: int, expected_h: int, label: str) -> np.ndarray:
    if field is None or not getattr(field, "filename", None):
        raise ApiError(400, "MISSING_FILE", f"Missing required file field: {label}")
    data = field.file.read()
    if not data:
        raise ApiError(400, "INVALID_RASTER", f"{label} is empty.")
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ApiError(400, "INVALID_RASTER", f"{label} must be a readable PNG.") from exc
    if arr.shape != (expected_h, expected_w):
        raise ApiError(
            422,
            "UNSUPPORTED_MODEL_INPUT",
            f"{label} shape {arr.shape} does not match grid {(expected_h, expected_w)}.",
        )
    return arr


def _write_debug_png(path: str, arr: np.ndarray) -> None:
    data = np.asarray(arr, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, 1.0)
    img = Image.fromarray((data * 255.0 + 0.5).astype(np.uint8), mode="L")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def texture_to_water_mask(texture: np.ndarray) -> np.ndarray:
    """Return True for water cells. API contract: white=land, black=water."""
    water = np.asarray(texture, dtype=np.float32) < 0.5
    if not np.any(water):
        raise ApiError(
            422,
            "UNSUPPORTED_MODEL_INPUT",
            "texture marks every cell as land; at least one water cell is required.",
        )
    return water


def save_received_maps(
    run_id: str,
    texture: np.ndarray,
    depth: np.ndarray,
    masks_by_id: dict[str, np.ndarray],
    output_dir: str,
) -> None:
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("_") or "run"
    run_dir = os.path.join(output_dir, safe_run_id)
    _write_debug_png(os.path.join(run_dir, "texture.png"), texture)
    _write_debug_png(os.path.join(run_dir, "depth.png"), depth)
    for mask_id, mask in sorted(masks_by_id.items()):
        safe_mask_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", mask_id).strip("_") or "mask"
        _write_debug_png(os.path.join(run_dir, f"mask_{safe_mask_id}.png"), mask.astype(np.float32))
    LOGGER.info(
        "run %s saved received map PNGs to %s (%d mask%s)",
        run_id,
        run_dir,
        len(masks_by_id),
        "" if len(masks_by_id) == 1 else "s",
    )


def _form_file(form: cgi.FieldStorage, name: str) -> cgi.FieldStorage | None:
    item = form[name] if name in form else None
    if isinstance(item, list):
        return item[0] if item else None
    return item


def parse_multipart_request(handler: BaseHTTPRequestHandler) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.startswith("multipart/form-data"):
        raise ApiError(400, "INVALID_REQUEST", "Content-Type must be multipart/form-data.")

    form = cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ={
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": handler.headers.get("Content-Length", "0"),
        },
    )

    request_field = form["request"] if "request" in form else None
    if request_field is None:
        raise ApiError(400, "INVALID_REQUEST", "Missing required form field: request")
    try:
        request = json.loads(request_field.value)
    except Exception as exc:
        raise ApiError(400, "INVALID_JSON", "The request form field must contain valid JSON.") from exc

    validate_required_request_fields(request)
    grid = request["grid"]
    width = int(grid["width"])
    height = int(grid["height"])

    texture = _read_png_field(_form_file(form, "texture"), width, height, "texture")
    depth = _read_png_field(_form_file(form, "depth"), width, height, "depth")

    uploaded_masks = form.getlist("mask") if "mask" in form else []
    masks_by_filename: dict[str, np.ndarray] = {}
    for mask_field in uploaded_masks:
        if not getattr(mask_field, "filename", None):
            continue
        masks_by_filename[os.path.basename(mask_field.filename)] = _read_png_field(
            mask_field, width, height, f"mask:{mask_field.filename}"
        )

    masks_by_id: dict[str, np.ndarray] = {}
    for mask_meta in request.get("masks") or []:
        mid = mask_meta.get("id")
        filename = os.path.basename(str(mask_meta.get("file", "")))
        if not mid or not filename:
            raise ApiError(400, "INVALID_REQUEST", "Each mask entry needs id and file.")
        if filename not in masks_by_filename:
            raise ApiError(400, "MISSING_FILE", f"Missing uploaded mask file referenced by request: {filename}")
        masks_by_id[str(mid)] = masks_by_filename[filename] > 0.0

    return request, texture, depth, masks_by_id


def validate_required_request_fields(request: dict[str, Any]) -> None:
    if not isinstance(request, dict):
        raise ApiError(400, "INVALID_REQUEST", "request must be a JSON object.")
    for key in ("model", "grid", "time", "species", "output"):
        if key not in request:
            raise ApiError(400, "INVALID_REQUEST", f"Missing required request field: {key}")
    grid = request["grid"]
    try:
        width = int(grid["width"])
        height = int(grid["height"])
    except Exception as exc:
        raise ApiError(400, "INVALID_REQUEST", "grid.width and grid.height are required integers.") from exc
    if width < 3 or height < 3:
        raise ApiError(422, "UNSUPPORTED_MODEL_INPUT", "grid.width and grid.height must both be >= 3.")

    time_cfg = request["time"]
    try:
        max_steps = int(time_cfg["max_steps"])
        sample_every = int(time_cfg.get("sample_every", 1))
    except Exception as exc:
        raise ApiError(400, "INVALID_REQUEST", "time.max_steps and time.sample_every must be integers.") from exc
    if max_steps < 0:
        raise ApiError(422, "UNSUPPORTED_MODEL_INPUT", "time.max_steps must be >= 0.")
    if sample_every <= 0:
        raise ApiError(422, "UNSUPPORTED_MODEL_INPUT", "time.sample_every must be > 0.")

    if not isinstance(request["species"], list) or not request["species"]:
        raise ApiError(400, "INVALID_REQUEST", "species must be a non-empty array.")

    out = request["output"]
    if out.get("dtype", "float32") != "float32":
        raise ApiError(422, "UNSUPPORTED_MODEL_INPUT", "Only output.dtype=float32 is supported.")
    if out.get("tensor_order", "frame,row,column,species") != "frame,row,column,species":
        raise ApiError(422, "UNSUPPORTED_MODEL_INPUT", "Only tensor_order=frame,row,column,species is supported.")
    if int(out.get("replicates", 1)) < 1:
        raise ApiError(422, "UNSUPPORTED_MODEL_INPUT", "output.replicates must be >= 1.")


def validate_pressures(request: dict[str, Any], masks_by_id: dict[str, np.ndarray]) -> None:
    for pressure in request.get("pressures") or []:
        raise ApiError(
            422,
            "UNSUPPORTED_PRESSURE",
            f"Pressure inputs are disabled in the current inference model: {pressure.get('type')!r}.",
            {"todo": "Add pressure effects to the environment transition before accepting pressure inputs."},
        )


def apply_uploaded_inputs(env: Any, texture: np.ndarray, depth: np.ndarray, request: dict[str, Any], masks_by_id: dict[str, np.ndarray]) -> None:
    # TODO: The current policies are not trained against texture/depth channels.
    # They are stored on the grid for future model configs that consume them.
    env.grid.add_map("texture", texture)
    env.grid.add_map("depth", depth)



def _capture_tensor(env: Any, internal_species: list[str]) -> np.ndarray:
    layers = []
    for sid in internal_species:
        if sid not in env.fgs:
            raise ApiError(422, "UNSUPPORTED_SPECIES", f"Species is not available in this model: {sid}")
        data = np.asarray(env.fgs[sid].biomass, dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        layers.append(np.maximum(data, 0.0))
    return np.stack(layers, axis=-1).astype(np.float32, copy=False)


def _run_one_replicate(
    model: ModelConfig,
    request: dict[str, Any],
    texture: np.ndarray,
    depth: np.ndarray,
    masks_by_id: dict[str, np.ndarray],
    seed: int | None,
    run_id: str,
    replicate_index: int,
    replicate_count: int,
    progress_base: float,
    progress_span: float,
    visualize: bool,
) -> tuple[np.ndarray, list[int]]:
    height = int(request["grid"]["height"])
    width = int(request["grid"]["width"])
    time_cfg = request["time"]
    max_steps = int(time_cfg["max_steps"])
    sample_every = int(time_cfg.get("sample_every", 1))
    include_final = bool(time_cfg.get("include_final", True))
    internal_species = _internal_species(list(request["species"]))

    with contextlib.redirect_stdout(io.StringIO()):
        water_mask = texture_to_water_mask(texture)
        LOGGER.info(
            "run %s replicate %d/%d texture water_cells=%d land_cells=%d",
            run_id,
            replicate_index + 1,
            replicate_count,
            int(water_mask.sum()),
            int(water_mask.size - water_mask.sum()),
        )
        env = build_env(
            model.project_path,
            (height, width),
            seed=seed,
            verbose=False,
            apply_natural_mortality=False,
            allowed_mask=water_mask,
        )
        policies, obs_mean, obs_var = load_policies_and_stats(env, model.checkpoint_dir, verbose=False)

    apply_uploaded_inputs(env, texture, depth, request, masks_by_id)
    env.policies = dict(policies)
    env.obs_mean = obs_mean
    env.obs_var = obs_var
    env.rebuild_batched_weights()

    viz = None
    if visualize:
        try:
            from lib.viz import LiveVisualizer

            internal_species = _internal_species(list(request["species"]))
            ndm_ids = [
                fid for fid, fg in env.fgs.items()
                if not getattr(fg, "is_decision_maker", False)
            ]
            viz = LiveVisualizer(
                fg_ids=list(env.fgs.keys()),
                grid_shape=(height, width),
                mode="inference",
                title=f"Mareld API {run_id}",
                plot_fg_ids=internal_species,
                ndm_ids=ndm_ids or None,
            )
            LOGGER.info("run %s visualization enabled for replicate %d", run_id, replicate_index + 1)
        except Exception as exc:
            LOGGER.warning("run %s visualization could not start: %r", run_id, exc)
            viz = None

    visual_b0 = {
        fid: max(float(fg.biomass.sum()), 1e-12)
        for fid, fg in env.fgs.items()
    }

    frames: list[np.ndarray] = []
    steps: list[int] = []
    progress_interval = max(1, max_steps // 20) if max_steps else 1
    last_logged_pct = -1

    def sample(step: int) -> None:
        frames.append(_capture_tensor(env, internal_species))
        steps.append(step)

    def update_progress(step: int, *, force_log: bool = False) -> None:
        nonlocal last_logged_pct
        within = 1.0 if max_steps == 0 else step / max_steps
        overall = progress_base + progress_span * within
        _set_run_state(run_id, "running", progress=overall)
        pct = int(round(overall * 100))
        should_log = force_log or pct != last_logged_pct and (step == 0 or step == max_steps or step % progress_interval == 0)
        if should_log:
            last_logged_pct = pct
            LOGGER.info(
                "run %s progress %d%% replicate=%d/%d step=%d/%d",
                run_id,
                pct,
                replicate_index + 1,
                replicate_count,
                step,
                max_steps,
            )

    def update_visual(step: int) -> None:
        nonlocal viz
        if viz is None:
            return
        try:
            viz.update_biomass(env.fgs, tick=step, extra={"run": run_id})
            for fid, fg in env.fgs.items():
                total = float(fg.biomass.sum())
                viz.update_series("biomass", fid,
                                  100.0 * total / visual_b0[fid],
                                  step=step)
            if not viz.pump_events():
                LOGGER.info("run %s visualization window closed at step %d; inference continues", run_id, step)
                viz.close()
                viz = None
        except Exception as exc:
            LOGGER.warning("run %s visualization update failed at step %d: %r", run_id, step, exc)
            viz = None

    try:
        sample(0)
        update_progress(0, force_log=True)
        update_visual(0)
        for step in range(1, max_steps + 1):
            observation = env.get_observation()
            actions = env.policy_controller.forward(observation)
            env.step(actions)
            if step % sample_every == 0:
                sample(step)
            update_progress(step)
            update_visual(step)
        if include_final and (not steps or steps[-1] != max_steps):
            sample(max_steps)
    finally:
        if viz is not None:
            try:
                viz.close()
            except Exception:
                pass

    return np.stack(frames, axis=0).astype(np.float32, copy=False), steps


def _summary_from_replicates(
    replicate_tensors: list[np.ndarray],
    steps: list[int],
    api_species: list[str],
    normalization: str,
) -> dict[str, Any]:
    totals = np.stack([t.sum(axis=(1, 2)) for t in replicate_tensors], axis=0).astype(np.float64)
    if normalization == "relative_to_initial":
        baseline = totals[:, :1, :]
        totals = np.divide(totals, baseline, out=np.zeros_like(totals), where=baseline > 0)
    elif normalization not in ("absolute", "none", None):
        raise ApiError(422, "UNSUPPORTED_MODEL_INPUT", f"Unsupported summary_normalization: {normalization!r}")

    # totals shape: [replicate, frame, species]. Summary contract wants
    # [group/species, frame].
    mean = totals.mean(axis=0).T
    if totals.shape[0] > 1:
        sem = totals.std(axis=0, ddof=1).T / np.sqrt(totals.shape[0])
        # TODO: Use a real Student t critical value when scipy or a small
        # lookup table is available. 1.96 is close for larger replicate counts.
        half_width = 1.96 * sem
    else:
        half_width = np.zeros_like(mean)

    ci_low = mean - half_width
    ci_high = mean + half_width
    return {
        "run_count": int(totals.shape[0]),
        "confidence_level": 0.95,
        "ci_method": "normal-approximation" if totals.shape[0] > 1 else "single-run",
        "normalization": "relative_to_initial" if normalization == "relative_to_initial" else "absolute",
        "grouping": "functional_group",
        "steps": steps,
        "groups": api_species,
        "group_species": [[s] for s in api_species],
        "mean": mean.tolist(),
        "ci_low": ci_low.tolist(),
        "ci_high": ci_high.tolist(),
    }


def execute_run(
    request: dict[str, Any],
    texture: np.ndarray,
    depth: np.ndarray,
    masks_by_id: dict[str, np.ndarray],
    *,
    visualize: bool = False,
) -> dict[str, Any]:
    run_id = str(request.get("run_id") or f"run-{uuid.uuid4().hex[:12]}")
    if run_id in RUNS:
        raise ApiError(409, "RUN_CONFLICT", f"run_id already exists: {run_id}")

    _set_run_state(run_id, "running", progress=0.0)
    started = time.monotonic()
    try:
        validate_pressures(request, masks_by_id)
        model = resolve_model(request["model"])

        output = request["output"]
        replicates = int(output.get("replicates", 1))
        api_species = list(request["species"])
        base_seed = request.get("seed")
        base_seed = int(base_seed) if base_seed is not None else None
        grid = request["grid"]
        time_cfg = request["time"]
        LOGGER.info(
            "run %s starting model=%s grid=%sx%s max_steps=%s sample_every=%s species=%s replicates=%d visualize=%s",
            run_id,
            model.id,
            grid["width"],
            grid["height"],
            time_cfg["max_steps"],
            time_cfg.get("sample_every", 1),
            api_species,
            replicates,
            visualize,
        )

        replicate_tensors: list[np.ndarray] = []
        steps: list[int] | None = None
        for idx in range(replicates):
            seed = None if base_seed is None else base_seed + idx
            LOGGER.info("run %s replicate %d/%d starting seed=%s", run_id, idx + 1, replicates, seed)
            tensor, run_steps = _run_one_replicate(
                model,
                request,
                texture,
                depth,
                masks_by_id,
                seed=seed,
                run_id=run_id,
                replicate_index=idx,
                replicate_count=replicates,
                progress_base=idx / replicates,
                progress_span=1.0 / replicates,
                visualize=visualize and idx == 0,
            )
            if steps is None:
                steps = run_steps
            elif steps != run_steps:
                raise ApiError(500, "MODEL_FAILED", "Replicate sampling steps did not match.")
            replicate_tensors.append(tensor)
            LOGGER.info("run %s replicate %d/%d completed", run_id, idx + 1, replicates)

        assert steps is not None
        tensor = np.mean(np.stack(replicate_tensors, axis=0), axis=0).astype(np.float32)
        tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        tensor = np.maximum(tensor, 0.0).astype(np.float32, copy=False)

        byte_limit = 100 * 1024 * 1024
        if tensor.nbytes > byte_limit:
            raise ApiError(
                422,
                "UNSUPPORTED_MODEL_INPUT",
                f"Sampled tensor is {tensor.nbytes} bytes, above the 100 MB practical limit.",
            )

        result = {
            "run_id": run_id,
            "world_size": int(request["grid"]["width"]),
            "species": api_species,
            "sample_every": int(time_cfg.get("sample_every", 1)),
            "include_final": bool(time_cfg.get("include_final", True)),
            "tick_duration_days": time_cfg.get("tick_duration_days"),
            "start_date": time_cfg.get("start_date"),
            "end_date": time_cfg.get("end_date"),
            "dtype": "float32",
            "shape": list(tensor.shape),
            "steps": steps,
            # TODO: The trainer fitness function is not exposed for inference
            # tensors yet. This placeholder is a deterministic ecological score:
            # final total biomass in the returned tensor.
            "fitness": float(tensor[-1].sum()) if tensor.size else 0.0,
            "episode_length": int(time_cfg["max_steps"]),
            "end_reason": "completed",
            "biomass_b64": base64.b64encode(np.ascontiguousarray(tensor).tobytes()).decode("ascii"),
        }

        if output.get("include_summary", False):
            result["summary"] = _summary_from_replicates(
                replicate_tensors,
                steps,
                api_species,
                output.get("summary_normalization", "absolute"),
            )

        envelope = {"run_id": run_id, "status": "completed", "result": result}
        RUNS[run_id] = envelope
        LOGGER.info(
            "run %s completed duration=%.2fs shape=%s tensor_bytes=%d",
            run_id,
            time.monotonic() - started,
            result["shape"],
            tensor.nbytes,
        )
        return envelope
    except ApiError as err:
        _set_run_state(run_id, "failed", error={"code": err.code, "message": err.message})
        LOGGER.warning("run %s failed code=%s message=%s", run_id, err.code, err.message)
        raise
    except PolicyCheckpointMismatchError as err:
        _set_run_state(run_id, "failed", error={"code": "MODEL_CHECKPOINT_MISMATCH", "message": str(err)})
        LOGGER.warning("run %s failed checkpoint mismatch", run_id)
        raise
    except Exception as exc:
        _set_run_state(run_id, "failed", error={"code": "MODEL_FAILED", "message": str(exc)})
        LOGGER.exception("run %s failed with unexpected exception", run_id)
        raise


class MareldApiHandler(BaseHTTPRequestHandler):
    server_version = "MareldInferenceAPI/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                _json_response(self, 200, {"ok": True})
            elif self.path == "/v1/models":
                _json_response(self, 200, [model_payload(m) for m in discover_models()])
            elif self.path == "/v1/mock/unsupported":
                _json_response(self, 200, {"unsupported": UNSUPPORTED_TODOS})
            else:
                match = re.fullmatch(r"/v1/runs/([^/?#]+)", self.path)
                if match:
                    run_id = match.group(1)
                    if run_id not in RUNS:
                        raise ApiError(404, "RUN_NOT_FOUND", f"Unknown run_id: {run_id}")
                    _json_response(self, 200, RUNS[run_id])
                else:
                    raise ApiError(404, "NOT_FOUND", "Route not found.")
        except ApiError as err:
            _error_response(self, err)
        except Exception as exc:
            _error_response(
                self,
                ApiError(500, "SERVICE_FAILED", "Unexpected service failure.", {"trace": traceback.format_exc()}),
            )

    def do_POST(self) -> None:
        try:
            if self.path != "/v1/runs":
                raise ApiError(404, "NOT_FOUND", "Route not found.")
            request, texture, depth, masks_by_id = parse_multipart_request(self)
            if not request.get("run_id"):
                request["run_id"] = f"run-{uuid.uuid4().hex[:12]}"
            run_id = str(request["run_id"])
            if run_id in RUNS:
                raise ApiError(409, "RUN_CONFLICT", f"run_id already exists: {run_id}")
            if bool(getattr(self.server, "save_received_maps", False)):
                try:
                    save_received_maps(
                        run_id,
                        texture,
                        depth,
                        masks_by_id,
                        str(getattr(self.server, "debug_map_dir", DEFAULT_DEBUG_MAP_DIR)),
                    )
                except Exception as exc:
                    LOGGER.warning("run %s could not save received map PNGs: %r", run_id, exc)
            envelope = execute_run(
                request,
                texture,
                depth,
                masks_by_id,
                visualize=bool(getattr(self.server, "visualize_inference", False)),
            )
            _json_response(self, 200, envelope)
        except ApiError as err:
            _error_response(self, err)
        except PolicyCheckpointMismatchError as err:
            _error_response(self, ApiError(422, "MODEL_CHECKPOINT_MISMATCH", str(err)))
        except Exception:
            _error_response(
                self,
                ApiError(500, "MODEL_FAILED", "Unexpected model failure.", {"trace": traceback.format_exc()}),
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Mareld pure inference API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--visual-inference",
        "--visual",
        action="store_true",
        help="Open the live Mareld visualizer during inference runs for debugging. Only the first replicate is visualized.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Service log verbosity.",
    )
    parser.add_argument(
        "--save-received-maps",
        action="store_true",
        help="Save uploaded texture/depth/mask rasters as PNGs under --debug-map-dir for visual debugging.",
    )
    parser.add_argument(
        "--debug-map-dir",
        default=DEFAULT_DEBUG_MAP_DIR,
        help="Directory for --save-received-maps output.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Pygame/SDL on macOS requires window creation on the main thread.
    # ThreadingHTTPServer handles requests in worker threads, so debug
    # visualization uses the single-threaded HTTPServer instead.
    server_cls = HTTPServer if args.visual_inference else ThreadingHTTPServer
    server = server_cls((args.host, args.port), MareldApiHandler)
    server.visualize_inference = bool(args.visual_inference)
    server.save_received_maps = bool(args.save_received_maps)
    server.debug_map_dir = os.path.abspath(args.debug_map_dir)
    print(f"Mareld inference API listening on http://{args.host}:{args.port}")
    print(f"Run logging level: {args.log_level}")
    print(f"Debug visualization: {'on' if args.visual_inference else 'off'}")
    print(f"Save received maps: {'on' if args.save_received_maps else 'off'}")
    if args.save_received_maps:
        print(f"Received map directory: {server.debug_map_dir}")
    print(f"Request handling: {'single-threaded' if args.visual_inference else 'threaded'}")
    print("Raster tensor orientation: row-major north/top to south/bottom, columns west/left to east/right.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
