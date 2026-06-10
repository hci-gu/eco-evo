"""Spawn-vikt-strategier för initial biomass-fördelning.

Kontrakt
========
Varje strategi är en ren funktion med signaturen::

    weights(grid_size: tuple[int, int],
            params:    dict,
            rng:       np.random.Generator,
            context:   dict | None = None) -> np.ndarray  # shape (H, W)

Returnen är **en icke-negativ vikt-array som summerar till 1.0** över alla
celler där `allowed_mask` (om given via `context['allowed_mask']`) är truthy.
Cell-värdet tolkas som "andel av total-biomassa som hamnar här" *innan*
per-cell-golvet `10·min_split` appliceras.

Designprinciper
---------------
- Strategier producerar *vikter*, inte ton. Total-biomass-skalning + golv
  hanteras av en separat allokator (`distribute_with_floor` i nästa steg,
  som ersätter `_spawn_biomass_distribution`s cluster-loop).
- Reference-grid-invarians: `scale`-parametern (i celler) tolkas i
  referens-grid-koordinater. Vid faktisk grid (H_act, W_act) skalas den
  med `sqrt(biomass_scale)` så fysisk klumpstorlek bevaras. Detta
  appliceras i `_apply_grid_scaling` innan strategi-anropet.
- Determinism: alla strategier tar en `rng` (numpy Generator). Samma
  `spawn_seed` + samma params + samma grid → identisk karta.
- Stoikiometri: vikterna normaliseras alltid till summan 1 i slutet av
  varje strategi, så att allokatorn kan göra `weights * total_b` utan att
  bry sig om absoluta nivåer.

Strategier (skiss)
------------------
- ``uniform``      Likformig fördelning över eligible cells (= dagens beteende
                   när min_per_cell == 0). Default för bakåtkompatibilitet.
- ``perlin``       fBm-Perlin noise. Sammanhängande fält av variabel storlek.
                   Params: scale, octaves, persistence, lacunarity,
                   threshold (cell-vikt = max(0, n - threshold)).
- ``colony``       N kolonier vid valda celler, gaussian-utsmetning kring
                   varje. Avsedd för seabirds/seals/porpoises där en
                   handfull tätt sammanhängande celler är realistiskt.
                   Params: n_colonies, sigma_cells, anchor (free | coast
                   | open_water; coast/open_water kräver depth_map i context).
- ``env_driven``   Vikt = linjär/exp-kombination av referens-fält i context.
                   Params: refs=[(name, weight, transform)], där transform
                   är 'linear' | 'exp' | 'invert' | 'gauss_smooth'.
                   Exempel: phytoplankton ~ -depth, zooplankton ~ phytoplankton
                   utjämnad gaussian.

Allt utom ``uniform`` är NotImplementedError i denna skiss — implementeras
i nästa steg när vi enats om API:t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# Spec-objekt: per-FG konfiguration som projektfilen/fg_library lagrar.
# ---------------------------------------------------------------------------

@dataclass
class StrategySpec:
    """Per-FG spawn-spec, lagras i fg_library.yaml som ``spawn:`` block.

    YAML-exempel::

        spawn:
          mode: perlin           # uniform | perlin | colony | env_driven
          scale: 12              # cells (i ref-grid-koord)
          octaves: 4
          persistence: 0.5
          lacunarity: 2.0
          threshold: 0.0
          seed: null             # null → ärver project_metadata.spawn_seed

    För ``colony``::

        spawn:
          mode: colony
          n_colonies: 3
          sigma_cells: 2.0
          anchor: coast          # free | coast | open_water

    För ``env_driven``::

        spawn:
          mode: env_driven
          refs:
            - {name: depth,         weight: -1.0, transform: linear}
            - {name: phytoplankton, weight:  0.8, transform: gauss_smooth, sigma: 3}
          floor: 0.0              # clamp negativ vikt
    """
    mode: str = "uniform"
    params: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None  # None → use project spawn_seed

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "StrategySpec":
        if not d:
            return cls()
        mode = str(d.get("mode", "uniform"))
        seed = d.get("seed", None)
        # Allt utöver mode/seed läggs in i params.
        params = {k: v for k, v in d.items() if k not in ("mode", "seed")}
        return cls(mode=mode, params=params,
                   seed=int(seed) if seed is not None else None)


# ---------------------------------------------------------------------------
# Registry — gör det enkelt att lägga till strategier utan att röra dispatch.
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {}


def register_strategy(name: str):
    """Decorator för att registrera en strategi-funktion under `name`."""
    def deco(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        _REGISTRY[name] = fn
        return fn
    return deco


def get_strategy(name: str) -> Callable[..., np.ndarray]:
    """Slå upp en strategi; faller tillbaka på ``uniform`` om okänt namn."""
    if name not in _REGISTRY:
        # Defensivt: okänd strategi → uniform + (i nästa steg) en log-varning.
        return _REGISTRY["uniform"]
    return _REGISTRY[name]


# ---------------------------------------------------------------------------
# Reference-grid-invarians: skala längdparametrar med sqrt(biomass_scale).
# ---------------------------------------------------------------------------

def _apply_grid_scaling(params: dict, context: Optional[dict]) -> dict:
    """Returnera en kopia av params där `scale`/`sigma_cells` skalats med
    sqrt(biomass_scale) så fysisk klumpstorlek är invariant mot grid-upplösning.

    `biomass_scale = (H_act·W_act)/(Rx·Ry)` förväntas i `context['biomass_scale']`.
    """
    if not context:
        return dict(params)
    scale_factor = float(context.get("biomass_scale", 1.0)) ** 0.5
    out = dict(params)
    for key in ("scale", "sigma_cells"):
        if key in out and out[key] is not None:
            try:
                out[key] = max(1e-6, float(out[key]) * scale_factor)
            except (TypeError, ValueError):
                pass
    return out


# ---------------------------------------------------------------------------
# Gemensam efterbehandling: applicera allowed_mask och normalisera till sum 1.
# ---------------------------------------------------------------------------

def _finalize(weights: np.ndarray,
              context: Optional[dict]) -> np.ndarray:
    """Maska bort otillåtna celler, klampa negativt, normalisera till sum=1."""
    w = np.asarray(weights, dtype=np.float64)
    if context and "allowed_mask" in context and context["allowed_mask"] is not None:
        mask = np.asarray(context["allowed_mask"]).reshape(w.shape).astype(bool)
        w = np.where(mask, w, 0.0)
    # Negativa vikter (kan uppstå i env_driven) klampas till 0.
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0.0:
        # Fall tillbaka på uniform över allowed-mask (eller hela griden) så
        # biomassa aldrig tappas vid en patologisk vikt-konfiguration.
        if context and context.get("allowed_mask") is not None:
            mask = np.asarray(context["allowed_mask"]).reshape(w.shape).astype(bool)
            n = int(mask.sum())
            if n > 0:
                w = mask.astype(np.float64) / n
                return w
        w = np.ones_like(w) / w.size
        return w
    return w / s


# ---------------------------------------------------------------------------
# Strategi: uniform (default, bakåtkompatibelt med nuvarande beteende)
# ---------------------------------------------------------------------------

@register_strategy("uniform")
def weights_uniform(grid_size: Tuple[int, int],
                    params: dict,
                    rng: np.random.Generator,
                    context: Optional[dict] = None) -> np.ndarray:
    """Likformig Dirichlet-stil-fördelning över eligible cells.

    Producerar samma kvalitativa output som `_spawn_biomass_distribution`s
    legacy-gren (när `min_per_cell <= 0`): per-cell-vikten är en oberoende
    Uniform(0,1)-draw, sedan normaliserad. Per-cell-värdena är *inte*
    identiska — men distributionen är.
    """
    H, W = grid_size
    draws = rng.random((H, W))
    return _finalize(draws, context)


# ---------------------------------------------------------------------------
# Strategi: perlin (fBm noise → sammanhängande fält)
# ---------------------------------------------------------------------------

@register_strategy("perlin")
def weights_perlin(grid_size: Tuple[int, int],
                   params: dict,
                   rng: np.random.Generator,
                   context: Optional[dict] = None) -> np.ndarray:
    """fBm-Perlin noise → vikter ∈ [0, 1] (efter threshold-subtraktion).

    Implementation-skiss (ej kopplad än):
      1. Sampla N=octaves lager Perlin/Simplex med våglängd
         scale_i = scale * lacunarity**(-i), amplitud
         a_i = persistence**i, summera.
      2. Normalisera till [0, 1] (per-realisering).
      3. Subtrahera `threshold` och klampa till 0 → ger tomma områden.
      4. Returnera _finalize(...).

    Beroende: använd `opensimplex` (ren python, snabbt nog för <500x500) eller
    en numpy-baserad value-noise om vi vill undvika tredjepartsberoenden.
    """
    p = _apply_grid_scaling(params, context)
    scale = float(p.get("scale", 10.0))
    octaves = int(p.get("octaves", 4))
    persistence = float(p.get("persistence", 0.5))
    lacunarity = float(p.get("lacunarity", 2.0))
    threshold = float(p.get("threshold", 0.0))
    # Använd rng för att seed:a noise-generatorn deterministiskt.
    noise_seed = int(rng.integers(0, 2**31 - 1))

    # TODO: byt ut placeholder mot riktig Perlin/Simplex-implementation.
    # Skiss-placeholder: smoothed gaussian random field via numpy FFT-filter,
    # vilket är likvärdigt med ett enkelt 1/f^alpha-spektrum (~ Perlin för
    # alpha ≈ 2*persistence).
    H, W = grid_size
    field = _gaussian_random_field(H, W, scale=scale, octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    seed=noise_seed)
    # Min-max-normalisera till [0,1].
    lo, hi = float(field.min()), float(field.max())
    if hi > lo:
        field = (field - lo) / (hi - lo)
    else:
        field = np.zeros_like(field)
    field = np.clip(field - threshold, 0.0, None)
    return _finalize(field, context)


def _gaussian_random_field(H: int, W: int, *,
                            scale: float, octaves: int,
                            persistence: float, lacunarity: float,
                            seed: int) -> np.ndarray:
    """Numpy-baserad fBm-approximation via summa av lågpassfiltrerat brus.

    Snabb, ren-numpy, ingen tredjepart. Inte sant Perlin/Simplex men ger
    visuellt likvärdiga sammanhängande fält för vårt syfte.
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((H, W), dtype=np.float64)
    amp = 1.0
    freq_scale = float(scale)
    for _ in range(max(1, octaves)):
        # Sampla vitt brus, lågpassfiltrera via gaussian-kärna i Fourier-rum.
        noise = rng.standard_normal((H, W))
        field += amp * _lowpass_gaussian(noise, sigma=max(1.0, freq_scale))
        amp *= persistence
        freq_scale /= lacunarity
    return field


def _lowpass_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Periodisk gaussian-smoothing via FFT (snabb, deps-fri)."""
    H, W = arr.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    # Gaussian-kärna i frekvensdomän: H(f) = exp(-2 π² σ² f²).
    kernel = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fy ** 2 + fx ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * kernel))


# ---------------------------------------------------------------------------
# Strategi: colony (N kolonier + gaussian-utsmetning)
# ---------------------------------------------------------------------------

@register_strategy("colony")
def weights_colony(grid_size: Tuple[int, int],
                   params: dict,
                   rng: np.random.Generator,
                   context: Optional[dict] = None) -> np.ndarray:
    """N kolonicentrum, gaussian-utsmetade. Avsedd för topp-predatorer.

    Adresserar frusen-topp-predator-fyndet: 12 t porpoises på 24 utspridda
    celler ger ingen rörelse-frihet; samma 12 t på 1 koloni à 5–10 sammanhängande
    celler ger en meningsfull lokal patch som policy `move` kan agera på.

    Params:
        n_colonies (int):       antal kolonicentrum (default 3).
        sigma_cells (float):    gaussian-spridning per koloni i celler (default 2.0).
        anchor (str):           'free' | 'coast' | 'open_water'. Default 'free'.
                                NOTE: djup-baserade ankare ('coast'/'open_water')
                                accepteras som parameter men påverkar inte
                                centrum-valet just nu — depth_map konsumeras
                                inte. Implementeras i ett senare steg när en
                                officiell depth-karta finns att referera till.
        amplitude_mode (str):   'uniform' (default, legacy) | 'jitter'.
                                I 'jitter'-läge får varje koloni en
                                oberoende centrum-amplitud Uniform(
                                amplitude_min, amplitude_max) och kolonier
                                kombineras med max (ej summa). Fältet
                                returneras *utan* sum-normalisering så
                                amplituderna bevaras i [0, amplitude_max].
                                Avsett för impacts (ljudkällor m.m.) där
                                varje källa har en egen styrka och summa-
                                bevarande inte är meningsfullt.
        amplitude_min (float):  Undre gräns för per-centrum-amplitud i
                                'jitter'-läge, relativt vmax (default 0.0).
        amplitude_max (float):  Övre gräns för per-centrum-amplitud i
                                'jitter'-läge, relativt vmax (default 1.0).
    """
    p = _apply_grid_scaling(params, context)
    H, W = grid_size
    n = int(p.get("n_colonies", 3))
    sigma = float(p.get("sigma_cells", 2.0))
    anchor = str(p.get("anchor", "free"))
    amplitude_mode = str(p.get("amplitude_mode", "uniform")).lower()
    amp_min = float(p.get("amplitude_min", 0.0))
    amp_max = float(p.get("amplitude_max", 1.0))
    # Sanity: clamp to [0,1] and ensure min<=max.
    amp_min = min(max(amp_min, 0.0), 1.0)
    amp_max = min(max(amp_max, 0.0), 1.0)
    if amp_max < amp_min:
        amp_min, amp_max = amp_max, amp_min

    # Välj kolonicentrum från eligible-pool (eller hela griden).
    centers = _pick_colony_centers(grid_size, n, anchor, context, rng)

    yy, xx = np.indices((H, W))
    inv_two_sigma2 = 1.0 / (2.0 * max(1e-6, sigma) ** 2)
    if amplitude_mode == "jitter":
        # Per-centrum-amplitud + max-kombination. Returnera fältet utan
        # _finalize-summa-normalisering (men respektera allowed_mask).
        field = np.zeros((H, W), dtype=np.float64)
        for (cy, cx) in centers:
            if amp_max > amp_min:
                amp = float(rng.uniform(amp_min, amp_max))
            else:
                amp = float(amp_min)
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            bulge = amp * np.exp(-d2 * inv_two_sigma2)
            field = np.maximum(field, bulge)
        if context and "allowed_mask" in context and context["allowed_mask"] is not None:
            mask = np.asarray(context["allowed_mask"]).reshape(field.shape).astype(bool)
            field = np.where(mask, field, 0.0)
        # Klampa numeriskt brus.
        return np.clip(field, 0.0, 1.0)
    # Legacy uniform-amplitud-läge: alla centrum bidrar med amplitud 1.0,
    # summa-kombination, sum-normalisering (biomassa-kontrakt).
    field = np.zeros((H, W), dtype=np.float64)
    for (cy, cx) in centers:
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        field += np.exp(-d2 * inv_two_sigma2)
    return _finalize(field, context)


def _pick_colony_centers(grid_size: Tuple[int, int], n: int, anchor: str,
                          context: Optional[dict],
                          rng: np.random.Generator) -> list:
    H, W = grid_size
    # Bygg eligibility-mask för centrum-val.
    if context and context.get("allowed_mask") is not None:
        base = np.asarray(context["allowed_mask"]).reshape(H, W).astype(bool)
    else:
        base = np.ones((H, W), dtype=bool)
    # NOTE: anchor='coast'/'open_water' är reserverade nyckelord men har
    # ingen effekt just nu. Djup-baserad filtrering är medvetet bortkopplad
    # tills en officiell depth-karta finns att referera till. Parametern
    # accepteras utan att påverka centrum-valet.
    _ = anchor  # explicit no-op
    idx = np.flatnonzero(base)
    if idx.size == 0:
        idx = np.arange(H * W)
    pick = rng.choice(idx, size=min(n, idx.size), replace=False)
    return [(int(p // W), int(p % W)) for p in pick]


# ---------------------------------------------------------------------------
# Strategi: env_driven (vikter från referens-fält i context)
# ---------------------------------------------------------------------------

@register_strategy("env_driven")
def weights_env_driven(grid_size: Tuple[int, int],
                       params: dict,
                       rng: np.random.Generator,
                       context: Optional[dict] = None) -> np.ndarray:
    """Vikt = linjär kombination av transformerade referens-fält.

    Params:
        refs: list[dict] med nycklarna
              name      : namn på fält i context['env_fields'][name]
              weight    : skalär (signed; negativa "invertera")
              transform : 'linear' | 'exp' | 'invert' | 'gauss_smooth'
              sigma     : (endast för gauss_smooth) kärnstorlek i celler
        floor: float, clamp under detta värde till 0 (default 0.0).
        noise_amp: float, lägg till en svag Perlin-overlay (default 0.0).

    NOTE: 'depth' får anges som ref-name (för framtida bruk) men kommer
    att ignoreras tills en officiell depth-karta exponeras via
    context['env_fields']['depth']. Just nu filtreras 'depth'-refs bort
    så djup inte påverkar vikten — i linje med policyn "depth som
    parameter men ingen effekt".
    """
    H, W = grid_size
    refs = params.get("refs", []) or []
    # Filtrera bort depth-refs tills djup-kartan är officiellt inkopplad.
    refs = [r for r in refs if str(r.get("name", "")).lower() != "depth"]
    fields = (context or {}).get("env_fields", {}) if context else {}
    out = np.zeros((H, W), dtype=np.float64)
    for ref in refs:
        name = ref.get("name")
        if name not in fields:
            continue
        f = np.asarray(fields[name], dtype=np.float64).reshape(H, W)
        weight = float(ref.get("weight", 1.0))
        transform = str(ref.get("transform", "linear"))
        if transform == "exp":
            f = np.exp(f - f.max())  # numerisk stabilitet
        elif transform == "invert":
            # Vänd: hög ref → låg vikt. Normalisera mot max.
            mx = float(f.max())
            if mx > 0:
                f = 1.0 - f / mx
            else:
                f = np.ones_like(f)
        elif transform == "gauss_smooth":
            sigma = float(ref.get("sigma", 2.0))
            f = _lowpass_gaussian(f, sigma=max(1.0, sigma))
        # 'linear' = identity.
        out += weight * f
    # Valfri brus-overlay för stokasticitet.
    noise_amp = float(params.get("noise_amp", 0.0))
    if noise_amp > 0:
        out = out + noise_amp * _lowpass_gaussian(
            rng.standard_normal((H, W)),
            sigma=float(params.get("noise_scale", 5.0)),
        )
    floor = float(params.get("floor", 0.0))
    out = np.clip(out - floor, 0.0, None)
    return _finalize(out, context)


# ---------------------------------------------------------------------------
# Top-level dispatch — enda inkopplings-punkten från config_loader.
# ---------------------------------------------------------------------------

def make_weights(spec: StrategySpec,
                 grid_size: Tuple[int, int],
                 project_seed: Optional[int],
                 context: Optional[dict] = None) -> np.ndarray:
    """Bygg vikt-arrayen för en FG enligt dess `StrategySpec`.

    `project_seed` används om `spec.seed` är None, så hela projektet kan ha
    en gemensam `spawn_seed` separat från ARS-träningens `--seed`.
    """
    seed = spec.seed if spec.seed is not None else (project_seed or 0)
    rng = np.random.default_rng(seed)
    fn = get_strategy(spec.mode)
    return fn(grid_size, spec.params, rng, context)
