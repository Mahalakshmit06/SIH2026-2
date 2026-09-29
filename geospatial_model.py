"""
MnBrain - Module 1
3D Subsurface Grade Modeling / Geospatial AI
================================================

Implementation of an Ordinary-Kriging-style spatial interpolation
workflow that produces a 3D block model of manganese (Mn) and iron (Fe)
grades for a selected pit.

All geological values are SYNTHETIC DATA generated from a seeded,
spatially correlated random field. They are intended for demonstration
only and do NOT represent real MOIL measurements.

Main entry point:
    get_kriged_block_model(pit_name, depth_m, confidence_cutoff) -> pandas.DataFrame

Method
------
1. A correlated 3D random field is generated on a grid (smoothed noise)
   plus a pit-specific deterministic trend.
2. Fictitious borehole samples are drawn from that field.
3. A spherical variogram (nugget / sill / range) models spatial continuity.
4. Ordinary Kriging estimates Mn and Fe grades at every block centre.
5. Prediction uncertainty is reported as a relative kriging variance and
   used to assign a UNFC 111 screening status.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

MAX_MODEL_DEPTH = 100.0        # metres
MIN_MODEL_DEPTH = 10.0         # metres
FIELD_SPACING = 20.0           # metres between correlated-field nodes
NOISE_SIGMA = (3.0, 3.0, 2.0)  # smoothing sigma of the random field

MIN_MN = 20.0                  # synthetic Mn range (display / clipping)
MAX_MN = 55.0
MIN_FE = 5.0
MAX_FE = 35.0


def list_pits() -> tuple[str, ...]:
    """Names of the pits available in the simulation."""
    return tuple(PIT_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Pit configuration (all values synthetic)
# ---------------------------------------------------------------------------

PIT_CONFIGS = {
    "Dongri Buzurg": {
        "code": "DB",
        "seed": 101,
        "extent_x": 400.0,
        "extent_y": 400.0,
        "nugget": 0.3,
        "sill": 35.0,               # partial sill C
        "range_m": 1000.0,          # variogram range a (synthetic)
        "anisotropy_z": 2.0,        # vertical lag multiplier (m of xy per 1 m z)
        "mn_center": 38.0,
        "mn_amp": 10.0,
        "fe_center": 17.0,
        "fe_amp": 7.0,
        "kriging_neighbors": 30,
        "block_cell": 25.0,
        "layer_thickness": 10.0,
        "n_holes": 169,
        "hole_jitter": 8.0,
    },
    "Balaghat": {
        "code": "BG",
        "seed": 207,
        "extent_x": 400.0,
        "extent_y": 400.0,
        "nugget": 0.3,
        "sill": 32.0,
        "range_m": 1100.0,
        "anisotropy_z": 2.0,
        "mn_center": 34.0,
        "mn_amp": 8.0,
        "fe_center": 20.0,
        "fe_amp": 6.0,
        "kriging_neighbors": 30,
        "block_cell": 25.0,
        "layer_thickness": 10.0,
        "n_holes": 169,
        "hole_jitter": 8.0,
    },
    "Generic Pit": {
        "code": "GP",
        "seed": 531,
        "extent_x": 400.0,
        "extent_y": 400.0,
        "nugget": 0.3,
        "sill": 38.0,
        "range_m": 950.0,
        "anisotropy_z": 2.0,
        "mn_center": 36.0,
        "mn_amp": 9.0,
        "fe_center": 18.0,
        "fe_amp": 6.5,
        "kriging_neighbors": 30,
        "block_cell": 25.0,
        "layer_thickness": 10.0,
        "n_holes": 169,
        "hole_jitter": 8.0,
    },
}

# Sample depths (m) at which every fictitious borehole is assayed,
# aligned with the 10 m block layers so each block layer is well supported.
SAMPLE_DEPTHS = tuple(5.0 + 10.0 * i for i in range(10))


def _validate_inputs(pit_name: str, depth_m: float, confidence_cutoff: float) -> dict:
    """Validate API inputs and return the pit configuration dictionary."""
    if not isinstance(pit_name, str) or pit_name not in PIT_CONFIGS:
        valid = ", ".join(list_pits())
        raise ValueError(f"Unknown pit '{pit_name}'. Valid pits: {valid}.")

    try:
        depth = float(depth_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth_m must be a numeric value in metres.") from exc
    if not MIN_MODEL_DEPTH <= depth <= MAX_MODEL_DEPTH:
        raise ValueError(
            f"depth_m must be between {MIN_MODEL_DEPTH:g} m and "
            f"{MAX_MODEL_DEPTH:g} m (received {depth_m})."
        )

    try:
        cutoff = float(confidence_cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_cutoff must be a numeric percentage.") from exc
    if not 0.0 < cutoff <= 100.0:
        raise ValueError(
            "confidence_cutoff must be greater than 0 and at most 100 "
            f"(received {confidence_cutoff})."
        )

    return PIT_CONFIGS[pit_name]


# ---------------------------------------------------------------------------
# Spherical variogram
# ---------------------------------------------------------------------------

def spherical_variogram(h: np.ndarray | float, nugget: float,
                        partial_sill: float, range_m: float) -> np.ndarray:
    """Spherical variogram model.

    gamma(h) = C0 + C * [1.5(h/a) - 0.5(h/a)^3]   for 0 < h <= a
    gamma(h) = C0 + C                             for h > a
    gamma(0) = 0

    Parameters
    ----------
    h : separation distance(s)
    nugget : C0
    partial_sill : C
    range_m : a
    """
    h = np.asarray(h, dtype=float)
    gamma = np.zeros_like(h)
    inside = (h > 0.0) & (h <= range_m)
    outside = h > range_m
    r = h[inside] / range_m
    gamma[inside] = nugget + partial_sill * (1.5 * r - 0.5 * r ** 3)
    gamma[outside] = nugget + partial_sill
    if gamma.ndim == 0:
        return float(gamma)
    return gamma


def get_variogram_parameters(pit_name: str) -> dict:
    """Variogram parameters used for a given pit (for display purposes)."""
    cfg = PIT_CONFIGS[pit_name]
    return {
        "nugget": cfg["nugget"],
        "partial_sill": cfg["sill"],
        "sill_total": cfg["nugget"] + cfg["sill"],
        "range_m": cfg["range_m"],
        "anisotropy_z": cfg["anisotropy_z"],
    }


def get_demonstration_samples(pit_name: str) -> pd.DataFrame:
    """Return the synthetic borehole sample frame for a demonstration pit.

    Output uses the standard ingestion schema so it can be routed through
    the same validation and kriging pipeline as an uploaded dataset:

        Borehole_ID, X, Y, Depth, Mn_Grade, Fe_Grade

    Deterministic per pit (fixed seed).
    """
    if not isinstance(pit_name, str) or pit_name not in PIT_CONFIGS:
        valid = ", ".join(list_pits())
        raise ValueError(f"Unknown pit '{pit_name}'. Valid pits: {valid}.")
    cfg = PIT_CONFIGS[pit_name]
    cache = _pit_cache(pit_name)
    n_depths = len(SAMPLE_DEPTHS)
    hole_idx = np.arange(len(cache["sx"])) // n_depths
    hole_ids = [f"{cfg['code']}-H{h:03d}" for h in range(int(hole_idx.max()) + 1)]
    ids = [hole_ids[h] for h in hole_idx.tolist()]
    return pd.DataFrame({
        "Borehole_ID": ids,
        "X": np.round(cache["sx"], 2),
        "Y": np.round(cache["sy"], 2),
        "Depth": np.round(cache["sz"], 1),
        "Mn_Grade": np.round(cache["mn_sample"], 2),
        "Fe_Grade": np.round(cache["fe_sample"], 2),
    })


# ---------------------------------------------------------------------------
# Synthetic geological field + borehole samples
# ---------------------------------------------------------------------------

def _trend_values(pit_name: str, x: np.ndarray, y: np.ndarray,
                  z: np.ndarray) -> np.ndarray:
    """Deterministic pit-specific spatial trend in [-1, 1]."""
    cfg = PIT_CONFIGS[pit_name]
    e = cfg["extent_x"]
    xn, yn = x / e, y / e
    zn = z / MAX_MODEL_DEPTH
    if pit_name == "Dongri Buzurg":
        return (0.55 * np.sin(np.pi * xn) * np.cos(np.pi * yn)
                + 0.25 * (1.0 - zn) + 0.10 * (yn - 0.5))
    if pit_name == "Balaghat":
        return (0.45 * np.cos(np.pi * (xn + yn)) + 0.20 * zn
                + 0.15 * np.sin(2 * np.pi * xn))
    # Generic Pit
    return (0.50 * np.sin(np.pi * (xn + 0.5 * yn)) * (1.0 - 0.5 * zn)
            + 0.15 * (0.5 - xn))


def _build_pit_cache(pit_name: str) -> dict:
    """Generate (once per pit) sample co-ords and correlated grade values."""
    cfg = PIT_CONFIGS[pit_name]
    rng = np.random.default_rng(cfg["seed"])

    e = cfg["extent_x"]
    nx = int(e / FIELD_SPACING) + 1
    ny = int(e / FIELD_SPACING) + 1
    nz = int(MAX_MODEL_DEPTH / FIELD_SPACING) + 1

    xs = np.linspace(0.0, e, nx)
    ys = np.linspace(0.0, e, ny)
    zs = np.linspace(0.0, MAX_MODEL_DEPTH, nz)
    xv, yv, zv = np.meshgrid(xs, ys, zs, indexing="ij")

    trend = _trend_values(pit_name, xv, yv, zv)

    # Correlated random fields: smoothed normal noise.
    noise_mn = gaussian_filter(rng.standard_normal((nx, ny, nz)),
                               sigma=NOISE_SIGMA, mode="reflect")
    noise_fe = gaussian_filter(rng.standard_normal((nx, ny, nz)),
                               sigma=NOISE_SIGMA, mode="reflect")

    def norm(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    mn_grid = np.clip(cfg["mn_center"]
                      + cfg["mn_amp"] * trend
                      + 4.5 * norm(noise_mn),
                      MIN_MN, MAX_MN)
    fe_grid = np.clip(cfg["fe_center"]
                      + cfg["fe_amp"] * (0.5 * (1.0 - trend))
                      + 3.2 * norm(noise_fe)
                      - 0.35 * (mn_grid - cfg["mn_center"]),
                      MIN_FE, MAX_FE)

    # Fictitious boreholes arranged on a jittered grid.
    n_side = int(np.ceil(np.sqrt(cfg["n_holes"])))
    holes = []
    center = (e / n_side) * 0.5
    grid_coords = [center + i * (e / n_side) for i in range(n_side)]
    for gx in grid_coords:
        for gy in grid_coords:
            jx = gx + rng.uniform(-cfg["hole_jitter"], cfg["hole_jitter"])
            jy = gy + rng.uniform(-cfg["hole_jitter"], cfg["hole_jitter"])
            holes.append((float(np.clip(jx, 2.0, e - 2.0)),
                          float(np.clip(jy, 2.0, e - 2.0))))

    sx, sy, sz = [], [], []
    for (hx, hy) in holes:
        for dz in SAMPLE_DEPTHS:
            sx.append(hx)
            sy.append(hy)
            sz.append(dz)
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    sz = np.asarray(sz, dtype=float)

    # Sample grades via trilinear interpolation of the field.
    ix = sx / FIELD_SPACING
    iy = sy / FIELD_SPACING
    iz = sz / FIELD_SPACING
    coords = np.vstack((ix, iy, iz))

    mn_sample = map_coordinates(mn_grid, coords, order=1, mode="nearest")
    fe_sample = map_coordinates(fe_grid, coords, order=1, mode="nearest")

    # Small assay noise (seeded), then re-clip.
    mn_sample = np.clip(mn_sample + rng.normal(0.0, 0.5, mn_sample.shape),
                        MIN_MN, MAX_MN)
    fe_sample = np.clip(fe_sample + rng.normal(0.0, 0.4, fe_sample.shape),
                        MIN_FE, MAX_FE)

    # Anisotropic scaling for distance computations.
    z_factor = cfg["anisotropy_z"]
    sample_scaled = np.column_stack((sx, sy, sz * z_factor))
    tree = cKDTree(sample_scaled)

    return {
        "sx": sx, "sy": sy, "sz": sz,
        "mn_sample": mn_sample, "fe_sample": fe_sample,
        "sample_scaled": sample_scaled,
        "tree": tree,
    }


_PIT_CACHE: dict[str, dict] = {}


def _pit_cache(pit_name: str) -> dict:
    """Return the cached sample/field data for a pit (deterministic)."""
    if pit_name not in _PIT_CACHE:
        _PIT_CACHE[pit_name] = _build_pit_cache(pit_name)
    return _PIT_CACHE[pit_name]


# ---------------------------------------------------------------------------
# Ordinary Kriging engine
# ---------------------------------------------------------------------------

def _ordinary_kriging(block_xyz: np.ndarray, cache: dict,
                      values: np.ndarray, cfg: dict,
                      k: int) -> tuple[float, float]:
    """Krige a single block.

    Solves the Ordinary Kriging system:
        [ Gamma  ]     [ gamma0 ]
        [ 1 1..1 0 ] * [ 1 ]  =  rhs
    producing weights lambda with sum(lambda) = 1.

    Returns (estimate, kriging_variance).
    """
    z_factor = cfg["anisotropy_z"]
    target = np.array([block_xyz[0], block_xyz[1], block_xyz[2] * z_factor])

    neighbor_count = min(k, len(values))
    distances, neighbor_idx = cache["tree"].query(target, k=neighbor_count)

    if neighbor_count == 1:
        # Degenerate fallback: nearest sample only.
        lam = np.array([1.0])
        gamma0 = spherical_variogram(np.array(distances), cfg["nugget"],
                                     cfg["sill"], cfg["range_m"])
        est = values[neighbor_idx]
        var = max(float(gamma0), 0.0)
        return float(est), var

    d = distances.astype(float)
    gamma0 = spherical_variogram(d, cfg["nugget"], cfg["sill"], cfg["range_m"])

    # Pairwise lags between selected neighbours.
    pts = cache["sample_scaled"][neighbor_idx]
    diff = pts[:, None, :] - pts[None, :, :]
    lag = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    Gamma = spherical_variogram(lag, cfg["nugget"], cfg["sill"], cfg["range_m"])
    np.fill_diagonal(Gamma, 0.0)
    n = neighbor_count
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = Gamma
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    rhs = np.append(gamma0, 1.0)

    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]

    lam = sol[:n]
    mu = sol[n]
    estimate = float(lam @ values[neighbor_idx])
    variance = float(lam @ gamma0 + mu)
    return estimate, max(variance, 0.0)


# ---------------------------------------------------------------------------
# Shared kriging core (demonstration dataset OR user-uploaded samples)
# ---------------------------------------------------------------------------

def krige_samples(sx: np.ndarray, sy: np.ndarray, sz: np.ndarray,
                  mn_grades: np.ndarray, fe_grades: np.ndarray,
                  depth_m: float, confidence_cutoff: float,
                  nugget: float, partial_sill: float, range_m: float,
                  anisotropy_z: float,
                  extent: tuple[float, float, float, float] = (0.0, 400.0, 0.0, 400.0),
                  block_cell: float = 25.0, layer_thickness: float = 10.0,
                  kriging_neighbors: int = 30, code: str = "UPLD",
                  clip_mn: tuple[float, float] = (0.0, 100.0),
                  clip_fe: tuple[float, float] = (0.0, 100.0)) -> pd.DataFrame:
    """Krige a 3D block model from sample arrays.

    Shared core used by the demonstration pipeline and by validated,
    user-uploaded geological data. Produces the standard block-model schema:

        Block_ID, X, Y, Z, Mn_Grade, Fe_Grade, Kriging_Variance,
        Confidence, UNFC_Code

    Variogram parameters (nugget, partial sill, range, z-anisotropy) are
    explicit model parameters; they are not claimed to be auto-fitted.
    """
    try:
        depth = float(depth_m)
        cutoff = float(confidence_cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth_m and confidence_cutoff must be numeric.") from exc
    if not MIN_MODEL_DEPTH <= depth <= MAX_MODEL_DEPTH:
        raise ValueError(
            f"depth_m must be between {MIN_MODEL_DEPTH:g} m and "
            f"{MAX_MODEL_DEPTH:g} m (received {depth_m})."
        )
    if not 0.0 < cutoff <= 100.0:
        raise ValueError(
            "confidence_cutoff must be greater than 0 and at most 100 "
            f"(received {confidence_cutoff})."
        )

    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    sz = np.asarray(sz, dtype=float)
    mn = np.asarray(mn_grades, dtype=float)
    fe = np.asarray(fe_grades, dtype=float)

    if sx.size == 0:
        raise ValueError("No valid geological samples available for kriging.")
    if not (sx.size == sy.size == sz.size == mn.size == fe.size):
        raise ValueError("Geological sample arrays have inconsistent lengths.")

    layer = float(layer_thickness)
    cell = float(block_cell)
    n_layers = max(1, int(round(depth / layer)))

    x0, x1 = float(extent[0]), float(extent[1])
    y0, y1 = float(extent[2]), float(extent[3])
    x_edges = np.arange(x0, x1 + 1e-9, cell)
    y_edges = np.arange(y0, y1 + 1e-9, cell)
    if x_edges.size < 2:
        x_edges = np.array([x0 - cell / 2, x0 + cell / 2])
    if y_edges.size < 2:
        y_edges = np.array([y0 - cell / 2, y0 + cell / 2])

    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    zc = np.array([layer * (i + 0.5) for i in range(n_layers)])

    xx, yy, zz = np.meshgrid(xc, yc, zc, indexing="ij")
    n_horiz = xc.size * yc.size

    z_factor = float(anisotropy_z)
    sample_scaled = np.column_stack((sx, sy, sz * z_factor))
    tree = cKDTree(sample_scaled)
    kcfg = {
        "anisotropy_z": z_factor,
        "nugget": float(nugget),
        "sill": float(partial_sill),
        "range_m": float(range_m),
    }
    cache = {"sample_scaled": sample_scaled, "tree": tree}

    mn_blocks = np.full(xx.size, np.nan)
    fe_blocks = np.full(xx.size, np.nan)
    var_blocks = np.full(xx.size, np.nan)
    k = int(kriging_neighbors)

    for i in range(xx.size):
        block = (float(xx.ravel()[i]), float(yy.ravel()[i]), float(zz.ravel()[i]))
        mn_blocks[i], var_blocks[i] = _ordinary_kriging(block, cache, mn, kcfg, k)
        fe_blocks[i], _ = _ordinary_kriging(block, cache, fe, kcfg, k)

    mn_blocks = np.clip(mn_blocks, float(clip_mn[0]), float(clip_mn[1]))
    fe_blocks = np.clip(fe_blocks, float(clip_fe[0]), float(clip_fe[1]))

    total_sill = kcfg["nugget"] + kcfg["sill"]
    var_pct = 100.0 * var_blocks / total_sill
    confidence = np.clip(100.0 * (1.0 - var_pct / 100.0), 0.0, 100.0)

    block_ids = []
    for i in range(xx.size):
        idx, iz = divmod(i, n_horiz)
        block_ids.append(f"{code}_{int(zz.ravel()[i]):03d}_{idx:04d}")

    unfc = np.where(var_pct < cutoff, "111", "UNVERIFIED")

    return pd.DataFrame({
        "Block_ID": block_ids,
        "X": xx.ravel(),
        "Y": yy.ravel(),
        "Z": zz.ravel(),
        "Mn_Grade": np.round(mn_blocks, 2),
        "Fe_Grade": np.round(fe_blocks, 2),
        "Kriging_Variance": np.round(var_pct, 3),
        "Confidence": np.round(confidence, 2),
        "UNFC_Code": unfc,
    })


MAX_GRID_CELLS = 32


def krige_uploaded_samples(sample_df: pd.DataFrame, depth_m: float,
                           confidence_cutoff: float = 3.0,
                           nugget: float = 0.3, partial_sill: float = 32.0,
                           range_m: float = 1000.0, anisotropy_z: float = 2.0,
                           block_cell: float = 25.0,
                           layer_thickness: float = 10.0,
                           kriging_neighbors: int = 30) -> pd.DataFrame:
    """Krige a 3D block model from a validated geological sample frame.

    Expected columns: Borehole_ID, X, Y, Depth, Mn_Grade, Fe_Grade.

    The block grid is derived from the observed sample extent so uploaded
    datasets replace the demonstration dataset without changing the
    Ordinary-Kriging algorithm. Grid resolution is capped so the dashboard
    stays interactive for large extents.
    """
    required = ["Borehole_ID", "X", "Y", "Depth", "Mn_Grade", "Fe_Grade"]
    if not isinstance(sample_df, pd.DataFrame):
        raise ValueError("sample_df must be a pandas DataFrame.")
    missing = [c for c in required if c not in sample_df.columns]
    if missing:
        raise ValueError(
            "Geological sample data is missing required column(s): "
            + ", ".join(missing) + "."
        )

    work = sample_df[required].copy()
    for col in required[1:]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if work.empty:
        raise ValueError("Geological sample data is empty.")
    work = work.dropna(subset=required[1:])
    if work.empty:
        raise ValueError("No valid numeric geological samples available.")

    if (work["Depth"] < 0).any():
        raise ValueError("Geological samples contain negative depths.")
    if ((work["Mn_Grade"] < 0) | (work["Mn_Grade"] > 100)).any():
        raise ValueError("Geological samples contain Mn grades outside 0-100%.")
    if ((work["Fe_Grade"] < 0) | (work["Fe_Grade"] > 100)).any():
        raise ValueError("Geological samples contain Fe grades outside 0-100%.")

    sx = work["X"].to_numpy(dtype=float)
    sy = work["Y"].to_numpy(dtype=float)
    sz = work["Depth"].to_numpy(dtype=float)
    mn = work["Mn_Grade"].to_numpy(dtype=float)
    fe = work["Fe_Grade"].to_numpy(dtype=float)

    x0, x1 = float(sx.min()), float(sx.max())
    y0, y1 = float(sy.min()), float(sy.max())
    if not np.isfinite([x0, x1, y0, y1]).all():
        raise ValueError("Geological coordinates contain non-finite values.")

    # Degenerate extent (single point cluster): pad to a one-cell box.
    if x1 - x0 < 1e-9:
        x0, x1 = x0 - block_cell / 2, x0 + block_cell / 2
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - block_cell / 2, y0 + block_cell / 2

    # Cap the grid so BOTH horizontal dimensions stay within the configured
    # maximum. The cell size is chosen large enough that neither span
    # produces more than MAX_GRID_CELLS cells, preserving full coverage.
    span_x, span_y = x1 - x0, y1 - y0
    cell = max(float(block_cell), span_x / MAX_GRID_CELLS,
               span_y / MAX_GRID_CELLS)

    clip_mn = (max(0.0, float(mn.min())), min(100.0, float(mn.max())))
    clip_fe = (max(0.0, float(fe.min())), min(100.0, float(fe.max())))

    return krige_samples(
        sx, sy, sz, mn, fe, depth_m, confidence_cutoff,
        nugget=nugget, partial_sill=partial_sill, range_m=range_m,
        anisotropy_z=anisotropy_z, extent=(x0, x1, y0, y1),
        block_cell=cell, layer_thickness=layer_thickness,
        kriging_neighbors=kriging_neighbors, code="UPLD",
        clip_mn=clip_mn, clip_fe=clip_fe,
    )


def get_kriged_block_model(pit_name: str, depth_m: float,
                           confidence_cutoff: float = 3.0) -> pd.DataFrame:
    """Build the 3D Ordinary-Kriging block model for a pit.

    Parameters
    ----------
    pit_name : one of list_pits()
    depth_m : modelled depth between 10 and 100 m
    confidence_cutoff : relative prediction-variance threshold (%) used to
        classify blocks as UNFC 111 screened when
        kriging_variance (%) < confidence_cutoff.

    Returns
    -------
    pandas.DataFrame with columns:
        Block_ID, X, Y, Z, Mn_Grade, Fe_Grade, Kriging_Variance,
        Confidence, UNFC_Code
    """
    cfg = _validate_inputs(pit_name, depth_m, confidence_cutoff)
    cache = _pit_cache(pit_name)
    return krige_samples(
        cache["sx"], cache["sy"], cache["sz"],
        cache["mn_sample"], cache["fe_sample"],
        depth_m, confidence_cutoff,
        nugget=cfg["nugget"], partial_sill=cfg["sill"],
        range_m=cfg["range_m"], anisotropy_z=cfg["anisotropy_z"],
        extent=(0.0, cfg["extent_x"], 0.0, cfg["extent_y"]),
        block_cell=cfg["block_cell"], layer_thickness=cfg["layer_thickness"],
        kriging_neighbors=cfg["kriging_neighbors"], code=cfg["code"],
        clip_mn=(MIN_MN, MAX_MN), clip_fe=(MIN_FE, MAX_FE),
    )


if __name__ == "__main__":
    import sys
    pit = sys.argv[1] if len(sys.argv) > 1 else "Balaghat"
    model = get_kriged_block_model(pit, 50, 3.0)
    print(model.head())
    print(f"\nTotal blocks: {len(model):,} | "
          f"Mn mean: {model.Mn_Grade.mean():.2f}% | "
          f"Fe mean: {model.Fe_Grade.mean():.2f}% | "
          f"UNFC 111: {(model.UNFC_Code == '111').sum():,}")