# Type stub for the `landtrendr` PyO3 extension module (src/python.rs).
# Maturin packages `<module_name>.pyi` from the project root into the wheel,
# giving IDEs and type checkers signatures for the native functions.

import numpy as np
from numpy.typing import NDArray

def pixel(
    values: NDArray[np.float32],
    years: NDArray[np.int32],
    max_segments: int = 6,
    spike_threshold: float = 0.9,
    recovery_threshold: float = 0.25,
    p_value_threshold: float = 0.05,
    best_model_proportion: float = 0.75,
    min_observations_needed: int = 6,
    vertex_count_overshoot: int = 3,
    prevent_one_year_recovery: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.uint8], float]:
    """Full per-pixel LandTrendr fit: returns (fitted, is_vertex, rmse).

    Defaults are the LT-GEE runParams. NaNs in `values` mark missing years.
    """

def pixel_debug(
    values: NDArray[np.float32],
    years: NDArray[np.int32],
    max_segments: int = 6,
    spike_threshold: float = 0.9,
    recovery_threshold: float = 0.25,
    p_value_threshold: float = 0.05,
    best_model_proportion: float = 0.75,
    min_observations_needed: int = 6,
    vertex_count_overshoot: int = 3,
    prevent_one_year_recovery: bool = True,
) -> tuple[NDArray[np.float32], list[int], list[int]]:
    """Vertex-selection debug tape: (despiked, candidate_vertex_indices,
    vetted_vertex_indices), for differential validation against LT-IDL."""

def raster_summary(
    data: NDArray[np.float32],
    years: NDArray[np.int32],
    max_segments: int = 6,
    spike_threshold: float = 0.9,
    recovery_threshold: float = 0.25,
    p_value_threshold: float = 0.05,
    best_model_proportion: float = 0.75,
    min_observations_needed: int = 6,
    vertex_count_overshoot: int = 3,
    prevent_one_year_recovery: bool = True,
) -> NDArray[np.float32]:
    """Raster-stack LandTrendr. `data` has shape (n_years, n_pixels) — the
    native layout of a loaded raster time series, e.g.
    stack.reshape(n_years, -1). Returns (4, n_pixels) summary bands
    [net_mag, year, rmse, peak_to_trough]."""

def ftvdiff(
    data: NDArray[np.float32],
    years: NDArray[np.int32],
    target_year: int,
    max_segments: int = 6,
    spike_threshold: float = 0.9,
    recovery_threshold: float = 0.25,
    p_value_threshold: float = 0.05,
    best_model_proportion: float = 0.75,
    min_observations_needed: int = 6,
    vertex_count_overshoot: int = 3,
    prevent_one_year_recovery: bool = True,
) -> NDArray[np.float32]:
    """Per-pixel fitted change *in* `target_year` (eMapR getLtFtvDiff):
    fitted[idx] - fitted[idx-1]. `data` has shape (n_years, n_pixels);
    returns n_pixels values, NaN where invalid."""

def loss_window(
    data: NDArray[np.float32],
    years: NDArray[np.int32],
    target_year: int,
    half_window: int = 1,
    max_segments: int = 6,
    spike_threshold: float = 0.9,
    recovery_threshold: float = 0.25,
    p_value_threshold: float = 0.05,
    best_model_proportion: float = 0.75,
    min_observations_needed: int = 6,
    vertex_count_overshoot: int = 3,
    prevent_one_year_recovery: bool = True,
) -> NDArray[np.float32]:
    """Windowed loss magnitude around `target_year`: sum of loss-direction
    fitted drops over [target_year - half_window, target_year + half_window].
    `data` has shape (n_years, n_pixels); returns n_pixels non-negative values,
    NaN where invalid. half_window=0 is the single-year loss."""

def segments(
    values: NDArray[np.float32],
    years: NDArray[np.int32],
    max_segments: int = 6,
    spike_threshold: float = 0.9,
    recovery_threshold: float = 0.25,
    p_value_threshold: float = 0.05,
    best_model_proportion: float = 0.75,
    min_observations_needed: int = 6,
    vertex_count_overshoot: int = 3,
    prevent_one_year_recovery: bool = True,
) -> NDArray[np.float32]:
    """Per-pixel segment table (standalone analog of GEE getSegmentData).

    Returns an (n_segments, 7) array, one row per fitted segment in vertex
    order, columns [start_year, end_year, start_val, end_val, magnitude,
    duration, rate]. Empty (0, 7) when the pixel is under-observed.

    Segmentation tracks LT-IDL closely — on the bundled validation pixels the
    vertex years match IDL exactly and fitted values agree to within ~8 NBRx1000
    (see idl-harness/). Values are read off the fitted trajectory, so a segment's
    magnitude reflects the fit, which on a fast recovery can overshoot the
    observed plateau exactly as IDL's anchored fit does."""

def raster_segments(
    data: NDArray[np.float32],
    years: NDArray[np.int32],
    max_segments: int = 6,
    spike_threshold: float = 0.9,
    recovery_threshold: float = 0.25,
    p_value_threshold: float = 0.05,
    best_model_proportion: float = 0.75,
    min_observations_needed: int = 6,
    vertex_count_overshoot: int = 3,
    prevent_one_year_recovery: bool = True,
) -> NDArray[np.float32]:
    """Raster-stack segment tables, NaN-padded to a fixed shape.

    `data` has shape (n_years, n_pixels). Returns (n_pixels, max_segments, 7):
    per pixel, up to max_segments segment rows (columns as in `segments`),
    remaining rows NaN. Parallelizes across pixels. See `segments` on how the
    decomposition relates to the IDL/GEE reference."""
