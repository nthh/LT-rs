//! PyO3 bindings for the standalone LandTrendr kernel.
//!
//! Exposes the full per-pixel result (fitted trajectory + vertices) so the port
//! can be compared vertex-for-vertex against native GEE LandTrendr.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::{landtrendr_flat as core_flat, landtrendr_pixel as core_pixel, LandTrendrParams};

fn params(
    max_segments: usize,
    spike_threshold: f32,
    recovery_threshold: f32,
    p_value_threshold: f64,
    best_model_proportion: f64,
    min_observations_needed: usize,
    vertex_count_overshoot: usize,
    prevent_one_year_recovery: bool,
) -> LandTrendrParams {
    LandTrendrParams {
        max_segments,
        spike_threshold,
        vertex_count_overshoot,
        recovery_threshold,
        p_value_threshold,
        best_model_proportion,
        min_observations_needed,
        prevent_one_year_recovery,
    }
}

/// Full per-pixel LandTrendr: returns (fitted, is_vertex, rmse).
///
/// Canonical LT-GEE defaults are the signature defaults (maxSegments 6, spike 0.9,
/// vertexCountOvershoot 3, preventOneYearRecovery true, recovery 0.25, pval 0.05,
/// bestModelProportion 0.75, minObs 6). NaNs in `values` mark missing years.
#[pyfunction]
#[pyo3(signature = (values, years, max_segments=6, spike_threshold=0.9, recovery_threshold=0.25, p_value_threshold=0.05, best_model_proportion=0.75, min_observations_needed=6, vertex_count_overshoot=3, prevent_one_year_recovery=true))]
fn landtrendr_pixel<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f32>,
    years: PyReadonlyArray1<'py, i32>,
    max_segments: usize,
    spike_threshold: f32,
    recovery_threshold: f32,
    p_value_threshold: f64,
    best_model_proportion: f64,
    min_observations_needed: usize,
    vertex_count_overshoot: usize,
    prevent_one_year_recovery: bool,
) -> (Py<PyArray1<f32>>, Py<PyArray1<u8>>, f32) {
    let p = params(
        max_segments, spike_threshold, recovery_threshold, p_value_threshold,
        best_model_proportion, min_observations_needed, vertex_count_overshoot,
        prevent_one_year_recovery,
    );
    let r = core_pixel(values.as_slice().unwrap(), years.as_slice().unwrap(), &p);
    let is_vertex: Vec<u8> = r.is_vertex.iter().map(|&b| b as u8).collect();
    (
        PyArray1::from_vec(py, r.fitted).into(),
        PyArray1::from_vec(py, is_vertex).into(),
        r.rmse,
    )
}

/// Raster-stack LandTrendr: 4 summary bands per pixel [net_mag, year, rmse, peak_to_trough].
#[pyfunction]
#[pyo3(signature = (data, pixel_count, band_count, years, max_segments=6, spike_threshold=0.9, recovery_threshold=0.25, p_value_threshold=0.05, best_model_proportion=0.75, min_observations_needed=6, vertex_count_overshoot=3, prevent_one_year_recovery=true))]
fn landtrendr_flat<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    pixel_count: usize,
    band_count: usize,
    years: PyReadonlyArray1<'py, i32>,
    max_segments: usize,
    spike_threshold: f32,
    recovery_threshold: f32,
    p_value_threshold: f64,
    best_model_proportion: f64,
    min_observations_needed: usize,
    vertex_count_overshoot: usize,
    prevent_one_year_recovery: bool,
) -> Py<PyArray1<f32>> {
    let p = params(
        max_segments, spike_threshold, recovery_threshold, p_value_threshold,
        best_model_proportion, min_observations_needed, vertex_count_overshoot,
        prevent_one_year_recovery,
    );
    let out = core_flat(
        data.as_slice().unwrap(), pixel_count, band_count,
        years.as_slice().unwrap(), &p,
    );
    PyArray1::from_vec(py, out).into()
}

#[pymodule]
fn lt_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(landtrendr_pixel, m)?)?;
    m.add_function(wrap_pyfunction!(landtrendr_flat, m)?)?;
    Ok(())
}
