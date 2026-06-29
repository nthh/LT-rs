//! LandTrendr temporal segmentation — a standalone Rust implementation.
//!
//! Per-pixel piecewise-linear segmentation of an annual spectral-index trajectory:
//!
//!   Kennedy, R.E., Yang, Z., Cohen, W.B. (2010). Detecting trends in forest
//!   disturbance and recovery using yearly Landsat time series: 1. LandTrendr —
//!   Temporal Segmentation Algorithms. Remote Sensing of Environment 114(12),
//!   2897–2910. https://doi.org/10.1016/j.rse.2010.07.008
//!
//! Validated against the native Google Earth Engine implementation:
//!   Kennedy, R.E. et al. (2018). Implementation of the LandTrendr Algorithm on
//!   Google Earth Engine. Remote Sensing 10(5), 691. https://doi.org/10.3390/rs10050691

// LandTrendr temporal segmentation
// ---------------------------------------------------------------------------

/// LandTrendr algorithm parameters.
pub struct LandTrendrParams {
    pub max_segments: usize,
    pub spike_threshold: f32,
    pub vertex_count_overshoot: usize,
    pub recovery_threshold: f32,
    pub p_value_threshold: f64,
    pub best_model_proportion: f64,
    pub min_observations_needed: usize,
    pub prevent_one_year_recovery: bool,
}

impl Default for LandTrendrParams {
    fn default() -> Self {
        Self {
            max_segments: 6,
            spike_threshold: 0.9,
            vertex_count_overshoot: 3,
            recovery_threshold: 0.25,
            p_value_threshold: 0.01,
            best_model_proportion: 1.25,
            min_observations_needed: 6,
            prevent_one_year_recovery: true,
        }
    }
}

/// Result of LandTrendr on a single pixel.
pub struct LandTrendrPixelResult {
    pub fitted: Vec<f32>,
    pub is_vertex: Vec<bool>,
    pub rmse: f32,
    pub segments: Vec<SegmentInfo>,
}

pub struct SegmentInfo {
    pub start_year: i32,
    pub end_year: i32,
    pub start_val: f32,
    pub end_val: f32,
    pub magnitude: f32,
    pub duration: i32,
    pub rate: f32,
}

/// Run LandTrendr segmentation on a single pixel time series.
pub fn landtrendr_pixel(
    values: &[f32],
    years: &[i32],
    params: &LandTrendrParams,
) -> LandTrendrPixelResult {
    let n = values.len();
    assert_eq!(n, years.len(), "values and years must have same length");
    assert!(n <= LT_MAX_N, "time series too long for workspace (max {})", LT_MAX_N);

    let n_valid = values.iter().filter(|v| !v.is_nan()).count();
    if n_valid < params.min_observations_needed {
        return LandTrendrPixelResult {
            fitted: values.to_vec(),
            is_vertex: vec![false; n],
            rmse: 0.0,
            segments: Vec::new(),
        };
    }

    // Delegate to the fast workspace-based implementation (single algorithm path)
    let mut ws = LandTrendrWorkspace::new();
    let selected = landtrendr_pixel_fast_core(values, years, n, params, &mut ws);

    // Extract full results from workspace
    let fitted = ws.fitted[..n].to_vec();
    let nv = ws.cand_n_verts[selected];
    let final_verts: Vec<usize> = ws.cand_verts[selected][..nv].to_vec();

    let mut is_vertex = vec![false; n];
    for &vi in &final_verts {
        is_vertex[vi] = true;
    }

    let mut sum_sq: f64 = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        if !values[i].is_nan() {
            let d = (values[i] - fitted[i]) as f64;
            sum_sq += d * d;
            count += 1;
        }
    }
    let rmse = if count > 0 { (sum_sq / count as f64).sqrt() as f32 } else { 0.0 };

    let segments = extract_segments(years, &fitted, &final_verts);

    LandTrendrPixelResult { fitted, is_vertex, rmse, segments }
}

// ---------------------------------------------------------------------------
// Shared helpers (used by both landtrendr_pixel and landtrendr_flat paths)
// ---------------------------------------------------------------------------

fn extract_segments(years: &[i32], fitted: &[f32], vertex_indices: &[usize]) -> Vec<SegmentInfo> {
    let mut verts = vertex_indices.to_vec();
    verts.sort_unstable();
    let mut segments = Vec::new();

    for i in 0..verts.len().saturating_sub(1) {
        let i_start = verts[i];
        let i_end = verts[i + 1];
        let start_year = years[i_start];
        let end_year = years[i_end];
        let start_val = fitted[i_start];
        let end_val = fitted[i_end];
        let magnitude = end_val - start_val;
        let duration = end_year - start_year;
        let rate = if duration > 0 {
            magnitude / duration as f32
        } else {
            0.0
        };

        segments.push(SegmentInfo {
            start_year,
            end_year,
            start_val,
            end_val,
            magnitude,
            duration,
            rate,
        });
    }

    segments
}

/// Approximate the F-distribution survival function (1 - CDF).
/// Uses Wilson-Hilferty normal approximation.
fn f_survival(f_stat: f64, df1: f64, df2: f64) -> f64 {
    if f_stat <= 0.0 {
        return 1.0;
    }

    let a = df1;
    let b = df2;
    let x = f_stat;

    let term1 = x.powf(1.0 / 3.0);
    let term2 = 1.0 - 2.0 / (9.0 * b);
    let term3 = (2.0 / (9.0 * b)).sqrt();
    let term4 = 1.0 - 2.0 / (9.0 * a);
    let term5 = (2.0 / (9.0 * a)).sqrt();

    let z = (term1 * term4 - term2) / (term1 * term1 * term5 * term5 + term3 * term3).sqrt();

    1.0 - normal_cdf(z)
}

/// Approximate normal CDF using Abramowitz & Stegun error function approx.
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-abs_x * abs_x).exp();

    0.5 * (1.0 + sign * y)
}

// ---------------------------------------------------------------------------
// LandTrendr fast path — zero-allocation per-pixel workspace
// ---------------------------------------------------------------------------
//
// The original landtrendr_pixel allocates ~92 Vecs per pixel call (fit_line
// returns Vec, identify_vertices/fit_segments allocate per-segment arrays).
// For 239K pixels this causes ~22M allocations, which dominate wall time
// in WASM's simple dlmalloc allocator. The fast path pre-allocates all
// buffers in a workspace struct and reuses them across pixels, reducing
// heap allocations to zero in the hot loop. Output is bit-identical.

const LT_MAX_N: usize = 128;
const LT_MAX_VERTS: usize = 24;
const LT_MAX_CANDIDATES: usize = 12;

struct LandTrendrWorkspace {
    vertices: [usize; LT_MAX_VERTS],
    work_verts: [usize; LT_MAX_VERTS],
    cand_verts: [[usize; LT_MAX_VERTS]; LT_MAX_CANDIDATES],
    cand_n_verts: [usize; LT_MAX_CANDIDATES],
    cand_ssr: [f64; LT_MAX_CANDIDATES],
    fitted: [f32; LT_MAX_N],
}

impl LandTrendrWorkspace {
    fn new() -> Self {
        Self {
            vertices: [0; LT_MAX_VERTS],
            work_verts: [0; LT_MAX_VERTS],
            cand_verts: [[0; LT_MAX_VERTS]; LT_MAX_CANDIDATES],
            cand_n_verts: [0; LT_MAX_CANDIDATES],
            cand_ssr: [0.0; LT_MAX_CANDIDATES],
            fitted: [0.0; LT_MAX_N],
        }
    }
}

/// Least-squares line fit returning (slope, intercept). Zero allocation.
#[inline]
fn fit_line_coeffs(values: &[f32], years: &[i32], start: usize, end: usize) -> (f64, f64) {
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut count = 0usize;
    for i in start..=end {
        let y = values[i] as f64;
        if !y.is_nan() {
            let x = years[i] as f64;
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            count += 1;
        }
    }
    if count < 2 {
        let mean = if count > 0 { sum_y / count as f64 } else { 0.0 };
        return (0.0, mean);
    }
    let denom = count as f64 * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return (0.0, sum_y / count as f64);
    }
    let slope = (count as f64 * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / count as f64;
    (slope, intercept)
}

/// vertex_angle with precomputed ranges (avoids O(n) min/max scan per call).
#[inline]
fn vertex_angle_fast(
    values: &[f32], years: &[i32],
    i_left: usize, i_center: usize, i_right: usize,
    year_range: f64, val_range: f64,
) -> f64 {
    let dy1 = (years[i_center] - years[i_left]) as f64 / year_range;
    let dv1 = (values[i_center] - values[i_left]) as f64 / val_range;
    let dy2 = (years[i_right] - years[i_center]) as f64 / year_range;
    let dv2 = (values[i_right] - values[i_center]) as f64 / val_range;
    let len1 = (dy1 * dy1 + dv1 * dv1).sqrt();
    let len2 = (dy2 * dy2 + dv2 * dv2).sqrt();
    if len1 == 0.0 || len2 == 0.0 { return 0.0; }
    let cos_angle = ((dy1 * dy2 + dv1 * dv2) / (len1 * len2)).clamp(-1.0, 1.0);
    let angle_deg = cos_angle.acos() * (180.0 / std::f64::consts::PI);
    (180.0 - angle_deg).abs()
}

#[inline]
fn compute_ranges(values: &[f32], n: usize, years: &[i32]) -> (f64, f64) {
    let year_range = (years[n - 1] - years[0]).max(1) as f64;
    let mut val_min = f64::INFINITY;
    let mut val_max = f64::NEG_INFINITY;
    for i in 0..n {
        let v = values[i] as f64;
        if !v.is_nan() {
            if v < val_min { val_min = v; }
            if v > val_max { val_max = v; }
        }
    }
    let val_range = if (val_max - val_min).abs() < 1e-15 { 1.0 } else { val_max - val_min };
    (year_range, val_range)
}

#[inline]
fn interpolate_nans_into(values: &[f32], years: &[i32], n: usize, out: &mut [f32]) {
    out[..n].copy_from_slice(&values[..n]);
    let mut valid_years = [0.0f64; LT_MAX_N];
    let mut valid_vals = [0.0f64; LT_MAX_N];
    let mut n_valid = 0usize;
    for i in 0..n {
        if !values[i].is_nan() {
            valid_years[n_valid] = years[i] as f64;
            valid_vals[n_valid] = values[i] as f64;
            n_valid += 1;
        }
    }
    if n_valid == 0 || n_valid == n { return; }
    let xs = &valid_years[..n_valid];
    let ys = &valid_vals[..n_valid];
    for i in 0..n {
        if out[i].is_nan() {
            let x = years[i] as f64;
            out[i] = if x <= xs[0] {
                ys[0] as f32
            } else if x >= xs[n_valid - 1] {
                ys[n_valid - 1] as f32
            } else {
                let mut val = ys[n_valid - 1];
                for j in 0..n_valid - 1 {
                    if x >= xs[j] && x <= xs[j + 1] {
                        let t = (x - xs[j]) / (xs[j + 1] - xs[j]);
                        val = ys[j] + t * (ys[j + 1] - ys[j]);
                        break;
                    }
                }
                val as f32
            };
        }
    }
}

#[inline]
fn despike_inplace(values: &mut [f32], n: usize, spike_threshold: f32) {
    if spike_threshold >= 1.0 { return; }
    for i in 1..n.saturating_sub(1) {
        let d_left = values[i] - values[i - 1];
        let d_right = values[i + 1] - values[i];
        if d_left * d_right >= 0.0 { continue; }
        let abs_left = d_left.abs();
        let abs_right = d_right.abs();
        let larger = abs_left.max(abs_right);
        let smaller = abs_left.min(abs_right);
        if larger == 0.0 { continue; }
        if smaller / larger >= spike_threshold {
            values[i] = (values[i - 1] + values[i + 1]) / 2.0;
        }
    }
}

/// Simultaneous least-squares piecewise-linear fit. Zero allocation (stack arrays).
///
/// Solves for vertex y-values that minimize total squared residual, then
/// interpolates fitted values. Uses Thomas' algorithm on the tridiagonal
/// normal equations — O(k) time for k vertices.
#[inline]
fn fit_segments_into(
    values: &[f32], years: &[i32],
    verts: &[usize], n_verts: usize,
    n: usize, fitted_out: &mut [f32],
) {
    if n_verts < 2 {
        fitted_out[..n].copy_from_slice(&values[..n]);
        return;
    }

    // Build tridiagonal normal equations (A^T A) y = (A^T b)
    let mut diag = [0.0f64; LT_MAX_VERTS];
    let mut off = [0.0f64; LT_MAX_VERTS];
    let mut rhs = [0.0f64; LT_MAX_VERTS];

    for s in 0..n_verts - 1 {
        let i_start = verts[s];
        let i_end = verts[s + 1];
        let year_start = years[i_start] as f64;
        let year_span = (years[i_end] - years[i_start]) as f64;

        for i in i_start..=i_end {
            let v = values[i];
            if v.is_nan() { continue; }
            let val = v as f64;
            let t = if year_span > 0.0 { (years[i] as f64 - year_start) / year_span } else { 0.0 };
            let w0 = 1.0 - t;
            let w1 = t;

            diag[s]     += w0 * w0;
            diag[s + 1] += w1 * w1;
            off[s]      += w0 * w1;
            rhs[s]      += w0 * val;
            rhs[s + 1]  += w1 * val;
        }
    }

    // Thomas' algorithm — forward sweep
    let mut c = [0.0f64; LT_MAX_VERTS];
    let mut d = [0.0f64; LT_MAX_VERTS];

    c[0] = if diag[0].abs() > 1e-15 { off[0] / diag[0] } else { 0.0 };
    d[0] = if diag[0].abs() > 1e-15 { rhs[0] / diag[0] } else { 0.0 };

    for i in 1..n_verts {
        let sub = off[i - 1];
        let denom = diag[i] - sub * c[i - 1];
        if denom.abs() < 1e-15 {
            c[i] = 0.0;
            d[i] = 0.0;
        } else {
            c[i] = if i < n_verts - 1 { off[i] / denom } else { 0.0 };
            d[i] = (rhs[i] - sub * d[i - 1]) / denom;
        }
    }

    // Back substitution → vertex y-values
    let mut y_verts = [0.0f64; LT_MAX_VERTS];
    y_verts[n_verts - 1] = d[n_verts - 1];
    for i in (0..n_verts - 1).rev() {
        y_verts[i] = d[i] - c[i] * y_verts[i + 1];
    }

    // Interpolate fitted values
    for i in 0..n { fitted_out[i] = f32::NAN; }
    for s in 0..n_verts - 1 {
        let i_start = verts[s];
        let i_end = verts[s + 1];
        let year_start = years[i_start] as f64;
        let year_span = (years[i_end] - years[i_start]) as f64;
        for i in i_start..=i_end {
            let t = if year_span > 0.0 { (years[i] as f64 - year_start) / year_span } else { 0.0 };
            fitted_out[i] = (y_verts[s] * (1.0 - t) + y_verts[s + 1] * t) as f32;
        }
    }
}

/// Identify vertices using iterative max-residual. Zero allocation.
fn identify_vertices_fast(
    values: &[f32], years: &[i32], n: usize,
    max_segments: usize, overshoot: usize,
    year_range: f64, val_range: f64,
    ws: &mut LandTrendrWorkspace,
) -> usize {
    let target = (max_segments + 1 + overshoot).min(n);
    ws.vertices[0] = 0;
    ws.vertices[1] = n - 1;
    let mut nv = 2usize;

    while nv < target {
        let mut best_residual = -1.0f64;
        let mut best_idx: Option<usize> = None;
        for s in 0..nv - 1 {
            let seg_start = ws.vertices[s];
            let seg_end = ws.vertices[s + 1];
            if seg_end - seg_start <= 1 { continue; }
            let (slope, intercept) = fit_line_coeffs(values, years, seg_start, seg_end);
            for i in (seg_start + 1)..seg_end {
                let fitted = intercept + slope * years[i] as f64;
                let residual = (values[i] as f64 - fitted).abs();
                if residual > best_residual {
                    best_residual = residual;
                    best_idx = Some(i);
                }
            }
        }
        match best_idx {
            Some(idx) if best_residual > 0.0 => {
                let mut exists = false;
                for i in 0..nv {
                    if ws.vertices[i] == idx { exists = true; break; }
                }
                if exists { break; }
                // Insert in sorted position
                let mut pos = nv;
                for i in 0..nv {
                    if ws.vertices[i] > idx { pos = i; break; }
                }
                let mut j = nv;
                while j > pos { ws.vertices[j] = ws.vertices[j - 1]; j -= 1; }
                ws.vertices[pos] = idx;
                nv += 1;
            }
            _ => break,
        }
    }

    // Cull to max_segments + 1 by importance. vertex_angle_fast returns the interior
    // angle: ~180 where the path runs straight through the vertex (least important,
    // remove first), ~0 at a sharp peak/trough (most important, keep). Remove the
    // STRAIGHTEST (max metric) each round — removing min would discard the sharp
    // disturbance vertices and keep noise, smearing real V-shaped events.
    let target_count = max_segments + 1;
    while nv > target_count {
        let mut max_angle = f64::NEG_INFINITY;
        let mut max_idx: Option<usize> = None;
        for i in 1..nv - 1 {
            let angle = vertex_angle_fast(
                values, years,
                ws.vertices[i - 1], ws.vertices[i], ws.vertices[i + 1],
                year_range, val_range,
            );
            if angle > max_angle { max_angle = angle; max_idx = Some(i); }
        }
        match max_idx {
            Some(idx) => {
                for i in idx..nv - 1 { ws.vertices[i] = ws.vertices[i + 1]; }
                nv -= 1;
            }
            None => break,
        }
    }
    nv
}

/// Build candidate models and select best via F-test. Zero allocation.
/// ws.fitted contains the selected model's fitted values on return.
fn fit_and_select_fast(
    values: &[f32], years: &[i32], n: usize,
    n_verts: usize, params: &LandTrendrParams,
    year_range: f64, val_range: f64,
    ws: &mut LandTrendrWorkspace,
) -> usize {
    let n_valid = values[..n].iter().filter(|v| !v.is_nan()).count();

    ws.work_verts[..n_verts].copy_from_slice(&ws.vertices[..n_verts]);
    let mut n_wv = n_verts;
    let mut n_cand = 0usize;

    while n_wv >= 2 && n_cand < LT_MAX_CANDIDATES {
        ws.cand_verts[n_cand][..n_wv].copy_from_slice(&ws.work_verts[..n_wv]);
        ws.cand_n_verts[n_cand] = n_wv;

        fit_segments_into(values, years, &ws.work_verts, n_wv, n, &mut ws.fitted);
        let mut ssr = 0.0f64;
        for i in 0..n {
            if !values[i].is_nan() {
                let d = (values[i] - ws.fitted[i]) as f64;
                ssr += d * d;
            }
        }
        ws.cand_ssr[n_cand] = ssr;
        n_cand += 1;

        if n_wv <= 2 { break; }

        // Build the next simpler candidate by removing the vertex whose removal
        // yields the LEAST increase in SSR/MSE — Kennedy 2010 §2.5.4 / Fig 3e
        // ("Remove vertex resulting in least increase in MSE"). The earlier
        // angle-change heuristic is a geometric proxy; the paper's criterion is
        // the actual fit cost, which keeps the recovery-curve vertices a sharp-
        // angle proxy would trade away for stable-period noise.
        let mut best_ssr = f64::INFINITY;
        let mut best_remove: Option<usize> = None;
        let mut trial = [0.0f32; LT_MAX_N];
        let mut tverts = [0usize; LT_MAX_VERTS];
        for r in 1..n_wv - 1 {
            let mut m = 0usize;
            for i in 0..n_wv {
                if i != r { tverts[m] = ws.work_verts[i]; m += 1; }
            }
            fit_segments_into(values, years, &tverts[..m], m, n, &mut trial);
            let mut ssr = 0.0f64;
            for i in 0..n {
                if !values[i].is_nan() {
                    let d = (values[i] - trial[i]) as f64;
                    ssr += d * d;
                }
            }
            if ssr < best_ssr { best_ssr = ssr; best_remove = Some(r); }
        }
        match best_remove {
            Some(idx) => {
                for i in idx..n_wv - 1 { ws.work_verts[i] = ws.work_verts[i + 1]; }
                n_wv -= 1;
            }
            None => break,
        }
    }

    if n_cand == 0 {
        let (slope, intercept) = fit_line_coeffs(values, years, 0, n - 1);
        for i in 0..n { ws.fitted[i] = (intercept + slope * years[i] as f64) as f32; }
        ws.cand_verts[0][0] = 0;
        ws.cand_verts[0][1] = n - 1;
        ws.cand_n_verts[0] = 2;
        return 0;
    }

    let full_ssr = ws.cand_ssr[0];
    let full_n_params = ws.cand_n_verts[0];
    if full_ssr < 1e-10 {
        fit_segments_into(
            values, years, &ws.cand_verts[0], ws.cand_n_verts[0], n, &mut ws.fitted,
        );
        return 0;
    }

    let mut selected = n_cand - 1;
    for idx in (0..n_cand).rev() {
        let model_n_params = ws.cand_n_verts[idx];
        if model_n_params >= full_n_params { continue; }
        let df1 = full_n_params - model_n_params;
        if n_valid <= full_n_params { continue; }
        let df2 = n_valid - full_n_params;
        if df1 == 0 || df2 <= 1 || full_ssr <= 0.0 { continue; }
        let f_stat = ((ws.cand_ssr[idx] - full_ssr) / df1 as f64) / (full_ssr / df2 as f64);
        let p_value = f_survival(f_stat, df1 as f64, df2 as f64);
        if p_value < params.p_value_threshold {
            selected = if idx > 0 { idx - 1 } else { 0 };
            break;
        }
    }

    // best_model_proportion check
    let full_rmse = (full_ssr / n_valid.max(1) as f64).sqrt();
    let sel_rmse = (ws.cand_ssr[selected] / n_valid.max(1) as f64).sqrt();
    if sel_rmse > 0.0 && full_rmse > 0.0 && sel_rmse / full_rmse > params.best_model_proportion {
        selected = 0;
    }

    fit_segments_into(
        values, years,
        &ws.cand_verts[selected], ws.cand_n_verts[selected], n, &mut ws.fitted,
    );

    selected
}

/// Core LandTrendr implementation — single algorithm path.
/// Runs despike → vertex identification → model selection → recovery clamp.
/// Returns selected candidate index; results are in ws.fitted and ws.cand_verts.
#[inline]
fn landtrendr_pixel_fast_core(
    values: &[f32], years: &[i32], n: usize,
    params: &LandTrendrParams, ws: &mut LandTrendrWorkspace,
) -> usize {
    let mut despiked = [0.0f32; LT_MAX_N];
    interpolate_nans_into(values, years, n, &mut despiked);
    despike_inplace(&mut despiked, n, params.spike_threshold);

    let (year_range, val_range) = compute_ranges(&despiked, n, years);

    let mut n_verts = identify_vertices_fast(
        &despiked, years, n,
        params.max_segments, params.vertex_count_overshoot,
        year_range, val_range, ws,
    );

    // prevent_one_year_recovery: a disturbance bottom immediately followed by a
    // single-year recovery is almost always residual cloud/shadow rather than real
    // regrowth (eMapR LT-GEE runParam). Drop the vertex that ENDS such a 1-year
    // recovery so the recovery is forced to span >=2 years (loss-down NBR: a drop
    // into v[i] then a rise out of v[i] over one year). Done on the vertex set
    // before model fitting, so every candidate inherits the constraint.
    if params.prevent_one_year_recovery {
        let mut i = 1;
        while i + 1 < n_verts {
            let (a, b, c) = (ws.vertices[i - 1], ws.vertices[i], ws.vertices[i + 1]);
            let drop_in = despiked[b] < despiked[a];          // disturbance into the bottom
            let rise_out = despiked[c] > despiked[b];          // recovery out of the bottom
            let one_year = years[c] - years[b] == 1;
            if drop_in && rise_out && one_year {
                for k in (i + 1)..n_verts - 1 { ws.vertices[k] = ws.vertices[k + 1]; }
                n_verts -= 1;                                  // removed the 1-yr recovery endpoint
            } else {
                i += 1;
            }
        }
    }

    let selected = fit_and_select_fast(
        &despiked, years, n,
        n_verts, params, year_range, val_range, ws,
    );

    // Recovery clamp: after fitting, constrain recovery segment slopes.
    // Clamp vertex endpoints so rate <= recovery_threshold, re-interpolate.
    if params.recovery_threshold < 1.0 {
        let nv = ws.cand_n_verts[selected];
        let verts = &ws.cand_verts[selected][..nv];
        let mut changed = false;
        for i in 0..nv.saturating_sub(1) {
            let si = verts[i];
            let ei = verts[i + 1];
            if ei < n && si < n {
                let mag = ws.fitted[ei] - ws.fitted[si];
                let dur = (years[ei] - years[si]) as f32;
                if mag > 0.0 && dur > 0.0 && mag / dur > params.recovery_threshold {
                    ws.fitted[ei] = ws.fitted[si] + params.recovery_threshold * dur;
                    changed = true;
                }
            }
        }
        if changed {
            for i in 0..nv.saturating_sub(1) {
                let si = verts[i];
                let ei = verts[i + 1];
                if ei < n && si < n {
                    let sv = ws.fitted[si];
                    let ev = ws.fitted[ei];
                    let span = (years[ei] - years[si]) as f32;
                    if span > 0.0 {
                        for j in (si + 1)..ei {
                            ws.fitted[j] = sv + (years[j] - years[si]) as f32 / span * (ev - sv);
                        }
                    }
                }
            }
        }
    }

    selected
}

/// Fast per-pixel LandTrendr.
/// Returns (net_magnitude, disturbance_year, rmse, peak_to_trough_magnitude).
///
/// - `net_magnitude` = fitted[last] - fitted[first] (net change; back-compat band 0).
/// - `peak_to_trough_magnitude` = fitted[trough_idx] - fitted[peak_idx] over the
///   FULL fitted trajectory, where peak_idx = argmax(fitted), trough_idx =
///   argmin(fitted); set to 0.0 when trough_idx <= peak_idx (monotonic rise / no
///   disturbance). This is the canonical LandTrendr disturbance-depth statistic
///   and matches the validated Python path (extract.py in the Bootleg-MTBS run):
///       peak_idx = argmax(fitted); trough_idx = argmin(fitted)
///       magnitude = fitted[trough_idx] - fitted[peak_idx]   (<= 0)
///       magnitude[trough_idx <= peak_idx] = 0.0
///   Returns NaN for both magnitudes when the pixel has insufficient valid
///   observations (so callers can mask on isfinite, matching extract.py's
///   `magnitude[~valid] = NaN`).
#[inline]
fn landtrendr_pixel_fast(
    values: &[f32], years: &[i32], n: usize,
    params: &LandTrendrParams, ws: &mut LandTrendrWorkspace,
) -> (f32, f32, f32, f32) {
    let n_valid = values[..n].iter().filter(|v| !v.is_nan()).count();
    if n_valid < params.min_observations_needed {
        // Insufficient data: net_change keeps its historical 0.0 sentinel for
        // band-0 back-compat; peak-to-trough is NaN so it masks out as invalid.
        return (0.0, f32::NAN, 0.0, f32::NAN);
    }

    let selected = landtrendr_pixel_fast_core(values, years, n, params, ws);

    // RMSE
    let mut sum_sq: f64 = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        if !values[i].is_nan() {
            let d = (values[i] - ws.fitted[i]) as f64;
            sum_sq += d * d;
            count += 1;
        }
    }
    let rmse = if count > 0 { (sum_sq / count as f64).sqrt() as f32 } else { 0.0 };

    let net_change = ws.fitted[n - 1] - ws.fitted[0];

    let verts = &ws.cand_verts[selected];
    let nv = ws.cand_n_verts[selected];
    let mut max_mag = 0.0f32;
    let mut dist_year = f32::NAN;
    for i in 0..nv.saturating_sub(1) {
        let magnitude = ws.fitted[verts[i + 1]] - ws.fitted[verts[i]];
        if magnitude < max_mag {
            max_mag = magnitude;
            dist_year = years[verts[i]] as f32;
        }
    }

    // Peak-to-trough over the full fitted trajectory (canonical disturbance
    // depth). argmax/argmin scan; tie-break = first index (matches numpy argmax/
    // argmin, which return the first occurrence). The fitted trajectory has no
    // NaNs over [0, n) for a fitted pixel, so no NaN guard is needed here.
    let mut peak_idx = 0usize;
    let mut trough_idx = 0usize;
    let mut peak_val = ws.fitted[0];
    let mut trough_val = ws.fitted[0];
    for i in 1..n {
        let v = ws.fitted[i];
        if v > peak_val {
            peak_val = v;
            peak_idx = i;
        }
        if v < trough_val {
            trough_val = v;
            trough_idx = i;
        }
    }
    let peak_to_trough = if trough_idx <= peak_idx {
        0.0 // monotonic rise (or trough precedes peak) => no disturbance
    } else {
        trough_val - peak_val // <= 0
    };

    (net_change, dist_year, rmse, peak_to_trough)
}

/// Run LandTrendr on a full raster stack.
///
/// `data`: flat slice (band_count bands of pixel_count pixels each)
/// `pixel_count`: pixels per band
/// `band_count`: number of annual observations
/// `years`: year for each band (length == band_count)
/// `params`: algorithm parameters
///
/// Returns flat Vec of pixel_count * 4, band-major:
///   [net_magnitude..., year..., rmse..., peak_to_trough_magnitude...]
///
/// Band 0 (net_magnitude) and bands 1/2 keep their original semantics for
/// back-compat. Band 3 (peak_to_trough_magnitude) is the canonical LandTrendr
/// disturbance-depth statistic (fitted trough - fitted peak over the full
/// trajectory; 0.0 for monotonic rises; NaN for under-observed pixels). See
/// `landtrendr_pixel_fast` for the exact definition.
pub fn landtrendr_flat(
    data: &[f32],
    pixel_count: usize,
    band_count: usize,
    years: &[i32],
    params: &LandTrendrParams,
) -> Vec<f32> {
    // Fast path: zero-allocation workspace for supported time series lengths
    if band_count <= LT_MAX_N {
        let mut magnitude_out = vec![0.0f32; pixel_count];
        let mut year_out = vec![f32::NAN; pixel_count];
        let mut rmse_out = vec![0.0f32; pixel_count];
        let mut ptt_out = vec![f32::NAN; pixel_count];
        let mut ws = LandTrendrWorkspace::new();
        let mut ts = vec![0.0f32; band_count];

        for px in 0..pixel_count {
            for t in 0..band_count {
                ts[t] = data[t * pixel_count + px];
            }
            let (mag, yr, rmse, ptt) =
                landtrendr_pixel_fast(&ts, years, band_count, params, &mut ws);
            magnitude_out[px] = mag;
            year_out[px] = yr;
            rmse_out[px] = rmse;
            ptt_out[px] = ptt;
        }

        let mut out = Vec::with_capacity(pixel_count * 4);
        out.extend_from_slice(&magnitude_out);
        out.extend_from_slice(&year_out);
        out.extend_from_slice(&rmse_out);
        out.extend_from_slice(&ptt_out);
        return out;
    }

    // Fallback for time series longer than LT_MAX_N
    let mut magnitude_out = vec![0.0f32; pixel_count];
    let mut year_out = vec![f32::NAN; pixel_count];
    let mut rmse_out = vec![0.0f32; pixel_count];
    let mut ptt_out = vec![f32::NAN; pixel_count];
    let mut ts = vec![0.0f32; band_count];

    for px in 0..pixel_count {
        for t in 0..band_count {
            ts[t] = data[t * pixel_count + px];
        }
        let result = landtrendr_pixel(&ts, years, params);
        // Net spectral change: last fitted value minus first
        let fitted = &result.fitted;
        magnitude_out[px] = fitted[fitted.len() - 1] - fitted[0];
        // Peak-to-trough over the full fitted trajectory (canonical disturbance
        // depth) — same definition as landtrendr_pixel_fast / extract.py.
        // Only computed for fitted pixels (>= min_observations_needed valid);
        // under-observed pixels leave ptt at the NaN init so they mask out.
        let n_valid_px = ts.iter().filter(|v| !v.is_nan()).count();
        if n_valid_px >= params.min_observations_needed {
            let mut peak_idx = 0usize;
            let mut trough_idx = 0usize;
            let mut peak_val = fitted[0];
            let mut trough_val = fitted[0];
            for i in 1..fitted.len() {
                let v = fitted[i];
                if v > peak_val { peak_val = v; peak_idx = i; }
                if v < trough_val { trough_val = v; trough_idx = i; }
            }
            ptt_out[px] = if trough_idx <= peak_idx { 0.0 } else { trough_val - peak_val };
        }
        // Year of greatest disturbance
        let mut max_mag: f32 = 0.0;
        let mut dist_year = f32::NAN;
        for seg in &result.segments {
            if seg.magnitude < max_mag {
                max_mag = seg.magnitude;
                dist_year = seg.start_year as f32;
            }
        }
        year_out[px] = dist_year;
        rmse_out[px] = result.rmse;
    }

    let mut out = Vec::with_capacity(pixel_count * 4);
    out.extend_from_slice(&magnitude_out);
    out.extend_from_slice(&year_out);
    out.extend_from_slice(&rmse_out);
    out.extend_from_slice(&ptt_out);
    out
}

/// LandTrendr per-year FTV (fitted-to-vertex) difference at a target year.
///
/// Returns `pixel_count` f32: `fitted[idx] - fitted[idx-1]`, where `idx` is the
/// position of `target_year` in `years`. This is eMapR's `getLtFtvDiff(year)`
/// signal (forestEnsembleFunctions L1795) — the fitted change *in* the target
/// year, which the forest-loss ensemble's `getLtProbabilities` stretches to a
/// 0-100 loss probability. It is distinct from `peak_to_trough` (band 3 of
/// `landtrendr_flat`), which is the largest disturbance over the *whole*
/// trajectory regardless of year. Signed in the input index's units; the caller
/// orients (loss sign) and stretches. NaN where `target_year` is absent, has no
/// prior year, or the pixel is under-observed.
pub fn landtrendr_ftvdiff_flat(
    data: &[f32],
    pixel_count: usize,
    band_count: usize,
    years: &[i32],
    target_year: i32,
    params: &LandTrendrParams,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; pixel_count];
    let idx = match years.iter().position(|&y| y == target_year) {
        Some(i) if i >= 1 => i,
        _ => return out, // target year absent or has no prior year -> all NaN
    };
    if band_count > LT_MAX_N {
        return out; // the validated fast-path fit only supports band_count <= LT_MAX_N
    }
    // Use the SAME fast-path fit as landtrendr_flat (despike -> vertices -> segment
    // fit, leaving the fitted trajectory in ws.fitted). landtrendr_pixel is a
    // separate, over-smoothing path — do not use it here.
    let mut ws = LandTrendrWorkspace::new();
    let mut ts = vec![0.0f32; band_count];
    for px in 0..pixel_count {
        for t in 0..band_count {
            ts[t] = data[t * pixel_count + px];
        }
        let n_valid = ts.iter().filter(|v| !v.is_nan()).count();
        if n_valid < params.min_observations_needed {
            continue; // leave NaN
        }
        landtrendr_pixel_fast_core(&ts, years, band_count, params, &mut ws);
        out[px] = ws.fitted[idx] - ws.fitted[idx - 1];
    }
    out
}

/// Windowed LandTrendr loss magnitude around a target year (loss = a fitted DECREASE).
///
/// Sums the loss-direction per-year fitted drops `max(0, fitted[y-1] - fitted[y])` over
/// `[target_year - half_window, target_year + half_window]`. When a disturbance is fit as a
/// multi-year ramp, the single-year `landtrendr_ftvdiff_flat` reads only ~1/N of a loss
/// spread over N years (low recall); the window recovers the full magnitude. `half_window = 0`
/// is the single-year loss (clamped to >= 0). Returns a NON-NEGATIVE loss magnitude in the
/// input index's units (loss-down convention); the caller stretches to a loss probability.
/// Trade-off: a window gives up some year precision (a loss in target+-1 counts toward
/// target). NaN where target_year is absent / has no prior year / the pixel is under-observed.
pub fn landtrendr_loss_window(
    data: &[f32], pixel_count: usize, band_count: usize, years: &[i32],
    target_year: i32, half_window: usize, params: &LandTrendrParams,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; pixel_count];
    let idx = match years.iter().position(|&y| y == target_year) {
        Some(i) if i >= 1 => i,
        _ => return out,
    };
    if band_count > LT_MAX_N {
        return out;
    }
    let lo = idx.saturating_sub(half_window).max(1);
    let hi = (idx + half_window).min(band_count - 1);
    let mut ws = LandTrendrWorkspace::new();
    let mut ts = vec![0.0f32; band_count];
    for px in 0..pixel_count {
        for t in 0..band_count {
            ts[t] = data[t * pixel_count + px];
        }
        if ts.iter().filter(|v| !v.is_nan()).count() < params.min_observations_needed {
            continue;
        }
        landtrendr_pixel_fast_core(&ts, years, band_count, params, &mut ws);
        let mut loss = 0.0f32;
        for y in lo..=hi {
            loss += (ws.fitted[y - 1] - ws.fitted[y]).max(0.0);
        }
        out[px] = loss;
    }
    out
}

#[cfg(feature = "python")]
mod python;
