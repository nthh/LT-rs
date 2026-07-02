//! Raster results must not depend on which pixels share a workspace or in
//! what order they are processed — i.e. `flat` on a stack must equal `flat`
//! on each pixel alone. Pixels 239/240 of the bundled box are a regression
//! case: with a reused, un-reset workspace, pixel 240's fit was contaminated
//! by pixel 239's leftover state (net_mag -0.12 instead of the validated
//! fresh-workspace -0.81).

use landtrendr::{flat, pixel, LandTrendrParams};

const YEARS: [i32; 33] = [
    1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995,
    1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
];

// Pixels 239 and 240 of data/nbr_1984_2016.npz (flattened band-major).
const PX_A: [f32; 33] = [
    0.800901, 0.767093, 0.806088, 0.799595, 0.806165, 0.796988, 0.798656,
    0.796194, 0.788024, 0.766055, 0.780225, 0.803805, 0.811181, f32::NAN,
    0.791653, 0.817029, 0.79544, 0.784025, 0.811748, 0.794779, 0.779283,
    0.787022, 0.781829, 0.766289, 0.740531, 0.731482, 0.746849, 0.714389,
    0.763078, 0.734704, 0.728457, 0.538626, 0.550555,
];
const PX_B: [f32; 33] = [
    0.806623, 0.781431, 0.785874, 0.797023, 0.802688, 0.732402, 0.802733,
    0.814364, 0.767752, 0.803099, 0.771571, 0.780535, 0.799409, f32::NAN,
    0.77325, 0.834149, 0.813577, 0.788592, 0.81134, 0.787354, 0.783755,
    0.794004, 0.782494, 0.78201, 0.702801, 0.731501, 0.724799, 0.739734,
    0.75518, 0.718098, 0.730819, 0.364815, 0.412738,
];

fn band_major(pixels: &[&[f32; 33]]) -> Vec<f32> {
    let mut data = vec![0.0f32; pixels.len() * 33];
    for (px, series) in pixels.iter().enumerate() {
        for t in 0..33 {
            data[t * pixels.len() + px] = series[t];
        }
    }
    data
}

#[test]
fn endpoint_vertex_is_never_removed() {
    // PX_B ends in a 1-year recovery (2015 bottom, 2016 uptick). The root
    // cause of the contamination above: prevent_one_year_recovery removed the
    // final vertex, leaving fitted[n-1] unwritten. The endpoint must survive.
    let r = pixel(&PX_B, &YEARS, &LandTrendrParams::default());
    assert!(r.is_vertex[32], "series endpoint lost its vertex");
    assert!(r.fitted[32].is_finite() && (r.fitted[32] - 0.4).abs() < 0.4,
        "fitted endpoint implausible: {}", r.fitted[32]);
}

#[test]
fn flat_is_order_independent() {
    let p = LandTrendrParams::default();
    let solo_a = flat(&PX_A, 1, 33, &YEARS, &p);
    let solo_b = flat(&PX_B, 1, 33, &YEARS, &p);

    let ab = flat(&band_major(&[&PX_A, &PX_B]), 2, 33, &YEARS, &p);
    let ba = flat(&band_major(&[&PX_B, &PX_A]), 2, 33, &YEARS, &p);

    for band in 0..4 {
        let eq = |x: f32, y: f32| (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-6;
        assert!(
            eq(ab[band * 2], solo_a[band]) && eq(ba[band * 2 + 1], solo_a[band]),
            "pixel A band {band} depends on stack position: solo {}, ab {}, ba {}",
            solo_a[band], ab[band * 2], ba[band * 2 + 1]
        );
        assert!(
            eq(ab[band * 2 + 1], solo_b[band]) && eq(ba[band * 2], solo_b[band]),
            "pixel B band {band} depends on stack position: solo {}, ab {}, ba {}",
            solo_b[band], ab[band * 2 + 1], ba[band * 2]
        );
    }
}
