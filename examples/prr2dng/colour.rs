//! DNG colour tags from the colour description in a ProRes RAW frame header.
//!
//! The header describes colour with white balance gains `wb = (wbR, 1, wbB)`
//! applied to the stored camera RGB, a matrix `M` from white-balanced camera
//! RGB to CIE XYZ, a correlated colour temperature `T` for the shot, and an
//! exposure gain. `M` is already chromatically adapted to D65: it maps the
//! white-balanced neutral `(1, 1, 1)` to the D65 white whatever `T` is.
//!
//! A DNG reader wants the camera's response under the shooting illuminant
//! instead, so the adaptation is undone with the Bradford transform from D65
//! to the white of `T`:
//!
//! ```text
//! camera RGB -> scene XYZ = Bradford(D65 -> white(T)) · M · diag(wb)
//! ColorMatrix1            = inverse of that, scaled so that the largest
//!                           camera channel of white(T) is 1
//! ForwardMatrix1          = Bradford(D65 -> D50) · M
//! AsShotNeutral           = (1/wbR, 1, 1/wbB)
//! BaselineExposure        = log2(gain)
//! ```
//!
//! With these tags a reader finds the white of `AsShotNeutral` at exactly the
//! header's colour temperature, and renders the as-shot frame as `M` does.
//! The white of `T` is on the Planckian locus below 4000 K and on the CIE
//! daylight locus from 4000 K up.

use proresraw_rs::FrameHeader;

/// A 3×3 matrix, row-major.
pub type Mat3 = [[f64; 3]; 3];

/// CIE 1931 chromaticities of the D65 and D50 whites.
const D65: (f64, f64) = (0.31271, 0.32902);
const D50: (f64, f64) = (0.34567, 0.35850);

/// XYZ to the Bradford cone response.
const BRADFORD: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// EXIF light source codes DNG accepts as calibration illuminants, with
/// their colour temperatures.
const ILLUMINANTS: [(u16, f64); 5] = [
    (17, 2856.0), // Standard light A
    (23, 5003.0), // D50
    (20, 5503.0), // D55
    (21, 6504.0), // D65
    (22, 7504.0), // D75
];

/// Colour tags of one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct DngColour {
    /// The colour temperature the matrices are built for, in kelvin.
    pub temperature: f64,
    /// `CalibrationIlluminant1`: the EXIF illuminant nearest to
    /// `temperature`. With a single matrix it is only a label.
    pub illuminant: u16,
    /// `ColorMatrix1`: XYZ to camera RGB under the shooting illuminant.
    pub color_matrix: Mat3,
    /// `ForwardMatrix1`: white-balanced camera RGB to XYZ (D50).
    pub forward_matrix: Mat3,
    /// `AsShotNeutral`.
    pub as_shot_neutral: [f64; 3],
    /// `BaselineExposure`, in stops.
    pub baseline_exposure: f64,
}

impl DngColour {
    /// The tags for `header`. A missing colour temperature (0) is taken as
    /// D65, which leaves `M` unadapted; non-positive gains count as 1.
    pub fn new(header: &FrameHeader) -> Self {
        let temperature = match header.wb_cct {
            0 => 6504.0,
            t => f64::from(t).clamp(1000.0, 25000.0),
        };
        let gain = |g: f32| {
            if g > 0.0 && g.is_finite() {
                f64::from(g)
            } else {
                1.0
            }
        };
        let wb = [gain(header.wb_red), 1.0, gain(header.wb_blue)];
        let m = header.color_matrix.map(|row| row.map(f64::from));
        let white = white_xy(temperature);
        let camera_to_scene = mul(&mul(&bradford(D65, white), &m), &diag(wb));
        let color_matrix = inverse(&camera_to_scene);
        let largest = apply(&color_matrix, xyz(white))
            .into_iter()
            .fold(f64::MIN, f64::max);
        let illuminant = ILLUMINANTS
            .iter()
            .min_by(|a, b| {
                let distance = |t: f64| (1.0 / t - 1.0 / temperature).abs();
                distance(a.1).total_cmp(&distance(b.1))
            })
            .map(|&(code, _)| code)
            .expect("illuminants");
        Self {
            temperature,
            illuminant,
            color_matrix: color_matrix.map(|row| row.map(|v| v / largest)),
            forward_matrix: mul(&bradford(D65, D50), &m),
            as_shot_neutral: wb.map(|g| 1.0 / g),
            baseline_exposure: gain(header.gain).log2(),
        }
    }
}

/// XYZ of the chromaticity `xy`, with Y = 1.
fn xyz((x, y): (f64, f64)) -> [f64; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Chromaticity of the white of colour temperature `t`.
fn white_xy(t: f64) -> (f64, f64) {
    if t < 4000.0 {
        planck_xy(t)
    } else {
        daylight_xy(t)
    }
}

/// The Planckian locus, from Krystek's rational approximation in CIE 1960 uv.
fn planck_xy(t: f64) -> (f64, f64) {
    let u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t)
        / (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t * t);
    let v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t)
        / (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t * t);
    let d = 2.0 * u - 8.0 * v + 4.0;
    (3.0 * u / d, 2.0 * v / d)
}

/// The CIE daylight locus, with D65 itself at 6500 and 6504 K.
fn daylight_xy(t: f64) -> (f64, f64) {
    if t == 6500.0 || t == 6504.0 {
        return D65;
    }
    let x = if t <= 7000.0 {
        -4.6070e9 / t.powi(3) + 2.9678e6 / t.powi(2) + 0.09911e3 / t + 0.244063
    } else {
        -2.0064e9 / t.powi(3) + 1.9018e6 / t.powi(2) + 0.24748e3 / t + 0.237040
    };
    (x, -3.0 * x * x + 2.870 * x - 0.275)
}

/// Bradford chromatic adaptation from white `from` to white `to`.
fn bradford(from: (f64, f64), to: (f64, f64)) -> Mat3 {
    let source = apply(&BRADFORD, xyz(from));
    let target = apply(&BRADFORD, xyz(to));
    let gains = [0, 1, 2].map(|i| target[i] / source[i]);
    mul(&mul(&inverse(&BRADFORD), &diag(gains)), &BRADFORD)
}

fn diag(v: [f64; 3]) -> Mat3 {
    [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]]
}

fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn apply(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| m[i][k] * v[k]).sum())
}

/// The inverse by cofactors; the matrices here are well conditioned.
fn inverse(m: &Mat3) -> Mat3 {
    let cofactor = |r: usize, c: usize| {
        let (r0, r1) = ((r + 1) % 3, (r + 2) % 3);
        let (c0, c1) = ((c + 1) % 3, (c + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    let det: f64 = (0..3).map(|c| m[0][c] * cofactor(0, c)).sum();
    std::array::from_fn(|i| std::array::from_fn(|j| cofactor(j, i) / det))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear Rec.709 to XYZ, its last column adjusted so that (1, 1, 1)
    /// maps exactly to D65, as a header matrix does.
    fn rec709() -> [[f32; 3]; 3] {
        let white = xyz(D65);
        let mut m = [
            [0.4124, 0.3576, 0.0],
            [0.2126, 0.7152, 0.0],
            [0.0193, 0.1192, 0.0],
        ];
        for (row, w) in m.iter_mut().zip(white) {
            row[2] = w - row[0] - row[1];
        }
        m.map(|row| row.map(|v| v as f32))
    }

    fn header(cct: u16, wb_red: f32, wb_blue: f32) -> FrameHeader {
        FrameHeader {
            version: 0,
            vendor: *b"test",
            width: 64,
            height: 32,
            crop: Default::default(),
            bayer_pattern: proresraw_rs::BayerPattern::Rggb,
            white_level: 65279,
            wb_red,
            wb_blue,
            color_matrix: rec709(),
            gain: 11.8,
            wb_cct: cct,
            flags: 0,
            alignment: 0,
            quant_matrix: [1; 64],
            curve: proresraw_rs::Curve::DEFAULT,
        }
    }

    fn close(a: [f64; 3], b: [f64; 3], tolerance: f64) -> bool {
        a.iter().zip(&b).all(|(a, b)| (a - b).abs() < tolerance)
    }

    #[test]
    fn the_shot_white_maps_to_the_as_shot_neutral() {
        for (cct, r, b) in [(5235, 1.0, 1.0), (3200, 1.7, 0.6), (5139, 2.419, 1.5486)] {
            let colour = DngColour::new(&header(cct, r, b));
            let neutral = apply(&colour.color_matrix, xyz(white_xy(f64::from(cct))));
            let scale = neutral[1] / colour.as_shot_neutral[1];
            let scaled = neutral.map(|v| v / scale);
            assert!(
                close(scaled, colour.as_shot_neutral, 1e-6),
                "{cct} K: {scaled:?}"
            );
            assert!((neutral.iter().fold(0.0f64, |a, &b| a.max(b)) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn the_forward_matrix_maps_neutral_to_d50() {
        let colour = DngColour::new(&header(5600, 1.0, 1.0));
        let white = apply(&colour.forward_matrix, [1.0; 3]);
        assert!(close(white, xyz(D50), 2e-4), "{white:?}");
        assert!((colour.baseline_exposure - 11.8f64.log2()).abs() < 1e-6);
        assert_eq!(colour.illuminant, 20);
    }

    #[test]
    fn at_d65_the_matrix_is_only_inverted() {
        let colour = DngColour::new(&header(0, 1.0, 1.0));
        assert_eq!(colour.temperature, 6504.0);
        assert_eq!(colour.illuminant, 21);
        let m = rec709().map(|row| row.map(f64::from));
        let identity = mul(&colour.color_matrix, &m);
        let scale = identity[0][0];
        for (i, row) in identity.iter().enumerate() {
            let expected = std::array::from_fn(|j| if i == j { scale } else { 0.0 });
            assert!(close(*row, expected, 1e-9), "{identity:?}");
        }
    }

    #[test]
    fn bradford_maps_white_to_white() {
        let adapted = apply(&bradford(D65, planck_xy(2856.0)), xyz(D65));
        assert!(close(adapted, xyz(planck_xy(2856.0)), 1e-12));
    }
}
