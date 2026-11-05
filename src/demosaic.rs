//! Apple-compatible raw conversion: noise-aware demosaicing, white
//! balance and the output colour matrix, applied to the linear Bayer
//! mosaic of a full-resolution decode.
//!
//! The conversion runs in three passes over `f32` values, in the order and
//! with the arithmetic of Apple's standard raw conversion:
//!
//! 1. **Green.** Sensels are normalized to `(v - black) / range`. At red
//!    and blue sites, green is estimated horizontally and vertically from
//!    the neighbouring greens with a same-colour curvature correction that
//!    is used only where it stands above the compression noise expected at
//!    that level, and the two estimates are blended by inverse error.
//!    Red and blue sites are clipped at the level that white balance will
//!    map to 1.0.
//! 2. **Chroma.** The missing red and blue come from the diagonal (at red
//!    and blue sites) or the horizontal and vertical (at green sites)
//!    neighbours, through a colour-ratio estimate that falls back to the
//!    plain average where the signal is within the noise. White balance
//!    gains are applied and every channel is clipped at 1.0.
//! 3. **Matrix.** White-balanced camera RGB goes through the header's
//!    camera-to-XYZ matrix, the XYZ to Rec.2020 matrix and the exposure
//!    gain, combined in `f64` into one 3×3 matrix.
//!
//! The noise model is driven by the frame's quantizer
//! ([`frame_quantizer`]): stronger quantization means more noise, so the
//! interpolation trusts gradients and colour ratios less.
//!
//! Results agree with GPU runs of Apple's conversion to within a few
//! `f32` units in the last place: GPUs differ among themselves in how they
//! round divisions and square roots, so no single bit pattern is
//! canonical.

use crate::decoder::{Frame, ParsedFrame};
use crate::header::{BLACK_LEVEL, FrameHeader};

/// XYZ (D65) to Rec.2020 primaries.
pub const XYZ_TO_REC2020: [[f64; 3]; 3] = [
    [1.716651187971268, -0.355670783776393, -0.25336628137366],
    [-0.666684351832489, 1.616481236634939, 0.015768545813911],
    [0.017639857445311, -0.042770613257809, 0.942103121235474],
];

/// The frame quantizer that drives the demosaic noise model: the first
/// quantization matrix entry times the largest tile qscale of the frame
/// ([`ParsedFrame::max_qscale`], which counts invalid tiles too, as Apple's
/// decoder does).
///
/// Decoded frames carry the same value ([`Frame::quantizer`]); this form
/// is for callers that only parse.
pub fn frame_quantizer(parsed: &ParsedFrame<'_>) -> u32 {
    crate::decoder::quantizer(&parsed.header, parsed.max_qscale)
}

/// Parameters of the green pass and the chroma pass, in `f32` exactly as
/// the passes consume them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemosaicParams {
    /// Header Bayer code (0 RGGB, 1 GRBG, 2 GBRG, 3 BGGR): the index
    /// `2 * (y & 1) + (x & 1)` of the red sensel.
    pub cfa_phase: i32,
    /// Black level of the linear values.
    pub black: i32,
    /// `1 / range`, where range is white minus black.
    pub one_over_range: f32,
    /// Noise model `(a, b, c)`: the variance of a normalized value `v` is
    /// `c + max(b, a * (v + black / range)²)`.
    pub noise: [f32; 3],
    /// Clip levels of red and blue sites before white balance: `1 / gain`.
    pub clip: [f32; 2],
    /// `black / range`.
    pub black_over_range: f32,
    /// White balance gains for red and blue.
    pub wb: [f32; 2],
}

impl DemosaicParams {
    /// Parameters for a frame with the given header and quantizer
    /// ([`Frame::quantizer`], or [`frame_quantizer`] for a parsed packet),
    /// using the header's white balance.
    pub fn new(header: &FrameHeader, quantizer: u32) -> Self {
        Self::with_white_balance(header, quantizer, header.wb_red, header.wb_blue)
    }

    /// Parameters with explicit red and blue white balance gains. A gain of
    /// zero stands for 1 in the clip levels.
    pub fn with_white_balance(header: &FrameHeader, quantizer: u32, red: f32, blue: f32) -> Self {
        let range = u32::from(header.white_level - BLACK_LEVEL);
        let q = f64::from(quantizer);
        let base = q * q * 5.0 + 64.0 / 3.0;
        let range2 = f64::from(range) * f64::from(range);
        let rangef = range as f32;
        let inverse = |g: f32| if g == 0.0 { 1.0 } else { 1.0 / g };
        DemosaicParams {
            cfa_phase: i32::from(header.bayer_pattern.code()),
            black: i32::from(BLACK_LEVEL),
            one_over_range: 1.0 / rangef,
            noise: [
                (base / 150_994_944.0) as f32,
                (base / (range2 * 256.0)) as f32,
                (64.0 / 3.0 / range2) as f32,
            ],
            clip: [inverse(red), inverse(blue)],
            black_over_range: f32::from(BLACK_LEVEL) / rangef,
            wb: [red, blue],
        }
    }
}

/// How far [`develop`] takes the conversion. Stages run in order, each
/// including the ones before it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// Demosaiced camera RGB without white balance. Every channel clips
    /// at sensor white (1.0).
    Demosaiced,
    /// Camera RGB with the white balance gains applied. Red and blue clip
    /// where their gain takes them to 1.0, so all channels clip together.
    #[default]
    WhiteBalanced,
    /// Scene-linear Rec.2020 through [`output_matrix`] with
    /// [`Exposure::Normalized`], Apple's default output.
    Rec2020,
}

/// Exposure scaling of [`output_matrix`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Exposure {
    /// The header gain times 10/9, which puts 90 % diffuse white at 1.0.
    #[default]
    Normalized,
    /// The header gain alone.
    Reflection,
}

/// The combined output matrix `XYZ_TO_REC2020 · camera_to_xyz · gain`
/// for white-balanced camera RGB, computed in `f64` and rounded per
/// entry, row-major.
pub fn output_matrix(header: &FrameHeader, exposure: Exposure) -> [f32; 9] {
    let gain = f64::from(header.gain);
    let gain = match exposure {
        Exposure::Normalized => gain * 10.0 / 9.0,
        Exposure::Reflection => gain,
    };
    let c = header.color_matrix;
    let g = XYZ_TO_REC2020;
    std::array::from_fn(|i| {
        let (row, col) = (i / 3, i % 3);
        let v = g[row][0] * f64::from(c[0][col])
            + g[row][1] * f64::from(c[1][col])
            + g[row][2] * f64::from(c[2][col]);
        (v * gain) as f32
    })
}

/// An RGB image of `f32` values, interleaved, normalized so that 0 is
/// black and 1 sensor white (before white balance and matrix).
#[derive(Debug, Clone, PartialEq)]
pub struct RgbImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Interleaved R, G, B, row-major, tightly packed.
    pub data: Vec<f32>,
}

/// An RGB image of `f32` values as three planes, normalized like
/// [`RgbImage`].
#[derive(Debug, Clone, PartialEq)]
pub struct RgbPlanes {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// The R, G and B planes, each row-major and tightly packed
    /// (`width * height` values).
    pub planes: [Vec<f32>; 3],
}

/// Run Apple's raw conversion on a decoded frame up to `stage`, with the
/// header's white balance and gain. The quantizer for the noise model
/// is the frame's own ([`Frame::quantizer`]).
///
/// ```no_run
/// # let packet = std::fs::read("frame.bin").unwrap();
/// use proresraw_rs::{Stage, decode_frame, develop};
/// let frame = decode_frame(&packet).unwrap();
/// let camera = develop(&frame, Stage::Demosaiced); // no white balance
/// ```
pub fn develop(frame: &Frame, stage: Stage) -> RgbImage {
    let params = DemosaicParams::new(&frame.header, frame.quantizer());
    develop_with(frame, &params, stage)
}

/// Like [`develop`] with explicit parameters, for example white balance
/// gains of your own ([`DemosaicParams::with_white_balance`]). At
/// [`Stage::Demosaiced`] the gains in `params` are ignored and every
/// channel clips at exactly 1.0, sensor white, so a consumer needs no
/// clip-level estimate of its own.
///
/// Values are normalized `f32`: 0 is black and 1 is the header white
/// level. The other decoders return unnormalized `u16` linear values
/// ([`crate::Frame::linearize`], [`crate::decode_frame_reduced`]), with
/// [`crate::BLACK_LEVEL`] and [`FrameHeader::white_level`] still to be
/// applied.
pub fn develop_with(frame: &Frame, params: &DemosaicParams, stage: Stage) -> RgbImage {
    let (width, height) = (frame.width(), frame.height());
    let mut data = vec![0f32; 3 * width * height];
    develop_into(frame, params, stage, &mut data);
    RgbImage {
        width,
        height,
        data,
    }
}

/// [`develop_with`] into a caller-owned buffer of interleaved RGB, for
/// reusing one buffer across frames.
///
/// # Panics
///
/// If `out` does not hold exactly `3 * width * height` values.
pub fn develop_into(frame: &Frame, params: &DemosaicParams, stage: Stage, out: &mut [f32]) {
    let (w, h) = (frame.width(), frame.height());
    assert_eq!(out.len(), 3 * w * h, "output size");
    Conversion::new(frame, params, stage).run_interleaved(frame, out);
}

/// [`develop_with`] with the result as three planes, the layout most raw
/// developers and GPU uploads want. Values are bit-identical to the
/// interleaved forms.
pub fn develop_planar(frame: &Frame, params: &DemosaicParams, stage: Stage) -> RgbPlanes {
    let (width, height) = (frame.width(), frame.height());
    let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; width * height]);
    let [r, g, b] = &mut planes;
    develop_planar_into(frame, params, stage, [r, g, b]);
    RgbPlanes {
        width,
        height,
        planes,
    }
}

/// [`develop_planar`] into caller-owned R, G and B planes, each row-major
/// and tightly packed.
///
/// # Panics
///
/// If a plane does not hold exactly `width * height` values.
pub fn develop_planar_into(
    frame: &Frame,
    params: &DemosaicParams,
    stage: Stage,
    out: [&mut [f32]; 3],
) {
    let (w, h) = (frame.width(), frame.height());
    for plane in &out {
        assert_eq!(plane.len(), w * h, "plane size");
    }
    Conversion::new(frame, params, stage).run_planar(frame, out);
}

/// Where one band of [`Conversion`] writes its finished rows: the band's
/// slice of an interleaved image, or its slice of each plane.
enum BandOut<'a> {
    Interleaved(&'a mut [f32]),
    Planar([&'a mut [f32]; 3]),
}

/// Per-band scratch: normalized rows and green-pass rows.
type Scratch = (Vec<f32>, Vec<[f32; 2]>);

/// Everything the bands of one conversion share.
struct Conversion {
    params: DemosaicParams,
    matrix: Option<[f32; 9]>,
    /// Linearization and normalization folded into one table.
    lut: Vec<f32>,
}

impl Conversion {
    fn new(frame: &Frame, params: &DemosaicParams, stage: Stage) -> Self {
        let mut p = *params;
        if stage == Stage::Demosaiced {
            p.wb = [1.0; 2];
            p.clip = [1.0; 2];
        }
        let matrix =
            (stage == Stage::Rec2020).then(|| output_matrix(&frame.header, Exposure::Normalized));
        let lut = frame
            .header
            .curve
            .lut()
            .iter()
            .map(|&v| p.one_over_range * (i32::from(v) - p.black) as f32)
            .collect();
        Conversion {
            params: p,
            matrix,
            lut,
        }
    }

    fn run_interleaved(&self, frame: &Frame, out: &mut [f32]) {
        let chunk = 3 * frame.width() * BAND;
        let band = |scratch: &mut Scratch, (index, rows): (usize, &mut [f32])| {
            self.band(frame, index, BandOut::Interleaved(rows), scratch);
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(chunk)
                .enumerate()
                .for_each_init(Scratch::default, band);
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut scratch = Scratch::default();
            out.chunks_mut(chunk)
                .enumerate()
                .for_each(|item| band(&mut scratch, item));
        }
    }

    fn run_planar(&self, frame: &Frame, [r, g, b]: [&mut [f32]; 3]) {
        let chunk = frame.width() * BAND;
        type Item<'a> = (usize, ((&'a mut [f32], &'a mut [f32]), &'a mut [f32]));
        let band = |scratch: &mut Scratch, (index, ((r, g), b)): Item<'_>| {
            self.band(frame, index, BandOut::Planar([r, g, b]), scratch);
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            r.par_chunks_mut(chunk)
                .zip(g.par_chunks_mut(chunk))
                .zip(b.par_chunks_mut(chunk))
                .enumerate()
                .for_each_init(Scratch::default, band);
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut scratch = Scratch::default();
            r.chunks_mut(chunk)
                .zip(g.chunks_mut(chunk))
                .zip(b.chunks_mut(chunk))
                .enumerate()
                .for_each(|item| band(&mut scratch, item));
        }
    }

    /// Run every pass over band `index`, writing its rows to `out`. A band
    /// needs green one row beyond it and normalized values three rows
    /// beyond, recomputed at the seams, so that its data stays in cache.
    fn band(&self, frame: &Frame, index: usize, mut out: BandOut<'_>, scratch: &mut Scratch) {
        let p = &self.params;
        let (w, h) = (frame.width(), frame.height());
        let y0 = index * BAND;
        let y1 = (y0 + BAND).min(h);
        let (g0, g1) = (y0.saturating_sub(1), (y1 + 1).min(h));
        let (n0, n1) = (g0.saturating_sub(2), (g1 + 2).min(h));
        let (norm, green) = scratch;
        norm.clear();
        for y in n0..n1 {
            norm.extend(frame.row(y).iter().map(|&code| self.lut[usize::from(code)]));
        }
        green.clear();
        green.resize((g1 - g0) * w, [0.0; 2]);
        for (y, row) in (g0..g1).zip(green.chunks_exact_mut(w)) {
            let src = |dy: isize| {
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize - n0;
                &norm[yy * w..yy * w + w]
            };
            green_row([src(-2), src(-1), src(0), src(1), src(2)], y, h, p, row);
        }
        let g = |yy: usize| &green[(yy - g0) * w..(yy - g0) * w + w];
        for (i, y) in (y0..y1).enumerate() {
            let (ym, yp) = mirrored(y, h);
            let rows = [g(ym), g(y), g(yp)];
            match &mut out {
                BandOut::Interleaved(o) => {
                    chroma_row(rows, y, p, &mut o[3 * w * i..3 * w * (i + 1)])
                }
                BandOut::Planar([r, g, b]) => {
                    let span = w * i..w * (i + 1);
                    let planes = [&mut r[span.clone()], &mut g[span.clone()], &mut b[span]];
                    chroma_row_planar(rows, y, p, planes);
                }
            }
        }
        if let Some(m) = &self.matrix {
            match out {
                BandOut::Interleaved(o) => matrix_rows(o, m),
                BandOut::Planar([r, g, b]) => matrix_planes([r, g, b], m),
            }
        }
    }
}

/// Rows per band in [`develop_with`].
const BAND: usize = 32;

/// The rows above and below `y`, mirrored at the picture edges.
#[inline(always)]
fn mirrored(y: usize, len: usize) -> (usize, usize) {
    (
        if y > 0 { y - 1 } else { y + 1 },
        if y < len - 1 { y + 1 } else { y - 1 },
    )
}

/// `a * b + c` with a single rounding. This is one instruction on targets
/// with FMA (for example `-C target-cpu=x86-64-v3`) and a library call
/// otherwise.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// Noise variance of a normalized value.
#[inline(always)]
fn variance(v: f32, off: f32, [a, b, c]: [f32; 3]) -> f32 {
    let t = off + v;
    c + b.max(a * (t * t))
}

/// Four-term dot product, accumulated with fused multiply-adds in the
/// order GPU builds of the conversion use.
#[inline(always)]
fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    fma(a[3], b[3], fma(a[2], b[2], fma(a[0], b[0], a[1] * b[1])))
}

/// Run `f` over every chunk of `chunk` elements with its index, in
/// parallel with the `parallel` feature.
fn for_rows<T: Send>(out: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(y, row)| f(y, row));
    }
    #[cfg(not(feature = "parallel"))]
    for (y, row) in out.chunks_mut(chunk).enumerate() {
        f(y, row);
    }
}

/// Pass 1: normalized mosaic to (red or blue, green) per sensel. Red and
/// blue are 0 at green sites. `linear` is `width` × `height` linear values.
pub fn green_pass(linear: &[u16], w: usize, h: usize, p: &DemosaicParams) -> Vec<[f32; 2]> {
    assert_eq!(linear.len(), w * h);
    let mut norm = vec![0f32; w * h];
    for_rows(&mut norm, w, |y, row| {
        for (n, &v) in row.iter_mut().zip(&linear[y * w..]) {
            *n = p.one_over_range * (i32::from(v) - p.black) as f32;
        }
    });
    green_pass_normalized(&norm, w, h, p)
}

/// [`green_pass`] on values already normalized to `(v - black) / range`.
fn green_pass_normalized(norm: &[f32], w: usize, h: usize, p: &DemosaicParams) -> Vec<[f32; 2]> {
    assert_eq!(norm.len(), w * h);
    let mut out = vec![[0f32; 2]; w * h];
    for_rows(&mut out, w, |y, row| {
        let src = |dy: isize| {
            let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
            &norm[yy * w..yy * w + w]
        };
        green_row([src(-2), src(-1), src(0), src(1), src(2)], y, h, p, row);
    });
    out
}

/// One row `y` of the green pass. `rows` are the normalized rows `y - 2`
/// to `y + 2`, clamped to the picture of height `h`.
#[inline(always)]
fn green_row_impl(rows: [&[f32]; 5], y: usize, h: usize, p: &DemosaicParams, out: &mut [[f32; 2]]) {
    let w = out.len();
    let bs = p.black as f32 * p.one_over_range;
    let var = |v: f32| variance(v, bs, p.noise);
    let red_row = (p.cfa_phase >> 1) as usize;
    let red_col = (p.cfa_phase & 1) as usize;
    let last_x = w - 1;
    let centre = rows[2];
    // Green sites keep their value.
    for (o, &n) in out.iter_mut().zip(centre) {
        *o = [0.0, n];
    }
    let phase = y % 2 == red_row;
    let first = if phase { red_col } else { 1 - red_col };
    let clip = if phase { p.clip[0] } else { p.clip[1] };
    let limit = [1.0, 1.0, clip, clip];
    for x in (first..w).step_by(2) {
        let c = centre[x].min(clip);
        let at = |dx: isize| centre[(x as isize + dx).clamp(0, last_x as isize) as usize];
        let mut hh = [at(-1), at(1), at(-2), at(2)];
        let mut vv = [rows[1][x], rows[3][x], rows[0][x], rows[4][x]];
        for i in 0..4 {
            hh[i] = hh[i].min(limit[i]);
            vv[i] = vv[i].min(limit[i]);
        }
        mirror(&mut hh, x, w, c);
        mirror(&mut vv, y, h, c);
        let var_h = hh.map(var);
        let var_v = vv.map(var);
        let sums = [hh[0] + hh[1], vv[0] + vv[1], hh[2] + hh[3], vv[2] + vv[3]];
        let mut ratio = [0f32; 2];
        if c > var(c).sqrt() {
            // c * 2 is exact, so fusing would not change these.
            let t = [
                c * 2.0 + sums[2],
                c * 2.0 + sums[3],
                c * -2.0 + sums[2],
                c * -2.0 + sums[3],
            ];
            for (dir, (d, vd)) in [(hh, var_h), (vv, var_v)].into_iter().enumerate() {
                let sd = vd.map(f32::sqrt);
                let a = d[0] - d[1];
                let b = d[2] - d[3];
                let any = a * b > 0.0
                    || (d[2] - c) * (d[3] - c) > 0.0
                    || sd[0] + sd[1] > a.abs()
                    || sd[2] + sd[3] > b.abs();
                if any && (0..4).all(|i| d[i] > sd[i]) {
                    ratio[dir] = t[2 + dir] / (t[dir] * 2.0);
                }
            }
        }
        let gh = sums[0] * 0.5 * (1.0 - ratio[0]);
        let gv = sums[1] * 0.5 * (1.0 - ratio[1]);
        let dh = [hh[0] - gh, hh[1] - gh, hh[2] - c, hh[3] - c];
        let dv = [vv[0] - gv, vv[1] - gv, vv[2] - c, vv[3] - c];
        let eh = dot4(dh, dh) + (var_h[0] + var_h[1]);
        let ev = dot4(dv, dv) + (var_v[0] + var_v[1]);
        out[x] = [c, gv + (gh - gv) * ev / (eh + ev)];
    }
}

/// Border handling of the green pass: neighbours past the edge mirror the
/// ones on the other side; a same-colour neighbour two away from a pixel
/// next to the edge takes the centre value.
#[inline(always)]
fn mirror(d: &mut [f32; 4], pos: usize, len: usize, c: f32) {
    if pos == 0 {
        *d = [d[1], d[1], d[3], d[3]];
    } else if pos == 1 {
        d[2] = c;
    } else if pos == len - 1 {
        *d = [d[0], d[0], d[2], d[2]];
    } else if pos == len - 2 {
        d[3] = c;
    }
}

/// The centre green of a chroma estimate and its noise standard deviation.
#[derive(Clone, Copy)]
struct Centre {
    g: f32,
    sigma: f32,
}

/// Chroma estimate from four neighbours `xx` of the wanted colour with
/// greens `gg`: their mean, moved towards the colour-ratio estimate
/// `centre · x / g` by the confidence in each. Falls back to the mean when
/// the centre or any neighbour is within its noise. The four lanes are
/// computed side by side so that they vectorize.
#[inline(always)]
fn estimate(xx: [f32; 4], gg: [f32; 4], c: Centre, off: f32, noise: [f32; 3]) -> f32 {
    let quarter = [0.25; 4];
    let mean_x = dot4(xx, quarter);
    // Also falls back when the comparison is unordered (NaN).
    if c.sigma.partial_cmp(&c.g) != Some(std::cmp::Ordering::Less) {
        return mean_x;
    }
    let var_x = xx.map(|v| variance(v, off, noise));
    let sx = var_x.map(f32::sqrt);
    let sg = gg.map(|v| variance(v, off, noise).sqrt());
    let mut above = true;
    for i in 0..4 {
        above &= (sx[i] < xx[i]) & (sg[i] < gg[i]);
    }
    if !above {
        return mean_x;
    }
    let (cg, cs) = (c.g, c.sigma);
    let lo: [f32; 4] = std::array::from_fn(|i| ((cg - cs) * (xx[i] - sx[i])) / (sg[i] + gg[i]));
    let mid: [f32; 4] = std::array::from_fn(|i| (cg * xx[i]) / gg[i]);
    let hi: [f32; 4] = std::array::from_fn(|i| ((cs + cg) * (sx[i] + xx[i])) / (gg[i] - sg[i]));
    let spread = dot4(
        std::array::from_fn(|i| mid[i] - lo[i]),
        std::array::from_fn(|i| hi[i] - mid[i]),
    );
    let mean_mid = dot4(mid, quarter);
    let dm = mid.map(|v| v - mean_mid);
    let b = fma(spread, 0.75, dot4(dm, dm));
    let dx = xx.map(|v| v - mean_x);
    let a = fma(dot4(var_x, [1.0; 4]), 0.75, dot4(dx, dx));
    let weight = a / (b + a);
    fma(mean_mid - mean_x, weight, mean_x)
}

/// Pass 2: (red or blue, green) per sensel to white-balanced camera RGB,
/// clipped at 1.0, interleaved.
pub fn chroma_pass(green: &[[f32; 2]], w: usize, h: usize, p: &DemosaicParams) -> Vec<f32> {
    assert_eq!(green.len(), w * h);
    let mut out = vec![0f32; 3 * w * h];
    let row = |y: usize| &green[y * w..y * w + w];
    for_rows(&mut out, 3 * w, |y, out| {
        let (ym, yp) = mirrored(y, h);
        chroma_row([row(ym), row(y), row(yp)], y, p, out);
    });
    out
}

/// One row `y` of the chroma pass from the green-pass rows above, at and
/// below it (mirrored at the picture edges), into interleaved RGB.
#[inline(always)]
fn chroma_row_impl(rows: [&[[f32; 2]]; 3], y: usize, p: &DemosaicParams, out: &mut [f32]) {
    let out = &mut out[..3 * rows[1].len()];
    chroma_row_with(rows, y, p, |x, rgb| {
        out[3 * x..3 * x + 3].copy_from_slice(&rgb);
    });
}

/// [`chroma_row_impl`] into one row of each of the R, G and B planes.
#[inline(always)]
fn chroma_row_planar_impl(
    rows: [&[[f32; 2]]; 3],
    y: usize,
    p: &DemosaicParams,
    [r, g, b]: [&mut [f32]; 3],
) {
    let w = rows[1].len();
    let (r, g, b) = (&mut r[..w], &mut g[..w], &mut b[..w]);
    chroma_row_with(rows, y, p, |x, rgb| {
        r[x] = rgb[0];
        g[x] = rgb[1];
        b[x] = rgb[2];
    });
}

/// The chroma pass of one row, handing each pixel's finished RGB to `put`.
#[inline(always)]
fn chroma_row_with(
    rows: [&[[f32; 2]]; 3],
    y: usize,
    p: &DemosaicParams,
    mut put: impl FnMut(usize, [f32; 3]),
) {
    let [up, mid, down] = rows;
    let w = mid.len();
    let off = p.black_over_range;
    let noise = p.noise;
    let on_red_row = y % 2 == (p.cfa_phase >> 1) as usize;
    let red_col = (p.cfa_phase & 1) as usize;
    // Column parity of the red or blue sites of this row.
    let rb_col = if on_red_row { red_col } else { 1 - red_col };
    for x in 0..w {
        let (xm, xp) = mirrored(x, w);
        let [c, g] = mid[x];
        let centre = Centre {
            g,
            sigma: variance(g, off, noise).sqrt(),
        };
        let (r, b) = if x % 2 == rb_col {
            let n = [up[xm], down[xm], up[xp], down[xp]];
            let e = estimate(n.map(|v| v[0]), n.map(|v| v[1]), centre, off, noise);
            if on_red_row { (c, e) } else { (e, c) }
        } else {
            // Each of the two opposite neighbours counts twice.
            let pair = |u: [f32; 2], v: [f32; 2]| {
                let xx = [u[0], v[0], u[0], v[0]];
                estimate(xx, [u[1], v[1], u[1], v[1]], centre, off, noise)
            };
            let eh = pair(mid[xm], mid[xp]);
            let ev = pair(up[x], down[x]);
            if on_red_row { (eh, ev) } else { (ev, eh) }
        };
        put(
            x,
            [(r * p.wb[0]).min(1.0), g.min(1.0), (b * p.wb[1]).min(1.0)],
        );
    }
}

/// Pass 3: multiply interleaved RGB by a row-major 3×3 matrix. Each
/// output is `b·m2 + (r·m0 + g·m1)` with both additions fused.
pub fn apply_matrix(rgb: &[f32], m: &[f32; 9]) -> Vec<f32> {
    let mut out = rgb.to_vec();
    for_rows(&mut out, 3 * 4096, |_, chunk| matrix_rows(chunk, m));
    out
}

/// [`apply_matrix`] in place.
#[inline(always)]
fn matrix_rows_impl(rgb: &mut [f32], m: &[f32; 9]) {
    for px in rgb.chunks_exact_mut(3) {
        let [r, g, b] = [px[0], px[1], px[2]];
        for (v, i) in px.iter_mut().zip([0, 3, 6]) {
            *v = fma(b, m[i + 2], fma(r, m[i], g * m[i + 1]));
        }
    }
}

/// [`matrix_rows_impl`] on R, G and B planes of equal length.
#[inline(always)]
fn matrix_planes_impl([r, g, b]: [&mut [f32]; 3], m: &[f32; 9]) {
    let n = r.len();
    let (g, b) = (&mut g[..n], &mut b[..n]);
    for i in 0..n {
        let [rv, gv, bv] = [r[i], g[i], b[i]];
        r[i] = fma(bv, m[2], fma(rv, m[0], gv * m[1]));
        g[i] = fma(bv, m[5], fma(rv, m[3], gv * m[4]));
        b[i] = fma(bv, m[8], fma(rv, m[6], gv * m[7]));
    }
}

/// Row kernels, dispatched at run time to copies compiled with AVX2 and
/// FMA when the CPU has them (feature `runtime-fma`). Both copies come
/// from the same source and give identical results: `mul_add` is a fused
/// multiply-add either way, one instruction in the fast copy and a library
/// call otherwise.
macro_rules! dispatch {
    ($(fn $name:ident => $imp:ident ($($arg:ident: $ty:ty),*);)*) => {$(
        #[inline]
        fn $name($($arg: $ty),*) {
            #[cfg(all(feature = "runtime-fma", target_arch = "x86_64", not(target_feature = "fma")))]
            if fast::available() {
                #[allow(unsafe_code)]
                // SAFETY: `available` checked that the CPU supports every
                // feature `fast::$name` is compiled with.
                return unsafe { fast::$name($($arg),*) };
            }
            $imp($($arg),*)
        }
    )*
        #[cfg(all(feature = "runtime-fma", target_arch = "x86_64", not(target_feature = "fma")))]
        #[allow(unsafe_code)]
        mod fast {
            use super::*;

            /// Whether the CPU has AVX2 and FMA (cached by the standard
            /// library).
            #[inline]
            pub(super) fn available() -> bool {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }

            $(
                #[target_feature(enable = "avx2,fma")]
                pub(super) unsafe fn $name($($arg: $ty),*) {
                    $imp($($arg),*)
                }
            )*
        }
    };
}

dispatch! {
    fn green_row => green_row_impl(
        rows: [&[f32]; 5], y: usize, h: usize, p: &DemosaicParams, out: &mut [[f32; 2]]
    );
    fn chroma_row => chroma_row_impl(
        rows: [&[[f32; 2]]; 3], y: usize, p: &DemosaicParams, out: &mut [f32]
    );
    fn chroma_row_planar => chroma_row_planar_impl(
        rows: [&[[f32; 2]]; 3], y: usize, p: &DemosaicParams, out: [&mut [f32]; 3]
    );
    fn matrix_rows => matrix_rows_impl(rgb: &mut [f32], m: &[f32; 9]);
    fn matrix_planes => matrix_planes_impl(rgb: [&mut [f32]; 3], m: &[f32; 9]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::BayerPattern;

    const PATTERNS: [BayerPattern; 4] = [
        BayerPattern::Rggb,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
        BayerPattern::Bggr,
    ];

    fn params(pattern: BayerPattern, wb: [f32; 2]) -> DemosaicParams {
        let range = 65279 - 256;
        DemosaicParams {
            cfa_phase: i32::from(pattern.code()),
            black: 256,
            one_over_range: 1.0 / range as f32,
            noise: [1e-6, 1e-9, 5e-9],
            clip: wb.map(|g| 1.0 / g),
            black_over_range: 256.0 / range as f32,
            wb,
        }
    }

    /// A mosaic with one linear value per colour.
    fn mosaic(pattern: BayerPattern, w: usize, h: usize, rgb: [u16; 3]) -> Vec<u16> {
        (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| match pattern.color_at(y, x) {
                    crate::header::Color::Red => rgb[0],
                    crate::header::Color::Green => rgb[1],
                    crate::header::Color::Blue => rgb[2],
                })
            })
            .collect()
    }

    #[test]
    fn flat_field_is_reproduced_exactly() {
        let (w, h) = (12, 10);
        for pattern in PATTERNS {
            let p = params(pattern, [1.0, 1.0]);
            let lin = mosaic(pattern, w, h, [20000; 3]);
            let v = p.one_over_range * (20000 - 256) as f32;
            let rgb = chroma_pass(&green_pass(&lin, w, h, &p), w, h, &p);
            assert!(rgb.iter().all(|&c| c == v), "{pattern:?}");
        }
    }

    #[test]
    fn colour_field_lands_on_the_right_channels() {
        let (w, h) = (16, 14);
        for pattern in PATTERNS {
            let p = params(pattern, [2.0, 1.5]);
            let raw = [12000u16, 30000, 16000];
            let lin = mosaic(pattern, w, h, raw);
            let rgb = chroma_pass(&green_pass(&lin, w, h, &p), w, h, &p);
            let norm = raw.map(|v| p.one_over_range * f32::from(v - 256));
            let want = [norm[0] * 2.0, norm[1], norm[2] * 1.5];
            for px in rgb.chunks_exact(3) {
                for c in 0..3 {
                    assert!(
                        (px[c] - want[c]).abs() < 1e-6,
                        "{pattern:?} {px:?} {want:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn red_and_blue_clip_where_white_balance_reaches_one() {
        let (w, h) = (8, 8);
        let p = params(BayerPattern::Rggb, [2.0, 4.0]);
        let lin = mosaic(BayerPattern::Rggb, w, h, [60000, 1000, 60000]);
        let rgb = chroma_pass(&green_pass(&lin, w, h, &p), w, h, &p);
        for px in rgb.chunks_exact(3) {
            assert_eq!((px[0], px[2]), (1.0, 1.0));
        }
    }

    #[test]
    fn noise_parameters_follow_the_quantizer() {
        let mut rng = crate::synth::Rng::new(3);
        let spec = crate::synth::Spec {
            valid_tiles: true,
            ..crate::synth::Spec::new(32, 16, BayerPattern::Rggb)
        };
        let packet = crate::synth::random_frame(&spec, &mut rng);
        let header = crate::parse_frame(&packet).unwrap().header;
        let p = DemosaicParams::with_white_balance(&header, 64, 2.0, 0.0);
        let range = 65023f64;
        let base = 5.0 * 64.0 * 64.0 + 64.0 / 3.0;
        assert_eq!(p.noise[0], (base / 150_994_944.0) as f32);
        assert_eq!(p.noise[1], (base / (range * range * 256.0)) as f32);
        assert_eq!(p.noise[2], (64.0 / 3.0 / (range * range)) as f32);
        assert_eq!(p.clip, [0.5, 1.0]);
        assert_eq!(p.wb, [2.0, 0.0]);
    }

    #[test]
    fn output_matrix_combines_gamut_matrix_and_gain() {
        let mut rng = crate::synth::Rng::new(4);
        let spec = crate::synth::Spec {
            gain: 2.0,
            ..crate::synth::Spec::new(32, 16, BayerPattern::Rggb)
        };
        let header = crate::parse_frame(&crate::synth::random_frame(&spec, &mut rng))
            .unwrap()
            .header;
        let reflection = output_matrix(&header, Exposure::Reflection);
        let normalized = output_matrix(&header, Exposure::Normalized);
        for i in 0..9 {
            let g = XYZ_TO_REC2020[i / 3][i % 3];
            assert_eq!(reflection[i], (g * 2.0) as f32);
            assert_eq!(normalized[i], (g * 2.0 * 10.0 / 9.0) as f32);
        }
    }

    #[test]
    fn quantizer_counts_every_tile_qscale_byte() {
        let mut rng = crate::synth::Rng::new(6);
        let spec = crate::synth::Spec {
            valid_tiles: true,
            quant_matrix: Some([3; 64]),
            ..crate::synth::Spec::new(64, 32, BayerPattern::Rggb)
        };
        let mut packet = crate::synth::random_frame(&spec, &mut rng);
        let parsed = crate::parse_frame(&packet).unwrap();
        let qscales: Vec<u8> = parsed.tiles.iter().map(|t| t[1]).collect();
        let max = *qscales.iter().max().unwrap();
        assert_eq!(frame_quantizer(&parsed), 3 * u32::from(max));
        // Make the first tile invalid (first stream size 0) with the
        // largest qscale: it is skipped by the decoder but still counts.
        let start = parsed.tiles[0].as_ptr() as usize - packet.as_ptr() as usize;
        packet[start + 1] = 255;
        packet[start + 2..start + 4].fill(0);
        let parsed = crate::parse_frame(&packet).unwrap();
        assert_eq!(frame_quantizer(&parsed), 3 * 255);
        let frame = crate::decode_frame(&packet).unwrap();
        assert_eq!(frame.skipped_tiles, [0]);
        assert_eq!(frame.quantizer(), 3 * 255);
        let reduced = crate::decode_frame_reduced(&packet, crate::Scale::Half).unwrap();
        assert_eq!(reduced.quantizer(), 3 * 255);
    }

    /// Each stage builds on the previous one. `Demosaiced` has no white
    /// balance: red and blue are neither scaled nor clipped early.
    #[test]
    fn develop_stages() {
        let mut rng = crate::synth::Rng::new(9);
        let spec = crate::synth::Spec {
            valid_tiles: true,
            wb: (2.0, 1.5),
            ..crate::synth::Spec::new(48, 32, BayerPattern::Rggb)
        };
        let packet = crate::synth::random_frame(&spec, &mut rng);
        let frame = crate::decode_frame(&packet).unwrap();
        let q = frame.quantizer();
        let img = develop(&frame, Stage::Demosaiced);
        let plain = DemosaicParams::with_white_balance(&frame.header, q, 1.0, 1.0);
        let (w, h) = (frame.width(), frame.height());
        let demosaiced = chroma_pass(&green_pass(&frame.linearize(), w, h, &plain), w, h, &plain);
        assert_eq!(img.data, demosaiced);
        let balanced = develop(&frame, Stage::WhiteBalanced).data;
        assert_ne!(img.data, balanced);
        let rec2020 = develop(&frame, Stage::Rec2020).data;
        let m = output_matrix(&frame.header, Exposure::Normalized);
        assert_eq!(rec2020, apply_matrix(&balanced, &m));
    }

    /// The planar and caller-owned forms carry the same values as the
    /// interleaved image, at every stage, including frames whose height is
    /// not a whole number of bands.
    #[test]
    fn planar_and_interleaved_outputs_agree() {
        let mut rng = crate::synth::Rng::new(10);
        let spec = crate::synth::Spec {
            valid_tiles: true,
            wb: (1.8, 1.3),
            ..crate::synth::Spec::new(80, 80, BayerPattern::Grbg)
        };
        let packet = crate::synth::random_frame(&spec, &mut rng);
        let frame = crate::decode_frame(&packet).unwrap();
        let params = DemosaicParams::new(&frame.header, frame.quantizer());
        let n = frame.width() * frame.height();
        assert!(frame.height() % BAND != 0);
        for stage in [Stage::Demosaiced, Stage::WhiteBalanced, Stage::Rec2020] {
            let image = develop_with(&frame, &params, stage);
            let mut reused = vec![f32::NAN; 3 * n];
            develop_into(&frame, &params, stage, &mut reused);
            assert_eq!(reused, image.data, "{stage:?}");
            let planar = develop_planar(&frame, &params, stage);
            assert_eq!((planar.width, planar.height), (image.width, image.height));
            for (c, plane) in planar.planes.iter().enumerate() {
                let want: Vec<f32> = image.data.iter().skip(c).step_by(3).copied().collect();
                assert_eq!(plane, &want, "{stage:?} channel {c}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "plane size")]
    fn planar_output_must_fit_the_frame() {
        let mut rng = crate::synth::Rng::new(11);
        let spec = crate::synth::Spec::new(32, 16, BayerPattern::Rggb);
        let frame = crate::decode_frame(&crate::synth::random_frame(&spec, &mut rng)).unwrap();
        let params = DemosaicParams::new(&frame.header, frame.quantizer());
        let (mut r, mut g, mut b) = (vec![0.0; 32 * 16], vec![0.0; 32 * 16], vec![0.0; 10]);
        develop_planar_into(&frame, &params, Stage::default(), [&mut r, &mut g, &mut b]);
    }

    #[test]
    fn develop_runs_on_a_decoded_frame() {
        let mut rng = crate::synth::Rng::new(5);
        let spec = crate::synth::Spec {
            valid_tiles: true,
            ..crate::synth::Spec::new(48, 32, BayerPattern::Bggr)
        };
        let packet = crate::synth::random_frame(&spec, &mut rng);
        let frame = crate::decode_frame(&packet).unwrap();
        assert!(frame.quantizer() > 0);
        let img = develop(&frame, Stage::default());
        assert_eq!(
            (img.width, img.height, img.data.len()),
            (48, 32, 48 * 32 * 3)
        );
        assert!(img.data.iter().all(|v| v.is_finite() && *v <= 1.0));
    }
}
