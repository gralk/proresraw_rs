//! Dequantization and the inverse DCT (bitstream description §7).
//!
//! Two transforms are provided:
//!
//! - [`Idct::Apple`] (the default) is the Apple-compatible transform: an
//!   integer row pass with per-row prescaled tables and no intermediate
//!   rounding, followed by a column pass in `f32` (tan/cos factorization)
//!   whose output is truncated after adding `32768.5`. It reproduces
//!   Apple's decoder output bit for bit.
//! - [`Idct::Ffmpeg`] is FFmpeg's "simple IDCT", which differs from the
//!   Apple-compatible transform by ±1–3 code values on a fraction of the
//!   samples of blocks with AC energy. It is kept to compare against
//!   FFmpeg's decoder.
//!
//! Both take int16 coefficients and an int16 quantizer table and return the
//! transform-domain code values (centred values + 32768, clamped).

/// Which inverse transform to run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Idct {
    /// Apple-compatible (bit-exact with Apple's decoder).
    #[default]
    Apple,
    /// FFmpeg's simple IDCT, bit-exact with FFmpeg's `prores_raw` decoder.
    Ffmpeg,
}

impl Idct {
    /// Dequantize one raster-order block with `q` and return the code
    /// values of its inverse transform, in raster order.
    pub fn transform(self, coefs: &[i16; 64], q: &[i16; 64]) -> [u16; 64] {
        match self {
            Idct::Apple => apple_block(coefs, q),
            Idct::Ffmpeg => ffmpeg_block(coefs, q),
        }
    }
}

/// Dequantized coefficient: the 32-bit product saturated to 16 bits.
#[inline(always)]
pub fn dequantize(coef: i16, q: i16) -> i16 {
    (i32::from(coef) * i32::from(q)).clamp(-32768, 32767) as i16
}

// ---------------------------------------------------------------------------
// Apple-compatible transform

/// Row-pass tables: the 8-point DCT basis prescaled per row. Set `s` holds
/// four tables applied to the coefficient pairs (x0,x2), (x4,x6), (x1,x3)
/// and (x5,x7); entry `2k`, `2k+1` weights the pair for output lane `k`.
pub(crate) const ROW_TABLES: [[[i32; 8]; 4]; 4] = [
    [
        [4096, 5352, 4096, 2217, 4096, -2217, 4096, -5352],
        [4096, 2217, -4096, -5352, -4096, 5352, 4096, -2217],
        [5681, 4816, 4816, -1130, 3218, -5681, 1130, -3218],
        [3218, 1130, -5681, -3218, 1130, 4816, 4816, -5681],
    ],
    [
        [5681, 7423, 5681, 3075, 5681, -3075, 5681, -7423],
        [5681, 3075, -5681, -7423, -5681, 7423, 5681, -3075],
        [7880, 6681, 6681, -1567, 4464, -7880, 1567, -4464],
        [4464, 1567, -7880, -4464, 1567, 6681, 6681, -7880],
    ],
    [
        [5352, 6992, 5352, 2896, 5352, -2896, 5352, -6992],
        [5352, 2896, -5352, -6992, -5352, 6992, 5352, -2896],
        [7423, 6293, 6293, -1477, 4205, -7423, 1477, -4205],
        [4205, 1477, -7423, -4205, 1477, 6293, 6293, -7423],
    ],
    [
        [4816, 6293, 4816, 2607, 4816, -2607, 4816, -6293],
        [4816, 2607, -4816, -6293, -4816, 6293, 4816, -2607],
        [6681, 5663, 5663, -1329, 3784, -6681, 1329, -3784],
        [3784, 1329, -6681, -3784, 1329, 5663, 5663, -6681],
    ],
];

/// Table set used by each coefficient row.
pub(crate) const ROW_SET: [usize; 8] = [0, 1, 2, 3, 0, 3, 2, 1];

/// Column-pass constants, as exact `f32` bit patterns.
pub(crate) const SCALE: f32 = f32::from_bits(0x3980_0000); // 2^-12
pub(crate) const BIAS: f32 = f32::from_bits(0x4700_0080); // 32768.5
pub(crate) const TAN1: f32 = f32::from_bits(0x3e4b_afaf); // tan(π/16)
pub(crate) const TAN2: f32 = f32::from_bits(0x3ed4_13cd); // tan(π/8)
pub(crate) const TAN3: f32 = f32::from_bits(0x3f2b_0dc1); // tan(3π/16)
pub(crate) const COS4: f32 = f32::from_bits(0x3f35_04f3); // cos(π/4)

/// One row of the integer pass: eight int32 outputs, no rounding or shift.
/// The products and sums cannot overflow for int16 input
/// (|even| + |odd| < 1.4·10⁹).
#[inline(always)]
fn row_pass(x: &[i16], t: &[[i32; 8]; 4]) -> [i32; 8] {
    let x: [i32; 8] = std::array::from_fn(|i| i32::from(x[i]));
    let pairs = [(x[0], x[2]), (x[4], x[6]), (x[1], x[3]), (x[5], x[7])];
    let dot = |j: usize, k: usize| pairs[j].0 * t[j][2 * k] + pairs[j].1 * t[j][2 * k + 1];
    let mut out = [0i32; 8];
    for k in 0..4 {
        let even = dot(0, k) + dot(1, k);
        let odd = dot(2, k) + dot(3, k);
        out[k] = even + odd;
        out[7 - k] = even - odd;
    }
    out
}

/// Truncate toward zero; NaN and out-of-range values give `i32::MIN`.
#[inline(always)]
fn trunc_to_i32(v: f32) -> i32 {
    if (-2147483648.0..2147483648.0).contains(&v) {
        v as i32
    } else {
        i32::MIN
    }
}

/// Biased column output to a code value: truncate, re-centre, saturate to
/// 16 bits and offset back to unsigned.
#[inline(always)]
fn output_code(v: f32) -> u16 {
    let centred = trunc_to_i32(v).wrapping_sub(32768).clamp(-32768, 32767);
    (centred as i16 as u16) ^ 0x8000
}

/// Apple-compatible dequantize + 8×8 inverse transform of one raster-order block.
pub fn apple_block(coefs: &[i16; 64], q: &[i16; 64]) -> [u16; 64] {
    let mut rows = [[0f32; 8]; 8];
    for (r, row) in rows.iter_mut().enumerate() {
        let d: [i16; 8] = std::array::from_fn(|c| dequantize(coefs[8 * r + c], q[8 * r + c]));
        let y = row_pass(&d, &ROW_TABLES[ROW_SET[r]]);
        for (o, v) in row.iter_mut().zip(y) {
            *o = v as f32 * SCALE;
        }
    }
    let mut out = [0u16; 64];
    for c in 0..8 {
        let r: [f32; 8] = std::array::from_fn(|i| rows[i][c]);
        // Even part. The bias rides on the DC term from the start.
        let p = r[0] + BIAS;
        let ea = p + r[4];
        let eb = p - r[4];
        let f = r[6] * TAN2 + r[2];
        let g = r[2] * TAN2 - r[6];
        let e0 = ea + f;
        let e3 = ea - f;
        let e1 = eb + g;
        let e2 = eb - g;
        // Odd part.
        let a = r[7] * TAN1 + r[1];
        let b = r[1] * TAN1 - r[7];
        let cc = r[5] * TAN3 + r[3];
        let d = r[5] - r[3] * TAN3;
        let o0 = a + cc;
        let t = a - cc;
        let o3 = b + d;
        let u = b - d;
        let o1 = (t + u) * COS4;
        let o2 = (t - u) * COS4;
        let y = [
            e0 + o0,
            e1 + o1,
            e2 + o2,
            e3 + o3,
            e3 - o3,
            e2 - o2,
            e1 - o1,
            e0 - o0,
        ];
        for (i, v) in y.into_iter().enumerate() {
            out[8 * i + c] = output_code(v);
        }
    }
    out
}

/// Row tables of the 4-point transform for even and odd coefficient rows:
/// the weight of (x0 ± x2) in the even part, then the weights of x1 and
/// x3 in the two odd outputs.
const ROW4_TABLES: [(i32, [i32; 4]); 2] = [
    (8192, [10703, 4433, 4433, -10703]),
    (10703, [13985, 5793, 5793, -13985]),
];

/// Column-pass scale of the 4-point transform: 2^-13.
const SCALE4: f32 = f32::from_bits(0x3900_0000);

/// Saturate a centred value to 16 bits and offset it to a code value.
#[inline(always)]
fn saturate_code(v: i32) -> u16 {
    (v.clamp(-32768, 32767) as i16 as u16) ^ 0x8000
}

/// Apple-compatible dequantize + 4×4 inverse transform, used for
/// quarter-resolution decoding. `coefs` is the top-left 4×4 corner of a
/// block in raster order; `q` is the full 8×8 quantizer table, of which
/// the matching corner is used. Returns 16 code values in raster order.
pub fn apple_block_4x4(coefs: &[i16; 16], q: &[i16; 64]) -> [u16; 16] {
    let mut rows = [[0f32; 4]; 4];
    for (r, row) in rows.iter_mut().enumerate() {
        let x: [i32; 4] =
            std::array::from_fn(|c| i32::from(dequantize(coefs[4 * r + c], q[8 * r + c])));
        let (k, t) = ROW4_TABLES[r & 1];
        let e0 = k * x[0] + k * x[2];
        let e1 = k * x[0] - k * x[2];
        let o0 = t[0] * x[1] + t[1] * x[3];
        let o1 = t[2] * x[1] + t[3] * x[3];
        let y = [e0 + o0, e1 + o1, e1 - o1, e0 - o0];
        for (o, v) in row.iter_mut().zip(y) {
            *o = v as f32 * SCALE4;
        }
    }
    let mut out = [0u16; 16];
    for c in 0..4 {
        let r0 = rows[0][c] + BIAS;
        let (r1, r2, r3) = (rows[1][c], rows[2][c], rows[3][c]);
        let e0 = r2 + r0;
        let e1 = r0 - r2;
        let f = r3 * TAN2 + r1;
        let g = r1 * TAN2 - r3;
        let y = [f + e0, g + e1, e1 - g, e0 - f];
        for (i, v) in y.into_iter().enumerate() {
            out[4 * i + c] = output_code(v);
        }
    }
    out
}

/// Apple-compatible dequantize + 2×2 inverse transform (sums and
/// differences, no scaling), used for eighth-resolution decoding. `coefs`
/// is the top-left 2×2 corner of a block in raster order; `q` is the full
/// 8×8 quantizer table. Returns 4 code values in raster order.
pub fn apple_block_2x2(coefs: &[i16; 4], q: &[i16; 64]) -> [u16; 4] {
    let d = [
        i32::from(dequantize(coefs[0], q[0])),
        i32::from(dequantize(coefs[1], q[1])),
        i32::from(dequantize(coefs[2], q[8])),
        i32::from(dequantize(coefs[3], q[9])),
    ];
    let (top_sum, top_diff) = (d[0] + d[1], d[0] - d[1]);
    let (bottom_sum, bottom_diff) = (d[2] + d[3], d[2] - d[3]);
    [
        saturate_code(top_sum + bottom_sum),
        saturate_code(top_diff + bottom_diff),
        saturate_code(top_sum - bottom_sum),
        saturate_code(top_diff - bottom_diff),
    ]
}

// ---------------------------------------------------------------------------
// FFmpeg simple IDCT

const W1: i32 = 22725;
const W2: i32 = 21407;
const W3: i32 = 19265;
const W4: i32 = 16384;
const W5: i32 = 12873;
const W6: i32 = 8867;
const W7: i32 = 4520;
const ROW_SHIFT: u32 = 13;
const COL_SHIFT: u32 = 15;

/// Centered transform output to the transform-domain code value.
pub fn code_value(centered: i32) -> u16 {
    (centered + 32768).clamp(0, 65535) as u16
}

/// FFmpeg's dequantize + simple IDCT of one raster-order block.
pub fn ffmpeg_block(coefs: &[i16; 64], q: &[i16; 64]) -> [u16; 64] {
    let mut block: [i32; 64] = std::array::from_fn(|i| i32::from(dequantize(coefs[i], q[i])));
    for row in block.chunks_exact_mut(8) {
        ffmpeg_row(row);
    }
    for col in 0..8 {
        ffmpeg_column(&mut block, col);
    }
    block.map(code_value)
}

/// Butterfly shared by both passes: even part from x0, x2, x4, x6 and odd
/// part from x1, x3, x5, x7, with `round` already folded into the DC term.
#[inline(always)]
fn butterfly(x: [i32; 8], round: i32) -> ([i32; 4], [i32; 4]) {
    let w = |k: i32, v: i32| k.wrapping_mul(v);
    let base = w(W4, x[0]).wrapping_add(round);
    let a = [
        base.wrapping_add(w(W2, x[2]))
            .wrapping_add(w(W4, x[4]))
            .wrapping_add(w(W6, x[6])),
        base.wrapping_add(w(W6, x[2]))
            .wrapping_sub(w(W4, x[4]))
            .wrapping_sub(w(W2, x[6])),
        base.wrapping_sub(w(W6, x[2]))
            .wrapping_sub(w(W4, x[4]))
            .wrapping_add(w(W2, x[6])),
        base.wrapping_sub(w(W2, x[2]))
            .wrapping_add(w(W4, x[4]))
            .wrapping_sub(w(W6, x[6])),
    ];
    let b = [
        w(W1, x[1])
            .wrapping_add(w(W3, x[3]))
            .wrapping_add(w(W5, x[5]))
            .wrapping_add(w(W7, x[7])),
        w(W3, x[1])
            .wrapping_sub(w(W7, x[3]))
            .wrapping_sub(w(W1, x[5]))
            .wrapping_sub(w(W5, x[7])),
        w(W5, x[1])
            .wrapping_sub(w(W1, x[3]))
            .wrapping_add(w(W7, x[5]))
            .wrapping_add(w(W3, x[7])),
        w(W7, x[1])
            .wrapping_sub(w(W5, x[3]))
            .wrapping_add(w(W3, x[5]))
            .wrapping_sub(w(W1, x[7])),
    ];
    (a, b)
}

#[inline(always)]
fn outputs(a: [i32; 4], b: [i32; 4], shift: u32) -> [i32; 8] {
    [
        a[0].wrapping_add(b[0]) >> shift,
        a[1].wrapping_add(b[1]) >> shift,
        a[2].wrapping_add(b[2]) >> shift,
        a[3].wrapping_add(b[3]) >> shift,
        a[3].wrapping_sub(b[3]) >> shift,
        a[2].wrapping_sub(b[2]) >> shift,
        a[1].wrapping_sub(b[1]) >> shift,
        a[0].wrapping_sub(b[0]) >> shift,
    ]
}

fn ffmpeg_row(row: &mut [i32]) {
    let x: [i32; 8] = row.try_into().expect("eight coefficients");
    let (a, b) = butterfly(x, 1 << (ROW_SHIFT - 1));
    row.copy_from_slice(&outputs(a, b, ROW_SHIFT));
}

fn ffmpeg_column(block: &mut [i32; 64], col: usize) {
    let mut x = [0i32; 8];
    for (i, v) in x.iter_mut().enumerate() {
        *v = block[8 * i + col];
    }
    // The rounding term is (1 << (COL_SHIFT - 1)) / W4 = 1, folded into x0.
    x[0] = x[0].wrapping_add(1);
    let (a, b) = butterfly(x, 0);
    for (i, v) in outputs(a, b, COL_SHIFT).into_iter().enumerate() {
        block[8 * i + col] = v;
    }
}

// ---------------------------------------------------------------------------
// Block generator shared by the tests and external test tooling.

/// Deterministic generator of (coefficients, quantizers) test blocks.
/// Exposed through the `testgen` feature so the reference data in
/// `tests/data/idct_*` can be regenerated; changing it invalidates them.
#[cfg(any(test, feature = "testgen"))]
#[doc(hidden)]
pub mod testgen {
    /// SplitMix64 stream.
    pub struct Gen(u64);

    impl Gen {
        /// Number of block kinds; block `i` is of kind `i % KINDS`.
        pub const KINDS: u64 = 6;

        /// Generator with the given seed.
        pub fn new(seed: u64) -> Self {
            Gen(seed)
        }

        /// Next raw 64-bit value.
        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }

        fn signed(&mut self, m: i32) -> i16 {
            (self.below(2 * m as u64 + 1) as i32 - m) as i16
        }

        /// Quantizers as a stream produces them: qmat 1..=`qmat` × qscale 1..=`qscale`.
        fn q(&mut self, qmat: u64, qscale: u64) -> [i16; 64] {
            let s = 1 + self.below(qscale);
            std::array::from_fn(|_| ((1 + self.below(qmat)) * s) as i16)
        }

        /// Block `i` of the stream.
        pub fn block(&mut self, i: u64) -> ([i16; 64], [i16; 64]) {
            let mut c = [0i16; 64];
            let q;
            match i % Self::KINDS {
                // DC only.
                0 => {
                    q = self.q(63, 255);
                    c[0] = self.next_u64() as i16;
                }
                // Sparse AC, often saturating.
                1 => {
                    q = self.q(63, 255);
                    c[0] = self.signed(2047);
                    for _ in 0..1 + self.below(9) {
                        let k = self.below(64) as usize;
                        c[k] = self.signed(2047);
                    }
                }
                // Dense, light quantization, little saturation.
                2 => {
                    q = self.q(16, 8);
                    c = std::array::from_fn(|_| self.signed(255));
                }
                // Dense with energy decaying away from DC.
                3 => {
                    q = self.q(8, 4);
                    c = std::array::from_fn(|k| {
                        let m = 4096 >> ((k / 8 + k % 8) / 2).min(12);
                        self.signed(m)
                    });
                }
                // Any coefficients, stream quantizers.
                4 => {
                    q = self.q(63, 255);
                    c = std::array::from_fn(|_| self.next_u64() as i16);
                }
                // Any coefficients and any quantizers.
                _ => {
                    q = std::array::from_fn(|_| self.next_u64() as i16);
                    c = std::array::from_fn(|_| self.next_u64() as i16);
                }
            }
            (c, q)
        }
    }

    /// FNV-1a over little-endian u16 outputs.
    pub mod fnv1a {
        /// Initial hash value.
        pub const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

        /// Fold one output block into the hash.
        pub fn update(mut h: u64, block: &[u16; 64]) -> u64 {
            for v in block {
                for b in v.to_le_bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            h
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONES: [i16; 64] = [1; 64];

    /// Float reference: the orthonormal 2-D IDCT scaled like the integer one.
    fn reference(coefs: &[i16; 64]) -> [f64; 64] {
        let mut out = [0.0; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut s = 0.0;
                for v in 0..8 {
                    for u in 0..8 {
                        let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                        let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                        s += cu
                            * cv
                            * f64::from(coefs[8 * v + u])
                            * ((2.0 * x as f64 + 1.0) * u as f64 * std::f64::consts::PI / 16.0)
                                .cos()
                            * ((2.0 * y as f64 + 1.0) * v as f64 * std::f64::consts::PI / 16.0)
                                .cos();
                    }
                }
                // DC-only blocks return the DC value: eight times the
                // orthonormal transform.
                out[8 * y + x] = s * 2.0 + 32768.0;
            }
        }
        out
    }

    fn random_blocks(n: usize) -> Vec<[i16; 64]> {
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) as i32 % 2001 - 1000) as i16
        };
        (0..n).map(|_| std::array::from_fn(|_| next())).collect()
    }

    #[test]
    fn dc_only_block_is_flat() {
        for idct in [Idct::Apple, Idct::Ffmpeg] {
            for dc in [-32768i16, -5000, -1, 0, 1, 800, 32767] {
                let mut b = [0i16; 64];
                b[0] = dc;
                let out = idct.transform(&b, &ONES);
                assert!(
                    out.iter().all(|&v| i32::from(v) == i32::from(dc) + 32768),
                    "{idct:?} {dc}"
                );
            }
        }
    }

    #[test]
    fn both_transforms_match_the_float_reference_closely() {
        for b in random_blocks(50) {
            let expected = reference(&b);
            for (idct, tol) in [(Idct::Apple, 3.0), (Idct::Ffmpeg, 2.0)] {
                for (got, want) in idct.transform(&b, &ONES).iter().zip(&expected) {
                    assert!(
                        (f64::from(*got) - want).abs() <= tol,
                        "{idct:?} {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn dequantizes_with_saturation() {
        let mut b = [0i16; 64];
        b[0] = 3000;
        let mut q = ONES;
        q[0] = 20;
        let mut sat = [0i16; 64];
        sat[0] = 32767;
        for idct in [Idct::Apple, Idct::Ffmpeg] {
            assert_eq!(idct.transform(&b, &q), idct.transform(&sat, &ONES));
        }
        assert_eq!(dequantize(-32768, -1), 32767);
        assert_eq!(dequantize(-300, 300), -32768);
    }

    #[test]
    fn code_values_are_centered_and_clamped() {
        assert_eq!(code_value(0), 32768);
        assert_eq!(code_value(-32768), 0);
        assert_eq!(code_value(-40000), 0);
        assert_eq!(code_value(32767), 65535);
        assert_eq!(code_value(40000), 65535);
        assert_eq!(output_code(-0.9), 0);
        assert_eq!(output_code(32768.5), 32768);
        assert_eq!(output_code(70000.0), 65535);
        assert_eq!(output_code(f32::NAN), 65535);
    }

    /// Reference records: coefficients, quantizers, expected code values
    /// (little-endian i16, i16, u16; 64 each).
    #[test]
    fn apple_matches_reference_vectors() {
        let data = include_bytes!("../tests/data/idct_vectors.bin");
        assert_eq!(data.len() % 384, 0);
        let words = |b: &[u8]| -> Vec<u16> {
            b.chunks_exact(2)
                .map(|w| u16::from_le_bytes([w[0], w[1]]))
                .collect()
        };
        for (i, rec) in data.chunks_exact(384).enumerate() {
            let c: [i16; 64] = std::array::from_fn(|k| words(&rec[..128])[k] as i16);
            let q: [i16; 64] = std::array::from_fn(|k| words(&rec[128..256])[k] as i16);
            let want: [u16; 64] = words(&rec[256..]).try_into().unwrap();
            assert_eq!(apple_block(&c, &q), want, "record {i}");
        }
    }

    /// Reference FNV-1a hash of the expected output of 2^20 generated
    /// blocks (seed 0x4a54).
    #[test]
    fn apple_matches_reference_hash() {
        let expected = include_str!("../tests/data/idct_hash.txt");
        let expected = u64::from_str_radix(expected.trim().trim_start_matches("0x"), 16).unwrap();
        let mut g = testgen::Gen::new(0x4a54);
        let mut h = testgen::fnv1a::OFFSET;
        for i in 0..1 << 20 {
            let (c, q) = g.block(i);
            h = testgen::fnv1a::update(h, &apple_block(&c, &q));
        }
        assert_eq!(h, expected);
    }
}
