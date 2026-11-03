//! Dequantization and the integer inverse DCT (bitstream description §7).

const W1: i32 = 22725;
const W2: i32 = 21407;
const W3: i32 = 19265;
const W4: i32 = 16384;
const W5: i32 = 12873;
const W6: i32 = 8867;
const W7: i32 = 4520;
const ROW_SHIFT: u32 = 13;
const COL_SHIFT: u32 = 15;

/// Dequantize one raster-order block in place and replace it with the
/// centered sensel values of its inverse transform.
pub fn inverse_transform(block: &mut [i32; 64], qmat: &[i32; 64]) {
    for (c, &q) in block.iter_mut().zip(qmat) {
        *c = (*c * q).clamp(-32768, 32767);
    }
    for row in block.chunks_exact_mut(8) {
        transform_row(row);
    }
    for col in 0..8 {
        transform_column(block, col);
    }
}

/// Centered transform output to the transform-domain code value.
pub fn code_value(centered: i32) -> u16 {
    (centered + 32768).clamp(0, 65535) as u16
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

fn transform_row(row: &mut [i32]) {
    let x: [i32; 8] = row.try_into().expect("eight coefficients");
    let (a, b) = butterfly(x, 1 << (ROW_SHIFT - 1));
    row.copy_from_slice(&outputs(a, b, ROW_SHIFT));
}

fn transform_column(block: &mut [i32; 64], col: usize) {
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

#[cfg(test)]
mod tests {
    use super::*;

    const ONES: [i32; 64] = [1; 64];

    /// Float reference: the orthonormal 2-D IDCT scaled like the integer one.
    fn reference(coefs: &[i32; 64]) -> [f64; 64] {
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
                // The integer transform returns DC-only blocks at the DC value,
                // eight times the orthonormal transform.
                out[8 * y + x] = s * 2.0;
            }
        }
        out
    }

    #[test]
    fn dc_only_block_is_flat() {
        let mut b = [0i32; 64];
        b[0] = 800;
        inverse_transform(&mut b, &ONES);
        assert!(b.iter().all(|&v| v == b[0]), "{b:?}");
        assert_eq!(b[0], 800);
    }

    #[test]
    fn matches_float_reference_closely() {
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) as i32 % 2001) - 1000
        };
        for _ in 0..50 {
            let mut b = [0i32; 64];
            for v in &mut b {
                *v = next();
            }
            let expected = reference(&b);
            inverse_transform(&mut b, &ONES);
            for (got, want) in b.iter().zip(&expected) {
                assert!((f64::from(*got) - want).abs() <= 2.0, "{got} vs {want}");
            }
        }
    }

    #[test]
    fn dequantizes_with_saturation() {
        let mut b = [0i32; 64];
        b[0] = 3000;
        let mut q = ONES;
        q[0] = 20;
        inverse_transform(&mut b, &q);
        let mut sat = [0i32; 64];
        sat[0] = 32767;
        inverse_transform(&mut sat, &ONES);
        assert_eq!(b, sat);
    }

    #[test]
    fn code_values_are_centered_and_clamped() {
        assert_eq!(code_value(0), 32768);
        assert_eq!(code_value(-32768), 0);
        assert_eq!(code_value(-40000), 0);
        assert_eq!(code_value(32767), 65535);
        assert_eq!(code_value(40000), 65535);
    }
}
