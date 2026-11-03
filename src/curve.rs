//! The linearization curve (bitstream description §3.5 and §8).

/// Eight control points mapping transform-domain code values to linear
/// sensor values. Point `i` sits at code value `i * 8192`; an implicit
/// ninth point of 65536 sits at code value 65536.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Curve(pub [u16; 8]);

impl Curve {
    /// The curve used when the header carries none: one doubling per segment.
    pub const DEFAULT: Curve = Curve([0, 512, 1024, 2048, 4096, 8192, 16384, 32768]);

    /// Linear value of one code value.
    pub fn linearize(&self, code: u16) -> u16 {
        let seg = usize::from(code >> 13);
        let frac = u32::from(code & 0x1FFF);
        let c0 = u32::from(self.0[seg]);
        let c1 = self.0.get(seg + 1).map_or(65536, |&c| u32::from(c));
        // c1 may be below c0 for a non-monotonic curve; the format wraps the
        // difference modulo 2^16, so do the same.
        let delta = c1.wrapping_sub(c0) & 0xFFFF;
        let out = (c0 * 8192 + delta * frac + 4096) >> 13;
        out.min(65535) as u16
    }

    /// Lookup table of every code value, for bulk linearization.
    pub fn lut(&self) -> Box<[u16; 65536]> {
        let mut lut = vec![0u16; 65536].into_boxed_slice();
        for (code, v) in lut.iter_mut().enumerate() {
            *v = self.linearize(code as u16);
        }
        lut.try_into().expect("65536 entries")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_curve_hits_control_points() {
        for (i, &p) in Curve::DEFAULT.0.iter().enumerate() {
            assert_eq!(Curve::DEFAULT.linearize((i * 8192) as u16), p);
        }
        assert_eq!(Curve::DEFAULT.linearize(65535), 65532);
    }

    #[test]
    fn interpolates_linearly_within_a_segment() {
        let c = Curve([0, 100, 200, 300, 400, 500, 600, 700]);
        assert_eq!(c.linearize(4096), 50);
        assert_eq!(c.linearize(8192 + 2048), 125);
        assert_eq!(u32::from(c.linearize(57344 + 4096)), (700 + 65536) / 2);
    }

    #[test]
    fn rounds_to_nearest() {
        let c = Curve([0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(c.linearize(4095), 0);
        assert_eq!(c.linearize(4096), 1);
    }

    #[test]
    fn lut_matches_pointwise() {
        let c = Curve([0, 224, 288, 468, 1162, 3339, 9431, 25377]);
        let lut = c.lut();
        for code in [0u16, 1, 8191, 8192, 30000, 65535] {
            assert_eq!(lut[usize::from(code)], c.linearize(code));
        }
    }
}
