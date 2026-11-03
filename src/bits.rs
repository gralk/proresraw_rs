//! MSB-first bit reader with the 32-bit zero-filled look-ahead window that
//! the entropy code is defined on (bitstream description §6.1).

use crate::codebook::Codebook;

/// Reads codewords from a component stream.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`; may run past the end.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Reader positioned at the first bit of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// The next 32 bits, zero-filled past the end of the data.
    fn window(&self) -> u32 {
        let byte = self.pos >> 3;
        let mut buf = [0u8; 8];
        if let Some(src) = self.data.get(byte..) {
            let n = src.len().min(8);
            buf[..n].copy_from_slice(&src[..n]);
        }
        let shifted = u64::from_be_bytes(buf) << (self.pos & 7);
        (shifted >> 32) as u32
    }

    /// Read one bit; zero past the end of the data.
    pub fn read_bit(&mut self) -> u32 {
        let bit = self.window() >> 31;
        self.pos += 1;
        bit
    }

    /// Decode one symbol of the hybrid code, or `None` at the end of the
    /// component: an all-zero window or a codeword longer than the window.
    pub fn read_symbol(&mut self, cb: Codebook) -> Option<u32> {
        let w = self.window();
        if w == 0 {
            return None;
        }
        let q = w.leading_zeros();
        let rice = cb.rice_order();
        if q <= cb.switch() {
            let after_one = (u64::from(w) << (q + 1)) & 0xFFFF_FFFF;
            let tail = (after_one >> (32 - rice)) as u32;
            self.pos += (q + 1 + rice) as usize;
            return Some((q << rice) | tail);
        }
        let exp = cb.exp_order();
        let len = 2 * q + exp - cb.switch();
        if len > 32 {
            return None;
        }
        self.pos += len as usize;
        let prefix = (u64::from(w) >> (32 - len)) as u32;
        // A leading one at bit `len - q - 1` guarantees prefix >= 1 << exp.
        Some(prefix + ((cb.switch() + 1) << rice) - (1 << exp))
    }
}

/// Test-only encoder of the hybrid code, written from the description.
#[cfg(test)]
pub(crate) mod testing {
    use crate::codebook::Codebook;

    pub(crate) struct BitWriter {
        bits: Vec<bool>,
    }

    impl BitWriter {
        pub fn new() -> Self {
            BitWriter { bits: Vec::new() }
        }

        pub fn put(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.bits.push((value >> i) & 1 == 1);
            }
        }

        /// Append the codeword of `v` and return its length in bits.
        pub fn symbol(&mut self, cb: Codebook, v: u32) -> u32 {
            let (s, e, r) = (cb.switch(), cb.exp_order(), cb.rice_order());
            let q = v >> r;
            if q <= s {
                self.put(0, q);
                self.put(1, 1);
                self.put(v & ((1 << r) - 1), r);
                q + 1 + r
            } else {
                let rest = v - ((s + 1) << r);
                let m = rest + (1 << e);
                let bits = 32 - m.leading_zeros();
                let zeros = s + 1 + (bits - 1 - e);
                self.put(0, zeros);
                self.put(m, bits);
                zeros + bits
            }
        }

        pub fn finish(self) -> Vec<u8> {
            let mut out = vec![0u8; self.bits.len().div_ceil(8)];
            for (i, &b) in self.bits.iter().enumerate() {
                if b {
                    out[i / 8] |= 0x80 >> (i % 8);
                }
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use testing::BitWriter;

    #[test]
    fn rice_codewords() {
        // s=1, e=3, r=2: "1" + 2 bits -> 0..3, "01" + 2 bits -> 4..7.
        let cb = Codebook(0x132);
        let mut r = BitReader::new(&[0b1110_1100]);
        assert_eq!(r.read_symbol(cb), Some(0b11));
        assert_eq!(r.read_symbol(cb), Some(4 + 0b10));
    }

    #[test]
    fn exp_golomb_codewords() {
        // s=0, e=3, r=1: values >= 2 use exp-Golomb(3) + 2.
        let cb = Codebook(0x031);
        // q=1: len = 2*1 + 3 - 0 = 5 bits "01abc" -> abc + 2 - 8 + 8.
        let mut r = BitReader::new(&[0b0110_1000]);
        assert_eq!(r.read_symbol(cb), Some(0b01101 + 2 - 8));
    }

    #[test]
    fn round_trips_through_the_test_encoder() {
        let cbs = [
            Codebook(0x2BC),
            Codebook(0x064),
            Codebook(0x031),
            Codebook(0),
            Codebook(0x433),
        ];
        let values = [0u32, 1, 2, 3, 7, 8, 15, 16, 100, 1000, 4095, 65535];
        for &cb in &cbs {
            for &v in &values {
                let mut w = BitWriter::new();
                let len = w.symbol(cb, v);
                let data = w.finish();
                let mut r = BitReader::new(&data);
                // Codewords longer than the 32-bit window are not valid and
                // read as the end of the component.
                let expected = (len <= 32).then_some(v);
                assert_eq!(r.read_symbol(cb), expected, "{cb:?} {v}");
            }
        }
    }

    #[test]
    fn ends_on_zero_window_and_past_the_end() {
        let mut r = BitReader::new(&[0, 0, 0, 0, 0x80]);
        assert_eq!(r.read_symbol(Codebook(0)), None);
        let mut r = BitReader::new(&[0b1000_0000]);
        assert_eq!(r.read_symbol(Codebook(0)), Some(0));
        assert_eq!(r.read_bit(), 0);
        assert_eq!(r.read_symbol(Codebook(0)), None);
        let mut r = BitReader::new(&[]);
        assert_eq!(r.read_symbol(Codebook(0)), None);
    }

    #[test]
    fn overlong_codeword_ends_the_component() {
        // 20 zeros then a one: q = 20 > s, len = 40 + 0 - 0 > 32.
        let mut r = BitReader::new(&[0, 0, 0b0000_1000, 0, 0, 0]);
        assert_eq!(r.read_symbol(Codebook(0)), None);
    }

    #[test]
    fn zero_fill_completes_a_trailing_codeword() {
        // s=0, r=4: "1" followed by only two data bits; the rest is zero fill.
        let mut r = BitReader::new(&[0b0000_0100]);
        r.pos = 5;
        assert_eq!(r.read_symbol(Codebook(0x004)), Some(0));
        assert_eq!(r.read_symbol(Codebook(0x004)), None);
    }
}
