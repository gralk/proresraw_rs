//! MSB-first bit reader and codeword decoder (bitstream description §6.1),
//! Apple-compatible down to malformed streams.
//!
//! The reader keeps a 64-bit window whose top 32 bits are the next bits of
//! the stream, plus a count of the valid bits below them, and refills it
//! with one big-endian 32-bit word whenever that count goes negative. Past
//! the end of the stream the refills are zero. For every codeword of at
//! most 32 bits this is the same as reading the stream bit by bit with zero
//! fill. Longer codewords are invalid; for them the shifts wrap (modulo 64
//! for the window, modulo 32 for the view), which is what Apple's decoder
//! produces for such streams.

use crate::codebook::Codebook;

/// Reads codewords from one component stream.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte to load.
    pos: usize,
    /// Whole 32-bit words may be loaded while `pos < words_end`.
    words_end: usize,
    /// Bits of the stream, MSB first; the top 32 are the current view.
    window: u64,
    /// Valid bits in the window beyond the top 32. Negative after an
    /// over-long codeword, after which the window no longer follows the stream.
    extra: i32,
}

/// Up to four bytes as the top bytes of a big-endian word.
fn partial_word(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .take(4)
        .enumerate()
        .fold(0, |w, (i, &b)| w | (u32::from(b) << (24 - 8 * i)))
}

impl<'a> BitReader<'a> {
    /// Reader over `data`, which starts `align` (0..=3) bytes past a 4-byte
    /// boundary. The bytes up to the next boundary after the first word are
    /// loaded separately so that later refills are whole aligned words; this
    /// only affects the result once an invalid codeword has been read.
    pub fn new(data: &'a [u8], align: usize) -> Self {
        let mut pos;
        let mut remaining = data.len();
        let first;
        let mut address = align;
        if remaining < 4 {
            first = partial_word(data);
            pos = remaining;
            address += remaining;
            remaining = 0;
        } else {
            first = u32::from_be_bytes(data[..4].try_into().expect("four bytes"));
            pos = 4;
            address += 4;
            remaining -= 4;
        }
        let mut low = 0u32;
        let mut extra = 0;
        let to_boundary = (4 - (address & 3)) & 3;
        if to_boundary != 0 && to_boundary <= remaining {
            low = partial_word(&data[pos..pos + to_boundary]);
            pos += to_boundary;
            remaining -= to_boundary;
            extra = 8 * to_boundary as i32;
        }
        BitReader {
            data,
            pos,
            words_end: pos + 4 * (remaining / 4),
            window: (u64::from(first) << 32) | u64::from(low),
            extra,
        }
    }

    /// Next 32-bit refill: a whole word, else the zero-filled tail.
    fn next_word(&mut self) -> u32 {
        if self.pos < self.words_end {
            let w = u32::from_be_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
            self.pos += 4;
            w
        } else {
            let w = partial_word(&self.data[self.pos..]);
            self.pos = self.data.len();
            w
        }
    }

    /// The top 32 bits of the window. Zero marks the end of a component.
    pub fn top(&self) -> u32 {
        (self.window >> 32) as u32
    }

    /// Drop `len` bits, refilling at most once.
    pub fn consume(&mut self, len: u32) {
        self.window = self.window.wrapping_shl(len);
        self.extra = self.extra.wrapping_sub(len as i32);
        if self.extra < 0 {
            let shift = (-i32::from(self.extra as i8)) as u32 & 0x3f;
            self.window |= u64::from(self.next_word()) << shift;
            self.extra += 32;
        }
    }

    /// Decode the codeword at the top of a non-zero window: its value and
    /// length in bits, without consuming it.
    pub fn peek_symbol(&self, cb: Codebook) -> (u32, u32) {
        codeword(self.top(), cb)
    }
}

/// Codeword decoder for a non-zero 32-bit view `p`: value and
/// length. Lengths above 32 are invalid; the shifts then wrap mod 32.
pub fn codeword(p: u32, cb: Codebook) -> (u32, u32) {
    debug_assert_ne!(p, 0);
    let lz = p.leading_zeros();
    let s = cb.switch();
    let r = cb.rice_order();
    if (p as i32) < 0 {
        return ((p & 0x7fff_ffff) >> (31u32.wrapping_sub(r) & 31), r + 1);
    }
    if lz <= s {
        let tail = ((p << ((lz + 1) & 31)) >> 1) >> (31u32.wrapping_sub(r) & 31);
        return ((lz << r).wrapping_add(tail), r + 1 + lz);
    }
    let e = cb.exp_order();
    let len = (2 * lz).wrapping_sub(s).wrapping_add(e);
    let value = ((s + 1) << r)
        .wrapping_add(p >> (32u32.wrapping_sub(len) & 31))
        .wrapping_sub(1 << e);
    (value, len)
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

    fn read(r: &mut BitReader<'_>, cb: Codebook) -> Option<u32> {
        if r.top() == 0 {
            return None;
        }
        let (v, len) = r.peek_symbol(cb);
        r.consume(len);
        Some(v)
    }

    #[test]
    fn rice_codewords() {
        // s=1, e=3, r=2: "1" + 2 bits -> 0..3, "01" + 2 bits -> 4..7.
        let cb = Codebook(0x132);
        let mut r = BitReader::new(&[0b1110_1100], 0);
        assert_eq!(read(&mut r, cb), Some(0b11));
        assert_eq!(read(&mut r, cb), Some(4 + 0b10));
    }

    #[test]
    fn exp_golomb_codewords() {
        // s=0, e=3, r=1: q=1: len = 2*1 + 3 - 0 = 5 bits "01abc" -> abc + 2 - 8 + 8.
        let mut r = BitReader::new(&[0b0110_1000], 0);
        assert_eq!(read(&mut r, Codebook(0x031)), Some(0b01101 + 2 - 8));
    }

    /// A stream of valid codewords reads the same at every alignment and
    /// across refills, with the window always matching a bit-by-bit view.
    #[test]
    fn round_trips_through_the_test_encoder_at_every_alignment() {
        let cbs = [
            Codebook(0x2BC),
            Codebook(0x064),
            Codebook(0x031),
            Codebook(0),
            Codebook(0x433),
        ];
        let values = [0u32, 1, 2, 3, 7, 8, 15, 16, 100, 1000, 4095, 65535];
        let mut w = BitWriter::new();
        let mut expected = Vec::new();
        for &cb in &cbs {
            for &v in &values {
                // Only codewords that fit the 32-bit view are valid.
                let mut probe = BitWriter::new();
                if probe.symbol(cb, v) <= 32 {
                    w.symbol(cb, v);
                    expected.push((cb, v));
                }
            }
        }
        w.put(1, 1); // stop the stream from ending in zero fill
        let data = w.finish();
        for align in 0..4 {
            let mut r = BitReader::new(&data, align);
            for &(cb, v) in &expected {
                assert_eq!(read(&mut r, cb), Some(v), "{cb:?} {v} align {align}");
            }
        }
    }

    #[test]
    fn ends_on_zero_window_and_past_the_end() {
        let mut r = BitReader::new(&[0, 0, 0, 0, 0x80], 0);
        assert_eq!(read(&mut r, Codebook(0)), None);
        let mut r = BitReader::new(&[0b1000_0000], 0);
        assert_eq!(read(&mut r, Codebook(0)), Some(0));
        assert_eq!(read(&mut r, Codebook(0)), None);
        let mut r = BitReader::new(&[], 3);
        assert_eq!(read(&mut r, Codebook(0)), None);
    }

    #[test]
    fn overlong_codeword_shifts_wrap() {
        // 20 zeros then a one: q = 20 > s, len = 40 > 32; the value shift
        // wraps to (32 - 40) & 31 = 24.
        let p = 1u32 << 11;
        assert_eq!(codeword(p, Codebook(0)), (1 + (p >> 24) - 1, 40));
    }
}
