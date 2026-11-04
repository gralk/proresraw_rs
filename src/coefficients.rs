//! Entropy decoding of one component stream into quantized coefficients
//! (bitstream description §6.2–§6.4). Symbol arithmetic wraps at 32 bits
//! and coefficients are stored as 16 bits, so that malformed streams give
//! the same coefficients as Apple's decoder.

use crate::bits::BitReader;
use crate::codebook::{
    AC_CB, AC_INITIAL, Codebook, DC_CB, DC_RESET, FIRST_DC, LN_CB, LN_INITIAL, RN_CB, RN_INITIAL,
    SCAN,
};

/// Largest number of blocks a component stream can hold (alignment 4).
pub const MAX_BLOCKS: usize = 16;

/// Next codebook from a table indexed by a clamped symbol.
fn next(table: &[Codebook], symbol: u32) -> Codebook {
    table[(symbol as usize).min(table.len() - 1)]
}

/// Decode a component stream into `coefs`, which holds `64 << log2_blocks`
/// raster-order coefficients, block after block. Every entry is written.
/// Values are stored truncated to 16 bits.
/// `align` is the stream's offset from a 4-byte boundary (see
/// [`BitReader::new`]).
pub fn decode_component(data: &[u8], align: usize, log2_blocks: u8, coefs: &mut [i16]) {
    let blocks = 1u32 << log2_blocks;
    let codes = 64 * blocks;
    let block_mask = blocks - 1;
    debug_assert_eq!(coefs.len(), codes as usize);
    coefs.fill(0);
    let mut r = BitReader::new(data, align);

    // DC: a sign-folded first value, then deltas with a running sign.
    if r.top() == 0 {
        return;
    }
    let (first, len) = r.peek_symbol(FIRST_DC);
    r.consume(len);
    let odd = first & 1;
    let mut dc = ((odd.wrapping_add(first) >> 1) ^ odd.wrapping_neg()).wrapping_add(odd) as i32;
    coefs[0] = dc as i16;
    let mut dc_cb = DC_RESET;
    let mut sign = 0u32;
    for block in 1..blocks {
        if r.top() == 0 {
            return;
        }
        let (symbol, len) = r.peek_symbol(dc_cb);
        sign ^= symbol & 1;
        let magnitude = ((symbol & 1).wrapping_add(symbol)) >> 1;
        let delta = (sign.wrapping_neg() ^ magnitude).wrapping_add(sign) as i32;
        dc_cb = next(&DC_CB, magnitude);
        sign = u32::from(delta < 0);
        r.consume(len);
        dc = dc.wrapping_add(delta);
        coefs[64 * block as usize] = dc as i16;
    }

    // AC: runs of non-zero levels alternating with runs of zeros, in
    // frequency-major order across the blocks.
    let place =
        |n: u32| usize::from(SCAN[(n >> log2_blocks) as usize]) + ((n & block_mask) << 6) as usize;
    // Level symbol, sign and 32-bit view to the stored coefficient.
    let level = |r: &mut BitReader<'_>, cb: Codebook| {
        let top = r.top();
        let (symbol, len) = r.peek_symbol(cb);
        let negative = ((top as i32) << (len & 31)) >> 31 != 0;
        r.consume(len + 1);
        let magnitude = (symbol as u16).wrapping_add(1);
        (
            symbol,
            if negative {
                magnitude.wrapping_neg()
            } else {
                magnitude
            } as i16,
        )
    };
    let mut ac_cb = AC_INITIAL;
    let mut rn_cb = RN_INITIAL;
    let mut ln_cb = LN_INITIAL;
    let mut n = blocks;
    if n >= codes {
        return;
    }
    while r.top() != 0 {
        let (run, len) = r.peek_symbol(ln_cb);
        r.consume(len);
        for _ in 0..run {
            if r.top() == 0 {
                return;
            }
            let (symbol, value) = level(&mut r, ac_cb);
            ac_cb = next(&AC_CB, symbol);
            coefs[place(n)] = value;
            n += 1;
            if n == codes {
                return;
            }
        }
        if r.top() == 0 {
            return;
        }
        let (zeros, len) = r.peek_symbol(rn_cb);
        rn_cb = next(&RN_CB, zeros);
        r.consume(len);
        n = n.wrapping_add(1).wrapping_add(zeros);
        if n >= codes || r.top() == 0 {
            return;
        }
        let (symbol, value) = level(&mut r, ac_cb);
        ac_cb = next(&AC_CB, symbol);
        ln_cb = next(&LN_CB, symbol);
        coefs[place(n)] = value;
        n += 1;
        if n >= codes {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::testing::BitWriter;

    #[test]
    fn empty_stream_is_all_zero() {
        let mut c = vec![7i16; 64 * 4];
        decode_component(&[], 0, 2, &mut c);
        assert!(c.iter().all(|&v| v == 0));
    }

    #[test]
    fn decodes_dc_values_with_running_sign() {
        // First DC -3 (symbol 5), then deltas +2, -1, 0, +4 across five blocks.
        let mut w = BitWriter::new();
        w.symbol(FIRST_DC, 5);
        w.symbol(DC_RESET, 3); // n=1: magnitude 2, odd symbol flips sign to negative? see below
        let data = w.finish();
        let mut c = vec![0i16; 64 * 2];
        decode_component(&data, 0, 1, &mut c);
        assert_eq!(c[0], -3);
        // symbol 3: sign ^= 1 -> negative, magnitude (3+1)>>1 = 2 -> delta -2.
        assert_eq!(c[64], -5);
    }

    #[test]
    fn dc_sign_persists_until_a_zero_delta() {
        let mut w = BitWriter::new();
        w.symbol(FIRST_DC, 0); // dc 0
        w.symbol(DC_RESET, 1); // odd: negative, magnitude 1 -> -1
        w.symbol(DC_CB[1], 2); // even: keep sign, magnitude 1 -> -1
        w.symbol(DC_CB[1], 0); // magnitude 0 clears the sign
        w.symbol(DC_CB[0], 2); // positive 1
        let data = w.finish();
        let mut c = vec![0i16; 64 * 8];
        decode_component(&data, 0, 3, &mut c);
        let dcs: Vec<i16> = (0..5).map(|b| c[b * 64]).collect();
        assert_eq!(dcs, [0, -1, -2, -2, -1]);
    }

    #[test]
    fn decodes_ac_runs() {
        // One block: DC 0, then a run of two levels (+1, -3), a zero run of
        // one (coded 0), then level +2 at the next position.
        let mut w = BitWriter::new();
        w.symbol(FIRST_DC, 0);
        w.symbol(LN_INITIAL, 2);
        w.symbol(AC_INITIAL, 0);
        w.put(0, 1);
        w.symbol(AC_CB[0], 2);
        w.put(1, 1);
        w.symbol(RN_INITIAL, 0);
        w.symbol(AC_CB[2], 1);
        w.put(0, 1);
        let data = w.finish();
        let mut c = vec![0i16; 64];
        decode_component(&data, 0, 0, &mut c);
        assert_eq!(c[usize::from(SCAN[1])], 1);
        assert_eq!(c[usize::from(SCAN[2])], -3);
        assert_eq!(c[usize::from(SCAN[3])], 0);
        assert_eq!(c[usize::from(SCAN[4])], 2);
        assert_eq!(c.iter().filter(|&&v| v != 0).count(), 3);
    }

    #[test]
    fn interleaves_blocks_by_frequency() {
        // Two blocks: positions 2 and 3 are scan index 1 of blocks 0 and 1.
        let mut w = BitWriter::new();
        w.symbol(FIRST_DC, 0);
        w.symbol(DC_RESET, 0);
        w.symbol(LN_INITIAL, 2);
        w.symbol(AC_INITIAL, 4);
        w.put(0, 1);
        w.symbol(AC_CB[4], 6);
        w.put(0, 1);
        let data = w.finish();
        let mut c = vec![0i16; 128];
        decode_component(&data, 0, 1, &mut c);
        assert_eq!(c[usize::from(SCAN[1])], 5);
        assert_eq!(c[64 + usize::from(SCAN[1])], 7);
    }
}
