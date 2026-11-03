//! Entropy decoding of one component stream into quantized coefficients
//! (bitstream description §6.2–§6.4).

use crate::bits::BitReader;
use crate::codebook::{
    AC_CB, AC_INITIAL, DC_CB, DC_RESET, FIRST_DC, LN_CB, LN_INITIAL, RN_CB, RN_INITIAL, SCAN,
};

/// Largest number of blocks a component stream can hold (alignment 4).
pub const MAX_BLOCKS: usize = 16;

/// Decode a component stream into `coefs`, which holds `64 << log2_blocks`
/// raster-order coefficients, block after block. Every entry is written.
pub fn decode_component(data: &[u8], log2_blocks: u8, coefs: &mut [i32]) {
    let blocks = 1usize << log2_blocks;
    let codes = 64 * blocks;
    let block_mask = blocks - 1;
    debug_assert_eq!(coefs.len(), codes);
    coefs.fill(0);
    let mut r = BitReader::new(data);

    let Some(first) = r.read_symbol(FIRST_DC) else {
        return;
    };
    let mut dc = ((first >> 1) as i32) ^ -((first & 1) as i32);
    coefs[0] = dc;
    let mut prev_symbol = first;
    let mut sign = 0i32;
    for block in 1..blocks {
        let cb = if block % 16 == 1 {
            DC_RESET
        } else {
            DC_CB[(((prev_symbol + 1) >> 1) as usize).min(DC_CB.len() - 1)]
        };
        let Some(symbol) = r.read_symbol(cb) else {
            return;
        };
        prev_symbol = symbol;
        sign ^= (symbol & 1) as i32;
        let magnitude = ((symbol + 1) >> 1) as i32;
        let delta = if sign != 0 { -magnitude } else { magnitude };
        sign = i32::from(delta < 0);
        dc += delta;
        coefs[block * 64] = dc;
    }

    let place = |n: usize| usize::from(SCAN[n >> log2_blocks]) + ((n & block_mask) << 6);
    let mut ac_cb = AC_INITIAL;
    let mut rn_cb = RN_INITIAL;
    let mut ln_cb = LN_INITIAL;
    let mut n = blocks;
    while n < codes {
        let Some(run) = r.read_symbol(ln_cb) else {
            return;
        };
        for _ in 0..run {
            let Some(level) = r.read_symbol(ac_cb) else {
                return;
            };
            ac_cb = AC_CB[(level as usize).min(AC_CB.len() - 1)];
            coefs[place(n)] = signed(level, r.read_bit());
            n += 1;
            if n == codes {
                return;
            }
        }
        let Some(zeros) = r.read_symbol(rn_cb) else {
            return;
        };
        rn_cb = RN_CB[(zeros as usize).min(RN_CB.len() - 1)];
        n += zeros as usize + 1;
        if n >= codes {
            return;
        }
        let Some(level) = r.read_symbol(ac_cb) else {
            return;
        };
        ac_cb = AC_CB[(level as usize).min(AC_CB.len() - 1)];
        ln_cb = LN_CB[(level as usize).min(LN_CB.len() - 1)];
        coefs[place(n)] = signed(level, r.read_bit());
        n += 1;
    }
}

/// Level symbol and sign bit to coefficient value.
fn signed(level: u32, sign_bit: u32) -> i32 {
    let magnitude = level as i32 + 1;
    if sign_bit != 0 { -magnitude } else { magnitude }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::testing::BitWriter;

    #[test]
    fn empty_stream_is_all_zero() {
        let mut c = vec![7i32; 64 * 4];
        decode_component(&[], 2, &mut c);
        assert!(c.iter().all(|&v| v == 0));
    }

    #[test]
    fn decodes_dc_values_with_running_sign() {
        // First DC -3 (symbol 5), then deltas +2, -1, 0, +4 across five blocks.
        let mut w = BitWriter::new();
        w.symbol(FIRST_DC, 5);
        w.symbol(DC_RESET, 3); // n=1: magnitude 2, odd symbol flips sign to negative? see below
        let data = w.finish();
        let mut c = vec![0i32; 64 * 2];
        decode_component(&data, 1, &mut c);
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
        let mut c = vec![0i32; 64 * 8];
        decode_component(&data, 3, &mut c);
        let dcs: Vec<i32> = (0..5).map(|b| c[b * 64]).collect();
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
        let mut c = vec![0i32; 64];
        decode_component(&data, 0, &mut c);
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
        let mut c = vec![0i32; 128];
        decode_component(&data, 1, &mut c);
        assert_eq!(c[usize::from(SCAN[1])], 5);
        assert_eq!(c[64 + usize::from(SCAN[1])], 7);
    }
}
