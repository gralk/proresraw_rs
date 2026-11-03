//! Adaptive codebook parameters and tables (bitstream description §6).

/// Parameters of one hybrid Golomb-Rice / exponential-Golomb code, packed
/// as `(switch << 8) | (exp_order << 4) | rice_order`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Codebook(pub u16);

impl Codebook {
    /// Largest leading-zero count decoded as a Rice codeword.
    pub fn switch(self) -> u32 {
        u32::from(self.0 >> 8)
    }

    /// Order of the exponential-Golomb part.
    pub fn exp_order(self) -> u32 {
        u32::from((self.0 >> 4) & 0xF)
    }

    /// Order of the Rice part.
    pub fn rice_order(self) -> u32 {
        u32::from(self.0 & 0xF)
    }
}

/// Code of the first DC coefficient of a component.
pub const FIRST_DC: Codebook = Codebook(0x2BC);
/// Code of the DC difference of blocks whose index is 1 modulo 16.
pub const DC_RESET: Codebook = Codebook(0x064);
/// Initial code of AC levels.
pub const AC_INITIAL: Codebook = Codebook(0x031);
/// Initial code of zero-run lengths.
pub const RN_INITIAL: Codebook = Codebook(0x000);
/// Initial code of non-zero-run lengths.
pub const LN_INITIAL: Codebook = Codebook(0x042);

/// Next DC difference code, indexed by the halved previous DC symbol.
pub const DC_CB: [Codebook; 13] = cb([
    0x010, 0x021, 0x032, 0x033, 0x033, 0x033, 0x044, 0x044, 0x044, 0x044, 0x044, 0x044, 0x076,
]);

/// Next AC level code, indexed by the previous level symbol.
pub const AC_CB: [Codebook; 95] = cb(ac_table());

/// Next zero-run code, indexed by the previous zero-run symbol.
pub const RN_CB: [Codebook; 28] = cb([
    0x200, 0x100, 0x000, 0x000, 0x211, 0x211, 0x111, 0x111, 0x011, 0x011, 0x021, 0x021, 0x222,
    0x022, 0x022, 0x022, 0x022, 0x022, 0x022, 0x022, 0x022, 0x022, 0x022, 0x022, 0x022, 0x032,
    0x032, 0x044,
]);

/// Next non-zero-run code, indexed by the level symbol that ended a zero run.
pub const LN_CB: [Codebook; 15] = cb([
    0x100, 0x111, 0x222, 0x222, 0x122, 0x122, 0x433, 0x433, 0x233, 0x233, 0x233, 0x233, 0x233,
    0x233, 0x033,
]);

const fn ac_table() -> [u16; 95] {
    let mut t = [0u16; 95];
    let runs: [(usize, u16); 10] = [
        (1, 0x000),
        (1, 0x211),
        (2, 0x111),
        (3, 0x222),
        (3, 0x122),
        (8, 0x233),
        (2, 0x133),
        (22, 0x244),
        (52, 0x355),
        (1, 0x166),
    ];
    let mut i = 0;
    let mut r = 0;
    while r < runs.len() {
        let (count, value) = runs[r];
        let mut k = 0;
        while k < count {
            t[i] = value;
            i += 1;
            k += 1;
        }
        r += 1;
    }
    assert!(i == 95);
    t
}

const fn cb<const N: usize>(raw: [u16; N]) -> [Codebook; N] {
    let mut out = [Codebook(0); N];
    let mut i = 0;
    while i < N {
        out[i] = Codebook(raw[i]);
        i += 1;
    }
    out
}

/// Scan position to raster coefficient index within an 8×8 block.
pub const SCAN: [u8; 64] = [
    0, 8, 1, 9, 16, 24, 17, 25, 2, 10, 3, 11, 18, 26, 19, 27, 32, 40, 33, 34, 41, 48, 56, 49, 42,
    35, 43, 50, 57, 58, 51, 59, 4, 12, 5, 6, 13, 20, 28, 21, 14, 7, 15, 22, 29, 36, 44, 37, 30, 23,
    31, 38, 45, 52, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpacks_parameters() {
        assert_eq!(
            (
                FIRST_DC.switch(),
                FIRST_DC.exp_order(),
                FIRST_DC.rice_order()
            ),
            (2, 11, 12)
        );
        assert_eq!(
            (
                LN_CB[6].switch(),
                LN_CB[6].exp_order(),
                LN_CB[6].rice_order()
            ),
            (4, 3, 3)
        );
    }

    #[test]
    fn ac_table_runs() {
        assert_eq!(AC_CB[0], Codebook(0));
        assert_eq!(AC_CB[9], Codebook(0x122));
        assert_eq!(AC_CB[10], Codebook(0x233));
        assert_eq!(AC_CB[19], Codebook(0x133));
        assert_eq!(AC_CB[20], Codebook(0x244));
        assert_eq!(AC_CB[41], Codebook(0x244));
        assert_eq!(AC_CB[42], Codebook(0x355));
        assert_eq!(AC_CB[93], Codebook(0x355));
        assert_eq!(AC_CB[94], Codebook(0x166));
    }

    #[test]
    fn scan_is_a_permutation() {
        let mut seen = [false; 64];
        for &i in &SCAN {
            assert!(!seen[usize::from(i)]);
            seen[usize::from(i)] = true;
        }
    }
}
