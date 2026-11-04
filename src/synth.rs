//! Synthetic frames for tests: a valid frame structure around arbitrary
//! tile payloads, so that decoders can be compared on inputs no camera
//! produces.

use crate::header::BayerPattern;
use crate::tile::TileGrid;

/// Small deterministic generator (xorshift64*).
pub struct Rng(u64);

impl Rng {
    /// Generator seeded with `seed` (any value).
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// Next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// Uniform value below `bound`.
    pub fn below(&mut self, bound: u32) -> u32 {
        self.next_u32() % bound
    }
}

/// Parameters of a synthetic frame.
pub struct Spec {
    /// Width in sensels, even.
    pub width: u16,
    /// Height in sensels, even.
    pub height: u16,
    /// Tile alignment, 0 to 4.
    pub alignment: u8,
    /// Colour filter arrangement.
    pub pattern: BayerPattern,
    /// Quantization matrix, written to the header when present.
    pub quant_matrix: Option<[u8; 64]>,
    /// Linearization curve, written to the header when present.
    pub curve: Option<[u16; 8]>,
}

/// A frame packet with random tile headers and component streams.
pub fn random_frame(spec: &Spec, rng: &mut Rng) -> Vec<u8> {
    let mut flags = u16::from(spec.alignment) << 1;
    let mut body = vec![0u8; 70];
    body[1] = 1;
    body[2..6].copy_from_slice(b"test");
    body[6..8].copy_from_slice(&spec.width.to_be_bytes());
    body[8..10].copy_from_slice(&spec.height.to_be_bytes());
    body[14..16].copy_from_slice(&spec.pattern.code().to_be_bytes());
    body[16..18].copy_from_slice(&0xFDFFu16.to_be_bytes());
    if let Some(q) = spec.quant_matrix {
        flags |= 1;
        body.extend_from_slice(&q);
    }
    if let Some(c) = spec.curve {
        flags |= 1 << 4;
        body.extend(c.iter().flat_map(|v| v.to_be_bytes()));
    }
    body[68..70].copy_from_slice(&flags.to_be_bytes());

    let grid = TileGrid::new(
        usize::from(spec.width),
        usize::from(spec.height),
        spec.alignment,
    );
    let tiles: Vec<Vec<u8>> = grid.tiles().map(|t| random_tile(t.blocks(), rng)).collect();

    let mut packet = Vec::new();
    packet.extend_from_slice(&0u32.to_be_bytes());
    packet.extend_from_slice(b"prrf");
    packet.extend_from_slice(&(body.len() as u16 + 2).to_be_bytes());
    packet.extend_from_slice(&body);
    for t in &tiles {
        packet.extend_from_slice(&(t.len() as u16).to_be_bytes());
    }
    for t in &tiles {
        packet.extend_from_slice(t);
    }
    let len = packet.len() as u32;
    packet[..4].copy_from_slice(&len.to_be_bytes());
    packet
}

/// One tile: an 8-byte header and four streams of random bits whose
/// lengths range from empty to generous for the block count. A zero
/// qscale or an empty stream makes the tile invalid, so a few tiles of
/// every frame are skipped.
fn random_tile(blocks: usize, rng: &mut Rng) -> Vec<u8> {
    let qscale = rng.below(256) as u8;
    let sizes: [usize; 4] = std::array::from_fn(|_| rng.below(48 * blocks as u32 + 1) as usize);
    let mut tile = vec![8 << 3, qscale];
    for s in &sizes[..3] {
        tile.extend_from_slice(&(*s as u16).to_be_bytes());
    }
    for s in sizes {
        // Bias towards zero bits so long codewords and early ends both occur.
        tile.extend((0..s).map(|_| (rng.next_u32() & rng.next_u32()) as u8));
    }
    tile
}

/// A varied set of specs covering every alignment and pattern, partial
/// edge tiles, absent and extreme quantization matrices and curves.
pub fn specs(rng: &mut Rng) -> Vec<Spec> {
    let patterns = [
        BayerPattern::Rggb,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
        BayerPattern::Bggr,
    ];
    let mut out = Vec::new();
    for alignment in 0..=4u8 {
        for (i, &pattern) in patterns.iter().enumerate() {
            let width = 2 * (1 + rng.below(300) as u16);
            let height = 2 * (1 + rng.below(40) as u16);
            // Quantization matrices and curves within what Apple's decoder
            // accepts: entries 1..=63, curves rising strictly.
            let quant_matrix = match i {
                0 => None,
                1 => Some([63; 64]),
                _ => Some(std::array::from_fn(|_| 1 + rng.below(63) as u8)),
            };
            let curve = (i % 2 == 1).then(|| {
                let mut c: [u16; 8] = std::array::from_fn(|_| rng.below(65536 - 8) as u16);
                c.sort_unstable();
                std::array::from_fn(|k| c[k] + k as u16)
            });
            out.push(Spec {
                width,
                height,
                alignment,
                pattern,
                quant_matrix,
                curve,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_frame;

    #[test]
    fn random_frames_decode_without_panicking() {
        let mut rng = Rng::new(1);
        for spec in specs(&mut rng) {
            let packet = random_frame(&spec, &mut rng);
            let frame = decode_frame(&packet).unwrap();
            assert_eq!(frame.width(), usize::from(spec.width));
            assert_eq!(frame.header.bayer_pattern, spec.pattern);
            let flat = frame.samples.iter().filter(|&&v| v == 32768).count();
            assert!(
                flat < frame.samples.len() / 2,
                "streams decoded to almost nothing"
            );
            frame.linearize();
        }
    }
}
