//! Synthetic frames for tests: a valid frame structure around arbitrary
//! tile payloads, so that decoders can be compared on inputs no camera
//! produces.

use crate::header::{BLACK_LEVEL, BayerPattern, Crop};
use crate::tile::{TileGeometry, TileGrid};

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

/// Parameters of a synthetic frame. [`Spec::new`] gives the defaults of
/// the version 1 header; the header fields exist so a caller can check what
/// it reads back from a decoded frame.
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
    /// Vendor tag.
    pub vendor: [u8; 4],
    /// Recommended crop.
    pub crop: Crop,
    /// White level of the linearized output, above [`crate::BLACK_LEVEL`].
    pub white_level: u16,
    /// White balance red and blue factors.
    pub wb: (f32, f32),
    /// Row-major camera RGB to XYZ matrix.
    pub color_matrix: [[f32; 3]; 3],
    /// Gain after the matrix.
    pub gain: f32,
    /// White balance correlated colour temperature.
    pub wb_cct: u16,
    /// Make every tile valid (non-zero qscale, non-empty streams), so no
    /// tile is skipped. Otherwise a few tiles of every frame are invalid.
    pub valid_tiles: bool,
}

impl Spec {
    /// A frame of `width` by `height` with alignment 3, no quantization
    /// matrix or curve, vendor `test`, no crop, white level 65279, unit white
    /// balance, the identity matrix, unit gain and 5600 K, random tiles of
    /// which some are invalid.
    pub fn new(width: u16, height: u16, pattern: BayerPattern) -> Self {
        Spec {
            width,
            height,
            alignment: 3,
            pattern,
            quant_matrix: None,
            curve: None,
            vendor: *b"test",
            crop: Crop::default(),
            white_level: 0xFDFF + 256,
            wb: (1., 1.),
            color_matrix: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            gain: 1.,
            wb_cct: 5600,
            valid_tiles: false,
        }
    }
}

/// A frame packet with random tile headers and component streams.
pub fn random_frame(spec: &Spec, rng: &mut Rng) -> Vec<u8> {
    frame(spec, |t| random_tile(t.blocks(), spec.valid_tiles, rng))
}

/// A frame packet whose tile payloads, in frame order, come from `payload`
/// given each tile's geometry; [`tile`] builds a payload from a qscale and
/// four streams.
pub fn frame(spec: &Spec, mut payload: impl FnMut(TileGeometry) -> Vec<u8>) -> Vec<u8> {
    let mut flags = u16::from(spec.alignment) << 1;
    let mut body = vec![0u8; 70];
    body[1] = 1;
    body[2..6].copy_from_slice(&spec.vendor);
    body[6..8].copy_from_slice(&spec.width.to_be_bytes());
    body[8..10].copy_from_slice(&spec.height.to_be_bytes());
    body[10..14].copy_from_slice(&[
        spec.crop.left,
        spec.crop.right,
        spec.crop.top,
        spec.crop.bottom,
    ]);
    body[14..16].copy_from_slice(&spec.pattern.code().to_be_bytes());
    body[16..18].copy_from_slice(&(spec.white_level - BLACK_LEVEL).to_be_bytes());
    body[18..22].copy_from_slice(&spec.wb.0.to_be_bytes());
    body[22..26].copy_from_slice(&spec.wb.1.to_be_bytes());
    for (i, v) in spec.color_matrix.iter().flatten().enumerate() {
        body[26 + 4 * i..30 + 4 * i].copy_from_slice(&v.to_be_bytes());
    }
    body[62..66].copy_from_slice(&spec.gain.to_be_bytes());
    body[66..68].copy_from_slice(&spec.wb_cct.to_be_bytes());
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
    let tiles: Vec<Vec<u8>> = grid.tiles().map(&mut payload).collect();

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
/// qscale or an empty stream makes the tile invalid, so unless `valid`
/// a few tiles of every frame are skipped.
fn random_tile(blocks: usize, valid: bool, rng: &mut Rng) -> Vec<u8> {
    let floor = u32::from(valid);
    let qscale = (floor + rng.below(256 - floor)) as u8;
    let sizes: [u32; 4] =
        std::array::from_fn(|_| floor + rng.below(48 * blocks as u32 + 1 - floor));
    // Bias towards zero bits so long codewords and early ends both occur.
    let streams = sizes.map(|s| {
        (0..s)
            .map(|_| (rng.next_u32() & rng.next_u32()) as u8)
            .collect::<Vec<u8>>()
    });
    tile(qscale, &streams.each_ref().map(Vec::as_slice))
}

/// A tile payload: the 8-byte header for `qscale` and the stream sizes,
/// then the four streams in stored order. A zero qscale or an empty stream
/// makes the tile invalid.
pub fn tile(qscale: u8, streams: &[&[u8]; 4]) -> Vec<u8> {
    let mut tile = vec![8 << 3, qscale];
    for s in &streams[..3] {
        tile.extend_from_slice(&(s.len() as u16).to_be_bytes());
    }
    for s in streams {
        tile.extend_from_slice(s);
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
                alignment,
                quant_matrix,
                curve,
                ..Spec::new(width, height, pattern)
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

    #[test]
    fn the_header_carries_the_spec_and_valid_tiles_are_never_skipped() {
        let mut rng = Rng::new(2);
        let spec = Spec {
            vendor: *b"atm0",
            crop: Crop {
                left: 16,
                right: 8,
                top: 4,
                bottom: 2,
            },
            white_level: 4095 + 256,
            wb: (1.5, 0.75),
            color_matrix: [[0.7, 0.2, 0.1], [0.3, 1.0, -0.3], [0.0, -0.4, 1.5]],
            gain: 11.8,
            wb_cct: 5235,
            valid_tiles: true,
            ..Spec::new(48, 32, BayerPattern::Bggr)
        };
        let frame = decode_frame(&random_frame(&spec, &mut rng)).unwrap();
        let h = &frame.header;
        assert_eq!(h.vendor, *b"atm0");
        assert_eq!(h.crop, spec.crop);
        assert_eq!(h.white_level, spec.white_level);
        assert_eq!((h.wb_red, h.wb_blue), spec.wb);
        assert_eq!(h.color_matrix, spec.color_matrix);
        assert_eq!(h.gain, spec.gain);
        assert_eq!(h.wb_cct, spec.wb_cct);
        assert!(frame.skipped_tiles.is_empty());
    }
}
