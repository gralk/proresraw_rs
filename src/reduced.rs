//! Reduced-resolution decoding: half, quarter or eighth size RGB straight
//! from the bitstream, as Apple's decoder offers for previews.
//!
//! Each output pixel covers one 2×2 Bayer cell (half), 4×4 sensels
//! (quarter) or 8×8 sensels (eighth). Its red, green and blue come from
//! the red stream, the first green stream and the blue stream of the tile;
//! the second green stream is not decoded. Quarter and eighth resolution
//! decode only the leading 16 or 4 coefficients of every block and run a
//! 4×4 or 2×2 inverse transform on them, so they cost a fraction of a full
//! decode. Values are linearized with the frame's curve, like
//! [`Frame::linearize`](crate::Frame::linearize).

use crate::coefficients::{MAX_BLOCKS, decode_component_reduced};
use crate::decoder::parse_frame;
use crate::error::Result;
use crate::header::FrameHeader;
use crate::idct::{apple_block, apple_block_2x2, apple_block_4x4};
use crate::tile::{TILE_UNIT, Tile, TileGeometry};

/// Output size of a reduced decode, relative to the sensel grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scale {
    /// One pixel per 2×2 Bayer cell.
    Half,
    /// One pixel per 4×4 sensels.
    Quarter,
    /// One pixel per 8×8 sensels.
    Eighth,
}

impl Scale {
    /// Sensels per output pixel along each axis: 2, 4 or 8.
    pub fn factor(self) -> usize {
        1 << self.log2_factor()
    }

    fn log2_factor(self) -> u8 {
        match self {
            Scale::Half => 1,
            Scale::Quarter => 2,
            Scale::Eighth => 3,
        }
    }

    /// `log2` of the coefficients decoded per block.
    fn log2_coefs(self) -> u8 {
        match self {
            Scale::Half => 6,
            Scale::Quarter => 4,
            Scale::Eighth => 2,
        }
    }

    /// Output pixels along one side of a transformed block: 8, 4 or 2.
    fn block_side(self) -> usize {
        TILE_UNIT >> self.log2_factor()
    }
}

/// A reduced-resolution decode: linear RGB, three samples per pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct RgbFrame {
    /// The frame header.
    pub header: FrameHeader,
    /// The output scale.
    pub scale: Scale,
    /// Linear R, G, B sensor values, interleaved, row-major over
    /// [`RgbFrame::stride`] × the coded height divided by the scale
    /// factor (padding of the last tiles included). Black and white levels
    /// are those of the full-resolution linear values.
    pub samples: Vec<u16>,
    /// Indices (frame order) of tiles skipped for an invalid tile header;
    /// their pixels stay 0.
    pub skipped_tiles: Vec<usize>,
    /// Largest qscale byte over all tiles, as in
    /// [`ParsedFrame::max_qscale`](crate::ParsedFrame::max_qscale).
    pub max_qscale: u8,
}

impl RgbFrame {
    /// Visible width in pixels: the sensel width divided by the scale
    /// factor, rounded down.
    pub fn width(&self) -> usize {
        usize::from(self.header.width) >> self.scale.log2_factor()
    }

    /// Visible height in pixels, rounded down like the width.
    pub fn height(&self) -> usize {
        usize::from(self.header.height) >> self.scale.log2_factor()
    }

    /// Distance between rows in pixels (multiply by 3 for samples).
    pub fn stride(&self) -> usize {
        self.header.coded_width() >> self.scale.log2_factor()
    }

    /// The visible part of one row, as interleaved RGB samples.
    pub fn row(&self, y: usize) -> &[u16] {
        let start = 3 * y * self.stride();
        &self.samples[start..start + 3 * self.width()]
    }

    /// The frame quantizer, as [`Frame::quantizer`](crate::Frame::quantizer)
    /// gives it for a full decode.
    pub fn quantizer(&self) -> u32 {
        crate::decoder::quantizer(&self.header, self.max_qscale)
    }

    /// Visible pixels as tightly packed interleaved RGB.
    pub fn to_packed(&self) -> Vec<u16> {
        (0..self.height())
            .flat_map(|y| self.row(y))
            .copied()
            .collect()
    }
}

/// Decode one frame packet at a reduced resolution with the
/// Apple-compatible inverse transforms.
pub fn decode_frame_reduced(packet: &[u8], scale: Scale) -> Result<RgbFrame> {
    let parsed = parse_frame(packet)?;
    let header = parsed.header;
    let shift = scale.log2_factor();
    let stride = header.coded_width() >> shift;
    let rows_per_tile = TILE_UNIT >> shift;
    let mut samples = vec![0u16; 3 * stride * (header.coded_height() >> shift)];
    let per_row = parsed.grid.tiles_per_row();
    let lut = header.curve.lut();
    let qmat = &header.quant_matrix;
    let base = packet.as_ptr() as usize;
    let tile_rows = samples.chunks_exact_mut(3 * stride * rows_per_tile);
    let payload_rows = parsed.tiles.chunks_exact(per_row);
    let decode_row = |(tile_row, (rows, payloads)): (usize, (&mut [u16], &[&[u8]]))| {
        let first_index = tile_row * per_row;
        let mut skipped = Vec::new();
        for (i, (geometry, payload)) in parsed.grid.row(tile_row).zip(payloads).enumerate() {
            match Tile::parse(payload, first_index + i) {
                Ok(tile) => {
                    let ctx = TileContext {
                        qmat,
                        lut: &lut,
                        scale,
                        base,
                        stride,
                    };
                    ctx.decode(&tile, geometry, rows);
                }
                Err(_) => skipped.push(first_index + i),
            }
        }
        skipped
    };
    #[cfg(feature = "parallel")]
    let skipped: Vec<Vec<usize>> = {
        use rayon::prelude::*;
        let rows: Vec<_> = tile_rows.zip(payload_rows).collect();
        rows.into_par_iter().enumerate().map(decode_row).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let skipped: Vec<Vec<usize>> = tile_rows
        .zip(payload_rows)
        .enumerate()
        .map(decode_row)
        .collect();
    Ok(RgbFrame {
        header,
        scale,
        samples,
        skipped_tiles: skipped.concat(),
        max_qscale: parsed.max_qscale,
    })
}

/// Per-frame state shared by the tiles.
struct TileContext<'a> {
    qmat: &'a [u8; 64],
    lut: &'a [u16; 65536],
    scale: Scale,
    base: usize,
    /// Output row stride in pixels.
    stride: usize,
}

impl TileContext<'_> {
    /// Decode the red, first green and blue streams of one tile into the
    /// tile row `rows`.
    fn decode(&self, tile: &Tile<'_>, geometry: TileGeometry, rows: &mut [u16]) {
        // qmat <= 63 and qscale <= 255 are validated, so the product fits i16.
        let q: [i16; 64] =
            std::array::from_fn(|i| (u16::from(self.qmat[i]) * u16::from(tile.qscale)) as i16);
        let log2_coefs = self.scale.log2_coefs();
        let per_block = 1usize << log2_coefs;
        let side = self.scale.block_side();
        let blocks = geometry.blocks();
        let x0 = geometry.x >> self.scale.log2_factor();
        let mut coefs = [0i16; 64 * MAX_BLOCKS];
        let coefs = &mut coefs[..per_block * blocks];
        // Streams are stored green, green, blue, red.
        for (channel, stream) in [(0, 3), (1, 0), (2, 2)] {
            let data = tile.streams[stream];
            let align = (data.as_ptr() as usize).wrapping_sub(self.base) & 3;
            decode_component_reduced(data, align, geometry.log2_blocks, log2_coefs, coefs);
            for (b, block) in coefs.chunks_exact(per_block).enumerate() {
                let mut values = [0u16; 64];
                let values = &mut values[..per_block];
                match self.scale {
                    Scale::Half => values.copy_from_slice(&apple_block(
                        block.try_into().expect("64 coefficients"),
                        &q,
                    )),
                    Scale::Quarter => values.copy_from_slice(&apple_block_4x4(
                        block.try_into().expect("16 coefficients"),
                        &q,
                    )),
                    Scale::Eighth => values.copy_from_slice(&apple_block_2x2(
                        block.try_into().expect("4 coefficients"),
                        &q,
                    )),
                }
                let bx = x0 + b * side;
                for (i, line) in values.chunks_exact(side).enumerate() {
                    let start = 3 * (i * self.stride + bx) + channel;
                    for (d, &v) in rows[start..].iter_mut().step_by(3).zip(line) {
                        *d = self.lut[usize::from(v)];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_frame;
    use crate::synth::{Rng, Spec, random_frame, specs};

    /// Half resolution runs the same 8×8 transform as a full decode, so
    /// every pixel equals the full-resolution linear values of the red,
    /// first green and blue sensels of its Bayer cell.
    #[test]
    fn half_resolution_matches_full_decode() {
        let mut rng = Rng::new(7);
        for spec in specs(&mut rng) {
            let spec = Spec {
                valid_tiles: true,
                ..spec
            };
            let packet = random_frame(&spec, &mut rng);
            let full = decode_frame(&packet).unwrap();
            let linear = full.linearize();
            let width = full.width();
            let [g, _, b, r] = full.header.bayer_pattern.stream_phases();
            let rgb = decode_frame_reduced(&packet, Scale::Half).unwrap();
            assert_eq!((rgb.width(), rgb.height()), (width / 2, full.height() / 2));
            for y in 0..rgb.height() {
                for x in 0..rgb.width() {
                    let at = |(pr, pc): (usize, usize)| linear[(2 * y + pr) * width + 2 * x + pc];
                    let px = &rgb.row(y)[3 * x..3 * x + 3];
                    assert_eq!(px, [at(r), at(g), at(b)], "{:?} ({x},{y})", spec.pattern);
                }
            }
        }
    }

    #[test]
    fn reduced_sizes_and_skipped_tiles() {
        let mut rng = Rng::new(8);
        for spec in specs(&mut rng) {
            let packet = random_frame(&spec, &mut rng);
            let full = decode_frame(&packet).unwrap();
            for scale in [Scale::Half, Scale::Quarter, Scale::Eighth] {
                let f = scale.factor();
                let rgb = decode_frame_reduced(&packet, scale).unwrap();
                assert_eq!(rgb.width(), full.width() / f);
                assert_eq!(rgb.height(), full.height() / f);
                assert_eq!(
                    rgb.samples.len(),
                    3 * full.stride() / f * full.header.coded_height() / f
                );
                assert_eq!(rgb.skipped_tiles, full.skipped_tiles);
                assert_eq!(rgb.to_packed().len(), 3 * rgb.width() * rgb.height());
            }
        }
    }
}
