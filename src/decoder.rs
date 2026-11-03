//! Frame decoding: frame layout, tile dispatch and sensel placement
//! (bitstream description §2, §4, §5).

use crate::coefficients::{MAX_BLOCKS, decode_component};
use crate::error::{Error, Result};
use crate::header::{FrameHeader, MIN_HEADER_SIZE};
use crate::idct::{code_value, inverse_transform};
use crate::tile::{STREAM_PHASES, TILE_UNIT, Tile, TileGeometry, TileGrid};

/// Byte length of the frame prefix: size, tag and header size.
const PREFIX_LEN: usize = 10;

/// A frame split into its header and tile payloads, before any decoding.
#[derive(Debug, Clone)]
pub struct ParsedFrame<'a> {
    /// The frame header.
    pub header: FrameHeader,
    /// The tile partition.
    pub grid: TileGrid,
    /// Tile payloads in frame order.
    pub tiles: Vec<&'a [u8]>,
}

/// Split a packet into its header and tiles without decoding them.
pub fn parse_frame(packet: &[u8]) -> Result<ParsedFrame<'_>> {
    let prefix = packet.get(..PREFIX_LEN).ok_or(Error::Truncated {
        what: "frame prefix",
        needed: PREFIX_LEN,
        available: packet.len(),
    })?;
    let declared = u32::from_be_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]);
    if declared as usize != packet.len() {
        return Err(Error::FrameSizeMismatch {
            declared,
            actual: packet.len(),
        });
    }
    let tag = [prefix[4], prefix[5], prefix[6], prefix[7]];
    if &tag != b"prrf" {
        return Err(Error::BadTag(tag));
    }
    let header_size = u16::from_be_bytes([prefix[8], prefix[9]]);
    if header_size < MIN_HEADER_SIZE {
        return Err(Error::HeaderTooShort(header_size));
    }
    let body_end = PREFIX_LEN + usize::from(header_size) - 2;
    let body = packet.get(PREFIX_LEN..body_end).ok_or(Error::Truncated {
        what: "frame header",
        needed: body_end,
        available: packet.len(),
    })?;
    let header = FrameHeader::parse(body)?;
    let grid = TileGrid::new(
        usize::from(header.width),
        usize::from(header.height),
        header.alignment,
    );
    let table_end = body_end + 2 * grid.len();
    let table = packet.get(body_end..table_end).ok_or(Error::Truncated {
        what: "tile size table",
        needed: table_end,
        available: packet.len(),
    })?;
    let mut offset = table_end;
    let mut tiles = Vec::with_capacity(grid.len());
    for (index, size) in table.chunks_exact(2).enumerate() {
        let size = usize::from(u16::from_be_bytes([size[0], size[1]]));
        let tile = packet
            .get(offset..offset + size)
            .ok_or(Error::TileOutOfBounds { tile: index })?;
        tiles.push(tile);
        offset += size;
    }
    Ok(ParsedFrame {
        header,
        grid,
        tiles,
    })
}

/// A decoded frame: transform-domain code values of every sensel.
///
/// Code values are what the inverse transform produces, before the
/// linearization curve. Apply [`FrameHeader::curve`] (for example with
/// [`Frame::linearize`]) to reach the linear sensor values that
/// [`crate::header::BLACK_LEVEL`] and [`FrameHeader::white_level`] refer to.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// The frame header.
    pub header: FrameHeader,
    /// Sensels in row-major order over the coded size, including the
    /// padding rows and columns of the last tiles.
    pub samples: Vec<u16>,
}

impl Frame {
    /// Visible width in sensels.
    pub fn width(&self) -> usize {
        usize::from(self.header.width)
    }

    /// Visible height in sensels.
    pub fn height(&self) -> usize {
        usize::from(self.header.height)
    }

    /// Distance between rows in `samples`: the coded width.
    pub fn stride(&self) -> usize {
        self.header.coded_width()
    }

    /// The visible part of one row.
    pub fn row(&self, y: usize) -> &[u16] {
        let start = y * self.stride();
        &self.samples[start..start + self.width()]
    }

    /// Visible rows, top to bottom.
    pub fn rows(&self) -> impl Iterator<Item = &[u16]> {
        (0..self.height()).map(|y| self.row(y))
    }

    /// Linear sensor values of the visible picture, tightly packed.
    pub fn linearize(&self) -> Vec<u16> {
        let lut = self.header.curve.lut();
        let mut out = Vec::with_capacity(self.width() * self.height());
        for row in self.rows() {
            out.extend(row.iter().map(|&c| lut[usize::from(c)]));
        }
        out
    }
}

/// Decode one frame packet (one QuickTime sample).
pub fn decode_frame(packet: &[u8]) -> Result<Frame> {
    let parsed = parse_frame(packet)?;
    let header = parsed.header;
    let stride = header.coded_width();
    let mut samples = vec![0u16; stride * header.coded_height()];
    let per_row = parsed.grid.tiles_per_row();
    let tile_rows = samples.chunks_exact_mut(stride * TILE_UNIT);
    let payload_rows = parsed.tiles.chunks_exact(per_row);
    let qmat = &header.quant_matrix;
    let decode_row = |(tile_row, (rows, payloads)): (usize, (&mut [u16], &[&[u8]]))| {
        let geometry = parsed.grid.row(tile_row);
        let first_index = tile_row * per_row;
        geometry
            .zip(payloads)
            .enumerate()
            .try_for_each(|(i, (geometry, payload))| {
                let tile = Tile::parse(payload, first_index + i)?;
                decode_tile(&tile, geometry, qmat, rows, stride);
                Ok(())
            })
    };
    tile_rows
        .zip(payload_rows)
        .enumerate()
        .try_for_each(decode_row)?;
    Ok(Frame { header, samples })
}

/// Decode one tile into `rows`, the 16 sensel rows of its tile row.
fn decode_tile(
    tile: &Tile<'_>,
    geometry: TileGeometry,
    qmat: &[u8; 64],
    rows: &mut [u16],
    stride: usize,
) {
    let mut scaled = [0i32; 64];
    for (s, &q) in scaled.iter_mut().zip(qmat) {
        *s = i32::from(q) * i32::from(tile.qscale);
    }
    let blocks = geometry.blocks();
    let mut coefs = [0i32; 64 * MAX_BLOCKS];
    let coefs = &mut coefs[..64 * blocks];
    for (stream, &(phase_row, phase_col)) in tile.streams.iter().zip(&STREAM_PHASES) {
        decode_component(stream, geometry.log2_blocks, coefs);
        for (b, block) in coefs.chunks_exact_mut(64).enumerate() {
            let block: &mut [i32; 64] = block.try_into().expect("64 coefficients");
            inverse_transform(block, &scaled);
            let x0 = geometry.x + b * TILE_UNIT + phase_col;
            for (i, line) in block.chunks_exact(8).enumerate() {
                let start = (2 * i + phase_row) * stride + x0;
                // Eight samples two apart span fifteen sensels.
                let dst = &mut rows[start..start + 15];
                for (d, &v) in dst.iter_mut().step_by(2).zip(line) {
                    *d = code_value(v);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_prefixes() {
        assert!(matches!(
            decode_frame(&[0; 4]),
            Err(Error::Truncated { .. })
        ));
        let mut p = vec![0u8; 80];
        p[..4].copy_from_slice(&80u32.to_be_bytes());
        p[4..8].copy_from_slice(b"xxxx");
        assert_eq!(decode_frame(&p), Err(Error::BadTag(*b"xxxx")));
        p[4..8].copy_from_slice(b"prrf");
        p[8..10].copy_from_slice(&61u16.to_be_bytes());
        assert_eq!(decode_frame(&p), Err(Error::HeaderTooShort(61)));
        p[..4].copy_from_slice(&81u32.to_be_bytes());
        assert!(matches!(
            decode_frame(&p),
            Err(Error::FrameSizeMismatch { .. })
        ));
    }

    /// A 16×16 frame with one tile whose streams are empty: every sensel
    /// decodes to the centre code value.
    #[test]
    fn empty_tile_decodes_to_mid_grey() {
        let mut body = vec![0u8; 70];
        body[6..8].copy_from_slice(&16u16.to_be_bytes());
        body[8..10].copy_from_slice(&16u16.to_be_bytes());
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(b"prrf");
        p.extend_from_slice(&72u16.to_be_bytes());
        p.extend_from_slice(&body);
        p.extend_from_slice(&8u16.to_be_bytes());
        p.extend_from_slice(&[8 << 3, 1, 0, 0, 0, 0, 0, 0]);
        let len = p.len() as u32;
        p[..4].copy_from_slice(&len.to_be_bytes());
        let frame = decode_frame(&p).unwrap();
        assert_eq!(
            (frame.width(), frame.height(), frame.stride()),
            (16, 16, 16)
        );
        assert!(frame.samples.iter().all(|&v| v == 32768));
        assert!(frame.linearize().iter().all(|&v| v == 4096));
    }

    #[test]
    fn missing_tile_is_an_error() {
        let mut body = vec![0u8; 70];
        body[6..8].copy_from_slice(&16u16.to_be_bytes());
        body[8..10].copy_from_slice(&16u16.to_be_bytes());
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(b"prrf");
        p.extend_from_slice(&72u16.to_be_bytes());
        p.extend_from_slice(&body);
        p.extend_from_slice(&40u16.to_be_bytes());
        p.extend_from_slice(&[0; 8]);
        let len = p.len() as u32;
        p[..4].copy_from_slice(&len.to_be_bytes());
        assert_eq!(decode_frame(&p), Err(Error::TileOutOfBounds { tile: 0 }));
    }
}
