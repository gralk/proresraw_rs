//! Frame decoding: frame layout, tile dispatch and sensel placement
//! (bitstream description §2, §4, §5).

use crate::coefficients::{MAX_BLOCKS, decode_component};
use crate::error::{Error, Result};
use crate::header::{FrameHeader, MIN_HEADER_SIZE};
use crate::idct::Idct;
use crate::tile::{TILE_UNIT, Tile, TileGeometry, TileGrid};

/// Byte length of the frame prefix that precedes the header body: the
/// frame size (`u32`, big-endian), the tag `prrf` and the header size
/// (`u16`, big-endian, counting itself and the body).
pub const PREFIX_LEN: usize = 10;

/// A frame split into its header and tile payloads, before any decoding.
#[derive(Debug, Clone)]
pub struct ParsedFrame<'a> {
    /// The frame header.
    pub header: FrameHeader,
    /// The tile partition.
    pub grid: TileGrid,
    /// Tile payloads in frame order.
    pub tiles: Vec<&'a [u8]>,
    /// Largest qscale byte over all tiles, read from the second byte of
    /// every tile whether or not its header is valid, as Apple's decoder
    /// does. Drives the demosaic noise model
    /// ([`crate::demosaic::frame_quantizer`]).
    pub max_qscale: u8,
}

/// The frame quantizer of a header and largest tile qscale: the first
/// quantization matrix entry times the qscale.
pub(crate) fn quantizer(header: &FrameHeader, max_qscale: u8) -> u32 {
    u32::from(header.quant_matrix[0]) * u32::from(max_qscale)
}

/// Parse the frame header from the start of a packet, without the tile
/// table or tiles.
///
/// Only the first [`header_len`] bytes are read, so a container reader
/// can summarize a clip from the start of its first sample: read
/// [`PREFIX_LEN`] bytes, then `header_len` of them. The frame size field
/// is not checked against the packet, which may be truncated here;
/// [`parse_frame`] checks it.
pub fn parse_header(packet: &[u8]) -> Result<FrameHeader> {
    let body_end = header_len(packet)?;
    let body = packet.get(PREFIX_LEN..body_end).ok_or(Error::Truncated {
        what: "frame header",
        needed: body_end,
        available: packet.len(),
    })?;
    FrameHeader::parse(body)
}

/// Bytes from the start of a packet to the end of its header: the
/// [`PREFIX_LEN`]-byte prefix plus the header body. Reads and checks the
/// prefix (tag and header size) only.
pub fn header_len(packet: &[u8]) -> Result<usize> {
    let prefix = packet.get(..PREFIX_LEN).ok_or(Error::Truncated {
        what: "frame prefix",
        needed: PREFIX_LEN,
        available: packet.len(),
    })?;
    let tag = [prefix[4], prefix[5], prefix[6], prefix[7]];
    if &tag != b"prrf" {
        return Err(Error::BadTag(tag));
    }
    let header_size = u16::from_be_bytes([prefix[8], prefix[9]]);
    if header_size < MIN_HEADER_SIZE {
        return Err(Error::HeaderTooShort(header_size));
    }
    Ok(PREFIX_LEN + usize::from(header_size) - 2)
}

/// Split a packet into its header and tiles without decoding them.
pub fn parse_frame(packet: &[u8]) -> Result<ParsedFrame<'_>> {
    let prefix = packet.get(..PREFIX_LEN).ok_or(Error::Truncated {
        what: "frame prefix",
        needed: PREFIX_LEN,
        available: packet.len(),
    })?;
    // Like Apple's decoder, accept any non-zero size up to the packet
    // length; the frame is read from the packet regardless.
    let declared = u32::from_be_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]);
    if declared == 0 || declared as usize > packet.len() {
        return Err(Error::FrameSizeMismatch {
            declared,
            actual: packet.len(),
        });
    }
    let body_end = header_len(packet)?;
    let header = parse_header(packet)?;
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
    let mut max_qscale = 0;
    for (index, size) in table.chunks_exact(2).enumerate() {
        let size = usize::from(u16::from_be_bytes([size[0], size[1]]));
        // The byte after the tile start, even for a tile too short to hold it.
        max_qscale = max_qscale.max(packet.get(offset + 1).copied().unwrap_or(0));
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
        max_qscale,
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
    /// Indices (frame order) of tiles skipped for an invalid tile header.
    /// Such tiles are skipped without writing their sensels, as Apple's
    /// decoder does; here they stay 0. Empty for a well-formed frame.
    pub skipped_tiles: Vec<usize>,
    /// Largest qscale byte over all tiles, as in
    /// [`ParsedFrame::max_qscale`].
    pub max_qscale: u8,
}

impl Frame {
    /// The quantizer that drives the demosaic noise model, the same value
    /// [`crate::demosaic::frame_quantizer`] gives for the parsed packet.
    pub fn quantizer(&self) -> u32 {
        quantizer(&self.header, self.max_qscale)
    }

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
    ///
    /// To fuse linearization into a pass of your own instead, index the
    /// curve's table ([`crate::Curve::lut`]) with the code values of
    /// [`Frame::row`].
    pub fn linearize(&self) -> Vec<u16> {
        let mut out = vec![0; self.width() * self.height()];
        self.linearize_into(&mut out);
        out
    }

    /// [`Frame::linearize`] into a caller-owned buffer of
    /// `width * height` values, for reusing it from frame to frame.
    ///
    /// # Panics
    ///
    /// If `out` does not hold exactly `width * height` values.
    pub fn linearize_into(&self, out: &mut [u16]) {
        let w = self.width();
        assert_eq!(out.len(), w * self.height(), "output size");
        if w == 0 {
            return;
        }
        let lut = self.header.curve.lut();
        let row = |(y, dst): (usize, &mut [u16])| {
            for (d, &c) in dst.iter_mut().zip(self.row(y)) {
                *d = lut[usize::from(c)];
            }
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out.par_chunks_exact_mut(w).enumerate().for_each(row);
        }
        #[cfg(not(feature = "parallel"))]
        out.chunks_exact_mut(w).enumerate().for_each(row);
    }
}

/// Decode one frame packet (one QuickTime sample) with the Apple-compatible
/// inverse transform ([`Idct::Apple`]).
pub fn decode_frame(packet: &[u8]) -> Result<Frame> {
    decode_frame_with(packet, Idct::default())
}

/// Decode one frame packet with the given inverse transform.
pub fn decode_frame_with(packet: &[u8], idct: Idct) -> Result<Frame> {
    let parsed = parse_frame(packet)?;
    let header = parsed.header;
    let stride = header.coded_width();
    let mut samples = vec![0u16; stride * header.coded_height()];
    let per_row = parsed.grid.tiles_per_row();
    let tile_rows = samples.chunks_exact_mut(stride * TILE_UNIT);
    let payload_rows = parsed.tiles.chunks_exact(per_row);
    let qmat = &header.quant_matrix;
    let phases = header.bayer_pattern.stream_phases();
    let base = packet.as_ptr() as usize;
    // Decode one tile row; returns the indices of the tiles it skipped.
    let decode_row = |(tile_row, (rows, payloads)): (usize, (&mut [u16], &[&[u8]]))| {
        let first_index = tile_row * per_row;
        let mut skipped = Vec::new();
        for (i, (geometry, payload)) in parsed.grid.row(tile_row).zip(payloads).enumerate() {
            match Tile::parse(payload, first_index + i) {
                Ok(tile) => decode_tile(&tile, geometry, qmat, phases, idct, base, rows, stride),
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
    Ok(Frame {
        header,
        samples,
        skipped_tiles: skipped.concat(),
        max_qscale: parsed.max_qscale,
    })
}

/// Decode one tile into `rows`, the 16 sensel rows of its tile row,
/// placing each stream on its Bayer phase.
#[allow(clippy::too_many_arguments)]
fn decode_tile(
    tile: &Tile<'_>,
    geometry: TileGeometry,
    qmat: &[u8; 64],
    phases: [(usize, usize); 4],
    idct: Idct,
    base: usize,
    rows: &mut [u16],
    stride: usize,
) {
    // qmat <= 63 and qscale <= 255 are validated, so the product fits i16.
    let scaled: [i16; 64] =
        std::array::from_fn(|i| (u16::from(qmat[i]) * u16::from(tile.qscale)) as i16);
    let blocks = geometry.blocks();
    let mut coefs = [0i16; 64 * MAX_BLOCKS];
    let coefs = &mut coefs[..64 * blocks];
    for (stream, (phase_row, phase_col)) in tile.streams.iter().zip(phases) {
        // The bit reader depends on the stream's offset modulo 4 (for
        // invalid codewords only); the packet counts as 4-byte aligned.
        let align = (stream.as_ptr() as usize).wrapping_sub(base) & 3;
        decode_component(stream, align, geometry.log2_blocks, coefs);
        for (b, block) in coefs.chunks_exact(64).enumerate() {
            let block: &[i16; 64] = block.try_into().expect("64 coefficients");
            let values = idct.transform(block, &scaled);
            let x0 = geometry.x + b * TILE_UNIT + phase_col;
            for (i, line) in values.chunks_exact(8).enumerate() {
                let start = (2 * i + phase_row) * stride + x0;
                // Eight samples two apart span fifteen sensels.
                let dst = &mut rows[start..start + 15];
                for (d, &v) in dst.iter_mut().step_by(2).zip(line) {
                    *d = v;
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
        // Four one-byte streams of zero bits.
        p.extend_from_slice(&12u16.to_be_bytes());
        p.extend_from_slice(&[8 << 3, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]);
        let len = p.len() as u32;
        p[..4].copy_from_slice(&len.to_be_bytes());
        let frame = decode_frame(&p).unwrap();
        assert_eq!(
            (frame.width(), frame.height(), frame.stride()),
            (16, 16, 16)
        );
        assert!(frame.skipped_tiles.is_empty());
        assert!(frame.samples.iter().all(|&v| v == 32768));
        assert!(frame.linearize().iter().all(|&v| v == 4096));
        let mut reused = vec![0; 16 * 16];
        frame.linearize_into(&mut reused);
        assert_eq!(reused, frame.linearize());

        // Empty streams make the tile invalid: it is skipped.
        let at = p.len() - 12;
        p[at - 2..at].copy_from_slice(&8u16.to_be_bytes());
        p.truncate(at + 8);
        p[at + 2..at + 8].fill(0);
        let len = p.len() as u32;
        p[..4].copy_from_slice(&len.to_be_bytes());
        let frame = decode_frame(&p).unwrap();
        assert_eq!(frame.skipped_tiles, [0]);
        assert!(frame.samples.iter().all(|&v| v == 0));
    }

    #[test]
    fn frame_size_may_be_below_the_packet_length() {
        let mut p = vec![0u8; 80];
        p[4..8].copy_from_slice(b"prrf");
        p[8..10].copy_from_slice(&61u16.to_be_bytes());
        for size in [0u32, 81] {
            p[..4].copy_from_slice(&size.to_be_bytes());
            assert!(matches!(
                decode_frame(&p),
                Err(Error::FrameSizeMismatch { .. })
            ));
        }
        // A size below the packet length passes the size check.
        p[..4].copy_from_slice(&20u32.to_be_bytes());
        assert_eq!(decode_frame(&p), Err(Error::HeaderTooShort(61)));
    }

    /// The header parses from the prefix and body alone, before the tile
    /// table, and agrees with a full parse.
    #[test]
    fn header_parses_from_the_packet_start() {
        let mut rng = crate::synth::Rng::new(13);
        let spec = crate::synth::Spec::new(48, 32, crate::BayerPattern::Bggr);
        let packet = crate::synth::random_frame(&spec, &mut rng);
        let len = header_len(&packet).unwrap();
        assert!(len < packet.len());
        let header = parse_header(&packet[..len]).unwrap();
        assert_eq!(header, parse_frame(&packet).unwrap().header);
        assert!(matches!(
            parse_header(&packet[..len - 1]),
            Err(Error::Truncated { .. })
        ));
        assert!(matches!(
            header_len(&packet[..PREFIX_LEN - 1]),
            Err(Error::Truncated { .. })
        ));
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
