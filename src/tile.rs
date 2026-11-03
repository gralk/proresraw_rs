//! Tile grid geometry and tile headers (bitstream description §4 and §5).

use crate::error::{Error, Result};

/// Height of every tile in sensel rows, and the width of one tile column.
pub const TILE_UNIT: usize = 16;

/// Byte length of the fixed tile header fields: length, qscale, three sizes.
const TILE_HEADER_FIXED: usize = 8;

/// Where one tile sits in the picture and how many blocks it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGeometry {
    /// First sensel column.
    pub x: usize,
    /// First sensel row.
    pub y: usize,
    /// `log2` of the number of 8×8 blocks per component; the tile is
    /// `16 << log2_blocks` sensels wide.
    pub log2_blocks: u8,
}

impl TileGeometry {
    /// Number of 8×8 blocks per component.
    pub fn blocks(&self) -> usize {
        1 << self.log2_blocks
    }

    /// Width in sensels.
    pub fn width(&self) -> usize {
        TILE_UNIT << self.log2_blocks
    }
}

/// The tile partition of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
    /// Picture width in 16-sensel columns.
    columns: usize,
    /// Picture height in 16-sensel rows.
    rows: usize,
    /// Largest tile is `1 << alignment` columns wide.
    alignment: u8,
}

impl TileGrid {
    /// Grid for a picture of the given sensel size and tile alignment.
    pub fn new(width: usize, height: usize, alignment: u8) -> Self {
        TileGrid {
            columns: width.div_ceil(TILE_UNIT),
            rows: height.div_ceil(TILE_UNIT),
            alignment,
        }
    }

    /// Tiles in one tile row: full-width tiles, then the binary split of the remainder.
    pub fn tiles_per_row(&self) -> usize {
        let full = self.columns >> self.alignment;
        let remainder = self.columns & ((1 << self.alignment) - 1);
        full + remainder.count_ones() as usize
    }

    /// Number of tile rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Total number of tiles.
    pub fn len(&self) -> usize {
        self.tiles_per_row() * self.rows
    }

    /// True when the picture has no tiles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Geometry of the tiles in one tile row, left to right.
    pub fn row(&self, tile_row: usize) -> impl Iterator<Item = TileGeometry> + '_ {
        let y = tile_row * TILE_UNIT;
        let mut column = 0usize;
        let mut remaining = self.columns;
        let mut log2 = self.alignment;
        std::iter::from_fn(move || {
            while remaining > 0 {
                let unit = 1usize << log2;
                if unit <= remaining {
                    let tile = TileGeometry {
                        x: column * TILE_UNIT,
                        y,
                        log2_blocks: log2,
                    };
                    column += unit;
                    remaining -= unit;
                    return Some(tile);
                }
                log2 -= 1;
            }
            None
        })
    }

    /// Geometry of every tile in frame order (row-major).
    pub fn tiles(&self) -> impl Iterator<Item = TileGeometry> + '_ {
        (0..self.rows).flat_map(move |r| self.row(r))
    }
}

/// The parsed header of one tile and its four component streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile<'a> {
    /// Quantizer scale multiplying the frame's quantization matrix.
    pub qscale: u8,
    /// Entropy-coded coefficient streams, in stored order (see
    /// [`STREAM_PHASES`] for which Bayer phase each one carries).
    pub streams: [&'a [u8]; 4],
}

/// Bayer phase (row parity, column parity) carried by each tile stream.
pub const STREAM_PHASES: [(usize, usize); 4] = [(1, 0), (0, 1), (1, 1), (0, 0)];

impl<'a> Tile<'a> {
    /// Split a tile payload into its header and component streams.
    pub fn parse(data: &'a [u8], index: usize) -> Result<Self> {
        let bad = || Error::BadTileHeader { tile: index };
        if data.len() < TILE_HEADER_FIXED {
            return Err(bad());
        }
        let header_len = usize::from(data[0] >> 3);
        let qscale = data[1];
        let sizes = [
            usize::from(u16::from_be_bytes([data[2], data[3]])),
            usize::from(u16::from_be_bytes([data[4], data[5]])),
            usize::from(u16::from_be_bytes([data[6], data[7]])),
        ];
        let body = data.get(header_len..).ok_or_else(bad)?;
        let first_three = sizes[0] + sizes[1] + sizes[2];
        if first_three > body.len() {
            return Err(bad());
        }
        let (s0, rest) = body.split_at(sizes[0]);
        let (s1, rest) = rest.split_at(sizes[1]);
        let (s2, s3) = rest.split_at(sizes[2]);
        Ok(Tile {
            qscale,
            streams: [s0, s1, s2, s3],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn widths(grid: &TileGrid, row: usize) -> Vec<usize> {
        grid.row(row).map(|t| t.width()).collect()
    }

    #[test]
    fn full_rows_use_the_largest_tiles() {
        let g = TileGrid::new(5888, 3312, 3);
        assert_eq!(g.tiles_per_row(), 46);
        assert_eq!(g.rows(), 207);
        assert_eq!(g.len(), 46 * 207);
        assert!(widths(&g, 0).iter().all(|&w| w == 128));
    }

    #[test]
    fn remainder_splits_into_halving_tiles() {
        // 4264 sensels = 266.5 columns -> 267 columns = 33 * 8 + 2 + 1.
        let g = TileGrid::new(4264, 2408, 3);
        assert_eq!(g.tiles_per_row(), 35);
        let w = widths(&g, 0);
        assert_eq!(w.len(), 35);
        assert!(w[..33].iter().all(|&w| w == 128));
        assert_eq!(&w[33..], &[32, 16]);
        let last = g.row(0).last().unwrap();
        assert_eq!(last.x, 4256);
        assert_eq!(last.log2_blocks, 0);
        assert_eq!(g.row(150).next().unwrap().y, 2400);
    }

    #[test]
    fn alignment_zero_gives_single_block_tiles() {
        let g = TileGrid::new(48, 16, 0);
        assert_eq!(widths(&g, 0), [16, 16, 16]);
        assert_eq!(g.tiles().count(), 3);
    }

    #[test]
    fn alignment_four_tiles_are_256_wide() {
        let g = TileGrid::new(4224, 3024, 4);
        assert_eq!(g.tiles_per_row(), 17);
        assert_eq!(
            widths(&g, 0),
            [256; 16].iter().copied().chain([128]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn tile_header_splits_streams() {
        let mut data = vec![8 << 3, 15, 0, 2, 0, 1, 0, 3];
        data.extend_from_slice(&[10, 11, 20, 30, 31, 32, 40, 41]);
        let t = Tile::parse(&data, 0).unwrap();
        assert_eq!(t.qscale, 15);
        assert_eq!(t.streams[0], &[10, 11]);
        assert_eq!(t.streams[1], &[20]);
        assert_eq!(t.streams[2], &[30, 31, 32]);
        assert_eq!(t.streams[3], &[40, 41]);
    }

    #[test]
    fn tile_header_rejects_oversized_streams() {
        let data = [8 << 3, 1, 0, 9, 0, 0, 0, 0, 1, 2];
        assert_eq!(Tile::parse(&data, 4), Err(Error::BadTileHeader { tile: 4 }));
        assert_eq!(
            Tile::parse(&data[..5], 4),
            Err(Error::BadTileHeader { tile: 4 })
        );
        let data = [16 << 3, 1, 0, 0, 0, 0, 0, 0, 1, 2];
        assert_eq!(Tile::parse(&data, 1), Err(Error::BadTileHeader { tile: 1 }));
    }
}
