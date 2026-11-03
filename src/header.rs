//! Frame header parsing (bitstream description §3).

use crate::curve::Curve;
use crate::error::{Error, Result};

/// Black level of linearized output, fixed by the format.
pub const BLACK_LEVEL: u16 = 256;

/// Byte length of the fixed part of the header body (everything before
/// the optional quantization matrix and curve).
const FIXED_BODY_LEN: usize = 70;

/// Smallest valid `header_size` field: the fixed body plus the field itself.
pub const MIN_HEADER_SIZE: u16 = (FIXED_BODY_LEN + 2) as u16;

/// Largest tile alignment the format allows (tiles up to 256 sensels wide).
pub const MAX_ALIGNMENT: u8 = 4;

/// Arrangement of the 2×2 colour filter array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayerPattern {
    /// Red at (0,0), blue at (1,1).
    Rggb,
    /// Green at (0,0), red at (0,1).
    Grbg,
    /// Green at (0,0), blue at (0,1).
    Gbrg,
    /// Blue at (0,0), red at (1,1).
    Bggr,
}

/// One colour filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    /// Red filter.
    Red,
    /// Green filter.
    Green,
    /// Blue filter.
    Blue,
}

impl BayerPattern {
    /// Pattern for the two-bit code stored in the header.
    pub fn from_code(code: u16) -> Self {
        match code & 3 {
            0 => BayerPattern::Rggb,
            1 => BayerPattern::Grbg,
            2 => BayerPattern::Gbrg,
            _ => BayerPattern::Bggr,
        }
    }

    /// The two-bit code stored in the header.
    pub fn code(self) -> u16 {
        match self {
            BayerPattern::Rggb => 0,
            BayerPattern::Grbg => 1,
            BayerPattern::Gbrg => 2,
            BayerPattern::Bggr => 3,
        }
    }

    /// Position of the red sensel within the 2×2 cell as (row, column).
    pub fn red_position(self) -> (usize, usize) {
        match self {
            BayerPattern::Rggb => (0, 0),
            BayerPattern::Grbg => (0, 1),
            BayerPattern::Gbrg => (1, 0),
            BayerPattern::Bggr => (1, 1),
        }
    }

    /// Colour of the sensel at the given coordinates.
    pub fn color_at(self, row: usize, col: usize) -> Color {
        let (red_row, red_col) = self.red_position();
        let (row, col) = (row & 1, col & 1);
        if (row, col) == (red_row, red_col) {
            Color::Red
        } else if (row, col) == (red_row ^ 1, red_col ^ 1) {
            Color::Blue
        } else {
            Color::Green
        }
    }
}

/// Sensel margins the camera recommends discarding after demosaicing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Crop {
    /// Columns on the left.
    pub left: u8,
    /// Columns on the right.
    pub right: u8,
    /// Rows at the top.
    pub top: u8,
    /// Rows at the bottom.
    pub bottom: u8,
}

/// Everything the frame header says about the picture.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameHeader {
    /// Bitstream version; 0 and 1 are known.
    pub version: u8,
    /// Four-character vendor tag, such as `atm0` or `apl0`.
    pub vendor: [u8; 4],
    /// Picture width in sensels.
    pub width: u16,
    /// Picture height in sensels.
    pub height: u16,
    /// Recommended crop margins.
    pub crop: Crop,
    /// Colour filter arrangement.
    pub bayer_pattern: BayerPattern,
    /// White level of the linearized output; black is [`BLACK_LEVEL`].
    pub white_level: u16,
    /// White balance red multiplier.
    pub wb_red: f32,
    /// White balance blue multiplier.
    pub wb_blue: f32,
    /// Row-major camera RGB to CIE XYZ (D65) matrix.
    pub color_matrix: [[f32; 3]; 3],
    /// Gain applied after the matrix.
    pub gain: f32,
    /// White balance correlated colour temperature in kelvin, informational.
    pub wb_cct: u16,
    /// Raw flags word; bits 0, 1–3 and 4 are decoded into the fields below.
    pub flags: u16,
    /// Tile alignment: the widest tile spans `16 << alignment` sensels.
    pub alignment: u8,
    /// Quantization matrix in raster order, all ones when absent.
    pub quant_matrix: [u8; 64],
    /// Linearization curve, the default when absent.
    pub curve: Curve,
}

impl FrameHeader {
    /// Parse the header body (the bytes after the `header_size` field).
    pub fn parse(body: &[u8]) -> Result<Self> {
        if body.len() < FIXED_BODY_LEN {
            return Err(Error::Truncated {
                what: "frame header",
                needed: FIXED_BODY_LEN,
                available: body.len(),
            });
        }
        let mut r = Reader { data: body, pos: 1 };
        let version = r.u8();
        if version > 1 {
            return Err(Error::UnsupportedVersion(version));
        }
        let vendor = r.bytes4();
        let width = r.u16();
        let height = r.u16();
        if width == 0 || height == 0 || width & 1 != 0 || height & 1 != 0 {
            return Err(Error::BadDimensions { width, height });
        }
        let crop = Crop {
            left: r.u8(),
            right: r.u8(),
            top: r.u8(),
            bottom: r.u8(),
        };
        let bayer_pattern = BayerPattern::from_code(r.u16());
        let white_level = r.u16().wrapping_add(BLACK_LEVEL);
        let wb_red = r.f32();
        let wb_blue = r.f32();
        let mut color_matrix = [[0.0f32; 3]; 3];
        for row in &mut color_matrix {
            for v in row {
                *v = r.f32();
            }
        }
        let gain = r.f32();
        let wb_cct = r.u16();
        let flags = r.u16();
        let alignment = ((flags >> 1) & 7) as u8;
        if alignment > MAX_ALIGNMENT {
            return Err(Error::BadAlignment(alignment));
        }
        let mut quant_matrix = [1u8; 64];
        if flags & 1 != 0 {
            quant_matrix = r.array::<64>("quantization matrix")?;
        }
        let curve = if (flags >> 4) & 1 != 0 {
            let raw = r.array::<16>("linearization curve")?;
            let mut points = [0u16; 8];
            for (p, pair) in points.iter_mut().zip(raw.chunks_exact(2)) {
                *p = u16::from_be_bytes([pair[0], pair[1]]);
            }
            Curve(points)
        } else {
            Curve::DEFAULT
        };
        Ok(FrameHeader {
            version,
            vendor,
            width,
            height,
            crop,
            bayer_pattern,
            white_level,
            wb_red,
            wb_blue,
            color_matrix,
            gain,
            wb_cct,
            flags,
            alignment,
            quant_matrix,
            curve,
        })
    }

    /// Width rounded up to whole tiles.
    pub fn coded_width(&self) -> usize {
        (usize::from(self.width) + 15) & !15
    }

    /// Height rounded up to whole tiles.
    pub fn coded_height(&self) -> usize {
        (usize::from(self.height) + 15) & !15
    }
}

/// Big-endian cursor over a byte slice whose fixed part was length-checked.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn u8(&mut self) -> u8 {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }

    fn u16(&mut self) -> u16 {
        let v = u16::from_be_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        v
    }

    fn bytes4(&mut self) -> [u8; 4] {
        let v = [
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ];
        self.pos += 4;
        v
    }

    fn f32(&mut self) -> f32 {
        f32::from_be_bytes(self.bytes4())
    }

    fn array<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N]> {
        let end = self.pos + N;
        let slice = self.data.get(self.pos..end).ok_or(Error::Truncated {
            what,
            needed: end,
            available: self.data.len(),
        })?;
        let mut out = [0u8; N];
        out.copy_from_slice(slice);
        self.pos = end;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal header body with the given flags and trailing bytes.
    fn body(flags: u16, tail: &[u8]) -> Vec<u8> {
        let mut b = vec![0u8; FIXED_BODY_LEN];
        b[1] = 0; // version
        b[2..6].copy_from_slice(b"test");
        b[6..8].copy_from_slice(&64u16.to_be_bytes());
        b[8..10].copy_from_slice(&32u16.to_be_bytes());
        b[10..14].copy_from_slice(&[1, 2, 3, 4]);
        b[14..16].copy_from_slice(&3u16.to_be_bytes());
        b[16..18].copy_from_slice(&0xFDFFu16.to_be_bytes());
        b[18..22].copy_from_slice(&2.0f32.to_be_bytes());
        b[22..26].copy_from_slice(&1.5f32.to_be_bytes());
        for i in 0..9 {
            b[26 + 4 * i..30 + 4 * i].copy_from_slice(&(i as f32).to_be_bytes());
        }
        b[62..66].copy_from_slice(&11.8f32.to_be_bytes());
        b[66..68].copy_from_slice(&5600u16.to_be_bytes());
        b[68..70].copy_from_slice(&flags.to_be_bytes());
        b.extend_from_slice(tail);
        b
    }

    #[test]
    fn parses_fixed_fields_and_defaults() {
        let h = FrameHeader::parse(&body(0b0110, &[])).unwrap();
        assert_eq!(h.version, 0);
        assert_eq!(&h.vendor, b"test");
        assert_eq!((h.width, h.height), (64, 32));
        assert_eq!(
            h.crop,
            Crop {
                left: 1,
                right: 2,
                top: 3,
                bottom: 4
            }
        );
        assert_eq!(h.bayer_pattern, BayerPattern::Bggr);
        assert_eq!(h.white_level, 65279);
        assert_eq!((h.wb_red, h.wb_blue), (2.0, 1.5));
        assert_eq!(h.color_matrix[1][2], 5.0);
        assert_eq!(h.gain, 11.8);
        assert_eq!(h.wb_cct, 5600);
        assert_eq!(h.alignment, 3);
        assert_eq!(h.quant_matrix, [1; 64]);
        assert_eq!(h.curve, Curve::DEFAULT);
    }

    #[test]
    fn parses_quant_matrix_and_curve() {
        let mut tail = vec![7u8; 64];
        for p in [0u16, 224, 288, 468, 1162, 3339, 9431, 25377] {
            tail.extend_from_slice(&p.to_be_bytes());
        }
        let h = FrameHeader::parse(&body(0b11_1001, &tail)).unwrap();
        assert_eq!(h.alignment, 4);
        assert_eq!(h.quant_matrix, [7; 64]);
        assert_eq!(h.curve.0, [0, 224, 288, 468, 1162, 3339, 9431, 25377]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(matches!(
            FrameHeader::parse(&[0; 10]),
            Err(Error::Truncated { .. })
        ));
        let mut b = body(0, &[]);
        b[1] = 2;
        assert_eq!(FrameHeader::parse(&b), Err(Error::UnsupportedVersion(2)));
        let mut b = body(0, &[]);
        b[7] = 65;
        assert!(matches!(
            FrameHeader::parse(&b),
            Err(Error::BadDimensions { .. })
        ));
        assert_eq!(
            FrameHeader::parse(&body(0b1010, &[])),
            Err(Error::BadAlignment(5))
        );
        assert!(matches!(
            FrameHeader::parse(&body(1, &[0; 10])),
            Err(Error::Truncated { .. })
        ));
    }

    #[test]
    fn bayer_pattern_colours() {
        use Color::*;
        let grid = |p: BayerPattern| {
            [
                p.color_at(0, 0),
                p.color_at(0, 1),
                p.color_at(1, 0),
                p.color_at(1, 1),
            ]
        };
        assert_eq!(grid(BayerPattern::Rggb), [Red, Green, Green, Blue]);
        assert_eq!(grid(BayerPattern::Grbg), [Green, Red, Blue, Green]);
        assert_eq!(grid(BayerPattern::Gbrg), [Green, Blue, Red, Green]);
        assert_eq!(grid(BayerPattern::Bggr), [Blue, Green, Green, Red]);
        for code in 0..4 {
            assert_eq!(BayerPattern::from_code(code).code(), code);
        }
    }

    #[test]
    fn coded_dimensions_round_up_to_tiles() {
        let mut b = body(0, &[]);
        b[6..8].copy_from_slice(&4264u16.to_be_bytes());
        b[8..10].copy_from_slice(&2408u16.to_be_bytes());
        let h = FrameHeader::parse(&b).unwrap();
        assert_eq!((h.coded_width(), h.coded_height()), (4272, 2416));
    }
}
