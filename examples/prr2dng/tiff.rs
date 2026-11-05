//! A minimal little-endian TIFF writer: one image file directory (IFD) and
//! its image data, as strips or tiles. Enough for a DNG whose raw image is
//! the main image.

use std::collections::BTreeMap;

/// A tag value. Every variant is a list; a single value is a list of one.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Byte(Vec<u8>),
    Ascii(String),
    Short(Vec<u16>),
    Long(Vec<u32>),
    Rational(Vec<(u32, u32)>),
    SRational(Vec<(i32, i32)>),
}

impl Value {
    /// TIFF field type code.
    fn kind(&self) -> u16 {
        match self {
            Value::Byte(_) => 1,
            Value::Ascii(_) => 2,
            Value::Short(_) => 3,
            Value::Long(_) => 4,
            Value::Rational(_) => 5,
            Value::SRational(_) => 10,
        }
    }

    /// Number of values; an ASCII string counts its terminating NUL.
    fn count(&self) -> usize {
        match self {
            Value::Byte(v) => v.len(),
            Value::Ascii(s) => s.len() + 1,
            Value::Short(v) => v.len(),
            Value::Long(v) => v.len(),
            Value::Rational(v) => v.len(),
            Value::SRational(v) => v.len(),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Value::Byte(v) => v.clone(),
            Value::Ascii(s) => [s.as_bytes(), &[0]].concat(),
            Value::Short(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Long(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Rational(v) => v
                .iter()
                .flat_map(|&(n, d)| [n.to_le_bytes(), d.to_le_bytes()].concat())
                .collect(),
            Value::SRational(v) => v
                .iter()
                .flat_map(|&(n, d)| [n.to_le_bytes(), d.to_le_bytes()].concat())
                .collect(),
        }
    }

    /// The closest signed rational with a power-of-ten denominator of at
    /// most 10⁶ (enough for colour matrices and exposure).
    pub fn srational(values: &[f64]) -> Value {
        Value::SRational(
            values
                .iter()
                .map(|&v| {
                    let mut d = 1_000_000i64;
                    while d > 1 && (v * d as f64).abs() >= f64::from(i32::MAX) {
                        d /= 10;
                    }
                    let n = (v * d as f64)
                        .round()
                        .clamp(f64::from(i32::MIN), f64::from(i32::MAX));
                    (n as i32, d as i32)
                })
                .collect(),
        )
    }

    /// As [`Value::srational`], for non-negative values.
    pub fn rational(values: &[f64]) -> Value {
        Value::Rational(
            values
                .iter()
                .map(|&v| {
                    let mut d = 1_000_000u64;
                    while d > 1 && v * d as f64 >= f64::from(u32::MAX) {
                        d /= 10;
                    }
                    (
                        (v * d as f64).round().clamp(0.0, f64::from(u32::MAX)) as u32,
                        d as u32,
                    )
                })
                .collect(),
        )
    }
}

/// How the image data is cut up.
pub enum Layout {
    /// Strips of `rows` rows each.
    Strips { rows: u32 },
    /// Tiles of `width` × `height` sensels, row by row.
    Tiles { width: u32, height: u32 },
}

/// Tags of the strip or tile layout, which [`write`] fills in.
const STRIP_OFFSETS: u16 = 273;
const ROWS_PER_STRIP: u16 = 278;
const STRIP_BYTE_COUNTS: u16 = 279;
const TILE_WIDTH: u16 = 322;
const TILE_LENGTH: u16 = 323;
const TILE_OFFSETS: u16 = 324;
const TILE_BYTE_COUNTS: u16 = 325;

/// A TIFF file with one IFD holding `tags` and the image `chunks` (strips
/// or tiles, in order) laid out as `layout` says. The layout tags are
/// added here.
pub fn write(mut tags: BTreeMap<u16, Value>, layout: Layout, chunks: &[Vec<u8>]) -> Vec<u8> {
    let sizes = chunks.iter().map(|c| c.len() as u32).collect::<Vec<_>>();
    let (offsets_tag, counts_tag) = match layout {
        Layout::Strips { rows } => {
            tags.insert(ROWS_PER_STRIP, Value::Long(vec![rows]));
            (STRIP_OFFSETS, STRIP_BYTE_COUNTS)
        }
        Layout::Tiles { width, height } => {
            tags.insert(TILE_WIDTH, Value::Long(vec![width]));
            tags.insert(TILE_LENGTH, Value::Long(vec![height]));
            (TILE_OFFSETS, TILE_BYTE_COUNTS)
        }
    };
    tags.insert(counts_tag, Value::Long(sizes));
    // Placeholder offsets of the right count, so the layout can be sized.
    tags.insert(offsets_tag, Value::Long(vec![0; chunks.len()]));

    // Header, then the IFD, then values too large for their entry, then data.
    let ifd_size = 2 + 12 * tags.len() + 4;
    let mut values_size = 0;
    for value in tags.values() {
        let len = value.bytes().len();
        if len > 4 {
            values_size += len + (len & 1);
        }
    }
    let mut offset = (8 + ifd_size + values_size) as u32;
    let offsets = chunks
        .iter()
        .map(|c| {
            let at = offset;
            offset += c.len() as u32 + (c.len() as u32 & 1);
            at
        })
        .collect();
    tags.insert(offsets_tag, Value::Long(offsets));

    let mut out = Vec::with_capacity(offset as usize);
    out.extend(b"II");
    out.extend(42u16.to_le_bytes());
    out.extend(8u32.to_le_bytes());
    out.extend((tags.len() as u16).to_le_bytes());
    let mut overflow = Vec::with_capacity(values_size);
    let values_start = 8 + ifd_size;
    for (&tag, value) in &tags {
        let bytes = value.bytes();
        out.extend(tag.to_le_bytes());
        out.extend(value.kind().to_le_bytes());
        out.extend((value.count() as u32).to_le_bytes());
        if bytes.len() <= 4 {
            let mut inline = [0; 4];
            inline[..bytes.len()].copy_from_slice(&bytes);
            out.extend(inline);
        } else {
            out.extend(((values_start + overflow.len()) as u32).to_le_bytes());
            overflow.extend(&bytes);
            if bytes.len() & 1 == 1 {
                overflow.push(0);
            }
        }
    }
    // No further IFD.
    out.extend(0u32.to_le_bytes());
    out.extend(overflow);
    for chunk in chunks {
        out.extend(chunk);
        if chunk.len() & 1 == 1 {
            out.push(0);
        }
    }
    debug_assert_eq!(out.len(), offset as usize);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_at(data: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([data[at], data[at + 1]])
    }
    fn u32_at(data: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(data[at..at + 4].try_into().unwrap())
    }

    /// (tag, type, count, value or offset) of each IFD entry.
    fn entries(data: &[u8]) -> Vec<(u16, u16, u32, u32)> {
        let ifd = u32_at(data, 4) as usize;
        (0..usize::from(u16_at(data, ifd)))
            .map(|i| {
                let at = ifd + 2 + 12 * i;
                (
                    u16_at(data, at),
                    u16_at(data, at + 2),
                    u32_at(data, at + 4),
                    u32_at(data, at + 8),
                )
            })
            .collect()
    }

    #[test]
    fn strips_and_large_values_land_where_the_entries_say() {
        let mut tags = BTreeMap::new();
        tags.insert(305, Value::Ascii("abc".into()));
        tags.insert(50712, Value::Short(vec![1, 2, 3]));
        tags.insert(256, Value::Long(vec![7]));
        let chunks = vec![vec![1, 2, 3], vec![4, 5]];
        let data = write(tags, Layout::Strips { rows: 1 }, &chunks);
        assert_eq!(&data[..4], b"II*\0");
        let entries = entries(&data);
        let tags: Vec<u16> = entries.iter().map(|e| e.0).collect();
        assert_eq!(tags, [256, 273, 278, 279, 305, 50712], "sorted");
        let find = |tag| *entries.iter().find(|e| e.0 == tag).unwrap();
        assert_eq!(find(256), (256, 4, 1, 7));
        // "abc\0" fits in the entry.
        assert_eq!(find(305).3.to_le_bytes(), *b"abc\0");
        let table = find(50712);
        assert_eq!(table.2, 3);
        let at = table.3 as usize;
        assert_eq!(at % 2, 0);
        assert_eq!(&data[at..at + 6], [1, 0, 2, 0, 3, 0]);
        let offsets = find(273);
        assert_eq!(offsets.2, 2);
        let counts = find(279);
        for i in 0..2 {
            let offset = u32_at(&data, offsets.3 as usize + 4 * i) as usize;
            let count = u32_at(&data, counts.3 as usize + 4 * i) as usize;
            assert_eq!(offset % 2, 0);
            assert_eq!(data[offset..offset + count], chunks[i]);
        }
    }

    #[test]
    fn a_single_tile_offset_fits_in_its_entry() {
        let data = write(
            BTreeMap::new(),
            Layout::Tiles {
                width: 16,
                height: 16,
            },
            &[vec![9; 5]],
        );
        let entries = entries(&data);
        let offset = entries.iter().find(|e| e.0 == 324).unwrap().3 as usize;
        assert_eq!(data[offset..offset + 5], [9; 5]);
        assert_eq!(data.len(), offset + 6, "padded to an even length");
    }

    #[test]
    fn rationals_keep_six_decimals() {
        assert_eq!(
            Value::srational(&[-0.3556711, 3.560669, 5000.0]),
            Value::SRational(vec![
                (-355671, 1_000_000),
                (3560669, 1_000_000),
                (500_000_000, 100_000)
            ])
        );
        assert_eq!(
            Value::rational(&[0.5]),
            Value::Rational(vec![(500_000, 1_000_000)])
        );
    }
}
