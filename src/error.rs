//! Decoder errors.

use std::fmt;

/// Why a frame could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The data ends before the named structure is complete.
    Truncated {
        /// What was being read.
        what: &'static str,
        /// Bytes the structure needs.
        needed: usize,
        /// Bytes that were available.
        available: usize,
    },
    /// The size field at the start of the frame is zero or exceeds the packet.
    FrameSizeMismatch {
        /// Size declared by the frame.
        declared: u32,
        /// Length of the packet.
        actual: usize,
    },
    /// A quantization matrix entry is outside 1..=63.
    BadQuantMatrix {
        /// Raster index of the entry.
        index: usize,
        /// Its value.
        value: u8,
    },
    /// The linearization curve does not rise strictly.
    BadCurve([u16; 8]),
    /// A version 0 header carries a linearization curve.
    UnexpectedCurve,
    /// The frame does not start with the `prrf` tag.
    BadTag([u8; 4]),
    /// The frame header is shorter than the fixed fields.
    HeaderTooShort(u16),
    /// The bitstream version is not one this decoder knows.
    UnsupportedVersion(u8),
    /// The frame dimensions are odd or zero.
    BadDimensions {
        /// Width in sensels.
        width: u16,
        /// Height in sensels.
        height: u16,
    },
    /// The tile alignment exceeds the largest supported tile width.
    BadAlignment(u8),
    /// A tile's size table entry does not fit inside the frame.
    TileOutOfBounds {
        /// Index of the tile in the frame.
        tile: usize,
    },
    /// A tile's header is invalid. The decoder skips such tiles (leaving
    /// their sensels unwritten, at 0) and reports
    /// them in [`crate::Frame::skipped_tiles`]; this error comes only from
    /// [`crate::tile::Tile::parse`].
    BadTileHeader {
        /// Index of the tile in the frame.
        tile: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated {
                what,
                needed,
                available,
            } => write!(f, "{what}: need {needed} bytes, have {available}"),
            Error::FrameSizeMismatch { declared, actual } => {
                write!(
                    f,
                    "frame declares {declared} bytes; the packet holds {actual}"
                )
            }
            Error::BadQuantMatrix { index, value } => {
                write!(
                    f,
                    "quantization matrix entry {index} is {value}, outside 1..=63"
                )
            }
            Error::BadCurve(points) => {
                write!(f, "linearization curve {points:?} does not rise strictly")
            }
            Error::UnexpectedCurve => write!(f, "version 0 header carries a linearization curve"),
            Error::BadTag(tag) => write!(f, "expected the 'prrf' frame tag, found {tag:?}"),
            Error::HeaderTooShort(n) => write!(f, "frame header of {n} bytes is shorter than 72"),
            Error::UnsupportedVersion(v) => write!(f, "unsupported bitstream version {v}"),
            Error::BadDimensions { width, height } => {
                write!(f, "dimensions {width}x{height} must be even and non-zero")
            }
            Error::BadAlignment(a) => write!(f, "tile alignment {a} exceeds 4"),
            Error::TileOutOfBounds { tile } => write!(f, "tile {tile} extends past the frame"),
            Error::BadTileHeader { tile } => write!(f, "tile {tile} has an invalid header"),
        }
    }
}

impl std::error::Error for Error {}

/// Result alias for decoder operations.
pub type Result<T> = std::result::Result<T, Error>;
