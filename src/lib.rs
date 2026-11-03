//! Apple ProRes RAW decoder in pure Rust.
//!
//! ProRes RAW stores the Bayer mosaic of the sensor, compressed per
//! 16-row tile with a DCT and an adaptive Golomb-Rice / exp-Golomb code.
//! This crate turns one frame packet (a QuickTime sample of an `aprn` or
//! `aprh` track) into the transform-domain code value of every sensel,
//! together with the header metadata a raw developer needs: the colour
//! filter arrangement, the linearization curve, black and white levels,
//! white balance factors, the camera-to-XYZ matrix and the recommended
//! crop. Apply the curve with [`Frame::linearize`] to obtain linear
//! sensor values.
//!
//! The implementation follows the bitstream description in
//! `docs/bitstream.md`. Demuxing is out of scope; any QuickTime reader
//! that yields samples works.
//!
//! ```no_run
//! let packet = std::fs::read("frame.bin").unwrap();
//! let frame = proresraw_rs::decode_frame(&packet).unwrap();
//! println!("{}x{} {:?}", frame.width(), frame.height(), frame.header.bayer_pattern);
//! let linear = frame.linearize();
//! ```

pub mod curve;
pub mod error;
pub mod header;

pub use curve::Curve;
pub use error::{Error, Result};
pub use header::{BLACK_LEVEL, BayerPattern, Color, Crop, FrameHeader};

/// QuickTime sample description FourCCs of ProRes RAW tracks.
pub const FOURCCS: [&[u8; 4]; 2] = [b"aprn", b"aprh"];

/// True when `fourcc` names a ProRes RAW track (`aprn` or `aprh`, any case).
pub fn is_prores_raw_fourcc(fourcc: &[u8; 4]) -> bool {
    let lower = fourcc.map(|b| b.to_ascii_lowercase());
    FOURCCS.contains(&&lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_fourccs() {
        assert!(is_prores_raw_fourcc(b"aprn"));
        assert!(is_prores_raw_fourcc(b"APRH"));
        assert!(!is_prores_raw_fourcc(b"apch"));
    }
}
