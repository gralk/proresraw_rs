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
//! `docs/bitstream.md`. [`mov`] finds the frames of a QuickTime clip;
//! any other QuickTime reader that yields samples works as well.
//!
//! With the `gpu` feature, [`gpu::GpuDecoder`] decodes on a wgpu device
//! the application provides and leaves the code values in a GPU buffer.
//!
//! ```no_run
//! let packet = std::fs::read("frame.bin").unwrap();
//! let frame = proresraw_rs::decode_frame(&packet).unwrap();
//! println!("{}x{} {:?}", frame.width(), frame.height(), frame.header.bayer_pattern);
//! let linear = frame.linearize();
//! ```

// The only unsafe code is the run-time FMA dispatch of the raw conversion
// (feature `runtime-fma`); without it, unsafe code is forbidden outright.
#![cfg_attr(not(feature = "runtime-fma"), forbid(unsafe_code))]

pub mod bits;
pub mod codebook;
pub mod coefficients;
pub mod curve;
pub mod decoder;
pub mod demosaic;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod header;
pub mod idct;
pub mod mov;
pub mod reduced;
/// Synthetic frames for decoder tests, exposed with the `testgen` feature.
#[cfg(any(test, feature = "testgen"))]
pub mod synth;
pub mod tile;

pub use curve::Curve;
pub use decoder::{Frame, ParsedFrame, decode_frame, decode_frame_with, parse_frame};
pub use demosaic::{
    DemosaicParams, Exposure, RgbImage, Stage, develop, develop_into, develop_with, frame_quantizer,
};
pub use error::{Error, Result};
#[cfg(feature = "gpu")]
pub use gpu::{GpuDecoder, GpuError, GpuFrame};
pub use header::{BLACK_LEVEL, BayerPattern, Color, Crop, FrameHeader};
pub use idct::Idct;
pub use reduced::{RgbFrame, Scale, decode_frame_reduced};

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
