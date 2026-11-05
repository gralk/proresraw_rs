//! One decoded ProRes RAW frame as a DNG file.
//!
//! The raw image is the main (and only) image of the file: 16-bit CFA data
//! of the visible picture. By default the samples are the decoder's code
//! values and the frame's linearization curve becomes the DNG
//! `LinearizationTable`, so a reader linearizes exactly as the decoder does
//! and the code values themselves survive. Black (256) and white levels are
//! in the linear domain in both formats.
//!
//! The image is stored uncompressed in one strip, or as 256×256 tiles of
//! lossless JPEG (DNG compression 7).

use std::collections::BTreeMap;

use rayon::prelude::*;

use proresraw_rs::{BLACK_LEVEL, Color, Frame, FrameHeader};

use crate::colour::DngColour;
use crate::lj92;
use crate::tiff::{self, Layout, Value};

/// Side of the square lossless JPEG tiles: a multiple of 16, as TIFF
/// requires, and the size DNG writers commonly use.
const TILE: usize = 256;

/// How the image data is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// One uncompressed strip.
    None,
    /// Lossless JPEG tiles.
    LosslessJpeg,
}

/// What the samples hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Samples {
    /// Code values, with the curve as a `LinearizationTable`.
    CodeValues,
    /// Linear values (the curve applied), with no table.
    Linear,
}

/// Facts about the clip that every frame's file repeats.
#[derive(Debug, Clone, Default)]
pub struct ClipInfo {
    pub make: Option<String>,
    pub model: Option<String>,
    /// Frames per second as a fraction.
    pub fps: Option<(u32, u32)>,
    /// `YYYY:MM:DD HH:MM:SS`, when the movie records its creation time.
    pub date_time: Option<String>,
}

impl ClipInfo {
    /// `UniqueCameraModel`. The ProRes RAW data of a camera differs from its
    /// native raw files, so the name is kept apart from the camera's own.
    fn unique_camera_model(&self, header: &FrameHeader) -> String {
        let camera = [self.make.as_deref(), self.model.as_deref()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" ");
        if camera.is_empty() {
            format!("ProRes RAW {}", header.vendor.escape_ascii())
        } else {
            format!("{camera} (ProRes RAW)")
        }
    }
}

/// The DNG file of `frame`.
pub fn write(
    frame: &Frame,
    clip: &ClipInfo,
    samples: Samples,
    compression: Compression,
) -> Vec<u8> {
    let mut tags = tags(&frame.header, clip, samples);
    let values = visible_samples(frame, samples);
    let (width, height) = (frame.width(), frame.height());
    match compression {
        Compression::None => {
            tags.insert(259, Value::Short(vec![1]));
            let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let rows = height as u32;
            tiff::write(tags, Layout::Strips { rows }, &[data])
        }
        Compression::LosslessJpeg => {
            tags.insert(259, Value::Short(vec![7]));
            let across = width.div_ceil(TILE);
            let tiles: Vec<Vec<u8>> = (0..across * height.div_ceil(TILE))
                .into_par_iter()
                .map(|i| {
                    let tile = tile(&values, width, height, i % across * TILE, i / across * TILE);
                    lj92::encode(&tile, TILE, TILE, 2)
                })
                .collect();
            let side = TILE as u32;
            tiff::write(
                tags,
                Layout::Tiles {
                    width: side,
                    height: side,
                },
                &tiles,
            )
        }
    }
}

/// The `TILE`-square tile at (`left`, `top`) of an even-sized CFA image.
/// Sensels past the right or bottom edge repeat the nearest edge sensel of
/// the same colour, which keeps the padding cheap to code.
fn tile(values: &[u16], width: usize, height: usize, left: usize, top: usize) -> Vec<u16> {
    // The last column or row of the same CFA phase as `i`, for `i` past `n`.
    let clamp = |i: usize, n: usize| if i < n { i } else { n - 2 + (i - n) % 2 };
    let mut tile = Vec::with_capacity(TILE * TILE);
    for y in top..top + TILE {
        let row = &values[clamp(y, height) * width..][..width];
        tile.extend((left..left + TILE).map(|x| row[clamp(x, width)]));
    }
    tile
}

/// The visible picture, row by row, as the file stores it.
fn visible_samples(frame: &Frame, samples: Samples) -> Vec<u16> {
    match samples {
        Samples::CodeValues => frame.rows().flatten().copied().collect(),
        Samples::Linear => frame.linearize(),
    }
}

/// Every tag except the compression and the image layout.
fn tags(header: &FrameHeader, clip: &ClipInfo, samples: Samples) -> BTreeMap<u16, Value> {
    let colour = DngColour::new(header);
    let (width, height) = (u32::from(header.width), u32::from(header.height));
    let mut tags = BTreeMap::new();
    let mut put = |tag: u16, value: Value| {
        tags.insert(tag, value);
    };
    put(254, Value::Long(vec![0])); // NewSubFileType: main image
    put(256, Value::Long(vec![width]));
    put(257, Value::Long(vec![height]));
    put(258, Value::Short(vec![16])); // BitsPerSample
    put(262, Value::Short(vec![32803])); // PhotometricInterpretation: CFA
    if let Some(make) = &clip.make {
        put(271, Value::Ascii(make.clone()));
    }
    if let Some(model) = &clip.model {
        put(272, Value::Ascii(model.clone()));
    }
    put(274, Value::Short(vec![1])); // Orientation: top left
    put(277, Value::Short(vec![1])); // SamplesPerPixel
    put(284, Value::Short(vec![1])); // PlanarConfiguration: chunky
    put(
        305,
        Value::Ascii(format!(
            "proresraw_rs prr2dng {}",
            env!("CARGO_PKG_VERSION")
        )),
    );
    if let Some(date_time) = &clip.date_time {
        put(306, Value::Ascii(date_time.clone()));
    }
    put(33421, Value::Short(vec![2, 2])); // CFARepeatPatternDim
    let cfa =
        [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(r, c)| match header.bayer_pattern.color_at(r, c) {
            Color::Red => 0,
            Color::Green => 1,
            Color::Blue => 2,
        });
    put(33422, Value::Byte(cfa.to_vec())); // CFAPattern
    put(50706, Value::Byte(vec![1, 4, 0, 0])); // DNGVersion
    put(50707, Value::Byte(vec![1, 3, 0, 0])); // DNGBackwardVersion
    put(50708, Value::Ascii(clip.unique_camera_model(header)));
    put(50710, Value::Byte(vec![0, 1, 2])); // CFAPlaneColor
    put(50711, Value::Short(vec![1])); // CFALayout: rectangular
    if samples == Samples::CodeValues {
        put(50712, Value::Short(header.curve.lut().to_vec())); // LinearizationTable
    }
    put(50713, Value::Short(vec![1, 1])); // BlackLevelRepeatDim
    put(50714, Value::Long(vec![u32::from(BLACK_LEVEL)])); // BlackLevel
    put(50717, Value::Long(vec![u32::from(header.white_level)])); // WhiteLevel
    let crop = header.crop;
    let crop_width = width.saturating_sub(u32::from(crop.left) + u32::from(crop.right));
    let crop_height = height.saturating_sub(u32::from(crop.top) + u32::from(crop.bottom));
    if crop_width > 0 && crop_height > 0 {
        put(50719, Value::Long(vec![crop.left.into(), crop.top.into()])); // DefaultCropOrigin
        put(50720, Value::Long(vec![crop_width, crop_height])); // DefaultCropSize
    }
    put(50721, Value::srational(colour.color_matrix.as_flattened())); // ColorMatrix1
    put(50728, Value::rational(&colour.as_shot_neutral)); // AsShotNeutral
    put(50730, Value::srational(&[colour.baseline_exposure])); // BaselineExposure
    put(50778, Value::Short(vec![colour.illuminant])); // CalibrationIlluminant1
    put(
        50964,
        Value::srational(colour.forward_matrix.as_flattened()),
    ); // ForwardMatrix1
    if let Some((num, den)) = clip.fps {
        put(51044, Value::SRational(vec![(num as i32, den as i32)])); // FrameRate
    }
    tags
}

/// `YYYY:MM:DD HH:MM:SS` (UTC) of a QuickTime time: seconds since 1904-01-01.
pub fn quicktime_date_time(seconds: u64) -> String {
    let days = (seconds / 86400) as i64;
    let time = seconds % 86400;
    // Days from 1904-01-01 to 1970-01-01, then the civil date of a Unix day
    // (Howard Hinnant's algorithm).
    let z = days - 24107 + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}:{month:02}:{day:02} {:02}:{:02}:{:02}",
        time / 3600,
        time / 60 % 60,
        time % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_repeats_the_edge_of_the_same_colour() {
        // A 4×2 image whose value is its column plus 10 × its row.
        let values: Vec<u16> = (0..8).map(|i| (i % 4 + 10 * (i / 4)) as u16).collect();
        let tile = tile(&values, 4, 2, 0, 0);
        assert_eq!(&tile[..8], [0, 1, 2, 3, 2, 3, 2, 3]);
        assert_eq!(&tile[TILE..TILE + 4], [10, 11, 12, 13]);
        // Rows past the bottom alternate between the last two rows.
        assert_eq!(tile[2 * TILE], 0);
        assert_eq!(tile[3 * TILE], 10);
    }

    #[test]
    fn quicktime_dates() {
        assert_eq!(quicktime_date_time(0), "1904:01:01 00:00:00");
        // 2021-10-23 12:34:56 UTC is Unix time 1634992496.
        assert_eq!(
            quicktime_date_time(1_634_992_496 + 2_082_844_800),
            "2021:10:23 12:34:56"
        );
        assert_eq!(
            quicktime_date_time(2_082_844_800 - 1),
            "1969:12:31 23:59:59"
        );
    }
}
