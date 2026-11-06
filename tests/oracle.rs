//! Bit-exact comparison against FFmpeg's `prores_raw` decoder.
//!
//! Fixtures are produced by `tools/make_fixtures.sh` from sample footage
//! and are not part of the repository. Set `PRORES_RAW_FIXTURES` to the
//! fixture directory to run these tests; they are skipped otherwise.
//! Each fixture is `NAME.bin` (one frame packet), with `NAME.raw`
//! (FFmpeg's `bayer_rggb16le` output, visible size, linearized) when
//! FFmpeg can decode it. FFmpeg refuses Bayer codes other than 0, so such
//! packets come without a reference and are only compared across this
//! crate's decoders.

use std::path::{Path, PathBuf};

use proresraw_rs::{Idct, decode_frame_with};

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os("PRORES_RAW_FIXTURES").map(PathBuf::from)
}

/// The `.bin` packets of the fixture directory, sorted.
fn packets(dir: &Path) -> Vec<PathBuf> {
    let mut packets: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("fixture directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .collect();
    packets.sort();
    assert!(!packets.is_empty(), "no .bin fixtures in {}", dir.display());
    packets
}

#[test]
fn matches_ffmpeg_linearized_output() {
    let Some(dir) = fixture_dir() else {
        eprintln!("PRORES_RAW_FIXTURES not set; skipping the oracle test");
        return;
    };
    let mut compared = 0;
    for packet_path in packets(&dir) {
        let Ok(reference) = std::fs::read(packet_path.with_extension("raw")) else {
            continue;
        };
        compared += 1;
        let packet = std::fs::read(&packet_path).unwrap();
        // FFmpeg uses the simple IDCT; the default (Apple-compatible) transform differs.
        let frame = decode_frame_with(&packet, Idct::Ffmpeg).unwrap();
        let linear = frame.linearize();
        assert_eq!(
            reference.len(),
            2 * linear.len(),
            "{}: reference size",
            packet_path.display()
        );
        let mismatches = linear
            .iter()
            .zip(reference.chunks_exact(2))
            .filter(|(got, want)| **got != u16::from_le_bytes([want[0], want[1]]))
            .count();
        assert_eq!(
            mismatches,
            0,
            "{}: {mismatches} sensels differ",
            packet_path.display()
        );
    }
    assert!(compared > 0, "no fixture has an FFmpeg reference");
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_matches_the_host_on_every_fixture() {
    let Some(dir) = fixture_dir() else {
        eprintln!("PRORES_RAW_FIXTURES not set; skipping the GPU fixture test");
        return;
    };
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("no GPU adapter; skipping the GPU fixture test");
        return;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let mut decoder = proresraw_rs::GpuDecoder::new(&device);
    for packet_path in packets(&dir) {
        let packet = std::fs::read(&packet_path).unwrap();
        let host = proresraw_rs::decode_frame(&packet).unwrap();
        let gpu = decoder
            .decode(&device, &queue, &packet)
            .unwrap()
            .download(&device, &queue)
            .unwrap();
        let mismatches = gpu
            .samples
            .iter()
            .zip(&host.samples)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches,
            0,
            "{}: {mismatches} sensels differ",
            packet_path.display()
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_develop_matches_the_host_on_every_fixture() {
    use proresraw_rs::{DemosaicParams, GpuDecoder, GpuDeveloper, Stage, develop_planar};
    let Some(dir) = fixture_dir() else {
        eprintln!("PRORES_RAW_FIXTURES not set; skipping the GPU develop fixture test");
        return;
    };
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("no GPU adapter; skipping the GPU develop fixture test");
        return;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let mut decoder = GpuDecoder::new(&device);
    let mut developer = GpuDeveloper::new(&device);
    for packet_path in packets(&dir) {
        let packet = std::fs::read(&packet_path).unwrap();
        let host_frame = proresraw_rs::decode_frame(&packet).unwrap();
        let params = DemosaicParams::new(&host_frame.header, host_frame.quantizer());
        let frame = decoder.decode(&device, &queue, &packet).unwrap();
        // The default bands (the whole frame where it fits), then bands of
        // 37 rows at the last stage.
        let runs = [
            (Stage::Demosaiced, None),
            (Stage::WhiteBalanced, None),
            (Stage::Rec2020, None),
            (Stage::Rec2020, Some(37)),
        ];
        for (stage, rows) in runs {
            developer.set_band_rows(rows);
            let host = develop_planar(&host_frame, &params, stage);
            let gpu = developer
                .develop(&device, &queue, &frame, &params, stage)
                .unwrap()
                .download(&device, &queue)
                .unwrap();
            let mismatches: usize = gpu
                .planes
                .iter()
                .zip(&host.planes)
                .map(|(g, h)| {
                    g.iter()
                        .zip(h)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count()
                })
                .sum();
            assert_eq!(
                mismatches,
                0,
                "{} {stage:?} bands {rows:?}: {mismatches} values differ",
                packet_path.display()
            );
        }
    }
}
