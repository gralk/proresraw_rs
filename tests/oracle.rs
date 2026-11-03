//! Bit-exact comparison against FFmpeg's `prores_raw` decoder.
//!
//! Fixtures are produced by `tools/make_fixtures.sh` from sample footage
//! and are not part of the repository. Set `PRORES_RAW_FIXTURES` to the
//! fixture directory to run these tests; they are skipped otherwise.
//! Each fixture is a pair `NAME.bin` (one frame packet) and `NAME.raw`
//! (FFmpeg's `bayer_rggb16le` output, visible size, linearized).

use std::path::PathBuf;

use proresraw_rs::decode_frame;

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os("PRORES_RAW_FIXTURES").map(PathBuf::from)
}

#[test]
fn matches_ffmpeg_linearized_output() {
    let Some(dir) = fixture_dir() else {
        eprintln!("PRORES_RAW_FIXTURES not set; skipping the oracle test");
        return;
    };
    let mut packets: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("fixture directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .collect();
    packets.sort();
    assert!(!packets.is_empty(), "no .bin fixtures in {}", dir.display());
    for packet_path in packets {
        let packet = std::fs::read(&packet_path).unwrap();
        let reference = std::fs::read(packet_path.with_extension("raw")).unwrap();
        let frame = decode_frame(&packet).unwrap();
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
}
