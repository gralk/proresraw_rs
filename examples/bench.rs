//! Time the crate's decoders and the raw conversion on frame packets, to
//! compare machines and builds.
//!
//! Usage: `bench [--runs N] FRAME.bin...`
//!
//! Each file is a size-prefixed packet (or a file of concatenated ones,
//! as written by `ffmpeg -c:v copy -f rawvideo`; the first is used). For
//! every file the tool prints the median and fastest time over `N` runs
//! (default 10, after one warm-up) of:
//!
//! - `decode`: [`decode_frame`];
//! - `linearize`: [`Frame::linearize_into`] into a reused buffer;
//! - `half`, `quarter`, `eighth`: [`decode_frame_reduced`], and `half+g2`
//!   with both greens averaged;
//! - `develop`: [`develop_with`] at `Demosaiced`, allocating its output;
//! - `develop into`: [`develop_planar_into`] at `Demosaiced` into reused
//!   planes.
//!
//! Build with `--release`. Times depend heavily on other load on the
//! machine: compare runs taken side by side.

use std::process::ExitCode;
use std::time::Instant;

use proresraw_rs::{
    DemosaicParams, Frame, Greens, Scale, Stage, decode_frame, decode_frame_reduced,
    decode_frame_reduced_with, develop_planar_into, develop_with,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut runs = 10usize;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--runs" => runs = args.next().ok_or("--runs needs a number")?.parse()?,
            _ => files.push(arg),
        }
    }
    if files.is_empty() || runs == 0 {
        return Err("usage: bench [--runs N] FRAME.bin...".into());
    }
    println!(
        "{} threads, {runs} runs, median / fastest in ms",
        rayon::current_num_threads()
    );
    for file in &files {
        let data = std::fs::read(file)?;
        let packet = first_packet(&data).ok_or("no packet in file")?;
        let frame = decode_frame(packet)?;
        let name = std::path::Path::new(file)
            .file_name()
            .map_or(file.as_str(), |n| n.to_str().unwrap_or(file));
        println!("\n{name}: {}x{}", frame.width(), frame.height());
        bench_frame(packet, &frame, runs)?;
    }
    Ok(())
}

fn bench_frame(packet: &[u8], frame: &Frame, runs: usize) -> proresraw_rs::Result<()> {
    let report = |what: &str, times: Vec<f64>| {
        let (median, fastest) = stats(times);
        println!("  {what:<14} {median:7.1} {fastest:7.1}");
    };
    report("decode", time(runs, || decode_frame(packet).map(drop))?);
    let mut linear = vec![0u16; frame.width() * frame.height()];
    report(
        "linearize",
        time(runs, || {
            frame.linearize_into(&mut linear);
            Ok(())
        })?,
    );
    for (what, scale) in [
        ("half", Scale::Half),
        ("quarter", Scale::Quarter),
        ("eighth", Scale::Eighth),
    ] {
        report(
            what,
            time(runs, || decode_frame_reduced(packet, scale).map(drop))?,
        );
    }
    report(
        "half+g2",
        time(runs, || {
            decode_frame_reduced_with(packet, Scale::Half, Greens::Average).map(drop)
        })?,
    );
    let params = DemosaicParams::new(&frame.header, frame.quantizer());
    report(
        "develop",
        time(runs, || {
            std::hint::black_box(develop_with(frame, &params, Stage::Demosaiced));
            Ok(())
        })?,
    );
    let n = frame.width() * frame.height();
    let mut planes = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    report(
        "develop into",
        time(runs, || {
            let [r, g, b] = &mut planes;
            develop_planar_into(frame, &params, Stage::Demosaiced, [r, g, b]);
            Ok(())
        })?,
    );
    Ok(())
}

/// Run `f` once to warm up, then `runs` times, returning milliseconds.
fn time(
    runs: usize,
    mut f: impl FnMut() -> proresraw_rs::Result<()>,
) -> proresraw_rs::Result<Vec<f64>> {
    f()?;
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            f()?;
            Ok(start.elapsed().as_secs_f64() * 1e3)
        })
        .collect()
}

/// Median and minimum.
fn stats(mut times: Vec<f64>) -> (f64, f64) {
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], times[0])
}

/// The first packet of a file of size-prefixed packets.
fn first_packet(data: &[u8]) -> Option<&[u8]> {
    let size = data.get(..4)?;
    let size = u32::from_be_bytes([size[0], size[1], size[2], size[3]]) as usize;
    data.get(..size)
}
