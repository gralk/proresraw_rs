//! Decode one frame from a file of concatenated ProRes RAW packets (as
//! written by `ffmpeg -i clip.mov -c:v copy -f rawvideo frames.bin`) and
//! write it as a 16-bit binary PGM.
//!
//! Usage: `decode FRAMES.bin OUT.pgm [--frame N] [--linear] [--gpu]`
//!
//! `--gpu` (feature `gpu`) decodes on the default wgpu adapter and reports
//! the steady-state time per frame over repeated decodes.

use std::process::ExitCode;

use proresraw_rs::{decode_frame, parse_frame};

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
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or("missing input file")?;
    let output = args.next().ok_or("missing output file")?;
    let mut frame_index = 0usize;
    let mut linear = false;
    let mut gpu = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frame" => frame_index = args.next().ok_or("--frame needs a number")?.parse()?,
            "--linear" => linear = true,
            "--gpu" => gpu = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let data = std::fs::read(&input)?;
    let packet = nth_packet(&data, frame_index).ok_or("frame index past the end of the file")?;
    let parsed = parse_frame(packet)?;
    eprintln!("{:#?}", parsed.header);
    eprintln!("{} tiles", parsed.tiles.len());
    let frame = if gpu {
        decode_on_gpu(packet)?
    } else {
        let start = std::time::Instant::now();
        let frame = decode_frame(packet)?;
        eprintln!("decoded in {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
        frame
    };
    let samples: Vec<u16> = if linear {
        frame.linearize()
    } else {
        frame.rows().flatten().copied().collect()
    };
    let mut pgm = format!("P5\n{} {}\n65535\n", frame.width(), frame.height()).into_bytes();
    pgm.extend(samples.iter().flat_map(|v| v.to_be_bytes()));
    std::fs::write(&output, pgm)?;
    Ok(())
}

#[cfg(feature = "gpu")]
fn decode_on_gpu(packet: &[u8]) -> Result<proresraw_rs::Frame, Box<dyn std::error::Error>> {
    use std::time::Instant;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))?;
    eprintln!("GPU: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))?;
    let mut decoder = proresraw_rs::GpuDecoder::new(&device);
    let frame = decoder.decode(&device, &queue, packet)?;
    let wait = |device: &wgpu::Device| device.poll(wgpu::PollType::wait_indefinitely());
    wait(&device)?;
    const RUNS: u32 = 20;
    let start = Instant::now();
    for _ in 0..RUNS {
        decoder.decode_into(&device, &queue, packet, &frame.samples)?;
    }
    wait(&device)?;
    let per_frame = start.elapsed().as_secs_f64() * 1e3 / f64::from(RUNS);
    eprintln!("GPU decode: {per_frame:.2} ms per frame over {RUNS} frames (upload included)");
    let start = Instant::now();
    let frame = frame.download(&device, &queue)?;
    eprintln!("download: {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    Ok(frame)
}

#[cfg(not(feature = "gpu"))]
fn decode_on_gpu(_: &[u8]) -> Result<proresraw_rs::Frame, Box<dyn std::error::Error>> {
    Err("built without the gpu feature".into())
}

/// The `n`th packet of a file of size-prefixed packets.
fn nth_packet(data: &[u8], n: usize) -> Option<&[u8]> {
    let mut offset = 0;
    for _ in 0..n {
        let size = data.get(offset..offset + 4)?;
        offset += u32::from_be_bytes([size[0], size[1], size[2], size[3]]) as usize;
    }
    let size = data.get(offset..offset + 4)?;
    let size = u32::from_be_bytes([size[0], size[1], size[2], size[3]]) as usize;
    data.get(offset..offset + size)
}
