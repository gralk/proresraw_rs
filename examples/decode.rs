//! Decode one frame from a file of concatenated ProRes RAW packets (as
//! written by `ffmpeg -i clip.mov -c:v copy -f rawvideo frames.bin`) and
//! write it as a 16-bit binary PGM.
//!
//! Usage: `decode FRAMES.bin OUT.pgm [--frame N] [--linear] [--gpu]
//! [--scale half|quarter|eighth] [--develop STAGE]`
//!
//! `--scale` decodes at reduced resolution and writes a 16-bit binary PPM
//! of linear RGB instead. `--develop STAGE` runs the raw conversion up to
//! `demosaiced`, `white-balanced` or `rec2020` and writes a float PFM.
//!
//! `--gpu` (feature `gpu`) decodes on the default wgpu adapter and reports
//! the steady-state time per frame over repeated decodes.

use std::process::ExitCode;

use proresraw_rs::{Scale, decode_frame, decode_frame_reduced, parse_frame};

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
    let mut scale = None;
    let mut develop = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frame" => frame_index = args.next().ok_or("--frame needs a number")?.parse()?,
            "--linear" => linear = true,
            "--gpu" => gpu = true,
            "--develop" => {
                develop = Some(match args.next().as_deref() {
                    Some("demosaiced") => proresraw_rs::Stage::Demosaiced,
                    Some("white-balanced") => proresraw_rs::Stage::WhiteBalanced,
                    Some("rec2020") => proresraw_rs::Stage::Rec2020,
                    _ => return Err("--develop needs demosaiced, white-balanced or rec2020".into()),
                })
            }
            "--scale" => {
                scale = Some(match args.next().as_deref() {
                    Some("half") => Scale::Half,
                    Some("quarter") => Scale::Quarter,
                    Some("eighth") => Scale::Eighth,
                    _ => return Err("--scale needs half, quarter or eighth".into()),
                })
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let data = std::fs::read(&input)?;
    let packet = nth_packet(&data, frame_index).ok_or("frame index past the end of the file")?;
    let parsed = parse_frame(packet)?;
    eprintln!("{:#?}", parsed.header);
    eprintln!("{} tiles", parsed.tiles.len());
    if let Some(scale) = scale {
        let start = std::time::Instant::now();
        let rgb = decode_frame_reduced(packet, scale)?;
        eprintln!(
            "decoded {scale:?} in {:.1} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
        let mut ppm = format!("P6\n{} {}\n65535\n", rgb.width(), rgb.height()).into_bytes();
        ppm.extend(rgb.to_packed().iter().flat_map(|v| v.to_be_bytes()));
        std::fs::write(&output, ppm)?;
        return Ok(());
    }
    let frame = if gpu {
        decode_on_gpu(packet)?
    } else {
        let start = std::time::Instant::now();
        let frame = decode_frame(packet)?;
        eprintln!("decoded in {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
        frame
    };
    if let Some(stage) = develop {
        let start = std::time::Instant::now();
        let q = proresraw_rs::frame_quantizer(&parsed);
        let rgb = proresraw_rs::develop(&frame, q, stage);
        eprintln!(
            "developed in {:.1} ms (quantizer {q})",
            start.elapsed().as_secs_f64() * 1e3
        );
        // PFM stores rows bottom to top.
        let mut pfm = format!("PF\n{} {}\n-1.0\n", rgb.width, rgb.height).into_bytes();
        for row in rgb.data.chunks_exact(3 * rgb.width).rev() {
            pfm.extend(row.iter().flat_map(|v| v.to_le_bytes()));
        }
        std::fs::write(&output, pfm)?;
        return Ok(());
    }
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
