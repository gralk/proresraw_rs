//! Decode one frame from a file of concatenated ProRes RAW packets (as
//! written by `ffmpeg -i clip.mov -c:v copy -f rawvideo frames.bin`) and
//! write it as a 16-bit binary PGM.
//!
//! Usage: `decode FRAMES.bin OUT.pgm [--frame N] [--linear]`

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
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frame" => frame_index = args.next().ok_or("--frame needs a number")?.parse()?,
            "--linear" => linear = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let data = std::fs::read(&input)?;
    let packet = nth_packet(&data, frame_index).ok_or("frame index past the end of the file")?;
    let parsed = parse_frame(packet)?;
    eprintln!("{:#?}", parsed.header);
    eprintln!("{} tiles", parsed.tiles.len());
    let start = std::time::Instant::now();
    let frame = decode_frame(packet)?;
    eprintln!("decoded in {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
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
