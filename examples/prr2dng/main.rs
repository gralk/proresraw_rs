//! Convert ProRes RAW clips to DNG image sequences.
//!
//! Usage: `prr2dng [OPTIONS] CLIP.mov...`
//!
//! Each clip must hold a ProRes RAW video track (`aprn` or `aprh`) whose
//! first frame parses; every clip is checked before any is converted. Frame
//! `n` of `CLIP.mov` is written to `OUT/CLIP/CLIP.nnnnnn.dng`.
//!
//! Options:
//! - `-o`, `--output DIR`: where the per-clip folders go (default: the
//!   current directory).

mod colour;
mod mov;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail, ensure};
use proresraw_rs::{FrameHeader, is_prores_raw_fourcc, parse_frame};

const USAGE: &str = "usage: prr2dng [-o DIR] CLIP.mov...";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

struct Options {
    clips: Vec<PathBuf>,
    output: PathBuf,
}

fn parse_args() -> Result<Options> {
    let mut options = Options {
        clips: Vec::new(),
        output: PathBuf::from("."),
    };
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("-o" | "--output") => {
                options.output = args.next().context("--output needs a folder")?.into();
            }
            Some("-h" | "--help") => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            Some(flag) if flag.starts_with('-') && flag.len() > 1 => {
                bail!("unknown option {flag}\n{USAGE}")
            }
            _ => options.clips.push(arg.into()),
        }
    }
    ensure!(!options.clips.is_empty(), "no clips given\n{USAGE}");
    Ok(options)
}

/// A clip whose video track holds ProRes RAW frames.
struct Clip {
    path: PathBuf,
    /// The file name without its extension: the output folder and file prefix.
    name: String,
    track: mov::Track,
    /// The first frame's header.
    header: FrameHeader,
}

impl Clip {
    /// Open `path` and check that it is a ProRes RAW clip: a QuickTime movie
    /// whose video track is `aprn` or `aprh` and whose first frame parses.
    fn open(path: &Path) -> Result<Self> {
        let name = path
            .file_stem()
            .context("the path has no file name")?
            .to_string_lossy()
            .into_owned();
        let file = File::open(path).context("cannot open the file")?;
        let track =
            mov::read_video_track(&file, is_prores_raw_fourcc).context("not a ProRes RAW clip")?;
        let packet = sample(&file, &track, 0)?;
        let header = parse_frame(&packet)
            .context("not a ProRes RAW clip: its first frame does not parse")?
            .header;
        Ok(Self {
            path: path.to_owned(),
            name,
            track,
            header,
        })
    }
}

/// The bytes of sample `index`.
fn sample(file: &File, track: &mov::Track, index: usize) -> Result<Vec<u8>> {
    let sample = track.samples[index];
    let mut packet = vec![0; sample.size as usize];
    mov::read_at(file, sample.offset, &mut packet)
        .with_context(|| format!("reading frame {index}"))?;
    Ok(packet)
}

fn run() -> Result<()> {
    let options = parse_args()?;
    let clips = options
        .clips
        .iter()
        .map(|path| Clip::open(path).with_context(|| path.display().to_string()))
        .collect::<Result<Vec<_>>>()?;
    for clip in &clips {
        let (num, den) = clip.track.fps;
        let camera = [
            clip.track
                .text(&["com.apple.proapps.manufacturer", "com.apple.quicktime.make"]),
            clip.track
                .text(&["com.apple.proapps.modelname", "com.apple.quicktime.model"]),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
        eprintln!(
            "{}: {} {}x{}, {} frames at {:.3} fps, {} -> {}",
            clip.path.display(),
            clip.track.fourcc.escape_ascii(),
            clip.header.width,
            clip.header.height,
            clip.track.samples.len(),
            f64::from(num) / f64::from(den),
            if camera.is_empty() {
                "unknown camera"
            } else {
                &camera
            },
            options.output.join(&clip.name).display(),
        );
        let colour = colour::DngColour::new(&clip.header);
        eprintln!(
            "  as shot {:.0} K, white balance {:.4}/{:.4}, baseline exposure {:+.2}",
            colour.temperature, clip.header.wb_red, clip.header.wb_blue, colour.baseline_exposure,
        );
        for warning in &clip.track.warnings {
            eprintln!("  warning: {warning}");
        }
    }
    Ok(())
}
