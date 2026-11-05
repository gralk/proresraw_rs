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
//! - `--linear`: store linear sensor values instead of the code values and
//!   their `LinearizationTable`.
//! - `--frames FIRST[-[LAST]]`: convert only these frames (0-based,
//!   inclusive).
//! - `--overwrite`: replace existing DNG files instead of stopping.
//!
//! A frame that fails to decode is reported and skipped; the others are
//! still written, and the exit status reports the failure.

mod colour;
mod dng;
mod mov;
mod tiff;

use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail, ensure};
use proresraw_rs::{FrameHeader, decode_frame, is_prores_raw_fourcc, parse_frame};

const USAGE: &str =
    "usage: prr2dng [-o DIR] [--linear] [--frames FIRST[-[LAST]]] [--overwrite] CLIP.mov...";

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
    samples: dng::Samples,
    /// First frame and, if given, last frame (inclusive).
    frames: (usize, Option<usize>),
    overwrite: bool,
}

fn parse_args() -> Result<Options> {
    let mut options = Options {
        clips: Vec::new(),
        output: PathBuf::from("."),
        samples: dng::Samples::CodeValues,
        frames: (0, None),
        overwrite: false,
    };
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("-o" | "--output") => {
                options.output = args.next().context("--output needs a folder")?.into();
            }
            Some("--linear") => options.samples = dng::Samples::Linear,
            Some("--overwrite") => options.overwrite = true,
            Some("--frames") => {
                let range = args.next().context("--frames needs a range")?;
                options.frames = parse_frames(&range.to_string_lossy())
                    .with_context(|| format!("--frames {}", range.to_string_lossy()))?;
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

/// `N` (one frame), `N-` (from N on) or `N-M` (N to M inclusive).
fn parse_frames(text: &str) -> Result<(usize, Option<usize>)> {
    let (first, last) = match text.split_once('-') {
        None => (text, Some(text)),
        Some((first, "")) => (first, None),
        Some((first, last)) => (first, Some(last)),
    };
    let first = first.parse().context("not a frame number")?;
    let last = last
        .map(|last| last.parse::<usize>().context("not a frame number"))
        .transpose()?;
    ensure!(last.is_none_or(|last| last >= first), "the range is empty");
    Ok((first, last))
}

/// A clip whose video track holds ProRes RAW frames.
struct Clip {
    path: PathBuf,
    /// The file name without its extension: the output folder and file prefix.
    name: String,
    file: File,
    track: mov::Track,
    /// The first frame's header.
    header: FrameHeader,
    info: dng::ClipInfo,
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
        let text = |keys: &[&str]| track.text(keys).map(str::to_owned);
        let info = dng::ClipInfo {
            make: text(&["com.apple.proapps.manufacturer", "com.apple.quicktime.make"]),
            model: text(&["com.apple.proapps.modelname", "com.apple.quicktime.model"]),
            fps: Some(track.fps),
            date_time: track.creation_time.map(dng::quicktime_date_time),
        };
        Ok(Self {
            path: path.to_owned(),
            name,
            file,
            track,
            header,
            info,
        })
    }

    /// The frames `options` selects, within the clip.
    fn frames(&self, options: &Options) -> Range<usize> {
        let count = self.track.samples.len();
        let (first, last) = options.frames;
        let end = last.map_or(count, |last| (last + 1).min(count));
        first.min(end)..end
    }

    /// Where frame `index` goes.
    fn output(&self, options: &Options, index: usize) -> PathBuf {
        options
            .output
            .join(&self.name)
            .join(format!("{}.{index:06}.dng", self.name))
    }

    fn report(&self, options: &Options) {
        let (num, den) = self.track.fps;
        let camera = [self.info.make.as_deref(), self.info.model.as_deref()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" ");
        let frames = self.frames(options);
        eprintln!(
            "{}: {} {}x{}, {} frames at {:.3} fps, {}",
            self.path.display(),
            self.track.fourcc.escape_ascii(),
            self.header.width,
            self.header.height,
            self.track.samples.len(),
            f64::from(num) / f64::from(den),
            if camera.is_empty() {
                "unknown camera"
            } else {
                &camera
            },
        );
        let colour = colour::DngColour::new(&self.header);
        eprintln!(
            "  as shot {:.0} K, white balance {:.4}/{:.4}, baseline exposure {:+.2}",
            colour.temperature, self.header.wb_red, self.header.wb_blue, colour.baseline_exposure,
        );
        for warning in &self.track.warnings {
            eprintln!("  warning: {warning}");
        }
        eprintln!(
            "  frames {}..{} -> {}",
            frames.start,
            frames.end,
            options.output.join(&self.name).display()
        );
    }

    /// Write the selected frames; the number that failed to decode.
    fn convert(&self, options: &Options) -> Result<usize> {
        let frames = self.frames(options);
        let folder = options.output.join(&self.name);
        std::fs::create_dir_all(&folder)
            .with_context(|| format!("creating {}", folder.display()))?;
        if !options.overwrite {
            if let Some(existing) = frames
                .clone()
                .map(|i| self.output(options, i))
                .find(|path| path.exists())
            {
                bail!("{} exists (use --overwrite to replace)", existing.display());
            }
        }
        let start = std::time::Instant::now();
        let mut failed = 0;
        for index in frames.clone() {
            let packet = sample(&self.file, &self.track, index)?;
            let frame = match decode_frame(&packet) {
                Ok(frame) => frame,
                Err(e) => {
                    eprintln!("\r  frame {index}: skipped, it does not decode: {e}");
                    failed += 1;
                    continue;
                }
            };
            if !frame.skipped_tiles.is_empty() {
                eprintln!(
                    "\r  frame {index}: {} invalid tiles left black",
                    frame.skipped_tiles.len()
                );
            }
            let data = dng::write(&frame, &self.info, options.samples);
            write_file(&self.output(options, index), &data)?;
            eprint!("\r  frame {}/{}", index + 1 - frames.start, frames.len());
        }
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "\r  {} frames in {seconds:.1} s ({:.1} fps)",
            frames.len() - failed,
            (frames.len() - failed) as f64 / seconds
        );
        Ok(failed)
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

/// Write `data` to `path` through a temporary file, so that an interrupted
/// run leaves no truncated DNG behind.
fn write_file(path: &Path, data: &[u8]) -> Result<()> {
    let partial = path.with_extension("dng.partial");
    std::fs::write(&partial, data).with_context(|| format!("writing {}", partial.display()))?;
    std::fs::rename(&partial, path).with_context(|| format!("writing {}", path.display()))
}

fn run() -> Result<()> {
    let options = parse_args()?;
    let clips = options
        .clips
        .iter()
        .map(|path| Clip::open(path).with_context(|| path.display().to_string()))
        .collect::<Result<Vec<_>>>()?;
    let mut failed = 0;
    for clip in &clips {
        clip.report(&options);
        failed += clip
            .convert(&options)
            .with_context(|| clip.path.display().to_string())?;
    }
    ensure!(
        failed == 0,
        "{failed} frames did not decode and were skipped"
    );
    Ok(())
}
