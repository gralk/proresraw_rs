//! The sample table of one video track of a QuickTime (or ISO base media) movie:
//! where each presented frame's bytes are, and the frame rate, plus the movie's
//! text metadata. Only the `moov` atom is read into memory; samples are read with
//! positional I/O ([`read_at`]), so one `File` serves every decoder thread
//! without a lock.
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use proresraw_rs::{decode_frame, is_prores_raw_fourcc, mov};
//! let file = std::fs::File::open("A001_C010.mov")?;
//! let track = mov::read_video_track(&file, is_prores_raw_fourcc)?;
//! let first = track.samples[0];
//! let mut packet = vec![0; first.size as usize];
//! mov::read_at(&file, first.offset, &mut packet)?;
//! let frame = decode_frame(&packet)?;
//! # Ok(())
//! # }
//! ```
use std::{fmt, fs::File};

/// Why a movie could not be read: a message naming the atom or structure at
/// fault, with the context it was read in, outermost first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<std::array::TryFromSliceError> for Error {
    fn from(e: std::array::TryFromSliceError) -> Self {
        Self(e.to_string())
    }
}

/// The result of reading a movie.
pub type Result<T> = std::result::Result<T, Error>;

/// Return an [`Error`] with a formatted message.
macro_rules! bail {
    ($($message:tt)+) => {
        return Err(Error(format!($($message)+)))
    };
}
/// [`bail!`] unless `condition` holds.
macro_rules! ensure {
    ($condition:expr, $($message:tt)+) => {
        if !$condition {
            bail!($($message)+);
        }
    };
}
/// Context for a missing value or a failure: `context: cause`.
trait Context<T> {
    fn context(self, message: impl fmt::Display) -> Result<T>;
    fn with_context<M: fmt::Display>(self, message: impl FnOnce() -> M) -> Result<T>;
}
impl<T> Context<T> for Option<T> {
    fn context(self, message: impl fmt::Display) -> Result<T> {
        self.ok_or_else(|| Error(message.to_string()))
    }
    fn with_context<M: fmt::Display>(self, message: impl FnOnce() -> M) -> Result<T> {
        self.ok_or_else(|| Error(message().to_string()))
    }
}
impl<T, E: fmt::Display> Context<T> for std::result::Result<T, E> {
    fn context(self, message: impl fmt::Display) -> Result<T> {
        self.map_err(|e| Error(format!("{message}: {e}")))
    }
    fn with_context<M: fmt::Display>(self, message: impl FnOnce() -> M) -> Result<T> {
        self.map_err(|e| Error(format!("{}: {e}", message())))
    }
}

/// Larger `moov` atoms are refused rather than read: real movies' are tens of KB.
const MAX_MOOV: u64 = 64 << 20;
/// More samples than any movie this reader is for (46 hours at 60 fps).
const MAX_SAMPLES: u64 = 10_000_000;
/// Atoms a QuickTime or ISO movie may start with.
const FIRST_ATOMS: [&[u8; 4]; 7] = [
    b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot",
];

/// Where one sample's bytes are in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sample {
    /// Byte offset of the sample in the file.
    pub offset: u64,
    /// The sample's length in bytes.
    pub size: u32,
}

/// One video track of a movie: its format, rate, samples and the movie's
/// metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    /// The sample description's fourcc, such as `aprh`.
    pub fourcc: [u8; 4],
    /// Media timescale over the most common sample duration, reduced.
    pub fps: (u32, u32),
    /// Whether the samples' durations differ (the iPhone records this way);
    /// [`Self::fps`] is then the most common one.
    pub variable_durations: bool,
    /// Byte offset and size of each presented sample, in order.
    pub samples: Vec<Sample>,
    /// Movie creation time, seconds since 1904-01-01 UTC (mvhd), when non-zero.
    pub creation_time: Option<u64>,
    /// How to turn the stored frames for display, from the track's matrix.
    /// Decoded frames, with their Bayer pattern and crop, are as stored.
    pub orientation: Orientation,
    /// Conditions a caller should report once per clip (edit list oddities etc.).
    pub warnings: Vec<String>,
    /// The movie's text metadata items (`moov/meta`, `mdta` keys), in file order.
    pub metadata: Vec<(String, String)>,
}

/// How a track's stored frames are turned or mirrored for display: the
/// eight symmetries of a rectangle, named as the EXIF and TIFF orientations
/// are. Each says what to do to the stored frame to show it upright.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// Shown as stored (EXIF 1).
    #[default]
    Identity,
    /// Mirrored left to right (EXIF 2).
    MirrorHorizontal,
    /// Turned 180° (EXIF 3).
    Rotate180,
    /// Mirrored top to bottom (EXIF 4).
    MirrorVertical,
    /// Mirrored about the main diagonal: rows become columns (EXIF 5).
    Transpose,
    /// Turned 90° clockwise (EXIF 6), as for a phone held upright.
    Rotate90,
    /// Mirrored about the anti-diagonal (EXIF 7).
    Transverse,
    /// Turned 90° anticlockwise (EXIF 8).
    Rotate270,
}

impl Orientation {
    /// The EXIF / TIFF `Orientation` value, 1 to 8.
    pub fn exif(self) -> u8 {
        self as u8 + 1
    }
    /// Whether display swaps the stored width and height.
    pub fn swaps_dimensions(self) -> bool {
        self.exif() >= 5
    }
}

impl Track {
    /// The first text metadata value stored under any of `keys`, in that order.
    pub fn text(&self, keys: &[&str]) -> Option<&str> {
        keys.iter().find_map(|key| {
            self.metadata
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
        })
    }
}

/// The first enabled video track whose sample description `accept`s, with its samples.
pub fn read_video_track(file: &File, accept: fn(&[u8; 4]) -> bool) -> Result<Track> {
    let length = file.metadata().context("reading the file's length")?.len();
    let moov = read_moov(file, length)?;
    parse_movie(&moov, length, accept)
}

/// Positional read of `buf.len()` bytes at `offset`.
pub fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> Result<()> {
    read_exact_at(file, offset, buf)
        .with_context(|| format!("reading {} bytes at byte {offset}", buf.len()))
}
#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}
#[cfg(windows)]
fn read_exact_at(file: &File, offset: u64, mut buf: &mut [u8]) -> std::io::Result<()> {
    use std::{io::ErrorKind, os::windows::fs::FileExt};
    let mut offset = offset;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// The body of the top-level `moov` atom.
fn read_moov(file: &File, length: u64) -> Result<Vec<u8>> {
    let mut position = 0;
    while position < length {
        let mut header = [0; 16];
        let available = (length - position).min(16) as usize;
        ensure!(available >= 8, "truncated atom header at byte {position}");
        read_at(file, position, &mut header[..available])?;
        let kind: [u8; 4] = header[4..8].try_into()?;
        ensure!(
            position > 0 || FIRST_ATOMS.contains(&&kind),
            "not a QuickTime movie: it starts with `{}`, not `ftyp` or `moov`",
            name(&kind)
        );
        let (header_size, size) = match u32::from_be_bytes(header[..4].try_into()?) {
            1 => {
                ensure!(available == 16, "truncated `{}` atom header", name(&kind));
                (16, u64::from_be_bytes(header[8..].try_into()?))
            }
            0 => (8, length - position),
            size => (8, u64::from(size)),
        };
        ensure!(
            size >= header_size,
            "invalid `{}` atom size {size} at byte {position}",
            name(&kind)
        );
        ensure!(
            size <= length - position,
            "truncated `{}` atom at byte {position}: it claims {size} bytes and the file has {} \
             left (an interrupted recording?)",
            name(&kind),
            length - position
        );
        if kind == *b"moov" {
            let body = size - header_size;
            ensure!(
                body <= MAX_MOOV,
                "the `moov` atom is {body} bytes, more than the {MAX_MOOV} this reader accepts"
            );
            let mut moov = vec![0; body as usize];
            read_at(file, position + header_size, &mut moov)?;
            return Ok(moov);
        }
        position += size;
    }
    if length == 0 {
        bail!("not a QuickTime movie: the file is empty");
    }
    bail!("the movie has no `moov` atom, so its frames cannot be found (an interrupted recording?)")
}

type Atom<'a> = ([u8; 4], &'a [u8]);

fn parse_movie(moov: &[u8], length: u64, accept: fn(&[u8; 4]) -> bool) -> Result<Track> {
    let atoms = children(moov)?;
    let mvhd = require(&atoms, b"mvhd", "the movie")?;
    let (creation_time, movie_timescale) = times(mvhd, b"mvhd")?;
    let mut rejected = None;
    for (_, trak) in atoms.iter().filter(|(kind, _)| kind == b"trak") {
        let Some(video) = video_track(trak)? else {
            continue;
        };
        if !accept(&video.fourcc) {
            rejected.get_or_insert(video.fourcc);
            continue;
        }
        let mut track = video.read(length, movie_timescale)?;
        track.creation_time = Some(creation_time).filter(|&t| t != 0);
        if let Some((_, meta)) = atoms.iter().find(|(kind, _)| kind == b"meta") {
            // Metadata is optional: a damaged item costs the tags, not the frames.
            match text_metadata(meta) {
                Ok(items) => track.metadata = items,
                Err(e) => track
                    .warnings
                    .push(format!("the movie's metadata is unreadable: {e:#}")),
            }
        }
        return Ok(track);
    }
    match rejected {
        Some(fourcc) => bail!(
            "the video track is `{}`, not a format this reader accepts",
            name(&fourcc)
        ),
        None => bail!("the movie has no enabled video track"),
    }
}

/// An enabled video track's media, before its sample table is resolved.
struct Video<'a> {
    fourcc: [u8; 4],
    timescale: u32,
    tkhd: &'a [u8],
    elst: Option<&'a [u8]>,
    stbl: Vec<Atom<'a>>,
}

/// `trak`'s media when it is an enabled video track.
fn video_track(trak: &[u8]) -> Result<Option<Video<'_>>> {
    let atoms = children(trak)?;
    let tkhd = require(&atoms, b"tkhd", "a track")?;
    if u32_at(tkhd, 0, b"tkhd")? & 1 == 0 {
        return Ok(None);
    }
    let mdia = children(require(&atoms, b"mdia", "a track")?)?;
    // Version and flags, component type, then the subtype.
    let handler = u32_at(require(&mdia, b"hdlr", "a track's media")?, 8, b"hdlr")?;
    if handler.to_be_bytes() != *b"vide" {
        return Ok(None);
    }
    let (_, timescale) = times(require(&mdia, b"mdhd", "the video track")?, b"mdhd")?;
    ensure!(timescale > 0, "the video track's timescale is 0");
    let minf = children(require(&mdia, b"minf", "the video track")?)?;
    let stbl = children(require(&minf, b"stbl", "the video track")?)?;
    let stsd = require(&stbl, b"stsd", "the video track")?;
    ensure!(
        u32_at(stsd, 4, b"stsd")? > 0,
        "the video track has no sample description"
    );
    // Version and flags, entry count, then the first entry's size and fourcc.
    let fourcc = u32_at(stsd, 12, b"stsd")?.to_be_bytes();
    let elst = match atoms.iter().find(|(kind, _)| kind == b"edts") {
        Some((_, edts)) => children(edts)?
            .into_iter()
            .find(|(kind, _)| kind == b"elst")
            .map(|(_, elst)| elst),
        None => None,
    };
    Ok(Some(Video {
        fourcc,
        timescale,
        tkhd,
        elst,
        stbl,
    }))
}

impl Video<'_> {
    fn read(&self, length: u64, movie_timescale: u32) -> Result<Track> {
        let sizes = self.sizes()?;
        ensure!(!sizes.is_empty(), "the video track has no samples");
        let durations = self.durations(sizes.len())?;
        let samples = self.locate(&sizes, length)?;
        let (mode, variable_durations) = common_duration(&durations);
        let divisor = gcd(self.timescale, mode);
        let mut warnings = Vec::new();
        let orientation = orientation(self.tkhd, &mut warnings)?;
        let presented = match self.elst {
            Some(elst) => edit(
                elst,
                &durations,
                self.timescale,
                movie_timescale,
                &mut warnings,
            )?,
            None => 0..samples.len(),
        };
        ensure!(
            !presented.is_empty(),
            "the video track's edit list presents none of its {} samples",
            samples.len()
        );
        Ok(Track {
            fourcc: self.fourcc,
            fps: (self.timescale / divisor, mode / divisor),
            variable_durations,
            samples: samples[presented].to_vec(),
            creation_time: None,
            orientation,
            warnings,
            metadata: Vec::new(),
        })
    }
    fn table(&self, kind: &[u8; 4]) -> Option<&[u8]> {
        self.stbl
            .iter()
            .find(|(k, _)| k == kind)
            .map(|(_, body)| *body)
    }
    /// Each sample's size, from `stsz` or `stz2`.
    fn sizes(&self) -> Result<Vec<u32>> {
        if let Some(stsz) = self.table(b"stsz") {
            let size = u32_at(stsz, 4, b"stsz")?;
            let count = sample_count(u32_at(stsz, 8, b"stsz")?)?;
            if size != 0 {
                return Ok(vec![size; count]);
            }
            return Ok(entries(stsz, 12, count, 4, b"stsz")?
                .chunks_exact(4)
                .map(be32)
                .collect());
        }
        let stz2 = self
            .table(b"stz2")
            .context("the video track has no sample sizes (`stsz` or `stz2`)")?;
        let field = *stz2.get(7).context("truncated `stz2` atom")?;
        let count = sample_count(u32_at(stz2, 8, b"stz2")?)?;
        Ok(match field {
            4 => entries(stz2, 12, count.div_ceil(2), 1, b"stz2")?
                .iter()
                .flat_map(|&b| [u32::from(b >> 4), u32::from(b & 15)])
                .take(count)
                .collect(),
            8 => entries(stz2, 12, count, 1, b"stz2")?
                .iter()
                .map(|&b| u32::from(b))
                .collect(),
            16 => entries(stz2, 12, count, 2, b"stz2")?
                .chunks_exact(2)
                .map(|b| u32::from(u16::from_be_bytes([b[0], b[1]])))
                .collect(),
            _ => bail!("`stz2` field size {field} is not 4, 8 or 16 bits"),
        })
    }
    /// Each sample's duration, from `stts`; there must be one per sample.
    fn durations(&self, samples: usize) -> Result<Vec<u32>> {
        let stts = require(&self.stbl, b"stts", "the video track")?;
        let count = u32_at(stts, 4, b"stts")? as usize;
        let table = entries(stts, 8, count, 8, b"stts")?;
        let total: u64 = table.chunks_exact(8).map(|e| u64::from(be32(e))).sum();
        ensure!(
            total == samples as u64,
            "the video track's sample table disagrees: `stts` times {total} samples, the sizes \
             list {samples}"
        );
        let mut durations = Vec::with_capacity(samples);
        for entry in table.chunks_exact(8) {
            let duration = be32(&entry[4..]);
            ensure!(duration > 0, "the video track has a sample of duration 0");
            durations.resize(durations.len() + be32(entry) as usize, duration);
        }
        Ok(durations)
    }
    /// Each sample's place in the file, from the chunk offsets and `stsc`.
    fn locate(&self, sizes: &[u32], length: u64) -> Result<Vec<Sample>> {
        let chunks: Vec<u64> = if let Some(stco) = self.table(b"stco") {
            let count = u32_at(stco, 4, b"stco")? as usize;
            entries(stco, 8, count, 4, b"stco")?
                .chunks_exact(4)
                .map(|e| u64::from(be32(e)))
                .collect()
        } else {
            let co64 = self
                .table(b"co64")
                .context("the video track has no chunk offsets (`stco` or `co64`)")?;
            let count = u32_at(co64, 4, b"co64")? as usize;
            entries(co64, 8, count, 8, b"co64")?
                .chunks_exact(8)
                .map(be64)
                .collect()
        };
        let stsc = require(&self.stbl, b"stsc", "the video track")?;
        let count = u32_at(stsc, 4, b"stsc")? as usize;
        let runs: Vec<(usize, usize)> = entries(stsc, 8, count, 12, b"stsc")?
            .chunks_exact(12)
            .map(|e| (be32(e) as usize, be32(&e[4..]) as usize))
            .collect();
        ensure!(
            runs.first().is_some_and(|&(first, _)| first == 1)
                && runs.windows(2).all(|w| w[0].0 < w[1].0)
                && runs.last().is_some_and(|&(last, _)| last <= chunks.len()),
            "the video track's `stsc` does not list runs of its {} chunks from chunk 1 in order",
            chunks.len()
        );
        let mut samples = Vec::with_capacity(sizes.len());
        let mut sizes_left = sizes.iter();
        for (i, &(first, per_chunk)) in runs.iter().enumerate() {
            let end = runs.get(i + 1).map_or(chunks.len(), |&(next, _)| next - 1);
            for &chunk in &chunks[first - 1..end] {
                let mut offset = chunk;
                for _ in 0..per_chunk {
                    let Some(&size) = sizes_left.next() else {
                        bail!(
                            "the video track's sample table disagrees: its chunks hold more than \
                             the {} samples it sizes",
                            sizes.len()
                        );
                    };
                    let end = offset.saturating_add(u64::from(size));
                    ensure!(
                        end <= length,
                        "sample {} (bytes {offset}..{end}) lies beyond the end of the file at \
                         {length} bytes (a truncated file?)",
                        samples.len()
                    );
                    samples.push(Sample { offset, size });
                    offset = end;
                }
            }
        }
        ensure!(
            samples.len() == sizes.len(),
            "the video track's sample table disagrees: its chunks hold {} of the {} samples it \
             sizes",
            samples.len(),
            sizes.len()
        );
        Ok(samples)
    }
}

/// The most common duration, ignoring the last sample (often cut short), and whether
/// any other occurs. Ties go to the shorter duration.
fn common_duration(durations: &[u32]) -> (u32, bool) {
    let body = &durations[..durations.len().saturating_sub(1).max(1)];
    let mut counts = std::collections::BTreeMap::<u32, usize>::new();
    body.iter()
        .for_each(|&d| *counts.entry(d).or_default() += 1);
    let mode = counts
        .iter()
        .rev()
        .max_by_key(|&(_, &count)| count)
        .map_or(1, |(&duration, _)| duration);
    (mode, counts.len() > 1)
}

/// The samples the first non-empty edit of `elst` presents: those starting within
/// the edit. Other edits are reported in `warnings` and ignored.
fn edit(
    elst: &[u8],
    durations: &[u32],
    timescale: u32,
    movie_timescale: u32,
    warnings: &mut Vec<String>,
) -> Result<std::ops::Range<usize>> {
    let version = *elst.first().context("truncated `elst` atom")?;
    let count = u32_at(elst, 4, b"elst")? as usize;
    // Segment duration (movie timescale), media time (media timescale), rate (16.16).
    let edits: Vec<(u64, i64, i32)> = match version {
        0 => entries(elst, 8, count, 12, b"elst")?
            .chunks_exact(12)
            .map(|e| {
                let time = be32(&e[4..]) as i32;
                (u64::from(be32(e)), i64::from(time), be32(&e[8..]) as i32)
            })
            .collect(),
        1 => entries(elst, 8, count, 20, b"elst")?
            .chunks_exact(20)
            .map(|e| (be64(e), be64(&e[8..]) as i64, be32(&e[16..]) as i32))
            .collect(),
        _ => bail!("`elst` version {version} is not supported"),
    };
    if edits.len() > 1 {
        warnings.push(format!(
            "the video track's edit list has {} edits; only the first that shows media is \
             honoured",
            edits.len()
        ));
    }
    if edits.first().is_some_and(|&(_, time, _)| time < 0) {
        warnings.push(
            "the video track's edit list starts with an empty edit (a delay), which is ignored"
                .into(),
        );
    }
    let Some(&(duration, start, rate)) = edits.iter().find(|&&(_, time, _)| time >= 0) else {
        if !edits.is_empty() {
            warnings.push(
                "the video track's edit list shows no media; every sample is presented".into(),
            );
        }
        return Ok(0..durations.len());
    };
    if rate != 0x1_0000 {
        warnings.push(format!(
            "the video track's edit plays at rate {}; frames are read at normal speed",
            f64::from(rate) / 65536.
        ));
    }
    // Compare in the media timescale scaled by the movie's, exactly.
    let start = u128::from(start as u64) * u128::from(movie_timescale);
    let end = match (duration, movie_timescale) {
        (0, _) => u128::MAX,
        (_, 0) => bail!("the movie's timescale is 0"),
        _ => start + u128::from(duration) * u128::from(timescale),
    };
    let mut time = 0u128;
    let mut first = durations.len();
    let mut last = durations.len();
    for (i, &duration) in durations.iter().enumerate() {
        if time >= end {
            last = i;
            break;
        }
        if time >= start && first == durations.len() {
            first = i;
        }
        time += u128::from(duration) * u128::from(movie_timescale);
    }
    Ok(first..last.max(first))
}

/// The UTF-8 items of a `meta` atom whose keys use the `mdta` namespace (the
/// QuickTime metadata scheme). Other value types, such as binary tables, are skipped.
fn text_metadata(meta: &[u8]) -> Result<Vec<(String, String)>> {
    // QuickTime's `meta` holds its children directly; the ISO form has a version
    // and flags first.
    let body = match meta.get(4..8) {
        Some(b"hdlr") => meta,
        _ => meta.get(4..).context("truncated `meta` atom")?,
    };
    let atoms = children(body)?;
    let (Some((_, keys)), Some((_, ilst))) = (
        atoms.iter().find(|(kind, _)| kind == b"keys"),
        atoms.iter().find(|(kind, _)| kind == b"ilst"),
    ) else {
        return Ok(Vec::new());
    };
    // Version and flags, the count, then sized entries: namespace, key.
    let count = u32_at(keys, 4, b"keys")? as usize;
    let mut names = Vec::with_capacity(count.min(1024));
    let mut rest = keys.get(8..).context("truncated `keys` atom")?;
    for _ in 0..count {
        let size = rest.get(..4).map(be32).context("truncated `keys` atom")? as usize;
        let entry = rest.get(8..size.max(8)).context("truncated `keys` atom")?;
        names.push(String::from_utf8_lossy(entry).into_owned());
        rest = &rest[size.max(8)..];
    }
    let mut items = Vec::new();
    // Each item's type is the 1-based index of its key.
    for (index, item) in children(ilst)? {
        let Some(name) = names.get((be32(&index) as usize).wrapping_sub(1)) else {
            continue;
        };
        for (kind, data) in children(item)? {
            // Type indicator (well-known type 1 is UTF-8), locale, then the value.
            if kind == *b"data" && data.len() >= 8 && be32(data) == 1 {
                let value = String::from_utf8_lossy(&data[8..]);
                items.push((name.clone(), value.trim_end_matches('\0').to_owned()));
            }
        }
    }
    Ok(items)
}

/// The creation time and timescale of a `mvhd` or `mdhd`.
fn times(body: &[u8], kind: &[u8; 4]) -> Result<(u64, u32)> {
    match body.first() {
        Some(0) => Ok((u64::from(u32_at(body, 4, kind)?), u32_at(body, 12, kind)?)),
        Some(1) => Ok((u64_at(body, 4, kind)?, u32_at(body, 20, kind)?)),
        Some(version) => bail!("`{}` version {version} is not supported", name(kind)),
        None => bail!("truncated `{}` atom", name(kind)),
    }
}

/// The orientation of a `tkhd`'s display matrix: the turn or mirror that
/// shows the stored frames upright. QuickTime maps a stored point `(x, y)`
/// to `(a·x + c·y + tx, b·x + d·y + ty)`; only the signs of the 16.16
/// entries `a b c d` matter, since the translation just moves the turned
/// frame back to the origin. A matrix that is none of a rectangle's eight
/// symmetries (a skew, an arbitrary angle) shows the frames as stored and
/// adds a warning; scale is ignored.
fn orientation(tkhd: &[u8], warnings: &mut Vec<String>) -> Result<Orientation> {
    // Version and flags, the times, track ID and duration, then 8 reserved
    // bytes and layer, alternate group, volume and 2 reserved: 40 bytes in
    // version 0, 52 with version 1's 64-bit times and duration.
    let at = match tkhd.first() {
        Some(0) => 40,
        Some(1) => 52,
        Some(version) => bail!("`tkhd` version {version} is not supported"),
        None => bail!("truncated `tkhd` atom"),
    };
    let entry = |i: usize| -> Result<f64> {
        Ok(f64::from(u32_at(tkhd, at + 4 * i, b"tkhd")? as i32) / 65536.0)
    };
    // Row-major a b u / c d v / tx ty w.
    let (a, b, c, d) = (entry(0)?, entry(1)?, entry(3)?, entry(4)?);
    let sign = |v: f64| i8::from(v > 0.0) - i8::from(v < 0.0);
    Ok(match (sign(a), sign(b), sign(c), sign(d)) {
        (1, 0, 0, 1) => Orientation::Identity,
        (-1, 0, 0, 1) => Orientation::MirrorHorizontal,
        (-1, 0, 0, -1) => Orientation::Rotate180,
        (1, 0, 0, -1) => Orientation::MirrorVertical,
        (0, 1, 1, 0) => Orientation::Transpose,
        (0, 1, -1, 0) => Orientation::Rotate90,
        (0, -1, -1, 0) => Orientation::Transverse,
        (0, -1, 1, 0) => Orientation::Rotate270,
        _ => {
            warnings.push(format!(
                "the video track's display matrix [{a} {b}; {c} {d}] is not a quarter turn \
                 or mirror, so the frames are shown as stored"
            ));
            Orientation::Identity
        }
    })
}

/// The child atoms of `data`.
fn children(data: &[u8]) -> Result<Vec<Atom<'_>>> {
    let mut atoms = Vec::new();
    let mut rest = data;
    while !rest.is_empty() {
        if rest.len() < 8 {
            // QuickTime allows a 32-bit zero after the last child.
            ensure!(rest.iter().all(|&b| b == 0), "truncated atom header");
            break;
        }
        let kind: [u8; 4] = rest[4..8].try_into()?;
        let (header, size) = match be32(rest) {
            1 => (16, u64_at(rest, 8, &kind)?),
            0 => (8, rest.len() as u64),
            size => (8, u64::from(size)),
        };
        ensure!(
            header as u64 <= size && size <= rest.len() as u64,
            "truncated `{}` atom: it claims {size} bytes and {} are left",
            name(&kind),
            rest.len()
        );
        atoms.push((kind, &rest[header..size as usize]));
        rest = &rest[size as usize..];
    }
    Ok(atoms)
}

fn require<'a>(atoms: &[Atom<'a>], kind: &[u8; 4], parent: &str) -> Result<&'a [u8]> {
    atoms
        .iter()
        .find(|(k, _)| k == kind)
        .map(|(_, body)| *body)
        .with_context(|| format!("{parent} has no `{}` atom", name(kind)))
}

/// `count` entries of `width` bytes at `at` in `kind`'s body.
fn entries<'a>(
    body: &'a [u8],
    at: usize,
    count: usize,
    width: usize,
    kind: &[u8; 4],
) -> Result<&'a [u8]> {
    count
        .checked_mul(width)
        .and_then(|bytes| body.get(at..at.checked_add(bytes)?))
        .with_context(|| {
            format!(
                "truncated `{}` atom: {count} entries do not fit in {} bytes",
                name(kind),
                body.len()
            )
        })
}

fn sample_count(count: u32) -> Result<usize> {
    ensure!(
        u64::from(count) <= MAX_SAMPLES,
        "the video track lists {count} samples, more than the {MAX_SAMPLES} this reader accepts"
    );
    Ok(count as usize)
}

/// The big-endian `u32` at the start of `bytes`, which holds at least four.
fn be32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
/// The big-endian `u64` at the start of `bytes`, which holds at least eight.
fn be64(bytes: &[u8]) -> u64 {
    (u64::from(be32(bytes)) << 32) | u64::from(be32(&bytes[4..]))
}
fn u32_at(body: &[u8], at: usize, kind: &[u8; 4]) -> Result<u32> {
    Ok(be32(entries(body, at, 1, 4, kind)?))
}
fn u64_at(body: &[u8], at: usize, kind: &[u8; 4]) -> Result<u64> {
    Ok(be64(entries(body, at, 1, 8, kind)?))
}

/// A fourcc for messages, with non-ASCII bytes escaped.
fn name(kind: &[u8; 4]) -> String {
    kind.escape_ascii().to_string()
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn prores_raw(fourcc: &[u8; 4]) -> bool {
        matches!(fourcc, b"aprn" | b"aprh")
    }
    fn read(path: &Path) -> Result<Track> {
        read_video_track(&File::open(path)?, prores_raw)
    }
    fn message(result: Result<Track>) -> String {
        format!("{:#}", result.expect_err("the movie should be rejected"))
    }
    /// Sample `i`'s bytes in the hand-built movies: `size` copies of `i`.
    fn payload(i: usize, size: u32) -> Vec<u8> {
        vec![i as u8; size as usize]
    }
    fn assert_samples(path: &Path, track: &Track, first: usize, sizes: &[u32]) {
        let file = File::open(path).unwrap();
        assert_eq!(track.samples.len(), sizes.len());
        for (i, (sample, &size)) in track.samples.iter().zip(sizes).enumerate() {
            assert_eq!(sample.size, size);
            let mut data = vec![0; size as usize];
            read_at(&file, sample.offset, &mut data).unwrap();
            assert_eq!(data, payload(first + i, size), "sample {i}");
        }
    }

    // Hand-built movies.

    fn atom(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend(kind);
        out.extend(body);
        out
    }
    fn full(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
        let mut head = (flags | u32::from(version) << 24).to_be_bytes().to_vec();
        head.extend(body);
        atom(kind, &head)
    }
    fn words(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_be_bytes()).collect()
    }

    #[derive(Clone)]
    enum Sizes {
        Stsz,
        Stz2(u8),
    }
    #[derive(Clone)]
    struct Spec {
        handler: [u8; 4],
        fourcc: [u8; 4],
        enabled: bool,
        timescale: u32,
        /// Per-sample durations, coded as runs.
        durations: Vec<u32>,
        sizes: Vec<u32>,
        size_table: Sizes,
        /// Samples per chunk, in order.
        chunks: Vec<u32>,
        co64: bool,
        /// (segment duration, media time).
        edits: Vec<(u32, i32)>,
        tkhd_version: u8,
        /// The display matrix's `a b c d`, 16.16.
        matrix: [i32; 4],
    }
    impl Default for Spec {
        fn default() -> Self {
            Self {
                handler: *b"vide",
                fourcc: *b"aprh",
                enabled: true,
                timescale: 600,
                durations: vec![20; 5],
                sizes: vec![3, 5, 7, 2, 4],
                size_table: Sizes::Stsz,
                chunks: vec![2, 2, 1],
                co64: false,
                edits: Vec::new(),
                tkhd_version: 0,
                matrix: [0x1_0000, 0, 0, 0x1_0000],
            }
        }
    }
    /// A `trak` whose samples start at byte `data` of the file, back to back.
    fn trak(spec: &Spec, data: u64) -> Vec<u8> {
        // Version and flags, then the fields before the matrix (36 or 48
        // bytes), the matrix and the width and height.
        let at = if spec.tkhd_version == 0 { 40 } else { 52 };
        let mut tkhd = vec![0; at + 44];
        tkhd[..4].copy_from_slice(
            &(u32::from(spec.enabled) | u32::from(spec.tkhd_version) << 24).to_be_bytes(),
        );
        let [a, b, c, d] = spec.matrix;
        let matrix = [a, b, 0, c, d, 0, 0, 0, 0x4000_0000];
        for (i, v) in matrix.iter().enumerate() {
            tkhd[at + 4 * i..][..4].copy_from_slice(&v.to_be_bytes());
        }
        let mut body = atom(b"tkhd", &tkhd);
        if !spec.edits.is_empty() {
            let mut elst = words(&[0, spec.edits.len() as u32]);
            for &(duration, time) in &spec.edits {
                elst.extend(words(&[duration, time as u32, 0x1_0000]));
            }
            body.extend(atom(b"edts", &atom(b"elst", &elst)));
        }
        let mut stsd_entry = vec![0; 78];
        stsd_entry[7] = 1;
        let mut stbl = full(
            b"stsd",
            0,
            0,
            &[words(&[1]), atom(&spec.fourcc, &stsd_entry)].concat(),
        );
        let mut runs: Vec<(u32, u32)> = Vec::new();
        for &d in &spec.durations {
            match runs.last_mut() {
                Some((count, duration)) if *duration == d => *count += 1,
                _ => runs.push((1, d)),
            }
        }
        let mut stts = words(&[runs.len() as u32]);
        runs.iter().for_each(|&(c, d)| stts.extend(words(&[c, d])));
        stbl.extend(full(b"stts", 0, 0, &stts));
        let mut stsc = Vec::new();
        let mut count = 0;
        for (i, &n) in spec.chunks.iter().enumerate() {
            if i == 0 || spec.chunks[i - 1] != n {
                stsc.extend(words(&[i as u32 + 1, n, 1]));
                count += 1;
            }
        }
        stbl.extend(full(b"stsc", 0, 0, &[words(&[count]), stsc].concat()));
        let n = spec.sizes.len() as u32;
        stbl.extend(match spec.size_table {
            Sizes::Stsz => full(
                b"stsz",
                0,
                0,
                &[words(&[0, n]), words(&spec.sizes)].concat(),
            ),
            Sizes::Stz2(bits) => {
                let table: Vec<u8> = match bits {
                    4 => spec
                        .sizes
                        .chunks(2)
                        .map(|p| (p[0] << 4 | p.get(1).copied().unwrap_or(0)) as u8)
                        .collect(),
                    8 => spec.sizes.iter().map(|&s| s as u8).collect(),
                    _ => spec
                        .sizes
                        .iter()
                        .flat_map(|&s| (s as u16).to_be_bytes())
                        .collect(),
                };
                full(
                    b"stz2",
                    0,
                    0,
                    &[vec![0, 0, 0, bits], words(&[n]), table].concat(),
                )
            }
        });
        let mut offsets = Vec::new();
        let mut at = data;
        let mut sizes = spec.sizes.iter();
        for &n in &spec.chunks {
            offsets.push(at);
            at += sizes
                .by_ref()
                .take(n as usize)
                .map(|&s| u64::from(s))
                .sum::<u64>();
        }
        stbl.extend(if spec.co64 {
            let mut table = words(&[offsets.len() as u32]);
            offsets.iter().for_each(|o| table.extend(o.to_be_bytes()));
            full(b"co64", 0, 0, &table)
        } else {
            let table: Vec<u32> = offsets.iter().map(|&o| o as u32).collect();
            full(
                b"stco",
                0,
                0,
                &[words(&[table.len() as u32]), words(&table)].concat(),
            )
        });
        let mut minf = full(b"vmhd", 0, 1, &[0; 8]);
        minf.extend(atom(b"stbl", &stbl));
        let mut mdia = full(b"mdhd", 0, 0, &words(&[0, 0, spec.timescale, 0, 0]));
        mdia.extend(full(
            b"hdlr",
            0,
            0,
            &[b"mhlr".as_slice(), &spec.handler, &[0; 13]].concat(),
        ));
        mdia.extend(atom(b"minf", &minf));
        body.extend(atom(b"mdia", &mdia));
        atom(b"trak", &body)
    }
    /// ftyp, wide, mdat holding every track's samples in turn, then moov.
    fn write_movie(dir: &Path, tracks: &[Spec], creation: u32) -> PathBuf {
        let mut file = atom(b"ftyp", b"qt  \0\0\x02\0qt  ");
        file.extend(atom(b"wide", &[]));
        let mut data = Vec::new();
        let mut starts = Vec::new();
        for spec in tracks {
            starts.push(file.len() as u64 + 8 + data.len() as u64);
            for (i, &size) in spec.sizes.iter().enumerate() {
                data.extend(payload(i, size));
            }
        }
        file.extend(atom(b"mdat", &data));
        let mut moov = full(b"mvhd", 0, 0, &words(&[creation, 0, 600, 0]));
        for (spec, &start) in tracks.iter().zip(&starts) {
            moov.extend(trak(spec, start));
        }
        file.extend(atom(b"moov", &moov));
        let path = dir.join("hand.mov");
        std::fs::write(&path, file).unwrap();
        path
    }
    fn movie(tracks: &[Spec]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = write_movie(dir.path(), tracks, 0);
        (dir, path)
    }

    #[test]
    fn chunks_of_several_samples_resolve_after_wide() {
        let spec = Spec::default();
        let dir = tempfile::tempdir().unwrap();
        let path = write_movie(dir.path(), std::slice::from_ref(&spec), 3_000_000_000);
        let track = read(&path).unwrap();
        assert_eq!(track.fourcc, *b"aprh");
        assert_eq!(track.fps, (30, 1));
        assert!(!track.variable_durations);
        assert_eq!(track.creation_time, Some(3_000_000_000));
        assert_samples(&path, &track, 0, &spec.sizes);
    }
    #[test]
    fn stco_and_co64_agree() {
        let (_dir, stco) = movie(&[Spec::default()]);
        let co64 = Spec {
            co64: true,
            ..Spec::default()
        };
        let (_dir64, co64) = movie(&[co64]);
        assert_eq!(read(&stco).unwrap(), read(&co64).unwrap());
    }
    #[test]
    fn compact_sizes_of_every_field_width() {
        for bits in [4, 8, 16] {
            let spec = Spec {
                size_table: Sizes::Stz2(bits),
                ..Spec::default()
            };
            let (_dir, path) = movie(std::slice::from_ref(&spec));
            let track = read(&path).unwrap();
            assert_samples(&path, &track, 0, &spec.sizes);
        }
    }
    #[test]
    fn edit_list_skips_samples_before_the_first_edit() {
        // Media time 15 falls inside sample 0: it starts before the edit, so it is
        // not presented (as the a7S III clip's 966 inside a 1001 sample).
        let spec = Spec {
            edits: vec![(80, 15)],
            ..Spec::default()
        };
        let (_dir, path) = movie(std::slice::from_ref(&spec));
        let track = read(&path).unwrap();
        assert!(track.warnings.is_empty());
        assert_samples(&path, &track, 1, &spec.sizes[1..]);
        // The edit's end bounds the samples too: 20 units from 20 is sample 1 only.
        let spec = Spec {
            edits: vec![(20, 20)],
            ..Spec::default()
        };
        let (_dir, path) = movie(std::slice::from_ref(&spec));
        assert_samples(&path, &read(&path).unwrap(), 1, &spec.sizes[1..2]);
    }
    #[test]
    fn odd_edit_lists_use_the_first_non_empty_edit_and_warn() {
        let spec = Spec {
            edits: vec![(40, -1), (40, 40), (20, 0)],
            ..Spec::default()
        };
        let (_dir, path) = movie(std::slice::from_ref(&spec));
        let track = read(&path).unwrap();
        assert_samples(&path, &track, 2, &spec.sizes[2..4]);
        assert_eq!(track.warnings.len(), 2, "{:?}", track.warnings);
        assert!(track.warnings[0].contains("3 edits"));
        assert!(track.warnings[1].contains("empty edit"));
        let past = Spec {
            edits: vec![(20, 1000)],
            ..Spec::default()
        };
        let (_dir, path) = movie(&[past]);
        assert!(message(read(&path)).contains("presents none"));
    }
    #[test]
    fn the_most_common_duration_sets_the_rate() {
        // As the iPhone: 20 units at 600 per second with an occasional 21; the
        // last sample's odd duration does not count.
        let spec = Spec {
            durations: vec![20, 20, 21, 20, 7],
            ..Spec::default()
        };
        let (_dir, path) = movie(&[spec]);
        let track = read(&path).unwrap();
        assert_eq!(track.fps, (30, 1));
        assert!(track.variable_durations);
        let last_only = Spec {
            durations: vec![20, 20, 20, 20, 7],
            ..Spec::default()
        };
        let (_dir, path) = movie(&[last_only]);
        assert!(!read(&path).unwrap().variable_durations);
    }
    #[test]
    fn other_tracks_and_disabled_video_are_skipped() {
        let sound = Spec {
            handler: *b"soun",
            fourcc: *b"lpcm",
            ..Spec::default()
        };
        let timecode = Spec {
            handler: *b"tmcd",
            fourcc: *b"tmcd",
            ..Spec::default()
        };
        let disabled = Spec {
            enabled: false,
            fourcc: *b"aprn",
            ..Spec::default()
        };
        let video = Spec {
            sizes: vec![9, 8, 7, 6, 5],
            ..Spec::default()
        };
        let (_dir, path) = movie(&[sound.clone(), timecode, disabled, video.clone()]);
        let track = read(&path).unwrap();
        assert_eq!(track.fourcc, *b"aprh");
        assert_eq!(
            track.samples.iter().map(|s| s.size).collect::<Vec<_>>(),
            video.sizes
        );
        let (_dir, path) = movie(&[sound]);
        assert_eq!(message(read(&path)), "the movie has no enabled video track");
    }
    #[test]
    fn display_matrices_set_the_orientation() {
        const ONE: i32 = 0x1_0000;
        use Orientation::*;
        for (matrix, orientation) in [
            ([ONE, 0, 0, ONE], Identity),
            ([-ONE, 0, 0, ONE], MirrorHorizontal),
            ([-ONE, 0, 0, -ONE], Rotate180),
            ([ONE, 0, 0, -ONE], MirrorVertical),
            ([0, ONE, ONE, 0], Transpose),
            // A phone held upright: the stored frame turns a quarter clockwise.
            ([0, ONE, -ONE, 0], Rotate90),
            ([0, -ONE, -ONE, 0], Transverse),
            ([0, -ONE, ONE, 0], Rotate270),
            // Scale is ignored.
            ([-2 * ONE, 0, 0, -ONE / 2], Rotate180),
        ] {
            for tkhd_version in [0, 1] {
                let (_dir, path) = movie(&[Spec {
                    matrix,
                    tkhd_version,
                    ..Spec::default()
                }]);
                let track = read(&path).unwrap();
                assert_eq!(
                    track.orientation, orientation,
                    "{matrix:?} in tkhd version {tkhd_version}"
                );
                assert!(track.warnings.is_empty(), "{:?}", track.warnings);
            }
        }
    }

    #[test]
    fn orientations_have_their_exif_values() {
        use Orientation::*;
        let all = [
            Identity,
            MirrorHorizontal,
            Rotate180,
            MirrorVertical,
            Transpose,
            Rotate90,
            Transverse,
            Rotate270,
        ];
        for (i, orientation) in all.into_iter().enumerate() {
            assert_eq!(orientation.exif() as usize, i + 1);
            assert_eq!(orientation.swaps_dimensions(), i >= 4);
        }
    }

    #[test]
    fn other_display_matrices_show_frames_as_stored() {
        // 45 degrees.
        let (_dir, path) = movie(&[Spec {
            matrix: [46341, 46341, -46341, 46341],
            ..Spec::default()
        }]);
        let track = read(&path).unwrap();
        assert_eq!(track.orientation, Orientation::Identity);
        assert_eq!(track.warnings.len(), 1, "{:?}", track.warnings);
        assert!(
            track.warnings[0].contains("not a quarter turn or mirror"),
            "{}",
            track.warnings[0]
        );
    }

    #[test]
    fn other_codecs_are_named() {
        let spec = Spec {
            fourcc: *b"apch",
            ..Spec::default()
        };
        let (_dir, path) = movie(&[spec]);
        assert!(message(read(&path)).contains("the video track is `apch`"));
    }
    #[test]
    fn broken_files_are_described() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_movie(dir.path(), &[Spec::default()], 0);
        let data = std::fs::read(&path).unwrap();
        // Cut inside the moov: the moov atom claims more than the file holds.
        std::fs::write(&path, &data[..data.len() - 10]).unwrap();
        let text = message(read(&path));
        assert!(text.contains("truncated `moov`"), "{text}");
        // A moov whose child claims more than the moov holds.
        let moov = data.len() - data.windows(4).rev().position(|w| w == b"moov").unwrap() - 4 - 4;
        let mut inner = data.clone();
        let stsz = inner.windows(4).position(|w| w == b"stsz").unwrap() - 4;
        inner[stsz..stsz + 4].copy_from_slice(&1000u32.to_be_bytes());
        std::fs::write(&path, &inner).unwrap();
        let text = message(read(&path));
        assert!(text.contains("truncated `stsz`"), "{text}");
        // No moov at all: the recording stopped before writing it.
        std::fs::write(&path, &data[..moov]).unwrap();
        assert!(message(read(&path)).contains("no `moov`"));
        // A sample beyond the end of the file.
        let mut beyond = data.clone();
        let stco = beyond.windows(4).position(|w| w == b"stco").unwrap() + 12;
        beyond[stco..stco + 4].copy_from_slice(&(data.len() as u32).to_be_bytes());
        std::fs::write(&path, &beyond).unwrap();
        assert!(message(read(&path)).contains("beyond the end of the file"));
        std::fs::write(&path, b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR").unwrap();
        assert!(message(read(&path)).contains("not a QuickTime movie"));
        std::fs::write(&path, b"").unwrap();
        assert!(message(read(&path)).contains("not a QuickTime movie"));
    }

    /// Text metadata is read from both forms of the `meta` atom: QuickTime's,
    /// whose children follow directly, and ISO's, which starts with a version
    /// and flags word.
    #[test]
    fn text_metadata_in_both_meta_forms() {
        let mut keys = words(&[0, 2]);
        for key in [b"com.apple.quicktime.make".as_slice(), b"com.example.table"] {
            keys.extend(words(&[key.len() as u32 + 8]));
            keys.extend(b"mdta");
            keys.extend(key);
        }
        let item = |index: u32, kind: u32, value: &[u8]| {
            let data = atom(b"data", &[words(&[kind, 0]), value.to_vec()].concat());
            atom(&index.to_be_bytes(), &data)
        };
        let ilst = [item(1, 1, b"Apple"), item(2, 0, &[1, 2, 3])].concat();
        let children = [
            full(
                b"hdlr",
                0,
                0,
                &[b"\0\0\0\0mdta".as_slice(), &[0; 12]].concat(),
            ),
            atom(b"keys", &keys),
            atom(b"ilst", &ilst),
        ]
        .concat();
        let expected = vec![("com.apple.quicktime.make".to_owned(), "Apple".to_owned())];
        assert_eq!(text_metadata(&children).unwrap(), expected);
        let iso = [vec![0; 4], children].concat();
        assert_eq!(text_metadata(&iso).unwrap(), expected);
    }

    /// The sample clips under `PRORES_RAW_FOOTAGE`: each first and last sample
    /// is a ProRes RAW frame (its size, then `prrf`).
    #[test]
    fn footage_clips() {
        let Some(root) = std::env::var_os("PRORES_RAW_FOOTAGE") else {
            return;
        };
        let root = PathBuf::from(root);
        for (clip, fourcc, frames, fps, variable, make) in [
            (
                "sony-fx6-prores_raw/FILMPLUS-ProResRaw-NinjaV-FX6-3.MOV",
                b"aprn",
                708,
                (50, 1),
                false,
                "Sony",
            ),
            (
                "lumix-s5-prores_raw/a7s III ProRes RAW HQ.mov",
                b"aprh",
                346,
                (24000, 1001),
                false,
                "Sony",
            ),
            (
                "iphone17-prores_raw/04_A001_09210311_C010.mov",
                b"aprh",
                712,
                (30, 1),
                true,
                "Apple Inc.",
            ),
        ] {
            let path = root.join(clip);
            if !path.is_file() {
                continue;
            }
            let file = File::open(&path).unwrap();
            let track = read_video_track(&file, prores_raw).unwrap();
            assert_eq!(&track.fourcc, fourcc, "{clip}");
            assert_eq!(track.samples.len(), frames, "{clip}");
            assert_eq!(track.fps, fps, "{clip}");
            assert_eq!(track.variable_durations, variable, "{clip}");
            assert!(track.creation_time.is_some(), "{clip}");
            assert!(track.warnings.is_empty(), "{clip}: {:?}", track.warnings);
            let found = track.text(&["com.apple.proapps.manufacturer", "com.apple.quicktime.make"]);
            assert_eq!(found, Some(make), "{clip}");
            assert_eq!(track.orientation, Orientation::Identity, "{clip}");
            for sample in [track.samples[0], track.samples[frames - 1]] {
                let mut head = [0; 8];
                read_at(&file, sample.offset, &mut head).unwrap();
                assert_eq!(be32(&head), sample.size, "{clip}");
                assert_eq!(&head[4..], b"prrf", "{clip}");
            }
        }
        // The Blackmagic Camera app stores the iPhone's 4:3 frames upside
        // down and turns them with the track matrix.
        let clip = "iphone17-prores_raw/01_A001_09211049_C068.mov";
        let path = root.join(clip);
        if path.is_file() {
            let track = read_video_track(&File::open(&path).unwrap(), prores_raw).unwrap();
            assert_eq!(track.orientation, Orientation::Rotate180, "{clip}");
        }
    }
}
