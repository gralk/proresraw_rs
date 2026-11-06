//! Decoding on a GPU through wgpu (feature `gpu`).
//!
//! The caller owns the device: build a [`GpuDecoder`] from it once, then
//! decode packets into storage buffers that stay on the GPU. The host
//! parses and validates the frame and tile headers; the GPU decodes the
//! component streams, bit-exactly with [`crate::decode_frame`], and
//! inverse-transforms the blocks with the Apple-compatible transform. Its
//! column pass is in `f32`; a driver may fuse a multiply and an add there,
//! so a few code values can differ from the host by one.
//!
//! Tiles with an invalid header are skipped as on the host: their sensels
//! are not written, so a fresh output buffer holds 0 there and a buffer
//! passed to [`GpuDecoder::decode_into`] keeps its contents.
//!
//! The output buffer holds the code values of the coded picture
//! ([`FrameHeader::coded_width`] × [`FrameHeader::coded_height`] sensels,
//! row-major) packed two per 32-bit word: the even column in the low 16
//! bits, the odd column in the high 16 bits. Apply the header curve to
//! reach linear values, as on the host.
//!
//! [`GpuDeveloper`] runs the raw conversion ([`crate::develop_planar`]) on
//! such a buffer, or on an uploaded host frame, into three `f32` planes
//! ([`GpuRgb`]) that stay on the GPU until downloaded.
//!
//! # Working with your own GPU memory
//!
//! [`GpuDecoder`] and [`GpuDeveloper`] are conveniences that own their
//! scratch and submit their own work. Underneath, one [`GpuPipelines`] per
//! device (shared behind an `Arc`) records into an encoder of yours with
//! buffers of yours:
//!
//! 1. Plan: [`decode_footprint`] (from [`crate::parse_header`] on the
//!    start of a packet) and [`develop_footprint`] list every buffer a
//!    frame needs as [`BufferRequest`]s, with when each is dead.
//! 2. Allocate, or hand out regions of larger blocks: storage bindings
//!    start at multiples of `min_storage_buffer_offset_alignment`, and are
//!    bound at exactly the requested size. All storage bindings are
//!    read-write, so they may share one buffer; the conversion's parameter
//!    block is a uniform and needs a buffer of its own.
//! 3. Record: [`GpuPipelines::record_decode`], then
//!    [`GpuPipelines::record_develop`] on its output, into one encoder,
//!    and submit when you choose. Uploads go through `queue.write_buffer`,
//!    which lands before the commands of the next submission, so frames
//!    recorded into one submission need their own upload buffers.
//!
//! The conversion runs in row bands, so its green-pass scratch is
//! [`develop_row_bytes`] per row of a band plus [`DEVELOP_GREEN_HALO`]
//! rows each side; [`default_band_rows`] picks the tallest band the device
//! allows, and [`DEVELOP_INPUT_HALO`] gives a planner the rows of code
//! values a band reads beyond its edges.

use std::fmt;
use std::sync::Arc;

mod develop;

pub use develop::{
    DEVELOP_GREEN_HALO, DEVELOP_INPUT_HALO, DevelopBuffers, DevelopFootprint, GpuDeveloper, GpuRgb,
    default_band_rows, develop_footprint, develop_row_bytes, plane_bytes,
};

use crate::codebook::{
    AC_CB, AC_INITIAL, Codebook, DC_CB, DC_RESET, FIRST_DC, LN_CB, LN_INITIAL, RN_CB, RN_INITIAL,
    SCAN,
};
use crate::decoder::{Frame, ParsedFrame, parse_frame};
use crate::error::Error;
use crate::header::FrameHeader;
use crate::idct::{BIAS, COS4, ROW_SET, ROW_TABLES, SCALE, TAN1, TAN2, TAN3};
use crate::tile::{TILE_UNIT, Tile};

/// Invocations per workgroup of the stream pass.
const STREAM_GROUP: u32 = 64;
/// Bytes of zero padding after the packet, so that the two-word window
/// never reads past the buffer.
const PACKET_PADDING: usize = 8;
/// 32-bit words of one stream table entry.
const STREAM_WORDS: usize = 4;
/// Coefficient words of one 16×16 square: four blocks of 64 halves.
const SQUARE_WORDS: u64 = 128;
/// 32-bit words of the decoder's parameter block.
const PARAMS_WORDS: u64 = 8 + 64;

/// Why a frame could not be decoded on the GPU.
#[derive(Debug)]
pub enum GpuError {
    /// The packet is malformed.
    Format(Error),
    /// A buffer the frame needs exceeds a limit of the device.
    TooLarge {
        /// Which buffer.
        what: &'static str,
        /// Bytes it needs.
        bytes: u64,
        /// The device's limit.
        limit: u64,
    },
    /// A buffer binding is smaller than the request it serves.
    BufferTooSmall {
        /// Which buffer.
        what: &'static str,
        /// Bytes available in the binding.
        size: u64,
        /// Bytes the frame needs.
        needed: u64,
    },
    /// A buffer lacks a usage its request needs.
    MissingUsage {
        /// Which buffer.
        what: &'static str,
        /// The usages the request needs.
        needs: wgpu::BufferUsages,
    },
    /// A binding's offset is not a multiple of the device's alignment for
    /// its kind of binding.
    Misaligned {
        /// Which buffer.
        what: &'static str,
        /// The binding's offset.
        offset: u64,
        /// The alignment the device requires.
        alignment: u64,
    },
    /// Waiting for the GPU failed.
    Poll(wgpu::PollError),
    /// Mapping a buffer for reading failed.
    Map(wgpu::BufferAsyncError),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::Format(e) => e.fmt(f),
            GpuError::TooLarge { what, bytes, limit } => {
                write!(f, "{what} needs {bytes} bytes, the device allows {limit}")
            }
            GpuError::BufferTooSmall { what, size, needed } => {
                write!(
                    f,
                    "{what} binding of {size} bytes, the frame needs {needed}"
                )
            }
            GpuError::MissingUsage { what, needs } => {
                write!(f, "{what} buffer lacks usage {needs:?}")
            }
            GpuError::Misaligned {
                what,
                offset,
                alignment,
            } => write!(
                f,
                "{what} binding at offset {offset}, not a multiple of {alignment}"
            ),
            GpuError::Poll(e) => write!(f, "waiting for the GPU: {e}"),
            GpuError::Map(e) => write!(f, "reading back from the GPU: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<Error> for GpuError {
    fn from(e: Error) -> Self {
        GpuError::Format(e)
    }
}

/// Codebook tables and the scan as WGSL constants, from the host's tables.
fn tables() -> String {
    fn list(values: impl Iterator<Item = u32>) -> String {
        values
            .map(|v| format!("{v}u"))
            .collect::<Vec<_>>()
            .join(", ")
    }
    fn table(name: &str, cbs: &[Codebook]) -> String {
        format!(
            "const {name} = array<u32, {}>({});\n",
            cbs.len(),
            list(cbs.iter().map(|c| u32::from(c.0)))
        )
    }
    let mut out = String::new();
    for (name, cb) in [
        ("FIRST_DC", FIRST_DC),
        ("DC_RESET", DC_RESET),
        ("AC_INITIAL", AC_INITIAL),
        ("RN_INITIAL", RN_INITIAL),
        ("LN_INITIAL", LN_INITIAL),
    ] {
        out += &format!("const {name} = {}u;\n", cb.0);
    }
    out += &table("DC_CB", &DC_CB);
    out += &table("AC_CB", &AC_CB);
    out += &table("RN_CB", &RN_CB);
    out += &table("LN_CB", &LN_CB);
    out += &format!(
        "const SCAN = array<u32, 64>({});\n",
        list(SCAN.iter().map(|&s| u32::from(s)))
    );
    out += &format!(
        "const ROW_SET = array<u32, 8>({});\n",
        list(ROW_SET.iter().map(|&s| s as u32))
    );
    let rows: Vec<String> = ROW_TABLES
        .iter()
        .flatten()
        .flatten()
        .map(|v| format!("{v}"))
        .collect();
    out += &format!("const ROW_TABLES = array<i32, 128>({});\n", rows.join(", "));
    for (name, v) in [
        ("SCALE", SCALE),
        ("BIAS", BIAS),
        ("TAN1", TAN1),
        ("TAN2", TAN2),
        ("TAN3", TAN3),
        ("COS4", COS4),
    ] {
        // Shortest round-trip decimal: parses back to the same f32.
        out += &format!("const {name} = {v:?}f;\n");
    }
    out
}

/// A growable GPU buffer.
struct Slot {
    buffer: Option<wgpu::Buffer>,
    label: &'static str,
    usage: wgpu::BufferUsages,
}

impl Slot {
    fn new(label: &'static str, usage: wgpu::BufferUsages) -> Self {
        Slot {
            buffer: None,
            label,
            usage,
        }
    }

    /// A buffer of at least `size` bytes, reallocated with headroom when
    /// the current one is too small.
    fn get(&mut self, device: &wgpu::Device, size: u64) -> &wgpu::Buffer {
        if self.buffer.as_ref().is_none_or(|b| b.size() < size) {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: (size + size / 4).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT),
                usage: self.usage,
                mapped_at_creation: false,
            }));
        }
        self.buffer.as_ref().expect("allocated above")
    }
}

/// Decodes ProRes RAW frames on one wgpu device.
///
/// Holds the pipelines and scratch buffers reused across frames; use one
/// decoder per thread. Frames decoded in a row on the same queue are
/// ordered, so scratch reuse is safe without waiting.
pub struct GpuDecoder {
    pipelines: Arc<GpuPipelines>,
    params: Slot,
    packet: Slot,
    streams: Slot,
    coefs: Slot,
    skipped: Slot,
    skipped_tiles: Vec<usize>,
    max_qscale: u8,
}

/// The compiled compute pipelines of the decoder and the raw conversion
/// for one device.
///
/// Compiling them is the expensive part of setting up GPU work; share one
/// set (behind an [`Arc`]) between every [`GpuDecoder`] and
/// [`GpuDeveloper`] on the device. They hold no per-frame state.
pub struct GpuPipelines {
    decode_layout: wgpu::BindGroupLayout,
    decode_streams: wgpu::ComputePipeline,
    transform: wgpu::ComputePipeline,
    develop: develop::Pipelines,
}

impl GpuPipelines {
    /// Compile the decoding and conversion pipelines for `device`.
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // Every binding is read-write, even those the shaders only read:
        // wgpu tracks usage per buffer within a dispatch, and a buffer
        // bound once read-only and once read-write conflicts. Read-write
        // throughout lets a caller sub-allocate every request from one
        // buffer, at no measurable cost.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prores raw"),
            entries: &[
                storage(0, false),
                storage(1, false),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prores raw"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let source = include_str!("gpu/decode.wgsl").replace("// TABLES", &tables());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prores raw"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        GpuPipelines {
            decode_streams: pipeline("decode_streams"),
            transform: pipeline("transform"),
            decode_layout: layout,
            develop: develop::Pipelines::new(device),
        }
    }
}

impl GpuDecoder {
    /// A decoder with its own pipelines, compiled for `device`. Prefer
    /// [`GpuDecoder::with_pipelines`] when several decoders or a
    /// [`GpuDeveloper`] share the device.
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_pipelines(Arc::new(GpuPipelines::new(device)))
    }

    /// A decoder using shared pipelines. Scratch buffers are allocated on
    /// the pipelines' device when the first frame is decoded.
    pub fn with_pipelines(pipelines: Arc<GpuPipelines>) -> Self {
        let copy_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        GpuDecoder {
            pipelines,
            params: Slot::new("prores raw params", copy_storage),
            packet: Slot::new("prores raw packet", copy_storage),
            streams: Slot::new("prores raw streams", copy_storage),
            coefs: Slot::new(
                "prores raw coefficients",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            skipped: Slot::new("prores raw skipped squares", copy_storage),
            skipped_tiles: Vec::new(),
            max_qscale: 0,
        }
    }

    /// Decode `packet` (one QuickTime sample) into a new buffer with
    /// `STORAGE | COPY_SRC` usage. The work is submitted to `queue`.
    pub fn decode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packet: &[u8],
    ) -> Result<GpuFrame, GpuError> {
        let header = crate::decoder::parse_header(packet)?;
        let samples = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prores raw samples"),
            size: sample_bytes(&header),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let info = self.submit(device, queue, packet, &samples)?;
        Ok(GpuFrame {
            header: info.header,
            samples,
            skipped_tiles: info.skipped_tiles,
            max_qscale: info.max_qscale,
        })
    }

    /// Largest tile qscale byte of the most recently submitted frame (see
    /// [`Frame::max_qscale`]).
    pub fn max_qscale(&self) -> u8 {
        self.max_qscale
    }

    /// The pipelines this decoder runs.
    pub fn pipelines(&self) -> &Arc<GpuPipelines> {
        &self.pipelines
    }

    /// Indices of the tiles of the most recently submitted frame that were
    /// skipped for an invalid tile header (see [`Frame::skipped_tiles`]).
    pub fn skipped_tiles(&self) -> &[usize] {
        &self.skipped_tiles
    }

    /// Decode `packet` into `samples`, a buffer with `STORAGE` usage of at
    /// least [`sample_bytes`] bytes, and return the frame header. The work
    /// is submitted to `queue`; later submissions on it see the result.
    pub fn decode_into(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packet: &[u8],
        samples: &wgpu::Buffer,
    ) -> Result<FrameHeader, GpuError> {
        Ok(self.submit(device, queue, packet, samples)?.header)
    }

    /// Record a decode into scratch from the slots and submit it.
    fn submit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packet: &[u8],
        samples: &wgpu::Buffer,
    ) -> Result<DecodeInfo, GpuError> {
        let header = crate::decoder::parse_header(packet)?;
        let f = decode_footprint(&header, packet.len());
        let buffers = DecodeBuffers {
            params: self
                .params
                .get(device, f.params.bytes)
                .as_entire_buffer_binding(),
            packet: self
                .packet
                .get(device, f.packet.bytes)
                .as_entire_buffer_binding(),
            streams: self
                .streams
                .get(device, f.streams.bytes)
                .as_entire_buffer_binding(),
            coefs: self
                .coefs
                .get(device, f.coefs.bytes)
                .as_entire_buffer_binding(),
            skipped: self
                .skipped
                .get(device, f.skipped.bytes)
                .as_entire_buffer_binding(),
            samples: samples.as_entire_buffer_binding(),
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prores raw decode"),
        });
        let info = self
            .pipelines
            .record_decode(device, queue, &mut encoder, packet, &buffers)?;
        queue.submit([encoder.finish()]);
        self.skipped_tiles.clone_from(&info.skipped_tiles);
        self.max_qscale = info.max_qscale;
        Ok(info)
    }

    /// Free the scratch buffers, which otherwise keep the size of the
    /// largest frame decoded so far. The next decode allocates them again,
    /// sized for its frame. Work already submitted is unaffected.
    pub fn release(&mut self) {
        for slot in [
            &mut self.params,
            &mut self.packet,
            &mut self.streams,
            &mut self.coefs,
            &mut self.skipped,
        ] {
            slot.buffer = None;
        }
    }
}

/// One buffer a GPU operation needs, as its footprint lists it: allocate a
/// buffer (or a region of one) of at least `bytes` with at least `usage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferRequest {
    /// What the buffer is for, usable as a debug label.
    pub label: &'static str,
    /// Bytes the operation reads or writes.
    pub bytes: u64,
    /// Usages the buffer must have.
    pub usage: wgpu::BufferUsages,
}

/// The buffers [`GpuPipelines::record_decode`] needs for one frame, in the
/// order of [`DecodeBuffers`].
///
/// Each binding must start at a multiple of the device's
/// `min_storage_buffer_offset_alignment`, and none may be larger than its
/// `max_storage_buffer_binding_size`; the decode binds exactly `bytes` of
/// each, so a larger region is fine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeFootprint {
    /// Geometry and quantization matrix. Uploaded.
    pub params: BufferRequest,
    /// The packet, zero-padded. Uploaded.
    pub packet: BufferRequest,
    /// The table of component streams, sized for every tile being valid.
    /// Uploaded.
    pub streams: BufferRequest,
    /// Dequantized coefficients: cleared and written by the stream pass,
    /// read by the transform pass, dead after it (2 bytes per sensel).
    pub coefs: BufferRequest,
    /// One bit per 16×16 square of a skipped tile. Uploaded.
    pub skipped: BufferRequest,
    /// The output code values ([`sample_bytes`]), written by the transform
    /// pass.
    pub samples: BufferRequest,
}

impl DecodeFootprint {
    /// The requests in [`DecodeBuffers`] order.
    pub fn requests(&self) -> [BufferRequest; 6] {
        [
            self.params,
            self.packet,
            self.streams,
            self.coefs,
            self.skipped,
            self.samples,
        ]
    }

    /// Total bytes of all requests.
    pub fn bytes(&self) -> u64 {
        self.requests().iter().map(|r| r.bytes).sum()
    }
}

/// What a frame with `header` in a packet of `packet_len` bytes needs from
/// the device to decode. The header comes from
/// [`crate::parse_header`] on the start of the packet, so a frame can be
/// planned before it is read in full.
pub fn decode_footprint(header: &FrameHeader, packet_len: usize) -> DecodeFootprint {
    use wgpu::BufferUsages as U;
    let layout = Layout::new(header);
    let tiles = crate::tile::TileGrid::new(
        usize::from(header.width),
        usize::from(header.height),
        header.alignment,
    )
    .len();
    let upload = U::STORAGE | U::COPY_DST;
    let request = |label, bytes: u64, usage| BufferRequest {
        label,
        bytes: bytes.max(4).next_multiple_of(4),
        usage,
    };
    DecodeFootprint {
        params: request("prores raw params", PARAMS_WORDS * 4, upload),
        // The upload ends with a padded word after the last whole one.
        packet: request(
            "prores raw packet",
            ((packet_len & !3) + 4 + PACKET_PADDING) as u64,
            upload,
        ),
        streams: request(
            "prores raw streams",
            (tiles * 4 * STREAM_WORDS * 4) as u64,
            upload,
        ),
        coefs: request("prores raw coefficients", layout.coef_bytes(), upload),
        skipped: request(
            "prores raw skipped squares",
            (layout.squares() as usize).div_ceil(32) as u64 * 4,
            upload,
        ),
        samples: request("prores raw samples", layout.sample_bytes(), U::STORAGE),
    }
}

/// Caller-provided buffers for [`GpuPipelines::record_decode`], one per
/// [`DecodeFootprint`] request. Bindings may be regions of larger buffers,
/// for example blocks of an arena.
#[derive(Debug, Clone)]
pub struct DecodeBuffers<'a> {
    /// See [`DecodeFootprint::params`].
    pub params: wgpu::BufferBinding<'a>,
    /// See [`DecodeFootprint::packet`].
    pub packet: wgpu::BufferBinding<'a>,
    /// See [`DecodeFootprint::streams`].
    pub streams: wgpu::BufferBinding<'a>,
    /// See [`DecodeFootprint::coefs`].
    pub coefs: wgpu::BufferBinding<'a>,
    /// See [`DecodeFootprint::skipped`].
    pub skipped: wgpu::BufferBinding<'a>,
    /// See [`DecodeFootprint::samples`].
    pub samples: wgpu::BufferBinding<'a>,
}

/// What [`GpuPipelines::record_decode`] learnt from the frame's headers.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeInfo {
    /// The frame header.
    pub header: FrameHeader,
    /// Indices of tiles skipped for an invalid header; their sensels are
    /// not written.
    pub skipped_tiles: Vec<usize>,
    /// Largest qscale byte over all tiles, as in [`Frame::max_qscale`].
    pub max_qscale: u8,
}

impl DecodeInfo {
    /// The frame quantizer, as [`Frame::quantizer`] gives it.
    pub fn quantizer(&self) -> u32 {
        crate::decoder::quantizer(&self.header, self.max_qscale)
    }
}

/// Check `binding` against `request` and return it narrowed to exactly
/// the requested bytes.
fn bind<'a>(
    binding: &wgpu::BufferBinding<'a>,
    request: &BufferRequest,
    alignment: u32,
) -> Result<wgpu::BufferBinding<'a>, GpuError> {
    let what = request.label;
    if !binding.buffer.usage().contains(request.usage) {
        return Err(GpuError::MissingUsage {
            what,
            needs: request.usage,
        });
    }
    let alignment = u64::from(alignment);
    if binding.offset % alignment != 0 {
        return Err(GpuError::Misaligned {
            what,
            offset: binding.offset,
            alignment,
        });
    }
    let rest = binding.buffer.size().saturating_sub(binding.offset);
    let size = binding.size.map_or(rest, |s| s.get().min(rest));
    if size < request.bytes {
        return Err(GpuError::BufferTooSmall {
            what,
            size,
            needed: request.bytes,
        });
    }
    Ok(wgpu::BufferBinding {
        buffer: binding.buffer,
        offset: binding.offset,
        size: wgpu::BufferSize::new(request.bytes),
    })
}

/// Fail when any request exceeds the device's storage binding limits.
fn check_limits(device: &wgpu::Device, requests: &[BufferRequest]) -> Result<(), GpuError> {
    let limits = device.limits();
    let limit = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
    for r in requests {
        if r.bytes > limit {
            return Err(GpuError::TooLarge {
                what: r.label,
                bytes: r.bytes,
                limit,
            });
        }
    }
    Ok(())
}

impl GpuPipelines {
    /// Record the decode of `packet` into `encoder`, reading and writing
    /// only `buffers` (see [`decode_footprint`]). Nothing is submitted, so
    /// a consumer can chain further passes in the same submission; the
    /// code values are in `buffers.samples` once it completes.
    ///
    /// The host parses and validates the headers here and uploads the
    /// packet, the stream table, the parameters and the skipped-square
    /// bitmap with `queue.write_buffer`. wgpu performs such writes before
    /// the commands of the next submission on `queue`, so decodes recorded
    /// into one submission need separate upload buffers; the coefficients
    /// and the output may be reused once the transform pass has run.
    pub fn record_decode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        packet: &[u8],
        buffers: &DecodeBuffers<'_>,
    ) -> Result<DecodeInfo, GpuError> {
        let parsed = parse_frame(packet)?;
        let footprint = decode_footprint(&parsed.header, packet.len());
        // Stream offsets are addressed in bits with 32-bit arithmetic.
        let bit_limit = u64::from(u32::MAX / 8);
        if footprint.packet.bytes > bit_limit {
            return Err(GpuError::TooLarge {
                what: "packet",
                bytes: footprint.packet.bytes,
                limit: bit_limit,
            });
        }
        check_limits(device, &footprint.requests())?;
        let alignment = device.limits().min_storage_buffer_offset_alignment;
        let b = |binding, request| bind(binding, request, alignment);
        let params_b = b(&buffers.params, &footprint.params)?;
        let packet_b = b(&buffers.packet, &footprint.packet)?;
        let streams_b = b(&buffers.streams, &footprint.streams)?;
        let coefs_b = b(&buffers.coefs, &footprint.coefs)?;
        let skipped_b = b(&buffers.skipped, &footprint.skipped)?;
        let samples_b = b(&buffers.samples, &footprint.samples)?;

        let layout = Layout::new(&parsed.header);
        let mut skipped_tiles = Vec::new();
        let (table, skipped_squares) = stream_table(packet, &parsed, &layout, &mut skipped_tiles);
        let streams = (table.len() / STREAM_WORDS) as u32;
        let params = params(&parsed.header, &layout, streams);
        queue.write_buffer(params_b.buffer, params_b.offset, &words_to_bytes(&params));
        if !table.is_empty() {
            queue.write_buffer(streams_b.buffer, streams_b.offset, &words_to_bytes(&table));
        }
        upload_padded(queue, packet_b.buffer, packet_b.offset, packet);
        queue.write_buffer(
            skipped_b.buffer,
            skipped_b.offset,
            &words_to_bytes(&skipped_squares),
        );

        let (coefs_buffer, coefs_offset) = (coefs_b.buffer, coefs_b.offset);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prores raw frame"),
            layout: &self.decode_layout,
            entries: &[
                binding_entry(0, params_b),
                binding_entry(1, packet_b),
                binding_entry(2, streams_b),
                binding_entry(3, coefs_b),
                binding_entry(4, samples_b),
                binding_entry(5, skipped_b),
            ],
        });
        // The stream pass ORs coefficients into zeroed words; one fill is
        // much cheaper than each invocation zeroing its scattered blocks.
        encoder.clear_buffer(coefs_buffer, coefs_offset, Some(footprint.coefs.bytes));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prores raw decode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.decode_streams);
            pass.set_bind_group(0, &bind_group, &[]);
            let (x, y) = groups(
                streams,
                device.limits().max_compute_workgroups_per_dimension,
            );
            pass.dispatch_workgroups(x, y, 1);
            pass.set_pipeline(&self.transform);
            pass.dispatch_workgroups(layout.squares_x, layout.squares_y, 1);
        }
        Ok(DecodeInfo {
            header: parsed.header,
            skipped_tiles,
            max_qscale: parsed.max_qscale,
        })
    }
}

/// Sizes of the per-frame GPU data.
#[derive(Debug, Clone, Copy)]
struct Layout {
    squares_x: u32,
    squares_y: u32,
}

impl Layout {
    fn new(header: &FrameHeader) -> Self {
        Layout {
            squares_x: (header.coded_width() / TILE_UNIT) as u32,
            squares_y: (header.coded_height() / TILE_UNIT) as u32,
        }
    }

    fn squares(&self) -> u64 {
        u64::from(self.squares_x) * u64::from(self.squares_y)
    }

    fn coef_bytes(&self) -> u64 {
        self.squares() * SQUARE_WORDS * 4
    }

    fn sample_bytes(&self) -> u64 {
        self.squares() * (TILE_UNIT * TILE_UNIT * 2) as u64
    }
}

/// A frame decoded on the GPU.
#[derive(Debug)]
pub struct GpuFrame {
    /// The frame header.
    pub header: FrameHeader,
    /// Code values of the coded picture, two per word (see the module docs).
    pub samples: wgpu::Buffer,
    /// Indices of tiles skipped for an invalid header; their sensels are 0.
    pub skipped_tiles: Vec<usize>,
    /// Largest qscale byte over all tiles, as in [`Frame::max_qscale`].
    pub max_qscale: u8,
}

impl GpuFrame {
    /// The frame quantizer, as [`Frame::quantizer`] gives it.
    pub fn quantizer(&self) -> u32 {
        crate::decoder::quantizer(&self.header, self.max_qscale)
    }

    /// Copy the code values to the host. Waits for the GPU.
    pub fn download(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Frame, GpuError> {
        let bytes = read_buffer(device, queue, &self.samples, sample_bytes(&self.header))?;
        Ok(Frame {
            header: self.header.clone(),
            samples: bytes
                .chunks_exact(2)
                .map(|h| u16::from_le_bytes([h[0], h[1]]))
                .collect(),
            skipped_tiles: self.skipped_tiles.clone(),
            max_qscale: self.max_qscale,
        })
    }
}

/// Bytes of the output buffer for a frame with this header.
pub fn sample_bytes(header: &FrameHeader) -> u64 {
    Layout::new(header).sample_bytes()
}

/// The parameter block: geometry, the phase of each stream and the
/// quantization matrix.
fn params(header: &FrameHeader, layout: &Layout, streams: u32) -> Vec<u32> {
    let mut phase_stream = [0u32; 4];
    for (stream, (row, col)) in header.bayer_pattern.stream_phases().into_iter().enumerate() {
        phase_stream[row * 2 + col] = stream as u32;
    }
    // The fourth word must be 0: the shader XORs it into products (`rounded`).
    let mut out = vec![layout.squares_x, layout.squares_y, streams, 0];
    out.extend(phase_stream);
    out.extend(header.quant_matrix.iter().map(|&q| u32::from(q)));
    out
}

/// One entry per component stream of every valid tile: byte offset, byte
/// length, first square and packed block count, qscale and stream index;
/// and a bitmap of the squares of skipped tiles, whose indices go to
/// `skipped_tiles`.
fn stream_table(
    packet: &[u8],
    parsed: &ParsedFrame<'_>,
    layout: &Layout,
    skipped_tiles: &mut Vec<usize>,
) -> (Vec<u32>, Vec<u32>) {
    let base = packet.as_ptr() as usize;
    let mut table = Vec::with_capacity(parsed.tiles.len() * 4 * STREAM_WORDS);
    let mut skipped = vec![0u32; (layout.squares() as usize).div_ceil(32).max(1)];
    skipped_tiles.clear();
    for (index, (geometry, payload)) in parsed.grid.tiles().zip(&parsed.tiles).enumerate() {
        let square =
            (geometry.y / TILE_UNIT) as u32 * layout.squares_x + (geometry.x / TILE_UNIT) as u32;
        let Ok(tile) = Tile::parse(payload, index) else {
            skipped_tiles.push(index);
            for sq in square..square + geometry.blocks() as u32 {
                skipped[(sq / 32) as usize] |= 1 << (sq % 32);
            }
            continue;
        };
        for (stream, data) in tile.streams.iter().enumerate() {
            // Streams are subslices of the packet; their address gives the offset.
            let offset = data.as_ptr() as usize - base;
            debug_assert!(offset + data.len() <= packet.len());
            table.extend([
                offset as u32,
                data.len() as u32,
                square,
                u32::from(geometry.log2_blocks)
                    | u32::from(tile.qscale) << 8
                    | (stream as u32) << 16,
            ]);
        }
    }
    (table, skipped)
}

fn binding_entry(binding: u32, buffer: wgpu::BufferBinding<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(buffer),
    }
}

/// Workgroup counts covering `streams` invocations within the per-dimension limit.
fn groups(streams: u32, max_per_dimension: u32) -> (u32, u32) {
    let total = streams.div_ceil(STREAM_GROUP).max(1);
    let x = total.min(max_per_dimension);
    (x, total.div_ceil(x))
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Write `data` to the start of `buffer` followed by zero padding, without
/// copying the bulk of it: the aligned prefix goes directly, the tail and
/// padding through a small array.
fn upload_padded(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
    let aligned = data.len() & !3;
    if aligned > 0 {
        queue.write_buffer(buffer, offset, &data[..aligned]);
    }
    let mut tail = [0u8; 4 + PACKET_PADDING];
    let rest = &data[aligned..];
    tail[..rest.len()].copy_from_slice(rest);
    queue.write_buffer(buffer, offset + aligned as u64, &tail);
}

/// Copy `size` bytes of `src` to the host.
fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, GpuError> {
    let [bytes] = read_buffers(device, queue, &[src], size)?;
    Ok(bytes)
}

/// Copy the first `size` bytes of each of `srcs` to the host, with one
/// submission and one wait.
fn read_buffers<const N: usize>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    srcs: &[&wgpu::Buffer; N],
    size: u64,
) -> Result<[Vec<u8>; N], GpuError> {
    let mut out: [Vec<u8>; N] = std::array::from_fn(|_| Vec::new());
    read_buffers_with(device, queue, srcs, size, |i, bytes| {
        out[i] = bytes.to_vec()
    })?;
    Ok(out)
}

/// Like [`read_buffers`], handing each buffer's mapped bytes to `f` with
/// its index instead of copying them.
fn read_buffers_with<const N: usize>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    srcs: &[&wgpu::Buffer; N],
    size: u64,
    mut f: impl FnMut(usize, &[u8]),
) -> Result<(), GpuError> {
    let staging = srcs.map(|_| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prores raw readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("prores raw readback"),
    });
    for (src, dst) in srcs.iter().zip(&staging) {
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
    }
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    for buffer in &staging {
        let tx = tx.clone();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
    }
    drop(tx);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(GpuError::Poll)?;
    // wgpu runs the callbacks before a waiting poll returns; a dropped
    // sender means a mapping never completed.
    for _ in 0..N {
        rx.recv()
            .map_err(|_| GpuError::Map(wgpu::BufferAsyncError))?
            .map_err(GpuError::Map)?;
    }
    for (i, buffer) in staging.iter().enumerate() {
        f(i, &buffer.slice(..).get_mapped_range());
        buffer.unmap();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coefficients::decode_component;
    use crate::idct::dequantize;
    use crate::synth::{Rng, random_frame, specs};

    /// A device, or `None` with a message when the machine has no adapter.
    pub(crate) fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = match pollster::block_on(instance.request_adapter(&Default::default())) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("no GPU adapter, skipping: {e}");
                return None;
            }
        };
        Some(pollster::block_on(adapter.request_device(&Default::default())).expect("device"))
    }

    /// The scratch layout the stream pass must produce, from the host decoder.
    fn host_coefficients(packet: &[u8]) -> Vec<u32> {
        let parsed = parse_frame(packet).unwrap();
        let layout = Layout::new(&parsed.header);
        let mut words = vec![0u32; (layout.coef_bytes() / 4) as usize];
        let base = packet.as_ptr() as usize;
        let mut coefs = vec![0i16; 64 * 16];
        for (index, (geometry, payload)) in parsed.grid.tiles().zip(&parsed.tiles).enumerate() {
            let Ok(tile) = Tile::parse(payload, index) else {
                continue;
            };
            let blocks = geometry.blocks();
            let square =
                (geometry.y / TILE_UNIT) * layout.squares_x as usize + geometry.x / TILE_UNIT;
            for (stream, data) in tile.streams.iter().enumerate() {
                let c = &mut coefs[..64 * blocks];
                decode_component(
                    data,
                    (data.as_ptr() as usize - base) & 3,
                    geometry.log2_blocks,
                    c,
                );
                for (b, block) in c.chunks_exact(64).enumerate() {
                    for (raster, &v) in block.iter().enumerate() {
                        let q =
                            i16::from(parsed.header.quant_matrix[raster]) * i16::from(tile.qscale);
                        let d = dequantize(v, q) as u16 as u32;
                        let word = (square + b) * 128 + stream * 32 + raster / 2;
                        words[word] |= d << (16 * (raster % 2));
                    }
                }
            }
        }
        words
    }

    /// Buffers for every request of `footprint`, the coefficients
    /// readable.
    fn allocate(device: &wgpu::Device, footprint: &DecodeFootprint) -> Vec<wgpu::Buffer> {
        footprint
            .requests()
            .iter()
            .map(|r| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(r.label),
                    size: r.bytes,
                    usage: r.usage | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect()
    }

    fn buffers(b: &[wgpu::Buffer]) -> DecodeBuffers<'_> {
        DecodeBuffers {
            params: b[0].as_entire_buffer_binding(),
            packet: b[1].as_entire_buffer_binding(),
            streams: b[2].as_entire_buffer_binding(),
            coefs: b[3].as_entire_buffer_binding(),
            skipped: b[4].as_entire_buffer_binding(),
            samples: b[5].as_entire_buffer_binding(),
        }
    }

    /// Decode `packet` into buffers of its footprint and read back the
    /// coefficients, which the transform pass leaves in place.
    fn gpu_coefficients(
        pipelines: &GpuPipelines,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packet: &[u8],
    ) -> Vec<u32> {
        let header = crate::parse_header(packet).unwrap();
        let footprint = decode_footprint(&header, packet.len());
        let b = allocate(device, &footprint);
        let mut encoder = device.create_command_encoder(&Default::default());
        pipelines
            .record_decode(device, queue, &mut encoder, packet, &buffers(&b))
            .unwrap();
        queue.submit([encoder.finish()]);
        read_buffer(device, queue, &b[3], footprint.coefs.bytes)
            .unwrap()
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    }

    #[test]
    fn stream_pass_matches_the_host() {
        let Some((device, queue)) = device() else {
            return;
        };
        let pipelines = GpuPipelines::new(&device);
        let mut rng = Rng::new(7);
        for spec in specs(&mut rng) {
            let packet = random_frame(&spec, &mut rng);
            let gpu = gpu_coefficients(&pipelines, &device, &queue, &packet);
            let host = host_coefficients(&packet);
            let first_diff = gpu.iter().zip(&host).position(|(a, b)| a != b);
            assert_eq!(
                first_diff, None,
                "alignment {} pattern {:?} {}x{}",
                spec.alignment, spec.pattern, spec.width, spec.height
            );
        }
    }

    #[test]
    fn decode_matches_the_host() {
        let Some((device, queue)) = device() else {
            return;
        };
        let mut decoder = GpuDecoder::new(&device);
        let mut rng = Rng::new(11);
        let (mut total, mut differ, mut worst) = (0usize, 0usize, 0i32);
        for spec in specs(&mut rng) {
            let packet = random_frame(&spec, &mut rng);
            let gpu = decoder
                .decode(&device, &queue, &packet)
                .unwrap()
                .download(&device, &queue)
                .unwrap();
            let host = crate::decode_frame(&packet).unwrap();
            assert_eq!(gpu.header, host.header);
            assert_eq!(gpu.skipped_tiles, host.skipped_tiles);
            total += host.samples.len();
            for (&g, &h) in gpu.samples.iter().zip(&host.samples) {
                if g != h {
                    differ += 1;
                    worst = worst.max((i32::from(g) - i32::from(h)).abs());
                }
            }
        }
        eprintln!("GPU vs host: {differ} of {total} samples differ, max |diff| {worst}");
        assert_eq!(differ, 0);
    }

    /// Every request of a footprint as a region of one shared buffer, the
    /// way an arena hands out blocks: decodes recorded into one encoder,
    /// each with its own regions, match the host.
    #[test]
    fn record_decode_into_regions_of_one_buffer() {
        let Some((device, queue)) = device() else {
            return;
        };
        let pipelines = GpuPipelines::new(&device);
        let align = u64::from(device.limits().min_storage_buffer_offset_alignment);
        let mut rng = Rng::new(31);
        let packets: Vec<Vec<u8>> = specs(&mut rng)
            .into_iter()
            .take(3)
            .map(|spec| random_frame(&spec, &mut rng))
            .collect();
        // Lay out every request of every frame, aligned.
        let mut offsets = Vec::new();
        let mut end = 0u64;
        for packet in &packets {
            let footprint = decode_footprint(&crate::parse_header(packet).unwrap(), packet.len());
            let frame: Vec<u64> = footprint
                .requests()
                .iter()
                .map(|r| {
                    let at = end.next_multiple_of(align);
                    end = at + r.bytes;
                    at
                })
                .collect();
            offsets.push((footprint, frame));
        }
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("arena"),
            size: end.next_multiple_of(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let region = |at: u64, bytes: u64| wgpu::BufferBinding {
            buffer: &arena,
            offset: at,
            size: wgpu::BufferSize::new(bytes),
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        for (packet, (footprint, at)) in packets.iter().zip(&offsets) {
            let r = footprint.requests();
            let buffers = DecodeBuffers {
                params: region(at[0], r[0].bytes),
                packet: region(at[1], r[1].bytes),
                streams: region(at[2], r[2].bytes),
                coefs: region(at[3], r[3].bytes),
                skipped: region(at[4], r[4].bytes),
                samples: region(at[5], r[5].bytes),
            };
            pipelines
                .record_decode(&device, &queue, &mut encoder, packet, &buffers)
                .unwrap();
        }
        queue.submit([encoder.finish()]);
        let all = read_buffer(&device, &queue, &arena, arena.size()).unwrap();
        for (packet, (footprint, at)) in packets.iter().zip(&offsets) {
            let host = crate::decode_frame(packet).unwrap();
            let bytes = &all[at[5] as usize..(at[5] + footprint.samples.bytes) as usize];
            let gpu: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|h| u16::from_le_bytes([h[0], h[1]]))
                .collect();
            assert_eq!(gpu, host.samples);
        }

        // A region off the device's alignment is refused.
        let (footprint, at) = &offsets[0];
        let r = footprint.requests();
        let shifted = DecodeBuffers {
            params: region(at[0], r[0].bytes),
            packet: region(at[1], r[1].bytes),
            streams: region(at[2], r[2].bytes),
            coefs: region(at[3] + 4, r[3].bytes),
            skipped: region(at[4], r[4].bytes),
            samples: region(at[5], r[5].bytes),
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(matches!(
            pipelines.record_decode(&device, &queue, &mut encoder, &packets[0], &shifted),
            Err(GpuError::Misaligned { .. })
        ));
    }

    #[test]
    fn decode_into_checks_the_output_buffer() {
        let Some((device, queue)) = device() else {
            return;
        };
        let mut decoder = GpuDecoder::new(&device);
        let mut rng = Rng::new(3);
        let spec = &specs(&mut rng)[0];
        let packet = random_frame(spec, &mut rng);
        let needed = sample_bytes(&parse_frame(&packet).unwrap().header);
        let buffer = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let small = buffer(needed - 4, wgpu::BufferUsages::STORAGE);
        assert!(matches!(
            decoder.decode_into(&device, &queue, &packet, &small),
            Err(GpuError::BufferTooSmall { .. })
        ));
        let copy_only = buffer(needed, wgpu::BufferUsages::COPY_DST);
        assert!(matches!(
            decoder.decode_into(&device, &queue, &packet, &copy_only),
            Err(GpuError::MissingUsage { .. })
        ));
        let fine = buffer(needed, wgpu::BufferUsages::STORAGE);
        assert!(decoder.decode_into(&device, &queue, &packet, &fine).is_ok());
        assert!(matches!(
            decoder.decode_into(&device, &queue, &packet[..20], &fine),
            Err(GpuError::Format(_))
        ));
    }

    #[test]
    fn release_frees_the_scratch() {
        let Some((device, queue)) = device() else {
            return;
        };
        let mut decoder = GpuDecoder::new(&device);
        let mut rng = Rng::new(32);
        let packet = random_frame(&specs(&mut rng)[0], &mut rng);
        let first = decoder.decode(&device, &queue, &packet).unwrap();
        assert!(decoder.coefs.buffer.is_some());
        decoder.release();
        assert!(decoder.coefs.buffer.is_none() && decoder.packet.buffer.is_none());
        let again = decoder.decode(&device, &queue, &packet).unwrap();
        assert_eq!(
            again.download(&device, &queue).unwrap(),
            first.download(&device, &queue).unwrap()
        );
    }

    #[test]
    fn gpu_types_can_be_shared_between_threads() {
        fn check<T: Send + Sync>() {}
        check::<GpuPipelines>();
        check::<GpuDecoder>();
        check::<GpuDeveloper>();
    }

    #[test]
    fn group_counts_respect_the_dimension_limit() {
        assert_eq!(groups(0, 65535), (1, 1));
        assert_eq!(groups(64, 65535), (1, 1));
        assert_eq!(groups(65, 65535), (2, 1));
        assert_eq!(groups(64 * 70000, 65535), (65535, 2));
    }
}
