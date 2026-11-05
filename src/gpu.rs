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

use std::fmt;

mod develop;

pub use develop::{GpuDeveloper, GpuRgb, plane_bytes};

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
    /// The output buffer is smaller than the frame's samples.
    OutputTooSmall {
        /// Size of the buffer.
        size: u64,
        /// Bytes the frame needs, see [`sample_bytes`].
        needed: u64,
    },
    /// The output buffer lacks `STORAGE` usage.
    OutputNotStorage,
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
            GpuError::OutputTooSmall { size, needed } => {
                write!(f, "output buffer of {size} bytes, the frame needs {needed}")
            }
            GpuError::OutputNotStorage => write!(f, "output buffer lacks STORAGE usage"),
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
    layout: wgpu::BindGroupLayout,
    decode_streams: wgpu::ComputePipeline,
    transform: wgpu::ComputePipeline,
    params: Slot,
    packet: Slot,
    streams: Slot,
    coefs: Slot,
    skipped: Slot,
    skipped_tiles: Vec<usize>,
    max_qscale: u8,
}

impl GpuDecoder {
    /// Compile the decoding pipelines for `device`.
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prores raw"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, false),
                storage(4, false),
                storage(5, true),
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
        let copy_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        GpuDecoder {
            decode_streams: pipeline("decode_streams"),
            transform: pipeline("transform"),
            layout,
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
        let parsed = parse_frame(packet)?;
        let samples = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prores raw samples"),
            size: sample_bytes(&parsed.header),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let header = self.submit(device, queue, packet, parsed, &samples)?;
        Ok(GpuFrame {
            header,
            samples,
            skipped_tiles: self.skipped_tiles.clone(),
            max_qscale: self.max_qscale,
        })
    }

    /// Largest tile qscale byte of the most recently submitted frame (see
    /// [`Frame::max_qscale`]).
    pub fn max_qscale(&self) -> u8 {
        self.max_qscale
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
        let parsed = parse_frame(packet)?;
        let needed = sample_bytes(&parsed.header);
        if samples.size() < needed {
            return Err(GpuError::OutputTooSmall {
                size: samples.size(),
                needed,
            });
        }
        if !samples.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(GpuError::OutputNotStorage);
        }
        self.submit(device, queue, packet, parsed, samples)
    }

    fn submit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packet: &[u8],
        parsed: ParsedFrame<'_>,
        samples: &wgpu::Buffer,
    ) -> Result<FrameHeader, GpuError> {
        let (mut encoder, bind_group, layout) =
            self.record_streams(device, queue, packet, &parsed, samples)?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prores raw transform"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.transform);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(layout.squares_x, layout.squares_y, 1);
        }
        queue.submit([encoder.finish()]);
        self.max_qscale = parsed.max_qscale;
        Ok(parsed.header)
    }

    /// Upload a parsed frame and record the stream pass. Returns the
    /// command encoder for further passes and the bind group of the frame.
    fn record_streams(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packet: &[u8],
        parsed: &ParsedFrame<'_>,
        samples: &wgpu::Buffer,
    ) -> Result<(wgpu::CommandEncoder, wgpu::BindGroup, Layout), GpuError> {
        let limits = device.limits();
        let layout = Layout::new(&parsed.header);
        let (table, skipped_squares) =
            stream_table(packet, parsed, &layout, &mut self.skipped_tiles);
        let check = |what, bytes: u64| {
            let limit =
                u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
            if bytes > limit {
                Err(GpuError::TooLarge { what, bytes, limit })
            } else {
                Ok(())
            }
        };
        let packet_bytes = (packet.len() + PACKET_PADDING) as u64;
        // Stream offsets are addressed in bits with 32-bit arithmetic.
        let bit_limit = u64::from(u32::MAX / 8);
        if packet_bytes > bit_limit {
            return Err(GpuError::TooLarge {
                what: "packet",
                bytes: packet_bytes,
                limit: bit_limit,
            });
        }
        check("packet", packet_bytes)?;
        check("stream table", (table.len() * 4) as u64)?;
        check("coefficients", layout.coef_bytes())?;
        check("samples", layout.sample_bytes())?;

        let params = params(&parsed.header, &layout, (table.len() / STREAM_WORDS) as u32);
        let params_buf = self.params.get(device, (params.len() * 4) as u64);
        queue.write_buffer(params_buf, 0, &words_to_bytes(&params));
        let streams_buf = self
            .streams
            .get(device, (table.len().max(STREAM_WORDS) * 4) as u64);
        queue.write_buffer(streams_buf, 0, &words_to_bytes(&table));
        let packet_buf = self.packet.get(device, packet_bytes);
        upload_padded(queue, packet_buf, packet);
        let coefs_buf = self.coefs.get(device, layout.coef_bytes());
        let skipped_buf = self.skipped.get(device, (skipped_squares.len() * 4) as u64);
        queue.write_buffer(skipped_buf, 0, &words_to_bytes(&skipped_squares));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prores raw frame"),
            layout: &self.layout,
            entries: &[
                entry(0, params_buf),
                entry(1, packet_buf),
                entry(2, streams_buf),
                entry(3, coefs_buf),
                entry(4, samples),
                entry(5, skipped_buf),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prores raw decode"),
        });
        // The stream pass ORs coefficients into zeroed words; one fill is
        // much cheaper than each invocation zeroing its scattered blocks.
        encoder.clear_buffer(coefs_buf, 0, Some(layout.coef_bytes()));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prores raw streams"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.decode_streams);
            pass.set_bind_group(0, &bind_group, &[]);
            let (x, y) = groups(
                (table.len() / STREAM_WORDS) as u32,
                limits.max_compute_workgroups_per_dimension,
            );
            pass.dispatch_workgroups(x, y, 1);
        }
        Ok((encoder, bind_group, layout))
    }

    /// Decode the coefficients of `packet` and read them back, in the
    /// scratch layout. For testing the stream pass on its own.
    #[cfg(test)]
    fn coefficients(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packet: &[u8],
    ) -> Result<Vec<u32>, GpuError> {
        let parsed = parse_frame(packet)?;
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let (encoder, _, layout) = self.record_streams(device, queue, packet, &parsed, &dummy)?;
        queue.submit([encoder.finish()]);
        let coefs = self
            .coefs
            .buffer
            .as_ref()
            .expect("allocated by record_streams");
        let bytes = read_buffer(device, queue, coefs, layout.coef_bytes())?;
        Ok(bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect())
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

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
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
fn upload_padded(queue: &wgpu::Queue, buffer: &wgpu::Buffer, data: &[u8]) {
    let aligned = data.len() & !3;
    if aligned > 0 {
        queue.write_buffer(buffer, 0, &data[..aligned]);
    }
    let mut tail = [0u8; 4 + PACKET_PADDING];
    let rest = &data[aligned..];
    tail[..rest.len()].copy_from_slice(rest);
    queue.write_buffer(buffer, aligned as u64, &tail);
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

    #[test]
    fn stream_pass_matches_the_host() {
        let Some((device, queue)) = device() else {
            return;
        };
        let mut decoder = GpuDecoder::new(&device);
        let mut rng = Rng::new(7);
        for spec in specs(&mut rng) {
            let packet = random_frame(&spec, &mut rng);
            let gpu = decoder.coefficients(&device, &queue, &packet).unwrap();
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
            Err(GpuError::OutputTooSmall { .. })
        ));
        let copy_only = buffer(needed, wgpu::BufferUsages::COPY_DST);
        assert!(matches!(
            decoder.decode_into(&device, &queue, &packet, &copy_only),
            Err(GpuError::OutputNotStorage)
        ));
        let fine = buffer(needed, wgpu::BufferUsages::STORAGE);
        assert!(decoder.decode_into(&device, &queue, &packet, &fine).is_ok());
        assert!(matches!(
            decoder.decode_into(&device, &queue, &packet[..20], &fine),
            Err(GpuError::Format(_))
        ));
    }

    #[test]
    fn group_counts_respect_the_dimension_limit() {
        assert_eq!(groups(0, 65535), (1, 1));
        assert_eq!(groups(64, 65535), (1, 1));
        assert_eq!(groups(65, 65535), (2, 1));
        assert_eq!(groups(64 * 70000, 65535), (65535, 2));
    }
}
