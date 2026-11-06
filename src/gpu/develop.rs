//! The raw conversion on the GPU: [`crate::develop_planar`], pass for pass.

use std::sync::Arc;

use super::{
    BufferRequest, GpuError, GpuFrame, GpuPipelines, Slot, bind, binding_entry, check_limits,
    read_buffers_with, sample_bytes,
};
use crate::decoder::Frame;
use crate::demosaic::{Conversion, DemosaicParams, RgbPlanes, Stage};
use crate::header::FrameHeader;

/// Invocations per side of a workgroup of both passes.
const GROUP_SIDE: u32 = 16;
/// Bytes of the parameter block of one band.
const PARAMS_BYTES: usize = 128;

/// Rows of code values a band of the conversion reads beyond each of its
/// edges (inside the picture): the green pass needs two rows around each
/// of its rows, and it covers one row beyond the band.
pub const DEVELOP_INPUT_HALO: usize = 3;

/// Green-pass rows a band computes beyond each of its edges (inside the
/// picture), which the chroma pass of the band reads.
pub const DEVELOP_GREEN_HALO: usize = 1;

/// Scratch bytes per green-pass row of a frame with this header: two `f32`
/// planes. A band of `rows` rows needs `rows + 2 * DEVELOP_GREEN_HALO`
/// green rows (fewer at the picture's edges).
pub fn develop_row_bytes(header: &FrameHeader) -> u64 {
    8 * u64::from(header.width)
}

/// Green-pass rows of the band starting at `y0` with `rows` rows, in a
/// picture `height` rows tall: (first, count).
fn green_rows(y0: usize, rows: usize, height: usize) -> (usize, usize) {
    let g0 = y0.saturating_sub(DEVELOP_GREEN_HALO);
    let g1 = (y0 + rows + DEVELOP_GREEN_HALO).min(height);
    (g0, g1 - g0)
}

/// The band height the conversion uses unless told otherwise: the whole
/// frame when its green planes fit the device's storage binding and buffer
/// limits, otherwise the tallest band whose green planes fit, rounded down
/// to a multiple of 16 rows.
pub fn default_band_rows(header: &FrameHeader, limits: &wgpu::Limits) -> usize {
    let height = usize::from(header.height);
    let limit = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
    let row = (develop_row_bytes(header) / 2).max(1);
    let fit = (limit / row) as usize;
    if fit >= height {
        return height;
    }
    let rows = fit.saturating_sub(2 * DEVELOP_GREEN_HALO);
    if rows >= 16 {
        rows / 16 * 16
    } else {
        rows.max(1)
    }
}

/// Runs the raw conversion on one wgpu device.
///
/// The input is the code values of a frame in the layout [`super::GpuDecoder`]
/// writes, so a frame can be decoded and converted without leaving the GPU;
/// [`GpuDeveloper::develop_frame`] uploads a host [`Frame`] instead. The
/// output is three `f32` planes (R, G, B), each `width * height` values,
/// row-major and tightly packed, with the values of
/// [`crate::develop_planar`]. The arithmetic is the host's, operation for
/// operation; a device may still round some divisions and square roots
/// differently from the host.
///
/// Holds scratch buffers reused across frames; use one per thread, all
/// sharing one [`GpuPipelines`].
pub struct GpuDeveloper {
    pipelines: Arc<GpuPipelines>,
    band_rows: Option<usize>,
    params: Slot,
    lut: Slot,
    samples: Slot,
    rb: Slot,
    gr: Slot,
}

/// A frame converted on the GPU: three planes of `f32` values.
#[derive(Debug)]
pub struct GpuRgb {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// The R, G and B planes, `width * height` values each, row-major,
    /// with `STORAGE | COPY_SRC` usage.
    pub planes: [wgpu::Buffer; 3],
    /// Indices of tiles skipped for an invalid header, from the decode.
    pub skipped_tiles: Vec<usize>,
}

impl GpuRgb {
    /// Copy the planes to the host. Waits for the GPU.
    pub fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<RgbPlanes, GpuError> {
        let n = self.width * self.height;
        let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; n]);
        let [r, g, b] = &mut planes;
        self.download_into(device, queue, [r, g, b])?;
        Ok(RgbPlanes {
            width: self.width,
            height: self.height,
            planes,
        })
    }

    /// Copy the planes into caller-owned slices of `width * height`
    /// values each, for reusing them from frame to frame. Waits for the
    /// GPU.
    ///
    /// # Panics
    ///
    /// If a slice does not hold exactly `width * height` values.
    pub fn download_into(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut out: [&mut [f32]; 3],
    ) -> Result<(), GpuError> {
        let n = self.width * self.height;
        for plane in &out {
            assert_eq!(plane.len(), n, "plane size");
        }
        let [r, g, b] = &self.planes;
        read_buffers_with(device, queue, &[r, g, b], (n * 4) as u64, |i, bytes| {
            copy_floats(bytes, out[i]);
        })
    }
}

/// Little-endian `f32` values from `bytes` into `out`, in parallel with the
/// `parallel` feature: reading mapped memory is slow on one thread.
fn copy_floats(bytes: &[u8], out: &mut [f32]) {
    const CHUNK: usize = 1 << 16;
    let copy = |(dst, src): (&mut [f32], &[u8])| {
        for (d, s) in dst.iter_mut().zip(src.chunks_exact(4)) {
            *d = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(CHUNK)
            .zip(bytes.par_chunks(4 * CHUNK))
            .for_each(copy);
    }
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(CHUNK)
        .zip(bytes.chunks(4 * CHUNK))
        .for_each(copy);
}

/// Bytes of one output plane for a frame with this header.
pub fn plane_bytes(header: &FrameHeader) -> u64 {
    (usize::from(header.width) * usize::from(header.height) * 4) as u64
}

/// The conversion's part of [`GpuPipelines`].
pub(super) struct Pipelines {
    layout: wgpu::BindGroupLayout,
    green: wgpu::ComputePipeline,
    chroma: wgpu::ComputePipeline,
}

impl Pipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                // The parameters: one block per band.
                has_dynamic_offset: binding == 0,
                min_binding_size: None,
            },
            count: None,
        };
        let storage = |read_only| wgpu::BufferBindingType::Storage { read_only };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prores raw develop"),
            // The parameters are a uniform: read on every invocation, they
            // belong in the constant cache, and as read-write storage the
            // conversion runs 40 % slower. Everything else is read-write
            // storage so that a caller may place it all in one buffer (see
            // the decode layout in `GpuPipelines::new`).
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform),
                buffer(1, storage(false)),
                buffer(2, storage(false)),
                buffer(3, storage(false)),
                buffer(4, storage(false)),
                buffer(5, storage(false)),
                buffer(6, storage(false)),
                buffer(7, storage(false)),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prores raw develop"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prores raw develop"),
            source: wgpu::ShaderSource::Wgsl(include_str!("develop.wgsl").into()),
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
        Pipelines {
            green: pipeline("green"),
            chroma: pipeline("chroma"),
            layout,
        }
    }
}

impl GpuDeveloper {
    /// A developer with its own pipelines, compiled for `device`. Prefer
    /// [`GpuDeveloper::with_pipelines`] to share them with decoders.
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_pipelines(Arc::new(GpuPipelines::new(device)))
    }

    /// A developer using shared pipelines. Scratch buffers are allocated
    /// when the first frame is converted.
    pub fn with_pipelines(pipelines: Arc<GpuPipelines>) -> Self {
        let copy_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        GpuDeveloper {
            pipelines,
            band_rows: None,
            params: Slot::new(
                "prores raw develop params",
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            lut: Slot::new("prores raw develop table", copy_storage),
            samples: Slot::new("prores raw develop samples", copy_storage),
            rb: Slot::new("prores raw red and blue", wgpu::BufferUsages::STORAGE),
            gr: Slot::new("prores raw green", wgpu::BufferUsages::STORAGE),
        }
    }

    /// The pipelines this developer runs.
    pub fn pipelines(&self) -> &Arc<GpuPipelines> {
        &self.pipelines
    }

    /// Convert in bands of `rows` rows, or with `None` (the default) in
    /// bands of [`default_band_rows`] for the device. Smaller bands need
    /// less scratch (see [`GpuPipelines::record_develop`]).
    pub fn set_band_rows(&mut self, rows: Option<usize>) {
        self.band_rows = rows;
    }

    /// Convert a frame decoded on the GPU into three new planes. The work
    /// is submitted to `queue` after the decode.
    pub fn develop(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &GpuFrame,
        params: &DemosaicParams,
        stage: Stage,
    ) -> Result<GpuRgb, GpuError> {
        let planes = new_planes(device, &frame.header);
        let [r, g, b] = &planes;
        self.develop_into(
            device,
            queue,
            &frame.header,
            params,
            stage,
            &frame.samples,
            [r, g, b],
        )?;
        Ok(GpuRgb {
            width: frame.header.width.into(),
            height: frame.header.height.into(),
            planes,
            skipped_tiles: frame.skipped_tiles.clone(),
        })
    }

    /// Upload a host [`Frame`]'s code values and convert them into three
    /// new planes.
    pub fn develop_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &Frame,
        params: &DemosaicParams,
        stage: Stage,
    ) -> Result<GpuRgb, GpuError> {
        let bytes: Vec<u8> = frame.samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        let samples = self.samples.get(device, bytes.len() as u64);
        queue.write_buffer(samples, 0, &bytes);
        let samples = samples.clone();
        let planes = new_planes(device, &frame.header);
        let [r, g, b] = &planes;
        self.develop_into(
            device,
            queue,
            &frame.header,
            params,
            stage,
            &samples,
            [r, g, b],
        )?;
        Ok(GpuRgb {
            width: frame.width(),
            height: frame.height(),
            planes,
            skipped_tiles: frame.skipped_tiles.clone(),
        })
    }

    /// Convert the code values in `samples` (the layout
    /// [`super::GpuDecoder`] writes, at least [`sample_bytes`] long) of a
    /// frame with `header` into the planes `out`, buffers with `STORAGE`
    /// usage of at least [`plane_bytes`] each. The work is submitted to
    /// `queue`; later submissions on it see the result.
    #[allow(clippy::too_many_arguments)]
    pub fn develop_into(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        header: &FrameHeader,
        params: &DemosaicParams,
        stage: Stage,
        samples: &wgpu::Buffer,
        out: [&wgpu::Buffer; 3],
    ) -> Result<(), GpuError> {
        let limits = device.limits();
        let band_rows = self
            .band_rows
            .unwrap_or_else(|| default_band_rows(header, &limits));
        let f = develop_footprint(header, band_rows, &limits);
        let buffers = DevelopBuffers {
            params: self
                .params
                .get(device, f.params.bytes)
                .as_entire_buffer_binding(),
            lut: self.lut.get(device, f.lut.bytes).as_entire_buffer_binding(),
            samples: samples.as_entire_buffer_binding(),
            rb: self.rb.get(device, f.rb.bytes).as_entire_buffer_binding(),
            gr: self.gr.get(device, f.gr.bytes).as_entire_buffer_binding(),
            red: out[0].as_entire_buffer_binding(),
            green: out[1].as_entire_buffer_binding(),
            blue: out[2].as_entire_buffer_binding(),
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prores raw develop"),
        });
        self.pipelines.record_develop(
            device,
            queue,
            &mut encoder,
            header,
            params,
            stage,
            band_rows,
            &buffers,
        )?;
        queue.submit([encoder.finish()]);
        Ok(())
    }
}

/// The buffers [`GpuPipelines::record_develop`] needs for one frame, in
/// the order of [`DevelopBuffers`] (whose `samples`, the decoder's output,
/// is not listed here: see [`super::DecodeFootprint::samples`]).
///
/// Storage bindings must start at multiples of the device's
/// `min_storage_buffer_offset_alignment`; the conversion binds exactly
/// `bytes` of each. The storage requests may all be regions of one buffer;
/// the parameters need a buffer of their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevelopFootprint {
    /// The conversion's constants, a uniform block. Uploaded. Its binding
    /// starts at a multiple of `min_uniform_buffer_offset_alignment`, and
    /// as wgpu tracks usage per buffer it must not share a buffer with the
    /// other (storage) bindings.
    pub params: BufferRequest,
    /// Code value to normalized linear value, 65536 `f32`. Uploaded.
    pub lut: BufferRequest,
    /// Green pass output, red or blue per sensel: written by the green
    /// pass, read by the chroma pass, dead after it.
    pub rb: BufferRequest,
    /// Green pass output, green per sensel; like `rb`.
    pub gr: BufferRequest,
    /// The output R plane, `width * height` `f32`.
    pub red: BufferRequest,
    /// The output G plane.
    pub green: BufferRequest,
    /// The output B plane.
    pub blue: BufferRequest,
}

impl DevelopFootprint {
    /// The requests in [`DevelopBuffers`] order, without `samples`.
    pub fn requests(&self) -> [BufferRequest; 7] {
        [
            self.params,
            self.lut,
            self.rb,
            self.gr,
            self.red,
            self.green,
            self.blue,
        ]
    }

    /// Total bytes of all requests.
    pub fn bytes(&self) -> u64 {
        self.requests().iter().map(|r| r.bytes).sum()
    }
}

/// What converting a frame with `header` in bands of `band_rows` rows
/// needs from the device besides its code values. `limits` gives the
/// uniform offset alignment, which spaces the bands' parameter blocks.
pub fn develop_footprint(
    header: &FrameHeader,
    band_rows: usize,
    limits: &wgpu::Limits,
) -> DevelopFootprint {
    use wgpu::BufferUsages as U;
    let plane = plane_bytes(header).max(4);
    let bands = Bands::new(header, band_rows, limits);
    let scratch = (bands.green_rows as u64 * develop_row_bytes(header) / 2).max(4);
    let request = |label, bytes, usage| BufferRequest {
        label,
        bytes,
        usage,
    };
    let upload = U::STORAGE | U::COPY_DST;
    DevelopFootprint {
        params: request(
            "prores raw develop params",
            bands.count as u64 * bands.stride,
            U::UNIFORM | U::COPY_DST,
        ),
        lut: request("prores raw develop table", 65536 * 4, upload),
        rb: request("prores raw red and blue", scratch, U::STORAGE),
        gr: request("prores raw green", scratch, U::STORAGE),
        red: request("prores raw red", plane, U::STORAGE),
        green: request("prores raw green plane", plane, U::STORAGE),
        blue: request("prores raw blue", plane, U::STORAGE),
    }
}

/// How a frame is cut into bands.
struct Bands {
    /// Rows per band (the last may have fewer).
    rows: usize,
    count: usize,
    /// Largest green-pass row count of any band.
    green_rows: usize,
    /// Bytes between bands' parameter blocks.
    stride: u64,
}

impl Bands {
    fn new(header: &FrameHeader, band_rows: usize, limits: &wgpu::Limits) -> Self {
        let height = usize::from(header.height).max(1);
        let rows = band_rows.clamp(1, height);
        Bands {
            rows,
            count: height.div_ceil(rows),
            green_rows: (rows + 2 * DEVELOP_GREEN_HALO).min(height),
            stride: (PARAMS_BYTES as u64)
                .next_multiple_of(u64::from(limits.min_uniform_buffer_offset_alignment)),
        }
    }
}

/// Caller-provided buffers for [`GpuPipelines::record_develop`]: the
/// frame's code values and one binding per [`DevelopFootprint`] request.
#[derive(Debug, Clone)]
pub struct DevelopBuffers<'a> {
    /// See [`DevelopFootprint::params`].
    pub params: wgpu::BufferBinding<'a>,
    /// See [`DevelopFootprint::lut`].
    pub lut: wgpu::BufferBinding<'a>,
    /// The frame's code values, as [`super::GpuPipelines::record_decode`]
    /// writes them ([`sample_bytes`]). Only read.
    pub samples: wgpu::BufferBinding<'a>,
    /// See [`DevelopFootprint::rb`].
    pub rb: wgpu::BufferBinding<'a>,
    /// See [`DevelopFootprint::gr`].
    pub gr: wgpu::BufferBinding<'a>,
    /// See [`DevelopFootprint::red`].
    pub red: wgpu::BufferBinding<'a>,
    /// See [`DevelopFootprint::green`].
    pub green: wgpu::BufferBinding<'a>,
    /// See [`DevelopFootprint::blue`].
    pub blue: wgpu::BufferBinding<'a>,
}

impl GpuPipelines {
    /// Record the conversion of a frame's code values into `encoder`,
    /// reading and writing only `buffers` (see [`develop_footprint`]).
    /// Nothing is submitted; the planes hold the result once the
    /// submission completes. The values are [`crate::develop_planar`]'s,
    /// bit for bit.
    ///
    /// The parameters and the table are uploaded with `queue.write_buffer`,
    /// which wgpu performs before the commands of the next submission, so
    /// conversions recorded into one submission need separate upload
    /// buffers. The code values may come from a decode recorded earlier in
    /// the same encoder.
    ///
    /// The conversion runs in bands of `band_rows` rows, all recorded here,
    /// one after another through the same green-pass planes, so those hold
    /// one band's rows rather than the frame's; the footprint must be for
    /// the same band height. Any band height gives the same values. Use
    /// [`default_band_rows`] for the device's limits, or a planner's
    /// choice with [`DEVELOP_INPUT_HALO`], [`DEVELOP_GREEN_HALO`] and
    /// [`develop_row_bytes`].
    #[allow(clippy::too_many_arguments)]
    pub fn record_develop(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        header: &FrameHeader,
        params: &DemosaicParams,
        stage: Stage,
        band_rows: usize,
        buffers: &DevelopBuffers<'_>,
    ) -> Result<(), GpuError> {
        let limits = device.limits();
        let footprint = develop_footprint(header, band_rows, &limits);
        let bands = Bands::new(header, band_rows, &limits);
        let samples = BufferRequest {
            label: "prores raw samples",
            bytes: sample_bytes(header),
            usage: wgpu::BufferUsages::STORAGE,
        };
        check_limits(device, &footprint.requests()[1..])?;
        check_limits(device, &[samples])?;
        let b =
            |binding, request| bind(binding, request, limits.min_storage_buffer_offset_alignment);
        let mut params_b = bind(
            &buffers.params,
            &footprint.params,
            limits.min_uniform_buffer_offset_alignment,
        )?;
        let lut_b = b(&buffers.lut, &footprint.lut)?;
        let samples_b = b(&buffers.samples, &samples)?;
        let rb_b = b(&buffers.rb, &footprint.rb)?;
        let gr_b = b(&buffers.gr, &footprint.gr)?;
        let red_b = b(&buffers.red, &footprint.red)?;
        let green_b = b(&buffers.green, &footprint.green)?;
        let blue_b = b(&buffers.blue, &footprint.blue)?;

        let conversion = Conversion::new(header, params, stage);
        let height = usize::from(header.height);
        let mut blocks = vec![0u8; footprint.params.bytes as usize];
        let mut dispatches = Vec::with_capacity(bands.count);
        for band in 0..bands.count {
            let y0 = band * bands.rows;
            let rows = bands.rows.min(height - y0);
            let (g0, g_rows) = green_rows(y0, rows, height);
            let block = params_block(header, &conversion, [y0, rows, g0, g_rows]);
            let at = band * bands.stride as usize;
            blocks[at..at + PARAMS_BYTES].copy_from_slice(&block);
            dispatches.push((rows as u32, g_rows as u32));
        }
        queue.write_buffer(params_b.buffer, params_b.offset, &blocks);
        // Each band binds its own block through a dynamic offset.
        params_b.size = wgpu::BufferSize::new(PARAMS_BYTES as u64);
        let lut: Vec<u8> = conversion
            .lut
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        queue.write_buffer(lut_b.buffer, lut_b.offset, &lut);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prores raw develop frame"),
            layout: &self.develop.layout,
            entries: &[
                binding_entry(0, params_b),
                binding_entry(1, lut_b),
                binding_entry(2, samples_b),
                binding_entry(3, rb_b),
                binding_entry(4, gr_b),
                binding_entry(5, red_b),
                binding_entry(6, green_b),
                binding_entry(7, blue_b),
            ],
        });
        let columns = u32::from(header.width).div_ceil(GROUP_SIDE);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prores raw develop"),
            timestamp_writes: None,
        });
        // wgpu orders dispatches that share read-write storage, so each
        // band's green pass waits for the previous band's chroma pass.
        for (band, (rows, g_rows)) in dispatches.into_iter().enumerate() {
            let offset = (band as u64 * bands.stride) as u32;
            pass.set_bind_group(0, &bind_group, &[offset]);
            pass.set_pipeline(&self.develop.green);
            pass.dispatch_workgroups(columns, g_rows.div_ceil(GROUP_SIDE), 1);
            pass.set_pipeline(&self.develop.chroma);
            pass.dispatch_workgroups(columns, rows.div_ceil(GROUP_SIDE), 1);
        }
        Ok(())
    }
}

/// Three output planes for a frame with this header.
fn new_planes(device: &wgpu::Device, header: &FrameHeader) -> [wgpu::Buffer; 3] {
    ["prores raw red", "prores raw green", "prores raw blue"].map(|label| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: plane_bytes(header).max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    })
}

/// The shader's `Params` block for one band: `band` is the output's first
/// row and row count, then the green pass's.
fn params_block(header: &FrameHeader, c: &Conversion, band: [usize; 4]) -> Vec<u8> {
    let p = &c.params;
    let f = |v: f32| v.to_bits();
    // The green pass offsets by black * (1 / range), the chroma pass by
    // black / range; the two may round differently.
    let green_off = p.black as f32 * p.one_over_range;
    let m = c.matrix.unwrap_or_default();
    let mut words = vec![
        u32::from(header.width),
        u32::from(header.height),
        header.coded_width() as u32,
        p.cfa_phase as u32,
        f(green_off),
        f(p.black_over_range),
        f(p.clip[0]),
        f(p.clip[1]),
        f(p.noise[0]),
        f(p.noise[1]),
        f(p.noise[2]),
        0,
        f(p.wb[0]),
        f(p.wb[1]),
        u32::from(c.matrix.is_some()),
        0,
    ];
    for row in m.chunks_exact(3) {
        words.extend(row.iter().map(|&v| f(v)));
        words.push(0);
    }
    words.extend(band.map(|v| v as u32));
    debug_assert_eq!(words.len() * 4, PARAMS_BYTES);
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuDecoder;
    use crate::gpu::tests::device;
    use crate::synth::{Rng, Spec, random_frame, specs};

    /// Differences between GPU and host planes: (count, largest |diff|).
    fn compare(gpu: &RgbPlanes, host: &RgbPlanes) -> (usize, f32) {
        assert_eq!((gpu.width, gpu.height), (host.width, host.height));
        let mut differ = 0;
        let mut worst = 0f32;
        for (g, h) in gpu.planes.iter().zip(&host.planes) {
            for (&a, &b) in g.iter().zip(h) {
                if a.to_bits() != b.to_bits() {
                    differ += 1;
                    worst = worst.max((a - b).abs());
                }
            }
        }
        (differ, worst)
    }

    #[test]
    fn develop_matches_the_host() {
        let Some((device, queue)) = device() else {
            return;
        };
        // One set of pipelines serves both.
        let mut decoder = GpuDecoder::new(&device);
        let mut developer = GpuDeveloper::with_pipelines(decoder.pipelines().clone());
        let mut rng = Rng::new(21);
        let (mut total, mut differ, mut worst) = (0, 0, 0f32);
        for spec in specs(&mut rng) {
            let spec = Spec {
                valid_tiles: true,
                wb: (1.9, 1.4),
                ..spec
            };
            let packet = random_frame(&spec, &mut rng);
            let host_frame = crate::decode_frame(&packet).unwrap();
            let params = DemosaicParams::new(&host_frame.header, host_frame.quantizer());
            let frame = decoder.decode(&device, &queue, &packet).unwrap();
            for stage in [Stage::Demosaiced, Stage::WhiteBalanced, Stage::Rec2020] {
                let host = crate::develop_planar(&host_frame, &params, stage);
                let gpu = developer
                    .develop(&device, &queue, &frame, &params, stage)
                    .unwrap()
                    .download(&device, &queue)
                    .unwrap();
                let (d, w) = compare(&gpu, &host);
                total += 3 * host.width * host.height;
                differ += d;
                worst = worst.max(w);
            }
        }
        eprintln!("GPU vs host develop: {differ} of {total} values differ, max |diff| {worst:e}");
        assert_eq!(differ, 0);
    }

    /// Any band height gives the host's values: bands of one row (whose
    /// green rows reach past both edges), heights around the workgroup
    /// size, and the whole frame.
    #[test]
    fn every_band_height_matches_the_host() {
        let Some((device, queue)) = device() else {
            return;
        };
        let mut decoder = GpuDecoder::new(&device);
        let mut developer = GpuDeveloper::with_pipelines(decoder.pipelines().clone());
        let mut rng = Rng::new(25);
        for spec in specs(&mut rng).into_iter().take(4) {
            let spec = Spec {
                valid_tiles: true,
                ..spec
            };
            let packet = random_frame(&spec, &mut rng);
            let host_frame = crate::decode_frame(&packet).unwrap();
            let params = DemosaicParams::new(&host_frame.header, host_frame.quantizer());
            let host = crate::develop_planar(&host_frame, &params, Stage::Rec2020);
            let frame = decoder.decode(&device, &queue, &packet).unwrap();
            for rows in [1, 2, 3, 15, 16, 17, host_frame.height()] {
                developer.set_band_rows(Some(rows));
                let gpu = developer
                    .develop(&device, &queue, &frame, &params, Stage::Rec2020)
                    .unwrap()
                    .download(&device, &queue)
                    .unwrap();
                assert_eq!(compare(&gpu, &host).0, 0, "{rows}-row bands");
            }
        }
    }

    #[test]
    fn bands_size_the_scratch() {
        let mut rng = Rng::new(26);
        let header = crate::parse_header(&random_frame(
            &Spec::new(320, 96, crate::BayerPattern::Rggb),
            &mut rng,
        ))
        .unwrap();
        let limits = wgpu::Limits::default();
        let stride = u64::from(limits.min_uniform_buffer_offset_alignment);
        for (rows, green_rows, bands) in [(1, 3, 96), (16, 18, 6), (40, 42, 3), (96, 96, 1)] {
            let f = develop_footprint(&header, rows, &limits);
            assert_eq!(f.rb.bytes, green_rows * 320 * 4, "{rows}");
            assert_eq!(f.gr.bytes, f.rb.bytes);
            assert_eq!(f.params.bytes, bands * stride);
            assert_eq!(f.red.bytes, 320 * 96 * 4);
        }
        assert_eq!(develop_row_bytes(&header), 320 * 8);
    }

    #[test]
    fn default_band_fits_the_binding_limit() {
        let mut rng = Rng::new(27);
        let header = crate::parse_header(&random_frame(
            &Spec::new(1000, 512, crate::BayerPattern::Rggb),
            &mut rng,
        ))
        .unwrap();
        let limits = |bytes: u32| wgpu::Limits {
            max_storage_buffer_binding_size: bytes,
            ..wgpu::Limits::default()
        };
        // The whole frame's green planes fit.
        assert_eq!(default_band_rows(&header, &limits(512 * 4000)), 512);
        // 200 rows fit: 198 with the halo, rounded down to 192.
        assert_eq!(default_band_rows(&header, &limits(200 * 4000)), 192);
        // Fewer than 16 rows fit: no rounding.
        assert_eq!(default_band_rows(&header, &limits(12 * 4000)), 10);
        assert_eq!(default_band_rows(&header, &limits(100)), 1);
        let rows = default_band_rows(&header, &limits(200 * 4000));
        assert!(develop_footprint(&header, rows, &limits(0)).rb.bytes <= 200 * 4000);
    }

    #[test]
    fn develop_into_checks_its_buffers() {
        let Some((device, queue)) = device() else {
            return;
        };
        let mut decoder = GpuDecoder::new(&device);
        let mut developer = GpuDeveloper::new(&device);
        let mut rng = Rng::new(23);
        let packet = random_frame(&Spec::new(64, 32, crate::BayerPattern::Rggb), &mut rng);
        let frame = decoder.decode(&device, &queue, &packet).unwrap();
        let params = DemosaicParams::new(&frame.header, frame.quantizer());
        let plane = plane_bytes(&frame.header);
        let buffer = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let (r, g) = (buffer(plane, storage), buffer(plane, storage));
        let mut run = |samples: &wgpu::Buffer, b: &wgpu::Buffer| {
            developer.develop_into(
                &device,
                &queue,
                &frame.header,
                &params,
                Stage::default(),
                samples,
                [&r, &g, b],
            )
        };
        assert!(matches!(
            run(&frame.samples, &buffer(plane - 4, storage)),
            Err(GpuError::BufferTooSmall { .. })
        ));
        assert!(matches!(
            run(&frame.samples, &buffer(plane, wgpu::BufferUsages::COPY_DST)),
            Err(GpuError::MissingUsage { .. })
        ));
        assert!(matches!(
            run(&buffer(16, storage), &buffer(plane, storage)),
            Err(GpuError::BufferTooSmall { .. })
        ));
        assert!(run(&frame.samples, &buffer(plane, storage)).is_ok());
    }

    /// Decode and conversion recorded into one encoder, every buffer a
    /// region of one shared buffer, as a consumer's arena would hand them
    /// out.
    #[test]
    fn decode_and_develop_in_one_submission_from_one_buffer() {
        let Some((device, queue)) = device() else {
            return;
        };
        let pipelines = GpuPipelines::new(&device);
        let mut rng = Rng::new(24);
        let spec = Spec {
            valid_tiles: true,
            ..Spec::new(200, 72, crate::BayerPattern::Gbrg)
        };
        let packet = random_frame(&spec, &mut rng);
        let header = crate::parse_header(&packet).unwrap();
        let decode = crate::gpu::decode_footprint(&header, packet.len());
        let limits = device.limits();
        let develop = develop_footprint(&header, 16, &limits);
        let align = u64::from(device.limits().min_storage_buffer_offset_alignment);
        let mut end = 0u64;
        let mut place = |bytes: u64| {
            let at = end.next_multiple_of(align);
            end = at + bytes;
            (at, bytes)
        };
        let d: Vec<_> = decode.requests().iter().map(|r| place(r.bytes)).collect();
        // The conversion's parameters are a uniform, in a buffer of their own.
        let v: Vec<_> = develop.requests()[1..]
            .iter()
            .map(|r| place(r.bytes))
            .collect();
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform"),
            size: develop.params.bytes,
            usage: develop.params.usage,
            mapped_at_creation: false,
        });
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("arena"),
            size: end.next_multiple_of(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let region = |(at, bytes): (u64, u64)| wgpu::BufferBinding {
            buffer: &arena,
            offset: at,
            size: wgpu::BufferSize::new(bytes),
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        let decode_buffers = crate::gpu::DecodeBuffers {
            params: region(d[0]),
            packet: region(d[1]),
            streams: region(d[2]),
            coefs: region(d[3]),
            skipped: region(d[4]),
            samples: region(d[5]),
        };
        let info = pipelines
            .record_decode(&device, &queue, &mut encoder, &packet, &decode_buffers)
            .unwrap();
        let params = DemosaicParams::new(&info.header, info.quantizer());
        let develop_buffers = DevelopBuffers {
            params: uniform.as_entire_buffer_binding(),
            lut: region(v[0]),
            samples: region(d[5]),
            rb: region(v[1]),
            gr: region(v[2]),
            red: region(v[3]),
            green: region(v[4]),
            blue: region(v[5]),
        };
        pipelines
            .record_develop(
                &device,
                &queue,
                &mut encoder,
                &info.header,
                &params,
                Stage::Rec2020,
                16,
                &develop_buffers,
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        let all = crate::gpu::read_buffer(&device, &queue, &arena, arena.size()).unwrap();
        let host = crate::develop_planar(
            &crate::decode_frame(&packet).unwrap(),
            &params,
            Stage::Rec2020,
        );
        for (c, &(at, bytes)) in v[3..].iter().enumerate() {
            let plane: Vec<f32> = all[at as usize..(at + bytes) as usize]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            assert!(
                plane
                    .iter()
                    .zip(&host.planes[c])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "channel {c}"
            );
        }
    }

    #[test]
    fn develop_frame_uploads_a_host_frame() {
        let Some((device, queue)) = device() else {
            return;
        };
        let mut developer = GpuDeveloper::new(&device);
        let mut rng = Rng::new(22);
        let spec = Spec {
            valid_tiles: true,
            ..Spec::new(96, 48, crate::BayerPattern::Rggb)
        };
        let frame = crate::decode_frame(&random_frame(&spec, &mut rng)).unwrap();
        let params = DemosaicParams::new(&frame.header, frame.quantizer());
        let gpu = developer
            .develop_frame(&device, &queue, &frame, &params, Stage::Rec2020)
            .unwrap()
            .download(&device, &queue)
            .unwrap();
        let host = crate::develop_planar(&frame, &params, Stage::Rec2020);
        assert_eq!(compare(&gpu, &host).0, 0);
    }
}
