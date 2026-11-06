//! The raw conversion on the GPU: [`crate::develop_planar`], pass for pass.

use std::sync::Arc;

use super::{GpuError, GpuFrame, GpuPipelines, Slot, entry, read_buffers_with, sample_bytes};
use crate::decoder::Frame;
use crate::demosaic::{Conversion, DemosaicParams, RgbPlanes, Stage};
use crate::header::FrameHeader;

/// Invocations per side of a workgroup of both passes.
const GROUP_SIDE: u32 = 16;
/// Bytes of the parameter block.
const PARAMS_BYTES: usize = 112;

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
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage = |read_only| wgpu::BufferBindingType::Storage { read_only };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prores raw develop"),
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform),
                buffer(1, storage(true)),
                buffer(2, storage(true)),
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
        let needed = sample_bytes(header);
        let plane = plane_bytes(header);
        for (buffer, needed) in std::iter::once((samples, needed)).chain(out.map(|b| (b, plane))) {
            if buffer.size() < needed {
                return Err(GpuError::OutputTooSmall {
                    size: buffer.size(),
                    needed,
                });
            }
            if !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
                return Err(GpuError::OutputNotStorage);
            }
        }
        let limits = device.limits();
        let limit = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        if plane > limit {
            return Err(GpuError::TooLarge {
                what: "plane",
                bytes: plane,
                limit,
            });
        }
        let conversion = Conversion::new(header, params, stage);
        let block = params_block(header, &conversion);
        let params_buf = self.params.get(device, PARAMS_BYTES as u64);
        queue.write_buffer(params_buf, 0, &block);
        let lut: Vec<u8> = conversion
            .lut
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let lut_buf = self.lut.get(device, lut.len() as u64);
        queue.write_buffer(lut_buf, 0, &lut);
        let rb = self.rb.get(device, plane);
        let gr = self.gr.get(device, plane);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prores raw develop frame"),
            layout: &self.pipelines.develop.layout,
            entries: &[
                entry(0, params_buf),
                entry(1, lut_buf),
                entry(2, samples),
                entry(3, rb),
                entry(4, gr),
                entry(5, out[0]),
                entry(6, out[1]),
                entry(7, out[2]),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prores raw develop"),
        });
        let groups = (
            u32::from(header.width).div_ceil(GROUP_SIDE),
            u32::from(header.height).div_ceil(GROUP_SIDE),
        );
        let develop = &self.pipelines.develop;
        for (label, pipeline) in [("green", &develop.green), ("chroma", &develop.chroma)] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        queue.submit([encoder.finish()]);
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

/// The shader's `Params` block.
fn params_block(header: &FrameHeader, c: &Conversion) -> Vec<u8> {
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
            Err(GpuError::OutputTooSmall { .. })
        ));
        assert!(matches!(
            run(&frame.samples, &buffer(plane, wgpu::BufferUsages::COPY_DST)),
            Err(GpuError::OutputNotStorage)
        ));
        assert!(matches!(
            run(&buffer(16, storage), &buffer(plane, storage)),
            Err(GpuError::OutputTooSmall { .. })
        ));
        assert!(run(&frame.samples, &buffer(plane, storage)).is_ok());
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
