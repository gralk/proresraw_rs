# proresraw_rs

Pure-Rust decoder for Apple ProRes RAW (`aprn` / `aprh` QuickTime
tracks). One frame packet in, the Bayer mosaic of the sensor out, as
transform-domain code values plus everything the frame header says about
them: colour filter arrangement, linearization curve, black and white
levels, white balance factors, camera-to-XYZ matrix, gain and the
recommended crop.

The decoder stops where a raw developer wants to start. Code values are
what the inverse transform produces; `Frame::linearize` applies the
header curve to reach the linear sensor values that the black level
(256) and the header white level refer to, so the same normalization used
for DNG or camera raw applies.

```rust
let packet = std::fs::read("frame.bin")?;           // one QuickTime sample
let frame = proresraw_rs::decode_frame(&packet)?;
let linear = frame.linearize();                      // width * height u16
println!("{:?} white {}", frame.header.bayer_pattern, frame.header.white_level);
```

Demuxing is out of scope: any QuickTime reader that yields samples works.
`examples/decode.rs` turns a file of concatenated packets (as written by
`ffmpeg -c:v copy -f rawvideo`) into a 16-bit PGM.

## Status

- Bitstream versions 0 (Atomos recorders) and 1 (iPhone 17) decode.
- Tile alignments 0 to 4, partial edge tiles, custom quantization
  matrices and linearization curves are handled.
- RGGB and BGGR frames verified; GRBG and GBRG follow the same rule but
  no sample has been seen.
- Apple-compatible by default: the inverse transform, the validity rules
  and the decoding of malformed streams follow Apple's decoder, bit for
  bit. Invalid tiles are skipped as Apple's decoder skips them and are
  listed in `Frame::skipped_tiles`.
- With `decode_frame_with(packet, Idct::Ffmpeg)`, bit-exact with FFmpeg
  master's C `prores_raw` decoder on seven Atomos clips (`aprn` and
  `aprh`, partial edge tiles, qscale 15 and 45). The two transforms
  differ by ±1–3 code values on about a third of the samples.
- Tile rows decode in parallel on rayon (feature `parallel`, on by
  default). A 5.8K frame decodes in about 45 ms on a desktop CPU.

## Reduced resolution

`decode_frame_reduced(packet, Scale::Half | Quarter | Eighth)` decodes
linear RGB with one pixel per 2×2, 4×4 or 8×8 sensels, as Apple's
decoder does for previews: the second green stream is skipped, and
quarter and eighth resolution read only the leading 16 or 4 coefficients
of every block and run a 4×4 or 2×2 inverse transform. The transforms
are Apple-compatible bit for bit; half resolution equals the full decode
sampled at the red, first green and blue sensels. On a 4K frame, half,
quarter and eighth take about 16, 7 and 4 ms.

## Raw conversion

`develop(&frame, frame_quantizer(&parsed), stage)` runs Apple's standard
raw conversion and returns `f32` RGB. `stage` says how far to go; each
stage includes the ones before it:

| `Stage` | output |
|---|---|
| `Demosaiced` | camera RGB from the noise-aware demosaic, whose noise model follows the frame's quantizer; no white balance, clipped at sensor white |
| `WhiteBalanced` (default) | the header's white balance applied; red and blue clip where their gains reach 1.0, so all channels clip together |
| `Rec2020` | scene-linear Rec.2020 through Apple's output matrix (camera to XYZ from the header, XYZ to Rec.2020, header gain × 10/9) |

Stop before `Rec2020` to do your own colour management.
`develop_with(&frame, &params, stage)` takes explicit `DemosaicParams`,
for example your own white balance gains. The passes are also public
(`demosaic::green_pass`, `chroma_pass`, `apply_matrix`, with
`output_matrix(&header, Exposure::Reflection)` for Apple's other exposure
scaling).

The quantizer (first quantization matrix entry times the largest tile
qscale, invalid tiles included) equals the value Apple's decoder computes
on every sample clip. Compared with GPU runs of Apple's conversion on 4K
frames, the matrix pass is bit-exact on the same input and the demosaic
passes agree to within 2·10⁻⁷ on a 0–1 scale. The remaining differences
come from GPU division rounding, which varies between GPUs. Log and HLG
output curves are not implemented.

`develop` processes bands of rows so that every pass runs in cache. It
takes about 50 ms for a 4K frame on 32 threads when built with FMA
instructions (`-C target-cpu=x86-64-v3` or `native`), and about twice
that on the baseline x86-64 target, where each fused multiply-add is a
library call. Results are identical either way.

## GPU decoding

With the `gpu` feature, `GpuDecoder` decodes on a wgpu device the
application provides, with the Apple-compatible transform. The output
stays on the GPU in a storage buffer, ready for the next compute stage:

```rust
let mut decoder = proresraw_rs::GpuDecoder::new(&device);
let header = proresraw_rs::parse_frame(&packet)?.header;
let samples = device.create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: proresraw_rs::gpu::sample_bytes(&header),
    usage: wgpu::BufferUsages::STORAGE,
    mapped_at_creation: false,
});
decoder.decode_into(&device, &queue, &packet, &samples)?;
```

The buffer holds the code values of the coded picture (width and height
rounded up to 16), row-major, two per 32-bit word with the even column in
the low half. `GpuFrame::download` copies a frame back as a host `Frame`.
The crate enables no wgpu backend itself; the application chooses them.

The host parses and validates the frame and tile headers and uploads the
packet with a table of component streams. One compute pass decodes each
stream on its own invocation and stores dequantized coefficients,
bit-exactly with the host; a second runs the inverse transform per 16×16
sensel square. The transform's column pass is in `f32`, and WGSL does
not promise correctly rounded arithmetic, so a device may differ from the
host by one code value on rare samples; on an RTX 3060 (Vulkan) the
output is bit-exact with the host on every test frame.

Steady-state time per frame on an RTX 3060 (upload included) against the
parallel host decoder on the same machine:

| clip | size | GPU | host |
|---|---|---|---|
| Panasonic S5 via Atomos, HQ | 5888×3312 | 14.7 ms | 32.6 ms |
| Sony a7S III via Atomos, HQ | 4264×2408 | 6.1 ms | 23.8 ms |
| Sony FX6 via Ninja V | 3872×2192 | 3.0 ms | 14.7 ms |
| iPhone 17 Pro, HQ | 4224×3024 | 11.4 ms | 23.7 ms |

Entropy decoding is serial within a stream and dominates the GPU time.

## Design

`docs/bitstream.md` is the format description the code is written from.
The modules follow it: `header` (§3), `tile` (§4–5), `bits` and
`codebook` (§6.0–6.1), `coefficients` (§6.2–6.4), `idct` (§7), `curve`
(§8) and `decoder` (§2, placement). `gpu` and `gpu/decode.wgsl` mirror
§6–7 on the GPU; the shader's tables are generated from `codebook` and
`idct`. No unsafe code.

Random frames from `synth` (every alignment and Bayer pattern, extreme
quantizers, invalid tiles) must decode identically on the host and the
GPU. The Apple-compatible transform is checked against reference vectors
in `tests/data/`. `tests/oracle.rs` compares the FFmpeg transform
against FFmpeg's `prores_raw` decoder when
`PRORES_RAW_FIXTURES` points at fixtures made by a tool outside the
repository; it is skipped otherwise. The FFmpeg build must have x86
assembly disabled: with AVX2 its Bayer path inherits a transposed IDCT
permutation and every block comes out transposed.

## License

MIT. The format description was written from public sources and sample
footage; see `docs/bitstream.md` for provenance.
