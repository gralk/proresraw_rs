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

`mov::read_video_track` finds a clip's frames: it reads only the
`moov` atom and returns the video track's samples (offset and size, in
presentation order after the first edit), frame rate, creation time and
the movie's text metadata, and `mov::read_at` reads a sample from any
thread. Any other QuickTime reader that yields samples works as well.
`examples/decode.rs` turns a file of concatenated packets (as written by
`ffmpeg -c:v copy -f rawvideo`) into a 16-bit PGM. `examples/prr2dng`
reads `.mov` clips directly and writes DNG sequences (see below).

### Orientation

A frame is decoded as the camera stored it. The header's width, height,
Bayer pattern and crop all describe that stored frame, and nothing in
the frame says which way is up: display orientation is recorded only by
the container, in the video track's display matrix (`tkhd`). Recorders
can store frames turned. The Blackmagic Camera app on the iPhone 17 Pro
stores 4:3 frames upside down and marks the track as turned 180°
(matrix `[-1 0; 0 -1]`), while its 16:9 frames are stored upright. The
frame headers of the two are otherwise alike. A reader that ignores the
matrix shows those clips upside down.

`mov::Track::orientation` reads the matrix as one of the eight EXIF
orientations (`Orientation::exif()` gives the 1–8 value). Turn the
picture after decoding, and remember that the Bayer pattern and crop
turn with it (RGGB turned 180° is BGGR). A matrix that is not a quarter
turn or mirror leaves the frames as stored, with a warning on the track.

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
takes about 50 ms for a 4K frame on 32 threads. On x86-64 the default
feature `runtime-fma` runs the conversion kernels with AVX2 and FMA
instructions when the CPU has them, through one small `unsafe` dispatch;
without the feature the crate forbids unsafe code, and the conversion
falls back to a library call per fused multiply-add, about half the
speed unless the build targets FMA itself (`-C target-cpu=x86-64-v3` or
`native`). Results are bit-identical on every path.

## DNG conversion

`examples/prr2dng` is a command-line tool that converts ProRes RAW
`.mov` clips to DNG image sequences. It reads the clips with the
crate's `mov` reader, so no FFmpeg step is needed.

### Building

```sh
cargo build --release --example prr2dng
```

The binary is `target/release/examples/prr2dng`. To put it on your
`PATH` (in `~/.cargo/bin`), install it instead:

```sh
cargo install --path . --example prr2dng
```

### Usage

```
prr2dng [-o DIR] [--linear] [--uncompressed] [--frames FIRST[-[LAST]]] [--overwrite] CLIP.mov...
```

| option | effect |
|---|---|
| `-o`, `--output DIR` | folder for the per-clip sequences (default: the current directory) |
| `--linear` | store linear values with no `LinearizationTable` (see below) |
| `--uncompressed` | store uncompressed samples instead of lossless JPEG |
| `--frames N`, `N-M`, `N-` | convert only these frames (0-based, inclusive) |
| `--overwrite` | replace existing DNG files instead of stopping |

Frame `n` of `CLIP.mov` is written to `DIR/CLIP/CLIP.nnnnnn.dng`,
numbered by its frame index in the clip. For example:

```sh
# Every frame of two clips, into ./A001_C010/ and ./A001_C011/
prr2dng A001_C010.mov A001_C011.mov

# Frames 100 to 199 of one clip, into /scratch/dng/A001_C010/
prr2dng -o /scratch/dng --frames 100-199 A001_C010.mov
```

Every clip is checked before anything is written: its video track must
be `aprn` or `aprh` and its first frame must parse, so a wrong file
stops the run up front. For each clip the tool prints the format,
frame count, frame rate, camera, as-shot white balance and exposure,
then its progress. Existing files stop the run unless `--overwrite` is
given; files are written under a temporary name and renamed when
complete, so an interrupted run leaves no truncated DNGs. A frame that
fails to decode is reported and skipped, and the tool then exits with an
error status.

### What the files contain

Each DNG holds the frame's 16-bit code values with the frame's curve as
a 65536-entry `LinearizationTable`, so readers linearize exactly as
`Frame::linearize` does. Black (256) and white levels apply after the
table, as in ProRes RAW. Colour comes from the header:

| DNG tag | from |
|---|---|
| `ColorMatrix1` | header matrix and white balance, its D65 adaptation undone (Bradford) to the header's colour temperature |
| `ForwardMatrix1` | header matrix adapted from D65 to D50 |
| `AsShotNeutral` | `(1/wbR, 1, 1/wbB)` |
| `BaselineExposure` | `log2(gain)` |
| `DefaultCrop*` | header crop margins |
| `Make`, `Model`, `FrameRate`, `DateTime` | movie metadata and track |

The image is stored as 256×256 lossless JPEG tiles, about 1.4× smaller
than uncompressed. `--linear` stores `Frame::linearize()` output with no
table instead. That is the precision decoders output, and it compresses
better (a 4K frame takes about 11 MB instead of 15 MB), but it drops the
code values' sub-LSB detail in the shadows. LibRaw decodes the
compressed and uncompressed files of every sample clip to identical
data, and darktable and RawTherapee render them.

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
`idct`. `mov` is the QuickTime container reader, outside the bitstream
description. No unsafe code.

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
