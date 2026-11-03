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
- Output matches an Apple-decoded DNG conversion of an Atomos clip to
  within the DNG's 12-bit quantization.
- Tile rows decode in parallel on rayon (feature `parallel`, on by
  default). A 5.8K frame decodes in about 45 ms on a desktop CPU.

## Design

`docs/bitstream.md` is the format description the code is written from.
The modules follow it: `header` (§3), `tile` (§4–5), `bits` and
`codebook` (§6.1), `coefficients` (§6.2–6.4), `idct` (§7), `curve`
(§8) and `decoder` (§2, placement). No unsafe code.

`tests/oracle.rs` compares against FFmpeg's `prores_raw` decoder when
`PRORES_RAW_FIXTURES` points at fixtures made by a tool outside the
repository; it is skipped otherwise.

## License

MIT. The format description was written from public reverse-engineering
results and sample footage; see `docs/bitstream.md` for provenance.
