# Apple ProRes RAW bitstream description

Written 2026-11-03 as the clean-room specification for `proresraw_rs`.
Sources: the publicly documented behaviour of the FFmpeg `prores_raw`
decoder (libavcodec, master as of 2026-09-23, reverse engineered by
Paul B Mahol and Lynne) and direct inspection of frames from an Atomos
Ninja V / Shogun 7 (Panasonic S5, Sony FX6, Sony a7S III sources) and an
iPhone 17 (Final Cut Camera). No FFmpeg code is reproduced here; the
implementation is written from this document. Codebook tables and
transform constants are facts about the format, not expression.

Reference-decoder facts that FFmpeg's commit history documents and that
this description relies on:

- a component ends when a 32-bit zero-filled window is all zero;
- the camera-supplied 8-point curve must be applied to reach linear
  sensor values;
- edge tiles halve in width step by step;
- dequantized coefficients saturate to 16 bits and the transform output
  is centered at zero and biased by 32768 on output.

All multi-byte integers are big-endian. Floats are IEEE 754 binary32.

## 1. Container

ProRes RAW frames are samples of a QuickTime video track whose sample
description FourCC is `aprn` (ProRes RAW) or `aprh` (ProRes RAW HQ).
The profile changes only the encoder's quantization, not the syntax.
Every frame is an intra frame.

## 2. Frame layout

```
frame
  u32   frame_size          byte length of the whole frame (equals the sample size)
  u32   'prrf'              frame tag
  u16   header_size         byte length of the frame header including this field (>= 62)
  frame_header              header_size - 2 bytes
  u16   tile_size[nb_tiles] byte length of each tile, in tile order
  tile[nb_tiles]            tile payloads, back to back
  vendor metadata           optional, until frame end (see §9)
```

`header_size` counts itself: the frame header body starts right after it
and is `header_size - 2` bytes long. Observed sizes: 136 (Atomos, with
quant matrix, no curve), 152 (iPhone, with matrix and curve).

## 3. Frame header body

Offsets are relative to the start of the body.

| offset | type | field |
|---:|---|---|
| 0 | u8 | reserved |
| 1 | u8 | version (0 and 1 are known) |
| 2 | 4 bytes | vendor tag (`atm0` Atomos, `peac` Panasonic, `apl0` Apple) |
| 6 | u16 | width in sensels, even |
| 8 | u16 | height in sensels, even |
| 10 | u8 ×4 | recommended crop: left, right, top, bottom (margins to drop after demosaic) |
| 14 | u16 | bayer pattern, low two bits (§3.1) |
| 16 | u16 | sensel value range: white level minus 256 (§3.2) |
| 18 | f32 | white balance red factor |
| 22 | f32 | white balance blue factor |
| 26 | f32 ×9 | color matrix, row-major, camera RGB to CIE XYZ (D65) |
| 62 | f32 | gain factor applied after the matrix |
| 66 | u16 | white balance correlated color temperature, Kelvin (informational) |
| 68 | u16 | flags (§3.3) |
| 70 | u8 ×64 | quantization matrix, present when flags bit 0 is set (§3.4) |
| 70 or 134 | u16 ×8 | linearization curve, present when flags bit 4 is set (§3.5) |

The minimum body is 60 bytes, hence `header_size >= 62`. Fields beyond
the known ones (the Atomos vendor writes nothing extra; Apple writes
nothing extra) are ignored.

### 3.1 Bayer pattern

The usual enumeration: 0 RGGB (red at even row, even column), 1 GRBG,
2 GBRG, 3 BGGR. The pattern decides where the four colour-ordered
component streams of every tile land (§5). Every Atomos clip observed
uses 0, confirmed RGGB by an Assimilate DNG conversion; the iPhone 17
writes 3, confirmed BGGR by colour rendering through the header matrix
and by sub-sensel phase correlation of the decoded phases (see
`footage-survey.md`). Codes 1 and 2 follow the same rule but no sample
has been seen.

### 3.2 Levels

Decoded linear values (§8) are 16-bit. Black is fixed at 256 and white
is `sensel_value_range + 256`. An Assimilate DNG conversion of an a7S
III clip carries BlackLevel 256 and WhiteLevel 65279 for a header range
of 0xFDFF, which confirms both the formula and that the levels apply to
the linearized output, not to the transform domain.

### 3.3 Flags

| bit | meaning |
|---:|---|
| 0 | quantization matrix present |
| 1-3 | tile alignment `align`, 0 to 4; the largest tile is `16 << align` sensels wide |
| 4 | linearization curve present |
| 5 | unknown, set by Panasonic and Apple |

### 3.4 Quantization matrix

64 unsigned bytes in raster order of the 8×8 coefficient block (row
major, DC first). When absent every entry is 1. Observed: Atomos writes
an all-ones matrix explicitly; Apple writes a perceptual matrix.

### 3.5 Linearization curve

Eight u16 control points `c[0..8]` at transform-domain inputs
`0, 8192, …, 57344`, with an implicit ninth point of 65536 at input
65536. The default when absent is `0, 512, 1024, 2048, 4096, 8192,
16384, 32768`, a doubling per segment. The curve is the camera's
inverse transfer function combined with any encoder shaping, and must be
applied to obtain linear sensor values (§8).

## 4. Tile grid

Tiles are 16 sensel rows tall and a multiple of 16 sensels wide. Let
`tw16 = ceil(width / 16)` and `nb_th = ceil(height / 16)`.

Each tile row is partitioned left to right into the largest power-of-two
units first: for `e = align, align-1, …, 0`, while `(1 << e)` sixteen-
sensel columns remain, emit a tile of `1 << e` columns with
`log2_nb_blocks = e`. The number of tiles per row is therefore
`tw16 >> align` plus the population count of `tw16 & ((1 << align) - 1)`,
and `nb_tiles = nb_tw * nb_th`. Tiles are stored row-major in that order.

The tile size table immediately follows the frame header. Tile payloads
follow the table back to back, so tile `i` starts at
`10 + (header_size - 2) + 2 * nb_tiles + sum(tile_size[0..i])`.

Rows and columns beyond `width`/`height` inside the last tiles are coded
and decoded; they are padding and are cropped on output. The coded size
is the size rounded up to 16.

## 5. Tile payload

```
tile
  u8    header_len << 3    byte length of this tile header, times 8 (value >> 3)
  u8    qscale             quantizer scale, multiplies the frame quant matrix
  u16   size[0..3]         byte length of the first three component streams
  (padding to header_len)
  component stream 0       size[0] bytes
  component stream 1       size[1] bytes
  component stream 2       size[2] bytes
  component stream 3       tile_size - header_len - size[0] - size[1] - size[2] bytes
```

Observed `header_len` is 8. A negative derived fourth size is an error.

The four streams carry the four Bayer phases, one 8×8 block per 16×16
sensel square. The streams are colour-ordered: green, green, blue, red.
With the red sensel of the pattern at `(r, c)` (row parity, column
parity), the streams land on:

| stream | colour | phase | RGGB | BGGR |
|---:|---|---|---|---|
| 0 | green | (1−r, c) | (1, 0) | (0, 1) |
| 1 | green | (r, 1−c) | (0, 1) | (1, 0) |
| 2 | blue | (1−r, 1−c) | (1, 1) | (0, 0) |
| 3 | red | (r, c) | (0, 0) | (1, 1) |

Placing the streams of a BGGR frame as if it were RGGB gives the right
colours per stream but shifts every sample one sensel diagonally, which
phase correlation against the green average detects.

A tile with `log2_nb_blocks = e` holds `nb_blocks = 1 << e` blocks per
component, block `b` covering sensel columns `16b .. 16b+15` of the tile.

## 6. Component stream: entropy coding

A component stream is read MSB first as a sequence of codewords with a
32-bit look-ahead window that is zero-filled past the end of the stream.

### 6.1 Codewords

Every symbol is an unsigned integer from an adaptive Golomb-Rice /
exponential-Golomb hybrid code described by a 12-bit parameter
`(s, e, r)` packed as `(s << 8) | (e << 4) | r`:

- `r` Rice order, `e` exp-Golomb order, `s` switch threshold.
- Let `q` be the number of leading zero bits in the window.
- If `q <= s`: Rice codeword. Skip the `q` zeros and the one bit, read
  `r` bits `t`; value = `(q << r) | t`. Length `q + 1 + r`.
- Otherwise: exp-Golomb codeword of order `e` offset by the Rice range.
  Total length `L = 2q - s + e`; value = (the first `L` window bits as an
  integer) + `((s + 1) << r)` − `(1 << e)`. Equivalently, after the
  first `s + 1` zeros, an ordinary order-`e` exp-Golomb code plus
  `(s + 1) << r`.
- If the whole window is zero (including the case of no bits left) or
  `L > 32`, the symbol is *end of component*: decoding of this component
  stops and all remaining coefficients are zero. No valid codeword is
  longer than 32 bits.

### 6.2 Coefficient order

Coefficients of all `nb_blocks` blocks of the component are interleaved
by frequency: position `n` (from 0) refers to block `n & (nb_blocks-1)`
at scan index `n >> log2_nb_blocks`. Positions `0 .. nb_blocks-1` are the
DC coefficients; `nb_codes = 64 * nb_blocks` positions in total. The
scan index maps to the raster coefficient index through the ProRes
"interlaced" zig-zag (§6.5).

### 6.3 DC coefficients

The first DC symbol uses parameter `0x2BC` `(s=2, e=11, r=12)` and is
sign-folded: `dc0 = (v >> 1) ^ -(v & 1)` (even v non-negative, odd v
negative).

Each following block `n = 1 .. nb_blocks-1` codes a difference symbol
`v` with parameter:

- `0x064` `(s=0, e=6, r=4)` when `n mod 16 == 1`;
- otherwise `DC_CB[min((v_prev + 1) >> 1, 12)]` where `v_prev` is the
  previous DC symbol and `DC_CB` is

  `[0x010, 0x021, 0x032, 0x033, 0x033, 0x033, 0x044, 0x044, 0x044, 0x044, 0x044, 0x044, 0x076]`.

The difference is reconstructed with a running sign: let `sign` start
at 0. `sign ^= v & 1`; `delta = ((v + 1) >> 1)` negated when `sign`;
`sign = delta < 0` (a zero delta clears the sign); `dc += delta`.

### 6.4 AC coefficients

Positions `nb_blocks .. nb_codes-1` are coded as alternating runs of
non-zero levels and runs of zeros. Three adaptive parameters persist
across the component: `ac = 0x031` `(s=0, e=3, r=1)`, `rn = 0x000`,
`ln = 0x042` `(s=0, e=4, r=2)`.

```
loop while n < nb_codes:
    count = symbol(ln)                      length of the non-zero run
    repeat count times:
        level = symbol(ac); ac = AC_CB[min(level, 94)]
        sign  = 1 bit
        coef[n] = ±(level + 1); n += 1; stop if n == nb_codes
    zeros = symbol(rn); rn = RN_CB[min(zeros, 27)]
    n += zeros + 1                          a zero run is zeros + 1 long
    stop if n >= nb_codes
    level = symbol(ac); ac = AC_CB[min(level, 94)]; ln = LN_CB[min(level, 14)]
    sign  = 1 bit
    coef[n] = ±(level + 1); n += 1
```

Any end-of-component symbol leaves the remaining coefficients zero.

Tables:

```
AC_CB[95] = 0x000, 0x211, 0x111, 0x111, 0x222, 0x222, 0x222, 0x122, 0x122, 0x122,
            0x233 ×8, 0x133 ×2, 0x244 ×22, 0x355 ×52, 0x166
RN_CB[28] = 0x200, 0x100, 0x000, 0x000, 0x211, 0x211, 0x111, 0x111, 0x011, 0x011,
            0x021, 0x021, 0x222, 0x022 ×12, 0x032, 0x032, 0x044
LN_CB[15] = 0x100, 0x111, 0x222, 0x222, 0x122, 0x122, 0x433, 0x433, 0x233 ×6, 0x033
```

(`×k` means the entry repeated k times; AC_CB indices 10–17 are 0x233,
18–19 are 0x133, 20–41 are 0x244, 42–93 are 0x355, 94 is 0x166.)

### 6.5 Scan

Scan index to raster index (row-major 8×8), the ProRes interlaced scan:

```
 0,  8,  1,  9, 16, 24, 17, 25,
 2, 10,  3, 11, 18, 26, 19, 27,
32, 40, 33, 34, 41, 48, 56, 49,
42, 35, 43, 50, 57, 58, 51, 59,
 4, 12,  5,  6, 13, 20, 28, 21,
14,  7, 15, 22, 29, 36, 44, 37,
30, 23, 31, 38, 45, 52, 60, 53,
46, 39, 47, 54, 61, 62, 55, 63
```

## 7. Dequantization and inverse transform

For each block, in raster coefficient order:

`d[i] = clamp(coef[i] * qmat[i] * qscale, -32768, 32767)`

followed by a separable integer 8×8 inverse DCT with 32-bit
intermediates, the "simple IDCT" with constants
`W1..W7 = 22725, 21407, 19265, 16384, 12873, 8867, 4520`
(cos(kπ/16)·√2·2¹⁴ rounded, W4 forced to 2¹⁴), row shift 13 and column
shift 15:

- Row pass, for each row: with rounding term `1 << 12` added to the W4·x0
  product, compute the even part `a0..a3` from `x0, x2, x4, x6` and the
  odd part `b0..b3` from `x1, x3, x5, x7`, outputs
  `y0 = (a0+b0)>>13, y7 = (a0-b0)>>13, y1 = (a1+b1)>>13, y6 = (a1-b1)>>13,
   y2 = (a2+b2)>>13, y5 = (a2-b2)>>13, y3 = (a3+b3)>>13, y4 = (a3-b3)>>13`.
- Column pass, same butterfly on each column of the row results, with
  the rounding term folded in as `W4 * (x0 + 1)` and arithmetic right
  shift by 15.

Butterfly (shared by both passes, x = inputs, arithmetic in wrapping
32-bit integers, shifts arithmetic):

```
a0 = W4*x0 + W2*x2 + W4*x4 + W6*x6
a1 = W4*x0 + W6*x2 - W4*x4 - W2*x6
a2 = W4*x0 - W6*x2 - W4*x4 + W2*x6
a3 = W4*x0 - W2*x2 + W4*x4 - W6*x6
b0 = W1*x1 + W3*x3 + W5*x5 + W7*x7
b1 = W3*x1 - W7*x3 - W1*x5 - W5*x7
b2 = W5*x1 - W1*x3 + W7*x5 + W3*x7
b3 = W7*x1 - W5*x3 + W3*x5 - W1*x7
```

The result is the centered sensel value `t`; the transform-domain code
value is `u = clamp(t + 32768, 0, 65535)`.

## 8. Linearization

Linear output `L(u)` interpolates the curve of §3.5 piecewise linearly
with 13-bit segments and round-to-nearest:

```
seg  = u >> 13            0..7
frac = u & 0x1FFF
c0   = curve[seg]
c1   = seg < 7 ? curve[seg + 1] : 65536
L    = min((c0 * 8192 + (c1 - c0) * frac + 4096) >> 13, 65535)
```

Black (256) and white (§3.2) are levels of `L`.

## 9. Trailer

Bytes after the last tile and before the frame end are vendor records:
`u32 size, 4-byte tag 'psim', Pascal string name, payload`, repeated,
optionally closed by `u32 8, 'eomd'`, then zero padding. Panasonic
writes `com.panasonic.Semi-Pro.optical_correction` with IEEE doubles.
The decoder ignores the trailer.

## 10. Observed header values

See `footage-survey.md` for the dumps of every sample clip.
