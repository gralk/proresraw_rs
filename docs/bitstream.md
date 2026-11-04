# Apple ProRes RAW bitstream description

Written 2026-11-03 as the clean-room specification for `proresraw_rs`,
revised 2026-11-04 for Apple-compatible decoding. Sources: the publicly
documented behaviour of the FFmpeg `prores_raw` decoder (libavcodec,
master as of 2026-09-23, by Paul B Mahol and Lynne) and direct
inspection of frames from an Atomos Ninja V / Shogun 7 (Panasonic S5,
Sony FX6, Sony a7S III sources) and an iPhone 17 (Final Cut Camera). No
FFmpeg code is reproduced here; the implementation is written from this
document. Codebook tables and transform constants are facts about the
format, not expression.

Where Apple's decoder differs from FFmpeg's (the inverse transform, the
validity rules, and what malformed streams decode to), this description
follows Apple's decoder. Those points are marked *Apple-compatible*.

Reference-decoder facts that FFmpeg's commit history documents and that
this description relies on:

- a component ends when the 32-bit view of the stream is all zero;
- the camera-supplied 8-point curve must be applied to reach linear
  sensor values;
- edge tiles halve in width step by step;
- dequantized coefficients saturate to 16 bits and the transform output
  is centered at zero and biased by 32768 on output.

All arithmetic described below on symbols and positions is unsigned
32-bit and wraps; this only matters for malformed streams.

All multi-byte integers are big-endian. Floats are IEEE 754 binary32.

## 1. Container

ProRes RAW frames are samples of a QuickTime video track whose sample
description FourCC is `aprn` (ProRes RAW) or `aprh` (ProRes RAW HQ).
The profile changes only the encoder's quantization, not the syntax.
Every frame is an intra frame.

## 2. Frame layout

```
frame
  u32   frame_size          byte length of the whole frame (normally the sample size)
  u32   'prrf'              frame tag
  u16   header_size         byte length of the frame header including this field (>= 72)
  frame_header              header_size - 2 bytes
  u16   tile_size[nb_tiles] byte length of each tile, in tile order
  tile[nb_tiles]            tile payloads, back to back
  vendor metadata           optional, until frame end (see §9)
```

`header_size` counts itself: the frame header body starts right after it
and is `header_size - 2` bytes long. Observed sizes: 136 (Atomos, with
quant matrix, no curve), 152 (iPhone, with matrix and curve).

*Apple-compatible:* `frame_size` must be non-zero and at most the sample
size; it does not otherwise limit reading, which uses the whole sample.
A tile whose size table entry reaches past the sample makes the frame
undecodable.

## 3. Frame header body

Offsets are relative to the start of the body.

| offset | type | field |
|---:|---|---|
| 0 | u8 | reserved |
| 1 | u8 | version, 0 or 1 |
| 2 | 4 bytes | vendor tag (`atm0` Atomos, `peac` Panasonic, `apl0` Apple) |
| 6 | u16 | width in sensels, even and non-zero |
| 8 | u16 | height in sensels, even and non-zero |
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

The fixed body is 70 bytes, hence `header_size >= 72`; the optional
matrix and curve must also fit within `header_size`. Fields beyond the
known ones (the Atomos vendor writes nothing extra; Apple writes nothing
extra) are ignored.

*Apple-compatible* validity rules, each of which makes the frame
undecodable: version other than 0 or 1; a curve in a version 0 header;
zero or odd dimensions; alignment above 4; a quantization matrix entry
outside 1..63; a curve that does not rise strictly.

### 3.1 Bayer pattern

The usual enumeration: 0 RGGB (red at even row, even column), 1 GRBG,
2 GBRG, 3 BGGR. The pattern decides where the four colour-ordered
component streams of every tile land (§5). Every Atomos clip observed
uses 0, confirmed RGGB by a DNG conversion of the same clip; the iPhone 17
writes 3, confirmed BGGR by colour rendering through the header matrix
and by sub-sensel phase correlation of the decoded phases (see
`footage-survey.md`). Codes 1 and 2 follow the same rule but no sample
has been seen.

### 3.2 Levels

Decoded linear values (§8) are 16-bit. Black is fixed at 256 and white
is `sensel_value_range + 256`. A DNG conversion of an a7S
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
Entries are 1..63 (§3), so `qmat[i] * qscale` always fits in 16 bits.

### 3.5 Linearization curve

Eight u16 control points `c[0..8]` at transform-domain inputs
`0, 8192, …, 57344`, with an implicit ninth point of 65536 at input
65536. The default when absent is `0, 512, 1024, 2048, 4096, 8192,
16384, 32768`, a doubling per segment. The curve is the camera's
inverse transfer function combined with any encoder shaping, and must be
applied to obtain linear sensor values (§8). Only version 1 headers carry
a curve, and its points rise strictly (§3).

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

Observed `header_len` is 8.

*Apple-compatible:* a tile is valid when `header_len >= 8`, `qscale` is
1..255, `size[0..3]` are non-zero and the derived fourth size is
positive. An invalid tile does not make the frame undecodable: it is
skipped and its sensels are not written (a decoder that allocates the
output leaves them 0).

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

A component stream is read MSB first as a sequence of codewords through
a 32-bit view of the stream that is zero-filled past its end.

### 6.0 Bit reader

For valid streams the view is simply the next 32 bits. To decode
malformed streams *Apple-compatibly*, the reader is defined exactly:

- State: a 64-bit window `W` whose top 32 bits are the view, a signed
  count `x` of valid bits below the view, and a byte position.
- Setup, for a stream starting `a` bytes (0..3) past a 4-byte boundary,
  taking the sample as 4-byte aligned: load the first 4 bytes (or all of
  a shorter stream, zero-filled) as the top word. Then, if the next
  4-byte boundary is `k > 0` bytes away and at least `k` bytes remain,
  load those `k` bytes into the bits just below (`x = 8k`); otherwise
  `x = 0`. Later loads are whole big-endian words while at least four
  bytes remain, then the remaining bytes zero-filled, then zeros.
- Consuming `len` bits: `W <<= len mod 64`; `x -= len`; if `x < 0`, load
  one word `w` and set `W |= w << (−(x as signed 8-bit) mod 64)`,
  `x += 32`. Only one word is loaded per step.

With every codeword at most 32 bits long this keeps the view equal to
the next 32 bits of the stream. Over-long codewords (§6.1) desynchronize
it, and later symbols are read from whatever the window then holds.

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
- If the whole view is zero (including the case of no bits left), the
  symbol is *end of component*: decoding of this component stops and all
  remaining coefficients are zero. The view is tested before every
  symbol, including before each level of a run.
- No valid codeword is longer than 32 bits. *Apple-compatible:* a longer
  one is still decoded with the formulas above, with every shift of the
  32-bit view taken modulo 32 (the value term reads `view >> ((32 − L)
  mod 32)`), and consumes `L` bits as in §6.0.

Symbols of valid codewords are below 2²⁴. Coefficients are stored as
16-bit values (§6.3, §6.4) before dequantization.

### 6.2 Coefficient order

Coefficients of all `nb_blocks` blocks of the component are interleaved
by frequency: position `n` (from 0) refers to block `n & (nb_blocks-1)`
at scan index `n >> log2_nb_blocks`. Positions `0 .. nb_blocks-1` are the
DC coefficients; `nb_codes = 64 * nb_blocks` positions in total. The
scan index maps to the raster coefficient index through the ProRes
"interlaced" zig-zag (§6.5).

### 6.3 DC coefficients

The first DC symbol uses parameter `0x2BC` `(s=2, e=11, r=12)` and is
sign-folded: `dc0 = ((((v & 1) + v) >> 1) ^ -(v & 1)) + (v & 1)`, that is
`v / 2` for even `v` and `-(v + 1) / 2` for odd `v`.

Each following block `n = 1 .. nb_blocks-1` codes a difference symbol
`v` with parameter:

- `0x064` `(s=0, e=6, r=4)` when `n mod 16 == 1`;
- otherwise `DC_CB[min((v_prev + 1) >> 1, 12)]` where `v_prev` is the
  previous DC symbol and `DC_CB` is

  `[0x010, 0x021, 0x032, 0x033, 0x033, 0x033, 0x044, 0x044, 0x044, 0x044, 0x044, 0x044, 0x076]`.

The difference is reconstructed with a running sign: let `sign` start
at 0. `sign ^= v & 1`; `delta = ((v & 1) + v) >> 1` negated when `sign`;
`sign = delta < 0` (a zero delta clears the sign); `dc += delta`. The
running `dc` is a 32-bit integer; each block stores its low 16 bits as a
signed coefficient.

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
        sign  = bit (L mod 32) of the view before the codeword, from the top
        coef[n] = ±(level + 1); n += 1; stop if n == nb_codes
    zeros = symbol(rn); rn = RN_CB[min(zeros, 27)]
    n += zeros + 1                          a zero run is zeros + 1 long
    stop if n >= nb_codes
    level = symbol(ac); ac = AC_CB[min(level, 94)]; ln = LN_CB[min(level, 14)]
    sign  = as above
    coef[n] = ±(level + 1); n += 1
```

The sign is the bit after the codeword, and the codeword and sign
consume `L + 1` bits together. (For a codeword of exactly 32 bits the
rule reads the view's top bit, which is 0.) `level + 1` is stored as its
low 16 bits, negated in 16 bits when the sign is 1. Any end-of-component
symbol leaves the remaining coefficients zero.

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

For each block, in raster coefficient order, with the 16-bit
coefficients of §6:

`d[i] = clamp(coef[i] * (qmat[i] * qscale), -32768, 32767)`

The quantizer is at most 63 × 255, so the product is exact in 32 bits.
The 8×8 block `d` is then inverse transformed. Two transforms are
described: the *Apple-compatible* one (§7.1), which is the default, and
FFmpeg's (§7.2). They agree on blocks with only a DC coefficient and
differ by ±1–3 code values on about a third of the samples of real
footage.

### 7.1 Apple-compatible transform

**Row pass**, integer, no rounding and no shift. Row `k` of `d` uses
table set `S = [0, 1, 2, 3, 0, 3, 2, 1][k]`. A set is four tables
`T0..T3` of eight values. For output lane `j` = 0..3:

```
even[j] = x0*T0[2j] + x2*T0[2j+1] + x4*T1[2j] + x6*T1[2j+1]
odd[j]  = x1*T2[2j] + x3*T2[2j+1] + x5*T3[2j] + x7*T3[2j+1]
y[j] = even[j] + odd[j];   y[7-j] = even[j] - odd[j]
```

The sums fit in 32 bits for any 16-bit input.

```
set 0  T0 = 4096, 5352, 4096, 2217, 4096,-2217, 4096,-5352
       T1 = 4096, 2217,-4096,-5352,-4096, 5352, 4096,-2217
       T2 = 5681, 4816, 4816,-1130, 3218,-5681, 1130,-3218
       T3 = 3218, 1130,-5681,-3218, 1130, 4816, 4816,-5681
set 1  T0 = 5681, 7423, 5681, 3075, 5681,-3075, 5681,-7423
       T1 = 5681, 3075,-5681,-7423,-5681, 7423, 5681,-3075
       T2 = 7880, 6681, 6681,-1567, 4464,-7880, 1567,-4464
       T3 = 4464, 1567,-7880,-4464, 1567, 6681, 6681,-7880
set 2  T0 = 5352, 6992, 5352, 2896, 5352,-2896, 5352,-6992
       T1 = 5352, 2896,-5352,-6992,-5352, 6992, 5352,-2896
       T2 = 7423, 6293, 6293,-1477, 4205,-7423, 1477,-4205
       T3 = 4205, 1477,-7423,-4205, 1477, 6293, 6293,-7423
set 3  T0 = 4816, 6293, 4816, 2607, 4816,-2607, 4816,-6293
       T1 = 4816, 2607,-4816,-6293,-4816, 6293, 4816,-2607
       T2 = 6681, 5663, 5663,-1329, 3784,-6681, 1329,-3784
       T3 = 3784, 1329,-6681,-3784, 1329, 5663, 5663,-6681
```

**Column pass**, IEEE binary32, every operation rounded to nearest-even
and none fused. Each row output is converted to binary32
(round-to-nearest-even) and multiplied by 2⁻¹². With `r0..r7` a column
of these values (`r_k` from row `k`) and the constants

```
SCALE = 2^-12     (0x39800000)    BIAS = 32768.5    (0x47000080)
TAN1  = tan(π/16)  (0x3e4bafaf)    TAN2 = tan(π/8)   (0x3ed413cd)
TAN3  = tan(3π/16) (0x3f2b0dc1)    COS4 = cos(π/4)   (0x3f3504f3)
```

evaluate, in this order:

```
p  = r0 + BIAS;   ea = p + r4;   eb = p - r4
f  = r6*TAN2 + r2;   g = r2*TAN2 - r6
e0 = ea + f;   e3 = ea - f;   e1 = eb + g;   e2 = eb - g
a  = r7*TAN1 + r1;   b = r1*TAN1 - r7
c  = r5*TAN3 + r3;   d = r5 - r3*TAN3
o0 = a + c;   t = a - c;   o3 = b + d;   u = b - d
o1 = (t + u)*COS4;   o2 = (t - u)*COS4
y0 = e0 + o0   y1 = e1 + o1   y2 = e2 + o2   y3 = e3 + o3
y4 = e3 - o3   y5 = e2 - o2   y6 = e1 - o1   y7 = e0 - o0
```

`y_i` is row `i` of the column. The bias already includes the output
offset of 32768 and the rounding half: the code value is `u =
clamp(trunc(y), 0, 65535)`, truncating toward zero.

### 7.2 FFmpeg transform

A separable integer 8×8 inverse DCT with 32-bit intermediates, the
"simple IDCT" with constants
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
