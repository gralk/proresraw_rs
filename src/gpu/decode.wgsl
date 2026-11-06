// ProRes RAW decoding on the GPU (bitstream description §6 and §7).
//
// Pass 1, `decode_streams`: one invocation per tile component stream reads
// its codewords and stores the dequantized, 16-bit-saturated coefficients
// of its blocks into the cleared coefficient buffer, bit-exactly with the
// host. Pass 2, `transform`: one workgroup per 16x16 sensel square
// inverse-transforms the four blocks of the square with the Apple-compatible transform
// and writes its code values. Integer arithmetic wraps, as on the host.

struct Params {
    squares_x: u32,
    squares_y: u32,
    streams: u32,
    // Always 0; see `rounded`.
    zero: u32,
    // Stored stream index of each Bayer phase, indexed by row parity * 2 + column parity.
    phase_stream: array<u32, 4>,
    qmat: array<u32, 64>,
}

// One component stream of one tile.
struct Stream {
    // Byte offset of the stream in the packet.
    offset: u32,
    // Byte length of the stream.
    len: u32,
    // Index of the 16x16 square holding the tile's first block.
    square: u32,
    // log2 block count | qscale << 8 | stored stream index << 16.
    info: u32,
}

// Bindings the passes only read are declared read_write too, so that a
// caller may place every buffer in one (see `GpuPipelines::new`).
@group(0) @binding(0) var<storage, read_write> params: Params;
// The packet, four bytes per word in memory order, padded with zeros.
@group(0) @binding(1) var<storage, read_write> packet: array<u32>;
@group(0) @binding(2) var<storage, read_write> streams: array<Stream>;
// Per square: four streams of 64 coefficients, two 16-bit values per word,
// the even raster index in the low half. Cleared before the stream pass.
@group(0) @binding(3) var<storage, read_write> coefs: array<u32>;
// The same buffer as the stream pass sees it: each coefficient is ORed into
// its cleared word, which needs no read.
@group(0) @binding(3) var<storage, read_write> coef_bits: array<atomic<u32>>;
// Code values, two per word, the even column in the low half.
@group(0) @binding(4) var<storage, read_write> samples: array<u32>;
// One bit per square: set when its tile was skipped for an invalid header.
@group(0) @binding(5) var<storage, read_write> skipped: array<u32>;

// TABLES

// --- Pass 1: entropy decoding -------------------------------------------
//
// The bit reader and codeword decoder match src/bits.rs: a 64-bit window,
// here two words, whose top 32 bits are the current view, refilled one
// big-endian word at a time. Invalid (over-long) codewords wrap exactly as
// on the host.

var<private> win_hi: u32;
var<private> win_lo: u32;
// Valid bits below the top 32; negative after an over-long codeword.
var<private> extra: i32;
// Next byte to load, end of whole words, end of the stream (packet offsets).
var<private> bpos: u32;
var<private> words_end: u32;
var<private> bend: u32;

// Word `i` of the packet read as a big-endian 32-bit value.
fn be_word(i: u32) -> u32 {
    let w = packet[i];
    return (w >> 24u) | ((w >> 8u) & 0xff00u) | ((w << 8u) & 0xff0000u) | (w << 24u);
}

fn byte_at(i: u32) -> u32 {
    return (packet[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

// Up to four bytes from `p` as the top bytes of a big-endian word.
fn partial_word(p: u32, n: u32) -> u32 {
    var w = 0u;
    for (var i = 0u; i < min(n, 4u); i++) {
        w |= byte_at(p + i) << (24u - 8u * i);
    }
    return w;
}

// Set up the reader on `len` bytes at packet offset `offset`. The packet
// is taken as 4-byte aligned, as on the host.
fn reader_init(offset: u32, len: u32) {
    var p = offset;
    var remaining = len;
    if remaining < 4u {
        win_hi = partial_word(p, remaining);
        p += remaining;
        remaining = 0u;
    } else {
        win_hi = partial_word(p, 4u);
        p += 4u;
        remaining -= 4u;
    }
    win_lo = 0u;
    extra = 0;
    let to_boundary = (4u - (p & 3u)) & 3u;
    if to_boundary != 0u && to_boundary <= remaining {
        win_lo = partial_word(p, to_boundary);
        p += to_boundary;
        remaining -= to_boundary;
        extra = i32(8u * to_boundary);
    }
    bpos = p;
    words_end = p + 4u * (remaining / 4u);
    bend = offset + len;
}

// Next refill: a whole (aligned) word, else the zero-filled tail.
fn next_word() -> u32 {
    if bpos < words_end {
        let w = be_word(bpos >> 2u);
        bpos += 4u;
        return w;
    }
    let w = partial_word(bpos, bend - bpos);
    bpos = bend;
    return w;
}

// Drop `len` bits, refilling at most once.
fn consume(len: u32) {
    let s = len & 63u;
    if s >= 32u {
        win_hi = win_lo << (s - 32u);
        win_lo = 0u;
    } else if s != 0u {
        win_hi = (win_hi << s) | (win_lo >> (32u - s));
        win_lo = win_lo << s;
    }
    extra -= i32(len);
    if extra < 0 {
        // The refill position uses the low byte of the count, sign-extended.
        let shift = u32(-((extra << 24u) >> 24u)) & 63u;
        let w = next_word();
        if shift >= 32u {
            win_hi |= w << (shift - 32u);
        } else if shift == 0u {
            win_lo |= w;
        } else {
            win_hi |= w >> (32u - shift);
            win_lo |= w << shift;
        }
        extra += 32;
    }
}

// Codeword decoder on a non-zero view `p`: (value, length).
fn codeword(p: u32, cb: u32) -> vec2<u32> {
    let lz = countLeadingZeros(p);
    let s = cb >> 8u;
    let r = cb & 15u;
    if (p & 0x80000000u) != 0u {
        return vec2<u32>((p & 0x7fffffffu) >> ((31u - r) & 31u), r + 1u);
    }
    if lz <= s {
        let tail = ((p << ((lz + 1u) & 31u)) >> 1u) >> ((31u - r) & 31u);
        return vec2<u32>((lz << r) + tail, r + 1u + lz);
    }
    let e = (cb >> 4u) & 15u;
    let len = 2u * lz - s + e;
    return vec2<u32>(((s + 1u) << r) + (p >> ((32u - len) & 31u)) - (1u << e), len);
}

// A 16-bit coefficient (sign-extended) times the quantizer, saturated to
// 16 bits. Quantizers are at most 63 * 255, so the product fits.
fn dequantize(c: i32, q: u32) -> i32 {
    let c16 = (c << 16u) >> 16u;
    return clamp(c16 * i32(q), -32768, 32767);
}

// Read a level codeword and its sign bit: (symbol, signed 16-bit value).
fn level(cb: u32) -> vec2<i32> {
    let top = win_hi;
    let cw = codeword(top, cb);
    let negative = ((top << (cw.y & 31u)) >> 31u) != 0u;
    consume(cw.y + 1u);
    var magnitude = i32((cw.x + 1u) & 0xffffu);
    if negative {
        magnitude = -magnitude;
    }
    return vec2<i32>(bitcast<i32>(cw.x), magnitude);
}

// Set once a zero run wraps the scan position back over coefficients
// already stored (see `decode_streams`).
var<private> revisiting: bool;

// Store coefficient `raster` of the block whose words start at `block`.
// Until the scan position wraps, each coefficient is stored at most once,
// so ORing it into its cleared word is exact; after that a store must
// replace the half it lands on, as on the host. Only the invocation of
// the block's stream touches its words, so the load and store need no
// atomicity between them.
fn put(block: u32, raster: u32, v: i32) {
    let shift = (raster & 1u) * 16u;
    let word = block + (raster >> 1u);
    let bits = (u32(v) & 0xffffu) << shift;
    if revisiting {
        let kept = atomicLoad(&coef_bits[word]) & ~(0xffffu << shift);
        atomicStore(&coef_bits[word], kept | bits);
    } else {
        atomicOr(&coef_bits[word], bits);
    }
}

@compute @workgroup_size(64)
fn decode_streams(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.x + id.y * groups.x * 64u;
    if index >= params.streams {
        return;
    }
    let st = streams[index];
    let log2 = st.info & 0xffu;
    let qscale = (st.info >> 8u) & 0xffu;
    let blocks = 1u << log2;
    let codes = 64u * blocks;
    let block_mask = blocks - 1u;
    let base = st.square * 128u + (st.info >> 16u) * 32u;
    revisiting = false;
    reader_init(st.offset, st.len);

    // DC: a sign-folded first value, then deltas with a running sign.
    if win_hi == 0u {
        return;
    }
    let q0 = params.qmat[0] * qscale;
    let first = codeword(win_hi, FIRST_DC);
    consume(first.y);
    let odd = first.x & 1u;
    var dc = bitcast<i32>((((odd + first.x) >> 1u) ^ (0u - odd)) + odd);
    put(base, 0u, dequantize(dc, q0));
    var dc_cb = DC_RESET;
    var sign = 0u;
    for (var b = 1u; b < blocks; b++) {
        if win_hi == 0u {
            return;
        }
        let cw = codeword(win_hi, dc_cb);
        sign ^= cw.x & 1u;
        let magnitude = ((cw.x & 1u) + cw.x) >> 1u;
        let delta = bitcast<i32>(((0u - sign) ^ magnitude) + sign);
        dc_cb = DC_CB[min(magnitude, 12u)];
        sign = select(0u, 1u, delta < 0);
        consume(cw.y);
        dc += delta;
        put(base + b * 128u, 0u, dequantize(dc, q0));
    }

    var ac_cb = AC_INITIAL;
    var rn_cb = RN_INITIAL;
    var ln_cb = LN_INITIAL;
    var n = blocks;
    if n >= codes {
        return;
    }
    while win_hi != 0u {
        let run = codeword(win_hi, ln_cb);
        consume(run.y);
        for (var i = 0u; i < run.x; i++) {
            if win_hi == 0u {
                return;
            }
            let lv = level(ac_cb);
            ac_cb = AC_CB[min(u32(lv.x), 94u)];
            let raster = SCAN[n >> log2];
            put(base + (n & block_mask) * 128u, raster, dequantize(lv.y, params.qmat[raster] * qscale));
            n++;
            if n == codes {
                return;
            }
        }
        if win_hi == 0u {
            return;
        }
        let zeros = codeword(win_hi, rn_cb);
        rn_cb = RN_CB[min(zeros.x, 27u)];
        consume(zeros.y);
        // An over-long codeword can give a run near 2^32, which wraps the
        // position back to coefficients already stored.
        let before = n;
        n = n + 1u + zeros.x;
        if n < before {
            revisiting = true;
        }
        if n >= codes || win_hi == 0u {
            return;
        }
        let lv = level(ac_cb);
        ac_cb = AC_CB[min(u32(lv.x), 94u)];
        ln_cb = LN_CB[min(u32(lv.x), 14u)];
        let raster = SCAN[n >> log2];
        put(base + (n & block_mask) * 128u, raster, dequantize(lv.y, params.qmat[raster] * qscale));
        n++;
        if n >= codes {
            return;
        }
    }
}

// --- Pass 2: inverse transform ------------------------------------------
//
// The Apple-compatible transform (see src/idct.rs): an integer row pass with
// per-row prescaled tables, then an f32 column pass. The int-to-float
// conversion rounds to nearest-even in integer code, since WGSL leaves its
// rounding to the implementation; the float operations follow the host's
// order, and every product is forced to round on its own (see `rounded`)
// so that no multiply-add is fused. WGSL does not promise correctly
// rounded f32 arithmetic, so other devices may still differ by a code
// value in rare cases.

// The four blocks of a square, stream-major, raster order: row-pass output
// as f32 bits, then code values.
var<workgroup> blocks: array<u32, 256>;

// i32 to f32, rounding to nearest-even like the host's conversion.
fn to_f32(x: i32) -> f32 {
    if x == 0 {
        return 0.0;
    }
    let ux = bitcast<u32>(x);
    let a = select(ux, 0u - ux, x < 0);
    var e = 31u - countLeadingZeros(a);
    var mant: u32;
    if e <= 23u {
        mant = a << (23u - e);
    } else {
        let shift = e - 23u;
        mant = a >> shift;
        let rem = a & ((1u << shift) - 1u);
        let half = 1u << (shift - 1u);
        if rem > half || (rem == half && (mant & 1u) == 1u) {
            mant += 1u;
            if mant == (1u << 24u) {
                mant >>= 1u;
                e += 1u;
            }
        }
    }
    return bitcast<f32>((ux & 0x80000000u) | ((e + 127u) << 23u) | (mant & 0x7fffffu));
}

// Row `k` of a block: eight int32 outputs, no rounding or shift.
fn row_pass(x: array<i32, 8>, k: u32) -> array<i32, 8> {
    let t = ROW_SET[k] * 32u;
    var even: array<i32, 4>;
    var odd: array<i32, 4>;
    for (var j = 0u; j < 4u; j++) {
        even[j] = x[0] * ROW_TABLES[t + 2u * j] + x[2] * ROW_TABLES[t + 2u * j + 1u]
            + x[4] * ROW_TABLES[t + 8u + 2u * j] + x[6] * ROW_TABLES[t + 8u + 2u * j + 1u];
        odd[j] = x[1] * ROW_TABLES[t + 16u + 2u * j] + x[3] * ROW_TABLES[t + 16u + 2u * j + 1u]
            + x[5] * ROW_TABLES[t + 24u + 2u * j] + x[7] * ROW_TABLES[t + 24u + 2u * j + 1u];
    }
    return array<i32, 8>(
        even[0] + odd[0], even[1] + odd[1], even[2] + odd[2], even[3] + odd[3],
        even[3] - odd[3], even[2] - odd[2], even[1] - odd[1], even[0] - odd[0],
    );
}

// `x` unchanged, but opaque to the compiler: XOR with a zero it cannot
// see keeps a product from being fused with the following addition, which
// would skip the product's rounding.
fn rounded(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ params.zero);
}

// The f32 column pass; returns the code values of one column.
fn column_pass(r: array<f32, 8>) -> array<u32, 8> {
    let p = r[0] + BIAS;
    let ea = p + r[4];
    let eb = p - r[4];
    let f = rounded(r[6] * TAN2) + r[2];
    let g = rounded(r[2] * TAN2) - r[6];
    let e0 = ea + f;
    let e3 = ea - f;
    let e1 = eb + g;
    let e2 = eb - g;
    let a = rounded(r[7] * TAN1) + r[1];
    let b = rounded(r[1] * TAN1) - r[7];
    let c = rounded(r[5] * TAN3) + r[3];
    let d = r[5] - rounded(r[3] * TAN3);
    let o0 = a + c;
    let t = a - c;
    let o3 = b + d;
    let u = b - d;
    let o1 = rounded((t + u) * COS4);
    let o2 = rounded((t - u) * COS4);
    let y = array<f32, 8>(e0 + o0, e1 + o1, e2 + o2, e3 + o3, e3 - o3, e2 - o2, e1 - o1, e0 - o0);
    var out: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) {
        // Truncation toward zero, then saturation to 0..65535.
        out[i] = u32(clamp(i32(y[i]), 0, 65535));
    }
    return out;
}

// Workgroup (x, y) is the square at those coordinates. Invocation t
// transforms row and then column t % 8 of stream t / 8, then writes four
// words of the square's output, unless the square's tile was skipped.
@compute @workgroup_size(32)
fn transform(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let square = group.y * params.squares_x + group.x;
    let stream = t >> 3u;
    let k = t & 7u;
    let block = stream * 64u;

    var x: array<i32, 8>;
    let row_words = square * 128u + stream * 32u + k * 4u;
    for (var j = 0u; j < 4u; j++) {
        let w = coefs[row_words + j];
        x[2u * j] = bitcast<i32>(w << 16u) >> 16u;
        x[2u * j + 1u] = bitcast<i32>(w) >> 16u;
    }
    let row = row_pass(x, k);
    for (var j = 0u; j < 8u; j++) {
        blocks[block + k * 8u + j] = bitcast<u32>(to_f32(row[j]) * SCALE);
    }
    workgroupBarrier();

    var col: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        col[i] = bitcast<f32>(blocks[block + i * 8u + k]);
    }
    let values = column_pass(col);
    for (var i = 0u; i < 8u; i++) {
        blocks[block + i * 8u + k] = values[i];
    }
    workgroupBarrier();

    // Nothing is written for a skipped tile.
    if ((skipped[square >> 5u] >> (square & 31u)) & 1u) != 0u {
        return;
    }
    // Sensel (r, c) of the square is element (r / 2, c / 2) of the block
    // of the stream on phase (r % 2, c % 2). Word w of a row holds columns
    // 2w and 2w + 1.
    let r = t >> 1u;
    let even = params.phase_stream[(r & 1u) * 2u] * 64u + (r >> 1u) * 8u;
    let odd = params.phase_stream[(r & 1u) * 2u + 1u] * 64u + (r >> 1u) * 8u;
    let out_row = (group.y * 16u + r) * params.squares_x * 8u + group.x * 8u;
    for (var j = 0u; j < 4u; j++) {
        let w = (t & 1u) * 4u + j;
        samples[out_row + w] = blocks[even + w] | (blocks[odd + w] << 16u);
    }
}
