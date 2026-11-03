// ProRes RAW decoding on the GPU (bitstream description §6 and §7).
//
// Pass 1, `decode_streams`: one invocation per tile component stream reads
// its codewords and stores the dequantized, 16-bit-saturated coefficients
// of its blocks. Pass 2, `transform`: one workgroup per 16x16 sensel square
// inverse-transforms the four blocks of the square and writes its code
// values. Integer arithmetic wraps, as on the host.

struct Params {
    squares_x: u32,
    squares_y: u32,
    streams: u32,
    _pad: u32,
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

@group(0) @binding(0) var<storage, read> params: Params;
// The packet, four bytes per word in memory order, padded with zeros.
@group(0) @binding(1) var<storage, read> packet: array<u32>;
@group(0) @binding(2) var<storage, read> streams: array<Stream>;
// Per square: four streams of 64 coefficients, two 16-bit values per word,
// the even raster index in the low half.
@group(0) @binding(3) var<storage, read_write> coefs: array<u32>;
// Code values, two per word, the even column in the low half.
@group(0) @binding(4) var<storage, read_write> samples: array<u32>;

// TABLES

// --- Pass 1: entropy decoding -------------------------------------------

var<private> start_bit: u32;
var<private> pos: u32;
var<private> limit: u32;

// Word `i` of the packet read as a big-endian 32-bit value.
fn be_word(i: u32) -> u32 {
    let w = packet[i];
    return (w >> 24u) | ((w >> 8u) & 0xff00u) | ((w << 8u) & 0xff0000u) | (w << 24u);
}

// The next 32 bits of the stream, zero-filled past its end.
fn window() -> u32 {
    if pos >= limit {
        return 0u;
    }
    let bit = start_bit + pos;
    let i = bit >> 5u;
    let shift = bit & 31u;
    var w = be_word(i) << shift;
    if shift != 0u {
        w |= be_word(i + 1u) >> (32u - shift);
    }
    let left = limit - pos;
    if left < 32u {
        w &= 0xffffffffu << (32u - left);
    }
    return w;
}

fn read_bit() -> u32 {
    let b = window() >> 31u;
    pos += 1u;
    return b;
}

// One symbol of the hybrid code with parameters `cb`, or -1 at the end of
// the component.
fn symbol(cb: u32) -> i32 {
    let w = window();
    if w == 0u {
        return -1;
    }
    let q = countLeadingZeros(w);
    let s = cb >> 8u;
    let e = (cb >> 4u) & 15u;
    let r = cb & 15u;
    if q <= s {
        var tail = 0u;
        if r != 0u {
            tail = (w << (q + 1u)) >> (32u - r);
        }
        pos += q + 1u + r;
        return i32((q << r) | tail);
    }
    let len = 2u * q + e - s;
    if len > 32u {
        return -1;
    }
    pos += len;
    return i32((w >> (32u - len)) + ((s + 1u) << r) - (1u << e));
}

// The exact product saturated to 16 bits. Coefficients stay below 2^24
// and quantizers below 2^16, so saturating the coefficient first keeps the
// 32-bit product in range.
fn dequantize(c: i32, q: u32) -> i32 {
    if q == 0u {
        return 0;
    }
    if c > 32767 {
        return 32767;
    }
    if c < -32768 {
        return -32768;
    }
    return clamp(c * i32(q), -32768, 32767);
}

// Store coefficient `raster` of the block whose words start at `block`.
fn put(block: u32, raster: u32, v: i32) {
    let i = block + (raster >> 1u);
    let shift = (raster & 1u) * 16u;
    coefs[i] = (coefs[i] & ~(0xffffu << shift)) | ((u32(v) & 0xffffu) << shift);
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
    start_bit = st.offset * 8u;
    pos = 0u;
    limit = st.len * 8u;

    for (var b = 0u; b < blocks; b++) {
        for (var k = 0u; k < 32u; k++) {
            coefs[base + b * 128u + k] = 0u;
        }
    }

    let first = symbol(FIRST_DC);
    if first < 0 {
        return;
    }
    let q0 = params.qmat[0] * qscale;
    var dc = (first >> 1u) ^ -(first & 1);
    put(base, 0u, dequantize(dc, q0));
    var prev = u32(first);
    var sign = 0;
    for (var b = 1u; b < blocks; b++) {
        var cb = DC_RESET;
        if b % 16u != 1u {
            cb = DC_CB[min((prev + 1u) >> 1u, 12u)];
        }
        let v = symbol(cb);
        if v < 0 {
            return;
        }
        prev = u32(v);
        sign ^= v & 1;
        let magnitude = (v + 1) >> 1u;
        var delta = magnitude;
        if sign != 0 {
            delta = -magnitude;
        }
        sign = select(0, 1, delta < 0);
        dc += delta;
        put(base + b * 128u, 0u, dequantize(dc, q0));
    }

    var ac_cb = AC_INITIAL;
    var rn_cb = RN_INITIAL;
    var ln_cb = LN_INITIAL;
    var n = blocks;
    loop {
        if n >= codes {
            return;
        }
        let run = symbol(ln_cb);
        if run < 0 {
            return;
        }
        for (var i = 0; i < run; i++) {
            let level = symbol(ac_cb);
            if level < 0 {
                return;
            }
            ac_cb = AC_CB[min(u32(level), 94u)];
            let raster = SCAN[n >> log2];
            var c = level + 1;
            if read_bit() != 0u {
                c = -c;
            }
            put(base + (n & block_mask) * 128u, raster, dequantize(c, params.qmat[raster] * qscale));
            n++;
            if n == codes {
                return;
            }
        }
        let zeros = symbol(rn_cb);
        if zeros < 0 {
            return;
        }
        rn_cb = RN_CB[min(u32(zeros), 27u)];
        // A zero run is zeros + 1 long; symbols are below 2^24, so this
        // cannot overflow.
        n += u32(zeros) + 1u;
        if n >= codes {
            return;
        }
        let level = symbol(ac_cb);
        if level < 0 {
            return;
        }
        ac_cb = AC_CB[min(u32(level), 94u)];
        ln_cb = LN_CB[min(u32(level), 14u)];
        let raster = SCAN[n >> log2];
        var c = level + 1;
        if read_bit() != 0u {
            c = -c;
        }
        put(base + (n & block_mask) * 128u, raster, dequantize(c, params.qmat[raster] * qscale));
        n++;
    }
}

// --- Pass 2: inverse transform ------------------------------------------

const W1 = 22725;
const W2 = 21407;
const W3 = 19265;
const W4 = 16384;
const W5 = 12873;
const W6 = 8867;
const W7 = 4520;

// The four blocks of a square, stream-major, raster order.
var<workgroup> blocks: array<i32, 256>;

// One 8-point pass of the integer transform on `x`, with `round` folded
// into the DC term and outputs shifted right by `shift`.
fn idct8(x: array<i32, 8>, round: i32, shift: u32) -> array<i32, 8> {
    let base = W4 * x[0] + round;
    let a0 = base + W2 * x[2] + W4 * x[4] + W6 * x[6];
    let a1 = base + W6 * x[2] - W4 * x[4] - W2 * x[6];
    let a2 = base - W6 * x[2] - W4 * x[4] + W2 * x[6];
    let a3 = base - W2 * x[2] + W4 * x[4] - W6 * x[6];
    let b0 = W1 * x[1] + W3 * x[3] + W5 * x[5] + W7 * x[7];
    let b1 = W3 * x[1] - W7 * x[3] - W1 * x[5] - W5 * x[7];
    let b2 = W5 * x[1] - W1 * x[3] + W7 * x[5] + W3 * x[7];
    let b3 = W7 * x[1] - W5 * x[3] + W3 * x[5] - W1 * x[7];
    return array<i32, 8>(
        (a0 + b0) >> shift,
        (a1 + b1) >> shift,
        (a2 + b2) >> shift,
        (a3 + b3) >> shift,
        (a3 - b3) >> shift,
        (a2 - b2) >> shift,
        (a1 - b1) >> shift,
        (a0 - b0) >> shift,
    );
}

fn code_value(centered: i32) -> u32 {
    return u32(clamp(centered + 32768, 0, 65535));
}

// Workgroup (x, y) is the square at those coordinates. Invocation t
// transforms row and then column t % 8 of stream t / 8, then writes four
// words of the square's output.
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
    let row = idct8(x, 1 << 12u, 13u);
    for (var j = 0u; j < 8u; j++) {
        blocks[block + k * 8u + j] = row[j];
    }
    workgroupBarrier();

    for (var i = 0u; i < 8u; i++) {
        x[i] = blocks[block + i * 8u + k];
    }
    // The column rounding term (1 << 14) / W4 = 1 is folded into x0.
    x[0] += 1;
    let col = idct8(x, 0, 15u);
    for (var i = 0u; i < 8u; i++) {
        blocks[block + i * 8u + k] = col[i];
    }
    workgroupBarrier();

    // Sensel (r, c) of the square is element (r / 2, c / 2) of the block
    // of the stream on phase (r % 2, c % 2). Word w of a row holds columns
    // 2w and 2w + 1.
    let r = t >> 1u;
    let even = params.phase_stream[(r & 1u) * 2u] * 64u + (r >> 1u) * 8u;
    let odd = params.phase_stream[(r & 1u) * 2u + 1u] * 64u + (r >> 1u) * 8u;
    let out_row = (group.y * 16u + r) * params.squares_x * 8u + group.x * 8u;
    for (var j = 0u; j < 4u; j++) {
        let w = (t & 1u) * 4u + j;
        samples[out_row + w] = code_value(blocks[even + w]) | (code_value(blocks[odd + w]) << 16u);
    }
}
