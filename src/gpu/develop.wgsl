// Apple-compatible raw conversion on the GPU, pass for pass the host's
// (src/demosaic.rs): the same operations in the same order, with fused
// multiply-adds exactly where the host fuses them.
//
// Pass 1, `green`: one invocation per sensel normalizes its 5x5 window of
// code values through the folded linearization table and writes the red or
// blue value (0 at green sites) and the green estimate to two planes.
// Pass 2, `chroma`: one invocation per pixel reads the 3x3 window of those
// planes and writes white-balanced camera RGB, clipped at 1.0, to three
// planes, through the output matrix when there is one.

struct Params {
    width: u32,
    height: u32,
    // Distance between rows of the code values, in sensels.
    stride: u32,
    // Header Bayer code: the index 2 * (y & 1) + (x & 1) of the red sensel.
    cfa: u32,
    // Black over range as the green pass computes it (black * 1 / range).
    green_off: f32,
    // Black over range as the chroma pass computes it (black / range).
    chroma_off: f32,
    clip_red: f32,
    clip_blue: f32,
    // Noise model (a, b, c, unused).
    noise: vec4<f32>,
    wb_red: f32,
    wb_blue: f32,
    // 1 when `matrix` applies.
    has_matrix: u32,
    // Always 0; see `rounded`.
    zero: u32,
    // Rows of the output matrix (fourth lanes unused).
    matrix: array<vec4<f32>, 3>,
}

// A uniform, for the constant cache. The storage bindings are all
// read_write, even those only read, so that a caller may place them all in
// one buffer (see `GpuPipelines::new`).
@group(0) @binding(0) var<uniform> params: Params;
// Code value to normalized linear value.
@group(0) @binding(1) var<storage, read_write> lut: array<f32>;
// Code values, two per word, the even column in the low half.
@group(0) @binding(2) var<storage, read_write> samples: array<u32>;
// Green pass output: red or blue (0 at green sites), and green.
@group(0) @binding(3) var<storage, read_write> rb: array<f32>;
@group(0) @binding(4) var<storage, read_write> gr: array<f32>;
// Output planes.
@group(0) @binding(5) var<storage, read_write> out_r: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_g: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_b: array<f32>;

// `x` unchanged, but opaque to the compiler: XOR with a zero it cannot
// see keeps a product from being fused with the following addition, which
// would skip the product's rounding.
fn rounded(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ params.zero);
}

// a / b, correctly rounded like the host's division, which GPU division
// is not required to be: the reciprocal is refined by one Newton step and
// the quotient corrected by its exact residual, both with fused
// multiply-adds. Divisors here are positive and normal.
fn quot(a: f32, b: f32) -> f32 {
    let r0 = 1.0 / b;
    let r = fma(fma(-b, r0, 1.0), r0, r0);
    let q0 = a * r;
    return fma(fma(-q0, b, a), r, q0);
}

// sqrt(x), correctly rounded like the host's, by correcting the device's
// square root with its exact residual. Arguments here are positive.
fn root(x: f32) -> f32 {
    let h = 0.5 * inverseSqrt(x);
    let s0 = sqrt(x);
    return fma(fma(-s0, s0, x), h, s0);
}

fn quot4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(quot(a.x, b.x), quot(a.y, b.y), quot(a.z, b.z), quot(a.w, b.w));
}

fn root4(x: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(root(x.x), root(x.y), root(x.z), root(x.w));
}

// Noise variance of a normalized value.
fn variance(v: f32, off: f32) -> f32 {
    let t = off + v;
    return params.noise.z + max(params.noise.y, params.noise.x * (t * t));
}

fn variance4(v: vec4<f32>, off: f32) -> vec4<f32> {
    return vec4<f32>(variance(v.x, off), variance(v.y, off), variance(v.z, off), variance(v.w, off));
}

// Four-term dot product, accumulated with fused multiply-adds in the
// host's order.
fn dot4(a: vec4<f32>, b: vec4<f32>) -> f32 {
    return fma(a.w, b.w, fma(a.z, b.z, fma(a.x, b.x, a.y * b.y)));
}

// The host's dot product with ones, w + (z + (x + y)). Spelled as `dot4`
// with a constant, a compiler may fold the multiplies away and reassociate
// the additions, so the partial sums are kept opaque.
fn sum4(v: vec4<f32>) -> f32 {
    return v.w + rounded(v.z + rounded(v.x + v.y));
}

// --- Pass 1: green --------------------------------------------------------

// Normalized value at (x, y), clamped to the picture.
fn norm(x: i32, y: i32) -> f32 {
    let cx = u32(clamp(x, 0, i32(params.width) - 1));
    let cy = u32(clamp(y, 0, i32(params.height) - 1));
    let i = cy * params.stride + cx;
    let code = (samples[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;
    return lut[code];
}

// Border handling: neighbours past the edge mirror the ones on the other
// side; a same-colour neighbour two away from a pixel next to the edge
// takes the centre value.
fn mirror(d: vec4<f32>, pos: u32, len: u32, c: f32) -> vec4<f32> {
    if pos == 0u {
        return vec4<f32>(d.y, d.y, d.w, d.w);
    } else if pos == 1u {
        return vec4<f32>(d.x, d.y, c, d.w);
    } else if pos == len - 1u {
        return vec4<f32>(d.x, d.x, d.z, d.z);
    } else if pos == len - 2u {
        return vec4<f32>(d.x, d.y, d.z, c);
    }
    return d;
}

@compute @workgroup_size(16, 16)
fn green(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let w = params.width;
    let h = params.height;
    if x >= w || y >= h {
        return;
    }
    let o = y * w + x;
    let xi = i32(x);
    let yi = i32(y);
    let red_row = params.cfa >> 1u;
    let red_col = params.cfa & 1u;
    let phase = (y & 1u) == red_row;
    let first = select(1u - red_col, red_col, phase);
    let centre = norm(xi, yi);
    if (x & 1u) != first {
        // Green sites keep their value.
        rb[o] = 0.0;
        gr[o] = centre;
        return;
    }
    let clip = select(params.clip_blue, params.clip_red, phase);
    let c = min(centre, clip);
    let limit = vec4<f32>(1.0, 1.0, clip, clip);
    var hh = min(vec4<f32>(norm(xi - 1, yi), norm(xi + 1, yi), norm(xi - 2, yi), norm(xi + 2, yi)), limit);
    var vv = min(vec4<f32>(norm(xi, yi - 1), norm(xi, yi + 1), norm(xi, yi - 2), norm(xi, yi + 2)), limit);
    hh = mirror(hh, x, w, c);
    vv = mirror(vv, y, h, c);
    let off = params.green_off;
    let var_h = variance4(hh, off);
    let var_v = variance4(vv, off);
    let sums = vec4<f32>(hh.x + hh.y, vv.x + vv.y, hh.z + hh.w, vv.z + vv.w);
    var ratio = vec2<f32>(0.0, 0.0);
    if c > root(variance(c, off)) {
        // c * 2 is exact, so fusing would not change these.
        let t = vec4<f32>(c * 2.0 + sums.z, c * 2.0 + sums.w, c * -2.0 + sums.z, c * -2.0 + sums.w);
        for (var dir = 0u; dir < 2u; dir++) {
            let d = select(vv, hh, dir == 0u);
            let sd = root4(select(var_v, var_h, dir == 0u));
            let a = d.x - d.y;
            let b = d.z - d.w;
            let any_ = rounded(a * b) > 0.0
                || rounded((d.z - c) * (d.w - c)) > 0.0
                || sd.x + sd.y > abs(a)
                || sd.z + sd.w > abs(b);
            if any_ && all(d > sd) {
                ratio[dir] = quot(t[2u + dir], t[dir] * 2.0);
            }
        }
    }
    let gh = rounded(sums.x * 0.5) * (1.0 - ratio.x);
    let gv = rounded(sums.y * 0.5) * (1.0 - ratio.y);
    let dh = vec4<f32>(hh.x - gh, hh.y - gh, hh.z - c, hh.w - c);
    let dv = vec4<f32>(vv.x - gv, vv.y - gv, vv.z - c, vv.w - c);
    // Opaque inner sums: the host adds them first, and a compiler may
    // otherwise reassociate the three-term sum.
    let eh = dot4(dh, dh) + rounded(var_h.x + var_h.y);
    let ev = dot4(dv, dv) + rounded(var_v.x + var_v.y);
    rb[o] = c;
    gr[o] = gv + quot(rounded((gh - gv) * ev), eh + ev);
}

// --- Pass 2: chroma (and matrix) ------------------------------------------

// The rows or columns before and after `p`, mirrored at the edges.
fn mirrored(p: u32, len: u32) -> vec2<u32> {
    return vec2<u32>(select(p + 1u, p - 1u, p > 0u), select(p - 1u, p + 1u, p < len - 1u));
}

// Chroma estimate from four neighbours `xx` of the wanted colour with
// greens `gg`, around a centre green `cg` with noise deviation `cs`.
fn estimate(xx: vec4<f32>, gg: vec4<f32>, cg: f32, cs: f32) -> f32 {
    let quarter = vec4<f32>(0.25);
    let mean_x = dot4(xx, quarter);
    if !(cs < cg) {
        return mean_x;
    }
    let off = params.chroma_off;
    let var_x = variance4(xx, off);
    let sx = root4(var_x);
    let sg = root4(variance4(gg, off));
    if !(all(sx < xx) && all(sg < gg)) {
        return mean_x;
    }
    let lo = quot4((cg - cs) * (xx - sx), sg + gg);
    let mid = quot4(cg * xx, gg);
    let hi = quot4((cs + cg) * (sx + xx), gg - sg);
    let spread = dot4(mid - lo, hi - mid);
    let mean_mid = dot4(mid, quarter);
    let dm = mid - mean_mid;
    let b = fma(spread, 0.75, dot4(dm, dm));
    let dx = xx - mean_x;
    let a = fma(sum4(var_x), 0.75, dot4(dx, dx));
    let weight = quot(a, b + a);
    return fma(mean_mid - mean_x, weight, mean_x);
}

// Red or blue and green of the green pass at (x, y).
fn at(x: u32, y: u32) -> vec2<f32> {
    let i = y * params.width + x;
    return vec2<f32>(rb[i], gr[i]);
}

// Estimate from two opposite neighbours, each counted twice.
fn pair(u: vec2<f32>, v: vec2<f32>, cg: f32, cs: f32) -> f32 {
    return estimate(vec4<f32>(u.x, v.x, u.x, v.x), vec4<f32>(u.y, v.y, u.y, v.y), cg, cs);
}

@compute @workgroup_size(16, 16)
fn chroma(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let w = params.width;
    let h = params.height;
    if x >= w || y >= h {
        return;
    }
    let mx = mirrored(x, w);
    let my = mirrored(y, h);
    let here = at(x, y);
    let c = here.x;
    let g = here.y;
    let cs = root(variance(g, params.chroma_off));
    let on_red_row = (y & 1u) == (params.cfa >> 1u);
    let red_col = params.cfa & 1u;
    let rb_col = select(1u - red_col, red_col, on_red_row);
    var r: f32;
    var b: f32;
    if (x & 1u) == rb_col {
        let n0 = at(mx.x, my.x);
        let n1 = at(mx.x, my.y);
        let n2 = at(mx.y, my.x);
        let n3 = at(mx.y, my.y);
        let e = estimate(vec4<f32>(n0.x, n1.x, n2.x, n3.x), vec4<f32>(n0.y, n1.y, n2.y, n3.y), g, cs);
        r = select(e, c, on_red_row);
        b = select(c, e, on_red_row);
    } else {
        let eh = pair(at(mx.x, y), at(mx.y, y), g, cs);
        let ev = pair(at(x, my.x), at(x, my.y), g, cs);
        r = select(ev, eh, on_red_row);
        b = select(eh, ev, on_red_row);
    }
    var rgb = vec3<f32>(min(r * params.wb_red, 1.0), min(g, 1.0), min(b * params.wb_blue, 1.0));
    if params.has_matrix != 0u {
        let m = params.matrix;
        rgb = vec3<f32>(
            fma(rgb.z, m[0].z, fma(rgb.x, m[0].x, rgb.y * m[0].y)),
            fma(rgb.z, m[1].z, fma(rgb.x, m[1].x, rgb.y * m[1].y)),
            fma(rgb.z, m[2].z, fma(rgb.x, m[2].x, rgb.y * m[2].y)),
        );
    }
    let o = y * w + x;
    out_r[o] = rgb.x;
    out_g[o] = rgb.y;
    out_b[o] = rgb.z;
}
