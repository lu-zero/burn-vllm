//! CubeCL kernels for the Qwen3.5 forward pass. Everything is f32 activations over
//! u32-word quantized weights so the same code runs on CUDA and wgpu (no f16/i8
//! storage requirements).

use burn_cubecl::cubecl::{self, prelude::*};

/// Maximum planes per cube the shared-memory reductions support.
pub const MAX_PLANES: usize = 32;
/// Shared-memory pair-tile layout for MoE descriptors: `[slot, count, p_0..p_{MAXM-1}]`.
pub const MOE_MAXM: usize = 8;
pub const MOE_DESC: usize = 2 + MOE_MAXM;

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

#[cube]
pub fn f16_to_f32(h: u32) -> f32 {
    let e = (h >> 10) & 0x1F;
    let m = h & 0x3FF;
    let sign = (h & 0x8000) << 16;
    let normal = f32::reinterpret(sign | ((e + 112) << 23) | (m << 13));
    let sub = f32::cast_from(m) * 5.960_464_5e-8;
    let sub = select(sign != 0, -sub, sub);
    select(e == 0, sub, normal)
}

#[cube]
fn sext8(byte: u32) -> f32 {
    f32::cast_from(i32::reinterpret(byte << 24) >> 24)
}

#[cube]
fn byte_at(w: &[u32], word_base: usize, byte: usize) -> u32 {
    (w[word_base + byte / 4] >> (((byte % 4) * 8) as u32)) & 0xFF
}

#[cube]
fn mxfp4_val(n: u32) -> f32 {
    let m = n & 7;
    let big = (4 + (m & 1) * 2) << ((m & 2) >> 1);
    let mag = f32::cast_from(select(m < 4, m, big));
    select((n & 8) != 0, -mag, mag)
}

#[cube]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + f32::exp(-x))
}

#[cube]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

// ---------------------------------------------------------------------------
// Block decoding: global 32-element block `b` -> 32 f32 values.
// ---------------------------------------------------------------------------

#[cube]
pub fn decode_block(w0: &[u32], w1: &[u32], b: usize, vals: &mut Array<f32>, #[comptime] fmt: u32) {
    if comptime!(fmt == 0) {
        // F32
        #[unroll]
        for j in 0..32usize {
            vals[j] = f32::reinterpret(w0[b * 32 + j]);
        }
    } else if comptime!(fmt == 1) {
        // Q8_0
        let d = f16_to_f32((w1[b / 2] >> (((b % 2) * 16) as u32)) & 0xFFFF);
        #[unroll]
        for i in 0..8usize {
            let w = w0[b * 8 + i];
            #[unroll]
            for k in 0..4usize {
                vals[i * 4 + k] = d * sext8((w >> comptime!((k * 8) as u32)) & 0xFF);
            }
        }
    } else if comptime!(fmt == 2) {
        // MXFP4
        let e = (w1[b / 4] >> (((b % 4) * 8) as u32)) & 0xFF;
        let d = select(e < 2, 0.0f32, f32::reinterpret((max(e, 1u32) - 1) << 23));
        #[unroll]
        for i in 0..4usize {
            let w = w0[b * 4 + i];
            #[unroll]
            for k in 0..4usize {
                let byte = (w >> comptime!((k * 8) as u32)) & 0xFF;
                vals[i * 4 + k] = d * mxfp4_val(byte & 0xF);
                vals[16 + i * 4 + k] = d * mxfp4_val(byte >> 4);
            }
        }
    } else {
        decode_kquant16(w0, b, 0, vals, 0usize, fmt);
        decode_kquant16(w0, b, 1, vals, 16usize, fmt);
    }
}

/// Decode values `[16*half, 16*half + 16)` of K-quant 32-block `b` into `vals[off..off+16]`.
#[cube]
pub fn decode_kquant16(
    w0: &[u32],
    b: usize,
    half: usize,
    vals: &mut Array<f32>,
    #[comptime] off: usize,
    #[comptime] fmt: u32,
) {
    if comptime!(fmt == 3) {
        // Q5_K: 44 words per superblock: d|dmin, scales[12], qh[32], qs[128].
        let base = (b / 8) * 44;
        let s = b % 8;
        let dd = w0[base];
        let d = f16_to_f32(dd & 0xFFFF);
        let dmin = f16_to_f32(dd >> 16);
        let sb = base + 1;
        let hi = s >= 4;
        let a = byte_at(w0, sb, s + 4);
        let lo_s = byte_at(w0, sb, s % 4);
        let sc = select(hi, (a & 0xF) | ((lo_s >> 6) << 4), byte_at(w0, sb, s) & 63);
        let mn = select(hi, (a >> 4) | ((byte_at(w0, sb, s) >> 6) << 4), a & 63);
        let d1 = d * f32::cast_from(sc);
        let m1 = dmin * f32::cast_from(mn);
        let nib_shift = ((s % 2) * 4) as u32;
        let qh_shift = s as u32;
        let ql_base = base + 12 + (s / 2) * 8 + half * 4;
        let qh_base = base + 4 + half * 4;
        #[unroll]
        for i in 0..4usize {
            let ql = w0[ql_base + i];
            let qh = w0[qh_base + i];
            #[unroll]
            for k in 0..4usize {
                let lo = (ql >> (comptime!((k * 8) as u32) + nib_shift)) & 0xF;
                let hi = ((qh >> (comptime!((k * 8) as u32) + qh_shift)) & 1) << 4;
                vals[off + i * 4 + k] = d1 * f32::cast_from(lo | hi) - m1;
            }
        }
    } else {
        // Q6_K: 53 words per (padded) superblock: ql[128], qh[64], scales[16], d.
        let base = (b / 8) * 53;
        let s = b % 8;
        let n = s / 4;
        let qq = s % 4;
        let d = f16_to_f32(w0[base + 52] & 0xFFFF);
        let ql_base = base + n * 16 + (qq % 2) * 8 + half * 4;
        let ql_shift = ((qq / 2) * 4) as u32;
        let qh_base = base + 32 + n * 8 + half * 4;
        let qh_shift = (qq * 2) as u32;
        let scale = d * sext8(byte_at(w0, base, 192 + n * 8 + qq * 2 + half));
        #[unroll]
        for i in 0..4usize {
            let ql = w0[ql_base + i];
            let qh = w0[qh_base + i];
            #[unroll]
            for k in 0..4usize {
                let lo = (ql >> (comptime!((k * 8) as u32) + ql_shift)) & 0xF;
                let hi = ((qh >> (comptime!((k * 8) as u32) + qh_shift)) & 3) << 4;
                let q = i32::cast_from(lo | hi) - 32;
                vals[off + i * 4 + k] = scale * f32::cast_from(q);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Cube-wide sum of `v`; result valid in every unit. Uses `smem[0..MAX_PLANES]`.
#[cube]
pub fn cube_sum(v: f32, smem: &mut Shared<[f32]>) -> f32 {
    let s = plane_sum(v);
    let np = (CUBE_DIM / PLANE_DIM) as usize;
    let plane = (UNIT_POS / PLANE_DIM) as usize;
    if UNIT_POS_PLANE == 0 {
        smem[plane] = s;
    }
    sync_cube();
    let mut tot = 0.0f32;
    for p in 0..np {
        tot += smem[p];
    }
    sync_cube();
    tot
}

#[cube]
pub fn cube_max(v: f32, smem: &mut Shared<[f32]>) -> f32 {
    let s = plane_max(v);
    let np = (CUBE_DIM / PLANE_DIM) as usize;
    let plane = (UNIT_POS / PLANE_DIM) as usize;
    if UNIT_POS_PLANE == 0 {
        smem[plane] = s;
    }
    sync_cube();
    let mut tot = smem[0];
    for p in 1..np {
        tot = max(tot, smem[p]);
    }
    sync_cube();
    tot
}

/// Linear cube index for grids folded over (x, y) to stay within wgpu's 65535 limit.
#[cube]
fn folded_cube() -> usize {
    (CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) as usize
}

pub type V4u = Vector<u32, Const<4>>;
pub type V4f = Vector<f32, Const<4>>;

/// Dot 16 decoded weights `vals[off..off+16]` with x vec4s `xv[xo..xo+4]`.
#[cube]
fn dot16(vals: &Array<f32>, #[comptime] off: usize, xv: &[V4f], xo: usize) -> f32 {
    let mut s = 0.0f32;
    #[unroll]
    for i in 0..4usize {
        let x = xv[xo + i];
        #[unroll]
        for k in 0..4usize {
            s += vals[off + i * 4 + k] * x.extract(k);
        }
    }
    s
}

/// Shared-memory table of the 16 E2M1 values (x2, matching ggml's kvalues_mxfp4), built
/// by the first 16 units. Must be reached by every unit of the cube.
#[cube]
fn mxfp4_lut() -> Shared<[f32]> {
    let mut lut = Shared::<[f32]>::new_slice(16usize);
    if UNIT_POS < 16 {
        lut[UNIT_POS as usize] = mxfp4_val(UNIT_POS);
    }
    sync_cube();
    lut
}

/// One plane computes lane-partial dot products of weight row `row` (global row index
/// into the packed tensor) against up to `maxm` activation rows. `xrow[m]` is the start of
/// activation row m in vec4 units; `acc[m]` receives this lane's partial sum.
///
/// Work units are one 128-bit weight load each (Q8_0: 16 weights, MXFP4: 32, F32: 4); the
/// K-quants decode 16 weights per unit from scalar words.
#[cube]
fn row_dot(
    w0: &[u32],
    w0v: &[V4u],
    w1: &[u32],
    lut: &Shared<[f32]>,
    xv: &[V4f],
    row: usize,
    k: usize,
    xrow: &Array<usize>,
    cnt: usize,
    acc: &mut Array<f32>,
    #[comptime] fmt: u32,
    #[comptime] maxm: usize,
) {
    let lane = UNIT_POS_PLANE as usize;
    let nl = PLANE_DIM as usize;
    let kb = k / 32;
    let mut vals = Array::<f32>::new(32usize);
    if comptime!(fmt == 0) {
        let units = k / 4;
        let base = row * units;
        let mut u = lane;
        while u < units {
            let w = w0v[base + u];
            #[unroll]
            for mi in 0..maxm {
                if mi < cnt {
                    let x = xv[xrow[mi] + u];
                    let mut s = 0.0f32;
                    #[unroll]
                    for j in 0..4usize {
                        s += f32::reinterpret(w.extract(j)) * x.extract(j);
                    }
                    acc[mi] += s;
                }
            }
            u += nl;
        }
    } else if comptime!(fmt == 1) {
        // Q8_0, one weight word (4 int8) per lane step: lanes read consecutive words and
        // consecutive activation vec4s, so both streams coalesce.
        let words = kb * 8;
        let base = row * words;
        let mut u0 = lane;
        while u0 < words {
            #[unroll]
            for r in 0..4usize {
                let u = u0 + r * nl;
                if u < words {
                    let wi = base + u;
                    let b = wi / 8;
                    let d = f16_to_f32((w1[b / 2] >> (((b % 2) * 16) as u32)) & 0xFFFF);
                    let w = w0[wi];
                    #[unroll]
                    for mi in 0..maxm {
                        if mi < cnt {
                            let x = xv[xrow[mi] + u];
                            let mut s = 0.0f32;
                            #[unroll]
                            for j in 0..4usize {
                                s += sext8((w >> comptime!((j * 8) as u32)) & 0xFF) * x.extract(j);
                            }
                            acc[mi] += d * s;
                        }
                    }
                }
            }
            u0 += 4 * nl;
        }
    } else if comptime!(fmt == 2) {
        // MXFP4, one word (8 nibbles: 4 low-half + 4 high-half values) per lane step.
        let words = kb * 4;
        let base = row * words;
        let mut u0 = lane;
        while u0 < words {
            #[unroll]
            for r in 0..4usize {
                let u = u0 + r * nl;
                if u < words {
                    let wi = base + u;
                    let b = wi / 4;
                    let e = (w1[b / 4] >> (((b % 4) * 8) as u32)) & 0xFF;
                    let d = select(e < 2, 0.0f32, f32::reinterpret((max(e, 1u32) - 1) << 23));
                    let w = w0[wi];
                    // vec4 offset of the low-half values within the row
                    let xo = (u / 4) * 8 + u % 4;
                    #[unroll]
                    for mi in 0..maxm {
                        if mi < cnt {
                            let xl = xv[xrow[mi] + xo];
                            let xh = xv[xrow[mi] + xo + 4];
                            let mut s = 0.0f32;
                            #[unroll]
                            for j in 0..4usize {
                                let byte = (w >> comptime!((j * 8) as u32)) & 0xFF;
                                s += lut[(byte & 0xF) as usize] * xl.extract(j)
                                    + lut[(byte >> 4) as usize] * xh.extract(j);
                            }
                            acc[mi] += d * s;
                        }
                    }
                }
            }
            u0 += 4 * nl;
        }
    } else {
        // K-quants: half a sub-block (16 values) per lane step, so the per-sub-block
        // scale decode is amortized over 16 weights.
        let units = kb * 2;
        let base = row * kb;
        let mut u = lane;
        while u < units {
            decode_kquant16(w0, base + u / 2, u % 2, &mut vals, 0usize, fmt);
            #[unroll]
            for mi in 0..maxm {
                if mi < cnt {
                    acc[mi] += dot16(&vals, 0usize, xv, xrow[mi] + u * 4);
                }
            }
            u += nl;
        }
    }
}

// ---------------------------------------------------------------------------
// Quantized matmul (small M): out[m, n] (+)= sum_k x[m, k] * W[n, k]
// One plane per output row; grid: (row groups folded over x/y, ceil(M / maxm) in z).
// ---------------------------------------------------------------------------

#[cube(launch)]
pub fn qmatmul(
    w0: &[u32],
    w0v: &[V4u],
    w1: &[u32],
    xv: &[V4f],
    out: &mut [f32],
    n: u32,
    k: u32,
    m: u32,
    #[comptime] fmt: u32,
    #[comptime] maxm: usize,
    #[comptime] accumulate: bool,
) {
    let lut = mxfp4_lut();
    let rows_per_cube = (CUBE_DIM / PLANE_DIM) as usize;
    let row = folded_cube() * rows_per_cube + (UNIT_POS / PLANE_DIM) as usize;
    let n = n as usize;
    let k = k as usize;
    let m0 = CUBE_POS_Z as usize * maxm;
    let cnt = min(maxm, m as usize - m0);
    let mut acc = Array::<f32>::new(maxm);
    let mut xrow = Array::<usize>::new(maxm);
    #[unroll]
    for mi in 0..maxm {
        acc[mi] = 0.0;
        xrow[mi] = (m0 + mi) * (k / 4);
    }
    if row < n {
        row_dot(
            w0, w0v, w1, &lut, xv, row, k, &xrow, cnt, &mut acc, fmt, maxm,
        );
    }
    #[unroll]
    for mi in 0..maxm {
        let s = plane_sum(acc[mi]);
        if row < n && UNIT_POS_PLANE == 0 && mi < cnt {
            let o = (m0 + mi) * n + row;
            if comptime!(accumulate) {
                out[o] += s;
            } else {
                out[o] = s;
            }
        }
    }
}

/// Dequantize a whole tensor to f32 (row-major [N, K]). One unit per 32-block.
#[cube(launch)]
pub fn dequant(w0: &[u32], w1: &[u32], out: &mut [f32], nblocks: u32, #[comptime] fmt: u32) {
    let b = (ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X) as usize;
    if b < nblocks as usize {
        let mut vals = Array::<f32>::new(32usize);
        decode_block(w0, w1, b, &mut vals, fmt);
        #[unroll]
        for j in 0..32usize {
            out[b * 32 + j] = vals[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Norms / elementwise
// ---------------------------------------------------------------------------

/// out[r, :] = x[r, :] * rsqrt(mean(x^2) + eps) * w   (one cube per row)
#[cube(launch)]
pub fn rmsnorm(x: &[f32], w: &[f32], out: &mut [f32], d: u32, eps: f32) {
    let mut smem = Shared::<[f32]>::new_slice(MAX_PLANES);
    let r = CUBE_POS_X as usize;
    let d = d as usize;
    let base = r * d;
    let mut ss = 0.0f32;
    let mut i = UNIT_POS as usize;
    while i < d {
        let v = x[base + i];
        ss += v * v;
        i += CUBE_DIM as usize;
    }
    let tot = cube_sum(ss, &mut smem);
    let inv = 1.0 / f32::sqrt(tot / f32::cast_from(d) + eps);
    let mut i = UNIT_POS as usize;
    while i < d {
        out[base + i] = x[base + i] * inv * w[i];
        i += CUBE_DIM as usize;
    }
}

#[cube(launch)]
pub fn add_inplace(x: &mut [f32], y: &[f32], n: u32) {
    let i = (ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X) as usize;
    if i < n as usize {
        x[i] += y[i];
    }
}

/// out[t, j] = silu(gu[t, j]) * gu[t, inter + j] over a fused [T, 2*inter] input.
#[cube(launch)]
pub fn swiglu(g: &[f32], u: &[f32], out: &mut [f32], n: u32) {
    let i = (ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X) as usize;
    if i < n as usize {
        out[i] = silu(g[i]) * u[i];
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (linear attention)
// ---------------------------------------------------------------------------

/// Causal depthwise conv1d (kernel 4) + SiLU with a rolling 3-step state.
/// x: [T, C], w: [C, 4], state: [3, C] (oldest first), out: [T, C]. One unit per channel.
#[cube(launch)]
pub fn gdn_conv(x: &[f32], w: &[f32], state: &mut [f32], out: &mut [f32], c: u32, t: u32) {
    let ch = ABSOLUTE_POS as usize;
    let c = c as usize;
    if ch < c {
        let w0 = w[ch * 4];
        let w1 = w[ch * 4 + 1];
        let w2 = w[ch * 4 + 2];
        let w3 = w[ch * 4 + 3];
        let mut h0 = state[ch];
        let mut h1 = state[c + ch];
        let mut h2 = state[2 * c + ch];
        for ti in 0..t as usize {
            let v = x[ti * c + ch];
            let y = w0 * h0 + w1 * h1 + w2 * h2 + w3 * v;
            out[ti * c + ch] = silu(y);
            h0 = h1;
            h1 = h2;
            h2 = v;
        }
        state[ch] = h0;
        state[c + ch] = h1;
        state[2 * c + ch] = h2;
    }
}

/// Gated delta rule recurrence + gated RMSNorm. One cube (DK units) per value head.
///
/// conv: [T, 2*NK*DK + NV*DV] (q | k | v after conv+silu), ab: [T, 2*NV] (alpha | beta
/// projections), z: [T, NV*DV], state: [NV, DK, DV], out: [T, NV*DV].
/// V-head `h` reads key head `h % NK` (GGUF tiled head order).
#[cube(launch)]
pub fn gdn_recurrent(
    conv: &[f32],
    alpha: &[f32],
    beta: &[f32],
    z: &[f32],
    ssm_a: &[f32],
    dt_bias: &[f32],
    norm_w: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    t: u32,
    nk: u32,
    nv: u32,
    eps: f32,
    #[comptime] dk: usize,
) {
    let mut smem = Shared::<[f32]>::new_slice(MAX_PLANES);
    let mut qs = Shared::<[f32]>::new_slice(dk);
    let mut ks = Shared::<[f32]>::new_slice(dk);
    let h = CUBE_POS_X as usize;
    let j = UNIT_POS as usize;
    let nk = nk as usize;
    let nv = nv as usize;
    let dv = dk;
    let kh = h % nk;
    let key_dim = nk * dk;
    let conv_dim = 2 * key_dim + nv * dv;
    let sbase = h * dk * dv;

    let mut s = Array::<f32>::new(dk);
    #[unroll]
    for i in 0..dk {
        s[i] = state[sbase + i * dv + j];
    }
    let a_h = ssm_a[h];
    let dt_h = dt_bias[h];
    let wj = norm_w[j];
    let qscale = 1.0 / f32::sqrt(f32::cast_from(dk));

    for ti in 0..t as usize {
        let row = ti * conv_dim;
        let qv = conv[row + kh * dk + j];
        let kv = conv[row + key_dim + kh * dk + j];
        let qn = cube_sum(qv * qv, &mut smem);
        let kn = cube_sum(kv * kv, &mut smem);
        qs[j] = qv / f32::sqrt(qn + 1e-6) * qscale;
        ks[j] = kv / f32::sqrt(kn + 1e-6);
        sync_cube();

        let av = alpha[ti * nv + h] + dt_h;
        let sp = select(av > 20.0, av, f32::ln(1.0 + f32::exp(av)));
        let g = f32::exp(a_h * sp);
        let bt = sigmoid(beta[ti * nv + h]);
        let vj = conv[row + 2 * key_dim + h * dv + j];

        let mut kvm = 0.0f32;
        #[unroll]
        for i in 0..dk {
            s[i] *= g;
            kvm += s[i] * ks[i];
        }
        let delta = (vj - kvm) * bt;
        let mut o = 0.0f32;
        #[unroll]
        for i in 0..dk {
            s[i] += ks[i] * delta;
            o += s[i] * qs[i];
        }
        let on = cube_sum(o * o, &mut smem);
        let zj = z[ti * nv * dv + h * dv + j];
        out[ti * nv * dv + h * dv + j] =
            o / f32::sqrt(on / f32::cast_from(dv) + eps) * wj * silu(zj);
    }
    #[unroll]
    for i in 0..dk {
        state[sbase + i * dv + j] = s[i];
    }
}

// ---------------------------------------------------------------------------
// Full attention (gated, GQA, partial NeoX RoPE)
// ---------------------------------------------------------------------------

/// Per-head RMSNorm + RoPE for q and k, scatter k/v into the cache.
/// qg: [T, NH, 2*HD] (q | gate per head), k/v: [T, NKV, HD],
/// q_out: [T, NH, HD], caches: [max_ctx, NKV, HD]. grid: (NH + NKV, T), cube: HD units.
#[cube(launch)]
pub fn attn_prep(
    qg: &[f32],
    k: &[f32],
    v: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    q_out: &mut [f32],
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    nh: u32,
    nkv: u32,
    pos0: u32,
    rope_dim: u32,
    theta: f32,
    eps: f32,
    #[comptime] hd: usize,
) {
    let mut smem = Shared::<[f32]>::new_slice(MAX_PLANES);
    let mut buf = Shared::<[f32]>::new_slice(hd);
    let head = CUBE_POS_X as usize;
    let t = CUBE_POS_Y as usize;
    let d = UNIT_POS as usize;
    let nh = nh as usize;
    let nkv = nkv as usize;
    let is_q = head < nh;
    let x = if is_q {
        qg[(t * nh + head) * 2 * hd + d]
    } else {
        k[(t * nkv + head - nh) * hd + d]
    };
    let wn = if is_q { q_norm[d] } else { k_norm[d] };
    let ss = cube_sum(x * x, &mut smem);
    let xn = x / f32::sqrt(ss / f32::cast_from(hd) + eps) * wn;
    buf[d] = xn;
    sync_cube();
    let rd = rope_dim as usize;
    let half = rd / 2;
    let pos = f32::cast_from(pos0 as usize + t);
    let mut y = xn;
    if d < rd {
        let i = d % half;
        let freq = f32::exp(-f32::ln(theta) * 2.0 * f32::cast_from(i) / f32::cast_from(rd));
        let ang = pos * freq;
        let c = f32::cos(ang);
        let s = f32::sin(ang);
        if d < half {
            y = xn * c - buf[d + half] * s;
        } else {
            y = xn * c + buf[d - half] * s;
        }
    }
    if is_q {
        q_out[(t * nh + head) * hd + d] = y;
    } else {
        let kvh = head - nh;
        let p = pos0 as usize + t;
        k_cache[(p * nkv + kvh) * hd + d] = y;
        v_cache[(p * nkv + kvh) * hd + d] = v[(t * nkv + kvh) * hd + d];
    }
}

/// Causal attention over the cache with online softmax, output gated by sigmoid(gate).
/// HD must be 256 (32 keys x 8 lanes per tile).
/// grid: (NH, T, splits). Each cube covers keys `[split * split_len, +split_len)` of its
/// token's causal range. With `partial`, it writes the unnormalized accumulator to
/// `out[(t, h, split), :]` and `(max, sum)` to `ml` for [`attention_combine`]; otherwise
/// (one split) it writes the gated, normalized output.
#[cube(launch)]
pub fn attention(
    q: &[f32],
    qg: &[f32],
    k_cache: &[V4f],
    v_cache: &[f32],
    out: &mut [f32],
    ml: &mut [f32],
    nh: u32,
    nkv: u32,
    pos0: u32,
    split_len: u32,
    scale: f32,
    #[comptime] hd: usize,
    #[comptime] partial: bool,
) {
    let mut qs = Shared::<[f32]>::new_slice(hd);
    let mut sc = Shared::<[f32]>::new_slice(32usize);
    let h = CUBE_POS_X as usize;
    let t = CUBE_POS_Y as usize;
    let split = CUBE_POS_Z as usize;
    let nsplit = CUBE_COUNT_Z as usize;
    let d = UNIT_POS as usize;
    let nh = nh as usize;
    let nkv = nkv as usize;
    let kvh = h / (nh / nkv);
    qs[d] = q[(t * nh + h) * hd + d] * scale;
    sync_cube();
    let nkeys = pos0 as usize + t + 1;
    let begin = split * split_len as usize;
    let end = min(nkeys, begin + split_len as usize);
    let kk = d / 8;
    let part = d % 8;
    let span = hd / 8;
    let mut m = f32::new(-1.0e30);
    let mut l = 0.0f32;
    let mut acc = 0.0f32;
    let mut s0 = begin;
    while s0 < end {
        let s = s0 + kk;
        let mut partial_dot = 0.0f32;
        if s < end {
            let kb = ((s * nkv + kvh) * hd + part * span) / 4;
            #[unroll]
            for i in 0..comptime!(hd / 32) {
                let kv = k_cache[kb + i];
                #[unroll]
                for j in 0..4usize {
                    partial_dot += qs[part * span + i * 4 + j] * kv.extract(j);
                }
            }
        }
        partial_dot += plane_shuffle_xor(partial_dot, 1);
        partial_dot += plane_shuffle_xor(partial_dot, 2);
        partial_dot += plane_shuffle_xor(partial_dot, 4);
        if part == 0 {
            sc[kk] = select(s < end, partial_dot, f32::new(-1.0e30));
        }
        sync_cube();
        let mut tmax = m;
        #[unroll]
        for i in 0..32usize {
            tmax = max(tmax, sc[i]);
        }
        let corr = f32::exp(m - tmax);
        l *= corr;
        acc *= corr;
        let cnt = min(32usize, end - s0);
        for i in 0..cnt {
            let pe = f32::exp(sc[i] - tmax);
            l += pe;
            acc += pe * v_cache[((s0 + i) * nkv + kvh) * hd + d];
        }
        m = tmax;
        sync_cube();
        s0 += 32;
    }
    if comptime!(partial) {
        let r = (t * nh + h) * nsplit + split;
        out[r * hd + d] = acc;
        if d == 0 {
            ml[r * 2] = m;
            ml[r * 2 + 1] = l;
        }
    } else {
        let gate = qg[(t * nh + h) * 2 * hd + hd + d];
        out[(t * nh + h) * hd + d] = acc / l * sigmoid(gate);
    }
}

/// Merge split-K attention partials and apply the output gate. grid: (NH, T), cube: HD.
#[cube(launch)]
pub fn attention_combine(
    parts: &[f32],
    ml: &[f32],
    qg: &[f32],
    out: &mut [f32],
    nh: u32,
    nsplit: u32,
    #[comptime] hd: usize,
) {
    let h = CUBE_POS_X as usize;
    let t = CUBE_POS_Y as usize;
    let d = UNIT_POS as usize;
    let nh = nh as usize;
    let ns = nsplit as usize;
    let r0 = (t * nh + h) * ns;
    let mut mx = f32::new(-1.0e30);
    for s in 0..ns {
        mx = max(mx, ml[(r0 + s) * 2]);
    }
    let mut l = 0.0f32;
    let mut acc = 0.0f32;
    for s in 0..ns {
        let w = f32::exp(ml[(r0 + s) * 2] - mx);
        l += ml[(r0 + s) * 2 + 1] * w;
        acc += parts[(r0 + s) * hd + d] * w;
    }
    let gate = qg[(t * nh + h) * 2 * hd + hd + d];
    out[(t * nh + h) * hd + d] = acc / l * sigmoid(gate);
}

// ---------------------------------------------------------------------------
// Mixture of experts
// ---------------------------------------------------------------------------

/// Top-k routing on softmax(logits): weights renormalized over the selected experts
/// (= softmax over the top-k logits). One unit per token.
/// logits: [T, stride] (first `ne` columns are the router), ids/wts: [T, topk].
#[cube(launch)]
pub fn moe_route(
    logits: &[f32],
    ids: &mut [u32],
    wts: &mut [f32],
    t: u32,
    ne: u32,
    stride: u32,
    #[comptime] topk: usize,
) {
    let tok = ABSOLUTE_POS as usize;
    if tok < t as usize {
        let base = tok * stride as usize;
        let ne = ne as usize;
        let mut sel = Array::<u32>::new(topk);
        let mut val = Array::<f32>::new(topk);
        #[unroll]
        for i in 0..topk {
            sel[i] = 0;
            val[i] = f32::new(-1.0e30);
        }
        for e in 0..ne {
            let v = logits[base + e];
            // insertion into a descending list
            if v > val[topk - 1] {
                let mut cur_v = v;
                let mut cur_i = e as u32;
                #[unroll]
                for i in 0..topk {
                    if cur_v > val[i] {
                        let tv = val[i];
                        let ti = sel[i];
                        val[i] = cur_v;
                        sel[i] = cur_i;
                        cur_v = tv;
                        cur_i = ti;
                    }
                }
            }
        }
        let mx = val[0];
        let mut sum = 0.0f32;
        #[unroll]
        for i in 0..topk {
            let e = f32::exp(val[i] - mx);
            val[i] = e;
            sum += e;
        }
        #[unroll]
        for i in 0..topk {
            ids[tok * topk + i] = sel[i];
            wts[tok * topk + i] = val[i] / sum;
        }
    }
}

/// Device-side descriptors for one (token, k) pair per tile with slot = expert id.
#[cube(launch)]
pub fn moe_direct_desc(ids: &[u32], desc: &mut [u32], npairs: u32) {
    let p = ABSOLUTE_POS as usize;
    if p < npairs as usize {
        desc[p * MOE_DESC] = ids[p];
        desc[p * MOE_DESC + 1] = 1;
        desc[p * MOE_DESC + 2] = p as u32;
    }
}

/// Pair rows of a descriptor tile as activation row offsets (vec4 units).
#[cube]
fn tile_rows(
    desc: &[u32],
    tile: usize,
    cnt: usize,
    div: u32,
    row_len: usize,
    xrow: &mut Array<usize>,
    #[comptime] maxm: usize,
) {
    #[unroll]
    for mi in 0..maxm {
        xrow[mi] = 0;
        if mi < cnt {
            xrow[mi] = (desc[tile * MOE_DESC + 2 + mi] / div) as usize * (row_len / 4);
        }
    }
}

/// Routed gate/up projections + SwiGLU, one plane per intermediate row. Tiles hold up
/// to `maxm` pairs (1 for decode). grid: (ceil(inter / rows_per_cube), tiles);
/// x row = pair / topk; act: [pairs, inter].
#[cube(launch)]
pub fn moe_gate_up(
    g0: &[u32],
    g0v: &[V4u],
    g1: &[u32],
    u0: &[u32],
    u0v: &[V4u],
    u1: &[u32],
    xv: &[V4f],
    desc: &[u32],
    act: &mut [f32],
    inter: u32,
    k: u32,
    topk: u32,
    #[comptime] fmt: u32,
    #[comptime] maxm: usize,
) {
    let lut = mxfp4_lut();
    let rows_per_cube = (CUBE_DIM / PLANE_DIM) as usize;
    let row = CUBE_POS_X as usize * rows_per_cube + (UNIT_POS / PLANE_DIM) as usize;
    let tile = CUBE_POS_Y as usize;
    let inter = inter as usize;
    let k = k as usize;
    let slot = desc[tile * MOE_DESC] as usize;
    let cnt = desc[tile * MOE_DESC + 1] as usize;
    let mut xrow = Array::<usize>::new(maxm);
    tile_rows(desc, tile, cnt, topk, k, &mut xrow, maxm);
    let mut ga = Array::<f32>::new(maxm);
    let mut ua = Array::<f32>::new(maxm);
    #[unroll]
    for mi in 0..maxm {
        ga[mi] = 0.0;
        ua[mi] = 0.0;
    }
    if row < inter {
        let wrow = slot * inter + row;
        row_dot(
            g0, g0v, g1, &lut, xv, wrow, k, &xrow, cnt, &mut ga, fmt, maxm,
        );
        row_dot(
            u0, u0v, u1, &lut, xv, wrow, k, &xrow, cnt, &mut ua, fmt, maxm,
        );
    }
    #[unroll]
    for mi in 0..maxm {
        let g = plane_sum(ga[mi]);
        let u = plane_sum(ua[mi]);
        if row < inter && UNIT_POS_PLANE == 0 && mi < cnt {
            let p = desc[tile * MOE_DESC + 2 + mi] as usize;
            act[p * inter + row] = silu(g) * u;
        }
    }
}

/// Routed down projection, one plane per hidden row.
/// grid: (ceil(hidden / rows_per_cube), tiles); y: [pairs, hidden].
#[cube(launch)]
pub fn moe_down(
    w0: &[u32],
    w0v: &[V4u],
    w1: &[u32],
    actv: &[V4f],
    desc: &[u32],
    y: &mut [f32],
    hidden: u32,
    inter: u32,
    #[comptime] fmt: u32,
    #[comptime] maxm: usize,
) {
    let lut = mxfp4_lut();
    let rows_per_cube = (CUBE_DIM / PLANE_DIM) as usize;
    let row = CUBE_POS_X as usize * rows_per_cube + (UNIT_POS / PLANE_DIM) as usize;
    let tile = CUBE_POS_Y as usize;
    let hidden = hidden as usize;
    let inter = inter as usize;
    let slot = desc[tile * MOE_DESC] as usize;
    let cnt = desc[tile * MOE_DESC + 1] as usize;
    let mut xrow = Array::<usize>::new(maxm);
    tile_rows(desc, tile, cnt, 1u32, inter, &mut xrow, maxm);
    let mut acc = Array::<f32>::new(maxm);
    #[unroll]
    for mi in 0..maxm {
        acc[mi] = 0.0;
    }
    if row < hidden {
        row_dot(
            w0,
            w0v,
            w1,
            &lut,
            actv,
            slot * hidden + row,
            inter,
            &xrow,
            cnt,
            &mut acc,
            fmt,
            maxm,
        );
    }
    #[unroll]
    for mi in 0..maxm {
        let s = plane_sum(acc[mi]);
        if row < hidden && UNIT_POS_PLANE == 0 && mi < cnt {
            let p = desc[tile * MOE_DESC + 2 + mi] as usize;
            y[p * hidden + row] = s;
        }
    }
}

/// x[t, d] += sum_k wts[t, k] * y[t*topk + k, d] + sigmoid(logits[t, gate_col]) * shared[t, d]
#[cube(launch)]
pub fn moe_combine(
    x: &mut [f32],
    y: &[f32],
    wts: &[f32],
    shared: &[f32],
    logits: &[f32],
    hidden: u32,
    total: u32,
    topk: u32,
    stride: u32,
    gate_col: u32,
) {
    let i = (ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X) as usize;
    if i < total as usize {
        let hidden = hidden as usize;
        let topk = topk as usize;
        let t = i / hidden;
        let d = i % hidden;
        let mut s = 0.0f32;
        for kk in 0..topk {
            let p = t * topk + kk;
            s += wts[p] * y[p * hidden + d];
        }
        let sg = sigmoid(logits[t * stride as usize + gate_col as usize]);
        x[i] += s + sg * shared[i];
    }
}
