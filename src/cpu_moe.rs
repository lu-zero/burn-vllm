//! Fast SIMD (AVX2+FMA+F16C) CPU quantized dot products and MoE expert execution.
//!
//! Computes missed experts directly from the mmap'd GGUF bytes (zero repack, zero PCIe
//! weight transfer) in parallel across CPU cores while the GPU executes the shared expert
//! and VRAM-resident hit experts.

use crate::quant::GgmlType;
use half::f16;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::LazyLock;

const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Precomputed E8M0 half-scale table: `E8M0_LUT[e] = 2^(e - 128)`.
const E8M0_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    // e < 2 maps to 0.0 (matching GPU kernel and avoiding x86 subnormal penalties).
    let mut e = 2usize;
    while e < 256 {
        t[e] = f32::from_bits(((e as u32) - 1) << 23);
        e += 1;
    }
    t
};

/// `SIGN_LUT[b]`: per-byte multipliers (0xFF = -1, 0x01 = +1) for the 8 sign bits of `b`,
/// applied with `_mm_sign_epi8`.
const SIGN_LUT: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut v = 0u64;
        let mut j = 0;
        while j < 8 {
            let byte = if (i >> j) & 1 == 1 { 0xFFu64 } else { 0x01u64 };
            v |= byte << (8 * j);
            j += 1;
        }
        t[i] = v;
        i += 1;
    }
    t
};
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Activation vector quantized per 32-element block to symmetric `i8` (`[-127, 127]`),
/// plus per-block scale `d` and exact f32 block sum `bsum` (for Q5_K's `-m1 * sum(x)`).
pub struct QuantX {
    pub qs: Vec<i8>,
    pub d: Vec<f32>,
    pub bsum: Vec<f32>,
}

impl QuantX {
    pub fn new() -> Self {
        Self {
            qs: Vec::new(),
            d: Vec::new(),
            bsum: Vec::new(),
        }
    }

    pub fn quantize_into(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len() % 32, 0);
        let nb = x.len() / 32;
        self.qs.resize(x.len(), 0);
        self.d.resize(nb, 0.0);
        self.bsum.resize(nb, 0.0);
        for (bi, xc) in x.chunks_exact(32).enumerate() {
            let mut amax = 0.0f32;
            let mut sum = 0.0f32;
            for &v in xc {
                let a = v.abs();
                if a > amax {
                    amax = a;
                }
                sum += v;
            }
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            self.d[bi] = scale;
            self.bsum[bi] = sum;
            let dst = &mut self.qs[bi * 32..(bi + 1) * 32];
            for (o, &v) in dst.iter_mut().zip(xc.iter()) {
                *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }

    pub fn from_f32(x: &[f32]) -> Self {
        let mut q = Self::new();
        q.quantize_into(x);
        q
    }
}

/// Fast quantized dot product using Q8-quantized activation `qx` (with f32 fallback).
#[inline]
pub fn dot_q8(ty: GgmlType, src: &[u8], qx: &QuantX, x_f32: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            // SAFETY: runtime check confirmed AVX2 + FMA + F16C.
            return unsafe {
                match ty {
                    GgmlType::Mxfp4 => avx2::dot_mxfp4_q8(src, qx),
                    GgmlType::Q5K => avx2::dot_q5k_q8(src, qx),
                    GgmlType::Q6K => avx2::dot_q6k_q8(src, qx),
                    GgmlType::Q4K => avx2::dot_q4k_q8(src, qx),
                    GgmlType::Q3K => avx2::dot_q3k_q8(src, qx),
                    GgmlType::Q8_0 => avx2::dot_q8_0_q8(src, qx),
                    GgmlType::Iq3S => avx2::dot_iq3s_q8(src, qx),
                    GgmlType::Iq3Xxs => avx2::dot_iq3xxs_q8(src, qx),
                    _ => dot_scalar(ty, src, x_f32),
                }
            };
        }
    }
    dot_scalar(ty, src, x_f32)
}

/// Exact f32 dot product (used as reference / fallback).
#[inline]
pub fn dot(ty: GgmlType, src: &[u8], x: &[f32]) -> f32 {
    let qx = QuantX::from_f32(x);
    dot_q8(ty, src, &qx, x)
}

fn dot_scalar(ty: GgmlType, src: &[u8], x: &[f32]) -> f32 {
    let (be, _bb) = ty.block();
    debug_assert_eq!(x.len() % be, 0);
    let mut sum = 0.0f32;
    match ty {
        GgmlType::Mxfp4 => {
            for (xc, b) in x.chunks_exact(32).zip(src.chunks_exact(17)) {
                let d = E8M0_LUT[b[0] as usize];
                let mut s = 0.0f32;
                for j in 0..16 {
                    s += MXFP4_VALUES[(b[1 + j] & 0xF) as usize] as f32 * xc[j]
                        + MXFP4_VALUES[(b[1 + j] >> 4) as usize] as f32 * xc[j + 16];
                }
                sum += d * s;
            }
        }
        GgmlType::Q5K => {
            for (xc, b) in x.chunks_exact(256).zip(src.chunks_exact(176)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                for s in 0..8 {
                    let (sc, m) = scale_min_k4(s, scales);
                    let (d1, m1) = (d * sc as f32, dmin * m as f32);
                    let ql = &qs[32 * (s / 2)..];
                    let xs = &xc[s * 32..(s + 1) * 32];
                    let mut sq = 0.0f32;
                    let mut sx = 0.0f32;
                    for l in 0..32 {
                        let lo = (ql[l] >> (4 * (s & 1))) & 0xF;
                        let hi = ((qh[l] >> s) & 1) << 4;
                        sq += (lo | hi) as f32 * xs[l];
                        sx += xs[l];
                    }
                    sum += d1 * sq - m1 * sx;
                }
            }
        }
        GgmlType::Q6K => {
            for (xc, b) in x.chunks_exact(256).zip(src.chunks_exact(210)) {
                let ql = &b[0..128];
                let qh = &b[128..192];
                let sc = &b[192..208];
                let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
                let mut sb = 0.0f32;
                for n in 0..2 {
                    let (ql, qh, sc, xn) =
                        (&ql[64 * n..], &qh[32 * n..], &sc[8 * n..], &xc[128 * n..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let qhv = qh[l];
                        let q1 = ((ql[l] & 0xF) | ((qhv & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qhv >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qhv >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qhv >> 6) & 3) << 4)) as i32 - 32;
                        sb += (sc[is] as i8) as f32 * (q1 as f32) * xn[l]
                            + (sc[is + 2] as i8) as f32 * (q2 as f32) * xn[l + 32]
                            + (sc[is + 4] as i8) as f32 * (q3 as f32) * xn[l + 64]
                            + (sc[is + 6] as i8) as f32 * (q4 as f32) * xn[l + 96];
                    }
                }
                sum += d * sb;
            }
        }
        GgmlType::Q8_0 => {
            for (xc, b) in x.chunks_exact(32).zip(src.chunks_exact(34)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let mut s = 0.0f32;
                for j in 0..32 {
                    s += (b[2 + j] as i8) as f32 * xc[j];
                }
                sum += d * s;
            }
        }
        GgmlType::F32 => {
            for (&xv, c) in x.iter().zip(src.chunks_exact(4)) {
                sum += f32::from_le_bytes(c.try_into().unwrap()) * xv;
            }
        }
        GgmlType::F16 => {
            for (&xv, c) in x.iter().zip(src.chunks_exact(2)) {
                sum += f16::from_le_bytes([c[0], c[1]]).to_f32() * xv;
            }
        }
        // Remaining formats (BF16, IQ*): dequantize the row, then plain f32 dot.
        GgmlType::Bf16 | GgmlType::Iq3S | GgmlType::Iq3Xxs | GgmlType::Iq2S => {
            let mut tmp = vec![0.0f32; x.len()];
            crate::quant::dequant(ty, src, &mut tmp);
            sum = tmp.iter().zip(x).map(|(a, b)| a * b).sum();
        }
        GgmlType::Q4K | GgmlType::Q3K => {
            let mut tmp = vec![0.0f32; x.len()];
            crate::quant::dequant(ty, src, &mut tmp);
            sum = tmp.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn f16_bits_to_f32(bits: u16) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
    }

    #[inline(always)]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let s = _mm_add_ps(s, shuf);
        let shuf = _mm_movehl_ps(shuf, s);
        let s = _mm_add_ss(s, shuf);
        _mm_cvtss_f32(s)
    }

    /// Signed 32 x `i8` dot product (where `a` is in `[-127, 127]` or `b` is in `[-127, 127]`):
    /// returns 8 `i32` partial sums.
    #[inline(always)]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn dot32_i8_i8(a: __m256i, b: __m256i, ones: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(a, a);
        let sy = _mm256_sign_epi8(b, a);
        let p16 = _mm256_maddubs_epi16(ax, sy);
        _mm256_madd_epi16(p16, ones)
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_mxfp4_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nb = qx.d.len();
            debug_assert!(src.len() >= nb * 17);
            let lut128 = _mm_loadu_si128(MXFP4_VALUES.as_ptr() as *const __m128i);
            let lut = _mm256_broadcastsi128_si256(lut128);
            let mask0f = _mm256_set1_epi8(0x0F);
            let ones = _mm256_set1_epi16(1);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut bp = src.as_ptr();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            let mut b = 0usize;
            while b + 2 <= nb {
                let d0 = _mm256_set1_ps(*E8M0_LUT.get_unchecked(*bp as usize) * *dp.add(b));
                let raw0 = _mm_loadu_si128(bp.add(1) as *const __m128i);
                let idx0 =
                    _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw0, 4), raw0), mask0f);
                let q0 = _mm256_shuffle_epi8(lut, idx0);
                let x0 = _mm256_loadu_si256(qp as *const __m256i);
                let p0 = _mm256_cvtepi32_ps(dot32_i8_i8(q0, x0, ones));
                acc0 = _mm256_fmadd_ps(d0, p0, acc0);

                let d1 =
                    _mm256_set1_ps(*E8M0_LUT.get_unchecked(*bp.add(17) as usize) * *dp.add(b + 1));
                let raw1 = _mm_loadu_si128(bp.add(18) as *const __m128i);
                let idx1 =
                    _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw1, 4), raw1), mask0f);
                let q1 = _mm256_shuffle_epi8(lut, idx1);
                let x1 = _mm256_loadu_si256(qp.add(32) as *const __m256i);
                let p1 = _mm256_cvtepi32_ps(dot32_i8_i8(q1, x1, ones));
                acc1 = _mm256_fmadd_ps(d1, p1, acc1);

                bp = bp.add(34);
                qp = qp.add(64);
                b += 2;
            }
            if b < nb {
                let d0 = _mm256_set1_ps(*E8M0_LUT.get_unchecked(*bp as usize) * *dp.add(b));
                let raw0 = _mm_loadu_si128(bp.add(1) as *const __m128i);
                let idx0 =
                    _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw0, 4), raw0), mask0f);
                let q0 = _mm256_shuffle_epi8(lut, idx0);
                let x0 = _mm256_loadu_si256(qp as *const __m256i);
                let p0 = _mm256_cvtepi32_ps(dot32_i8_i8(q0, x0, ones));
                acc0 = _mm256_fmadd_ps(d0, p0, acc0);
            }
            hsum256_ps(_mm256_add_ps(acc0, acc1))
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q5k_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 176);
            let mask0f = _mm256_set1_epi8(0x0F);
            let bit4 = _mm256_set1_epi8(0x10);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut min_acc = 0.0f32;
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            let bp_sum = qx.bsum.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 176);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
                let dmin = f16_bits_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
                let scales = std::slice::from_raw_parts(b.add(4), 12);
                let qh = _mm256_loadu_si256(b.add(16) as *const __m256i);
                let qs = b.add(48);
                let mut hm = _mm256_set1_epi8(1);
                let base_bi = sb * 8;
                for pair in 0..4usize {
                    let ql = _mm256_loadu_si256(qs.add(pair * 32) as *const __m256i);

                    // Even subblock
                    {
                        let s = pair * 2;
                        let bi = base_bi + s;
                        let (sc, m) = scale_min_k4(s, scales);
                        let h =
                            _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, hm), hm), bit4);
                        let q = _mm256_or_si256(_mm256_and_si256(ql, mask0f), h);
                        let x = _mm256_loadu_si256(qp as *const __m256i);
                        let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), ones);
                        let scale = _mm256_set1_ps(d * (sc as f32) * *dp.add(bi));
                        acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc);
                        min_acc += dmin * (m as f32) * *bp_sum.add(bi);
                        qp = qp.add(32);
                        hm = _mm256_add_epi8(hm, hm);
                    }
                    // Odd subblock
                    {
                        let s = pair * 2 + 1;
                        let bi = base_bi + s;
                        let (sc, m) = scale_min_k4(s, scales);
                        let h =
                            _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, hm), hm), bit4);
                        let q =
                            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql, 4), mask0f), h);
                        let x = _mm256_loadu_si256(qp as *const __m256i);
                        let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), ones);
                        let scale = _mm256_set1_ps(d * (sc as f32) * *dp.add(bi));
                        acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc);
                        min_acc += dmin * (m as f32) * *bp_sum.add(bi);
                        qp = qp.add(32);
                        hm = _mm256_add_epi8(hm, hm);
                    }
                }
            }
            hsum256_ps(acc) - min_acc
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q6k_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 210);
            let mask0f = _mm256_set1_epi8(0x0F);
            let mask03 = _mm256_set1_epi8(0x03);
            let bias32 = _mm256_set1_epi8(32);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 210);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b.add(208), *b.add(209)]));
                for n in 0..2usize {
                    let ql = b.add(64 * n);
                    let qh = b.add(128 + 32 * n);
                    let sc = b.add(192 + 8 * n) as *const i8;
                    let ql0 = _mm256_loadu_si256(ql as *const __m256i);
                    let ql1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
                    let qhv = _mm256_loadu_si256(qh as *const __m256i);

                    let q1 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(ql0, mask0f),
                            _mm256_slli_epi16(_mm256_and_si256(qhv, mask03), 4),
                        ),
                        bias32,
                    );
                    let q2 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(ql1, mask0f),
                            _mm256_slli_epi16(
                                _mm256_and_si256(_mm256_srli_epi16(qhv, 2), mask03),
                                4,
                            ),
                        ),
                        bias32,
                    );
                    let q3 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask0f),
                            _mm256_slli_epi16(
                                _mm256_and_si256(_mm256_srli_epi16(qhv, 4), mask03),
                                4,
                            ),
                        ),
                        bias32,
                    );
                    let q4 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask0f),
                            _mm256_slli_epi16(
                                _mm256_and_si256(_mm256_srli_epi16(qhv, 6), mask03),
                                4,
                            ),
                        ),
                        bias32,
                    );

                    // Each 32-element block has two 16-element sub-scales: lo 128-bit uses sc[2k],
                    // hi 128-bit uses sc[2k+1].
                    let base_bi = sb * 8 + n * 4;
                    let qs_arr = [q1, q2, q3, q4];
                    for (k, &qv) in qs_arr.iter().enumerate() {
                        let xv = _mm256_loadu_si256(qp.add(k * 32) as *const __m256i);
                        let p32 = dot32_i8_i8(qv, xv, ones);
                        let pf = _mm256_cvtepi32_ps(p32);
                        let dx = *dp.add(base_bi + k);
                        let slo = d * dx * (*sc.add(k * 2) as f32);
                        let shi = d * dx * (*sc.add(k * 2 + 1) as f32);
                        let sv = _mm256_set_m128(_mm_set1_ps(shi), _mm_set1_ps(slo));
                        acc = _mm256_fmadd_ps(sv, pf, acc);
                    }
                    qp = qp.add(128);
                }
            }
            hsum256_ps(acc)
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q8_0_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nb = qx.d.len();
            debug_assert!(src.len() >= nb * 34);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut bp = src.as_ptr();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for b in 0..nb {
                let dw = f16_bits_to_f32(u16::from_le_bytes([*bp, *bp.add(1)]));
                let q = _mm256_loadu_si256(bp.add(2) as *const __m256i);
                let x = _mm256_loadu_si256(qp as *const __m256i);
                // x is in [-127, 127], so abs(x) never overflows i8.
                let p32 = dot32_i8_i8(x, q, ones);
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(dw * *dp.add(b)),
                    _mm256_cvtepi32_ps(p32),
                    acc,
                );
                bp = bp.add(34);
                qp = qp.add(32);
            }
            hsum256_ps(acc)
        }
    }
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q4k_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 144);
            let mask0f = _mm256_set1_epi8(0x0F);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut min_acc = 0.0f32;
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            let bp_sum = qx.bsum.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 144);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
                let dmin = f16_bits_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
                let scales = std::slice::from_raw_parts(b.add(4), 12);
                let qs = b.add(16);
                let base_bi = sb * 8;
                for pair in 0..4usize {
                    let ql = _mm256_loadu_si256(qs.add(pair * 32) as *const __m256i);
                    for odd in 0..2usize {
                        let s = pair * 2 + odd;
                        let bi = base_bi + s;
                        let (sc, m) = scale_min_k4(s, scales);
                        let q = if odd == 0 {
                            _mm256_and_si256(ql, mask0f)
                        } else {
                            _mm256_and_si256(_mm256_srli_epi16(ql, 4), mask0f)
                        };
                        let x = _mm256_loadu_si256(qp as *const __m256i);
                        let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), ones);
                        let scale = _mm256_set1_ps(d * (sc as f32) * *dp.add(bi));
                        acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc);
                        min_acc += dmin * (m as f32) * *bp_sum.add(bi);
                        qp = qp.add(32);
                    }
                }
            }
            hsum256_ps(acc) - min_acc
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q3k_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 110);
            let mask03 = _mm256_set1_epi8(0x03);
            let four = _mm256_set1_epi8(4);
            let zero = _mm256_setzero_si256();
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 110);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b.add(108), *b.add(109)]));
                let sc = std::slice::from_raw_parts(b.add(96), 12);
                let hm = _mm256_loadu_si256(b as *const __m256i);
                let q0 = _mm256_loadu_si256(b.add(32) as *const __m256i);
                let q1 = _mm256_loadu_si256(b.add(64) as *const __m256i);
                for s in 0..8usize {
                    let (n, j) = (s / 4, s % 4);
                    let qn = if n == 0 { q0 } else { q1 };
                    let q2 = match j {
                        0 => _mm256_and_si256(qn, mask03),
                        1 => _mm256_and_si256(_mm256_srli_epi16(qn, 2), mask03),
                        2 => _mm256_and_si256(_mm256_srli_epi16(qn, 4), mask03),
                        _ => _mm256_and_si256(_mm256_srli_epi16(qn, 6), mask03),
                    };
                    let bit = _mm256_set1_epi8((1u32 << (n * 4 + j)) as i8);
                    let clear = _mm256_cmpeq_epi8(_mm256_and_si256(hm, bit), zero);
                    let q = _mm256_sub_epi8(q2, _mm256_and_si256(clear, four));
                    let x = _mm256_loadu_si256(qp as *const __m256i);
                    let p32 = dot32_i8_i8(q, x, ones);
                    let dx = d * *dp.add(sb * 8 + s);
                    let slo = dx * crate::quant::q3k_scale(sc, s * 2) as f32;
                    let shi = dx * crate::quant::q3k_scale(sc, s * 2 + 1) as f32;
                    let sv = _mm256_set_m128(_mm_set1_ps(shi), _mm_set1_ps(slo));
                    acc = _mm256_fmadd_ps(sv, _mm256_cvtepi32_ps(p32), acc);
                    qp = qp.add(32);
                }
            }
            hsum256_ps(acc)
        }
    }
    /// IQ3_S: 8 grid pairs per 32-block expanded to signed i8, then an int8 dot.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_iq3s_q8(src: &[u8], qx: &QuantX) -> f32 {
        use crate::iq_grids::IQ3S_GRID;
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 110);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 110);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
                for s in 0..8usize {
                    let qs = b.add(2 + 8 * s);
                    let qh = *b.add(66 + s) as usize;
                    let signs = b.add(74 + 4 * s);
                    let mut g = [0u64; 4];
                    let mut m = [0u64; 4];
                    for l in 0..4usize {
                        let i1 = *qs.add(2 * l) as usize | ((qh << (8 - 2 * l)) & 256);
                        let i2 = *qs.add(2 * l + 1) as usize | ((qh << (7 - 2 * l)) & 256);
                        g[l] = IQ3S_GRID[i1] as u64 | ((IQ3S_GRID[i2] as u64) << 32);
                        m[l] = SIGN_LUT[*signs.add(l) as usize];
                    }
                    let gv = _mm256_set_epi64x(g[3] as i64, g[2] as i64, g[1] as i64, g[0] as i64);
                    let mv = _mm256_set_epi64x(m[3] as i64, m[2] as i64, m[1] as i64, m[0] as i64);
                    let q = _mm256_sign_epi8(gv, mv);
                    let x = _mm256_loadu_si256(qp as *const __m256i);
                    let p32 = dot32_i8_i8(q, x, ones);
                    let sc = 1 + 2 * ((*b.add(106 + s / 2) >> (4 * (s % 2))) & 0xF) as u32;
                    let scale = _mm256_set1_ps(d * sc as f32 * *dp.add(sb * 8 + s));
                    acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc);
                    qp = qp.add(32);
                }
            }
            hsum256_ps(acc)
        }
    }

    /// IQ3_XXS: like IQ3_S but signs come from 7-bit indices into ksigns.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_iq3xxs_q8(src: &[u8], qx: &QuantX) -> f32 {
        use crate::iq_grids::IQ3XXS_GRID;
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 98);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 98);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
                for s in 0..8usize {
                    let qs = b.add(2 + 8 * s);
                    let aux = u32::from_le_bytes([
                        *b.add(66 + 4 * s),
                        *b.add(67 + 4 * s),
                        *b.add(68 + 4 * s),
                        *b.add(69 + 4 * s),
                    ]);
                    let mut g = [0u64; 4];
                    let mut m = [0u64; 4];
                    for l in 0..4usize {
                        g[l] = IQ3XXS_GRID[*qs.add(2 * l) as usize] as u64
                            | ((IQ3XXS_GRID[*qs.add(2 * l + 1) as usize] as u64) << 32);
                        let sg = crate::quant::ksign(((aux >> (7 * l)) & 127) as usize);
                        m[l] = SIGN_LUT[sg as usize];
                    }
                    let gv = _mm256_set_epi64x(g[3] as i64, g[2] as i64, g[1] as i64, g[0] as i64);
                    let mv = _mm256_set_epi64x(m[3] as i64, m[2] as i64, m[1] as i64, m[0] as i64);
                    let q = _mm256_sign_epi8(gv, mv);
                    let x = _mm256_loadu_si256(qp as *const __m256i);
                    let p32 = dot32_i8_i8(q, x, ones);
                    let scale =
                        _mm256_set1_ps(d * (0.5 + (aux >> 28) as f32) * 0.5 * *dp.add(sb * 8 + s));
                    acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc);
                    qp = qp.add(32);
                }
            }
            hsum256_ps(acc)
        }
    }
}

static MOE_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let n = std::env::var("BURN_CPU_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| {
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        })
        .unwrap_or_else(|| {
            let avail = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8);
            // On hybrid/SMT CPUs, using ~5/8 of logical threads (capped at 10)
            // maximizes DRAM bandwidth without hyperthread contention.
            (avail * 5 / 8).clamp(4, 10)
        })
        .max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("moe-cpu-{i}"))
        .build()
        .expect("build moe rayon pool")
});

/// Run `cpu` on the MoE pool while `main` runs on the calling thread; returns once both
/// finish. Lets the host keep enqueueing GPU work while missed experts are computed.
pub fn overlap<R>(cpu: impl FnOnce() + Send, main: impl FnOnce() -> R) -> R {
    MOE_POOL.in_place_scope(|s| {
        s.spawn(|_| cpu());
        main()
    })
}
/// One missed expert for single-token decode (`t == 1`), zero heap allocations.
#[derive(Clone, Copy)]
pub struct CpuDecodeJob<'a> {
    pub gate_raw: &'a [u8],
    pub up_raw: &'a [u8],
    pub down_raw: &'a [u8],
    pub weight: f32,
}

/// One missed expert to compute on CPU (general multi-token prefill).
pub struct CpuExpertJob<'a> {
    pub gate_raw: &'a [u8],
    pub up_raw: &'a [u8],
    pub down_raw: &'a [u8],
    /// `(token_idx, routing_weight)` pairs routed to this expert.
    pub pairs: Vec<(usize, f32)>,
}

/// Zero-overhead single-token decode CPU MoE using the spin-wait worker pool.
#[allow(clippy::too_many_arguments)]
pub fn run_experts_cpu_decode(
    gate_ty: GgmlType,
    up_ty: GgmlType,
    down_ty: GgmlType,
    jobs: &[CpuDecodeJob<'_>],
    x: &[f32],
    hidden: usize,
    inter: usize,
    out: &mut [f32],
) {
    let m = jobs.len();
    if m == 0 {
        return;
    }
    let gate_row_bytes = gate_ty.row_bytes(hidden);
    let up_row_bytes = up_ty.row_bytes(hidden);
    let down_row_bytes = down_ty.row_bytes(inter);
    struct DecodeScratch {
        qx0: QuantX,
        acts: Vec<f32>,
        qacts: Vec<QuantX>,
    }
    thread_local! {
        static SCRATCH: RefCell<DecodeScratch> = RefCell::new(DecodeScratch {
            qx0: QuantX::new(),
            acts: Vec::new(),
            qacts: Vec::new(),
        });
    }

    let x0 = &x[..hidden];
    SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.qx0.quantize_into(x0);
        sc.acts.resize(m * inter, 0.0);
        while sc.qacts.len() < m {
            sc.qacts.push(QuantX::new());
        }
        let DecodeScratch { qx0, acts, qacts } = &mut *sc;
        let qx0: &QuantX = qx0;
        let acts_slice = &mut acts[..m * inter];

        MOE_POOL.install(|| {
            // Pass 1: gate + up + silu across all missed experts in one parallel dispatch.
            acts_slice
                .par_chunks_mut(32)
                .enumerate()
                .for_each(|(ci, dst)| {
                    let start = ci * 32;
                    for (k, o) in dst.iter_mut().enumerate() {
                        let idx = start + k;
                        let mi = idx / inter;
                        let r = idx % inter;
                        let job = &jobs[mi];
                        let g = dot_q8(
                            gate_ty,
                            &job.gate_raw[r * gate_row_bytes..(r + 1) * gate_row_bytes],
                            qx0,
                            x0,
                        );
                        let u = dot_q8(
                            up_ty,
                            &job.up_raw[r * up_row_bytes..(r + 1) * up_row_bytes],
                            qx0,
                            x0,
                        );
                        *o = silu(g) * u;
                    }
                });

            for mi in 0..m {
                qacts[mi].quantize_into(&acts_slice[mi * inter..(mi + 1) * inter]);
            }
            let qacts_ref = &qacts[..m];

            // Pass 2: down projection + weighted sum into `out[0..hidden]`.
            out[..hidden]
                .par_chunks_mut(64)
                .enumerate()
                .for_each(|(ci, dst)| {
                    let d0 = ci * 64;
                    for (k, o) in dst.iter_mut().enumerate() {
                        let d = d0 + k;
                        let off = d * down_row_bytes;
                        let mut s = 0.0f32;
                        for (mi, job) in jobs.iter().enumerate() {
                            let act = &acts_slice[mi * inter..(mi + 1) * inter];
                            s += job.weight
                                * dot_q8(
                                    down_ty,
                                    &job.down_raw[off..off + down_row_bytes],
                                    &qacts_ref[mi],
                                    act,
                                );
                        }
                        *o = s;
                    }
                });
        });
    });
}

/// Run a set of missed experts on CPU and accumulate their weighted outputs into
/// `out` (`[t, hidden]`, zero-initialized by the caller).
#[allow(clippy::too_many_arguments)]
pub fn run_experts_cpu(
    gate_ty: GgmlType,
    up_ty: GgmlType,
    down_ty: GgmlType,
    jobs: &[CpuExpertJob<'_>],
    x: &[f32],
    t: usize,
    hidden: usize,
    inter: usize,
    out: &mut [f32],
) {
    if jobs.is_empty() {
        return;
    }
    if t == 1 {
        let djobs: Vec<CpuDecodeJob<'_>> = jobs
            .iter()
            .map(|j| CpuDecodeJob {
                gate_raw: j.gate_raw,
                up_raw: j.up_raw,
                down_raw: j.down_raw,
                weight: j.pairs[0].1,
            })
            .collect();
        run_experts_cpu_decode(gate_ty, up_ty, down_ty, &djobs, x, hidden, inter, out);
        return;
    }
    let gate_row_bytes = gate_ty.row_bytes(hidden);
    let up_row_bytes = up_ty.row_bytes(hidden);
    let down_row_bytes = down_ty.row_bytes(inter);

    // General multi-token path (prefill chunks, speculative verify windows): quantize each
    // token's activation once, then split the work by (expert, row) so every thread streams
    // an expert row once for all tokens routed to it.
    let qxs: Vec<QuantX> = (0..t)
        .map(|tok| QuantX::from_f32(&x[tok * hidden..(tok + 1) * hidden]))
        .collect();
    let mut off = Vec::with_capacity(jobs.len() + 1);
    off.push(0usize);
    for j in jobs {
        off.push(off.last().unwrap() + j.pairs.len() * inter);
    }
    let mut acts = vec![0.0f32; *off.last().unwrap()];
    let acts_ptr = acts.as_mut_ptr() as usize;
    MOE_POOL.install(|| {
        // Pass 1: gate + up + silu for every (expert, row, pair).
        (0..jobs.len() * inter)
            .into_par_iter()
            .with_min_len(16)
            .for_each(|idx| {
                let (ji, r) = (idx / inter, idx % inter);
                let job = &jobs[ji];
                let g_row = &job.gate_raw[r * gate_row_bytes..(r + 1) * gate_row_bytes];
                let u_row = &job.up_raw[r * up_row_bytes..(r + 1) * up_row_bytes];
                let dst = acts_ptr as *mut f32;
                for (pi, &(tok, _)) in job.pairs.iter().enumerate() {
                    let xt = &x[tok * hidden..(tok + 1) * hidden];
                    let g = dot_q8(gate_ty, g_row, &qxs[tok], xt);
                    let u = dot_q8(up_ty, u_row, &qxs[tok], xt);
                    // SAFETY: each (job, pair, row) slot is written by exactly one task.
                    unsafe { *dst.add(off[ji] + pi * inter + r) = silu(g) * u };
                }
            });
        let qacts: Vec<Vec<QuantX>> = jobs
            .iter()
            .enumerate()
            .map(|(ji, job)| {
                (0..job.pairs.len())
                    .map(|pi| {
                        let s = off[ji] + pi * inter;
                        QuantX::from_f32(&acts[s..s + inter])
                    })
                    .collect()
            })
            .collect();
        // Pass 2: down projection, accumulated per output column so tasks never share one.
        let out_ptr = out.as_mut_ptr() as usize;
        (0..hidden).into_par_iter().with_min_len(16).for_each(|d| {
            let dst = out_ptr as *mut f32;
            for (ji, job) in jobs.iter().enumerate() {
                let d_row = &job.down_raw[d * down_row_bytes..(d + 1) * down_row_bytes];
                for (pi, &(tok, wt)) in job.pairs.iter().enumerate() {
                    let s = off[ji] + pi * inter;
                    let y = wt * dot_q8(down_ty, d_row, &qacts[ji][pi], &acts[s..s + inter]);
                    // SAFETY: column `d` of every token row belongs to this task alone.
                    unsafe { *dst.add(tok * hidden + d) += y };
                }
            }
        });
    });
}
