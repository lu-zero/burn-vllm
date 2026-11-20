//! GGML quant types: host-side dequantization (reference + embedding lookup) and
//! repacking into the 4-byte-aligned layouts the GPU kernels read.
//!
//! GPU layouts (all indexed by the global 32-element block index `b`):
//! - `F32`:   w0 = raw f32 bits.
//! - `Q8_0`:  w0 = int8 quants, 8 words per block; w1 = f16 scales, 2 per word.
//! - `MXFP4`: w0 = packed nibbles, 4 words per block; w1 = E8M0 exponents, 4 per word.
//! - `Q5_K`:  w0 = raw 176-byte superblocks (44 words), w1 unused.
//! - `Q6_K`:  w0 = raw 210-byte superblocks padded to 212 bytes (53 words), w1 unused.

use anyhow::{Result, bail};
use half::f16;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    Q5K,
    Q6K,
    Mxfp4,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            13 => Self::Q5K,
            14 => Self::Q6K,
            39 => Self::Mxfp4,
            _ => bail!("unsupported ggml type {v}"),
        })
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Mxfp4 => (32, 17),
        }
    }

    pub fn row_bytes(self, n: usize) -> usize {
        let (be, bb) = self.block();
        assert!(n % be == 0, "{self:?}: {n} not a multiple of {be}");
        n / be * bb
    }

    /// The comptime format id the kernels dispatch on.
    pub fn gpu_fmt(self) -> u32 {
        match self {
            Self::F32 | Self::F16 => fmt::F32,
            Self::Q8_0 => fmt::Q8_0,
            Self::Mxfp4 => fmt::MXFP4,
            Self::Q5K => fmt::Q5K,
            Self::Q6K => fmt::Q6K,
        }
    }
}

/// Kernel format ids.
pub mod fmt {
    pub const F32: u32 = 0;
    pub const Q8_0: u32 = 1;
    pub const MXFP4: u32 = 2;
    pub const Q5K: u32 = 3;
    pub const Q6K: u32 = 4;

    /// u32 words of w0 per 32-element block (K-quants: per 256-element superblock).
    pub fn w0_words(f: u32, numel: usize) -> usize {
        match f {
            F32 => numel,
            Q8_0 => numel / 32 * 8,
            MXFP4 => numel / 32 * 4,
            Q5K => numel / 256 * 44,
            Q6K => numel / 256 * 53,
            _ => unreachable!(),
        }
    }

    pub fn w1_words(f: u32, numel: usize) -> usize {
        match f {
            Q8_0 => (numel / 32).div_ceil(2),
            MXFP4 => (numel / 32).div_ceil(4),
            _ => 1,
        }
    }
}

const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn e8m0_half(e: u8) -> f32 {
    // 2^(e - 128), matching ggml's GGML_E8M0_TO_FP32_HALF.
    if e < 2 {
        (e as f32 - 128.0).exp2()
    } else {
        f32::from_bits(((e as u32) - 1) << 23)
    }
}

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

/// Dequantize `out.len()` elements from `src` (which starts on a block boundary).
pub fn dequant(ty: GgmlType, src: &[u8], out: &mut [f32]) {
    let (be, bb) = ty.block();
    assert_eq!(out.len() % be, 0);
    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16::from_le_bytes(c.try_into().unwrap()).to_f32();
            }
        }
        GgmlType::Q8_0 => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                for j in 0..32 {
                    y[j] = d * (b[2 + j] as i8) as f32;
                }
            }
        }
        GgmlType::Mxfp4 => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let d = e8m0_half(b[0]);
                for j in 0..16 {
                    y[j] = d * MXFP4_VALUES[(b[1 + j] & 0xF) as usize] as f32;
                    y[j + 16] = d * MXFP4_VALUES[(b[1 + j] >> 4) as usize] as f32;
                }
            }
        }
        GgmlType::Q5K => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                for s in 0..8 {
                    let (sc, m) = scale_min_k4(s, scales);
                    let (d1, m1) = (d * sc as f32, dmin * m as f32);
                    let ql = &qs[32 * (s / 2)..];
                    for l in 0..32 {
                        let lo = (ql[l] >> (4 * (s & 1))) & 0xF;
                        let hi = ((qh[l] >> s) & 1) << 4;
                        y[s * 32 + l] = d1 * (lo | hi) as f32 - m1;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let ql = &b[0..128];
                let qh = &b[128..192];
                let sc = &b[192..208];
                let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
                for n in 0..2 {
                    for l in 0..32 {
                        let is = l / 16;
                        let (ql, qh, sc) = (&ql[64 * n..], qh[32 * n + l], &sc[8 * n..]);
                        let q1 = ((ql[l] & 0xF) | ((qh & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh >> 6) & 3) << 4)) as i32 - 32;
                        let y = &mut y[128 * n..];
                        y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
    }
}

/// A tensor repacked into the GPU layout described in the module docs.
pub struct Packed {
    pub fmt: u32,
    pub w0: Vec<u32>,
    pub w1: Vec<u32>,
}

const PAR_CHUNK_BLOCKS: usize = 1 << 14;

/// Repack raw ggml bytes (a whole number of blocks) into the GPU layout.
pub fn repack(ty: GgmlType, src: &[u8]) -> Packed {
    let (be, bb) = ty.block();
    let nblocks = src.len() / bb;
    let numel = nblocks * be;
    let f = ty.gpu_fmt();
    let mut w0 = vec![0u32; fmt::w0_words(f, numel)];
    let mut w1 = vec![0u32; fmt::w1_words(f, numel)];
    match ty {
        GgmlType::F32 | GgmlType::Q5K => copy_words(&mut w0, src),
        GgmlType::F16 => {
            for (o, c) in w0.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16::from_le_bytes([c[0], c[1]]).to_f32().to_bits();
            }
        }
        GgmlType::Q8_0 => {
            let w0b: &mut [u8] = bytemuck::cast_slice_mut(&mut w0);
            w0b.par_chunks_mut(32 * PAR_CHUNK_BLOCKS)
                .zip(src.par_chunks(34 * PAR_CHUNK_BLOCKS))
                .for_each(|(dst, s)| {
                    for (d, b) in dst.chunks_exact_mut(32).zip(s.chunks_exact(34)) {
                        d.copy_from_slice(&b[2..34]);
                    }
                });
            let w1h: &mut [u16] = bytemuck::cast_slice_mut(&mut w1);
            for (i, b) in src.chunks_exact(34).enumerate() {
                w1h[i] = u16::from_le_bytes([b[0], b[1]]);
            }
        }
        GgmlType::Mxfp4 => {
            let w0b: &mut [u8] = bytemuck::cast_slice_mut(&mut w0);
            w0b.par_chunks_mut(16 * PAR_CHUNK_BLOCKS)
                .zip(src.par_chunks(17 * PAR_CHUNK_BLOCKS))
                .for_each(|(dst, s)| {
                    for (d, b) in dst.chunks_exact_mut(16).zip(s.chunks_exact(17)) {
                        d.copy_from_slice(&b[1..17]);
                    }
                });
            let w1b: &mut [u8] = bytemuck::cast_slice_mut(&mut w1);
            for (i, b) in src.chunks_exact(17).enumerate() {
                w1b[i] = b[0];
            }
        }
        GgmlType::Q6K => {
            let w0b: &mut [u8] = bytemuck::cast_slice_mut(&mut w0);
            for (d, b) in w0b.chunks_exact_mut(212).zip(src.chunks_exact(210)) {
                d[..210].copy_from_slice(b);
            }
        }
    }
    Packed { fmt: f, w0, w1 }
}

fn copy_words(dst: &mut [u32], src: &[u8]) {
    dst.par_chunks_mut(1 << 16)
        .zip(src.par_chunks(4 << 16))
        .for_each(|(d, s)| {
            for (o, c) in d.iter_mut().zip(s.chunks_exact(4)) {
                *o = u32::from_le_bytes(c.try_into().unwrap());
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0() {
        assert_eq!(e8m0_half(128), 1.0);
        assert_eq!(e8m0_half(127), 0.5);
        assert_eq!(e8m0_half(130), 4.0);
    }
}
