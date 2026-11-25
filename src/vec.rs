//! Backend-neutral vector operations.
//!
//! One implementation is selected at startup (AVX2 on x86_64, NEON DotProd on aarch64,
//! portable scalar otherwise) and published as a `dyn VecBackend`. Call sites no longer
//! need `#[cfg]`, `match GgmlType`, or ad-hoc feature booleans; adding a format or an
//! architecture means adding one impl here.
//!
//! Each backend intentionally reproduces the reduction order of the code it replaces on
//! that architecture. Logits are byte-identical to the previous implementation on the
//! same host; cross-architecture tiny differences remain expected.

use crate::cpu_moe::QuantX;
use crate::quant::GgmlType;
use std::sync::LazyLock;

pub trait VecBackend: Send + Sync {
    fn name(&self) -> &'static str;

    /// Quantized dot of a GGUF weight row against a per-32-block Q8 activation.
    fn dot_q8(&self, ty: GgmlType, src: &[u8], qx: &QuantX, x: &[f32]) -> f32;

    /// Per-32-block symmetric i8 quantization: scales, exact block sums and codes.
    fn quantize_into(&self, x: &[f32], qs: &mut [i8], d: &mut [f32], bsum: &mut [f32]);
}

struct Scalar;

impl VecBackend for Scalar {
    fn name(&self) -> &'static str {
        "scalar"
    }

    fn dot_q8(&self, ty: GgmlType, src: &[u8], _qx: &QuantX, x: &[f32]) -> f32 {
        crate::cpu_moe::dot_scalar(ty, src, x)
    }

    fn quantize_into(&self, x: &[f32], qs: &mut [i8], d: &mut [f32], bsum: &mut [f32]) {
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
            d[bi] = scale;
            bsum[bi] = sum;
            for (o, &v) in qs[bi * 32..(bi + 1) * 32].iter_mut().zip(xc.iter()) {
                *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
struct Avx2;

#[cfg(target_arch = "x86_64")]
impl VecBackend for Avx2 {
    fn name(&self) -> &'static str {
        "avx2"
    }

    fn dot_q8(&self, ty: GgmlType, src: &[u8], qx: &QuantX, x: &[f32]) -> f32 {
        // SAFETY: this impl is only selected after avx2_available() returned true.
        unsafe { crate::cpu_moe::avx2_dot_q8(ty, src, qx, x) }
    }

    fn quantize_into(&self, x: &[f32], qs: &mut [i8], d: &mut [f32], bsum: &mut [f32]) {
        // SAFETY: selected only on an AVX2+FMA+F16C host.
        unsafe { crate::cpu_moe::avx2_quantize_into(x, qs, d, bsum) }
    }
}

#[cfg(target_arch = "aarch64")]
struct Neon;

#[cfg(target_arch = "aarch64")]
impl VecBackend for Neon {
    fn name(&self) -> &'static str {
        "neon"
    }

    fn dot_q8(&self, ty: GgmlType, src: &[u8], qx: &QuantX, x: &[f32]) -> f32 {
        crate::cpu_moe::neon_dot_q8(ty, src, qx).unwrap_or_else(|| Self::fallback(ty, src, qx, x))
    }

    fn quantize_into(&self, x: &[f32], qs: &mut [i8], d: &mut [f32], bsum: &mut [f32]) {
        Scalar.quantize_into(x, qs, d, bsum)
    }
}

#[cfg(target_arch = "aarch64")]
impl Neon {
    fn fallback(ty: GgmlType, src: &[u8], _qx: &QuantX, x: &[f32]) -> f32 {
        crate::cpu_moe::dot_scalar(ty, src, x)
    }
}

static BACKEND: LazyLock<&'static dyn VecBackend> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu_moe::avx2_available() {
        return with_name(&Avx2);
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu_moe::neon_available() {
        return with_name(&Neon);
    }
    with_name(&Scalar)
});

fn with_name(backend: &'static dyn VecBackend) -> &'static dyn VecBackend {
    eprintln!("vec backend: {}", backend.name());
    backend
}

#[inline]
pub fn backend() -> &'static dyn VecBackend {
    *BACKEND
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_is_selected() {
        assert!(!backend().name().is_empty());
    }

    #[test]
    fn scalar_quantization_is_deterministic() {
        let x: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.125).collect();
        let mut a = QuantX::new();
        let mut b = QuantX::new();
        a.qs.resize(128, 0);
        a.d.resize(4, 0.0);
        a.bsum.resize(4, 0.0);
        b.qs.resize(128, 0);
        b.d.resize(4, 0.0);
        b.bsum.resize(4, 0.0);
        Scalar.quantize_into(&x, &mut a.qs, &mut a.d, &mut a.bsum);
        Scalar.quantize_into(&x, &mut b.qs, &mut b.d, &mut b.bsum);
        assert_eq!(a.qs, b.qs);
        assert_eq!(a.d, b.d);
        assert_eq!(a.bsum, b.bsum);
    }
}
