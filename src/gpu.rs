//! Device context, raw 4-byte-element buffers and kernel launch wrappers.

use crate::kernels::{self, MOE_DESC, MOE_MAXM};
use burn_backend::{DType, Shape};
use burn_cubecl::{
    CubeDevice,
    cubecl::{bytes::Bytes, client::Client, prelude::*, server::Handle},
    kernel::{
        into_contiguous,
        matmul::{MatmulStrategy, matmul},
    },
    tensor::CubeTensor,
};
use burn_std::{Metadata, strides};

const WG: u32 = 256;
const MAX_GRID: u32 = 65535;

/// A device buffer of `len` 4-byte elements (f32 or u32).
#[derive(Clone)]
pub struct Buf {
    pub h: Handle,
    pub len: usize,
}

impl Buf {
    fn arg(&self) -> BufferArg {
        unsafe { BufferArg::from_raw_parts(self.h.clone(), self.len) }
    }

    /// The same memory viewed as 128-bit vectors (`len / 4` of them).
    fn argv(&self) -> BufferArg {
        unsafe { BufferArg::from_raw_parts(self.h.clone(), (self.len / 4).max(1)) }
    }
}

/// Quantized (or f32) weight matrix [n, k] in GPU layout.
pub struct QWeight {
    pub fmt: u32,
    pub n: usize,
    pub k: usize,
    pub w0: Buf,
    pub w1: Buf,
}

impl QWeight {
    pub fn bytes(&self) -> usize {
        (self.w0.len + self.w1.len) * 4
    }
}

pub struct Gpu {
    pub client: Client,
    pub device: CubeDevice,
    pub plane: u32,
    /// Placeholder for kernel arguments a comptime flag leaves unused.
    dummy: Buf,
}

/// Split a 1-D cube count over (x, y) to respect per-dimension grid limits.
fn fold(n: u32) -> (u32, u32) {
    if n <= MAX_GRID {
        (n.max(1), 1)
    } else {
        (MAX_GRID, n.div_ceil(MAX_GRID))
    }
}

fn grid_1d(n: usize) -> CubeCount {
    let (x, y) = fold((n as u32).div_ceil(WG));
    CubeCount::Static(x, y, 1)
}

/// Comptime tile width for MoE kernels covering tiles of up to `max_cnt` pairs.
fn tile_width(max_cnt: usize) -> usize {
    match max_cnt {
        0 | 1 => 1,
        2 => 2,
        3 | 4 => 4,
        _ => MOE_MAXM,
    }
}

impl Gpu {
    pub fn new(device: CubeDevice) -> Self {
        let client = device.client();
        let hw = &client.properties().hardware;
        let plane = hw.plane_size_max.max(hw.plane_size_min).max(8);
        let dummy = Buf {
            h: client.empty(4),
            len: 1,
        };
        Self {
            client,
            device,
            plane,
            dummy,
        }
    }

    pub fn alloc(&self, len: usize) -> Buf {
        Buf {
            h: self.client.empty(len.max(1) * 4),
            len,
        }
    }

    pub fn zeros(&self, len: usize) -> Buf {
        self.upload_u32(&vec![0u32; len.max(1)])
    }

    pub fn upload_f32(&self, v: &[f32]) -> Buf {
        Buf {
            h: self.client.create_from_slice(bytemuck::cast_slice(v)),
            len: v.len(),
        }
    }

    pub fn upload_u32(&self, v: &[u32]) -> Buf {
        Buf {
            h: self.client.create_from_slice(bytemuck::cast_slice(v)),
            len: v.len(),
        }
    }

    /// Write `data` into `dst` starting at element `offset`.
    pub fn write_at(&self, dst: &Buf, offset: usize, data: Vec<u32>) {
        self.write_many(vec![(dst, offset, data)]);
    }

    /// Batched [`Self::write_at`].
    ///
    /// Each destination handle is bounded on both ends: wgpu copies the whole bound
    /// range (padding past the data with staging garbage), so an open-ended range would
    /// clobber everything after the written region.
    pub fn write_many(&self, writes: Vec<(&Buf, usize, Vec<u32>)>) {
        for (dst, offset, data) in writes {
            assert!(offset + data.len() <= dst.len);
            let start = (offset * 4) as u64;
            let end = dst.h.size() - start - (data.len() * 4) as u64;
            let h = dst.h.clone().offset_start(start).offset_end(end);
            self.client.write(&h, Bytes::from_elems(data));
        }
    }

    pub fn read_f32(&self, b: &Buf) -> Vec<f32> {
        let bytes = self.client.read_one_unchecked(b.h.clone());
        bytemuck::cast_slice::<u8, f32>(&bytes[..b.len * 4]).to_vec()
    }

    pub fn read_u32(&self, b: &Buf) -> Vec<u32> {
        let bytes = self.client.read_one_unchecked(b.h.clone());
        bytemuck::cast_slice::<u8, u32>(&bytes[..b.len * 4]).to_vec()
    }

    /// Threads per cube for the plane-per-row kernels and the rows each cube covers
    /// (computed with the largest plane size, so the grid never undercounts).
    fn rows_cube(&self) -> (u32, u32) {
        let threads = (4 * self.plane).min(256);
        (threads, threads / self.plane)
    }

    /// out[m, n] (+)= x[m, k] . W[n, k] for small m (decode-style, weight streamed once
    /// per `maxm` rows of x).
    pub fn qmatmul(&self, w: &QWeight, x: &Buf, m: usize, out: &Buf, accumulate: bool) {
        assert!(x.len >= m * w.k && out.len >= m * w.n);
        let maxm = if m >= 8 {
            8
        } else if m >= 4 {
            4
        } else if m >= 2 {
            2
        } else {
            1
        };
        let (threads, rows) = self.rows_cube();
        let rpp = if w.fmt == 1 && maxm >= 2 { 4usize } else { 1 };
        let (gx, gy) = fold((w.n as u32).div_ceil(rows * rpp as u32));
        let gz = (m as u32).div_ceil(maxm as u32);
        kernels::qmatmul::launch(
            &self.client,
            CubeCount::Static(gx, gy, gz),
            CubeDim::new_1d(threads),
            w.w0.arg(),
            w.w0.argv(),
            w.w1.arg(),
            x.argv(),
            out.arg(),
            w.n as u32,
            w.k as u32,
            m as u32,
            w.fmt,
            maxm,
            rpp,
            accumulate,
        );
    }

    /// Full f32 dequantization of a weight into a fresh [n, k] buffer.
    pub fn dequant(&self, w: &QWeight) -> Buf {
        let nb = w.n * w.k / 32;
        let out = self.alloc(w.n * w.k);
        kernels::dequant::launch(
            &self.client,
            grid_1d(nb),
            CubeDim::new_1d(WG),
            w.w0.arg(),
            w.w1.arg(),
            out.arg(),
            nb as u32,
            w.fmt,
        );
        out
    }

    /// Large-M linear layer: dequantize then run burn's tiled matmul.
    /// Returns a fresh [m, n] buffer.
    pub fn linear_gemm(&self, w: &QWeight, x: &Buf, m: usize) -> Buf {
        let wd = self.dequant(w);
        let lhs = CubeTensor::new_contiguous(
            self.client.clone(),
            self.device.clone(),
            Shape::new([m, w.k]),
            x.h.clone(),
            DType::F32,
        );
        // W^T as a strided [k, n] view of the row-major [n, k] dequantized weight.
        let rhs = CubeTensor::new(
            self.client.clone(),
            wd.h.clone(),
            Metadata::new(Shape::new([w.k, w.n]), strides![1, w.k]),
            self.device.clone(),
            DType::F32,
        );
        let out =
            matmul(lhs, rhs, None, MatmulStrategy::default(), DType::F32).expect("matmul setup");
        // The runtime may pitch output rows (e.g. n = 257); consumers index [m, n] densely.
        let out = into_contiguous(out);
        Buf {
            h: out.handle,
            len: m * w.n,
        }
    }

    /// Dequant + tiled GEMM wins once enough tokens share the dequantized weight; its
    /// fixed dequant/launch cost never pays off for small outputs (router, ssm_alpha/beta).
    fn use_gemm(w: &QWeight, m: usize, gemm_min_m: usize) -> bool {
        m >= gemm_min_m && w.n >= 1024
    }

    /// out = x @ W^T for any m, choosing the streamed or GEMM path.
    pub fn linear(&self, w: &QWeight, x: &Buf, m: usize, gemm_min_m: usize) -> Buf {
        if Self::use_gemm(w, m, gemm_min_m) {
            self.linear_gemm(w, x, m)
        } else {
            let out = self.alloc(m * w.n);
            self.qmatmul(w, x, m, &out, false);
            out
        }
    }

    /// acc += x @ W^T
    pub fn linear_acc(&self, w: &QWeight, x: &Buf, m: usize, acc: &Buf, gemm_min_m: usize) {
        if Self::use_gemm(w, m, gemm_min_m) {
            let y = self.linear_gemm(w, x, m);
            self.add_inplace(acc, &y, m * w.n);
        } else {
            self.qmatmul(w, x, m, acc, true);
        }
    }

    pub fn rmsnorm(&self, x: &Buf, w: &Buf, rows: usize, d: usize, eps: f32) -> Buf {
        let out = self.alloc(rows * d);
        kernels::rmsnorm::launch(
            &self.client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_1d(WG),
            x.arg(),
            w.arg(),
            out.arg(),
            d as u32,
            eps,
        );
        out
    }

    pub fn add_inplace(&self, x: &Buf, y: &Buf, n: usize) {
        kernels::add_inplace::launch(
            &self.client,
            grid_1d(n),
            CubeDim::new_1d(WG),
            x.arg(),
            y.arg(),
            n as u32,
        );
    }

    pub fn swiglu(&self, g: &Buf, u: &Buf, n: usize) -> Buf {
        let out = self.alloc(n);
        kernels::swiglu::launch(
            &self.client,
            grid_1d(n),
            CubeDim::new_1d(WG),
            g.arg(),
            u.arg(),
            out.arg(),
            n as u32,
        );
        out
    }

    /// `snap`: rollback buffer for the state after each token but the last (see kernel).
    pub fn gdn_conv(
        &self,
        x: &Buf,
        w: &Buf,
        state: &Buf,
        snap: Option<&Buf>,
        c: usize,
        t: usize,
    ) -> Buf {
        let out = self.alloc(t * c);
        kernels::gdn_conv::launch(
            &self.client,
            CubeCount::Static((c as u32).div_ceil(WG), 1, 1),
            CubeDim::new_1d(WG),
            x.arg(),
            w.arg(),
            state.arg(),
            out.arg(),
            snap.unwrap_or(&self.dummy).arg(),
            c as u32,
            t as u32,
            snap.is_some(),
        );
        out
    }

    /// dst[0..n] = src[src_off..src_off + n]
    pub fn copy_f32(&self, dst: &Buf, src: &Buf, src_off: usize, n: usize) {
        kernels::copy_f32::launch(
            &self.client,
            grid_1d(n),
            CubeDim::new_1d(WG),
            dst.arg(),
            src.arg(),
            src_off as u32,
            n as u32,
        );
    }

    /// Rows [a | b] of width 2d from two [rows, d] buffers.
    pub fn concat_rows(&self, a: &Buf, b: &Buf, rows: usize, d: usize) -> Buf {
        let out = self.alloc(rows * 2 * d);
        kernels::concat_rows::launch(
            &self.client,
            grid_1d(rows * 2 * d),
            CubeDim::new_1d(WG),
            a.arg(),
            b.arg(),
            out.arg(),
            d as u32,
            (rows * 2 * d) as u32,
        );
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_recurrent(
        &self,
        conv: &Buf,
        alpha: &Buf,
        beta: &Buf,
        z: &Buf,
        ssm_a: &Buf,
        dt_bias: &Buf,
        norm_w: &Buf,
        state: &Buf,
        snap: Option<&Buf>,
        t: usize,
        nk: usize,
        nv: usize,
        dk: usize,
        eps: f32,
    ) -> Buf {
        let out = self.alloc(t * nv * dk);
        kernels::gdn_recurrent::launch(
            &self.client,
            CubeCount::Static(nv as u32, 1, 1),
            CubeDim::new_1d(dk as u32),
            conv.arg(),
            alpha.arg(),
            beta.arg(),
            z.arg(),
            ssm_a.arg(),
            dt_bias.arg(),
            norm_w.arg(),
            state.arg(),
            out.arg(),
            snap.unwrap_or(&self.dummy).arg(),
            t as u32,
            nk as u32,
            nv as u32,
            eps,
            dk,
            snap.is_some(),
        );
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_prep(
        &self,
        qg: &Buf,
        k: &Buf,
        v: &Buf,
        q_norm: &Buf,
        k_norm: &Buf,
        k_cache: &Buf,
        v_cache: &Buf,
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos0: usize,
        rope_dim: usize,
        theta: f32,
        eps: f32,
    ) -> Buf {
        let q = self.alloc(t * nh * hd);
        kernels::attn_prep::launch(
            &self.client,
            CubeCount::Static((nh + nkv) as u32, t as u32, 1),
            CubeDim::new_1d(hd as u32),
            qg.arg(),
            k.arg(),
            v.arg(),
            q_norm.arg(),
            k_norm.arg(),
            q.arg(),
            k_cache.arg(),
            v_cache.arg(),
            nh as u32,
            nkv as u32,
            pos0 as u32,
            rope_dim as u32,
            theta,
            eps,
            hd,
        );
        q
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &Buf,
        qg: &Buf,
        k_cache: &Buf,
        v_cache: &Buf,
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos0: usize,
    ) -> Buf {
        assert_eq!(
            hd, 256,
            "attention kernel assumes 32 keys x 8 lanes per tile"
        );
        let out = self.alloc(t * nh * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let nkeys = pos0 + t;
        // Decode: split the keys across cubes (flash-decoding) once one cube per head
        // would leave the device mostly idle.
        let split_len = (nkeys.div_ceil(32).div_ceil(32) * 32).max(256);
        let nsplit = if t == 1 { nkeys.div_ceil(split_len) } else { 1 };
        if nsplit <= 1 {
            kernels::attention::launch(
                &self.client,
                CubeCount::Static(nh as u32, t as u32, 1),
                CubeDim::new_1d(hd as u32),
                q.arg(),
                qg.arg(),
                k_cache.argv(),
                v_cache.arg(),
                out.arg(),
                self.dummy.arg(),
                nh as u32,
                nkv as u32,
                pos0 as u32,
                nkeys as u32,
                scale,
                hd,
                false,
            );
            return out;
        }
        let parts = self.alloc(t * nh * nsplit * hd);
        let ml = self.alloc(t * nh * nsplit * 2);
        kernels::attention::launch(
            &self.client,
            CubeCount::Static(nh as u32, t as u32, nsplit as u32),
            CubeDim::new_1d(hd as u32),
            q.arg(),
            qg.arg(),
            k_cache.argv(),
            v_cache.arg(),
            parts.arg(),
            ml.arg(),
            nh as u32,
            nkv as u32,
            pos0 as u32,
            split_len as u32,
            scale,
            hd,
            true,
        );
        kernels::attention_combine::launch(
            &self.client,
            CubeCount::Static(nh as u32, t as u32, 1),
            CubeDim::new_1d(hd as u32),
            parts.arg(),
            ml.arg(),
            qg.arg(),
            out.arg(),
            nh as u32,
            nsplit as u32,
            hd,
        );
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_prep_paged(
        &self,
        qg: &Buf,
        k: &Buf,
        v: &Buf,
        q_norm: &Buf,
        k_norm: &Buf,
        k_pool: &Buf,
        v_pool: &Buf,
        page_table: &Buf,
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos0: usize,
        page_size: usize,
        rope_dim: usize,
        theta: f32,
        eps: f32,
    ) -> Buf {
        let q = self.alloc(t * nh * hd);
        kernels::attn_prep_paged::launch(
            &self.client,
            CubeCount::Static((nh + nkv) as u32, t as u32, 1),
            CubeDim::new_1d(hd as u32),
            qg.arg(),
            k.arg(),
            v.arg(),
            q_norm.arg(),
            k_norm.arg(),
            q.arg(),
            k_pool.arg(),
            v_pool.arg(),
            page_table.arg(),
            nh as u32,
            nkv as u32,
            pos0 as u32,
            page_size as u32,
            rope_dim as u32,
            theta,
            eps,
            hd,
        );
        q
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_paged(
        &self,
        q: &Buf,
        qg: &Buf,
        k_pool: &Buf,
        v_pool: &Buf,
        page_table: &Buf,
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos0: usize,
        page_size: usize,
    ) -> Buf {
        assert_eq!(
            hd, 256,
            "attention kernel assumes 32 keys x 8 lanes per tile"
        );
        let out = self.alloc(t * nh * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let nkeys = pos0 + t;
        let target_cubes = 256usize;
        let want_splits = if t <= 8 && nkeys >= 1024 {
            (target_cubes / (nh * t)).clamp(1, 32)
        } else {
            1
        };
        let split_len = if want_splits > 1 {
            (nkeys.div_ceil(want_splits).div_ceil(32) * 32).max(256)
        } else {
            nkeys.max(1)
        };
        let nsplit = if want_splits > 1 {
            nkeys.div_ceil(split_len)
        } else {
            1
        };
        if nsplit <= 1 {
            kernels::attention_paged::launch(
                &self.client,
                CubeCount::Static(nh as u32, t as u32, 1),
                CubeDim::new_1d(hd as u32),
                q.arg(),
                qg.arg(),
                k_pool.argv(),
                v_pool.arg(),
                page_table.arg(),
                out.arg(),
                self.dummy.arg(),
                nh as u32,
                nkv as u32,
                pos0 as u32,
                page_size as u32,
                nkeys as u32,
                scale,
                hd,
                false,
            );
            return out;
        }
        let parts = self.alloc(t * nh * nsplit * hd);
        let ml = self.alloc(t * nh * nsplit * 2);
        kernels::attention_paged::launch(
            &self.client,
            CubeCount::Static(nh as u32, t as u32, nsplit as u32),
            CubeDim::new_1d(hd as u32),
            q.arg(),
            qg.arg(),
            k_pool.argv(),
            v_pool.arg(),
            page_table.arg(),
            parts.arg(),
            ml.arg(),
            nh as u32,
            nkv as u32,
            pos0 as u32,
            page_size as u32,
            split_len as u32,
            scale,
            hd,
            true,
        );
        kernels::attention_combine::launch(
            &self.client,
            CubeCount::Static(nh as u32, t as u32, 1),
            CubeDim::new_1d(hd as u32),
            parts.arg(),
            ml.arg(),
            qg.arg(),
            out.arg(),
            nh as u32,
            nsplit as u32,
            hd,
        );
        out
    }

    pub fn moe_route(
        &self,
        logits: &Buf,
        t: usize,
        ne: usize,
        stride: usize,
        topk: usize,
    ) -> (Buf, Buf) {
        let ids = self.alloc(t * topk);
        let wts = self.alloc(t * topk);
        kernels::moe_route::launch(
            &self.client,
            CubeCount::Static((t as u32).div_ceil(64), 1, 1),
            CubeDim::new_1d(64),
            logits.arg(),
            ids.arg(),
            wts.arg(),
            t as u32,
            ne as u32,
            stride as u32,
            topk,
        );
        (ids, wts)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_route_pack(
        &self,
        logits: &Buf,
        h2: &Buf,
        t: usize,
        ne: usize,
        stride: usize,
        hidden: usize,
        topk: usize,
    ) -> (Buf, Buf, Buf) {
        let ids = self.alloc(t * topk);
        let wts = self.alloc(t * topk);
        let pack = self.alloc(t * (2 * topk + hidden));
        kernels::moe_route_pack::launch(
            &self.client,
            CubeCount::Static(t as u32, 1, 1),
            CubeDim::new_1d(WG),
            logits.arg(),
            h2.arg(),
            ids.arg(),
            wts.arg(),
            pack.arg(),
            t as u32,
            ne as u32,
            stride as u32,
            hidden as u32,
            topk,
        );
        (ids, wts, pack)
    }

    pub fn moe_direct_desc(&self, ids: &Buf, npairs: usize) -> Buf {
        let desc = self.alloc(npairs * MOE_DESC);
        kernels::moe_direct_desc::launch(
            &self.client,
            grid_1d(npairs),
            CubeDim::new_1d(WG),
            ids.arg(),
            desc.arg(),
            npairs as u32,
        );
        desc
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_gate_up(
        &self,
        gate: (&Buf, &Buf),
        up: (&Buf, &Buf),
        fmt: u32,
        x: &Buf,
        desc: &Buf,
        ntiles: usize,
        act: &Buf,
        inter: usize,
        k: usize,
        topk: usize,
        max_cnt: usize,
    ) {
        let (threads, rows) = self.rows_cube();
        kernels::moe_gate_up::launch(
            &self.client,
            CubeCount::Static((inter as u32).div_ceil(rows), ntiles as u32, 1),
            CubeDim::new_1d(threads),
            gate.0.arg(),
            gate.0.argv(),
            gate.1.arg(),
            up.0.arg(),
            up.0.argv(),
            up.1.arg(),
            x.argv(),
            desc.arg(),
            act.arg(),
            inter as u32,
            k as u32,
            topk as u32,
            fmt,
            tile_width(max_cnt),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_down(
        &self,
        w: (&Buf, &Buf),
        fmt: u32,
        act: &Buf,
        desc: &Buf,
        ntiles: usize,
        y: &Buf,
        hidden: usize,
        inter: usize,
        max_cnt: usize,
    ) {
        let (threads, rows) = self.rows_cube();
        kernels::moe_down::launch(
            &self.client,
            CubeCount::Static((hidden as u32).div_ceil(rows), ntiles as u32, 1),
            CubeDim::new_1d(threads),
            w.0.arg(),
            w.0.argv(),
            w.1.arg(),
            act.argv(),
            desc.arg(),
            y.arg(),
            hidden as u32,
            inter as u32,
            fmt,
            tile_width(max_cnt),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine(
        &self,
        x: &Buf,
        y: &Buf,
        wts: &Buf,
        shared: &Buf,
        logits: &Buf,
        cpu_pack: Option<&Buf>,
        t: usize,
        hidden: usize,
        topk: usize,
        stride: usize,
        gate_col: usize,
    ) {
        let total = t * hidden;
        let npairs = t * topk;
        let (cpu_buf, has_cpu) = match cpu_pack {
            Some(b) => (b, true),
            None => (shared, false),
        };
        kernels::moe_combine::launch(
            &self.client,
            grid_1d(total),
            CubeDim::new_1d(WG),
            x.arg(),
            y.arg(),
            wts.arg(),
            shared.arg(),
            logits.arg(),
            cpu_buf.arg(),
            hidden as u32,
            total as u32,
            topk as u32,
            stride as u32,
            gate_col as u32,
            npairs as u32,
            has_cpu,
        );
    }

    /// Argmax along the last dimension for `[rows, n]` logits; returns `rows` token ids.
    pub fn argmax_rows(&self, logits: &Buf, rows: usize, n: usize) -> Vec<u32> {
        const NSPLIT: usize = 64;
        let part_v = self.alloc(rows * NSPLIT);
        let part_i = self.alloc(rows * NSPLIT);
        let out = self.alloc(rows);
        kernels::argmax_stage1::launch(
            &self.client,
            CubeCount::Static(NSPLIT as u32, rows as u32, 1),
            CubeDim::new_1d(WG),
            logits.arg(),
            part_v.arg(),
            part_i.arg(),
            n as u32,
            NSPLIT,
        );
        kernels::argmax_stage2::launch(
            &self.client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_1d(NSPLIT as u32),
            part_v.arg(),
            part_i.arg(),
            out.arg(),
            NSPLIT,
        );
        self.read_u32(&out)
    }
    pub fn sync(&self) {
        futures_lite::future::block_on(self.client.sync()).expect("device sync");
    }
}
