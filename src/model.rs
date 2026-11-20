//! Qwen3.5 MoE (hybrid Gated-DeltaNet / gated full attention) forward pass.

use crate::gguf::{Gguf, TensorInfo};
use crate::gpu::{Buf, Gpu, QWeight};
use crate::kernels::{MOE_DESC, MOE_MAXM};
use crate::quant::{self, fmt};
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub n_experts: usize,
    pub topk: usize,
    pub expert_inter: usize,
    pub shared_inter: usize,
    pub conv_k: usize,
    /// Key/value head dim of the linear-attention layers.
    pub ssm_dk: usize,
    /// Number of key heads (ssm.group_count).
    pub ssm_nk: usize,
    /// Number of value heads (ssm.time_step_rank).
    pub ssm_nv: usize,
    pub full_attn_interval: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.meta_str("general.architecture")?;
        ensure!(
            arch == "qwen35moe",
            "unsupported architecture {arch} (want qwen35moe)"
        );
        let k = |s: &str| g.meta_u64(&format!("{arch}.{s}")).map(|v| v as usize);
        let vocab = g.tensor("token_embd.weight")?.dims[1] as usize;
        let c = Self {
            n_layers: k("block_count")?,
            hidden: k("embedding_length")?,
            vocab,
            n_heads: k("attention.head_count")?,
            n_kv: k("attention.head_count_kv")?,
            head_dim: k("attention.key_length")?,
            rope_dim: k("rope.dimension_count")?,
            rope_theta: g.meta_f64(&format!("{arch}.rope.freq_base"))? as f32,
            eps: g.meta_f64(&format!("{arch}.attention.layer_norm_rms_epsilon"))? as f32,
            n_experts: k("expert_count")?,
            topk: k("expert_used_count")?,
            expert_inter: k("expert_feed_forward_length")?,
            shared_inter: k("expert_shared_feed_forward_length")?,
            conv_k: k("ssm.conv_kernel")?,
            ssm_dk: k("ssm.state_size")?,
            ssm_nk: k("ssm.group_count")?,
            ssm_nv: k("ssm.time_step_rank")?,
            full_attn_interval: k("full_attention_interval")?,
        };
        ensure!(c.topk <= MOE_MAXM, "topk {} > {MOE_MAXM}", c.topk);
        ensure!(c.conv_k == 4, "conv kernel {} unsupported", c.conv_k);
        ensure!(
            k("ssm.inner_size")? == c.ssm_nv * c.ssm_dk,
            "ssm inner size mismatch"
        );
        Ok(c)
    }

    pub fn is_full_attn(&self, layer: usize) -> bool {
        (layer + 1) % self.full_attn_interval == 0
    }

    fn conv_dim(&self) -> usize {
        2 * self.ssm_nk * self.ssm_dk + self.ssm_nv * self.ssm_dk
    }
}

struct GdnW {
    qkv: QWeight,
    z: QWeight,
    alpha: QWeight,
    beta: QWeight,
    out: QWeight,
    conv_w: Buf,
    ssm_a: Buf,
    dt_bias: Buf,
    norm: Buf,
}

struct AttnW {
    q: QWeight,
    k: QWeight,
    v: QWeight,
    o: QWeight,
    q_norm: Buf,
    k_norm: Buf,
}

enum Mixer {
    Gdn(GdnW),
    Attn(AttnW),
}

/// One expert tensor stack ([n_experts, rows, k]) mapped to a slot pool on the device.
struct ExpertTensor {
    info: TensorInfo,
    fmt: u32,
    expert_bytes: usize,
    w0_per: usize,
    w1_per: usize,
    w0: Buf,
    w1: Buf,
}

impl ExpertTensor {
    fn new(gpu: &Gpu, info: &TensorInfo, n_experts: usize, slots: usize) -> Self {
        let per_numel = (info.numel() as usize) / n_experts;
        let f = info.ty.gpu_fmt();
        let w0_per = fmt::w0_words(f, per_numel);
        let w1_per = if matches!(f, fmt::Q8_0 | fmt::MXFP4) {
            fmt::w1_words(f, per_numel)
        } else {
            0
        };
        let (w0, w1) = gpu.client.memory_persistent_allocation((), |_| {
            (
                gpu.alloc(w0_per * slots),
                gpu.alloc((w1_per * slots).max(1)),
            )
        });
        Self {
            info: info.clone(),
            fmt: f,
            expert_bytes: info.nbytes / n_experts,
            w0_per,
            w1_per,
            w0,
            w1,
        }
    }

    fn raw<'a>(&self, g: &'a Gguf, e: usize) -> &'a [u8] {
        let all = g.bytes(&self.info);
        &all[e * self.expert_bytes..(e + 1) * self.expert_bytes]
    }

    fn arg(&self) -> (&Buf, &Buf) {
        (&self.w0, &self.w1)
    }
}

/// Expert residency for one layer: either all experts resident (slot = expert id) or
/// an LRU cache of `slots` experts refilled from the mmap'd GGUF on demand.
struct ExpertStore {
    gate: ExpertTensor,
    up: ExpertTensor,
    down: ExpertTensor,
    slots: usize,
    slot_of: Vec<i32>,
    expert_in: Vec<i32>,
    stamp: Vec<u64>,
}

impl ExpertStore {
    fn resident(&self, n_experts: usize) -> bool {
        self.slots == n_experts
    }

    /// Load experts `(expert, slot)` into their slots: repack in parallel on the host,
    /// then enqueue the device writes (stream-ordered with earlier kernels). Batches are
    /// bounded so the pinned staging they need stays small.
    fn load_many(&mut self, gpu: &Gpu, g: &Gguf, list: &[(usize, usize)]) {
        let tensors = [&self.gate, &self.up, &self.down];
        for batch in list.chunks(16) {
            let packed: Vec<(usize, usize, quant::Packed)> = batch
                .par_iter()
                .flat_map_iter(|&(e, slot)| (0..3).map(move |ti| (e, slot, ti)))
                .map(|(e, slot, ti)| {
                    (
                        slot,
                        ti,
                        quant::repack(tensors[ti].info.ty, tensors[ti].raw(g, e)),
                    )
                })
                .collect();
            let mut writes = Vec::with_capacity(packed.len() * 2);
            for (slot, ti, p) in packed {
                let t = tensors[ti];
                writes.push((&t.w0, slot * t.w0_per, p.w0));
                if t.w1_per > 0 {
                    writes.push((&t.w1, slot * t.w1_per, p.w1));
                }
            }
            gpu.write_many(writes);
        }
        for &(e, slot) in list {
            if self.expert_in[slot] >= 0 {
                self.slot_of[self.expert_in[slot] as usize] = -1;
            }
            self.expert_in[slot] = e as i32;
            self.slot_of[e] = slot as i32;
        }
    }
}

struct MoeW {
    /// Router [n_experts + 1, hidden] in f32; the last row is the shared-expert gate.
    router: QWeight,
    gate_sh: QWeight,
    up_sh: QWeight,
    down_sh: QWeight,
    experts: ExpertStore,
}

struct Layer {
    attn_norm: Buf,
    post_norm: Buf,
    mixer: Mixer,
    moe: MoeW,
}

pub struct Model {
    pub cfg: Config,
    pub gpu: Gpu,
    gguf: Gguf,
    layers: Vec<Layer>,
    out_norm: Buf,
    lm_head: QWeight,
    embd: TensorInfo,
    /// Rows at or above which dense layers dequantize + use the tiled GEMM.
    pub gemm_min_m: usize,
    clock: u64,
    pub stats: Stats,
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub expert_hits: u64,
    pub expert_misses: u64,
    /// Host time spent repacking + enqueueing expert uploads.
    pub upload_secs: f64,
    /// Host time blocked reading routing decisions back from the device.
    pub route_sync_secs: f64,
}

impl Stats {
    pub fn since(&self, earlier: &Stats) -> Stats {
        Stats {
            expert_hits: self.expert_hits - earlier.expert_hits,
            expert_misses: self.expert_misses - earlier.expert_misses,
            upload_secs: self.upload_secs - earlier.upload_secs,
            route_sync_secs: self.route_sync_secs - earlier.route_sync_secs,
        }
    }

    /// One-line summary; `None` when no host-routed expert lookups happened.
    pub fn summary(&self) -> Option<String> {
        let lookups = self.expert_hits + self.expert_misses;
        (lookups > 0).then(|| {
            format!(
                "{} lookups, hit rate {:.1}%, upload {:.2}s, routing sync {:.2}s",
                lookups,
                100.0 * self.expert_hits as f64 / lookups as f64,
                self.upload_secs,
                self.route_sync_secs
            )
        })
    }
}

pub struct LoadOptions {
    /// Expert slots per layer; `None` = fit to the VRAM budget.
    pub expert_slots: Option<usize>,
    /// VRAM budget in bytes for weights + cache (None = device-reported total).
    pub vram_budget: Option<u64>,
    pub max_ctx: usize,
}

pub struct State {
    pub pos: usize,
    max_ctx: usize,
    kv: Vec<Option<(Buf, Buf)>>,
    conv: Vec<Option<Buf>>,
    ssm: Vec<Option<Buf>>,
}

fn upload_weight(gpu: &Gpu, g: &Gguf, name: &str) -> Result<QWeight> {
    let t = g.tensor(name)?;
    ensure!(t.dims.len() == 2, "{name}: expected 2-D, got {:?}", t.dims);
    let p = quant::repack(t.ty, g.bytes(t));
    Ok(QWeight {
        fmt: p.fmt,
        k: t.dims[0] as usize,
        n: t.dims[1] as usize,
        w0: gpu.upload_u32(&p.w0),
        w1: gpu.upload_u32(&p.w1),
    })
}

fn f32_tensor(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name)?;
    let mut v = vec![0f32; t.numel() as usize];
    quant::dequant(t.ty, g.bytes(t), &mut v);
    Ok(v)
}

fn upload_f32(gpu: &Gpu, g: &Gguf, name: &str) -> Result<Buf> {
    Ok(gpu.upload_f32(&f32_tensor(g, name)?))
}

impl Model {
    pub fn load(gpu: Gpu, gguf: Gguf, opts: &LoadOptions) -> Result<Self> {
        let cfg = Config::from_gguf(&gguf)?;
        let t0 = Instant::now();
        let g = &gguf;
        let blk = |i: usize, s: &str| format!("blk.{i}.{s}");

        // Size the expert pools before allocating anything.
        let mut expert_bytes_layer = 0usize;
        let mut dense_bytes = 0usize;
        for t in g.tensors.values() {
            if t.name.contains("_exps.") {
                expert_bytes_layer = expert_bytes_layer.max(0) + t.nbytes;
            } else if t.name != "token_embd.weight" {
                dense_bytes += t.nbytes;
            }
        }
        let expert_bytes_per = expert_bytes_layer / (cfg.n_layers * cfg.n_experts);
        let kv_bytes =
            cfg.n_layers / cfg.full_attn_interval * 2 * opts.max_ctx * cfg.n_kv * cfg.head_dim * 4;
        let total_vram = opts
            .vram_budget
            .or(gpu.client.properties().memory.max_memory())
            .unwrap_or(8 << 30) as usize;
        // GPU layouts carry a little padding; scratch for a 512-token prefill chunk.
        let reserve = (dense_bytes as f64 * 1.08) as usize + kv_bytes + (1200 << 20);
        let slots = match opts.expert_slots {
            Some(s) => s.clamp(cfg.topk, cfg.n_experts),
            None => {
                let avail = total_vram.saturating_sub(reserve);
                let per_slot = (expert_bytes_per as f64 * 1.08) as usize * cfg.n_layers;
                (avail / per_slot).clamp(cfg.topk, cfg.n_experts)
            }
        };
        eprintln!(
            "vram budget {:.2} GiB: dense {:.2} GiB, kv {:.2} GiB, experts {}/{} slots per layer ({:.2} GiB)",
            total_vram as f64 / (1u64 << 30) as f64,
            dense_bytes as f64 / (1u64 << 30) as f64,
            kv_bytes as f64 / (1u64 << 30) as f64,
            slots,
            cfg.n_experts,
            (slots * expert_bytes_per * cfg.n_layers) as f64 / (1u64 << 30) as f64,
        );

        let gpu_ref = &gpu;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        gpu.client
            .memory_persistent_allocation((), |_| -> Result<()> {
                for i in 0..cfg.n_layers {
                    let mixer = if cfg.is_full_attn(i) {
                        Mixer::Attn(AttnW {
                            q: upload_weight(gpu_ref, g, &blk(i, "attn_q.weight"))?,
                            k: upload_weight(gpu_ref, g, &blk(i, "attn_k.weight"))?,
                            v: upload_weight(gpu_ref, g, &blk(i, "attn_v.weight"))?,
                            o: upload_weight(gpu_ref, g, &blk(i, "attn_output.weight"))?,
                            q_norm: upload_f32(gpu_ref, g, &blk(i, "attn_q_norm.weight"))?,
                            k_norm: upload_f32(gpu_ref, g, &blk(i, "attn_k_norm.weight"))?,
                        })
                    } else {
                        Mixer::Gdn(GdnW {
                            qkv: upload_weight(gpu_ref, g, &blk(i, "attn_qkv.weight"))?,
                            z: upload_weight(gpu_ref, g, &blk(i, "attn_gate.weight"))?,
                            alpha: upload_weight(gpu_ref, g, &blk(i, "ssm_alpha.weight"))?,
                            beta: upload_weight(gpu_ref, g, &blk(i, "ssm_beta.weight"))?,
                            out: upload_weight(gpu_ref, g, &blk(i, "ssm_out.weight"))?,
                            conv_w: upload_f32(gpu_ref, g, &blk(i, "ssm_conv1d.weight"))?,
                            ssm_a: upload_f32(gpu_ref, g, &blk(i, "ssm_a"))?,
                            dt_bias: upload_f32(gpu_ref, g, &blk(i, "ssm_dt.bias"))?,
                            norm: upload_f32(gpu_ref, g, &blk(i, "ssm_norm.weight"))?,
                        })
                    };
                    let mut router = f32_tensor(g, &blk(i, "ffn_gate_inp.weight"))?;
                    router.extend(f32_tensor(g, &blk(i, "ffn_gate_inp_shexp.weight"))?);
                    ensure!(
                        router.len() == (cfg.n_experts + 1) * cfg.hidden,
                        "router shape"
                    );
                    let router = QWeight {
                        fmt: fmt::F32,
                        n: cfg.n_experts + 1,
                        k: cfg.hidden,
                        w0: gpu_ref.upload_u32(bytemuck::cast_slice(&router)),
                        w1: gpu_ref.upload_u32(&[0]),
                    };
                    let gate_t = g.tensor(&blk(i, "ffn_gate_exps.weight"))?;
                    let up_t = g.tensor(&blk(i, "ffn_up_exps.weight"))?;
                    let down_t = g.tensor(&blk(i, "ffn_down_exps.weight"))?;
                    ensure!(
                        gate_t.ty == up_t.ty,
                        "layer {i}: gate/up expert types differ"
                    );
                    let ne = cfg.n_experts;
                    let experts = ExpertStore {
                        gate: ExpertTensor::new(gpu_ref, gate_t, ne, slots),
                        up: ExpertTensor::new(gpu_ref, up_t, ne, slots),
                        down: ExpertTensor::new(gpu_ref, down_t, ne, slots),
                        slots,
                        slot_of: vec![-1; ne],
                        expert_in: vec![-1; slots],
                        stamp: vec![0; slots],
                    };
                    layers.push(Layer {
                        attn_norm: upload_f32(gpu_ref, g, &blk(i, "attn_norm.weight"))?,
                        post_norm: upload_f32(gpu_ref, g, &blk(i, "post_attention_norm.weight"))?,
                        mixer,
                        moe: MoeW {
                            router,
                            gate_sh: upload_weight(gpu_ref, g, &blk(i, "ffn_gate_shexp.weight"))?,
                            up_sh: upload_weight(gpu_ref, g, &blk(i, "ffn_up_shexp.weight"))?,
                            down_sh: upload_weight(gpu_ref, g, &blk(i, "ffn_down_shexp.weight"))?,
                            experts,
                        },
                    });
                }
                Ok(())
            })?;
        let (out_norm, lm_head) =
            gpu.client
                .memory_persistent_allocation((), |_| -> Result<_> {
                    Ok((
                        upload_f32(gpu_ref, g, "output_norm.weight")?,
                        upload_weight(gpu_ref, g, "output.weight")?,
                    ))
                })?;
        let embd = g.tensor("token_embd.weight")?.clone();
        let mut model = Self {
            cfg,
            gpu,
            gguf,
            layers,
            out_norm,
            lm_head,
            embd,
            // Measured crossover (`--bench`, RTX 5050): qmatmul is ~18 us per token on
            // 8192x2048 Q8_0, dequant + GEMM ~1.3 ms flat.
            gemm_min_m: 80,
            clock: 0,
            stats: Stats::default(),
        };
        // Fill every slot up front: resident mode loads all experts, cache mode warms up
        // with the first experts.
        for li in 0..model.layers.len() {
            let store = &mut model.layers[li].moe.experts;
            let all: Vec<(usize, usize)> = (0..store.slots).map(|s| (s, s)).collect();
            store.load_many(&model.gpu, &model.gguf, &all);
            model.gpu.sync();
            eprint!("\rloading experts: layer {}/{}", li + 1, model.cfg.n_layers);
        }
        eprintln!();
        eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(model)
    }

    pub fn new_state(&self, max_ctx: usize) -> State {
        let c = &self.cfg;
        let mut st = State {
            pos: 0,
            max_ctx,
            kv: vec![],
            conv: vec![],
            ssm: vec![],
        };
        for i in 0..c.n_layers {
            if c.is_full_attn(i) {
                let n = max_ctx * c.n_kv * c.head_dim;
                st.kv.push(Some((self.gpu.zeros(n), self.gpu.zeros(n))));
                st.conv.push(None);
                st.ssm.push(None);
            } else {
                st.kv.push(None);
                st.conv
                    .push(Some(self.gpu.zeros((c.conv_k - 1) * c.conv_dim())));
                st.ssm
                    .push(Some(self.gpu.zeros(c.ssm_nv * c.ssm_dk * c.ssm_dk)));
            }
        }
        st
    }

    fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden;
        let row_bytes = self.embd.ty.row_bytes(h);
        let all = self.gguf.bytes(&self.embd);
        let mut x = vec![0f32; tokens.len() * h];
        for (i, &t) in tokens.iter().enumerate() {
            let src = &all[t as usize * row_bytes..(t as usize + 1) * row_bytes];
            quant::dequant(self.embd.ty, src, &mut x[i * h..(i + 1) * h]);
        }
        x
    }

    /// Run `tokens` through the model, appending to `state`; returns last-token logits.
    pub fn forward(&mut self, tokens: &[u32], st: &mut State) -> Result<Vec<f32>> {
        let t = tokens.len();
        ensure!(t > 0, "empty input");
        if st.pos + t > st.max_ctx {
            bail!("context overflow: {} + {} > {}", st.pos, t, st.max_ctx);
        }
        let c = self.cfg.clone();
        let gm = self.gemm_min_m;
        let x = self.gpu.upload_f32(&self.embed(tokens));
        for li in 0..c.n_layers {
            let h = self
                .gpu
                .rmsnorm(&x, &self.layers[li].attn_norm, t, c.hidden, c.eps);
            let gpu = &self.gpu;
            match &self.layers[li].mixer {
                Mixer::Gdn(w) => {
                    let qkv = gpu.linear(&w.qkv, &h, t, gm);
                    let z = gpu.linear(&w.z, &h, t, gm);
                    let a = gpu.linear(&w.alpha, &h, t, gm);
                    let b = gpu.linear(&w.beta, &h, t, gm);
                    let conv = gpu.gdn_conv(
                        &qkv,
                        &w.conv_w,
                        st.conv[li].as_ref().unwrap(),
                        c.conv_dim(),
                        t,
                    );
                    let o = gpu.gdn_recurrent(
                        &conv,
                        &a,
                        &b,
                        &z,
                        &w.ssm_a,
                        &w.dt_bias,
                        &w.norm,
                        st.ssm[li].as_ref().unwrap(),
                        t,
                        c.ssm_nk,
                        c.ssm_nv,
                        c.ssm_dk,
                        c.eps,
                    );
                    gpu.linear_acc(&w.out, &o, t, &x, gm);
                }
                Mixer::Attn(w) => {
                    let (kc, vc) = st.kv[li].as_ref().unwrap();
                    let qg = gpu.linear(&w.q, &h, t, gm);
                    let k = gpu.linear(&w.k, &h, t, gm);
                    let v = gpu.linear(&w.v, &h, t, gm);
                    let q = gpu.attn_prep(
                        &qg,
                        &k,
                        &v,
                        &w.q_norm,
                        &w.k_norm,
                        kc,
                        vc,
                        t,
                        c.n_heads,
                        c.n_kv,
                        c.head_dim,
                        st.pos,
                        c.rope_dim,
                        c.rope_theta,
                        c.eps,
                    );
                    let o =
                        gpu.attention(&q, &qg, kc, vc, t, c.n_heads, c.n_kv, c.head_dim, st.pos);
                    gpu.linear_acc(&w.o, &o, t, &x, gm);
                }
            }
            let h2 = self
                .gpu
                .rmsnorm(&x, &self.layers[li].post_norm, t, c.hidden, c.eps);
            self.moe(li, &h2, &x, t)?;
        }
        st.pos += t;
        let last = Buf {
            h: x.h.clone().offset_start(((t - 1) * c.hidden * 4) as u64),
            len: c.hidden,
        };
        let xn = self.gpu.rmsnorm(&last, &self.out_norm, 1, c.hidden, c.eps);
        let logits = self.gpu.alloc(c.vocab);
        self.gpu.qmatmul(&self.lm_head, &xn, 1, &logits, false);
        Ok(self.gpu.read_f32(&logits))
    }

    fn moe(&mut self, li: usize, h2: &Buf, x: &Buf, t: usize) -> Result<()> {
        let c = &self.cfg;
        let (ne, topk, inter, hidden) = (c.n_experts, c.topk, c.expert_inter, c.hidden);
        let gm = self.gemm_min_m;
        let gpu = &self.gpu;
        let w = &self.layers[li].moe;
        let logits = gpu.linear(&w.router, h2, t, gm);
        let (ids, wts) = gpu.moe_route(&logits, t, ne, ne + 1, topk);

        let g = gpu.linear(&w.gate_sh, h2, t, gm);
        let u = gpu.linear(&w.up_sh, h2, t, gm);
        let a = gpu.swiglu(&g, &u, t * c.shared_inter);
        let sh = gpu.linear(&w.down_sh, &a, t, gm);

        let npairs = t * topk;
        let act = gpu.alloc(npairs * inter);
        let y = gpu.alloc(npairs * hidden);

        if w.experts.resident(ne) && t <= 2 {
            // Fully resident: no host round-trip, one tile per (token, k) pair.
            let desc = gpu.moe_direct_desc(&ids, npairs);
            self.run_experts(li, &desc, npairs, 1, h2, &act, &y);
        } else {
            let ts = Instant::now();
            let ids_host = gpu.read_u32(&ids);
            self.stats.route_sync_secs += ts.elapsed().as_secs_f64();
            let mut by_expert: Vec<Vec<u32>> = vec![Vec::new(); ne];
            for (p, &e) in ids_host.iter().enumerate() {
                by_expert[e as usize].push(p as u32);
            }
            let store = &self.layers[li].moe.experts;
            let mut needed: Vec<usize> = (0..ne).filter(|&e| !by_expert[e].is_empty()).collect();
            // Resident experts first so they are served before any eviction.
            needed.sort_by_key(|&e| store.slot_of[e] < 0);
            let slots = store.slots;
            // Slot writes are ordered after kernels already queued on the stream (CUDA
            // stream order; cubecl-wgpu submits pending work before a queue write), so a
            // later round may reuse slots an earlier round read.
            for round in needed.chunks(slots) {
                let st = &self.layers[li].moe.experts;
                // Slots already holding this round's experts must not be evicted by it.
                let mut pinned = vec![false; slots];
                for &e in round {
                    if st.slot_of[e] >= 0 {
                        pinned[st.slot_of[e] as usize] = true;
                    }
                }
                let mut desc = Vec::new();
                let mut max_cnt = 0;
                let mut misses = Vec::new();
                let st = &mut self.layers[li].moe.experts;
                for &e in round {
                    self.clock += 1;
                    let slot = if st.slot_of[e] >= 0 {
                        self.stats.expert_hits += 1;
                        st.slot_of[e] as usize
                    } else {
                        self.stats.expert_misses += 1;
                        let victim = (0..slots)
                            .filter(|&s| !pinned[s])
                            .min_by_key(|&s| st.stamp[s])
                            .context("no evictable expert slot")?;
                        misses.push((e, victim));
                        victim
                    };
                    pinned[slot] = true;
                    st.stamp[slot] = self.clock;
                    for chunk in by_expert[e].chunks(MOE_MAXM) {
                        max_cnt = max_cnt.max(chunk.len());
                        desc.push(slot as u32);
                        desc.push(chunk.len() as u32);
                        desc.extend((0..MOE_MAXM).map(|i| chunk.get(i).copied().unwrap_or(0)));
                    }
                }
                let tl = Instant::now();
                st.load_many(&self.gpu, &self.gguf, &misses);
                self.stats.upload_secs += tl.elapsed().as_secs_f64();
                let ntiles = desc.len() / MOE_DESC;
                let desc = self.gpu.upload_u32(&desc);
                self.run_experts(li, &desc, ntiles, max_cnt, h2, &act, &y);
            }
        }
        self.gpu
            .moe_combine(x, &y, &wts, &sh, &logits, t, hidden, topk, ne + 1, ne);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_experts(
        &self,
        li: usize,
        desc: &Buf,
        ntiles: usize,
        max_cnt: usize,
        h2: &Buf,
        act: &Buf,
        y: &Buf,
    ) {
        let c = &self.cfg;
        let st = &self.layers[li].moe.experts;
        let (inter, hidden) = (c.expert_inter, c.hidden);
        let (g, u, d) = (st.gate.arg(), st.up.arg(), st.down.arg());
        self.gpu.moe_gate_up(
            g,
            u,
            st.gate.fmt,
            h2,
            desc,
            ntiles,
            act,
            inter,
            hidden,
            c.topk,
            max_cnt,
        );
        self.gpu
            .moe_down(d, st.down.fmt, act, desc, ntiles, y, hidden, inter, max_cnt);
    }
}
