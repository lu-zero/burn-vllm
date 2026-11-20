//! Qwen3.5 MoE (hybrid Gated-DeltaNet / gated full attention) forward pass.

/// Opt-in (`BURN_PROF=1`) per-phase device-synchronized timing of `forward`.
pub struct Prof {
    on: bool,
    t: Instant,
    acc: [f64; PROF_N],
}

const PROF_N: usize = 8;
const PROF_NAMES: [&str; PROF_N] = [
    "gdn",
    "attn",
    "ffn/route",
    "shared",
    "experts",
    "upload",
    "cpu",
    "head",
];

impl Prof {
    fn new() -> Self {
        Self {
            on: std::env::var_os("BURN_PROF").is_some(),
            t: Instant::now(),
            acc: [0.0; PROF_N],
        }
    }

    /// Wait for the device, then charge the time since the previous tick to `cat`.
    fn tick(&mut self, gpu: &Gpu, cat: usize) {
        if self.on {
            gpu.sync();
            self.acc[cat] += self.t.elapsed().as_secs_f64();
            self.t = Instant::now();
        }
    }

    pub fn summary(&self) -> Option<String> {
        self.on.then(|| {
            let total: f64 = self.acc.iter().sum();
            let mut s = format!("profile (total {total:.2}s):");
            for (n, a) in PROF_NAMES.iter().zip(self.acc) {
                s += &format!(" {n} {a:.2}s");
            }
            s
        })
    }
}

use crate::cpu_moe::{self, CpuDecodeJob, CpuExpertJob};
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
    /// FFN width of dense (non-MoE) models; 0 for MoE.
    pub dense_inter: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.meta_str("general.architecture")?;
        let moe = arch == "qwen35moe";
        ensure!(
            moe || arch == "qwen35",
            "unsupported architecture {arch} (want qwen35 or qwen35moe)"
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
            n_experts: if moe { k("expert_count")? } else { 0 },
            topk: if moe { k("expert_used_count")? } else { 0 },
            expert_inter: if moe {
                k("expert_feed_forward_length")?
            } else {
                0
            },
            shared_inter: if moe {
                k("expert_shared_feed_forward_length")?
            } else {
                0
            },
            conv_k: k("ssm.conv_kernel")?,
            ssm_dk: k("ssm.state_size")?,
            ssm_nk: k("ssm.group_count")?,
            ssm_nv: k("ssm.time_step_rank")?,
            full_attn_interval: k("full_attention_interval")?,
            dense_inter: if moe { 0 } else { k("feed_forward_length")? },
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
        // Codebook formats keep one shared codebook in w1 (indexed absolutely by the
        // kernels), so w1 is not slot-pooled.
        let codebook = info.ty.codebook();
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
                match &codebook {
                    Some(cb) => gpu.upload_u32(cb),
                    None => gpu.alloc((w1_per * slots).max(1)),
                },
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
    freq: Vec<u32>,
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

    /// Repack a single expert on the host (for overlapped promotion before enqueueing DMA).
    fn repack_one(&self, g: &Gguf, e: usize) -> [quant::Packed; 3] {
        [
            quant::repack(self.gate.info.ty, self.gate.raw(g, e)),
            quant::repack(self.up.info.ty, self.up.raw(g, e)),
            quant::repack(self.down.info.ty, self.down.raw(g, e)),
        ]
    }

    /// Enqueue device writes for a pre-repacked expert and update slot residency tables.
    fn commit_one(&mut self, gpu: &Gpu, e: usize, slot: usize, packed: [quant::Packed; 3]) {
        let tensors = [&self.gate, &self.up, &self.down];
        let mut writes = Vec::with_capacity(6);
        for (ti, p) in packed.into_iter().enumerate() {
            let t = tensors[ti];
            writes.push((&t.w0, slot * t.w0_per, p.w0));
            if t.w1_per > 0 {
                writes.push((&t.w1, slot * t.w1_per, p.w1));
            }
        }
        gpu.write_many(writes);
        if self.expert_in[slot] >= 0 {
            self.slot_of[self.expert_in[slot] as usize] = -1;
        }
        self.expert_in[slot] = e as i32;
        self.slot_of[e] = slot as i32;
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

struct DenseW {
    gate: QWeight,
    up: QWeight,
    down: QWeight,
}

struct Layer {
    attn_norm: Buf,
    post_norm: Buf,
    mixer: Mixer,
}

pub struct Model {
    pub cfg: Config,
    pub gpu: Gpu,
    gguf: Gguf,
    layers: Vec<Layer>,
    /// Per-layer routed-MoE weights (empty for dense models).
    moes: Vec<MoeW>,
    /// Per-layer dense FFN weights (empty for MoE models).
    denses: Vec<DenseW>,
    out_norm: Buf,
    lm_head: QWeight,
    embd: TensorInfo,
    /// Rows at or above which dense layers dequantize + use the tiled GEMM.
    pub gemm_min_m: usize,
    clock: u64,
    pub stats: Stats,
    pub prof: Prof,
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub expert_hits: u64,
    pub expert_misses: u64,
    pub promotions: u64,
    /// Host time spent repacking + enqueueing expert uploads.
    pub upload_secs: f64,
    /// Host time blocked reading routing decisions back from the device.
    pub route_sync_secs: f64,
    /// Host time computing missed experts on CPU (overlapped with GPU).
    pub cpu_secs: f64,
}

impl Stats {
    pub fn since(&self, earlier: &Stats) -> Stats {
        Stats {
            expert_hits: self.expert_hits - earlier.expert_hits,
            expert_misses: self.expert_misses - earlier.expert_misses,
            promotions: self.promotions - earlier.promotions,
            upload_secs: self.upload_secs - earlier.upload_secs,
            route_sync_secs: self.route_sync_secs - earlier.route_sync_secs,
            cpu_secs: self.cpu_secs - earlier.cpu_secs,
        }
    }

    /// One-line summary; `None` when no host-routed expert lookups happened.
    pub fn summary(&self) -> Option<String> {
        let lookups = self.expert_hits + self.expert_misses;
        (lookups > 0).then(|| {
            format!(
                "{} lookups, hit rate {:.1}%, cpu {:.2}s, upload {:.2}s ({} promos), routing sync {:.2}s",
                lookups,
                100.0 * self.expert_hits as f64 / lookups as f64,
                self.cpu_secs,
                self.upload_secs,
                self.promotions,
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
        let expert_bytes_per = if cfg.n_experts > 0 {
            expert_bytes_layer / (cfg.n_layers * cfg.n_experts)
        } else {
            0
        };
        let kv_bytes =
            cfg.n_layers / cfg.full_attn_interval * 2 * opts.max_ctx * cfg.n_kv * cfg.head_dim * 4;
        let total_vram = opts
            .vram_budget
            .or(gpu.client.properties().memory.max_memory())
            .unwrap_or(8 << 30) as usize;
        // GPU layouts carry a little padding; scratch for a 512-token prefill chunk.
        let reserve = (dense_bytes as f64 * 1.08) as usize + kv_bytes + (640 << 20);
        let slots = if cfg.n_experts == 0 {
            0
        } else {
            match opts.expert_slots {
                Some(s) => s.clamp(cfg.topk, cfg.n_experts),
                None => {
                    let avail = total_vram.saturating_sub(reserve);
                    let per_slot = (expert_bytes_per as f64 * 1.08) as usize * cfg.n_layers;
                    (avail / per_slot).clamp(cfg.topk, cfg.n_experts)
                }
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
        let mut moes = Vec::new();
        let mut denses = Vec::new();
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
                    if cfg.n_experts == 0 {
                        denses.push(DenseW {
                            gate: upload_weight(gpu_ref, g, &blk(i, "ffn_gate.weight"))?,
                            up: upload_weight(gpu_ref, g, &blk(i, "ffn_up.weight"))?,
                            down: upload_weight(gpu_ref, g, &blk(i, "ffn_down.weight"))?,
                        });
                    } else {
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
                            freq: vec![0; ne],
                        };
                        moes.push(MoeW {
                            router,
                            gate_sh: upload_weight(gpu_ref, g, &blk(i, "ffn_gate_shexp.weight"))?,
                            up_sh: upload_weight(gpu_ref, g, &blk(i, "ffn_up_shexp.weight"))?,
                            down_sh: upload_weight(gpu_ref, g, &blk(i, "ffn_down_shexp.weight"))?,
                            experts,
                        });
                    }
                    layers.push(Layer {
                        attn_norm: upload_f32(gpu_ref, g, &blk(i, "attn_norm.weight"))?,
                        post_norm: upload_f32(gpu_ref, g, &blk(i, "post_attention_norm.weight"))?,
                        mixer,
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
            moes,
            denses,
            out_norm,
            lm_head,
            embd,
            // Measured crossover (`--bench`, RTX 5050): qmatmul is ~18 us per token on
            // 8192x2048 Q8_0, dequant + GEMM ~1.3 ms flat.
            gemm_min_m: 80,
            clock: 0,
            stats: Stats::default(),
            prof: Prof::new(),
        };
        // Fill every slot up front: resident mode loads all experts, cache mode warms up
        // with the first experts.
        for li in 0..model.moes.len() {
            let store = &mut model.moes[li].experts;
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
        let mut pending_promo: Option<(usize, usize, usize)> = None;
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
            self.prof.tick(
                &self.gpu,
                if matches!(self.layers[li].mixer, Mixer::Gdn(_)) {
                    0
                } else {
                    1
                },
            );
            let h2 = self
                .gpu
                .rmsnorm(&x, &self.layers[li].post_norm, t, c.hidden, c.eps);
            if self.cfg.n_experts == 0 {
                let d = &self.denses[li];
                let gm = self.gemm_min_m;
                let g = self.gpu.linear(&d.gate, &h2, t, gm);
                let u = self.gpu.linear(&d.up, &h2, t, gm);
                let a = self.gpu.swiglu(&g, &u, t * c.dense_inter);
                self.gpu.linear_acc(&d.down, &a, t, &x, gm);
            } else {
                self.moe(li, &h2, &x, t, &mut pending_promo)?;
            }
            self.prof.tick(&self.gpu, 2);
        }
        st.pos += t;
        let last = Buf {
            h: x.h.clone().offset_start(((t - 1) * c.hidden * 4) as u64),
            len: c.hidden,
        };
        let xn = self.gpu.rmsnorm(&last, &self.out_norm, 1, c.hidden, c.eps);
        let logits = self.gpu.alloc(c.vocab);
        self.gpu.qmatmul(&self.lm_head, &xn, 1, &logits, false);
        // Overlap any final-layer promotion repack with the ~2 ms lm_head matmul.
        if let Some((prev_li, prev_e, prev_slot)) = pending_promo.take() {
            let tl = Instant::now();
            let packed = self.moes[prev_li].experts.repack_one(&self.gguf, prev_e);
            self.moes[prev_li]
                .experts
                .commit_one(&self.gpu, prev_e, prev_slot, packed);
            self.moes[prev_li].experts.stamp[prev_slot] = self.clock;
            self.stats.upload_secs += tl.elapsed().as_secs_f64();
            self.stats.promotions += 1;
        }
        self.prof.tick(&self.gpu, 7);
        Ok(self.gpu.read_f32(&logits))
    }

    fn moe(
        &mut self,
        li: usize,
        h2: &Buf,
        x: &Buf,
        t: usize,
        pending_promo: &mut Option<(usize, usize, usize)>,
    ) -> Result<()> {
        let c = &self.cfg;
        let (ne, topk, inter, hidden) = (c.n_experts, c.topk, c.expert_inter, c.hidden);
        let gm = self.gemm_min_m;
        let npairs = t * topk;

        if self.moes[li].experts.resident(ne) && t <= 2 {
            // Fully resident: no host round-trip, one tile per (token, k) pair.
            let gpu = &self.gpu;
            let w = &self.moes[li];
            let logits = gpu.linear(&w.router, h2, t, gm);
            let (ids, wts) = gpu.moe_route(&logits, t, ne, ne + 1, topk);
            let g = gpu.linear(&w.gate_sh, h2, t, gm);
            let u = gpu.linear(&w.up_sh, h2, t, gm);
            let a = gpu.swiglu(&g, &u, t * c.shared_inter);
            let sh = gpu.linear(&w.down_sh, &a, t, gm);
            let act = gpu.alloc(npairs * inter);
            let y = gpu.alloc(npairs * hidden);
            let desc = gpu.moe_direct_desc(&ids, npairs);
            self.run_experts(li, &desc, npairs, 1, h2, &act, &y);
            self.gpu
                .moe_combine(x, &y, &wts, &sh, &logits, None, t, hidden, topk, ne + 1, ne);
            return Ok(());
        }

        if t <= 64 {
            // Hybrid GPU + CPU path:
            // 1. Route and pack (ids, wts, h2) in a single device-to-host read BEFORE
            //    launching the shared expert, so the sync returns immediately after routing.
            let logits = self.gpu.linear(&self.moes[li].router, h2, t, gm);
            let (_ids, wts, pack) =
                self.gpu
                    .moe_route_pack(&logits, h2, t, ne, ne + 1, hidden, topk);
            // Overlap previous layer's promotion repack with this layer's GPU mixer + router!
            let ready_promo = pending_promo.take().map(|(prev_li, prev_e, prev_slot)| {
                let tl = Instant::now();
                let packed = self.moes[prev_li].experts.repack_one(&self.gguf, prev_e);
                self.stats.upload_secs += tl.elapsed().as_secs_f64();
                (prev_li, prev_e, prev_slot, packed)
            });
            let ts = Instant::now();
            let pack_host = self.gpu.read_u32(&pack);
            self.stats.route_sync_secs += ts.elapsed().as_secs_f64();
            if let Some((prev_li, prev_e, prev_slot, packed)) = ready_promo {
                let tl = Instant::now();
                self.moes[prev_li]
                    .experts
                    .commit_one(&self.gpu, prev_e, prev_slot, packed);
                self.moes[prev_li].experts.stamp[prev_slot] = self.clock;
                self.stats.upload_secs += tl.elapsed().as_secs_f64();
                self.stats.promotions += 1;
            }

            let ids_host = &pack_host[..npairs];
            let wts_host: &[f32] = bytemuck::cast_slice(&pack_host[npairs..2 * npairs]);
            let h2_host: &[f32] =
                bytemuck::cast_slice(&pack_host[2 * npairs..2 * npairs + t * hidden]);

            // 2. Partition routed pairs into VRAM-resident hits (GPU) and misses (CPU).
            if t == 1 {
                let mut hit_desc = [0u32; MOE_MAXM * MOE_DESC];
                let mut hit_tiles = 0usize;
                let mut miss_experts = [0usize; MOE_MAXM];
                let mut miss_weights = [0.0f32; MOE_MAXM];
                let mut n_miss = 0usize;
                let mut hit_wts = [0.0f32; MOE_MAXM];

                {
                    let st = &mut self.moes[li].experts;
                    for (p, &eu) in ids_host.iter().enumerate() {
                        let e = eu as usize;
                        self.clock += 1;
                        st.freq[e] = st.freq[e].saturating_add(1);
                        if st.slot_of[e] >= 0 {
                            let slot = st.slot_of[e] as usize;
                            st.stamp[slot] = self.clock;
                            self.stats.expert_hits += 1;
                            let off = hit_tiles * MOE_DESC;
                            hit_desc[off] = slot as u32;
                            hit_desc[off + 1] = 1;
                            hit_desc[off + 2] = p as u32;
                            hit_tiles += 1;
                            hit_wts[p] = wts_host[p];
                        } else {
                            self.stats.expert_misses += 1;
                            miss_experts[n_miss] = e;
                            miss_weights[n_miss] = wts_host[p];
                            n_miss += 1;
                        }
                    }
                }

                // 3. Immediately enqueue shared expert + hit experts on the GPU so they run
                //    in parallel with CPU execution of the missed experts.
                let w = &self.moes[li];
                let g = self.gpu.linear(&w.gate_sh, h2, 1, gm);
                let u = self.gpu.linear(&w.up_sh, h2, 1, gm);
                let a = self.gpu.swiglu(&g, &u, c.shared_inter);
                let sh = self.gpu.linear(&w.down_sh, &a, 1, gm);

                let y = self.gpu.alloc(npairs * hidden);
                if hit_tiles > 0 {
                    let act = self.gpu.alloc(npairs * inter);
                    let desc_buf = self.gpu.upload_u32(&hit_desc[..hit_tiles * MOE_DESC]);
                    self.run_experts(li, &desc_buf, hit_tiles, 1, h2, &act, &y);
                }

                // 4. Compute missed experts on CPU while GPU runs shared + hit experts.
                if n_miss > 0 {
                    let st = &self.moes[li].experts;
                    let mut jobs_arr = [CpuDecodeJob {
                        gate_raw: &[],
                        up_raw: &[],
                        down_raw: &[],
                        weight: 0.0,
                    }; MOE_MAXM];
                    for i in 0..n_miss {
                        let e = miss_experts[i];
                        jobs_arr[i] = CpuDecodeJob {
                            gate_raw: st.gate.raw(&self.gguf, e),
                            up_raw: st.up.raw(&self.gguf, e),
                            down_raw: st.down.raw(&self.gguf, e),
                            weight: miss_weights[i],
                        };
                    }
                    let tc = Instant::now();
                    let mut cpu_pack = vec![0.0f32; topk + hidden];
                    cpu_pack[..topk].copy_from_slice(&hit_wts[..topk]);
                    cpu_moe::run_experts_cpu_decode(
                        st.gate.info.ty,
                        st.up.info.ty,
                        st.down.info.ty,
                        &jobs_arr[..n_miss],
                        h2_host,
                        hidden,
                        inter,
                        &mut cpu_pack[topk..],
                    );
                    self.stats.cpu_secs += tc.elapsed().as_secs_f64();
                    let cpu_pack_buf = self.gpu.upload_f32(&cpu_pack);
                    self.gpu.moe_combine(
                        x,
                        &y,
                        &wts,
                        &sh,
                        &logits,
                        Some(&cpu_pack_buf),
                        1,
                        hidden,
                        topk,
                        ne + 1,
                        ne,
                    );

                    // 5. LFU cache promotion: defer repack to next layer's GPU mixer window.
                    let st = &self.moes[li].experts;
                    if let Some(&best_e) =
                        miss_experts[..n_miss].iter().max_by_key(|&&e| st.freq[e])
                    {
                        let best_f = st.freq[best_e];
                        let victim = (0..st.slots)
                            .min_by_key(|&s| {
                                let cur = st.expert_in[s];
                                let f = if cur >= 0 { st.freq[cur as usize] } else { 0 };
                                (f, st.stamp[s])
                            })
                            .unwrap();
                        let cur = st.expert_in[victim];
                        let cur_f = if cur >= 0 { st.freq[cur as usize] } else { 0 };
                        let should_promote = (cur_f == 0 && best_f >= 2) || (best_f >= cur_f + 6);
                        if should_promote {
                            *pending_promo = Some((li, best_e, victim));
                        }
                    }
                } else {
                    self.gpu.moe_combine(
                        x,
                        &y,
                        &wts,
                        &sh,
                        &logits,
                        None,
                        1,
                        hidden,
                        topk,
                        ne + 1,
                        ne,
                    );
                }
                return Ok(());
            }

            // Multi-token (2..=64) hybrid path.
            let mut by_expert: Vec<Vec<u32>> = vec![Vec::new(); ne];
            for (p, &e) in ids_host.iter().enumerate() {
                by_expert[e as usize].push(p as u32);
            }
            let mut hit_desc = Vec::new();
            let mut max_cnt = 0usize;
            let mut miss_experts: Vec<usize> = Vec::new();
            let mut hit_wts: Vec<f32> = Vec::new();

            {
                let st = &mut self.moes[li].experts;
                for e in 0..ne {
                    let pairs = &by_expert[e];
                    if pairs.is_empty() {
                        continue;
                    }
                    self.clock += 1;
                    st.freq[e] = st.freq[e].saturating_add(pairs.len() as u32);
                    if st.slot_of[e] >= 0 {
                        let slot = st.slot_of[e] as usize;
                        st.stamp[slot] = self.clock;
                        self.stats.expert_hits += pairs.len() as u64;
                        for chunk in pairs.chunks(MOE_MAXM) {
                            max_cnt = max_cnt.max(chunk.len());
                            hit_desc.push(slot as u32);
                            hit_desc.push(chunk.len() as u32);
                            hit_desc
                                .extend((0..MOE_MAXM).map(|i| chunk.get(i).copied().unwrap_or(0)));
                        }
                    } else {
                        self.stats.expert_misses += pairs.len() as u64;
                        miss_experts.push(e);
                    }
                }
            }

            // 3. Immediately enqueue shared expert + hit experts on the GPU so they run
            //    in parallel with CPU execution of the missed experts.
            let w = &self.moes[li];
            let g = self.gpu.linear(&w.gate_sh, h2, t, gm);
            let u = self.gpu.linear(&w.up_sh, h2, t, gm);
            let a = self.gpu.swiglu(&g, &u, t * c.shared_inter);
            let sh = self.gpu.linear(&w.down_sh, &a, t, gm);

            let y = self.gpu.alloc(npairs * hidden);
            if !hit_desc.is_empty() {
                let act = self.gpu.alloc(npairs * inter);
                let ntiles = hit_desc.len() / MOE_DESC;
                let desc_buf = self.gpu.upload_u32(&hit_desc);
                self.run_experts(li, &desc_buf, ntiles, max_cnt, h2, &act, &y);
            }

            // 4. Compute missed experts on CPU while GPU runs shared + hit experts.
            if !miss_experts.is_empty() {
                hit_wts.extend_from_slice(wts_host);
                let st = &self.moes[li].experts;
                let jobs: Vec<CpuExpertJob<'_>> = miss_experts
                    .iter()
                    .map(|&e| {
                        let pairs: Vec<(usize, f32)> = by_expert[e]
                            .iter()
                            .map(|&p| {
                                hit_wts[p as usize] = 0.0;
                                ((p as usize) / topk, wts_host[p as usize])
                            })
                            .collect();
                        CpuExpertJob {
                            gate_raw: st.gate.raw(&self.gguf, e),
                            up_raw: st.up.raw(&self.gguf, e),
                            down_raw: st.down.raw(&self.gguf, e),
                            pairs,
                        }
                    })
                    .collect();
                let tc = Instant::now();
                let mut cpu_pack = vec![0.0f32; npairs + t * hidden];
                cpu_pack[..npairs].copy_from_slice(&hit_wts);
                cpu_moe::run_experts_cpu(
                    st.gate.info.ty,
                    st.up.info.ty,
                    st.down.info.ty,
                    &jobs,
                    h2_host,
                    t,
                    hidden,
                    inter,
                    &mut cpu_pack[npairs..],
                );
                self.stats.cpu_secs += tc.elapsed().as_secs_f64();
                let cpu_pack_buf = self.gpu.upload_f32(&cpu_pack);
                self.gpu.moe_combine(
                    x,
                    &y,
                    &wts,
                    &sh,
                    &logits,
                    Some(&cpu_pack_buf),
                    t,
                    hidden,
                    topk,
                    ne + 1,
                    ne,
                );
            } else {
                self.gpu
                    .moe_combine(x, &y, &wts, &sh, &logits, None, t, hidden, topk, ne + 1, ne);
            }
            return Ok(());
        }

        // Large-batch prefill (t > 64): stream missed experts onto GPU in slot-sized rounds.
        self.prof.tick(&self.gpu, 2);
        let gpu = &self.gpu;
        let w = &self.moes[li];
        let logits = gpu.linear(&w.router, h2, t, gm);
        let (ids, wts) = gpu.moe_route(&logits, t, ne, ne + 1, topk);
        self.prof.tick(gpu, 2);
        let g = gpu.linear(&w.gate_sh, h2, t, gm);
        let u = gpu.linear(&w.up_sh, h2, t, gm);
        let a = gpu.swiglu(&g, &u, t * c.shared_inter);
        let sh = gpu.linear(&w.down_sh, &a, t, gm);
        self.prof.tick(gpu, 3);
        let act = gpu.alloc(npairs * inter);
        let y = gpu.alloc(npairs * hidden);

        let ts = Instant::now();
        let ids_host = gpu.read_u32(&ids);
        self.stats.route_sync_secs += ts.elapsed().as_secs_f64();
        let mut by_expert: Vec<Vec<u32>> = vec![Vec::new(); ne];
        for (p, &e) in ids_host.iter().enumerate() {
            by_expert[e as usize].push(p as u32);
        }
        let store = &self.moes[li].experts;
        let mut needed: Vec<usize> = (0..ne).filter(|&e| !by_expert[e].is_empty()).collect();
        needed.sort_by_key(|&e| store.slot_of[e] < 0);
        let slots = store.slots;
        for round in needed.chunks(slots) {
            let st = &self.moes[li].experts;
            let mut pinned = vec![false; slots];
            for &e in round {
                if st.slot_of[e] >= 0 {
                    pinned[st.slot_of[e] as usize] = true;
                }
            }
            let mut desc = Vec::new();
            let mut max_cnt = 0;
            let mut misses = Vec::new();
            let st = &mut self.moes[li].experts;
            for &e in round {
                self.clock += 1;
                st.freq[e] = st.freq[e].saturating_add(by_expert[e].len() as u32);
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
            self.prof.tick(&self.gpu, 5);
            let ntiles = desc.len() / MOE_DESC;
            let desc = self.gpu.upload_u32(&desc);
            self.run_experts(li, &desc, ntiles, max_cnt, h2, &act, &y);
            self.prof.tick(&self.gpu, 4);
        }
        self.gpu
            .moe_combine(x, &y, &wts, &sh, &logits, None, t, hidden, topk, ne + 1, ne);
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
        let st = &self.moes[li].experts;
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
