mod cpu_moe;
mod gguf;
mod gpu;
mod iq_grids;
mod kernels;
mod model;
mod quant;
mod sampler;
mod selftest;
mod tokenizer;

use anyhow::{Context, Result, bail};
use burn_cubecl::cubecl;
use std::{io::Write, path::PathBuf, time::Instant};

const DEFAULT_MODEL: &str = "/home/theo/.cache/huggingface/hub/models--unsloth--Qwen3.5-35B-A3B-GGUF/snapshots/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-MXFP4_MOE.gguf";

const USAGE: &str = "usage: burn-vllm [options] [--prompt TEXT]
  --model PATH          GGUF file (qwen35moe)
  --backend cuda|wgpu   runtime (default: cuda if built, else wgpu)
  --prompt TEXT         user message (default: a short question)
  --raw                 feed the prompt verbatim (no chat template)
  --no-think            pre-fill an empty <think></think> block
  -n N                  max new tokens (default 256)
  --temp F --top-k N --top-p F --seed N   sampling (temp 0 = greedy)
  --ctx N               KV-cache capacity in tokens (default 8192)
  --chunk N             prefill chunk size (default 512)
  --vram-gb F           VRAM budget used to size the expert cache
  --expert-slots N      experts kept on device per layer (256 = all resident)
  --selftest            check GPU kernels against CPU references and exit
  --bench               time decode-shaped kernels and exit";

struct Args {
    model: PathBuf,
    backend: Option<String>,
    prompt: String,
    raw: bool,
    think: bool,
    n: usize,
    sampling: sampler::Params,
    ctx: usize,
    chunk: usize,
    vram_gb: Option<f64>,
    expert_slots: Option<usize>,
    selftest: bool,
    bench: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: PathBuf::from(DEFAULT_MODEL),
        backend: None,
        prompt: "Explain in two sentences why the sky is blue.".into(),
        raw: false,
        think: true,
        n: 256,
        sampling: sampler::Params::default(),
        ctx: 8192,
        chunk: 512,
        vram_gb: None,
        expert_slots: None,
        selftest: false,
        bench: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--model" => a.model = val()?.into(),
            "--backend" => a.backend = Some(val()?),
            "--prompt" | "-p" => a.prompt = val()?,
            "--raw" => a.raw = true,
            "--no-think" => a.think = false,
            "-n" => a.n = val()?.parse()?,
            "--temp" => a.sampling.temperature = val()?.parse()?,
            "--top-k" => a.sampling.top_k = val()?.parse()?,
            "--top-p" => a.sampling.top_p = val()?.parse()?,
            "--seed" => a.sampling.seed = val()?.parse()?,
            "--ctx" => a.ctx = val()?.parse()?,
            "--chunk" => a.chunk = val()?.parse()?,
            "--vram-gb" => a.vram_gb = Some(val()?.parse()?),
            "--expert-slots" => a.expert_slots = Some(val()?.parse()?),
            "--selftest" => a.selftest = true,
            "--bench" => a.bench = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}\n{USAGE}"),
        }
    }
    Ok(a)
}

fn device(backend: Option<&str>) -> Result<cubecl::Device> {
    Ok(match backend {
        #[cfg(feature = "cuda")]
        Some("cuda") | None => cubecl::Device::Cuda(Default::default()),
        #[cfg(feature = "wgpu")]
        Some("wgpu") => cubecl::Device::Wgpu(Default::default()),
        #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
        None => cubecl::Device::Wgpu(Default::default()),
        Some(b) => bail!("backend {b} not available in this build"),
    })
}

/// Persist compiled kernels across runs (off by default in cubecl). Without it every
/// process recompiles each matmul variant (~0.3-0.4 s apiece), which dominated short
/// prefills. `cubecl.toml` and `CUBECL_*` overrides still apply.
fn enable_kernel_cache() {
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    let mut cfg = CubeClRuntimeConfig::from_current_dir().override_from_env();
    cfg.compilation.cache = true;
    CubeClRuntimeConfig::try_set(cfg);
}

fn main() -> Result<()> {
    let args = parse_args()?;
    env_logger::init();
    enable_kernel_cache();
    let gpu = gpu::Gpu::new(device(args.backend.as_deref())?);
    eprintln!("device: {} (plane {})", gpu.client.name(), gpu.plane);
    let gguf = gguf::Gguf::open(&args.model)?;
    if args.selftest {
        return selftest::run(&gpu, &gguf);
    }
    if args.bench {
        return selftest::bench(&gpu, &gguf);
    }
    let tok = tokenizer::Tokenizer::from_gguf(&gguf)?;
    let opts = model::LoadOptions {
        expert_slots: args.expert_slots,
        vram_budget: args.vram_gb.map(|g| (g * (1u64 << 30) as f64) as u64),
        max_ctx: args.ctx,
    };
    let mut model = model::Model::load(gpu, gguf, &opts)?;

    let text = if args.raw {
        args.prompt.clone()
    } else {
        let think = if args.think {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        };
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{think}",
            args.prompt
        )
    };
    let prompt = tok.encode(&text)?;
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_id(s))
        .chain([tok.eos])
        .collect();
    eprintln!("prompt: {} tokens", prompt.len());

    let mut st = model.new_state(args.ctx);
    let mut sampler = sampler::Sampler::new(args.sampling.clone());
    let t0 = Instant::now();
    let mut logits = Vec::new();
    for chunk in prompt.chunks(args.chunk.max(1)) {
        logits = model.forward(chunk, &mut st)?;
    }
    let prefill = t0.elapsed().as_secs_f64();
    let stats0 = model.stats.clone();

    print!("{text}");
    let mut out_bytes = Vec::new();
    let mut printed = 0;
    let t1 = Instant::now();
    let mut generated = 0;
    for _ in 0..args.n {
        let next = sampler.sample(&logits);
        if stop.contains(&next) {
            break;
        }
        generated += 1;
        out_bytes.extend(tok.token_bytes(next));
        // Print the longest valid UTF-8 prefix not yet printed.
        let valid = match std::str::from_utf8(&out_bytes[printed..]) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        print!(
            "{}",
            std::str::from_utf8(&out_bytes[printed..printed + valid]).unwrap()
        );
        std::io::stdout().flush()?;
        printed += valid;
        if st.pos >= args.ctx {
            break;
        }
        logits = model.forward(&[next], &mut st)?;
    }
    println!();
    let decode = t1.elapsed().as_secs_f64();
    let s = &model.stats;
    eprintln!(
        "prefill: {} tok in {:.2}s ({:.1} tok/s) | decode: {} tok in {:.2}s ({:.2} tok/s)",
        prompt.len(),
        prefill,
        prompt.len() as f64 / prefill,
        generated,
        decode,
        generated as f64 / decode,
    );
    for (phase, st) in [("prefill", stats0.clone()), ("decode", s.since(&stats0))] {
        if let Some(line) = st.summary() {
            eprintln!("{phase} experts: {line}");
        }
    }
    if let Some(p) = model.prof.summary() {
        eprintln!("{p}");
    }
    Ok(())
}
