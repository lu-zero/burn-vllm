# burn-vllm

A Rust/[CubeCL](https://github.com/tracel-ai/cubecl) port of the Qwen3.5 MoE path of
`vllm.cpp`, built on `burn-cubecl` 0.23.0-pre.2. No nvcc: kernels are written in CubeCL and
JIT-compiled at runtime (NVRTC on CUDA, WGSL/SPIR-V on wgpu), so the build is a plain
`cargo build` and the same kernels run on NVIDIA, AMD, Intel and Apple GPUs.

## Status

- Model: `qwen35moe` GGUF (Qwen3.5-35B-A3B): 30 Gated-DeltaNet layers + 10 gated
  full-attention layers, 256-expert top-8 MoE with a gated shared expert.
- Weight formats read directly from the GGUF: F32, Q8_0, MXFP4, Q5_K, Q6_K.
- Single sequence, chunked prefill, contiguous f32 KV cache (paged attention is not ported
  yet).
- Experts larger than VRAM: each layer keeps an LRU pool of expert slots on the device and
  refills misses from the mmap'd GGUF. With enough VRAM every expert is resident and decode
  never syncs with the host for routing.

## Usage

```sh
cargo build --release
./target/release/burn-vllm --prompt "Why is the sky blue?" -n 200
./target/release/burn-vllm --backend wgpu --prompt "..."
./target/release/burn-vllm --selftest     # GPU kernels vs CPU references on real tensors
```

Options: `--model PATH`, `--temp/--top-k/--top-p/--seed` (`--temp 0` = greedy), `--ctx N`,
`--chunk N` (prefill chunk), `--vram-gb F` (budget used to size the expert cache),
`--expert-slots N` (per layer; 256 = all resident), `--no-think`, `--raw`.

## Layout

| File | Contents |
|---|---|
| `src/gguf.rs` | mmap GGUF reader |
| `src/quant.rs` | ggml block formats: CPU dequant + repack into 4-byte-aligned GPU layouts |
| `src/kernels.rs` | CubeCL kernels: quantized matmul / dequant, RMSNorm, GDN conv + recurrence, RoPE + gated attention, MoE routing / grouped experts / combine |
| `src/gpu.rs` | buffers, launch wrappers, dequant + burn tiled GEMM for prefill |
| `src/model.rs` | Qwen3.5 MoE weights, expert residency, forward pass |
| `src/tokenizer.rs` | byte-level BPE from GGUF metadata (qwen35 pre-tokenizer) |
| `src/selftest.rs` | kernel correctness checks |
