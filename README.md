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
`--expert-slots N` (per layer; 256 = all resident), `--mtp K`, `--no-think`, `--raw`.

### OpenAI-compatible server

```sh
./target/release/burn-vllm --serve 127.0.0.1:8000 --temp 0.7
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":64,"stream":true}'
```

Endpoints: `POST /v1/chat/completions` (streaming and not, `stop`, `max_tokens` /
`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stream_options.include_usage`,
`chat_template_kwargs.enable_thinking`), `POST /v1/completions` (raw prompt, no chat template),
`GET /v1/models`. The `model` field is accepted and ignored; the served name is the GGUF's file
name. Limits: one sequence at a time (requests queue), `n` must be 1, and `tools` are ignored.
With thinking on (the default), the model's reasoning is returned in `reasoning_content` (chat
message and stream deltas) and the answer in `content`; the `<think>` tags are not returned.

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
| `src/generate.rs` | one generation turn (prefill, greedy / sampled / MTP decode) shared by the CLI and server; chat template |
| `src/server.rs` | OpenAI-compatible HTTP API on axum; one engine thread owns the model |
