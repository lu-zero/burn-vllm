//! One generation turn on a loaded model, shared by the CLI and the server: prefill with
//! RadixAttention prefix reuse, then decode greedily, by sampling, or with MTP speculation.

use crate::{
    model::{Model, State, Stats},
    sampler,
};
use anyhow::Result;
use std::{fmt::Write, time::Instant};

/// Chat-ML prompt that ends where the model writes its answer. `think` pre-opens a
/// `<think>` block; otherwise the block is closed empty.
pub fn chat_prompt<'a>(
    messages: impl IntoIterator<Item = (&'a str, &'a str)>,
    think: bool,
) -> String {
    let mut text = String::new();
    for (role, content) in messages {
        let _ = write!(text, "<|im_start|>{role}\n{content}<|im_end|>\n");
    }
    text.push_str("<|im_start|>assistant\n");
    text.push_str(if think {
        "<think>\n"
    } else {
        "<think>\n\n</think>\n\n"
    });
    text
}

pub struct Params {
    pub max_new: usize,
    /// Prefill chunk size in tokens.
    pub chunk: usize,
    pub sampling: sampler::Params,
    /// MTP draft tokens per step (0 = plain decoding). Only used if the model has an MTP head.
    pub spec_k: usize,
    /// Tokens that end generation and are not emitted.
    pub stop: Vec<u32>,
}

#[derive(Default)]
pub struct SpecStats {
    pub steps: usize,
    pub drafted: usize,
    pub accepted: usize,
}

/// Decode path chosen for one turn. The requested draft count is clamped to zero when the
/// model has no MTP head; the chosen mode then drives both prefill and decode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DecodeMode {
    Speculative(usize),
    Greedy,
    Sampled,
}

/// Speculative decoding verifies the pending token plus `k` drafts in one pass, so it must
/// reserve `k + 1` context slots ahead of the plain decode paths.
fn has_room_for_verify(st: &State, k: usize) -> bool {
    st.pos + k + 1 < st.max_ctx
}

#[derive(Default)]
pub struct Outcome {
    pub radix_hit: usize,
    pub generated: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub prefill_stats: Stats,
    pub decode_stats: Stats,
    pub spec: Option<SpecStats>,
}

/// Prefill `prompt` into `st` (a fresh state from `Model::new_state`), then decode until a stop
/// token, `p.max_new` tokens, or the context fills. `on_token` receives each emitted token and
/// returns `true` to stop early.
pub fn generate(
    model: &mut Model,
    st: &mut State,
    prompt: &[u32],
    p: &Params,
    mut on_token: impl FnMut(u32) -> Result<bool>,
) -> Result<Outcome> {
    let spec_k = if model.has_mtp() { p.spec_k } else { 0 };
    let mut sampler = sampler::Sampler::new(p.sampling.clone());
    let mode = if spec_k > 0 {
        DecodeMode::Speculative(spec_k)
    } else if sampler.is_greedy() {
        DecodeMode::Greedy
    } else {
        DecodeMode::Sampled
    };
    let stats_before = model.stats.clone();

    let t0 = Instant::now();
    let (mut logits, mut h_prev, radix_hit) = if matches!(mode, DecodeMode::Speculative(_)) {
        let (l, h, hit) = model.prefill_spec(prompt, p.chunk, st)?;
        (l, Some(h), hit)
    } else {
        let (l, hit) = model.prefill_radix(prompt, p.chunk, st)?;
        (l, None, hit)
    };
    let prefill_secs = t0.elapsed().as_secs_f64();
    let stats0 = model.stats.clone();

    let t1 = Instant::now();
    let mut generated = 0usize;
    let mut emit = |next: u32| -> Result<bool> {
        if p.stop.contains(&next) {
            return Ok(true);
        }
        generated += 1;
        on_token(next)
    };
    let mut spec = SpecStats::default();
    match mode {
        DecodeMode::Speculative(k) => {
            let mut pending_token = sampler.sample(&logits);
            let mut emitted = 0usize;
            while emitted < p.max_new {
                if emit(pending_token)? {
                    break;
                }
                emitted += 1;
                if emitted >= p.max_new || !has_room_for_verify(st, k) {
                    break;
                }
                let hidden = h_prev
                    .as_ref()
                    .expect("speculative prefill provides a hidden state");
                let out = model.spec_step(st, pending_token, hidden, k, &mut sampler)?;
                spec.steps += 1;
                spec.drafted += out.drafted;
                spec.accepted += out.accepted.len();
                let mut stopped_while_drafting = false;
                for d in out.accepted {
                    if emitted >= p.max_new || emit(d)? {
                        stopped_while_drafting = true;
                        break;
                    }
                    emitted += 1;
                }
                if stopped_while_drafting {
                    break;
                }
                pending_token = out.next;
                h_prev = Some(out.hidden);
            }
        }
        DecodeMode::Greedy => {
            let mut next = sampler.sample(&logits);
            for _ in 0..p.max_new {
                if emit(next)? || st.pos >= st.max_ctx {
                    break;
                }
                next = model.forward_greedy(&[next], st)?;
            }
        }
        DecodeMode::Sampled => {
            for _ in 0..p.max_new {
                let next = sampler.sample(&logits);
                if emit(next)? || st.pos >= st.max_ctx {
                    break;
                }
                logits = model.forward(&[next], st)?;
            }
        }
    }
    drop(emit);

    Ok(Outcome {
        radix_hit,
        generated,
        prefill_secs,
        decode_secs: t1.elapsed().as_secs_f64(),
        prefill_stats: stats0.since(&stats_before),
        decode_stats: model.stats.since(&stats0),
        spec: matches!(mode, DecodeMode::Speculative(_)).then_some(spec),
    })
}
