//! Inkling-2 (`inkling2_loop`) GGUF inventory validation.
//!
//! The APEX checkpoints use a 24-layer looped MoE architecture that shares no tensor names
//! with Qwen3.5. Before a forward pass can be implemented, every tensor family must be
//! accounted for; this module checks the real artifact's inventory against the published
//! W0 tensor census so loading failures point at a specific missing family.

use crate::gguf::{Gguf, TensorInfo};
use anyhow::{Result, ensure};

/// Tensor dimensions in gguf order (`dims[0]` is the contiguous dimension).
fn dims<'a>(g: &'a Gguf, name: &str, want: &[u64]) -> Result<&'a TensorInfo> {
    let t = g.tensor(name)?;
    ensure!(
        t.dims.as_slice() == want,
        "{name}: dims {:?}, expected {want:?}",
        t.dims
    );
    Ok(t)
}

/// Validate the global and per-layer tensor inventory of an Inkling-2 checkpoint.
pub fn validate(g: &Gguf) -> Result<()> {
    ensure!(
        g.meta_str("general.architecture")? == "inkling2_loop",
        "not an inkling2_loop GGUF"
    );
    let n_layers = g.meta_u64("inkling2_loop.block_count")? as usize;
    let hidden = g.meta_u64("inkling2_loop.embedding_length")? as usize as u64;
    let vocab = g.meta_u64("inkling2_loop.vocab_size")? as usize as u64;
    let n_experts = g.meta_u64("inkling2_loop.expert_count")? as usize as u64;

    dims(g, "token_embd.weight", &[hidden, vocab])?;
    dims(g, "output.weight", &[hidden, vocab])?;
    dims(g, "output_norm.weight", &[hidden])?;

    for i in 0..n_layers {
        let b = format!("blk.{i}.");
        let has = |s: &str| g.tensors.contains_key(&format!("{b}{s}"));
        // Attention: 24 q/gate heads x 128, 8 kv heads x 128, plus relative-position bias.
        dims(g, &format!("{b}attn_q.weight"), &[hidden, 3072])?;
        dims(g, &format!("{b}attn_gate.weight"), &[hidden, 3072])?;
        dims(g, &format!("{b}attn_k.weight"), &[hidden, 1024])?;
        dims(g, &format!("{b}attn_v.weight"), &[hidden, 1024])?;
        dims(g, &format!("{b}attn_output.weight"), &[3072, hidden])?;
        dims(g, &format!("{b}attn_rel.weight"), &[hidden, 384])?;
        dims(g, &format!("{b}attn_rel_bias.weight"), &[16, 1024])?;
        dims(g, &format!("{b}attn_rel_clip.weight"), &[16, 24])?;
        dims(g, &format!("{b}attn_sink_proj.weight"), &[hidden, 24])?;
        dims(g, &format!("{b}attn_q_clip.weight"), &[128, 24])?;
        // Depth-4 shortconv and the MoE router.
        dims(g, &format!("{b}attn_k_conv.weight"), &[4, 1024])?;
        dims(g, &format!("{b}attn_v_conv.weight"), &[4, 1024])?;
        dims(g, &format!("{b}attn_post_conv.weight"), &[4, hidden])?;
        dims(g, &format!("{b}ffn_post_conv.weight"), &[4, hidden])?;
        dims(g, &format!("{b}ffn_gate_inp.weight"), &[hidden, n_experts])?;
        dims(g, &format!("{b}exp_probs_b.bias"), &[n_experts])?;
        let gu = g.tensor(&format!("{b}ffn_gate_up_exps.weight"))?;
        ensure!(
            gu.dims.as_slice() == [hidden, 2 * 1152, n_experts],
            "{b}ffn_gate_up_exps.weight: unexpected dims {:?}",
            gu.dims
        );
        let down = g.tensor(&format!("{b}ffn_down_exps.weight"))?;
        ensure!(
            down.dims.as_slice() == [1152, hidden, n_experts],
            "{b}ffn_down_exps.weight: unexpected dims {:?}",
            down.dims
        );
        // Full-attention layers (5, 11, 17, 23) carry a 3-slot residual query;
        // layer 0 has no attention residual at all.
        if i == 0 {
            ensure!(
                !has("attn_res_query.weight"),
                "blk.0: unexpected attention residual query"
            );
        } else if i % 6 == 5 {
            dims(g, &format!("{b}attn_res_query.weight"), &[hidden, 3])?;
        } else {
            dims(g, &format!("{b}attn_res_query.weight"), &[hidden])?;
        }
        dims(g, &format!("{b}ffn_res_query.weight"), &[hidden])?;
        // Loop-core layers 6..17 repeat three passes with rank-64 gated controllers.
        if (6..18).contains(&i) {
            for stream in ["q", "k", "v", "ffn"] {
                dims(
                    g,
                    &format!("{b}loop_{stream}_ctrl_down.weight"),
                    &[4608, 64],
                )?;
                dims(
                    g,
                    &format!("{b}loop_{stream}_ctrl_forget.weight"),
                    &[64, hidden],
                )?;
                dims(
                    g,
                    &format!("{b}loop_{stream}_ctrl_gain.weight"),
                    &[64, hidden],
                )?;
            }
            dims(g, &format!("{b}loop_msg_query.weight"), &[hidden, 4])?;
        } else {
            ensure!(
                !has("loop_msg_query.weight"),
                "{b}: unexpected loop controllers"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Gguf;
    use std::path::Path;

    #[test]
    fn apex_balanced_inventory() {
        let path =
            Path::new("/mnt/models/inkling2-mini-apex-gguf/Inkling-2-Mini-APEX-Balanced.gguf");
        if !path.exists() {
            eprintln!("skipping: APEX GGUF not present");
            return;
        }
        let g = Gguf::open(path).unwrap();
        validate(&g).unwrap();
    }
}
