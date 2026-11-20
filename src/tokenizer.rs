//! Byte-level BPE tokenizer built from GGUF `tokenizer.ggml.*` metadata (gpt2 model,
//! qwen35 pre-tokenizer).

use crate::gguf::Gguf;
use anyhow::{Context, Result, bail};
use fancy_regex::Regex;
use std::collections::HashMap;

const QWEN35_PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    tokens: Vec<String>,
    vocab: HashMap<String, u32>,
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Special (control / user-defined) tokens, longest first, matched verbatim.
    specials: Vec<(String, u32)>,
    is_special: Vec<bool>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pre: Regex,
    pub eos: u32,
}

fn bytes_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (b'!' as u32..=b'~' as u32).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    map
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let model = g.meta_str("tokenizer.ggml.model")?;
        if model != "gpt2" {
            bail!("unsupported tokenizer model {model}");
        }
        let pre = g.meta_str("tokenizer.ggml.pre").unwrap_or("default");
        if pre != "qwen35" && pre != "qwen2" {
            eprintln!("warning: pre-tokenizer {pre:?} not recognized, using qwen35 regex");
        }
        let tokens: Vec<String> = g
            .meta_array("tokenizer.ggml.tokens")?
            .iter()
            .map(|v| v.as_str().map(str::to_owned).context("token not a string"))
            .collect::<Result<_>>()?;
        let types: Vec<u64> = g
            .meta_array("tokenizer.ggml.token_type")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(1))
            .collect();
        let vocab: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let mut merges = HashMap::new();
        for (rank, m) in g.meta_array("tokenizer.ggml.merges")?.iter().enumerate() {
            let m = m.as_str().context("merge not a string")?;
            let (a, b) = m.split_once(' ').context("bad merge")?;
            let (Some(&ia), Some(&ib), Some(&im)) =
                (vocab.get(a), vocab.get(b), vocab.get(&format!("{a}{b}")))
            else {
                continue;
            };
            merges.entry((ia, ib)).or_insert((rank as u32, im));
        }
        let is_special: Vec<bool> = types.iter().map(|&t| t == 3 || t == 4).collect();
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| is_special[*i] && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let byte_to_char = bytes_to_unicode();
        let char_to_byte = byte_to_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        let eos = g.meta_u64("tokenizer.ggml.eos_token_id")? as u32;
        Ok(Self {
            tokens,
            vocab,
            merges,
            specials,
            is_special,
            byte_to_char,
            char_to_byte,
            pre: Regex::new(QWEN35_PRE)?,
            eos,
        })
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Earliest special-token occurrence (longest wins at equal positions).
            let mut best: Option<(usize, &str, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(p) = rest.find(s.as_str())
                    && best.is_none_or(|(bp, bs, _)| p < bp || (p == bp && s.len() > bs.len()))
                {
                    best = Some((p, s, *id));
                }
            }
            let (plain, special) = match best {
                Some((p, s, id)) => (&rest[..p], Some((s.len(), id))),
                None => (rest, None),
            };
            self.encode_plain(plain, &mut out)?;
            match special {
                Some((len, id)) => {
                    out.push(id);
                    rest = &rest[plain.len() + len..];
                }
                None => break,
            }
        }
        Ok(out)
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for m in self.pre.find_iter(text) {
            let piece = m?.as_str();
            let mut syms: Vec<u32> = piece
                .bytes()
                .map(|b| {
                    let c = self.byte_to_char[b as usize];
                    self.vocab
                        .get(c.encode_utf8(&mut [0; 4]) as &str)
                        .copied()
                        .context("byte symbol missing from vocab")
                })
                .collect::<Result<_>>()?;
            loop {
                let best = syms
                    .windows(2)
                    .enumerate()
                    .filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&(r, m)| (r, i, m)))
                    .min();
                let Some((_, i, m)) = best else { break };
                syms[i] = m;
                syms.remove(i + 1);
            }
            out.extend_from_slice(&syms);
        }
        Ok(())
    }

    /// Raw bytes for a token (special tokens render as their literal text).
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        if self.is_special[id as usize] {
            return t.as_bytes().to_vec();
        }
        t.chars()
            .map(|c| self.char_to_byte.get(&c).copied().unwrap_or(b'?'))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let path = std::path::Path::new(crate::DEFAULT_MODEL);
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let g = Gguf::open(path).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        for text in [
            "Hello, world! It's 2026.",
            "<|im_start|>user\nnaïve café — 東京 🚀\n\n  indented\ttab<|im_end|>\n",
            "1234567 x=y+z;\r\n",
        ] {
            let ids = tok.encode(text).unwrap();
            let bytes: Vec<u8> = ids.iter().flat_map(|&i| tok.token_bytes(i)).collect();
            assert_eq!(String::from_utf8(bytes).unwrap(), text);
        }
        let ids = tok.encode("<|im_start|>user").unwrap();
        assert_eq!(ids[0], tok.token_id("<|im_start|>").unwrap());
        // Digits split one per token (\p{N} alternative).
        assert_eq!(tok.encode("123").unwrap().len(), 3);
    }
}
