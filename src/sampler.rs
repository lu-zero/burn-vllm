//! Host-side token sampling (temperature / top-k / top-p) with a small xorshift RNG.

#[derive(Clone, Debug)]
pub struct Params {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for Params {
    /// Qwen3.5 recommended sampling (general.sampling.* in the GGUF).
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            seed: 0x5eed,
        }
    }
}

pub struct Sampler {
    p: Params,
    state: u64,
}

impl Sampler {
    pub fn new(p: Params) -> Self {
        let state = p.seed.max(1);
        Self { p, state }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let argmax = || {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };
        if self.p.temperature <= 0.0 || self.p.top_k == 1 {
            return argmax();
        }
        let k = if self.p.top_k == 0 {
            logits.len()
        } else {
            self.p.top_k.min(logits.len())
        };
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        if k < idx.len() {
            idx.select_nth_unstable_by(k, |&a, &b| {
                logits[b as usize].total_cmp(&logits[a as usize])
            });
            idx.truncate(k);
        }
        idx.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        let mx = logits[idx[0] as usize];
        let mut probs: Vec<f32> = idx
            .iter()
            .map(|&i| ((logits[i as usize] - mx) / self.p.temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        let mut cut = probs.len();
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if acc >= self.p.top_p {
                cut = i + 1;
                break;
            }
        }
        let total: f32 = probs[..cut].iter().sum();
        let mut r = self.next_f32() * total;
        for i in 0..cut {
            r -= probs[i];
            if r <= 0.0 {
                return idx[i];
            }
        }
        idx[cut - 1]
    }
}
