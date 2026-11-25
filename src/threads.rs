//! Central CPU thread budget.
//!
//! Every component used to pick its own thread count: Rayon defaulted to all 128 logical
//! CPUs, `MOE_POOL` to 10, and Tokio to 128. That oversubscribes the host and makes results
//! depend on whichever pool happened to initialize first. This module owns one budget and
//! hands each component an explicit share before any pool is created.
//!
//! Call [`CpuBudget::init`] once at the very start of `main`, before the first Rayon
//! iterator, `MOE_POOL` access or Tokio runtime is built.

const TOTAL_VAR: &str = "BURN_CPU_BUDGET";
const DECODE_VAR: &str = "BURN_DECODE_THREADS";
const MOE_VAR: &str = "BURN_CPU_THREADS";
const SERVER_VAR: &str = "BURN_SERVER_THREADS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuBudget {
    /// Total threads this process is allowed to use.
    pub total: usize,
    /// Global Rayon pool: Inkling-2/Kolibri matvecs, expert repacking, MC scoring.
    pub decode: usize,
    /// `cpu_moe::MOE_POOL`: missed-expert CPU work on the GPU path.
    pub moe: usize,
    /// Tokio runtime workers for the OpenAI-compatible server.
    pub server: usize,
}

impl CpuBudget {
    /// Compute a budget from explicit overrides. `None` means "use the default".
    /// Shares are clamped so the three pools together never exceed `total`.
    pub fn compute(
        total: Option<usize>,
        decode: Option<usize>,
        moe: Option<usize>,
        server: Option<usize>,
    ) -> Self {
        let total = total
            .or(env_usize(TOTAL_VAR))
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(8)
            .max(1);
        let mut decode = decode
            .or(env_usize(DECODE_VAR))
            .unwrap_or(total.min(64))
            .clamp(1, total);
        let moe = moe
            .or(env_usize(MOE_VAR))
            .unwrap_or((decode / 8).clamp(1, 10))
            .clamp(1, total);
        let server = server
            .or(env_usize(SERVER_VAR))
            .unwrap_or((total / 32).clamp(1, 4))
            .clamp(1, total);
        // Keep the sum within the budget on small hosts; on this 128-CPU machine the
        // defaults (64 + 8 + 4) already leave headroom for other work.
        if decode + moe + server > total {
            decode = total.saturating_sub(moe + server).max(1);
        }
        Self {
            total,
            decode,
            moe,
            server,
        }
    }

    pub fn from_env() -> Self {
        Self::compute(None, None, None, None)
    }

    /// Override the total budget (e.g. from `--cpu-threads`) before publishing shares.
    pub fn with_total(mut self, total: Option<usize>) -> Self {
        if let Some(n) = total {
            self = Self::compute(Some(n), None, None, None);
        }
        self
    }

    /// Publish the shares through the environment variables each pool already reads.
    /// Rayon's global pool and `MOE_POOL` are `LazyLock`s, so this must run before either
    /// is first touched.
    pub fn init(self) {
        // SAFETY: called single-threaded at the top of main, before any pool initializes.
        unsafe {
            std::env::set_var("RAYON_NUM_THREADS", self.decode.to_string());
            std::env::set_var(MOE_VAR, self.moe.to_string());
            std::env::set_var(SERVER_VAR, self.server.to_string());
        }
        eprintln!(
            "cpu budget: total={} decode={} moe={} server={}",
            self.total, self.decode, self.moe, self.server
        );
    }

    /// Tokio reads this at runtime construction, after `init`, so it can use the share.
    pub fn server_workers() -> usize {
        env_usize(SERVER_VAR).unwrap_or_else(|| Self::from_env().server)
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_budget_leaves_headroom() {
        let b = CpuBudget::compute(Some(128), None, None, None);
        assert_eq!(b.decode, 64);
        assert_eq!(b.moe, 8);
        assert_eq!(b.server, 4);
        assert!(b.decode + b.moe + b.server <= b.total);
    }

    #[test]
    fn small_host_partitions_without_oversubscribing() {
        let b = CpuBudget::compute(Some(8), None, None, None);
        assert_eq!(b.moe, 1);
        assert_eq!(b.server, 1);
        assert_eq!(b.decode, 6);
        assert!(b.decode + b.moe + b.server <= b.total);
    }

    #[test]
    fn overrides_are_honored_and_clamped() {
        let b = CpuBudget::compute(Some(16), Some(100), Some(2), Some(3));
        assert_eq!(b.decode, 11); // 16 - 2 - 3
        assert_eq!(b.moe, 2);
        assert_eq!(b.server, 3);
    }
}
