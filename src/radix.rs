//! Paged KV-cache pool and RadixAttention prefix tree with GDN state checkpoints.
//!
//! Full-attention layers store FP16-packed K/V (`u32`, 2x f16 per word) in fixed-size
//! physical pages (`KV_PAGE_SIZE` tokens). A radix tree tracks cached prefix blocks and
//! evicts unreferenced leaf blocks in LRU order when physical pages run low.
//!
//! Because Qwen3.5 is a hybrid Gated-DeltaNet + full-attention architecture, a reusable
//! prefix requires both the full-attention KV pages and a GDN recurrent/conv state
//! checkpoint at or before the matched prefix length.

use crate::gpu::{Buf, Gpu};
use std::collections::HashMap;
use std::sync::Arc;

pub const KV_PAGE_SIZE: usize = 16;

/// Host-RAM snapshot of all GDN layers' recurrent and convolution states at a prefix boundary.
#[derive(Clone)]
pub struct GdnCheckpoint {
    /// Conv rolling state per layer (`None` for full-attention layers).
    pub conv: Vec<Option<Vec<f32>>>,
    /// SSM recurrent state per layer (`None` for full-attention layers).
    pub ssm: Vec<Option<Vec<f32>>>,
    /// Main-model normed hidden state at the last token before the checkpoint (`[hidden]`).
    pub last_hidden: Option<Vec<f32>>,
}
/// Physical paged KV pool shared across active sequences and cached RadixAttention prefixes.
pub struct KvPool {
    pub num_pages: usize,
    pub page_size: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    /// Per full-attention layer (including optional MTP layer): `[num_pages, page_size, n_kv, hd/2]` u32.
    pub k_pools: Vec<Buf>,
    pub v_pools: Vec<Buf>,
    free_pages: Vec<u32>,
    /// Number of active sequences + radix-tree references holding each page.
    /// When `in_radix[p]` is true, the radix tree holds 1 implicit ownership (not in `ref_counts`);
    /// `ref_counts[p]` tracks active sequence pins so `ref_counts[p] == 0` means evictable.
    ref_counts: Vec<u32>,
    in_radix: Vec<bool>,
}

impl KvPool {
    pub fn new(
        gpu: &Gpu,
        n_attn_layers: usize,
        num_pages: usize,
        n_kv: usize,
        head_dim: usize,
    ) -> Self {
        assert_eq!(head_dim % 2, 0, "FP16-packed KV requires even head_dim");
        let words_per_layer = num_pages * KV_PAGE_SIZE * n_kv * (head_dim / 2);
        let (k_pools, v_pools) = gpu.client.memory_persistent_allocation((), |_| {
            let mut ks = Vec::with_capacity(n_attn_layers);
            let mut vs = Vec::with_capacity(n_attn_layers);
            for _ in 0..n_attn_layers {
                ks.push(gpu.alloc(words_per_layer));
                vs.push(gpu.alloc(words_per_layer));
            }
            (ks, vs)
        });
        let free_pages = (0..num_pages as u32).rev().collect();
        Self {
            num_pages,
            page_size: KV_PAGE_SIZE,
            n_kv,
            head_dim,
            k_pools,
            v_pools,
            free_pages,
            ref_counts: vec![0; num_pages],
            in_radix: vec![false; num_pages],
        }
    }

    pub fn free_count(&self) -> usize {
        self.free_pages.len()
    }

    pub fn active_count(&self) -> usize {
        self.ref_counts.iter().filter(|&&c| c > 0).count()
    }

    pub fn cached_count(&self) -> usize {
        self.in_radix.iter().filter(|&&b| b).count()
    }

    /// Allocate one physical page, evicting an unreferenced LRU radix leaf if needed.
    pub fn alloc_page(&mut self, radix: &mut RadixTree) -> Option<u32> {
        if let Some(p) = self.free_pages.pop() {
            self.ref_counts[p as usize] = 1;
            return Some(p);
        }
        if radix.evict_one(&self.ref_counts, &mut self.in_radix, &mut self.free_pages) {
            if let Some(p) = self.free_pages.pop() {
                self.ref_counts[p as usize] = 1;
                return Some(p);
            }
        }
        None
    }

    /// Pin an existing page for an active sequence.
    pub fn pin(&mut self, page_id: u32) {
        self.ref_counts[page_id as usize] += 1;
    }

    /// Unpin a page when a sequence releases it. If not retained in the radix tree,
    /// returns immediately to the free list.
    pub fn unpin(&mut self, page_id: u32) {
        let idx = page_id as usize;
        debug_assert!(self.ref_counts[idx] > 0);
        self.ref_counts[idx] -= 1;
        if self.ref_counts[idx] == 0 && !self.in_radix[idx] {
            self.free_pages.push(page_id);
        }
    }

    /// Mark a page as owned by a radix tree node (if not already marked).
    fn mark_radix(&mut self, page_id: u32) {
        self.in_radix[page_id as usize] = true;
    }
}

type BlockKey = [u32; KV_PAGE_SIZE];

struct RadixNode {
    /// Page-aligned token blocks along this incoming edge (`len % KV_PAGE_SIZE == 0`).
    tokens: Vec<u32>,
    /// Physical KV page IDs corresponding 1:1 with each `KV_PAGE_SIZE` block in `tokens`.
    pages: Vec<u32>,
    /// Optional GDN state checkpoint valid right after `tokens` of this node.
    gdn_ckpt: Option<Arc<GdnCheckpoint>>,
    /// Children keyed by their first `KV_PAGE_SIZE`-token block.
    children: HashMap<BlockKey, usize>,
    parent: Option<usize>,
    last_access: u64,
}

pub struct PrefixMatch {
    /// Number of matched tokens (multiple of `KV_PAGE_SIZE`).
    pub matched_tokens: usize,
    /// Pinned physical page IDs for `0..matched_tokens / KV_PAGE_SIZE`.
    pub pages: Vec<u32>,
    /// GDN checkpoint at `matched_tokens` (always `Some` when `matched_tokens > 0` and
    /// `require_gdn` is true).
    pub gdn_ckpt: Option<Arc<GdnCheckpoint>>,
}

/// Radix prefix tree mapping `KV_PAGE_SIZE`-aligned token blocks to physical KV pages
/// and optional GDN state checkpoints.
pub struct RadixTree {
    nodes: Vec<Option<RadixNode>>,
    free_ids: Vec<usize>,
    clock: u64,
    max_gdn_ckpts: usize,
    gdn_ckpt_count: usize,
}

impl RadixTree {
    pub fn new(max_gdn_ckpts: usize) -> Self {
        let root = RadixNode {
            tokens: Vec::new(),
            pages: Vec::new(),
            gdn_ckpt: None,
            children: HashMap::new(),
            parent: None,
            last_access: 0,
        };
        Self {
            nodes: vec![Some(root)],
            free_ids: Vec::new(),
            clock: 0,
            max_gdn_ckpts,
            gdn_ckpt_count: 0,
        }
    }

    fn alloc_node(&mut self, node: RadixNode) -> usize {
        if let Some(id) = self.free_ids.pop() {
            self.nodes[id] = Some(node);
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(Some(node));
            id
        }
    }

    fn free_node(&mut self, id: usize) {
        if let Some(n) = self.nodes[id].take() {
            if n.gdn_ckpt.is_some() {
                self.gdn_ckpt_count = self.gdn_ckpt_count.saturating_sub(1);
            }
            self.free_ids.push(id);
        }
    }

    fn block_key(tokens: &[u32]) -> BlockKey {
        tokens[..KV_PAGE_SIZE].try_into().unwrap()
    }

    /// Split node `nid` after `keep_blocks` (`1 <= keep_blocks < node.pages.len()`).
    fn split_node(&mut self, nid: usize, keep_blocks: usize) {
        let keep_toks = keep_blocks * KV_PAGE_SIZE;
        let orig = self.nodes[nid].take().unwrap();
        let child_tokens = orig.tokens[keep_toks..].to_vec();
        let child_pages = orig.pages[keep_blocks..].to_vec();
        let child_key = Self::block_key(&child_tokens);
        let child = RadixNode {
            tokens: child_tokens,
            pages: child_pages,
            gdn_ckpt: orig.gdn_ckpt,
            children: orig.children,
            parent: Some(nid),
            last_access: orig.last_access,
        };
        let child_id = self.alloc_node(child);
        // Reparent grandchildren to `child_id`.
        let grand_ids: Vec<usize> = self.nodes[child_id]
            .as_ref()
            .unwrap()
            .children
            .values()
            .copied()
            .collect();
        for gid in grand_ids {
            if let Some(g) = self.nodes[gid].as_mut() {
                g.parent = Some(child_id);
            }
        }
        let mut parent_children = HashMap::new();
        parent_children.insert(child_key, child_id);
        self.nodes[nid] = Some(RadixNode {
            tokens: orig.tokens[..keep_toks].to_vec(),
            pages: orig.pages[..keep_blocks].to_vec(),
            gdn_ckpt: None,
            children: parent_children,
            parent: orig.parent,
            last_access: orig.last_access,
        });
    }

    /// Find the longest cached `KV_PAGE_SIZE`-aligned prefix of `tokens`.
    /// When `require_gdn` is true, truncates back to the deepest node on the path that
    /// holds a `GdnCheckpoint` (and leaves at least 1 token unconsumed if `leave_last` so
    /// logits/last-token forward still runs).
    pub fn match_prefix(
        &mut self,
        tokens: &[u32],
        pool: &mut KvPool,
        require_gdn: bool,
        leave_last: bool,
    ) -> PrefixMatch {
        let usable_len = if leave_last {
            tokens.len().saturating_sub(1)
        } else {
            tokens.len()
        };
        let max_blocks = usable_len / KV_PAGE_SIZE;
        if max_blocks == 0 {
            return PrefixMatch {
                matched_tokens: 0,
                pages: Vec::new(),
                gdn_ckpt: None,
            };
        }

        self.clock += 1;
        let now = self.clock;
        let mut cur = 0usize;
        let mut pos_blocks = 0usize;
        let mut path_pages = Vec::new();
        let mut best_blocks = 0usize;
        let mut best_ckpt: Option<Arc<GdnCheckpoint>> = None;

        while pos_blocks < max_blocks {
            let key = Self::block_key(&tokens[pos_blocks * KV_PAGE_SIZE..]);
            let next_id = self.nodes[cur]
                .as_ref()
                .and_then(|n| n.children.get(&key).copied());
            let Some(nid) = next_id else {
                break;
            };
            let node = self.nodes[nid].as_mut().unwrap();
            node.last_access = now;
            let nblocks = node.pages.len();
            let mut matched = 0usize;
            while matched < nblocks && pos_blocks + matched < max_blocks {
                let a = &node.tokens[matched * KV_PAGE_SIZE..(matched + 1) * KV_PAGE_SIZE];
                let b = &tokens[(pos_blocks + matched) * KV_PAGE_SIZE
                    ..(pos_blocks + matched + 1) * KV_PAGE_SIZE];
                if a == b {
                    path_pages.push(node.pages[matched]);
                    matched += 1;
                } else {
                    break;
                }
            }
            pos_blocks += matched;
            if matched == nblocks {
                if let Some(ckpt) = &node.gdn_ckpt {
                    best_blocks = pos_blocks;
                    best_ckpt = Some(Arc::clone(ckpt));
                } else if !require_gdn {
                    best_blocks = pos_blocks;
                }
                cur = nid;
            } else {
                if !require_gdn {
                    best_blocks = pos_blocks;
                }
                break;
            }
        }

        let keep_blocks = if require_gdn {
            best_blocks
        } else {
            path_pages.len()
        };
        path_pages.truncate(keep_blocks);
        for &p in &path_pages {
            pool.pin(p);
        }
        PrefixMatch {
            matched_tokens: keep_blocks * KV_PAGE_SIZE,
            pages: path_pages,
            gdn_ckpt: if keep_blocks > 0 { best_ckpt } else { None },
        }
    }

    /// Insert a `KV_PAGE_SIZE`-aligned prefix (full pages `pages[..nblocks]`) and an
    /// optional GDN checkpoint at the end of those `nblocks * KV_PAGE_SIZE` tokens.
    pub fn insert_prefix(
        &mut self,
        tokens: &[u32],
        pages: &[u32],
        gdn_ckpt: Option<Arc<GdnCheckpoint>>,
        pool: &mut KvPool,
    ) {
        let nblocks = (tokens.len() / KV_PAGE_SIZE).min(pages.len());
        if nblocks == 0 {
            return;
        }
        self.clock += 1;
        let now = self.clock;
        let mut cur = 0usize;
        let mut pos_blocks = 0usize;

        while pos_blocks < nblocks {
            let key = Self::block_key(&tokens[pos_blocks * KV_PAGE_SIZE..]);
            let next_id = self.nodes[cur]
                .as_ref()
                .and_then(|n| n.children.get(&key).copied());
            let Some(nid) = next_id else {
                // Insert new leaf with remaining blocks.
                let rem_toks = tokens[pos_blocks * KV_PAGE_SIZE..nblocks * KV_PAGE_SIZE].to_vec();
                let rem_pages = pages[pos_blocks..nblocks].to_vec();
                for &p in &rem_pages {
                    pool.mark_radix(p);
                }
                let had_ckpt = gdn_ckpt.is_some();
                let leaf = RadixNode {
                    tokens: rem_toks,
                    pages: rem_pages,
                    gdn_ckpt,
                    children: HashMap::new(),
                    parent: Some(cur),
                    last_access: now,
                };
                let leaf_id = self.alloc_node(leaf);
                self.nodes[cur]
                    .as_mut()
                    .unwrap()
                    .children
                    .insert(key, leaf_id);
                if had_ckpt {
                    self.gdn_ckpt_count += 1;
                    self.trim_gdn_ckpts();
                }
                return;
            };

            let node_nblocks = self.nodes[nid].as_ref().unwrap().pages.len();
            let mut matched = 0usize;
            {
                let node = self.nodes[nid].as_ref().unwrap();
                while matched < node_nblocks && pos_blocks + matched < nblocks {
                    let a = &node.tokens[matched * KV_PAGE_SIZE..(matched + 1) * KV_PAGE_SIZE];
                    let b = &tokens[(pos_blocks + matched) * KV_PAGE_SIZE
                        ..(pos_blocks + matched + 1) * KV_PAGE_SIZE];
                    if a == b {
                        matched += 1;
                    } else {
                        break;
                    }
                }
            }
            if matched < node_nblocks {
                self.split_node(nid, matched);
            }
            pos_blocks += matched;
            let node = self.nodes[nid].as_mut().unwrap();
            node.last_access = now;
            if pos_blocks == nblocks {
                if gdn_ckpt.is_some() {
                    if node.gdn_ckpt.is_none() {
                        self.gdn_ckpt_count += 1;
                    }
                    node.gdn_ckpt = gdn_ckpt;
                    self.trim_gdn_ckpts();
                }
                return;
            }
            cur = nid;
        }
    }

    fn trim_gdn_ckpts(&mut self) {
        while self.gdn_ckpt_count > self.max_gdn_ckpts {
            let victim = self
                .nodes
                .iter()
                .enumerate()
                .skip(1)
                .filter_map(|(id, n)| {
                    let n = n.as_ref()?;
                    n.gdn_ckpt.as_ref().map(|_| (id, n.last_access))
                })
                .min_by_key(|&(_, ts)| ts)
                .map(|(id, _)| id);
            let Some(vid) = victim else { break };
            if let Some(n) = self.nodes[vid].as_mut() {
                n.gdn_ckpt = None;
                self.gdn_ckpt_count = self.gdn_ckpt_count.saturating_sub(1);
            }
        }
    }

    /// Evict the least-recently-used unreferenced leaf node, freeing its physical pages.
    pub fn evict_one(
        &mut self,
        ref_counts: &[u32],
        in_radix: &mut [bool],
        free_pages: &mut Vec<u32>,
    ) -> bool {
        let victim = self
            .nodes
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(id, n)| {
                let n = n.as_ref()?;
                if !n.children.is_empty() {
                    return None;
                }
                if n.pages.iter().any(|&p| ref_counts[p as usize] > 0) {
                    return None;
                }
                Some((id, n.last_access))
            })
            .min_by_key(|&(_, ts)| ts)
            .map(|(id, _)| id);
        let Some(vid) = victim else {
            return false;
        };
        let node = self.nodes[vid].as_ref().unwrap();
        let key = Self::block_key(&node.tokens);
        let parent = node.parent;
        for &p in &node.pages {
            in_radix[p as usize] = false;
            free_pages.push(p);
        }
        if let Some(pid) = parent
            && let Some(pnode) = self.nodes[pid].as_mut()
        {
            pnode.children.remove(&key);
        }
        self.free_node(vid);
        true
    }
}
