use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use indexmap::IndexSet;
use petgraph::prelude::NodeIndex;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::graph::{GraphId, GraphInstance, GraphLoader};

#[derive(Debug, Clone)]
pub struct SampleConfig {
    pub nodes: usize,
    pub samples: usize,
    pub seed: Option<u64>,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            nodes: 4,
            samples: 1,
            seed: None,
        }
    }
}

pub struct PatternSampler;

impl PatternSampler {
    pub fn sample(target: &GraphInstance, config: SampleConfig) -> Result<Vec<GraphInstance>> {
        if config.nodes == 0 {
            return Err(anyhow!("Requested node count must be greater than zero"));
        }
        if config.samples == 0 {
            return Err(anyhow!("Requested sample count must be greater than zero"));
        }
        if config.nodes > target.node_count() {
            return Err(anyhow!(
                "Requested subgraph with {} nodes, but target only has {} nodes",
                config.nodes,
                target.node_count()
            ));
        }

        let indices: Vec<NodeIndex> = target.graph.node_indices().collect();
        if indices.is_empty() {
            return Err(anyhow!("Target graph has no nodes"));
        }

        let max_attempts = config.samples.saturating_mul(32).max(config.samples);
        let base_seed = config.seed.unwrap_or_else(random_seed);
        let used_signatures: Arc<Mutex<HashSet<Vec<GraphId>>>> =
            Arc::new(Mutex::new(HashSet::new()));

        let attempts = 0..max_attempts;
        let mut collected: Vec<GraphInstance> = attempts
            .into_par_iter()
            .filter_map(|attempt| {
                let mut rng =
                    Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(attempt as u64));
                let start_idx = *indices.get(rng.gen_range(0..indices.len()))?;
                let collected = bfs_collect(target, start_idx, config.nodes);
                if collected.len() < config.nodes {
                    return None;
                }
                let node_ids: Vec<GraphId> = collected
                    .into_iter()
                    .filter_map(|idx| target.reverse_lookup.get(&idx).cloned())
                    .take(config.nodes)
                    .collect();
                if node_ids.len() < config.nodes {
                    return None;
                }
                let mut signature = node_ids.clone();
                signature.sort();
                {
                    let mut guard = used_signatures.lock().ok()?;
                    if !guard.insert(signature) {
                        return None;
                    }
                }
                GraphLoader::induced_subgraph(target, &node_ids).ok()
            })
            .collect();

        if collected.len() < config.samples {
            return Err(anyhow!(
                "Unable to sample {} unique subgraphs of size {} within attempt budget",
                config.samples,
                config.nodes
            ));
        }

        collected.truncate(config.samples);
        Ok(collected)
    }
}

fn bfs_collect(graph: &GraphInstance, start: NodeIndex, limit: usize) -> IndexSet<NodeIndex> {
    let mut visited: IndexSet<NodeIndex> = IndexSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if visited.len() >= limit {
            break;
        }
        for neighbor in graph.graph.neighbors(node) {
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
                if visited.len() >= limit {
                    break;
                }
            }
        }
    }

    visited
}

fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
