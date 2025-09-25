use std::collections::{HashSet, VecDeque};

use anyhow::{Result, anyhow};
use indexmap::IndexSet;
use petgraph::prelude::NodeIndex;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::graph::construction::GraphLoader;
use crate::graph::model::GraphInstance;

#[derive(Debug, Clone)]
pub struct SampleConfig {
    pub nodes: usize,
    pub samples: usize,
    pub seed: Option<u64>,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            nodes: 5,
            samples: 1,
            seed: None,
        }
    }
}

pub struct PatternSampler;

impl PatternSampler {
    pub fn sample_patterns(
        target: &GraphInstance,
        config: SampleConfig,
    ) -> Result<Vec<GraphInstance>> {
        if config.nodes == 0 {
            return Err(anyhow!("Sample node count must be greater than zero"));
        }
        if config.samples == 0 {
            return Err(anyhow!("Sample count must be greater than zero"));
        }
        if config.nodes > target.graph.node_count() {
            return Err(anyhow!(
                "Requested node count {} exceeds graph size {}",
                config.nodes,
                target.graph.node_count()
            ));
        }

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed.unwrap_or_else(random_seed));
        let node_indices: Vec<NodeIndex> = target.graph.node_indices().collect();
        if node_indices.is_empty() {
            return Err(anyhow!("Target graph has no nodes"));
        }

        let mut unique_signatures = HashSet::new();
        let mut results = Vec::new();
        let mut attempts = 0usize;
        let max_attempts = config.samples * target.graph.node_count() * 10;

        while results.len() < config.samples && attempts < max_attempts {
            attempts += 1;
            let start = node_indices[rng.gen_range(0..node_indices.len())];
            let visited = bfs_collect(target, start, config.nodes);
            if visited.len() < config.nodes {
                continue;
            }

            let mut node_ids = IndexSet::new();
            for idx in visited {
                if let Some(id) = target.reverse_lookup.get(&idx) {
                    node_ids.insert(id.clone());
                }
            }
            if node_ids.len() < config.nodes {
                continue;
            }

            let mut signature_vec: Vec<_> = node_ids.iter().cloned().collect();
            signature_vec.sort();
            let signature = signature_vec.join("|");
            if !unique_signatures.insert(signature) {
                continue;
            }

            let subgraph = GraphLoader::induced_subgraph(target, &node_ids)?;
            results.push(subgraph);
        }

        if results.len() < config.samples {
            Err(anyhow!(
                "Unable to sample {} distinct pattern graphs with {} nodes each",
                config.samples,
                config.nodes
            ))
        } else {
            Ok(results)
        }
    }
}

fn bfs_collect(target: &GraphInstance, start: NodeIndex, limit: usize) -> IndexSet<NodeIndex> {
    let mut visited = IndexSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if visited.len() >= limit {
            break;
        }
        for neighbor in target.graph.neighbors(node) {
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
                if visited.len() >= limit {
                    break;
                }
            }
        }
    }

    let mut trimmed = IndexSet::new();
    for node in visited.into_iter().take(limit) {
        trimmed.insert(node);
    }
    trimmed
}

fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
