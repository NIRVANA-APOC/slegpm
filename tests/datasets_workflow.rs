use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use indexmap::IndexSet;
use petgraph::algo::isomorphism::is_isomorphic_matching;
use slegpm::cache::{CacheMetadata, PreprocessCache};
use slegpm::graph::{EdgeAttributes, GraphInstance, GraphWriter, NodeAttributes};
use slegpm::spectral::{spectral_profile, LaplacianMatrix};

use slegpm::pipeline::candidate::{AnchorSelector, CandidateGenerator, SubgraphExtractor};
use slegpm::{
    DatasetLoader, GraphPreprocessor, MatchingWorkflow, PatternSampler, PreprocessedGraph,
    SampleConfig, WorkflowConfig,
};
use vf2::subgraph_isomorphisms;

use slegpm::spectral::spectral_distance;

const DATASETS: [&str; 5] = ["dblp", "hprd", "human", "wordnet", "yeast"];
const PATTERN_COUNT: usize = 10;
const PATTERN_NODES: usize = 6;
const SPECTRAL_DIM: usize = 12;
const EPSILON: f64 = 5e-3;

#[test]
fn dataset_loading_and_preprocessing() -> Result<()> {
    let loader = DatasetLoader::default();
    let cache = PreprocessCache::default();

    for dataset in DATASETS {
        let graph_path = dataset_graph_path(dataset);
        let graph = loader
            .load(relative_dataset_path(dataset))
            .with_context(|| format!("load dataset graph for {}", dataset))?;
        assert!(graph.node_count() > 0, "dataset {} has no nodes", dataset);

        let preprocessed = GraphPreprocessor::from_path(&graph_path, SPECTRAL_DIM)
            .with_context(|| format!("preprocess graph for {}", dataset))?;
        assert_eq!(
            preprocessed.node_count(),
            graph.node_count(),
            "preprocessed node count mismatch for {}",
            dataset
        );

        let spectral = preprocessed
            .spectral()
            .with_context(|| format!("materialize spectral profile for {}", dataset))?;
        assert!(!spectral.eigenvalues.is_empty());

        let laplacian = preprocessed
            .laplacian()
            .with_context(|| format!("materialize laplacian for {}", dataset))?;

        match laplacian.as_ref() {
            LaplacianMatrix::Dense(dense) => {
                assert_eq!(dense.size, graph.node_count());
                let recomputed =
                    spectral_profile(&LaplacianMatrix::Dense(dense.clone()), SPECTRAL_DIM)?;
                assert_eq!(recomputed.eigenvalues.len(), spectral.eigenvalues.len());
            }
            LaplacianMatrix::Sparse(sparse) => {
                assert_eq!(sparse.size, graph.node_count());
                let recomputed = spectral_profile(laplacian.as_ref(), SPECTRAL_DIM)?;
                assert_eq!(recomputed.eigenvalues.len(), spectral.eigenvalues.len());
            }
        }

        let metadata = CacheMetadata::from_path(&graph_path)?;
        if let Some(entry) = cache
            .load(&graph_path, SPECTRAL_DIM, &metadata)
            .with_context(|| format!("read preprocess cache for {}", dataset))?
        {
            let cached_spectral = cache.load_spectral(&entry)?;
            assert_eq!(
                cached_spectral.eigenvalues.len(),
                spectral.eigenvalues.len()
            );
            let cached_laplacian = cache.load_laplacian(&entry)?;
            match cached_laplacian {
                LaplacianMatrix::Dense(dense) => {
                    assert_eq!(dense.size, graph.node_count());
                }
                LaplacianMatrix::Sparse(sparse) => {
                    assert_eq!(sparse.size, graph.node_count());
                }
            }
        } else {
            panic!("expected cache entry for {}", dataset);
        }
    }

    Ok(())
}

#[test]
fn yeast_pattern_matching_end_to_end() -> Result<()> {
    let dataset = "yeast";
    let target_path = dataset_graph_path(dataset);
    let target_pre = Arc::new(
        GraphPreprocessor::from_path(&target_path, SPECTRAL_DIM)
            .with_context(|| "preprocess target graph")?,
    );

    let patterns_dir = dataset_patterns_dir(dataset);
    fs::create_dir_all(&patterns_dir)
        .with_context(|| format!("create pattern directory for {}", dataset))?;

    let samples = PatternSampler::sample(
        target_pre.graph(),
        SampleConfig {
            nodes: PATTERN_NODES,
            samples: PATTERN_COUNT,
            seed: Some(42),
        },
    )
    .with_context(|| "sample pattern graphs")?;

    assert_eq!(
        samples.len(),
        PATTERN_COUNT,
        "expected {} patterns for {}",
        PATTERN_COUNT,
        dataset
    );

    let mut pattern_paths = Vec::with_capacity(samples.len());
    for (idx, sample) in samples.iter().enumerate() {
        let pattern_path = patterns_dir.join(format!("pattern_{}.json", idx));
        GraphWriter::write_to_path(sample, &pattern_path)
            .with_context(|| format!("write pattern {}", idx))?;
        pattern_paths.push(pattern_path);
    }

    let mut config = WorkflowConfig::default();
    config.anchor_count = Some(4);
    config.epsilon = EPSILON;
    config.spectral_dim = SPECTRAL_DIM;
    config.wl_iterations = 3;

    for path in pattern_paths {
        let pattern_pre = Arc::new(
            GraphPreprocessor::from_path(&path, SPECTRAL_DIM)
                .with_context(|| format!("preprocess sampled pattern at {:?}", path))?,
        );

        let workflow = MatchingWorkflow::new(
            config.clone(),
            Arc::clone(&pattern_pre),
            Arc::clone(&target_pre),
        );
        let summary = workflow
            .execute()
            .with_context(|| "run matching workflow")?;

        for matched in &summary.matches {
            assert!(is_isomorphic_matching(
                &pattern_pre.graph().graph,
                &matched.graph,
                |a: &NodeAttributes, b: &NodeAttributes| node_equivalent(a, b, config.epsilon),
                |a: &EdgeAttributes, b: &EdgeAttributes| edge_equivalent(a, b, config.epsilon),
            ));
        }

        let observed: IndexSet<Vec<String>> = summary
            .matches
            .iter()
            .map(|graph| graph_signature(graph))
            .collect();
        let expected = vf2_expected_matches(pattern_pre.as_ref(), target_pre.as_ref(), &config)?;

        assert_eq!(
            observed.len(),
            expected.len(),
            "match count mismatch for {}",
            dataset
        );
        assert_eq!(observed, expected, "match sets differ for {}", dataset);
    }

    Ok(())
}

fn dataset_graph_path(dataset: &str) -> PathBuf {
    Path::new("datasets").join(dataset).join("data_graph.json")
}

fn relative_dataset_path(dataset: &str) -> PathBuf {
    PathBuf::from(dataset).join("data_graph.json")
}

fn dataset_patterns_dir(dataset: &str) -> PathBuf {
    Path::new("datasets").join(dataset).join("patterns")
}

fn graph_signature(graph: &Arc<GraphInstance>) -> Vec<String> {
    let mut ids: Vec<_> = graph.reverse_lookup.values().cloned().collect();
    ids.sort();
    ids
}

fn vf2_expected_matches(
    pattern: &PreprocessedGraph,
    target: &PreprocessedGraph,
    config: &WorkflowConfig,
) -> Result<IndexSet<Vec<String>>> {
    let pattern_graph = pattern.graph();
    let target_graph = target.graph();
    let pattern_spectral = pattern.spectral()?;
    let anchors = AnchorSelector::select(pattern_graph, config.anchor_count);
    let candidates = CandidateGenerator::filter_candidates(
        pattern_graph,
        target_graph,
        &anchors,
        config.epsilon,
    )?;
    let subgraphs =
        SubgraphExtractor::extract(target_graph, &candidates, pattern_graph.node_count())?;

    let mut results: IndexSet<Vec<String>> = IndexSet::new();
    for candidate in subgraphs {
        let Ok(candidate_pre) = GraphPreprocessor::from_instance(candidate, config.spectral_dim)
        else {
            continue;
        };

        let Ok(candidate_spectral) = candidate_pre.spectral() else {
            continue;
        };
        let distance = spectral_distance(pattern_spectral.as_ref(), candidate_spectral.as_ref());
        let mut ids: Vec<_> = candidate_pre
            .graph()
            .reverse_lookup
            .values()
            .cloned()
            .collect();
        ids.sort();

        let spectral_pass = distance <= config.epsilon;

        let builder = subgraph_isomorphisms(&pattern_graph.graph, &candidate_pre.graph().graph)
            .node_eq(|a: &NodeAttributes, b: &NodeAttributes| node_equivalent(a, b, config.epsilon))
            .edge_eq(|a: &EdgeAttributes, b: &EdgeAttributes| {
                edge_equivalent(a, b, config.epsilon)
            });
        let vf2_pass = builder.first().is_some();

        if spectral_pass && vf2_pass {
            results.insert(ids);
        }
    }

    Ok(results)
}

fn node_equivalent(left: &NodeAttributes, right: &NodeAttributes, epsilon: f64) -> bool {
    if left.label != right.label {
        return false;
    }
    match (left.weight, right.weight) {
        (Some(l), Some(r)) => (l - r).abs() <= epsilon,
        (None, None) => true,
        _ => false,
    }
}

fn edge_equivalent(left: &EdgeAttributes, right: &EdgeAttributes, epsilon: f64) -> bool {
    match (left.weight, right.weight) {
        (Some(l), Some(r)) => (l - r).abs() <= epsilon,
        (None, None) => true,
        _ => false,
    }
}
