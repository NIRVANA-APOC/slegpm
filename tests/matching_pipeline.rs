use anyhow::Result;
use petgraph::algo::isomorphism::is_isomorphic_matching;
use slegpm::MatchingWorkflow;
use slegpm::WorkflowConfig;
use slegpm::graph::construction::GraphLoader;
use slegpm::graph::model::{EdgeAttributes, NodeAttributes};
use slegpm::pipeline::candidate::{AnchorSelector, CandidateGenerator, SubgraphExtractor};
use slegpm::pipeline::preprocess::GraphPreprocessor;

fn small_target_graph() -> String {
    r#"{
        "directed": false,
        "nodes": [
            {"id": "a", "attributes": {"label": "A"}},
            {"id": "b", "attributes": {"label": "B"}},
            {"id": "c", "attributes": {"label": "C"}},
            {"id": "d", "attributes": {"label": "D"}}
        ],
        "edges": [
            {"source": "a", "target": "b", "attributes": {"weight": 1.0}},
            {"source": "b", "target": "c", "attributes": {"weight": 1.0}},
            {"source": "c", "target": "d", "attributes": {"weight": 1.0}},
            {"source": "a", "target": "c", "attributes": {"weight": 1.0}}
        ]
    }"#
    .to_string()
}

fn path_pattern_graph() -> String {
    r#"{
        "directed": false,
        "nodes": [
            {"id": "b", "attributes": {"label": "B"}},
            {"id": "c", "attributes": {"label": "C"}},
            {"id": "d", "attributes": {"label": "D"}}
        ],
        "edges": [
            {"source": "b", "target": "c", "attributes": {"weight": 1.0}},
            {"source": "c", "target": "d", "attributes": {"weight": 1.0}}
        ]
    }"#
    .to_string()
}

#[test]
fn candidate_generation_covers_pattern() -> Result<()> {
    let target = GraphLoader::from_json_str(&small_target_graph())?;
    let pattern = GraphLoader::from_json_str(&path_pattern_graph())?;

    let anchors = AnchorSelector::select_anchors(&pattern, Some(1));
    let candidates = CandidateGenerator::filter_candidates(&pattern, &target, &anchors, 1e-6)?;
    assert!(!candidates.is_empty());

    let subgraphs =
        SubgraphExtractor::extract_subgraphs(&target, &candidates, pattern.graph.node_count())?;
    assert!(!subgraphs.is_empty());
    Ok(())
}

#[test]
fn end_to_end_small_workflow_matches_pattern() -> Result<()> {
    let pattern_pre =
        GraphPreprocessor::preprocess_graph(GraphLoader::from_json_str(&path_pattern_graph())?)?;
    let target_pre =
        GraphPreprocessor::preprocess_graph(GraphLoader::from_json_str(&small_target_graph())?)?;

    let config = WorkflowConfig {
        anchor_count: Some(1),
        epsilon: 1e-6,
        ..WorkflowConfig::default()
    };
    let workflow = MatchingWorkflow::new(config, pattern_pre.clone(), target_pre.clone());
    let summary = workflow.execute()?;
    assert!(
        !summary.matches.is_empty(),
        "small workflow expected a match, audit={:?}",
        summary.audit_log
    );

    let pattern_graph = &pattern_pre.instance.graph;
    let epsilon = 1e-6;
    let mut found = false;
    for candidate in &summary.matches {
        if candidate.graph.node_count() == pattern_graph.node_count()
            && candidate.graph.edge_count() >= pattern_graph.edge_count()
            && is_isomorphic_matching(
                pattern_graph,
                &candidate.graph,
                |a: &NodeAttributes, b: &NodeAttributes| a.label == b.label,
                |a: &EdgeAttributes, b: &EdgeAttributes| match (a.weight, b.weight) {
                    (Some(x), Some(y)) => (x - y).abs() <= epsilon,
                    (None, None) => true,
                    _ => false,
                },
            )
        {
            found = true;
            break;
        }
    }
    assert!(found, "expected to recover the pattern as a match");

    Ok(())
}
