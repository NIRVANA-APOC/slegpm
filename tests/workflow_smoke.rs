use slegpm::graph::model::{EdgeAttributes, NodeAttributes};
use slegpm::{
    GraphLoader, GraphPreprocessor, GraphWriter, MatchingWorkflow, PatternSampler, SampleConfig,
    WorkflowConfig,
};
use std::fmt::format;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn temp_path(name: &str) -> PathBuf {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let mut path = std::env::temp_dir();
    path.push(format!("slegpm_{}_{}.json", name, epoch));
    path
}

#[test]
fn smoke_runs_end_to_end() {
    let pattern = r#"
    {
        "nodes": [
            {"id": "p0", "attributes": {"label": "A"}},
            {"id": "p1", "attributes": {"label": "B"}}
        ],
        "edges": [
            {"source": "p0", "target": "p1", "attributes": {"weight": 1.0}}
        ],
        "directed": false
    }
    "#;

    let target = r#"
    {
        "nodes": [
            {"id": "t0", "attributes": {"label": "A"}},
            {"id": "t1", "attributes": {"label": "B"}},
            {"id": "t2", "attributes": {"label": "C"}}
        ],
        "edges": [
            {"source": "t0", "target": "t1", "attributes": {"weight": 1.0}},
            {"source": "t1", "target": "t2", "attributes": {"weight": 1.0}}
        ],
        "directed": false
    }
    "#;

    let pattern_path = temp_path("pattern");
    let target_path = temp_path("target");

    fs::write(&pattern_path, pattern).expect("write pattern graph");
    fs::write(&target_path, target).expect("write target graph");

    let mut config = WorkflowConfig::default();
    config.anchor_count = Some(1);
    config.epsilon = 1e-3;
    config.dominance_threshold = 0.5;

    let pattern_pre = GraphPreprocessor::load_from_path(&pattern_path).expect("preprocess pattern");
    let target_pre = GraphPreprocessor::load_from_path(&target_path).expect("preprocess target");

    let workflow = MatchingWorkflow::new(config, pattern_pre, target_pre);
    let result = workflow.execute().expect("workflow execution");

    assert!(
        !result.matches.is_empty(),
        "expected at least one match, audit={:?}",
        result.audit_log
    );
    assert!(
        result.online_duration >= Duration::from_millis(0),
        "online duration should be non-negative"
    );
    assert!(
        result.matching_duration >= Duration::from_millis(0),
        "matching duration should be non-negative"
    );

    let _ = fs::remove_file(pattern_path);
    let _ = fs::remove_file(target_path);
}

#[test]
fn dataset_template_graph_is_compatible() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate_paths = [
        base.join("dataset/yeast/template_graph.json"),
        base.join("datasets/yeast/template_graph.json"),
    ];
    let load_path = candidate_paths
        .into_iter()
        .find(|path| path.exists())
        .expect("template graph path");
    let preprocessed =
        GraphPreprocessor::load_from_path(&load_path).expect("dataset graph preprocessing");
    assert_eq!(preprocessed.instance.graph.node_count(), 3);
    assert_eq!(preprocessed.instance.graph.edge_count(), 4); // undirected edges mirrored
    let first_node = preprocessed
        .instance
        .graph
        .node_indices()
        .next()
        .expect("first node");
    let attrs = preprocessed
        .instance
        .graph
        .node_weight(first_node)
        .expect("node weight");
    assert!(attrs.extra.contains_key("name"));
}

#[test]
fn graph_round_trip_serialization() {
    let json = r#"
    {
        "nodes": [
            {"id": "a", "attributes": {"label": "X", "name": "alpha"}},
            {"id": "b", "attributes": {"label": "Y"}}
        ],
        "edges": [
            {"source": "a", "target": "b", "attributes": {"weight": 2.5}}
        ],
        "directed": false,
        "graph_attributes": {"test": true}
    }
    "#;

    let graph = GraphLoader::from_json_str(json).expect("load original graph");
    let exported = GraphWriter::to_json_string(&graph).expect("serialize graph");
    let round_trip = GraphLoader::from_json_str(&exported).expect("roundtrip load");

    assert_eq!(round_trip.graph.node_count(), graph.graph.node_count());
    assert_eq!(round_trip.graph.edge_count(), graph.graph.edge_count());
    assert_eq!(
        round_trip
            .graph_attributes
            .get("test")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn pattern_sampler_produces_connected_subgraphs() {
    let target_json = r#"
    {
        "nodes": [
            {"id": "0", "attributes": {}},
            {"id": "1", "attributes": {}},
            {"id": "2", "attributes": {}},
            {"id": "3", "attributes": {}},
            {"id": "4", "attributes": {}}
        ],
        "edges": [
            {"source": "0", "target": "1", "attributes": {}},
            {"source": "1", "target": "2", "attributes": {}},
            {"source": "2", "target": "3", "attributes": {}},
            {"source": "3", "target": "4", "attributes": {}},
            {"source": "4", "target": "0", "attributes": {}}
        ],
        "directed": false
    }
    "#;

    let target = GraphLoader::from_json_str(target_json).expect("load target graph");
    let config = SampleConfig {
        nodes: 3,
        samples: 2,
        seed: Some(42),
    };
    let patterns = PatternSampler::sample_patterns(&target, config).expect("sample patterns");
    assert_eq!(patterns.len(), 2);
    for pattern in patterns {
        assert_eq!(pattern.graph.node_count(), 3);
        assert!(pattern.graph.edge_count() >= 2);
    }
}

#[test]
fn sampled_patterns_match_source() {
    use petgraph::algo::isomorphism::{is_isomorphic_matching, is_isomorphic_subgraph_matching};

    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("datasets/yeast/data_graph.json");
    let file = std::fs::File::open(&dataset_path).expect("open dataset graph");
    let target = GraphLoader::from_reader(file).expect("load dataset graph");

    println!(
        "Target N: {}, E: {}",
        target.graph.node_count(),
        target.graph.edge_count()
    );

    let config = SampleConfig {
        nodes: 6,
        samples: 2,
        seed: Some(123),
    };
    let patterns = PatternSampler::sample_patterns(&target, config).expect("sample patterns");

    patterns.iter().enumerate().for_each(|(idx, pattern)| {
        GraphWriter::write_to_path(
            &pattern,
            &dataset_path
                .parent()
                .unwrap()
                .join("test_data")
                .join(format!("pattern_{}", idx))
                .with_extension("json"),
        )
        .unwrap()
    });

    assert_eq!(patterns.len(), 2);

    let target_pre =
        GraphPreprocessor::preprocess_graph(target.clone()).expect("preprocess target");
    let workflow_config = WorkflowConfig {
        anchor_count: Some(3),
        epsilon: 1e-3,
        dominance_threshold: 0.0,
    };

    for pattern in patterns {
        let pattern_pre =
            GraphPreprocessor::preprocess_graph(pattern.clone()).expect("preprocess pattern");
        let workflow = MatchingWorkflow::new(
            workflow_config.clone(),
            pattern_pre.clone(),
            target_pre.clone(),
        );
        let result = workflow.execute().expect("workflow execution");
        println!("AUDIT LOG: {:#?}", result.audit_log);
        assert!(
            !result.matches.is_empty(),
            "expected sampled pattern to match its source graph, audit={:?}",
            result.audit_log
        );
        assert!(
            result.matching_duration >= Duration::from_millis(0),
            "matching duration should be non-negative"
        );

        let mut has_isomorphic = true;
        let mut fail_match = Vec::new();
        println!("Matches: {}", result.matches.len());
        for matched in &result.matches {
            println!(
                "N: {}, E: {}",
                matched.graph.node_count(),
                matched.graph.edge_count()
            );

            println!(
                "N: {}, E: {}",
                pattern_pre.instance.graph.node_count(),
                pattern_pre.instance.graph.edge_count()
            );
            println!("{:50}", "-");
            if !is_isomorphic_matching(
                &pattern_pre.instance.graph,
                &matched.graph,
                |a, b| node_compatible(a, b),
                |a, b| edge_compatible(a, b),
            ) {
                has_isomorphic = false;
                fail_match.push(matched);
                break;
            }
        }
        assert!(
            has_isomorphic,
            "no isomorphic match found for sampled pattern"
        );
        println!("Failed pattern matches: {}", fail_match.len());
        for fail_match in fail_match {
            println!("Failed match: {:?}", fail_match);
        }
    }
}

fn node_compatible(a: &NodeAttributes, b: &NodeAttributes) -> bool {
    match (&a.label, &b.label) {
        (Some(lhs), Some(rhs)) if lhs != rhs => return false,
        (Some(_), None) | (None, Some(_)) => return false,
        _ => {}
    }
    match (a.weight, b.weight) {
        (Some(lhs), Some(rhs)) => (lhs - rhs).abs() <= 1e-3,
        (None, None) => true,
        _ => false,
    }
}

fn edge_compatible(a: &EdgeAttributes, b: &EdgeAttributes) -> bool {
    match (a.weight, b.weight) {
        (Some(lhs), Some(rhs)) => (lhs - rhs).abs() <= 1e-3,
        (None, None) => true,
        _ => false,
    }
}
