use criterion::{Criterion, black_box, criterion_group, criterion_main};
use indexmap::IndexMap;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use slegpm::graph::model::{EdgeAttributes, GraphInstance, LabeledGraph, NodeAttributes};
use slegpm::spectral::{NormalizedLaplacianBuilder, SpectralVectorExtractor};

fn random_graph(nodes: usize, probability: f64, seed: u64) -> GraphInstance {
    let mut graph = LabeledGraph::with_capacity(nodes, nodes * nodes);
    let mut node_lookup = IndexMap::new();
    let mut reverse_lookup = IndexMap::new();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    for i in 0..nodes {
        let attrs = NodeAttributes {
            label: Some(format!("v{i}")),
            weight: None,
            extra: IndexMap::new(),
        };
        let idx = graph.add_node(attrs);
        let id = format!("v{i}");
        node_lookup.insert(id.clone(), idx);
        reverse_lookup.insert(idx, id);
    }

    for i in 0..nodes {
        for j in (i + 1)..nodes {
            if rng.r#gen::<f64>()() <= probability {
                let edge = EdgeAttributes {
                    weight: Some(1.0),
                    extra: IndexMap::new(),
                };
                let source = node_lookup[&format!("v{i}")];
                let target = node_lookup[&format!("v{j}")];
                graph.add_edge(source, target, edge.clone());
                graph.add_edge(target, source, edge);
            }
        }
    }

    GraphInstance {
        graph,
        node_lookup,
        reverse_lookup,
        graph_attributes: IndexMap::new(),
        directed: false,
    }
}

fn bench_spectral_pipeline(c: &mut Criterion) {
    let graph_small = random_graph(64, 0.15, 42);
    let graph_medium = random_graph(256, 0.08, 7);

    let mut group = c.benchmark_group("spectral_pipeline");

    group.bench_function("laplacian_64", |b| {
        b.iter(|| {
            let lap = NormalizedLaplacianBuilder::build(&graph_small).expect("laplacian");
            black_box(lap);
        });
    });

    group.bench_function("spectral_64", |b| {
        let lap = NormalizedLaplacianBuilder::build(&graph_small).expect("laplacian");
        b.iter(|| {
            let profile = SpectralVectorExtractor::compute_profile(&lap).expect("profile");
            black_box(profile);
        });
    });

    group.bench_function("spectral_256", |b| {
        let lap = NormalizedLaplacianBuilder::build(&graph_medium).expect("laplacian");
        b.iter(|| {
            let profile = SpectralVectorExtractor::compute_profile(&lap).expect("profile");
            black_box(profile.clone());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_spectral_pipeline);
criterion_main!(benches);
