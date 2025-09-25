use petgraph::visit::EdgeRef;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use indexmap::IndexMap;
use serde_json::Value;

use crate::graph::model::{
    EdgeAttributes, GraphInstance, NodeAttributes, RawEdge, RawGraph, RawNode,
};

/// Helper for exporting graphs back to JSON files compatible with the loader format.
pub struct GraphWriter;

impl GraphWriter {
    pub fn to_raw_graph(graph: &GraphInstance) -> RawGraph {
        let mut nodes = Vec::new();
        for (id, idx) in &graph.node_lookup {
            let attributes = build_node_attributes(
                graph
                    .graph
                    .node_weight(*idx)
                    .expect("node weight must exist"),
            );
            nodes.push(RawNode {
                id: id.clone(),
                attributes,
            });
        }

        let mut edges = Vec::new();
        for edge_ref in graph.graph.edge_references() {
            let source = graph
                .reverse_lookup
                .get(&edge_ref.source())
                .cloned()
                .expect("source id");
            let target = graph
                .reverse_lookup
                .get(&edge_ref.target())
                .cloned()
                .expect("target id");
            if !graph.directed {
                let src_ord = source.cmp(&target);
                if src_ord == std::cmp::Ordering::Greater {
                    continue;
                }
                if src_ord == std::cmp::Ordering::Equal && edge_ref.source() != edge_ref.target() {
                    continue;
                }
            }
            let attributes = build_edge_attributes(edge_ref.weight());
            edges.push(RawEdge {
                source,
                target,
                attributes,
            });
        }

        RawGraph {
            nodes,
            edges,
            graph_attributes: graph.graph_attributes.clone(),
            directed: graph.directed,
        }
    }

    pub fn to_json_string(graph: &GraphInstance) -> Result<String> {
        let raw = Self::to_raw_graph(graph);
        Ok(serde_json::to_string_pretty(&raw)?)
    }

    pub fn write_to_path(graph: &GraphInstance, path: &Path) -> Result<()> {
        let json = Self::to_json_string(graph)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}

fn build_node_attributes(node: &NodeAttributes) -> IndexMap<String, Value> {
    let mut map = IndexMap::new();
    if let Some(label) = &node.label {
        map.insert("label".to_string(), Value::String(label.clone()));
    }
    if let Some(weight) = node.weight {
        if let Some(number) = serde_json::Number::from_f64(weight) {
            map.insert("weight".to_string(), Value::Number(number));
        }
    }
    for (key, value) in &node.extra {
        map.insert(key.clone(), value.clone());
    }
    map
}

fn build_edge_attributes(edge: &EdgeAttributes) -> IndexMap<String, Value> {
    let mut map = IndexMap::new();
    if let Some(weight) = edge.weight {
        if let Some(number) = serde_json::Number::from_f64(weight) {
            map.insert("weight".to_string(), Value::Number(number));
        }
    }
    for (key, value) in &edge.extra {
        map.insert(key.clone(), value.clone());
    }
    map
}
