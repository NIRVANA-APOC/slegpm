use std::io::Read;

use anyhow::{Result, anyhow};
use indexmap::{IndexMap, IndexSet};
use petgraph::prelude::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::graph::model::{EdgeAttributes, GraphId, GraphInstance, NodeAttributes, RawGraph};

/// High-level loader responsible for turning JSON representations into in-memory graphs.
#[derive(Debug, Default)]
pub struct GraphLoader;

impl GraphLoader {
    /// Parse a JSON string into a graph instance.
    pub fn from_json_str(json: &str) -> Result<GraphInstance> {
        let raw: RawGraph = serde_json::from_str(json)?;
        Self::from_raw_graph(raw)
    }

    /// Read JSON graph data from a reader.
    pub fn from_reader<R: Read>(mut reader: R) -> Result<GraphInstance> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        Self::from_json_str(&buf)
    }

    /// Extract an induced subgraph over the provided node identifiers.
    pub fn induced_subgraph(
        graph: &GraphInstance,
        node_ids: &IndexSet<GraphId>,
    ) -> Result<GraphInstance> {
        let mut retain_indices = IndexSet::new();

        for node_id in node_ids {
            let idx = graph
                .node_lookup
                .get(node_id)
                .ok_or_else(|| anyhow!("Node id '{}' not found in graph", node_id))?;
            retain_indices.insert(*idx);
        }

        let mut new_graph = crate::graph::model::LabeledGraph::with_capacity(
            retain_indices.len(),
            retain_indices.len(),
        );
        let mut node_lookup = IndexMap::new();
        let mut reverse_lookup = IndexMap::new();
        let mut index_mapping: IndexMap<NodeIndex, NodeIndex> = IndexMap::new();

        for idx in &retain_indices {
            if let Some(weight) = graph.graph.node_weight(*idx) {
                let new_idx = new_graph.add_node(weight.clone());
                let node_id = graph
                    .reverse_lookup
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| idx.index().to_string());
                node_lookup.insert(node_id.clone(), new_idx);
                reverse_lookup.insert(new_idx, node_id);
                index_mapping.insert(*idx, new_idx);
            }
        }

        for edge in graph.graph.edge_references() {
            if let (Some(&new_source), Some(&new_target)) = (
                index_mapping.get(&edge.source()),
                index_mapping.get(&edge.target()),
            ) {
                new_graph.add_edge(new_source, new_target, edge.weight().clone());
            }
        }

        Ok(GraphInstance {
            graph: new_graph,
            node_lookup,
            reverse_lookup,
            graph_attributes: graph.graph_attributes.clone(),
            directed: graph.directed,
        })
    }

    /// Map external node ids to internal indexes. Will be used by candidate generation.
    pub fn resolve_node_id(graph: &GraphInstance, node_id: &GraphId) -> Option<NodeIndex> {
        graph.node_lookup.get(node_id).copied()
    }

    fn from_raw_graph(raw: RawGraph) -> Result<GraphInstance> {
        let node_count = raw.nodes.len();
        let mut graph =
            crate::graph::model::LabeledGraph::with_capacity(node_count, raw.edges.len());
        let mut node_lookup = IndexMap::new();
        let mut reverse_lookup = IndexMap::new();

        for raw_node in raw.nodes {
            let mut attributes = raw_node.attributes;
            let label = extract_label(&mut attributes);
            let weight = extract_weight(&mut attributes);
            let node_attr = NodeAttributes {
                label,
                weight,
                extra: attributes,
            };
            let idx = graph.add_node(node_attr);
            node_lookup.insert(raw_node.id.clone(), idx);
            reverse_lookup.insert(idx, raw_node.id);
        }

        for raw_edge in raw.edges {
            let source_idx = *node_lookup
                .get(&raw_edge.source)
                .ok_or_else(|| anyhow!("Unknown source node id: {}", raw_edge.source))?;
            let target_idx = *node_lookup
                .get(&raw_edge.target)
                .ok_or_else(|| anyhow!("Unknown target node id: {}", raw_edge.target))?;

            let mut attributes = raw_edge.attributes;
            let weight = extract_weight(&mut attributes);
            let edge_attr = EdgeAttributes {
                weight,
                extra: attributes,
            };
            graph.add_edge(source_idx, target_idx, edge_attr.clone());
            if !raw.directed && source_idx != target_idx {
                graph.add_edge(target_idx, source_idx, edge_attr);
            }
        }

        Ok(GraphInstance {
            graph,
            node_lookup,
            reverse_lookup,
            graph_attributes: raw.graph_attributes,
            directed: raw.directed,
        })
    }
}

fn extract_label(attrs: &mut IndexMap<String, serde_json::Value>) -> Option<String> {
    attrs.shift_remove("label").and_then(value_to_string)
}

fn extract_weight(attrs: &mut IndexMap<String, serde_json::Value>) -> Option<f64> {
    attrs.shift_remove("weight").and_then(|value| match value {
        serde_json::Value::Number(num) => num.as_f64(),
        serde_json::Value::String(s) => s.parse::<f64>().ok(),
        serde_json::Value::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
        _ => None,
    })
}

fn value_to_string(value: serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Number(num) => Some(num.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Null => None,
        other => Some(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph_json() -> String {
        r#"{
            "directed": false,
            "nodes": [
                {"id": "u", "attributes": {"label": "U"}},
                {"id": "v", "attributes": {"label": "V"}},
                {"id": "w", "attributes": {"label": "W"}}
            ],
            "edges": [
                {"source": "u", "target": "v", "attributes": {"weight": 1.0}},
                {"source": "v", "target": "w", "attributes": {}}
            ]
        }"#
        .to_string()
    }

    #[test]
    fn load_json_graph_counts_match() {
        let graph = GraphLoader::from_json_str(&sample_graph_json()).expect("load graph");
        assert_eq!(graph.graph.node_count(), 3);
        assert_eq!(
            graph.graph.edge_count(),
            4,
            "undirected edges should be duplicated"
        );
        assert!(graph.node_lookup.contains_key("u"));
        assert!(graph.reverse_lookup.values().any(|id| id == "w"));
    }

    #[test]
    fn induced_subgraph_preserves_structure() {
        let graph = GraphLoader::from_json_str(&sample_graph_json()).expect("load graph");
        let mut nodes = IndexSet::new();
        nodes.insert("u".to_string());
        nodes.insert("v".to_string());
        let subgraph = GraphLoader::induced_subgraph(&graph, &nodes).expect("subgraph");
        assert_eq!(subgraph.graph.node_count(), 2);
        assert_eq!(subgraph.graph.edge_count(), 2);
        assert!(subgraph.node_lookup.contains_key("u"));
    }
}
