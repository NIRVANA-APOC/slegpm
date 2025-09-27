use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use indexmap::IndexMap;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

pub type GraphId = String;

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct NodeAttributes {
    pub label: Option<String>,
    pub weight: Option<f64>,
    #[serde(default)]
    pub extra: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct EdgeAttributes {
    pub weight: Option<f64>,
    #[serde(default)]
    pub extra: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct GraphInstance {
    pub graph: Graph<NodeAttributes, EdgeAttributes>,
    pub node_lookup: IndexMap<GraphId, NodeIndex>,
    pub reverse_lookup: IndexMap<NodeIndex, GraphId>,
    pub graph_attributes: IndexMap<String, serde_json::Value>,
    pub directed: bool,
}

impl GraphInstance {
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn neighbors(&self, id: &GraphId) -> Result<Vec<GraphId>> {
        let idx = *self
            .node_lookup
            .get(id)
            .ok_or_else(|| anyhow!("Node '{id}' not found"))?;
        let mut around = Vec::new();
        for neighbor in self.graph.neighbors(idx) {
            if let Some(id) = self.reverse_lookup.get(&neighbor) {
                around.push(id.clone());
            }
        }
        Ok(around)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawGraph {
    pub nodes: Vec<RawNode>,
    pub edges: Vec<RawEdge>,
    #[serde(default)]
    pub graph_attributes: IndexMap<String, serde_json::Value>,
    #[serde(default)]
    pub directed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawNode {
    pub id: GraphId,
    #[serde(default)]
    pub attributes: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawEdge {
    pub source: GraphId,
    pub target: GraphId,
    #[serde(default)]
    pub attributes: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Default)]
pub struct GraphLoader;

impl GraphLoader {
    pub fn from_json_str(json: &str) -> Result<GraphInstance> {
        let raw: RawGraph = serde_json::from_str(json).context("parse graph json")?;
        Self::from_raw(raw)
    }

    pub fn from_reader<R: Read>(mut reader: R) -> Result<GraphInstance> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        Self::from_json_str(&buf)
    }

    pub fn from_path(path: &Path) -> Result<GraphInstance> {
        let file = File::open(path).with_context(|| format!("open graph file {:?}", path))?;
        Self::from_reader(file)
    }

    pub fn induced_subgraph(graph: &GraphInstance, nodes: &[GraphId]) -> Result<GraphInstance> {
        let mut retain = IndexMap::new();
        for id in nodes {
            let idx = graph
                .node_lookup
                .get(id)
                .ok_or_else(|| anyhow!("Node '{id}' not present in graph"))?;
            retain.insert(id.clone(), *idx);
        }

        let mut new_graph = Graph::with_capacity(retain.len(), retain.len());
        let mut node_lookup = IndexMap::new();
        let mut reverse_lookup = IndexMap::new();

        for (id, idx) in retain.iter() {
            let attrs = graph
                .graph
                .node_weight(*idx)
                .ok_or_else(|| anyhow!("Missing attributes for node '{id}'"))?;
            let new_idx = new_graph.add_node(attrs.clone());
            node_lookup.insert(id.clone(), new_idx);
            reverse_lookup.insert(new_idx, id.clone());
        }

        let mut seen = HashSet::new();
        for edge in graph.graph.edge_references() {
            let Some(source_id) = graph.reverse_lookup.get(&edge.source()) else {
                continue;
            };
            let Some(target_id) = graph.reverse_lookup.get(&edge.target()) else {
                continue;
            };
            if !node_lookup.contains_key(source_id) || !node_lookup.contains_key(target_id) {
                continue;
            }
            let new_source = node_lookup[source_id];
            let new_target = node_lookup[target_id];
            if graph.directed {
                new_graph.add_edge(new_source, new_target, edge.weight().clone());
                continue;
            }
            let mut key = [source_id.clone(), target_id.clone()];
            key.sort();
            if !seen.insert((key[0].clone(), key[1].clone())) {
                continue;
            }
            new_graph.add_edge(new_source, new_target, edge.weight().clone());
            if new_source != new_target {
                new_graph.add_edge(new_target, new_source, edge.weight().clone());
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

    pub fn resolve(graph: &GraphInstance, node_id: &GraphId) -> Option<NodeIndex> {
        graph.node_lookup.get(node_id).copied()
    }

    fn from_raw(raw: RawGraph) -> Result<GraphInstance> {
        let mut graph = Graph::with_capacity(raw.nodes.len(), raw.edges.len());
        let mut node_lookup = IndexMap::new();
        let mut reverse_lookup = IndexMap::new();

        for raw_node in raw.nodes {
            let (label, weight, extra) = split_node_attributes(raw_node.attributes);
            let attrs = NodeAttributes {
                label,
                weight,
                extra,
            };
            let idx = graph.add_node(attrs);
            node_lookup.insert(raw_node.id.clone(), idx);
            reverse_lookup.insert(idx, raw_node.id);
        }

        for raw_edge in raw.edges {
            let Some(source) = node_lookup.get(&raw_edge.source) else {
                return Err(anyhow!("Unknown source node '{}'", raw_edge.source));
            };
            let Some(target) = node_lookup.get(&raw_edge.target) else {
                return Err(anyhow!("Unknown target node '{}'", raw_edge.target));
            };
            let (weight, extra) = split_edge_attributes(raw_edge.attributes);
            let attrs = EdgeAttributes { weight, extra };
            graph.add_edge(*source, *target, attrs.clone());
            if !raw.directed && source != target {
                graph.add_edge(*target, *source, attrs);
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

pub struct GraphWriter;

impl GraphWriter {
    pub fn write_to_path(graph: &GraphInstance, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create graph parent directory {:?}", parent))?;
        }
        let raw = Self::to_raw_graph(graph);
        let json = serde_json::to_string_pretty(&raw)?;
        fs::write(path, json).with_context(|| format!("write graph to {:?}", path))?;
        Ok(())
    }

    fn to_raw_graph(graph: &GraphInstance) -> RawGraph {
        let nodes = graph
            .graph
            .node_indices()
            .map(|idx| RawNode {
                id: graph
                    .reverse_lookup
                    .get(&idx)
                    .expect("node id present")
                    .clone(),
                attributes: assemble_node_attributes(
                    graph.graph.node_weight(idx).expect("node weight"),
                ),
            })
            .collect();

        let mut edges = Vec::new();
        let mut seen = HashSet::new();
        for edge in graph.graph.edge_references() {
            let source = graph.reverse_lookup.get(&edge.source()).expect("source id");
            let target = graph.reverse_lookup.get(&edge.target()).expect("target id");
            if !graph.directed {
                let mut key = [source.clone(), target.clone()];
                key.sort();
                if !seen.insert((key[0].clone(), key[1].clone())) {
                    continue;
                }
            }
            edges.push(RawEdge {
                source: source.clone(),
                target: target.clone(),
                attributes: assemble_edge_attributes(edge.weight()),
            });
        }

        RawGraph {
            nodes,
            edges,
            graph_attributes: graph.graph_attributes.clone(),
            directed: graph.directed,
        }
    }
}

fn assemble_node_attributes(attrs: &NodeAttributes) -> IndexMap<String, serde_json::Value> {
    let mut map = attrs.extra.clone();
    if let Some(label) = &attrs.label {
        map.insert(
            "label".to_string(),
            serde_json::Value::String(label.clone()),
        );
    }
    if let Some(weight) = attrs.weight {
        if let Some(num) = serde_json::Number::from_f64(weight) {
            map.insert("weight".to_string(), serde_json::Value::Number(num));
        }
    }
    map
}

fn assemble_edge_attributes(attrs: &EdgeAttributes) -> IndexMap<String, serde_json::Value> {
    let mut map = attrs.extra.clone();
    if let Some(weight) = attrs.weight {
        if let Some(num) = serde_json::Number::from_f64(weight) {
            map.insert("weight".to_string(), serde_json::Value::Number(num));
        }
    }
    map
}

fn split_node_attributes(
    mut attributes: IndexMap<String, serde_json::Value>,
) -> (
    Option<String>,
    Option<f64>,
    IndexMap<String, serde_json::Value>,
) {
    let label = attributes
        .swap_remove("label")
        .and_then(|value| match value {
            serde_json::Value::String(s) => Some(s),
            serde_json::Value::Number(n) => Some(n.to_string()),
            serde_json::Value::Bool(b) => Some(b.to_string()),
            _ => None,
        });
    let weight = attributes
        .swap_remove("weight")
        .and_then(|value| match value {
            serde_json::Value::Number(n) => n.as_f64(),
            serde_json::Value::String(s) => s.parse::<f64>().ok(),
            serde_json::Value::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
            _ => None,
        });
    (label, weight, attributes)
}

fn split_edge_attributes(
    mut attributes: IndexMap<String, serde_json::Value>,
) -> (Option<f64>, IndexMap<String, serde_json::Value>) {
    let weight = attributes
        .swap_remove("weight")
        .and_then(|value| match value {
            serde_json::Value::Number(n) => n.as_f64(),
            serde_json::Value::String(s) => s.parse::<f64>().ok(),
            serde_json::Value::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
            _ => None,
        });
    (weight, attributes)
}
