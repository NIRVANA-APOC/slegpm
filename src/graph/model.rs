use indexmap::IndexMap;
use petgraph::{graph::Graph, prelude::NodeIndex};
use serde::{Deserialize, Serialize};

pub type GraphId = String;

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct NodeAttributes {
    pub label: Option<String>,
    pub weight: Option<f64>,
    #[serde(flatten)]
    pub extra: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct EdgeAttributes {
    pub weight: Option<f64>,
    #[serde(flatten)]
    pub extra: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawGraph {
    pub nodes: Vec<RawNode>,
    pub edges: Vec<RawEdge>,
    #[serde(default)]
    pub graph_attributes: IndexMap<String, serde_json::Value>,
    #[serde(default)]
    pub directed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawNode {
    pub id: GraphId,
    #[serde(default)]
    pub attributes: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEdge {
    pub source: GraphId,
    pub target: GraphId,
    #[serde(default)]
    pub attributes: IndexMap<String, serde_json::Value>,
}

pub type PatternGraph = GraphInstance;
pub type TargetGraph = GraphInstance;

pub type LabeledGraph = Graph<NodeAttributes, EdgeAttributes>;

#[derive(Debug, Clone)]
pub struct GraphInstance {
    pub graph: LabeledGraph,
    pub node_lookup: IndexMap<GraphId, NodeIndex>,
    pub reverse_lookup: IndexMap<NodeIndex, GraphId>,
    pub graph_attributes: IndexMap<String, serde_json::Value>,
    pub directed: bool,
}
