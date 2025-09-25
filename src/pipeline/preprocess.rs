use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use log::{debug, info};
use nalgebra::DMatrix;

use crate::graph::construction::GraphLoader;
use crate::graph::model::GraphInstance;
use crate::spectral::features::SpectralProfile;
use crate::spectral::{NormalizedLaplacianBuilder, SpectralVectorExtractor};

#[derive(Debug, Clone)]
pub struct PreprocessedGraph {
    pub instance: GraphInstance,
    pub laplacian: DMatrix<f64>,
    pub spectral: SpectralProfile,
}

impl PreprocessedGraph {
    pub fn node_count(&self) -> usize {
        self.instance.graph.node_count()
    }
}

#[derive(Debug, Default)]
pub struct GraphPreprocessor;

impl GraphPreprocessor {
    pub fn load_from_path(path: &Path) -> Result<PreprocessedGraph> {
        info!("Offline preprocessing: loading graph from {:?}", path);
        let file = File::open(path).with_context(|| format!("Failed to open {:?}", path))?;
        let instance = GraphLoader::from_reader(file)?;
        Self::preprocess_graph(instance)
    }

    pub fn preprocess_graph(instance: GraphInstance) -> Result<PreprocessedGraph> {
        info!(
            "Offline preprocessing: computing spectral features (|V|={}, |E|={})",
            instance.graph.node_count(),
            instance.graph.edge_count()
        );
        let laplacian = NormalizedLaplacianBuilder::build(&instance)?;
        let spectral = SpectralVectorExtractor::compute_profile(&laplacian)?;
        debug!(
            "Offline preprocessing complete: eigen spectrum length {}",
            spectral.eigenvalues.len()
        );
        Ok(PreprocessedGraph {
            instance,
            laplacian,
            spectral,
        })
    }
}
