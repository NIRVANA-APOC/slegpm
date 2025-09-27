use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::graph::{GraphInstance, GraphLoader};

const DEFAULT_ROOT: &str = "datasets";

#[derive(Debug, Clone)]
pub struct DatasetLoader {
    root: PathBuf,
}

impl DatasetLoader {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn default() -> Self {
        Self::new(DEFAULT_ROOT)
    }

    pub fn with_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.root = root.into();
        self
    }

    pub fn load(&self, relative: impl AsRef<Path>) -> Result<GraphInstance> {
        let path = self.root.join(relative);
        GraphLoader::from_path(&path).with_context(|| format!("load dataset from {:?}", path))
    }
}
