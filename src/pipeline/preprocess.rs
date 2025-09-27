use std::fmt;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};

use once_cell::sync::OnceCell;

use crate::cache::{CacheMetadata, CachedPreprocessEntry, PreprocessCache};
use crate::graph::{GraphInstance, GraphLoader};
use crate::spectral::{normalized_laplacian, spectral_profile, LaplacianMatrix, SpectralProfile};

#[derive(Debug, Clone)]
pub struct PreprocessedGraph {
    pub instance: Arc<GraphInstance>,
    laplacian: LazyValue<LaplacianMatrix>,
    spectral: LazyValue<SpectralProfile>,
}

impl PreprocessedGraph {
    pub fn node_count(&self) -> usize {
        self.instance.node_count()
    }

    pub fn graph(&self) -> &GraphInstance {
        &self.instance
    }

    pub fn instance_arc(&self) -> Arc<GraphInstance> {
        Arc::clone(&self.instance)
    }

    pub fn laplacian(&self) -> Result<Arc<LaplacianMatrix>> {
        self.laplacian.get()
    }

    pub fn spectral(&self) -> Result<Arc<SpectralProfile>> {
        self.spectral.get()
    }
}

#[derive(Debug, Default)]
pub struct GraphPreprocessor;

impl GraphPreprocessor {
    pub fn from_path(path: &Path, spectral_dim: usize) -> Result<PreprocessedGraph> {
        let instance =
            GraphLoader::from_path(path).with_context(|| format!("open graph file {:?}", path))?;
        let cache = PreprocessCache::default();
        let metadata = CacheMetadata::from_path(path).ok();

        if let Some(meta) = metadata.as_ref() {
            if let Some(entry) = cache.load(path, spectral_dim, meta)? {
                return Self::from_cache(instance, cache, entry);
            }
        }

        let preprocessed = Self::from_instance(instance, spectral_dim)?;
        if let Some(meta) = metadata.as_ref() {
            let spectral = preprocessed
                .spectral()
                .with_context(|| "access spectral profile for caching")?;
            let laplacian = preprocessed
                .laplacian()
                .with_context(|| "access laplacian for caching")?;
            cache.store(
                path,
                spectral_dim,
                meta,
                spectral.as_ref(),
                laplacian.as_ref(),
            )?;
        }
        Ok(preprocessed)
    }

    pub fn from_instance(
        instance: GraphInstance,
        spectral_dim: usize,
    ) -> Result<PreprocessedGraph> {
        let laplacian = normalized_laplacian(&instance)?;
        let spectral = spectral_profile(&laplacian, spectral_dim.max(1))?;
        Ok(Self::assemble(instance, laplacian, spectral))
    }

    fn from_cache(
        instance: GraphInstance,
        cache: PreprocessCache,
        entry: CachedPreprocessEntry,
    ) -> Result<PreprocessedGraph> {
        let cache_for_laplacian = cache.clone();
        let laplacian_entry = entry.clone();
        let laplacian = LazyValue::deferred(move || {
            cache_for_laplacian
                .load_laplacian(&laplacian_entry)
                .with_context(|| "load cached laplacian")
        });

        let cache_for_spectral = cache.clone();
        let spectral_entry = entry.clone();
        let spectral = LazyValue::deferred(move || {
            cache_for_spectral
                .load_spectral(&spectral_entry)
                .with_context(|| "load cached spectral profile")
        });

        Ok(Self::assemble_lazy(instance, laplacian, spectral))
    }

    fn assemble(
        instance: GraphInstance,
        laplacian: LaplacianMatrix,
        spectral: SpectralProfile,
    ) -> PreprocessedGraph {
        Self::assemble_lazy(
            instance,
            LazyValue::ready(laplacian),
            LazyValue::ready(spectral),
        )
    }

    fn assemble_lazy(
        instance: GraphInstance,
        laplacian: LazyValue<LaplacianMatrix>,
        spectral: LazyValue<SpectralProfile>,
    ) -> PreprocessedGraph {
        PreprocessedGraph {
            instance: Arc::new(instance),
            laplacian,
            spectral,
        }
    }
}

#[derive(Clone)]
struct LazyValue<T> {
    cell: Arc<OnceCell<Arc<T>>>,
    loader: Arc<LazyLoader<T>>,
}

enum LazyLoader<T> {
    Ready(Arc<T>),
    Deferred(Arc<dyn Fn() -> Result<T> + Send + Sync>),
}

impl<T> LazyValue<T>
where
    T: Send + Sync + 'static,
{
    fn ready(value: T) -> Self {
        let arc = Arc::new(value);
        let cell = Arc::new(OnceCell::new());
        let _ = cell.set(Arc::clone(&arc));
        Self {
            cell,
            loader: Arc::new(LazyLoader::Ready(arc)),
        }
    }

    fn deferred<F>(loader: F) -> Self
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        Self {
            cell: Arc::new(OnceCell::new()),
            loader: Arc::new(LazyLoader::Deferred(Arc::new(loader))),
        }
    }

    fn get(&self) -> Result<Arc<T>> {
        match self.loader.as_ref() {
            LazyLoader::Ready(value) => Ok(Arc::clone(value)),
            LazyLoader::Deferred(loader) => {
                let loader = Arc::clone(loader);
                self.cell
                    .get_or_try_init(|| loader().map(Arc::new))
                    .map(Arc::clone)
            }
        }
    }
}

impl<T> fmt::Debug for LazyValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let initialized = self.cell.get().is_some();
        f.debug_struct("LazyValue")
            .field("initialized", &initialized)
            .finish()
    }
}
