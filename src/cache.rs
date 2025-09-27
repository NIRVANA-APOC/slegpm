use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::spectral::{LaplacianMatrix, SpectralProfile};

const CACHE_DIR: &str = "cache";
const PREPROCESS_SUBDIR: &str = "preprocessed";
const METADATA_FILE: &str = "meta.json";
const SPECTRAL_FILE: &str = "spectral.json";
const LAPLACIAN_FILE: &str = "laplacian.json";
const CACHE_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub len: u64,
    pub modified: u64,
}

impl CacheMetadata {
    pub fn from_path(path: &Path) -> Result<Self> {
        let metadata =
            fs::metadata(path).with_context(|| format!("read metadata for {:?}", path))?;
        let len = metadata.len();
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or_default();
        Ok(Self { len, modified })
    }

    pub fn matches(&self, other: &Self) -> bool {
        self.len == other.len && self.modified == other.modified
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessCache {
    root: PathBuf,
}

impl Default for PreprocessCache {
    fn default() -> Self {
        Self::new(CACHE_DIR)
    }
}

impl PreprocessCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn load(
        &self,
        dataset_path: &Path,
        spectral_dim: usize,
        metadata: &CacheMetadata,
    ) -> Result<Option<CachedPreprocessEntry>> {
        let dir = self.cache_dir(dataset_path, spectral_dim)?;
        let meta_path = dir.join(METADATA_FILE);
        if !meta_path.exists() {
            return Ok(None);
        }

        let meta: CachedMeta = read_json(&meta_path)
            .with_context(|| format!("deserialize preprocess metadata from {:?}", meta_path))?;
        if meta.version != CACHE_VERSION
            || meta.spectral_dim != spectral_dim
            || !meta.metadata.matches(metadata)
        {
            return Ok(None);
        }

        let spectral_path = dir.join(SPECTRAL_FILE);
        let laplacian_path = dir.join(LAPLACIAN_FILE);
        if !spectral_path.exists() || !laplacian_path.exists() {
            return Ok(None);
        }

        Ok(Some(CachedPreprocessEntry::new(dir)))
    }

    pub fn store(
        &self,
        dataset_path: &Path,
        spectral_dim: usize,
        metadata: &CacheMetadata,
        spectral: &SpectralProfile,
        laplacian: &LaplacianMatrix,
    ) -> Result<()> {
        let dir = self.cache_dir(dataset_path, spectral_dim)?;
        fs::create_dir_all(&dir).with_context(|| format!("create cache directory {:?}", dir))?;

        let meta = CachedMeta {
            version: CACHE_VERSION,
            metadata: metadata.clone(),
            spectral_dim,
        };

        write_json(&dir.join(METADATA_FILE), &meta)
            .with_context(|| format!("write preprocess metadata to {:?}", dir))?;
        write_json(&dir.join(SPECTRAL_FILE), spectral)
            .with_context(|| format!("write cached spectral profile to {:?}", dir))?;
        write_json(&dir.join(LAPLACIAN_FILE), laplacian)
            .with_context(|| format!("write cached laplacian to {:?}", dir))?;

        Ok(())
    }

    pub fn load_spectral(&self, entry: &CachedPreprocessEntry) -> Result<SpectralProfile> {
        let path = entry.spectral_path();
        read_json(&path).with_context(|| format!("read cached spectral profile from {:?}", path))
    }

    pub fn load_laplacian(&self, entry: &CachedPreprocessEntry) -> Result<LaplacianMatrix> {
        let path = entry.laplacian_path();
        read_json(&path).with_context(|| format!("read cached laplacian from {:?}", path))
    }

    fn cache_dir(&self, dataset_path: &Path, spectral_dim: usize) -> Result<PathBuf> {
        let canonical = dataset_path
            .canonicalize()
            .unwrap_or_else(|_| dataset_path.to_path_buf());
        let mut hasher = blake3::Hasher::new();
        hasher.update(canonical.to_string_lossy().as_bytes());
        hasher.update(&spectral_dim.to_le_bytes());
        let hash = hasher.finalize();
        let dirname = hash.to_hex().to_string();
        Ok(self.root.join(PREPROCESS_SUBDIR).join(dirname))
    }
}

#[derive(Debug, Clone)]
pub struct CachedPreprocessEntry {
    dir: PathBuf,
}

impl CachedPreprocessEntry {
    fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    fn spectral_path(&self) -> PathBuf {
        self.dir.join(SPECTRAL_FILE)
    }

    fn laplacian_path(&self) -> PathBuf {
        self.dir.join(LAPLACIAN_FILE)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedMeta {
    version: u32,
    metadata: CacheMetadata,
    spectral_dim: usize,
}

fn read_json<T>(path: &Path) -> Result<T>
where
    T: DeserializeOwned,
{
    let file = File::open(path).with_context(|| format!("open cached json file {:?}", path))?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader)
        .with_context(|| format!("deserialize cached json file {:?}", path))
}

fn write_json<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create cache parent directory {:?}", parent))?;
    }
    let file = File::create(path).with_context(|| format!("create cache json file {:?}", path))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, value)
        .with_context(|| format!("serialize cache json file {:?}", path))
}
