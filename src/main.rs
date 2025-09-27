use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use log::info;

use slegpm::{
    GraphPreprocessor, GraphWriter, MatchingWorkflow, PatternSampler, PreprocessedGraph,
    SampleConfig, WorkflowConfig,
};

fn init_logging() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .try_init();
}

fn parse_args() -> Result<Vec<String>> {
    let mut args = env::args().skip(1);
    let dataset = args.next();
    if let Some(extra) = args.next() {
        anyhow::bail!("Unexpected extra argument: {extra}");
    }

    if let Some(dataset) = dataset {
        Ok(vec![dataset])
    } else {
        let entries = fs::read_dir("datasets").context("list datasets directory")?;
        let mut datasets: Vec<String> = entries
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.is_dir() && path.join("data_graph.json").is_file() {
                    Some(entry.file_name().to_string_lossy().into_owned())
                } else {
                    None
                }
            })
            .collect();
        datasets.sort();
        if datasets.is_empty() {
            anyhow::bail!("No datasets available under datasets");
        }
        Ok(datasets)
    }
}

fn main() -> Result<()> {
    init_logging();
    let datasets = parse_args()?;
    let config = WorkflowConfig::default();

    info!("Starting offline stage");
    let mut offline_datasets = Vec::with_capacity(datasets.len());
    for dataset in &datasets {
        info!("Dataset {}: offline stage start", dataset);
        let offline = prepare_offline(dataset, &config)?;
        offline_datasets.push((dataset.clone(), offline));
    }
    info!("Offline stage completed");

    info!("Starting online stage");
    for (dataset, offline) in &offline_datasets {
        info!("Dataset {}: online stage start", dataset);
        execute_online(dataset, &config, offline)?;
    }
    info!("Online stage completed");

    cleanup_cache()
}

struct OfflineDataset {
    target: Arc<PreprocessedGraph>,
    patterns: Vec<OfflinePattern>,
}

struct OfflinePattern {
    path: PathBuf,
    graph: Arc<PreprocessedGraph>,
}

fn prepare_offline(dataset: &str, config: &WorkflowConfig) -> Result<OfflineDataset> {
    let offline_start = Instant::now();
    let dataset_root = Path::new("datasets").join(dataset);
    let target_path = dataset_root.join("data_graph.json");
    let patterns_dir = dataset_root.join("patterns");

    fs::create_dir_all(&patterns_dir)
        .with_context(|| format!("ensure pattern directory {:?}", patterns_dir))?;

    let target = GraphPreprocessor::from_path(&target_path, config.spectral_dim)
        .with_context(|| format!("preprocess target graph at {:?}", target_path))?;
    let target = Arc::new(target);
    let target_graph = target.graph();
    info!(
        "Dataset {}: target graph nodes {}, edges {}",
        dataset,
        target_graph.node_count(),
        target_graph.edge_count()
    );

    let mut pattern_files = collect_pattern_files(&patterns_dir)
        .with_context(|| format!("enumerate pattern graphs in {:?}", patterns_dir))?;
    if pattern_files.is_empty() {
        sample_patterns(&patterns_dir, target.graph())
            .with_context(|| format!("auto-sample patterns into {:?}", patterns_dir))?;
        pattern_files = collect_pattern_files(&patterns_dir)
            .with_context(|| format!("enumerate pattern graphs in {:?}", patterns_dir))?;
    }
    if pattern_files.is_empty() {
        anyhow::bail!(
            "Failed to prepare any pattern graphs under {:?}",
            patterns_dir
        );
    }

    target
        .spectral()
        .with_context(|| format!("precompute spectral profile for {:?}", target_path))?;
    target
        .laplacian()
        .with_context(|| format!("precompute laplacian for {:?}", target_path))?;

    let mut patterns = Vec::with_capacity(pattern_files.len());
    for path in pattern_files {
        let preprocessed = GraphPreprocessor::from_path(&path, config.spectral_dim)
            .with_context(|| format!("preprocess pattern graph at {:?}", path))?;
        let graph = Arc::new(preprocessed);
        graph
            .spectral()
            .with_context(|| format!("precompute spectral profile for {:?}", path))?;
        graph
            .laplacian()
            .with_context(|| format!("precompute laplacian for {:?}", path))?;
        patterns.push(OfflinePattern { path, graph });
    }

    let offline_duration = offline_start.elapsed();
    info!(
        "Dataset {} offline duration {:?} (patterns {})",
        dataset,
        offline_duration,
        patterns.len()
    );

    Ok(OfflineDataset { target, patterns })
}

fn execute_online(dataset: &str, config: &WorkflowConfig, offline: &OfflineDataset) -> Result<()> {
    let mut total_stats = AggregatedStats::default();

    for pattern in &offline.patterns {
        let workflow = MatchingWorkflow::new(
            config.clone(),
            Arc::clone(&pattern.graph),
            Arc::clone(&offline.target),
        );
        let summary = workflow.execute()?;

        info!(
            "Pattern {:?}: matches {} (spectral {}, wl {}, fingerprint rejects {}, vf2 calls {})",
            pattern
                .path
                .file_name()
                .unwrap_or_else(|| OsStr::new("<unknown>")),
            summary.stats.matches,
            summary.stats.spectral_pass,
            summary.stats.wl_pass,
            summary.stats.fingerprint_rejects,
            summary.stats.vf2_calls
        );
        info!(
            "Pattern {:?}: online {:?}, matching {:?}",
            pattern
                .path
                .file_name()
                .unwrap_or_else(|| OsStr::new("<unknown>")),
            summary.online_duration,
            summary.matching_duration
        );

        total_stats.accumulate(&summary);
    }

    let averages = total_stats
        .averages()
        .context("compute average statistics")?;
    // Average metrics: matches (higher is better), spectral_pass (higher is better)
    // wl_pass (higher is better), vf2_calls (lower is better)
    info!(
        "Dataset {} average: matches {:.2}, spectral {:.2}, wl {:.2}, fingerprint rejects {:.2}, vf2 calls {:.2}",
        dataset,
        averages.matches,
        averages.spectral_pass,
        averages.wl_pass,
        averages.fingerprint_rejects,
        averages.vf2_calls
    );
    // Average durations: online phase and matching phase (lower is better)
    info!(
        "Dataset {} average durations: online {:?}, matching {:?}",
        dataset, averages.online_duration, averages.matching_duration
    );

    Ok(())
}

fn cleanup_cache() -> Result<()> {
    let cache_dir = Path::new("cache");
    if cache_dir.exists() {
        fs::remove_dir_all(cache_dir)
            .with_context(|| format!("remove cache directory {:?}", cache_dir))?;
        info!("Cleared cache directory at {:?}", cache_dir);
    }
    Ok(())
}

fn collect_pattern_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = fs::read_dir(dir)?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(OsStr::to_str)
                    .map(|ext| ext.eq_ignore_ascii_case("json"))
                    .unwrap_or(false)
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn sample_patterns(dir: &Path, target_graph: &slegpm::GraphInstance) -> Result<()> {
    let nodes = target_graph.node_count().min(6).max(1);
    let samples = PatternSampler::sample(
        target_graph,
        SampleConfig {
            nodes,
            samples: 10,
            seed: Some(42),
        },
    )?;

    for (idx, pattern) in samples.into_iter().enumerate() {
        let path = dir.join(format!("pattern_auto_{idx:02}.json"));
        GraphWriter::write_to_path(&pattern, &path)
            .with_context(|| format!("write sampled pattern to {:?}", path))?;
    }
    Ok(())
}

#[derive(Default)]
struct AggregatedStats {
    matches: f64,
    spectral_pass: f64,
    wl_pass: f64,
    vf2_calls: f64,
    fingerprint_rejects: f64,
    total_online: DurationAggregate,
    total_matching: DurationAggregate,
    count: usize,
}

impl AggregatedStats {
    fn accumulate(&mut self, summary: &slegpm::MatchingSummary) {
        self.matches += summary.stats.matches as f64;
        self.spectral_pass += summary.stats.spectral_pass as f64;
        self.wl_pass += summary.stats.wl_pass as f64;
        self.vf2_calls += summary.stats.vf2_calls as f64;
        self.fingerprint_rejects += summary.stats.fingerprint_rejects as f64;
        self.total_online.add(summary.online_duration);
        self.total_matching.add(summary.matching_duration);
        self.count += 1;
    }

    fn averages(&self) -> Result<AveragedStats> {
        if self.count == 0 {
            anyhow::bail!("no pattern statistics accumulated");
        }
        let count = self.count as f64;
        Ok(AveragedStats {
            matches: self.matches / count,
            spectral_pass: self.spectral_pass / count,
            wl_pass: self.wl_pass / count,
            vf2_calls: self.vf2_calls / count,
            fingerprint_rejects: self.fingerprint_rejects / count,
            online_duration: self.total_online.average(count),
            matching_duration: self.total_matching.average(count),
        })
    }
}

struct DurationAggregate {
    micros: f64,
}

impl Default for DurationAggregate {
    fn default() -> Self {
        Self { micros: 0.0 }
    }
}

impl DurationAggregate {
    fn add(&mut self, duration: std::time::Duration) {
        self.micros += duration.as_secs_f64() * 1_000_000.0;
    }

    fn average(&self, count: f64) -> std::time::Duration {
        let avg_micros = if count > 0.0 {
            self.micros / count
        } else {
            0.0
        };
        std::time::Duration::from_secs_f64(avg_micros / 1_000_000.0)
    }
}

struct AveragedStats {
    matches: f64,
    spectral_pass: f64,
    wl_pass: f64,
    vf2_calls: f64,
    fingerprint_rejects: f64,
    online_duration: std::time::Duration,
    matching_duration: std::time::Duration,
}
