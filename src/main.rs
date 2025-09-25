use std::path::PathBuf;

use anyhow::Result;
use log::info;
use slegpm::{GraphPreprocessor, MatchingWorkflow, WorkflowConfig};

const DEFAULT_PATTERN_PATH: &str =
    r#"D:\0A实验室\Me\WWW2026投稿\slegpm\datasets\yeast\test_data\pattern_0.json"#;
const DEFAULT_TARGET_PATH: &str =
    r#"D:\0A实验室\Me\WWW2026投稿\slegpm\datasets\yeast\data_graph.json"#;
const DEFAULT_ANCHOR_OVERRIDE: usize = 3;

fn init_logging() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .try_init();
}

fn parse_args() -> (PathBuf, PathBuf, usize) {
    let mut args = std::env::args().skip(1);
    let pattern_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PATTERN_PATH));
    let target_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_TARGET_PATH));
    let anchor_override = args
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ANCHOR_OVERRIDE);
    (pattern_path, target_path, anchor_override)
}

fn main() -> Result<()> {
    init_logging();

    let (pattern_path, target_path, anchor_override) = parse_args();

    info!("Pattern graph path: {:?}", pattern_path);
    info!("Target graph path: {:?}", target_path);
    info!("Anchor override: {}", anchor_override);

    info!("Starting offline preprocessing (pattern)");
    let pattern = GraphPreprocessor::load_from_path(&pattern_path)?;
    info!("Starting offline preprocessing (target)");
    let target = GraphPreprocessor::load_from_path(&target_path)?;

    let config = WorkflowConfig {
        anchor_count: Some(anchor_override),
        epsilon: 1e-3,
        ..WorkflowConfig::default()
    };

    let workflow = MatchingWorkflow::new(config, pattern, target);
    let summary = workflow.execute()?;

    for (idx, entry) in summary.audit_log.iter().enumerate() {
        println!("\n--------------------------------------------------------------------------");
        println!("Candidate {}:\n{}", idx + 1, entry);
    }

    println!("Matched subgraphs: {}", summary.matches.len());
    println!("Online duration: {:?}", summary.online_duration);
    println!("Matching duration: {:?}", summary.matching_duration);

    Ok(())
}
