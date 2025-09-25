# Spectral-Laplacian Enhanced Graph Pattern Matching (SLEGPM)

SLEGPM implements the spectral vector dominance pipeline described in the accompanying technical report. Offline preprocessing computes reusable spectral features, while the online stage performs anchor-driven candidate generation, spectral/WL pruning, dominance verification, and iterative matching. The code base is written in Rust and targets medium-scale biological or bibliographic graphs.

## Quick Start

```bash
# Build and run with the bundled defaults
cargo run --release

# Provide explicit pattern/target paths and anchor count
cargo run --release -- <pattern.json> <target.json> <anchor_count>

# Execute the end-to-end smoke test
env RUST_LOG=info cargo test --test workflow_smoke -- --nocapture
```

The CLI defaults point to the `datasets/yeast` sample set. The program prints offline/online timings and a detailed audit log for the matching phase.

## Data Format

Graphs are stored as JSON documents matching `datasets/yeast/template_graph.json`:

```json
{
  "directed": false,
  "graph_attributes": { ... },
  "nodes": [
    { "id": "A", "attributes": { "label": "Gene", "weight": 1.0, ... } }
  ],
  "edges": [
    { "source": "A", "target": "B", "attributes": { "weight": 0.5, ... } }
  ]
}
```

* Node and edge attribute maps accept arbitrary key/value pairs. Unknown keys are preserved under the `extra` field when loaded.
* Missing weights fall back to `1.0`. For undirected graphs the loader inserts symmetric edges automatically.

## Architecture Overview

**Offline phase**
1. Parse graph JSON into an attributed `petgraph::Graph` (`GraphLoader`).
2. Build the normalized Laplacian \( \hat{L} = D^{-1/2} L D^{-1/2} \).
3. Extract spectral features (leading eigenvalues, L2 norm, trace) via dense or Lanczos solvers.

**Online phase**
1. Select anchors in the pattern graph and locate candidate nodes in the target graph.
2. Expand anchor neighborhoods to obtain candidate subgraphs.
3. Run parallel spectral dominance and Weisfeiler-Lehman (WL) pruning to discard infeasible candidates.
4. Check Cauchy interlacing and dominance ratios to retain only candidates consistent with the report's theoretical guarantees.
5. Iteratively refine candidates, generate precise subgraphs, and finish with VF2 (petgraph) to confirm subgraph isomorphism.

Logging is emitted for each milestone and can be tuned with `RUST_LOG`.

## Module Reference

| Module | Responsibilities | Key Techniques | Third-party crates | Report mapping |
| --- | --- | --- | --- | --- |
| `src/main.rs` | CLI entry point; logging init; argument parsing; end-to-end orchestration | Default path overrides, anchor override | `anyhow`, `log`, `env_logger` | Aligns with the report's "offline preprocessing + online matching" execution harness |
| `graph::construction` | Load JSON graphs into `GraphInstance` with attribute preservation | Stable ID <-> index maps; automatic undirected edge duplication | `serde_json`, `anyhow`, `indexmap`, `petgraph` | Data loading stage prior to spectral analysis |
| `graph::model` | Node/edge attribute structs and the `GraphInstance` container | `serde` derives; `extra` map for opaque fields | `serde`, `indexmap`, `petgraph` | Canonical graph state used throughout the algorithm |
| `graph::serialization` | Export `GraphInstance` back to JSON | Deterministic node/edge ordering; optional weight handling | `serde_json`, `indexmap`, `anyhow` | Supports the report's "sample-and-verify" experiments |
| `spectral::normalized_laplacian` | Construct normalized Laplacian matrices | Degree matrix computation, symmetric assembly | `nalgebra`, `anyhow` | Implements "compute Laplacian matrix" (offline reusable step) |
| `spectral::spectral_vector` | Compute spectral profiles | Dense `SymmetricEigen`; Lanczos with multi-seed restarts; parallel reductions | `nalgebra`, `ndarray`, `rayon`, `rand`, `rand_xoshiro`, `anyhow` | Implements "spectral vector extraction" with Lanczos acceleration as proposed |
| `spectral::features` | `SpectralProfile` container | Automatic norm and trace bookkeeping | `ndarray` | Stores spectral indicators for dominance reasoning |
| `pipeline::preprocess` | Offline workflow wrapper | Structured logging around load -> Laplacian -> spectral steps | `log`, `anyhow`, `nalgebra` | Matches the report's offline preprocessing block |
| `pipeline::candidate` | Anchor selection, candidate filtering, subgraph induction | Degree/attribute scoring; BFS expansions; parallel filtering | `anyhow`, `indexmap`, `rayon`, `petgraph` | Implements candidate generation / iteration seeds |
| `pipeline::matching` | Spectral/WL pruning, dominance checks, iterative refinement, VF2 fallback | Parallel candidate evaluation; WL histogram comparison; audit reporting | `rayon`, `log`, `petgraph`, `anyhow`, `indexmap` | Aggregates the report's "coarse filter -> precise filter -> VF2 verification" pipeline |
| `pipeline::workflow` | Online coordinator, duration tracking, audit aggregation | Anchor -> candidate -> refinement -> matching pipeline | `log` | Ties together the online phase described in the report |
| `sampling::pattern_sampler` | Sample connected pattern graphs from a large target | Randomized BFS with deduplication and seeding | `rand`, `indexmap`, `petgraph`, `anyhow` | Implements the report's self-validation sampling experiments |
| `verify::interlacing` | Cauchy interlacing checks with explanations | Bounds logging, epsilon tolerances | (std, `SpectralProfile`) | Verifies necessary spectral dominance condition |
| `verify::dominance` | Combine interlacing, norm, trace checks into a scored report | Weighted satisfaction ratio; human-readable audit | (std, `SpectralProfile`) | Encodes the dominance heuristic discussed in the report |
| `tests::workflow_smoke` | End-to-end regression test | Dataset-backed sampling, workflow execution, VF2 assertions | `petgraph` | Confirms the pipeline recovers sampled patterns |

Exports are centralised in `src/lib.rs` and `src/pipeline/mod.rs` to ease reuse in tooling or experiments.

## Dependency Inventory

| Crate | Purpose |
| --- | --- |
| `anyhow` | Ergonomic error propagation |
| `env_logger` / `log` | Structured progress logging |
| `indexmap` | Stable maps/sets for deterministic signatures |
| `nalgebra` | Matrix algebra, eigen decomposition |
| `ndarray` | Lightweight spectral vector storage |
| `petgraph` | Graph representation and VF2 matching |
| `rand`, `rand_xoshiro` | Random sampling and Lanczos seeds |
| `rayon` | Parallel candidate and spectral processing |
| `serde`, `serde_json` | JSON I/O for graph data |
| `thiserror` | Reserved for custom error enums |

## Mapping to the Original Algorithm

| Report step | Code implementation | Notes |
| --- | --- | --- |
| Graph ingestion & normalization | `graph::construction`, `pipeline::preprocess` | Preserves rich attributes; ensures Laplacian correctness |
| Normalized Laplacian construction | `spectral::normalized_laplacian` | Direct translation of \( D^{-1/2} L D^{-1/2} \) |
| Spectral vector computation (offline) | `spectral::spectral_vector` | Dense solver for small graphs; Lanczos for large graphs | 
| Anchor selection & candidate nodes | `pipeline::candidate::AnchorSelector` / `CandidateGenerator` | Mirrors heuristic anchor-driven filtering |
| Candidate subgraph expansion | `pipeline::candidate::SubgraphExtractor` | Ensures connectivity and bounded size |
| Spectral dominance coarse filter | `pipeline::matching::spectral_dominates` | Implements component-wise dominance with norm/trace checks |
| WL signature pruning | `pipeline::matching::compute_wl_signature` and related filters | Multi-iteration WL histogram comparisons |
| Dominance & interlacing checks | `verify::{interlacing, dominance}` | Emits audited explanations for each candidate |
| Iterative precise filtering | `MatchingOrchestrator::run_iterative_matching` | Generates precise candidates, tracks statistics, controls fallback |
| VF2 exact verification | `pipeline::matching::run_vf2` | Only invoked after passing custom filters |
| Sampling & self-validation | `sampling::pattern_sampler`, `tests::workflow_smoke` | Matches the report's evaluation methodology |

## Testing and Validation

`tests/workflow_smoke.rs` performs an integration test that:
1. Samples two connected patterns from `datasets/yeast/data_graph.json`.
2. Runs the full workflow with logging enabled.
3. Asserts that each sampled pattern yields at least one match.
4. Uses VF2 with custom attribute comparators to confirm subgraph isomorphism.

Run this test after tuning thresholds or heuristics to ensure the pipeline still recovers sampled patterns.

## Logging and Performance Notes

* Set `RUST_LOG=debug` to inspect spectral/WL pruning decisions, dominance reports, and VF2 outcomes.
* Parallelism (`rayon`) feeds candidate generation, spectral extraction, and Lanczos restarts.
* `MatchingWorkflow::execute` returns both the total online duration and the matching-phase duration to gauge optimisation progress.

## Suggested Next Steps

1. Calibrate WL tolerance and spectral equivalence thresholds so that sampled-pattern tests pass while maintaining pruning strength.
2. Add persistence (e.g., disk caches) for offline spectral profiles across runs.
3. Experiment with advanced candidate enumeration or incremental refinement strategies for very large targets.

Consult the "Mapping to the Original Algorithm" table above to trace each report requirement back to its concrete implementation.
