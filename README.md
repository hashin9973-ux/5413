# 5413

5413 is an environment-independent adaptive learning runtime. Its long-term direction is an autonomous experimentalist: a system that learns predictive models of unfamiliar environments, selects informative experiments, plans actions under explicit safety constraints, and improves through measured evidence.

The current implementation includes:

- typed, domain-neutral environment contracts and schema-validated observations;
- explicit action authorization, with read-only default policy;
- append-only experience capture with a versioned, checksummed file journal, bounded decoding, torn-tail recovery, episode replay, and durable idempotency keys;
- a separately versioned append-only checkpoint log with torn-tail recovery, monotonic/idempotent snapshot writes, environment/schema binding, and deterministic callback-driven replay after a checkpoint;
- immutable, checksummed model-artifact persistence with versioned parameter serialization and restoration for built-in statistical and contextual models;
- durable representation-registry snapshots preserving schemas, versions, evaluation evidence, and candidate/active/rejected/retired lifecycle state;
- versioned, held-out-evaluated representations and explicit mappings;
- recursive structural profiling for nested objects and sequences;
- a pluggable predictive-model contract, an action-conditioned statistical model, and contextual ridge regression;
- empirical model ensembles that report within-model spread and between-model disagreement separately;
- held-out prediction evaluation with a dataset-ID leakage guard;
- disagreement-based experiment ranking with declared action-risk, utility, and cost constraints;
- bounded beam-search planning that scores predicted states and checks candidate actions against the environment's action policy;
- a receding-horizon runtime that replans from fresh observations, durably journals action intents before adapter effects, reauthorizes before execution, and records prediction error against observed outcomes;
- episode-scoped transition extraction that reconstructs completed `(state, action, next_state)` samples from journal events and excludes incomplete or mismatched actions;
- an auditable hypothesis registry with typed claims, unique evidence references, log-likelihood-ratio updates, posterior confidence, decision thresholds, and validated snapshot export/import;
- examples for physics simulation, finite-state workflows, representation discovery, persistent journaling, and the world-model/experiment/planning loop.

The implementation remains dependency-free and forbids unsafe Rust in the core crate.

## Current boundaries

This is a meaningful first intelligence loop, not a claim of general intelligence. Predictive models currently emit numeric next-state estimates. The contextual model is a batch ridge-regression baseline, and the ensemble is empirical rather than Bayesian. The experiment selector uses a disagreement-based information proxy rather than exact expected information gain. Planning is bounded beam search over caller-supplied action proposals; the runtime executes one action at a time and replans from fresh observations. It does not synthesize arguments or guarantee goal satisfaction. Durable intent keys block automatic retries after ambiguous adapter failures; this is a conservative at-most-once policy, not exactly-once external execution.

Temporal and relational models, causal identification beyond caller-supplied likelihood ratios, cross-environment transfer evaluation, online model updates from runtime feedback, a persistent executive supervisor, and meta-learning strategy synthesis remain future work. Model artifact references are caller-supplied stable identifiers, not cryptographic content hashes; artifact compaction is not implemented.

## Development

~~~sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p engine-core --example simulator_demo
cargo run -p engine-core --example finite_state_demo
cargo run -p engine-core --example representation_discovery_demo
cargo run -p engine-core --example experience_journal_demo
cargo run -p engine-core --example world_model_demo
~~~

See [docs/architecture.md](docs/architecture.md) for contracts, safety boundaries, and implementation limits, and [docs/roadmap.md](docs/roadmap.md) for staged milestones.
