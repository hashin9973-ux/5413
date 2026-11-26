# 5413

5413 is an environment-independent adaptive learning runtime. Its long-term direction is an autonomous experimentalist: a system that learns predictive models of unfamiliar environments, selects informative experiments, plans actions under explicit safety constraints, and improves through measured evidence.

The current implementation includes:

- typed, domain-neutral environment contracts and schema-validated observations;
- explicit action authorization, with read-only default policy;
- append-only experience capture with a versioned, checksummed file journal, bounded decoding, torn-tail recovery, episode replay, and durable idempotency keys;
- a separately versioned append-only checkpoint log with torn-tail recovery, monotonic/idempotent snapshot writes, environment/schema binding, and deterministic callback-driven replay after a checkpoint;
- versioned, held-out-evaluated representations and explicit mappings;
- recursive structural profiling for nested objects and sequences;
- a pluggable predictive-model contract, an action-conditioned statistical model, and contextual ridge regression;
- empirical model ensembles that report within-model spread and between-model disagreement separately;
- held-out prediction evaluation with a dataset-ID leakage guard;
- disagreement-based experiment ranking with declared action-risk, utility, and cost constraints;
- bounded beam-search planning that scores predicted states and checks candidate actions against the environment's action policy;
- examples for physics simulation, finite-state workflows, representation discovery, persistent journaling, and the world-model/experiment/planning loop.

The implementation remains dependency-free and forbids unsafe Rust in the core crate.

## Current boundaries

This is a meaningful first intelligence loop, not a claim of general intelligence. Predictive models currently emit numeric next-state estimates. The contextual model is a batch ridge-regression baseline, and the ensemble is empirical rather than Bayesian. The experiment selector uses a disagreement-based information proxy rather than exact expected information gain. Planning is bounded beam search over caller-supplied action proposals; it does not execute actions, synthesize arguments, or guarantee goal satisfaction. Plans must be re-authorized immediately before execution.

Model artifact persistence (checkpoints currently retain artifact references only), temporal and relational models, causal identification, cross-environment transfer evaluation, a persistent executive runtime, and meta-learning strategy synthesis remain future work.

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
