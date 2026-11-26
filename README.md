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
- disagreement-based candidate-action ranking with declared action-risk, utility, and cost constraints;
- Bayesian expected-information-gain ranking for binary-outcome experiments under caller-supplied likelihoods, with resource/risk budgets and irreversible-action filtering;
- bounded beam-search planning that scores predicted states and checks candidate actions against the environment's action policy;
- a receding-horizon runtime that replans from fresh observations, durably journals action intents before adapter effects, reauthorizes before execution, and records prediction error against observed outcomes;
- episode-scoped transition extraction that reconstructs completed `(state, action, next_state)` samples from journal events and excludes incomplete or mismatched actions;
- an auditable hypothesis registry with typed claims, unique evidence references, log-likelihood-ratio updates, posterior confidence, decision thresholds, and validated snapshot export/import;
- a predictive-model lifecycle registry with held-out evaluation, calibration gates, same-environment/metric/dataset comparisons, resource-aware promotion, and version retirement;
- a paired randomized-trial effect estimator with duplicate-block checks, treatment-minus-control estimates, standard errors, and explicitly normal-approximation intervals;
- expected-information-gain experiment ranking for binary observations, with declared likelihoods, risk/resource budgets, and irreversible-action filtering;
- cross-environment transfer benchmarks that require distinct source/target environments and disjoint dataset IDs, reporting gain, win rate, standard error, and resource cost, plus few-shot adaptation curves with area-under-improvement measurements;
- typed declarative strategy graphs with port validation, cycle/reachability checks, cost estimates, held-out same-dataset comparisons, and guarded candidate promotion/retirement lifecycle;
- an integrated research-cycle orchestrator that executes one selected intervention, updates durable hypothesis evidence, trains from the resulting episode, evaluates on a separate holdout, and promotes a versioned predictive model through explicit gates;
- examples for physics simulation, finite-state workflows, representation discovery, persistent journaling, world-model planning, and the end-to-end autonomous research cycle.

The implementation remains dependency-free and forbids unsafe Rust in the core crate.

## Current boundaries

This is a meaningful first intelligence loop, not a claim of general intelligence. Predictive models currently emit numeric next-state estimates. The contextual model is a batch ridge-regression baseline, and the ensemble is empirical rather than Bayesian. The disagreement-based candidate-action ranker uses a model-disagreement proxy. Separately, the binary hypothesis experiment selector computes exact expected entropy reduction under caller-supplied likelihoods; neither mechanism can validate its caller's assumptions about the environment. Planning is bounded beam search over caller-supplied action proposals; the runtime executes one action at a time and replans from fresh observations. It does not synthesize arguments or guarantee goal satisfaction. Durable intent keys block automatic retries after ambiguous adapter failures; this is a conservative at-most-once policy, not exactly-once external execution.

Temporal and relational predictive models, symbolic rule induction, robust multi-outcome causal estimators, autonomous construction of action mappings, adaptive multi-episode supervision, executable strategy-operator composition, and meta-learning from repeated benchmark evidence remain future work. The current research cycle is bounded to one selected intervention per invocation and requires caller-provided action mappings, likelihood models, validation data, outcome classifier, and evaluation function. Model artifact references are caller-supplied stable identifiers, not cryptographic content hashes; artifact compaction is not implemented.

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
cargo run -p engine-core --example autonomous_research_demo
~~~

See [docs/architecture.md](docs/architecture.md) for contracts, safety boundaries, and implementation limits, and [docs/roadmap.md](docs/roadmap.md) for staged milestones.
