# Development roadmap

The roadmap distinguishes implemented capabilities from research directions. A checked box means code and tests exist in the repository; it does not imply a production-grade or universally general solution.

## Milestone 1 — Environment-independent foundation

- [x] Typed environment descriptors and observation contracts
- [x] Dynamic observation schemas
- [x] Action proposals and authorization boundary
- [x] In-memory experience ledger
- [x] Versioned representation registry
- [x] Baseline numeric transition learner
- [x] Simulator adapter example
- [x] Second, structurally different environment adapter
- [x] CI execution with formatting, linting, tests, and runnable examples

## Milestone 2 — Reliable experience and replay

- [x] Durable append-only file journal with synchronized writes and incomplete-tail recovery
- [x] Episode replay in original append order
- [x] Versioned record encoding and corruption detection
- [x] Versioned append-only checkpoints with partial-tail recovery, sequence high-water marks, and callback-driven deterministic replay
- [x] Durable idempotency keys and duplicate retry suppression (single-writer contract)
- [x] Fault-injected partial-write and sync-failure recovery tests
- [ ] Journal format rewrite/migration tooling (v1 read compatibility is implemented)
- [ ] Indexed event retrieval and bounded-memory segment compaction
- [x] Versioned immutable model-artifact persistence and restoration for built-in statistical and contextual models
- [x] Durable representation-registry snapshot export/import through the immutable artifact store
- [ ] Artifact compaction, cryptographic content addressing, and full training-manifest provenance

## Milestone 3 — Dynamic representation learning

- [x] Candidate representation generation from observed object structures
- [x] Evidence-backed schema validity scoring on separately identified held-out data
- [x] Versioned field mapping and transformation contracts
- [x] Basic source-field loss/coverage reporting and target schema validation
- [x] Recursive structural profiling of nested objects and sequences
- [ ] Compile recursive structural profiles into recursively validated schemas
- [ ] Semantic representation quality scoring beyond schema validity
- [ ] Automated mapping discovery and held-out rollback experiments
- [ ] Temporal, relational, and entity-level representation learning

## Milestone 4 — Predictive world models

- [x] Interchangeable predictive-model interface with model identity and training dataset provenance
- [x] Versioned model artifact storage and restore for built-in statistical/contextual models
- [x] Model registry with held-out promotion gates, resource penalty, calibration requirement, and rollback by version retirement
- [x] Action-conditioned statistical delta model
- [x] Contextual numeric ridge-regression model with dynamically discovered features
- [x] Empirical model ensemble with separate aleatoric and disagreement-based epistemic uncertainty estimates
- [x] Held-out model evaluation with MAE, RMSE, spread, empirical coverage, and dataset-ID leakage guard
- [ ] Temporal sequence models and relational dynamics
- [ ] Symbolic transition-rule induction and competing hypotheses
- [ ] Better calibrated predictive distributions and proper scoring rules
- [ ] Persisted, versioned model artifacts and reproducible training manifests

## Milestone 5 — Experimentation and planning

- [x] Candidate experiment ranking using model disagreement, task utility, estimated cost, and declared action risk
- [x] Bounded beam-search planner over model-predicted states
- [x] Planner-side action-schema and risk authorization checks
- [x] Exact binary-outcome expected information gain and posterior-aware hypothesis updates under caller-supplied likelihoods
- [ ] Multi-outcome experimental design and online calibration of likelihoods
- [ ] Controlled causal interventions with confounder-aware evaluation
- [ ] Goal predicates, hard constraints, resource budgets, and terminal conditions
- [x] Receding-horizon runtime with fresh-state replanning, durable pre-execution intent, immediate reauthorization, observed outcome capture, and prediction-error metrics
- [x] Episode-scoped transition extraction from completed journaled action/observation sequences
- [x] Bounded one-step research-cycle orchestration linking experiment selection, authorized runtime execution, durable evidence, episode training, held-out evaluation, and model promotion
- [ ] Safe exploration policies with explicit reset and recovery contracts

## Milestone 6 — Reasoning and evidence

- [x] Typed hypothesis claims and competing hypothesis registry
- [x] Unique evidence references, provenance dataset IDs, and log-likelihood-ratio accumulation
- [x] Posterior confidence and explicit support/refutation/inconclusive thresholds
- [x] Strict versioned snapshot serialization and restore validation
- [x] Paired randomized-trial difference-in-means estimator with uncertainty and strict pair validation
- [ ] Unpaired/multi-arm estimators, randomization inference, confounder-aware estimators, and sensitivity analysis
- [x] Binary-outcome expected-information-gain ranking from caller-supplied likelihood models, risk and cost penalties, and irreversible-action constraints
- [ ] Automatic hypothesis generation, multi-outcome experiment design, and online calibration of experiment likelihoods

## Milestone 7 — Cross-environment generalization

- [x] Transfer benchmark records with source/target environment identities and dataset-leakage checks
- [x] Aggregate transfer improvement, win rate, case-level standard error, and resource cost
- [x] Few-shot transfer curves over strictly increasing adaptation sample counts, with direction-aware improvement and area-under-curve metric
- [ ] Multi-task transfer matrices and negative-transfer diagnosis

- [ ] Common benchmark protocol across discrete, continuous, workflow, and relational environments
- [ ] Sample-efficiency curves against random and passive-learning baselines
- [ ] Calibration, robustness, and out-of-distribution benchmark suites
- [ ] Transfer evaluation against independent-learning baselines
- [ ] Environment adaptation without environment-specific changes in the intelligence core

## Milestone 8 — Meta-learning

- [x] Typed, versioned declarative strategy definitions and DAG validation
- [x] Same-heldout strategy evaluation, resource weighting, guarded candidate promotion, and version retirement
- [ ] Trusted strategy operator execution and bounded composition/synthesis
- [ ] Strategy selection conditioned on measured environment characteristics
- [ ] Held-out learning-efficiency benchmarks and candidate rollback
- [ ] Persistent strategy memory and provenance

## Milestone 9 — Controlled algorithm improvement

- [ ] Operator proposals from measured failure analysis
- [ ] Sandboxed implementation experiments
- [ ] Reproducible build and benchmark pipeline
- [ ] Security review and regression gates
- [ ] Explicit promotion to a trusted operator registry

## Exit criteria for generality

A new environment must be integrated through an adapter and its declared schemas/actions. It must not require physics-specific or other domain-specific changes to the intelligence core. Transfer must be measured against independent-learning baselines, not inferred from shared interface compatibility alone. Autonomous execution must remain behind explicit authorization, bounded resource policies, observable outcomes, and rollback-capable state transitions.
