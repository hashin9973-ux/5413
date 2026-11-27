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
- [x] Artifact-store poison-after-I/O-failure semantics with injected partial-write tail recovery and sync-error reopen/idempotent-retry validation
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
- [x] Temporal numeric profiles over ordered observations, including drift, variance, extrema, missing transitions, and direction changes
- [x] Relationship graph extraction from caller-declared identity/reference fields with unresolved-reference policy
- [x] Distribution-based field-role candidates for numeric, boolean, categorical, identifier-like, structured, and heterogeneous values
- [ ] Learned temporal/relational representations and semantic concept induction

## Milestone 4 — Predictive world models

- [x] Interchangeable predictive-model interface with model identity and training dataset provenance
- [x] Versioned model artifact storage and restore for built-in statistical/contextual models
- [x] Model registry with held-out promotion gates, resource penalty, calibration requirement, rollback by version retirement, and snapshot-before-publication promotion semantics
- [x] Action-conditioned statistical delta model
- [x] Overflow-resistant online delta mean/variance with versioned learner artifacts and legacy-state migration
- [x] Contextual numeric ridge-regression model with dynamically discovered features
- [x] Empirical model ensemble with separate aleatoric and disagreement-based epistemic uncertainty estimates
- [x] Overflow-resistant ensemble mean and population-spread calculations, scaled aleatoric/epistemic aggregation, stable total-standard-deviation composition, and invalid uncertainty rejection
- [x] Held-out model evaluation with MAE, RMSE, spread, empirical coverage, and dataset-ID leakage guard
- [x] Stable contextual residual-spread fitting for very large finite residuals without intermediate square overflow
- [ ] Temporal sequence models and relational dynamics
- [ ] Symbolic transition-rule induction and competing hypotheses
- [ ] Better calibrated predictive distributions and proper scoring rules
- [ ] Persisted, versioned model artifacts and reproducible training manifests

## Milestone 5 — Experimentation and planning

- [x] Candidate experiment ranking using model disagreement, task utility, estimated cost, and declared action risk
- [x] Bounded beam-search planner over model-predicted states
- [x] Planner-side action-schema and risk authorization checks
- [x] Exact binary-outcome expected information gain and posterior-aware hypothesis updates under caller-supplied likelihoods
- [x] Categorical multi-outcome experiment ranking, one-step authorized execution, likelihood-ratio evidence updates, and durable hypothesis snapshots
- [ ] Online calibration of experiment likelihoods
- [ ] Controlled causal interventions with confounder-aware evaluation
- [ ] Goal predicates, hard constraints, resource budgets, and terminal conditions
- [x] Receding-horizon runtime with fresh-state replanning, durable pre-execution intent, immediate reauthorization, observed outcome capture, and prediction-error metrics
- [x] Overflow-resistant runtime and held-out MAE/RMSE accumulation plus scaled planner uncertainty aggregation and finite-score guards
- [x] Episode-scoped transition extraction from completed journaled action/observation sequences
- [x] Bounded one-step research-cycle orchestration linking experiment selection, authorized runtime execution, durable evidence, episode training, held-out evaluation, and model promotion
- [x] Preserve completed runtime reports across subsequent evidence update/persistence and learning failures; preflight both possible likelihood updates for numerical validity
- [x] Preserve incomplete experiment runtime reports in typed errors; verify journal-recovered incomplete outcomes and post-effect observations without updating hypotheses or training candidates
- [x] Stage hypothesis evidence registry updates and publish in-memory state only after immutable snapshot persistence; preserve explicit failure outcomes without diverging live registry state
- [x] Reset-aware multi-episode exploration with reset capability preflight, unique episode/execution IDs, durable reset-request/completion journal events, global action/resource budgets, and preserved completed reports
- [ ] Durable reset lifecycle events, environment recovery policies after ambiguous failures, and automatic compensation where adapters support it

## Milestone 6 — Reasoning and evidence

- [x] Typed hypothesis claims and competing hypothesis registry
- [x] Unique evidence references, provenance dataset IDs, and log-likelihood-ratio accumulation
- [x] Posterior confidence and explicit support/refutation/inconclusive thresholds
- [x] Strict versioned snapshot serialization and restore validation
- [x] Paired randomized-trial difference-in-means estimator with uncertainty and strict pair validation
- [x] Independent-unit randomized multi-arm treatment-vs-control estimates, declared allocation checks, Welch-style standard errors, normal-approximation intervals, Holm-Bonferroni adjustment, and immutable study persistence
- [x] Exact exhaustive paired sign-flip randomization inference under an explicitly attested independent fair-swap assignment mechanism, with a hard enumeration cap
- [x] Exact bounded multi-arm fixed-allocation randomization test using the maximum absolute treatment-control contrast as an omnibus statistic
- [ ] Confounder-aware estimators and sensitivity analysis
- [x] Binary-outcome expected-information-gain ranking from caller-supplied likelihood models, risk and cost penalties, and irreversible-action constraints
- [ ] Automatic hypothesis generation and online calibration of experiment likelihoods

## Milestone 7 — Cross-environment generalization

- [x] Transfer benchmark records with source/target environment identities and dataset-leakage checks
- [x] Aggregate transfer improvement, win rate, case-level standard error, and resource cost
- [x] Few-shot transfer curves over strictly increasing adaptation sample counts, with direction-aware improvement and area-under-curve metric
- [x] Sparse directed multi-task transfer matrices with per-pair improvement, win/loss rates, uncertainty, and resource cost
- [x] Overflow-resistant transfer means, standard errors, and resource-cost averages in summaries and directed matrices, with rejection of unrepresentable score improvements
- [x] Configurable negative-transfer diagnosis with minimum case count and negative-outcome rate
- [ ] Multi-task transfer graph optimization and source-selection policy

- [ ] Common benchmark protocol across discrete, continuous, workflow, and relational environments
- [x] Per-seed sample-efficiency curves against caller-declared reference strategies, with area under improvement and samples-to-target metrics
- [x] Aggregate repeated learning-curve seeds only within matching protocol signatures, with per-budget standard errors, AUC uncertainty, and target-reach statistics
- [x] Executable paired benchmark-protocol runner across seeds and sample budgets, with strict result-provenance checks, resource/evaluation budgets, aggregate summaries, and immutable persistence
- [ ] Confidence intervals robust to non-normal seed distributions
- [ ] Calibration, robustness, and out-of-distribution benchmark suites
- [ ] Transfer evaluation against independent-learning baselines
- [ ] Environment adaptation without environment-specific changes in the intelligence core

## Milestone 8 — Meta-learning

- [x] Typed, versioned declarative strategy definitions and DAG validation
- [x] Same-heldout strategy evaluation, resource weighting, guarded candidate promotion, and version retirement
- [x] Deterministic DAG interpreter over caller-supplied trusted operators with runtime port/type checks and declared cost-budget enforcement
- [x] Bounded unary-chain synthesis from trusted operator specs with depth, cost, expansion, and candidate limits
- [x] Multi-input DAG composition/synthesis from typed operator ports with bounded branch/merge search, explicit determinism policy, cost and expansion limits
- [ ] Strategy search guided by measured task performance
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
