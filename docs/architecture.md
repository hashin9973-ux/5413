# 5413 architecture contract

## Purpose

5413 is designed as an environment-independent adaptive learning runtime. Its target is an autonomous experimentalist that forms predictive models of unfamiliar environments, actively seeks evidence, plans under constraints, and improves learning strategies through reproducible evaluation.

The current implementation provides the first integrated learning loop, not general intelligence. The code must preserve the distinction between measured capabilities, heuristics, and future research directions.

## Core boundaries

1. **Environment adapter:** describes observations, actions, lifecycle capabilities, and feedback.
2. **Experience journal:** preserves append-only historical evidence with sequence and idempotency guarantees.
3. **Representation system:** discovers structure and manages candidate/active representation versions.
4. **World models:** train on identified datasets and produce numeric next-state predictions with explicit uncertainty metadata.
5. **Experimentation:** ranks caller-supplied candidate actions using model disagreement, expected utility, cost, and declared risk.
6. **Planning:** searches over model-predicted states while checking action schemas and risk policy.
7. **Evaluation:** scores held-out predictions and blocks dataset-ID reuse.
8. **Execution boundary:** callers must re-authorize a selected proposal immediately before passing it to an environment adapter.

## Invariants

- No physics-specific types in the core API.
- Observations are preserved independently of derived representations.
- Runtime observations are validated against declared, versioned schemas.
- Environment actions require an opaque authorized-action value.
- The default action authorization policy permits read-only actions only.
- Candidate representations are not active until explicitly promoted.
- At most one representation version per representation ID is active in the registry.
- Learning algorithms reject invalid numeric data rather than silently accepting NaN or infinity.
- Model and representation evaluation uses a separately identified dataset; dataset IDs are a provenance guard, not proof of disjoint samples.
- Plans contain proposals, not authorized execution capabilities.
- Strategy and algorithm changes require evaluation before promotion.

## Implemented capabilities

- Recursive, extensible data values, dynamic object schemas, and environment/action descriptors.
- Action-risk authorization bound to environment ID and version.
- Append-only experience ledger and dependency-free file journal with v2 writes/v1 read compatibility, CRC-32 corruption detection, synchronized appends, bounded decoding, torn-tail recovery, ledger reconstruction, ordered episode replay, durable idempotency keys, and deterministic I/O fault-injection tests.
- Versioned candidate/active/rejected/retired representation lifecycle with held-out schema validation.
- First-level schema inference and recursive structure profiles for nested objects and sequences. Recursive profiles describe observed structure; they are not yet recursively executable schema definitions or semantic concepts.
- Explicit, schema-validated field mappings with identity, affine numeric, and constant transforms.
- Baseline action-conditioned numeric mean-delta learner.
- PredictiveModel interface with versioned identity, training dataset provenance, fit reports, and structured predictions.
- ActionDeltaModel, a statistical wrapper over the baseline delta learner.
- ContextualLinearModel, a dependency-free batch ridge-regression baseline. It derives numeric state features and categorical action features from supplied training samples; it models numeric deltas only.
- EnsembleWorldModel, which aggregates member predictions. Within-model residual spread contributes to aleatoric uncertainty; member disagreement contributes to epistemic uncertainty. This is an empirical ensemble, not Bayesian inference.
- Held-out model evaluation reports MAE, RMSE, mean predicted spread, and empirical 95% interval coverage where uncertainty is marked calibrated. Evaluation checks the supplied dataset ID against model training provenance; it cannot independently verify the underlying sample split.
- Experiment ranking using a disagreement-based information-gain proxy, supplied utility/cost estimates, and descriptor-declared action risk. It filters invalid or unauthorized proposals but does not execute interventions.
- Bounded beam-search planning over caller-supplied action proposals with risk/schema checks, configurable uncertainty penalties, and a caller-supplied utility function. The planner returns proposals and predicted states; the caller must re-authorize before execution.
- Runnable examples for a particle simulator, finite-state workflow, representation discovery/mapping, persistent journal replay, and a synthetic world-model/experiment/planning loop.

## Predictive-model semantics and limitations

Predictive-model fitting receives an explicit training dataset ID. Held-out evaluation rejects an evaluation ID equal to the model's training ID. The caller remains responsible for genuinely disjoint datasets; identifiers are not cryptographic provenance.

The statistical delta model estimates per-action mean numeric changes. The contextual linear model fits regularized least-squares models from numeric fields in the current object and one-hot action features. Missing features are treated as zero, so the caller should maintain stable observation semantics or encode missingness explicitly. The current regression model is batch-oriented and retains fitted coefficients in memory; it does not implement incremental updates, temporal sequence modeling, arbitrary distributions, or model artifact persistence.

A field estimate includes an estimated mean, aleatoric spread, epistemic spread, sample count, and an uncertainty-calibration flag. A false calibration flag means the system does not have enough residual degrees of freedom to claim calibrated uncertainty. The current empirical ensemble estimates epistemic disagreement from member means plus member-level estimates; it is not a Bayesian posterior and does not guarantee statistically calibrated intervals.

The held-out evaluator reports empirical interval coverage only for estimates marked calibrated. It does not use a proper probabilistic scoring rule yet. For rigorous benchmarking, training/evaluation splits must be constructed outside the model and recorded reproducibly.

## Experimentation and planning semantics

The experiment ranker treats model disagreement relative to estimated outcome spread as an information-gain proxy. It is not exact expected information gain and does not perform Bayesian posterior updates. Utility and cost estimates are supplied by the caller. It will not invent environment actions or arguments.

The planner uses bounded beam search over caller-provided action proposals. It evaluates a caller-supplied scalar utility function on predicted states and penalizes uncertainty, including a configurable penalty for uncalibrated predictions. It is not a general constraint solver, does not provide terminal-state guarantees, and does not execute actions. Every selected proposal must pass through the normal authorization and environment validation path again at execution time.

## Deliberately not implemented yet

- Checkpoint/restore semantics, indexed event retrieval, journal compaction, and model artifact persistence.
- Representation-registry persistence and recovery.
- Async/streaming adapters and concurrent or multi-writer journal operation.
- Tensor/image/audio payload storage and modality-specific encoders.
- Temporal, relational, neural, and symbolic world models.
- Causal identification, controlled intervention analysis, and exact expected information gain.
- Cross-environment transfer benchmarks and independently measured generalization.
- Goal/attention management, resource-constrained execution, and a persistent autonomous executive runtime.
- Meta-learning strategy graphs, strategy memory, sandboxed operator synthesis, and executable self-improvement.

These are roadmap items, not implied properties of the current implementation.

## Dependency rules

- Core types are centralized initially; extraction into separate crates requires demonstrated reuse and stable dependency boundaries.
- Environment adapters depend on core contracts, never the reverse.
- Core learning and representation modules do not depend on a concrete environment adapter.
- Environment actions are executed only after authorization.
- Candidate models and representations are evaluated outside the trusted path before promotion.
- Predictive models must not obtain implicit environment execution capabilities.
- Experimental strategies are proposals and must be evaluated before entering any trusted registry.

## Representation lifecycle and discovery

A representation proposal records its source training dataset ID and evidence references. A candidate is evaluated on a separately identified validation dataset; the registry rejects reuse of the same dataset ID and promotion is blocked until the candidate meets its minimum sample and valid-fraction policy. This is an explicit provenance guard, not cryptographic proof that datasets are disjoint. Promoting a newer version retires the previously active version without deleting its record. Raw experience remains the historical source of truth.

The initial schema inference routine handles top-level object samples. Recursive structure profiling additionally handles nested objects and sequence elements, unions field names, tracks observed presence counts and null counts, widens integer/number mixtures, and generalizes incompatible types to Any. Configurable depth and node limits bound traversal. Profiles describe structure; they do not claim semantic equivalence, causal relations, or a complete recursive schema language.

## Action safety

An authorized action is bound to environment ID and version. Adapters validate that binding, the declared action, its risk, and its argument schema before applying effects. Irreversible operations require a policy that explicitly permits them. The planner and experiment ranker check policies while generating proposals, but those checks are not substitutes for re-authorization at the moment of execution. External-action support must add capability, permission, resource, and environment-specific safety checks.

## Experience journal guarantees and limits

ExperienceJournal is a single-writer append-only file journal. Each new record is length-framed and carries format version 2 plus a CRC-32 checksum; the reader remains compatible with version-1 records. The encoded payload is capped at 16 MiB, nested values at 128 levels, and decoded collection slots at 100,000 per record (both per-collection and aggregate). Strings are rejected before copying when they would exceed the payload cap. An append is added to the in-memory ledger only after the frame write and File::sync_all succeed. Reopening rebuilds the ledger and preserves global sequence order; episode replay filters that order by episode ID. An incomplete final frame is truncated to the last complete frame, while complete-frame checksum/encoding/sequence errors and unknown format versions fail closed. CRC-32 detects accidental corruption, not tampering. Idempotent append stores a caller-supplied stable key with the record. Retrying that key with the same episode and event returns the original sequence without another frame; reusing the key for different content is rejected. This guarantee assumes one active writer. The journal does not provide cross-process writer coordination, checkpoint compaction, or rewrite-based migration tooling. After an I/O error during append, the journal instance is poisoned and must be reopened because commit status may be ambiguous.

The journal's stress and fault-injection tests validate deterministic software recovery paths. They do not establish storage throughput or reproduce every filesystem, hardware, or physical power-loss failure mode. Durability remains bounded by the host filesystem and storage stack.
