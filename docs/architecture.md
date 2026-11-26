# 5413 architecture contract

## Purpose

5413 is designed as an environment-independent adaptive learning runtime. Its target is an autonomous experimentalist that forms predictive models of unfamiliar environments, actively seeks evidence, plans under constraints, and improves learning strategies through reproducible evaluation.

The current implementation provides the first integrated learning loop, not general intelligence. The code must preserve the distinction between measured capabilities, heuristics, and future research directions.

## Core boundaries

1. **Environment adapter:** describes observations, actions, lifecycle capabilities, and feedback.
2. **Experience journal:** preserves append-only historical evidence with sequence and idempotency guarantees.
3. **Representation system:** discovers structure and manages candidate/active representation versions.
4. **World models:** train on identified datasets and produce numeric next-state predictions with explicit uncertainty metadata.
5. **Experimentation:** includes a disagreement-based action ranker and a separate binary-hypothesis selector using caller-supplied likelihoods for expected entropy reduction.
6. **Planning:** searches over model-predicted states while checking action schemas and risk policy.
7. **Evaluation:** scores held-out predictions and blocks dataset-ID reuse.
8. **Receding-horizon runtime:** observes, plans, records intent, reauthorizes, executes one step, observes again, and records prediction error.
9. **Execution boundary:** adapters remain responsible for enforcing environment invariants.

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
- Separate append-only checkpoint log with versioned/checksummed frames, monotonic snapshot sequence validation, idempotent duplicate writes, partial-tail recovery, and callback-driven restoration from the last applied journal sequence.
- Versioned candidate/active/rejected/retired representation lifecycle with held-out schema validation.
- First-level schema inference and recursive structure profiles for nested objects and sequences. Recursive profiles describe observed structure; they are not yet recursively executable schema definitions or semantic concepts.
- Explicit, schema-validated field mappings with identity, affine numeric, and constant transforms.
- Baseline action-conditioned numeric mean-delta learner.
- PredictiveModel interface with versioned identity, training dataset provenance, fit reports, and structured predictions.
- ActionDeltaModel, a statistical wrapper over the baseline delta learner.
- ContextualLinearModel, a dependency-free batch ridge-regression baseline. It derives numeric state features and categorical action features from supplied training samples; it models numeric deltas only.
- EnsembleWorldModel, which aggregates member predictions. Within-model residual spread contributes to aleatoric uncertainty; member disagreement contributes to epistemic uncertainty. This is an empirical ensemble, not Bayesian inference.
- Held-out model evaluation reports MAE, RMSE, mean predicted spread, and empirical 95% interval coverage where uncertainty is marked calibrated. Evaluation checks the supplied dataset ID against model training provenance; it cannot independently verify the underlying sample split.
- Candidate-action ranking using ensemble disagreement as an information proxy, supplied task utility/cost estimates, and descriptor-declared action risk. It filters invalid or unauthorized proposals but does not execute interventions.
- Binary hypothesis experiment selection using exact expected entropy reduction under caller-supplied likelihoods P(y|H) and P(y|not-H), with hard resource/risk/irreversibility constraints.
- A bounded `run_autonomous_research_cycle` orchestration path that selects one experiment, executes one authorized action, updates and persists hypothesis evidence, extracts completed episode transitions, trains a candidate, evaluates supplied holdout samples, stores the model, and applies model-registry promotion gates. Its integration contract is explicit and still requires caller-provided action mappings, likelihoods, outcome classifier, holdout, and evaluator.
- Bounded beam-search planning over caller-supplied action proposals with risk/schema checks, configurable uncertainty penalties, and a caller-supplied utility function. The planner returns proposals and predicted states; the caller must re-authorize before execution.
- Immutable, checksummed model-artifact storage with explicit versioned serialization and restoration for the built-in statistical and contextual linear models.
- Representation-registry state can be serialized and stored as immutable artifacts, preserving schema definitions, evaluation evidence, and lifecycle state with strict restore-time invariants.
- Receding-horizon runtime with one-step replanning, durable pre-action intents, immediate reauthorization, outcome persistence, and numeric prediction-error metrics.
- Evidence-backed `HypothesisRegistry`: typed claims, unique evidence references, explicit dataset provenance, log-likelihood-ratio updates, posterior confidence, decision thresholds, and strict artifact round-trip validation.
- `ModelRegistry` tracks immutable model artifact references and only promotes candidates evaluated on the same held-out dataset, metric, and environment contract as the incumbent. Policies can require calibrated uncertainty and account for resource costs; promotion retires the prior active version.
- `rank_experiments` in `experiment_selection.rs` calculates expected entropy reduction for binary outcomes from caller-supplied likelihoods, then filters by resource/risk/irreversibility policy. `experimentation.rs` separately ranks action proposals using model disagreement, utility, cost, and action risk. The former is only as sound as its likelihoods and dependence assumptions.
- `PairedRandomizedStudy` computes a paired treatment-minus-control mean effect and standard error after requiring caller-attested randomized assignment, unique pair IDs, finite outcomes, and at least two pairs. Its 95% interval is a normal approximation, not an exact small-sample interval or randomization test.
- `TransferBenchmark` records baseline versus transferred model scores across distinct source/target environments, rejects reused dataset IDs, and aggregates metric-direction-aware improvement, positive-transfer rate, standard error across cases, and resource cost. `FewShotTransferCurve` also records adaptation performance across increasing target sample counts.
- Declarative `StrategyGraph` composition over a trusted operator catalogue, with exact typed ports, DAG/cycle checks, output reachability, aggregate cost, and same-holdout promotion gates with retirement of replaced versions. `execute_strategy_graph` invokes implementations only through a caller-supplied `StrategyOperatorExecutor`; it enforces deterministic topological order, runtime types, exact input/output sets, finite values, and a declared aggregate cost ceiling. `synthesize_unary_strategy_chains` enumerates type-compatible unary operator chains using deterministic bounded breadth-first search; it returns declarative graphs only and does not generate executable source.
- `extract_transition_samples` reconstructs completed state/action/next-state samples by episode, using the latest valid observation after a matching completed action; incomplete/mismatched actions and invalid state values are not trained on.
- `run_autonomous_research_cycle` ties the single-experiment learning loop together and returns experiment evidence plus a separate learning result, so promotion rejection does not erase the recorded experiment report.
- Runnable examples for a particle simulator, finite-state workflow, representation discovery/mapping, persistent journal replay, and a synthetic world-model/experiment/planning loop.

## Predictive-model semantics and limitations

Predictive-model fitting receives an explicit training dataset ID. Held-out evaluation rejects an evaluation ID equal to the model's training ID. The caller remains responsible for genuinely disjoint datasets; identifiers are not cryptographic provenance.

The statistical delta model estimates per-action mean numeric changes. The contextual linear model fits regularized least-squares models from numeric fields in the current object and one-hot action features. Missing features are treated as zero, so the caller should maintain stable observation semantics or encode missingness explicitly. The current regression model is batch-oriented and does not implement incremental updates, temporal sequence modeling, or arbitrary distributions. Its fitted parameters can be exported as a versioned `DataValue` artifact and stored through `ModelArtifactStore`.

A field estimate includes an estimated mean, aleatoric spread, epistemic spread, sample count, and an uncertainty-calibration flag. A false calibration flag means the system does not have enough residual degrees of freedom to claim calibrated uncertainty. The current empirical ensemble estimates epistemic disagreement from member means plus member-level estimates; it is not a Bayesian posterior and does not guarantee statistically calibrated intervals.

The held-out evaluator reports empirical interval coverage only for estimates marked calibrated. It does not use a proper probabilistic scoring rule yet. For rigorous benchmarking, training/evaluation splits must be constructed outside the model and recorded reproducibly.

## Experimentation and planning semantics

Strategy graphs are metadata, not arbitrary executable code; operator implementations must be supplied by a trusted runtime. Strategy promotion compares candidates and incumbents on the same validation dataset and applies a declared minimum improvement and resource-cost weight. This is a local evaluation gate, not proof of statistical significance; benchmark design and uncertainty-aware comparisons remain future work.

The hypothesis registry combines caller-supplied log likelihood ratios in log-odds space, under an explicit conditional-independence assumption. It prevents duplicate evidence references and direct reuse of the hypothesis source dataset, but it cannot determine whether evidence is truly independent or whether a caller-supplied likelihood ratio is statistically justified. Its statuses are decision thresholds, not proof. The experiment ranker treats model disagreement relative to estimated outcome spread as an information-gain proxy. It is not exact expected information gain and does not perform Bayesian posterior updates. Utility and cost estimates are supplied by the caller. It will not invent environment actions or arguments.

The planner uses bounded beam search over caller-provided action proposals. It evaluates a caller-supplied scalar utility function on predicted states and penalizes uncertainty, including a configurable penalty for uncalibrated predictions. The runtime executes only the first action of each plan, records durable idempotency intent before calling the adapter, reauthorizes immediately before execution, observes the actual next state, and records prediction error. It does not provide terminal-state guarantees or synthesize action arguments. If an adapter call fails after intent is recorded, side-effect status is ambiguous and the execution ID is blocked from automatic retry. This is an at-most-once safety policy, not an exactly-once guarantee. A caller must reconcile ambiguous executions using adapter-specific evidence.

## Deliberately not implemented yet

- Indexed event retrieval, journal/checkpoint/artifact compaction, and cryptographic content addressing. Checkpoints store external model-artifact references; actual built-in model parameters are persisted separately through `ModelArtifactStore`.
- Representation-registry persistence and recovery.
- Async/streaming adapters and concurrent or multi-writer journal operation.
- Tensor/image/audio payload storage and modality-specific encoders.
- Temporal, relational, neural, and symbolic world models.
- Causal identification, controlled intervention analysis, and exact expected information gain.
- Cross-environment transfer benchmarks and independently measured generalization.
- Online model refitting from runtime feedback, goal/attention management, resource-constrained execution, and a persistent autonomous executive supervisor.
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

The initial schema inference routine handles top-level object samples. Recursive structure profiling additionally handles nested objects and sequence elements, unions field names, tracks observed presence counts and null counts, widens integer/number mixtures, and generalizes incompatible types to Any. Configurable depth and node limits bound traversal. Temporal profiling summarizes consecutive numeric deltas in a caller-ordered stream. Relationship extraction requires caller-declared identity and reference fields and an explicit unresolved-reference policy. Field-role profiles identify statistical candidates from value distributions and missingness; they do not infer human meaning from field names. These profiles do not claim causal relations or a complete recursive schema language.

## Action safety

An authorized action is bound to environment ID and version. Adapters validate that binding, the declared action, its risk, and its argument schema before applying effects. Irreversible operations require a policy that explicitly permits them. The planner and experiment ranker check policies while generating proposals, but those checks are not substitutes for re-authorization at the moment of execution. External-action support must add capability, permission, resource, and environment-specific safety checks.

## Checkpoint and restoration guarantees and limits

`CheckpointStore` stores snapshots in a separate single-writer append-only file. Every frame has a magic marker, format version, payload length, and CRC-32 checksum. The store synchronizes each append, treats identical latest snapshots as idempotent, rejects sequence regression and same-sequence conflicts, truncates an incomplete final frame on reopen, and fails closed on complete-frame corruption. Snapshots bind environment ID/version, state-schema ID/version, last-applied experience sequence, logical state, and external model-artifact references. `restore_state` verifies contiguous journal sequence numbers, environment/schema identity, and that the checkpoint high-water mark exists in the journal, then applies only later records through a caller-provided callback. The callback must be deterministic and idempotent with respect to its inputs for reproducible replay. The restore function validates finite state after every applied event. It does not execute an environment or automatically interpret event variants. Checkpoint storage does not yet coordinate multiple writers or compact old snapshots. Learned model artifacts live in a separate immutable store and are referenced by checkpoints; artifact compaction and cryptographic integrity/authenticity are not implemented.

## Experience journal guarantees and limits

ExperienceJournal is a single-writer append-only file journal. Each new record is length-framed and carries format version 3 plus a CRC-32 checksum; the reader remains compatible with version-1 and version-2 records. Version 3 adds durable reset-request and reset-completion event tags while retaining the existing frame header and idempotency-key layout. The encoded payload is capped at 16 MiB, nested values at 128 levels, and decoded collection slots at 100,000 per record (both per-collection and aggregate). Strings are rejected before copying when they would exceed the payload cap. An append is added to the in-memory ledger only after the frame write and File::sync_all succeed. Reopening rebuilds the ledger and preserves global sequence order; episode replay filters that order by episode ID. An incomplete final frame is truncated to the last complete frame, while complete-frame checksum/encoding/sequence errors and unknown format versions fail closed. CRC-32 detects accidental corruption, not tampering. Idempotent append stores a caller-supplied stable key with the record. Retrying that key with the same episode and event returns the original sequence without another frame; reusing the key for different content is rejected. This guarantee assumes one active writer. The journal does not provide cross-process writer coordination, checkpoint compaction, or rewrite-based migration tooling. After an I/O error during append, the journal instance is poisoned and must be reopened because commit status may be ambiguous.

The journal's stress and fault-injection tests validate deterministic software recovery paths. They do not establish storage throughput or reproduce every filesystem, hardware, or physical power-loss failure mode. Durability remains bounded by the host filesystem and storage stack.


## Active-model restoration

`ModelArtifactStore::load_active_ensemble_from_registry` restores a persisted model-registry snapshot, selects records marked Active for the exact requested environment ID/version, reloads their predictive model artifacts, and verifies model ID, version, and training-dataset ID against each registry record before creating an `EnsembleWorldModel`. Rejected, retired, candidate, and other-environment models are excluded. An environment with no compatible active model returns the existing empty-ensemble model error; callers must decide whether to stop, use a separately authorized cold-start policy, or run discovery.


### Post-execution evidence handling

After a completed environment action, `AutonomousResearchReport` retains the runtime report and exposes `EvidenceCommitOutcome`: persisted update, rejected in-memory update, or successful in-memory update with failed durable snapshot. The cycle precomputes both possible log-likelihood updates and checks posterior log-odds finiteness before the environment effect. This avoids losing the recorded experiment result when the separate hypothesis-persistence operation fails. The journal remains the primary durable record of the observed action and outcome.


## Model-disagreement exploration

`rank_model_disagreement_actions` ranks caller-supplied action proposals using one-step predicted task utility and the root-sum-square of ensemble epistemic standard deviations, less declared resource and risk penalties. It requires at least two models, validates each action through the environment descriptor and `ActionAuthorizer`, and applies hard maximum cost/risk filters before ranking. The disagreement term is scale-dependent and is an exploration heuristic, not a calibrated mutual-information estimate; reward and risk estimates remain supplied by the caller/environment integration.


## Dynamic active exploration runtime

`run_receding_horizon_with_selector` calls a typed action-selector callback against each newly validated observation before planning and execution. `run_model_disagreement_exploration` uses this hook to rank caller-supplied actions at every step with the current environment descriptor and state. A configurable top-K shortlist passes into the existing bounded beam planner, which scores short action sequences; only its first action is executed, then the runtime observes and re-ranks. It returns per-step ranking snapshots and the normal runtime report. Each selected proposal is still passed through `ActionAuthorizer` again immediately before execution; action intent and outcome are durably journaled; the caller's maximum-step budget bounds the episode. Model disagreement remains a scale-dependent heuristic, not a formal information-gain guarantee.


## Sample-efficiency benchmark records

`LearningEfficiencyCurve` stores a single-seed comparison between a learner strategy and a reference strategy at strictly increasing training-sample budgets. Scores are direction-aware; the evaluation dataset ID must differ from the training dataset ID; every point stores both strategies' declared resource cost. The curve reports raw trapezoidal area under improvement and samples required to cross a configured improvement threshold. Repeated seeds are separate records in `LearningEfficiencyBenchmark`. Aggregation requires exact matching environment/task/strategy/metric/evaluation-size/sample-budget signatures and unique seeds; it reports per-budget standard errors, AUC standard error, target reach rate, and mean sample budget among successful runs. The standard errors are descriptive normal-approximation summaries, not robust confidence intervals, and an executable benchmark harness remains future work. Dataset identifiers are provenance labels and do not prove disjoint underlying samples.


## Executable learning-efficiency protocol

`BenchmarkProtocol` declares an environment/task, learner and reference strategies, metric direction, increasing training budgets, at least two unique seeds with per-seed training and evaluation dataset IDs, target improvement, evaluation-count and resource caps, and an immutable output reference. The runner executes paired learner/reference trials at every budget, validates the executor's echoed provenance and score/cost fields, constructs per-seed `LearningEfficiencyCurve` values, aggregates repeated seeds, and persists the complete `LearningEfficiencyBenchmark` only after all trials pass. Strategy execution remains provided by the caller. The checks validate declared provenance and do not cryptographically attest dataset contents, random-number generation, or evaluation code.


## Categorical-outcome experiment design

`rank_multi_outcome_experiments` validates caller-supplied categorical likelihood distributions and computes expected posterior entropy across all declared outcomes. `run_multi_outcome_experiment` chooses a positive-utility candidate, executes one supplied action through the authorized runtime, classifies the final observation using a caller-provided function, and records the selected outcome's log-likelihood ratio. It preflights all likelihood ratios before action execution and preserves the runtime report when the outcome is unclassified, hypothesis update is rejected, or artifact persistence fails. Probability distributions must be strictly positive and sum to one under both hypothesis states so finite log-likelihood evidence remains representable. Likelihood calibration and outcome classification are integration responsibilities.


## Reset-aware safe exploration

`run_safe_exploration` invokes the active explorer one action at a time, using a fresh runtime execution ID for each step and one episode ID per environment episode. Multi-episode requests are rejected before side effects unless the descriptor declares reset support; every reset result must return a new non-empty episode ID and the environment descriptor must remain stable. The supervisor enforces total step and declared resource-cost budgets against the exact action proposal recovered from the journal. Completed step/episode reports are retained when a later reset or exploration call fails. Before each reset call, the supervisor appends an idempotent reset-request event; after a reset returns, it appends a reset-completion event containing both episode IDs. Reusing the same execution ID after a recorded reset request is refused so the adapter is not blindly reset twice after an ambiguous failure. This is conservative at-most-once reset invocation, not exactly-once completion; a crash between reset request and completion must be resolved by the environment integration.
