#![forbid(unsafe_code)]

mod regression_metrics;

pub mod action;
pub mod active_exploration;
pub mod benchmark;
pub mod benchmark_protocol;
pub mod calibration;
pub mod causal;
pub mod checkpoint;
pub mod discovery;
pub mod environment;
pub mod experience;
pub mod experience_journal;
pub mod experiment_selection;
pub mod experimentation;
pub mod learning;
pub mod mapping;
pub mod meta_learning;
pub mod model_artifact;
pub mod model_registry;
pub mod multi_outcome;
pub mod planning;
pub mod reasoning;
pub mod representation;
pub mod research_cycle;
pub mod runtime;
pub mod safe_exploration;
pub mod schema;
pub mod structure;
pub mod value;
pub mod world_model;

pub use action::{
    ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionProposal, ActionRisk,
    AuthorizationError, AuthorizedAction,
};
pub use active_exploration::{
    run_model_disagreement_exploration, ActiveExplorationError, ActiveExplorationReport,
    ActiveExplorationRequest,
};
pub use benchmark::{
    BenchmarkError, FewShotTransferCurve, LearningEfficiencyBenchmark, LearningEfficiencyCurve,
    LearningEfficiencyPoint, LearningEfficiencySummary, NegativeTransferDiagnosis,
    NegativeTransferPolicy, ScoreDirection, TransferAdaptationPoint, TransferBenchmark,
    TransferEvaluation, TransferMatrix, TransferMatrixCell, TransferSummary,
};
pub use benchmark_protocol::{
    run_benchmark_protocol, BenchmarkProtocol, BenchmarkProtocolError, BenchmarkProtocolExecutor,
    BenchmarkProtocolReport, BenchmarkSeedPlan, BenchmarkStrategyRole, BenchmarkTrialRecord,
    BenchmarkTrialRequest, BenchmarkTrialResult,
};
pub use calibration::{
    CalibratedPrediction, ConformalCalibrationError, ConformalCalibrator, FieldCalibration,
    PredictionInterval,
};
pub use causal::{
    AssignmentDesign, CausalEffectEstimate, CausalError, MultiArmEffectEstimate,
    MultiArmRandomizationInference, MultiArmRandomizationInferencePolicy, PairedOutcome,
    PairedRandomizationInference, PairedRandomizedStudy, RandomizationInferencePolicy,
    RandomizedArmOutcome, RandomizedMultiArmEstimate, RandomizedMultiArmStudy,
};
pub use checkpoint::{
    restore_state, CheckpointAppendOutcome, CheckpointError, CheckpointSnapshot, CheckpointStore,
    RestorationReport, RestoreError, RestoreTarget,
};
pub use discovery::{
    propose_representation, DiscoveryError, DiscoverySummary, RepresentationProposal,
};
pub use environment::{
    ActionOutcome, EnvironmentAdapter, EnvironmentDescriptor, EnvironmentError, LifecycleOutcome,
    ObservationFrame,
};
pub use experience::{ExperienceEvent, ExperienceLedger, ExperienceRecord, LedgerError};
pub use experience_journal::{ExperienceJournal, IdempotentAppendOutcome, JournalError};
pub use experiment_selection::{
    rank_model_disagreement_actions, ExperimentSelectionError, ExperimentSelectionPolicy,
    ModelExplorationCandidate, ModelExplorationPolicy, ModelExplorationRanking,
    RankedExplorationAction,
};
pub use experimentation::{
    rank_experiments, ExperimentCandidate, ExperimentError, ExperimentPolicy, ExperimentRanking,
    ExperimentScore,
};
pub use learning::{
    AdaptiveLearningPolicy, AdaptiveLearningUpdate, AdaptiveNumericDeltaModel,
    CrossActionAdaptiveLearningPolicy, CrossActionAdaptiveLearningUpdate,
    CrossActionAdaptiveNumericDeltaModel, FieldPrediction, LearningError, NumericDeltaModel,
    TransitionSample, WindowedNumericDeltaModel,
};
pub use mapping::{
    FieldMapping, MappingError, MappingReport, MappingResult, MappingTransform,
    RepresentationMapping, RepresentationRef,
};
pub use meta_learning::{
    execute_strategy_graph, synthesize_strategy_dags, synthesize_unary_strategy_chains,
    InputBinding, OperatorCatalogue, OperatorSpec, PortSpec, StrategyDagSynthesisPolicy,
    StrategyError, StrategyEvaluation, StrategyExecutionError, StrategyExecutionReport,
    StrategyGraph, StrategyNode, StrategyOperatorExecutor, StrategyPolicy, StrategyRecord,
    StrategyRegistry, StrategySpec, StrategyStatus, StrategySynthesisError,
    StrategySynthesisPolicy, StrategySynthesisResult, SynthesizedStrategy,
};
pub use model_artifact::{ArtifactPutOutcome, ModelArtifactError, ModelArtifactStore};
pub use model_registry::{
    ModelCandidateSpec, ModelEvaluation as RegistryModelEvaluation, ModelPromotionPolicy,
    ModelRecord, ModelRegistry, ModelRegistryError, ModelStatus,
};
pub use multi_outcome::{
    rank_multi_outcome_experiments, run_multi_outcome_experiment, MultiOutcomeActionMapping,
    MultiOutcomeEvidenceCommitOutcome, MultiOutcomeExecutionConfig,
    MultiOutcomeExperimentCallbacks, MultiOutcomeExperimentCandidate,
    MultiOutcomeExperimentContext, MultiOutcomeExperimentError, MultiOutcomeExperimentReport,
    MultiOutcomeExperimentRequest, MultiOutcomeSelectionPolicy, OutcomeLikelihood,
    RankedMultiOutcomeExperiment,
};
pub use planning::{plan, Plan, PlanStep, PlanningConfig, PlanningError};
pub use reasoning::{
    EvidenceRecord, HypothesisPolicy, HypothesisRecord, HypothesisRegistry, HypothesisSpec,
    HypothesisStatus, ReasoningError,
};
pub use representation::{
    EvaluationPolicy, RegistryError, RepresentationEvaluation, RepresentationKind,
    RepresentationRecord, RepresentationRegistry, RepresentationSpec, RepresentationStatus,
};
pub use research_cycle::{
    run_autonomous_research_cycle, AutonomousResearchCallbacks, AutonomousResearchConfig,
    AutonomousResearchContext, AutonomousResearchError, AutonomousResearchReport,
    AutonomousResearchRequest, EvidenceCommitOutcome, ExperimentActionMapping,
    ExperimentExecutionConfig, LearningCycleConfig, LearningCycleError, LearningCycleReport,
};
pub use runtime::{
    run_receding_horizon, run_receding_horizon_with_selector, PredictionErrorMetrics,
    RuntimeConfig, RuntimeError, RuntimeReport, RuntimeStepReport, RuntimeTermination,
};
pub use safe_exploration::{
    run_safe_exploration, SafeExplorationEpisodeReport, SafeExplorationError,
    SafeExplorationFailure, SafeExplorationReport, SafeExplorationRequest,
    SafeExplorationTermination,
};
pub use schema::{
    FieldSpec, SchemaDefinition, SchemaDefinitionError, SchemaValidationError, ValueKind,
};
pub use structure::{
    analyze_temporal_sequence, discover_relationships, infer_recursive_structure, profile_fields,
    DiscoveredKind, DiscoveryLimits, FieldProfile, FieldProfilePolicy, FieldRole, RelationEdge,
    RelationshipGraph, SemanticProfile, StructureDiscoveryError, StructureNode, StructureProfile,
    TemporalFieldProfile, TemporalProfile, UnresolvedReferencePolicy,
};
pub use value::{DataValue, DataValueError};
pub use world_model::{
    evaluate_ensemble, evaluate_model, restore_predictive_model, ActionDeltaModel,
    ContextualLinearModel, EnsembleWorldModel, FieldEstimate, ModelEvaluation, ModelFitReport,
    ModelIdentity, ModelPrediction, PredictiveModel, TrainablePredictiveModel, WorldModelError,
};
