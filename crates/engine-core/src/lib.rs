#![forbid(unsafe_code)]

pub mod action;
pub mod checkpoint;
pub mod discovery;
pub mod environment;
pub mod experience;
pub mod experience_journal;
pub mod experimentation;
pub mod learning;
pub mod mapping;
pub mod meta_learning;
pub mod model_artifact;
pub mod planning;
pub mod representation;
pub mod reasoning;
pub mod runtime;
pub mod schema;
pub mod structure;
pub mod value;
pub mod world_model;

pub use action::{
    ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionProposal, ActionRisk,
    AuthorizationError, AuthorizedAction,
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
pub use experimentation::{
    rank_experiments, ExperimentCandidate, ExperimentError, ExperimentPolicy, ExperimentRanking,
    ExperimentScore,
};
pub use learning::{FieldPrediction, LearningError, NumericDeltaModel, TransitionSample};
pub use mapping::{
    FieldMapping, MappingError, MappingReport, MappingResult, MappingTransform,
    RepresentationMapping, RepresentationRef,
};
pub use model_artifact::{ArtifactPutOutcome, ModelArtifactError, ModelArtifactStore};
pub use planning::{plan, Plan, PlanStep, PlanningConfig, PlanningError};
pub use representation::{
    EvaluationPolicy, RegistryError, RepresentationEvaluation, RepresentationKind,
    RepresentationRecord, RepresentationRegistry, RepresentationSpec, RepresentationStatus,
};
pub use reasoning::{
    EvidenceRecord, HypothesisPolicy, HypothesisRecord, HypothesisRegistry, HypothesisSpec,
    HypothesisStatus, ReasoningError,
};
pub use meta_learning::{
    InputBinding, OperatorCatalogue, OperatorSpec, PortSpec, StrategyError, StrategyEvaluation,
    StrategyGraph, StrategyNode, StrategyPolicy, StrategyRecord, StrategyRegistry, StrategySpec,
    StrategyStatus,
};
pub use runtime::{
    run_receding_horizon, PredictionErrorMetrics, RuntimeConfig, RuntimeError, RuntimeReport,
    RuntimeStepReport, RuntimeTermination,
};
pub use schema::{
    FieldSpec, SchemaDefinition, SchemaDefinitionError, SchemaValidationError, ValueKind,
};
pub use structure::{
    infer_recursive_structure, DiscoveredKind, DiscoveryLimits, StructureDiscoveryError,
    StructureNode, StructureProfile,
};
pub use value::{DataValue, DataValueError};
pub use world_model::{
    evaluate_ensemble, evaluate_model, restore_predictive_model, ActionDeltaModel,
    ContextualLinearModel, EnsembleWorldModel, FieldEstimate, ModelEvaluation, ModelFitReport,
    ModelIdentity, ModelPrediction, PredictiveModel, WorldModelError,
};
