#![forbid(unsafe_code)]

pub mod action;
pub mod discovery;
pub mod environment;
pub mod experience;
pub mod learning;
pub mod mapping;
pub mod representation;
pub mod schema;
pub mod value;

pub use action::{
    ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionProposal, ActionRisk,
    AuthorizationError, AuthorizedAction,
};
pub use environment::{
    ActionOutcome, EnvironmentAdapter, EnvironmentDescriptor, EnvironmentError, LifecycleOutcome,
    ObservationFrame,
};
pub use discovery::{
    propose_representation, DiscoveryError, DiscoverySummary, RepresentationProposal,
};
pub use experience::{ExperienceEvent, ExperienceLedger, ExperienceRecord, LedgerError};
pub use learning::{FieldPrediction, LearningError, NumericDeltaModel, TransitionSample};
pub use mapping::{
    FieldMapping, MappingError, MappingReport, MappingResult, MappingTransform,
    RepresentationMapping, RepresentationRef,
};
pub use representation::{
    EvaluationPolicy, RegistryError, RepresentationEvaluation, RepresentationKind,
    RepresentationRecord, RepresentationRegistry, RepresentationSpec, RepresentationStatus,
};
pub use schema::{
    FieldSpec, SchemaDefinition, SchemaDefinitionError, SchemaValidationError, ValueKind,
};
pub use value::{DataValue, DataValueError};
