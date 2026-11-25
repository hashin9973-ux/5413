#![forbid(unsafe_code)]

pub mod action;
pub mod environment;
pub mod experience;
pub mod learning;
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
pub use experience::{ExperienceEvent, ExperienceLedger, ExperienceRecord, LedgerError};
pub use learning::{FieldPrediction, LearningError, NumericDeltaModel, TransitionSample};
pub use representation::{
    RegistryError, RepresentationKind, RepresentationRecord, RepresentationRegistry,
    RepresentationSpec, RepresentationStatus,
};
pub use schema::{
    FieldSpec, SchemaDefinition, SchemaDefinitionError, SchemaValidationError, ValueKind,
};
pub use value::{DataValue, DataValueError};
