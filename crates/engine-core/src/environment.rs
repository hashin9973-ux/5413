use crate::action::{ActionDescriptor, AuthorizedAction};
use crate::schema::SchemaDefinition;
use crate::value::DataValue;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvironmentDescriptor {
    pub id: String,
    pub version: String,
    pub observation_schemas: Vec<SchemaDefinition>,
    pub actions: Vec<ActionDescriptor>,
    pub supports_reset: bool,
}

impl EnvironmentDescriptor {
    pub fn validate(&self) -> Result<(), EnvironmentError> {
        if self.id.trim().is_empty() {
            return Err(EnvironmentError::InvalidDescriptor(
                "environment id must not be empty".to_owned(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(EnvironmentError::InvalidDescriptor(
                "environment version must not be empty".to_owned(),
            ));
        }

        let mut schema_ids = std::collections::BTreeSet::new();
        for schema in &self.observation_schemas {
            schema
                .validate_definition()
                .map_err(|error| EnvironmentError::InvalidDescriptor(error.to_string()))?;
            if !schema_ids.insert((schema.id.as_str(), schema.version)) {
                return Err(EnvironmentError::InvalidDescriptor(format!(
                    "duplicate observation schema {}@{}",
                    schema.id, schema.version
                )));
            }
        }

        let mut action_ids = std::collections::BTreeSet::new();
        for action in &self.actions {
            if action.id.trim().is_empty() {
                return Err(EnvironmentError::InvalidDescriptor(
                    "action id must not be empty".to_owned(),
                ));
            }
            if !action_ids.insert(action.id.as_str()) {
                return Err(EnvironmentError::InvalidDescriptor(format!(
                    "duplicate action id: {}",
                    action.id
                )));
            }
            action
                .arguments
                .validate_definition()
                .map_err(|error| EnvironmentError::InvalidDescriptor(error.to_string()))?;
        }
        Ok(())
    }

    pub fn validate_observation(
        &self,
        observation: &ObservationFrame,
    ) -> Result<(), EnvironmentError> {
        if observation.environment_id != self.id {
            return Err(EnvironmentError::InvalidObservation(
                "observation belongs to a different environment".to_owned(),
            ));
        }

        let schema = self
            .observation_schemas
            .iter()
            .find(|schema| {
                schema.id == observation.schema_id && schema.version == observation.schema_version
            })
            .ok_or_else(|| {
                EnvironmentError::InvalidObservation(format!(
                    "unknown observation schema {}@{}",
                    observation.schema_id, observation.schema_version
                ))
            })?;

        schema
            .validate(&observation.payload)
            .map_err(|error| EnvironmentError::InvalidObservation(error.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObservationFrame {
    pub environment_id: String,
    pub session_id: String,
    pub episode_id: String,
    pub sequence: u64,
    pub observed_at_millis: u64,
    pub schema_id: String,
    pub schema_version: u64,
    pub payload: DataValue,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionOutcome {
    pub action_id: String,
    pub completed: bool,
    pub observations: Vec<ObservationFrame>,
    pub feedback: DataValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LifecycleOutcome {
    pub episode_id: String,
}

pub trait EnvironmentAdapter {
    fn descriptor(&self) -> &EnvironmentDescriptor;

    fn observe(&mut self) -> Result<ObservationFrame, EnvironmentError>;

    fn execute(&mut self, action: AuthorizedAction) -> Result<ActionOutcome, EnvironmentError>;

    fn reset(&mut self) -> Result<LifecycleOutcome, EnvironmentError> {
        Err(EnvironmentError::UnsupportedOperation("reset"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvironmentError {
    UnsupportedOperation(&'static str),
    InvalidDescriptor(String),
    InvalidObservation(String),
    InvalidAction(String),
    Internal(String),
}

impl std::fmt::Display for EnvironmentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOperation(operation) => {
                write!(formatter, "environment does not support {operation}")
            }
            Self::InvalidDescriptor(message)
            | Self::InvalidObservation(message)
            | Self::InvalidAction(message)
            | Self::Internal(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for EnvironmentError {}
