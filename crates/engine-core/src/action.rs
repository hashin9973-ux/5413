use crate::environment::EnvironmentDescriptor;
use crate::schema::SchemaValidationError;
use crate::value::DataValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActionRisk {
    ReadOnly,
    Reversible,
    Irreversible,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionDescriptor {
    pub id: String,
    pub arguments: crate::schema::SchemaDefinition,
    pub risk: ActionRisk,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionProposal {
    pub action_id: String,
    pub arguments: DataValue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionAuthorizationPolicy {
    pub maximum_risk: ActionRisk,
}

impl ActionAuthorizationPolicy {
    pub fn new(maximum_risk: ActionRisk) -> Self {
        Self { maximum_risk }
    }
}

impl Default for ActionAuthorizationPolicy {
    fn default() -> Self {
        Self::new(ActionRisk::ReadOnly)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuthorizedAction {
    action_id: String,
    arguments: DataValue,
    risk: ActionRisk,
}

impl AuthorizedAction {
    pub fn action_id(&self) -> &str {
        &self.action_id
    }

    pub fn arguments(&self) -> &DataValue {
        &self.arguments
    }

    pub fn risk(&self) -> ActionRisk {
        self.risk
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ActionAuthorizer {
    policy: ActionAuthorizationPolicy,
}

impl ActionAuthorizer {
    pub fn new(policy: ActionAuthorizationPolicy) -> Self {
        Self { policy }
    }

    pub fn authorize(
        &self,
        environment: &EnvironmentDescriptor,
        proposal: ActionProposal,
    ) -> Result<AuthorizedAction, AuthorizationError> {
        let descriptor = environment
            .actions
            .iter()
            .find(|action| action.id == proposal.action_id)
            .ok_or_else(|| AuthorizationError::UnknownAction(proposal.action_id.clone()))?;

        if descriptor.risk > self.policy.maximum_risk {
            return Err(AuthorizationError::RiskExceedsPolicy {
                requested: descriptor.risk,
                maximum: self.policy.maximum_risk,
            });
        }

        descriptor
            .arguments
            .validate(&proposal.arguments)
            .map_err(AuthorizationError::InvalidArguments)?;

        Ok(AuthorizedAction {
            action_id: proposal.action_id,
            arguments: proposal.arguments,
            risk: descriptor.risk,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthorizationError {
    UnknownAction(String),
    RiskExceedsPolicy {
        requested: ActionRisk,
        maximum: ActionRisk,
    },
    InvalidArguments(SchemaValidationError),
}

impl std::fmt::Display for AuthorizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownAction(action) => write!(formatter, "unknown action: {action}"),
            Self::RiskExceedsPolicy {
                requested,
                maximum,
            } => write!(
                formatter,
                "action risk {requested:?} exceeds policy maximum {maximum:?}"
            ),
            Self::InvalidArguments(error) => {
                write!(formatter, "action arguments are invalid: {error}")
            }
        }
    }
}

impl std::error::Error for AuthorizationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::EnvironmentDescriptor;
    use crate::schema::{FieldSpec, SchemaDefinition, ValueKind};

    fn descriptor() -> EnvironmentDescriptor {
        EnvironmentDescriptor {
            id: "test-environment".to_owned(),
            version: "1".to_owned(),
            observation_schemas: Vec::new(),
            actions: vec![ActionDescriptor {
                id: "mutate".to_owned(),
                arguments: SchemaDefinition {
                    id: "mutate-args".to_owned(),
                    version: 1,
                    fields: vec![FieldSpec {
                        name: "amount".to_owned(),
                        kind: ValueKind::Integer,
                        required: true,
                    }],
                    allow_additional_fields: false,
                },
                risk: ActionRisk::Reversible,
            }],
            supports_reset: false,
        }
    }

    #[test]
    fn rejects_actions_above_policy_risk() {
        let authorizer = ActionAuthorizer::default();
        let proposal = ActionProposal {
            action_id: "mutate".to_owned(),
            arguments: DataValue::object([("amount".to_owned(), DataValue::Integer(1))]),
        };
        assert!(matches!(
            authorizer.authorize(&descriptor(), proposal),
            Err(AuthorizationError::RiskExceedsPolicy { .. })
        ));
    }

    #[test]
    fn authorizes_declared_action_with_valid_arguments() {
        let authorizer =
            ActionAuthorizer::new(ActionAuthorizationPolicy::new(ActionRisk::Reversible));
        let proposal = ActionProposal {
            action_id: "mutate".to_owned(),
            arguments: DataValue::object([("amount".to_owned(), DataValue::Integer(1))]),
        };
        assert!(authorizer.authorize(&descriptor(), proposal).is_ok());
    }
}
