use engine_core::{
    ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionOutcome, ActionProposal,
    ActionRisk, DataValue, EnvironmentAdapter, EnvironmentDescriptor, EnvironmentError,
    ExperienceEvent, ExperienceLedger, FieldSpec, ObservationFrame, SchemaDefinition, ValueKind,
};
use std::collections::BTreeMap;
use std::error::Error;

struct WorkflowSimulator {
    descriptor: EnvironmentDescriptor,
    phase: String,
    episode_id: String,
    sequence: u64,
}

impl WorkflowSimulator {
    fn new() -> Self {
        let observation_schema = SchemaDefinition {
            id: "workflow-state".to_owned(),
            version: 1,
            fields: vec![
                FieldSpec {
                    name: "phase".to_owned(),
                    kind: ValueKind::Text,
                    required: true,
                },
                FieldSpec {
                    name: "terminal".to_owned(),
                    kind: ValueKind::Boolean,
                    required: true,
                },
            ],
            allow_additional_fields: false,
        };
        let action_arguments = SchemaDefinition {
            id: "workflow-action-arguments".to_owned(),
            version: 1,
            fields: Vec::new(),
            allow_additional_fields: false,
        };

        Self {
            descriptor: EnvironmentDescriptor {
                id: "finite-state-workflow".to_owned(),
                version: "1".to_owned(),
                observation_schemas: vec![observation_schema],
                actions: vec![
                    ActionDescriptor {
                        id: "begin".to_owned(),
                        arguments: action_arguments.clone(),
                        risk: ActionRisk::Reversible,
                    },
                    ActionDescriptor {
                        id: "finish".to_owned(),
                        arguments: action_arguments,
                        risk: ActionRisk::Reversible,
                    },
                ],
                supports_reset: true,
            },
            phase: "idle".to_owned(),
            episode_id: "workflow-episode-0".to_owned(),
            sequence: 0,
        }
    }

    fn frame(&mut self) -> Result<ObservationFrame, EnvironmentError> {
        let sequence = self.sequence;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| EnvironmentError::Internal("sequence exhausted".to_owned()))?;

        Ok(ObservationFrame {
            environment_id: self.descriptor.id.clone(),
            session_id: "workflow-demo-session".to_owned(),
            episode_id: self.episode_id.clone(),
            sequence,
            observed_at_millis: sequence.saturating_mul(500),
            schema_id: "workflow-state".to_owned(),
            schema_version: 1,
            payload: DataValue::Object(BTreeMap::from([
                ("phase".to_owned(), DataValue::Text(self.phase.clone())),
                (
                    "terminal".to_owned(),
                    DataValue::Bool(self.phase == "complete"),
                ),
            ])),
        })
    }
}

impl EnvironmentAdapter for WorkflowSimulator {
    fn descriptor(&self) -> &EnvironmentDescriptor {
        &self.descriptor
    }

    fn observe(&mut self) -> Result<ObservationFrame, EnvironmentError> {
        self.frame()
    }

    fn execute(
        &mut self,
        action: engine_core::AuthorizedAction,
    ) -> Result<ActionOutcome, EnvironmentError> {
        self.descriptor.validate_authorized_action(&action)?;

        match action.action_id() {
            "begin" if self.phase == "idle" => self.phase = "active".to_owned(),
            "finish" if self.phase == "active" => self.phase = "complete".to_owned(),
            "begin" | "finish" => {
                return Err(EnvironmentError::InvalidAction(format!(
                    "transition {} is not allowed from phase {}",
                    action.action_id(),
                    self.phase
                )));
            }
            unknown => {
                return Err(EnvironmentError::InvalidAction(format!(
                    "unknown action: {unknown}"
                )));
            }
        }

        let observation = self.frame()?;
        Ok(ActionOutcome {
            action_id: action.action_id().to_owned(),
            completed: true,
            observations: vec![observation],
            feedback: DataValue::Object(BTreeMap::from([(
                "phase".to_owned(),
                DataValue::Text(self.phase.clone()),
            )])),
        })
    }

    fn reset(&mut self) -> Result<engine_core::LifecycleOutcome, EnvironmentError> {
        self.phase = "idle".to_owned();
        self.episode_id = "workflow-episode-reset".to_owned();
        self.sequence = 0;
        Ok(engine_core::LifecycleOutcome {
            episode_id: self.episode_id.clone(),
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut environment = WorkflowSimulator::new();
    let descriptor = environment.descriptor().clone();
    descriptor.validate()?;

    let authorizer = ActionAuthorizer::new(ActionAuthorizationPolicy::new(ActionRisk::Reversible));
    let mut ledger = ExperienceLedger::default();

    let mut observation = environment.observe()?;
    descriptor.validate_observation(&observation)?;
    ledger.append(
        observation.episode_id.clone(),
        ExperienceEvent::Observation(observation.clone()),
    )?;

    for action_id in ["begin", "finish"] {
        let proposal = ActionProposal {
            action_id: action_id.to_owned(),
            arguments: DataValue::Object(BTreeMap::new()),
        };
        ledger.append(
            observation.episode_id.clone(),
            ExperienceEvent::ActionProposed {
                action_id: proposal.action_id.clone(),
                arguments: proposal.arguments.clone(),
            },
        )?;

        let authorized = authorizer.authorize(&descriptor, proposal)?;
        let outcome = environment.execute(authorized)?;
        observation = outcome
            .observations
            .first()
            .cloned()
            .ok_or("workflow action returned no observation")?;
        descriptor.validate_observation(&observation)?;
        ledger.append(
            observation.episode_id.clone(),
            ExperienceEvent::ActionCompleted(outcome),
        )?;
        ledger.append(
            observation.episode_id.clone(),
            ExperienceEvent::Observation(observation.clone()),
        )?;
    }

    println!(
        "workflow reached phase {:?}; recorded {} experience events",
        observation.payload.get("phase"),
        ledger.len()
    );
    Ok(())
}
