use engine_core::{
    ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionOutcome, ActionProposal,
    ActionRisk, DataValue, EnvironmentAdapter, EnvironmentDescriptor, EnvironmentError,
    ExperienceEvent, ExperienceLedger, FieldSpec, NumericDeltaModel, ObservationFrame,
    SchemaDefinition, TransitionSample, ValueKind,
};
use std::collections::BTreeMap;
use std::error::Error;

struct ParticleSimulator {
    descriptor: EnvironmentDescriptor,
    position: f64,
    velocity: f64,
    acceleration: f64,
    step: u64,
    observation_sequence: u64,
}

impl ParticleSimulator {
    fn new() -> Self {
        let observation_schema = SchemaDefinition {
            id: "particle-state".to_owned(),
            version: 1,
            fields: vec![
                FieldSpec {
                    name: "position".to_owned(),
                    kind: ValueKind::Number,
                    required: true,
                },
                FieldSpec {
                    name: "velocity".to_owned(),
                    kind: ValueKind::Number,
                    required: true,
                },
                FieldSpec {
                    name: "step".to_owned(),
                    kind: ValueKind::Integer,
                    required: true,
                },
            ],
            allow_additional_fields: false,
        };

        let action_arguments = SchemaDefinition {
            id: "advance-arguments".to_owned(),
            version: 1,
            fields: Vec::new(),
            allow_additional_fields: false,
        };

        Self {
            descriptor: EnvironmentDescriptor {
                id: "one-dimensional-particle-simulator".to_owned(),
                version: "1".to_owned(),
                observation_schemas: vec![observation_schema],
                actions: vec![ActionDescriptor {
                    id: "advance".to_owned(),
                    arguments: action_arguments,
                    risk: ActionRisk::Reversible,
                }],
                supports_reset: true,
            },
            position: 0.0,
            velocity: 1.0,
            acceleration: 0.25,
            step: 0,
            observation_sequence: 0,
        }
    }

    fn current_frame(&mut self) -> Result<ObservationFrame, EnvironmentError> {
        let sequence = self.observation_sequence;
        self.observation_sequence = self
            .observation_sequence
            .checked_add(1)
            .ok_or_else(|| EnvironmentError::Internal("observation sequence exhausted".into()))?;

        let payload = DataValue::Object(BTreeMap::from([
            ("position".to_owned(), DataValue::Number(self.position)),
            ("velocity".to_owned(), DataValue::Number(self.velocity)),
            ("step".to_owned(), DataValue::Integer(self.step as i64)),
        ]));

        Ok(ObservationFrame {
            environment_id: self.descriptor.id.clone(),
            session_id: "demo-session".to_owned(),
            episode_id: "particle-episode-0".to_owned(),
            sequence,
            observed_at_millis: self.step.saturating_mul(1_000),
            schema_id: "particle-state".to_owned(),
            schema_version: 1,
            payload,
        })
    }
}

impl EnvironmentAdapter for ParticleSimulator {
    fn descriptor(&self) -> &EnvironmentDescriptor {
        &self.descriptor
    }

    fn observe(&mut self) -> Result<ObservationFrame, EnvironmentError> {
        self.current_frame()
    }

    fn execute(
        &mut self,
        action: engine_core::AuthorizedAction,
    ) -> Result<ActionOutcome, EnvironmentError> {
        if action.action_id() != "advance" {
            return Err(EnvironmentError::InvalidAction(format!(
                "unsupported action: {}",
                action.action_id()
            )));
        }

        self.position += self.velocity;
        self.velocity += self.acceleration;
        self.step = self
            .step
            .checked_add(1)
            .ok_or_else(|| EnvironmentError::Internal("simulation step exhausted".into()))?;

        let observation = self.current_frame()?;
        Ok(ActionOutcome {
            action_id: "advance".to_owned(),
            completed: true,
            observations: vec![observation],
            feedback: DataValue::Object(BTreeMap::from([(
                "simulated".to_owned(),
                DataValue::Bool(true),
            )])),
        })
    }

    fn reset(&mut self) -> Result<engine_core::LifecycleOutcome, EnvironmentError> {
        self.position = 0.0;
        self.velocity = 1.0;
        self.step = 0;
        self.observation_sequence = 0;
        Ok(engine_core::LifecycleOutcome {
            episode_id: "particle-episode-reset".to_owned(),
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut environment = ParticleSimulator::new();
    let descriptor = environment.descriptor().clone();
    descriptor.validate()?;

    let authorizer = ActionAuthorizer::new(ActionAuthorizationPolicy::new(ActionRisk::Reversible));
    let mut ledger = ExperienceLedger::default();
    let mut learner = NumericDeltaModel::default();

    let mut previous = environment.observe()?;
    descriptor.validate_observation(&previous)?;
    ledger.append(
        previous.episode_id.clone(),
        ExperienceEvent::Observation(previous.clone()),
    )?;

    for _ in 0..6 {
        let proposal = ActionProposal {
            action_id: "advance".to_owned(),
            arguments: DataValue::Object(BTreeMap::new()),
        };
        ledger.append(
            previous.episode_id.clone(),
            ExperienceEvent::ActionProposed {
                action_id: proposal.action_id.clone(),
                arguments: proposal.arguments.clone(),
            },
        )?;

        let authorized = authorizer.authorize(&descriptor, proposal)?;
        let outcome = environment.execute(authorized)?;
        let next = outcome
            .observations
            .first()
            .cloned()
            .ok_or("simulator action returned no observation")?;
        descriptor.validate_observation(&next)?;

        learner.observe_transition(&TransitionSample {
            action_id: "advance".to_owned(),
            before: previous.payload.clone(),
            after: next.payload.clone(),
        })?;

        ledger.append(
            next.episode_id.clone(),
            ExperienceEvent::ActionCompleted(outcome),
        )?;
        ledger.append(
            next.episode_id.clone(),
            ExperienceEvent::Observation(next.clone()),
        )?;

        let predictions = learner.predict_next("advance", &next.payload)?;
        if let Some(prediction) = predictions.get("position") {
            println!(
                "step {}: learned next position {:.3} ({} samples, delta stddev {:.3})",
                next.sequence,
                prediction.predicted_value,
                prediction.sample_count,
                prediction.delta_standard_deviation
            );
        }

        previous = next;
    }

    println!(
        "recorded {} experience events across {} learned transitions",
        ledger.len(),
        learner.transitions_seen()
    );
    Ok(())
}
