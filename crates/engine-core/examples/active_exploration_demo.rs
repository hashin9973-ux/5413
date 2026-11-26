//! Runs bounded active exploration using ensemble disagreement on every fresh observation.

use engine_core::action::{
    ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionProposal, ActionRisk,
    AuthorizedAction,
};
use engine_core::active_exploration::{
    run_model_disagreement_exploration, ActiveExplorationRequest,
};
use engine_core::environment::{
    ActionOutcome, EnvironmentAdapter, EnvironmentDescriptor, EnvironmentError, ObservationFrame,
};
use engine_core::experiment_selection::{
    ModelExplorationCandidate, ModelExplorationPolicy,
};
use engine_core::experience_journal::ExperienceJournal;
use engine_core::learning::TransitionSample;
use engine_core::planning::PlanningConfig;
use engine_core::schema::{FieldSpec, SchemaDefinition, ValueKind};
use engine_core::world_model::{ActionDeltaModel, EnsembleWorldModel, PredictiveModel};
use engine_core::{DataValue, RuntimeConfig};
use std::error::Error;
use std::path::PathBuf;

struct CounterEnvironment {
    descriptor: EnvironmentDescriptor,
    position: f64,
    sequence: u64,
}

impl CounterEnvironment {
    fn new() -> Self {
        Self {
            descriptor: EnvironmentDescriptor {
                id: "active-counter".to_owned(),
                version: "v1".to_owned(),
                observation_schemas: vec![SchemaDefinition {
                    id: "counter-state".to_owned(),
                    version: 1,
                    fields: vec![FieldSpec {
                        name: "position".to_owned(),
                        kind: ValueKind::Number,
                        required: true,
                    }],
                    allow_additional_fields: false,
                }],
                actions: vec![
                    ActionDescriptor {
                        id: "advance".to_owned(),
                        arguments: empty_arguments("advance-args"),
                        risk: ActionRisk::Reversible,
                    },
                    ActionDescriptor {
                        id: "wait".to_owned(),
                        arguments: empty_arguments("wait-args"),
                        risk: ActionRisk::ReadOnly,
                    },
                ],
                supports_reset: false,
            },
            position: 0.0,
            sequence: 0,
        }
    }

    fn frame(&mut self) -> ObservationFrame {
        let frame = ObservationFrame {
            environment_id: "active-counter".to_owned(),
            session_id: "active-session".to_owned(),
            episode_id: "active-exploration-episode".to_owned(),
            sequence: self.sequence,
            observed_at_millis: self.sequence,
            schema_id: "counter-state".to_owned(),
            schema_version: 1,
            payload: DataValue::object([(
                "position".to_owned(),
                DataValue::Number(self.position),
            )]),
        };
        self.sequence += 1;
        frame
    }
}

impl EnvironmentAdapter for CounterEnvironment {
    fn descriptor(&self) -> &EnvironmentDescriptor {
        &self.descriptor
    }

    fn observe(&mut self) -> Result<ObservationFrame, EnvironmentError> {
        Ok(self.frame())
    }

    fn execute(
        &mut self,
        action: AuthorizedAction,
    ) -> Result<ActionOutcome, EnvironmentError> {
        match action.action_id() {
            "advance" => self.position += 1.0,
            "wait" => {}
            action_id => {
                return Err(EnvironmentError::InvalidAction(format!(
                    "unsupported action {action_id}"
                )))
            }
        }
        Ok(ActionOutcome {
            action_id: action.action_id().to_owned(),
            completed: true,
            observations: Vec::new(),
            feedback: DataValue::Null,
        })
    }
}

fn empty_arguments(id: &str) -> SchemaDefinition {
    SchemaDefinition {
        id: id.to_owned(),
        version: 1,
        fields: Vec::new(),
        allow_additional_fields: false,
    }
}

fn sample(action_id: &str, before: f64, after: f64) -> TransitionSample {
    TransitionSample {
        action_id: action_id.to_owned(),
        before: DataValue::object([("position".to_owned(), DataValue::Number(before))]),
        after: DataValue::object([("position".to_owned(), DataValue::Number(after))]),
    }
}

fn trained_ensemble() -> Result<EnsembleWorldModel, Box<dyn Error>> {
    let mut first = ActionDeltaModel::new("counter-model-a", 1)?;
    first.fit(
        "counter-training-a",
        &[
            sample("advance", 0.0, 1.0),
            sample("advance", 10.0, 11.0),
            sample("wait", 0.0, 0.0),
            sample("wait", 10.0, 10.0),
        ],
    )?;

    let mut second = ActionDeltaModel::new("counter-model-b", 1)?;
    second.fit(
        "counter-training-b",
        &[
            sample("advance", 0.0, 3.0),
            sample("advance", 10.0, 13.0),
            sample("wait", 0.0, 0.0),
            sample("wait", 10.0, 10.0),
        ],
    )?;

    Ok(EnsembleWorldModel::new(vec![
        Box::new(first) as Box<dyn PredictiveModel>,
        Box::new(second) as Box<dyn PredictiveModel>,
    ])?)
}

fn journal_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "5413-active-exploration-{}.bin",
        std::process::id()
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = journal_path();
    let mut journal = ExperienceJournal::open(&path)?;
    let mut environment = CounterEnvironment::new();
    let candidates = vec![
        ModelExplorationCandidate {
            id: "advance".to_owned(),
            action: ActionProposal {
                action_id: "advance".to_owned(),
                arguments: DataValue::object(std::iter::empty()),
            },
            resource_cost: 1.0,
        },
        ModelExplorationCandidate {
            id: "wait".to_owned(),
            action: ActionProposal {
                action_id: "wait".to_owned(),
                arguments: DataValue::object(std::iter::empty()),
            },
            resource_cost: 0.25,
        },
    ];

    let report = run_model_disagreement_exploration(
        &mut environment,
        &trained_ensemble()?,
        ActiveExplorationRequest {
            candidate_actions: candidates,
            selection_policy: ModelExplorationPolicy {
                maximum_resource_cost: 5.0,
                maximum_action_risk: ActionRisk::Reversible,
                task_utility_weight: 0.1,
                disagreement_weight: 1.0,
                resource_cost_weight: 0.01,
                risk_weight: 0.1,
            },
            runtime: RuntimeConfig {
                execution_id: "active-exploration-demo-1".to_owned(),
                episode_id: "active-exploration-episode".to_owned(),
                maximum_steps: 3,
                authorizer: ActionAuthorizer::new(ActionAuthorizationPolicy::new(
                    ActionRisk::Reversible,
                )),
                planning: PlanningConfig {
                    horizon: 1,
                    beam_width: 1,
                    uncertainty_penalty: 0.0,
                    uncalibrated_prediction_penalty: 0.0,
                },
            },
        },
        &mut journal,
        |state| {
            state
                .get("position")
                .and_then(DataValue::as_f64)
                .unwrap_or(0.0)
        },
    )?;

    for (step, ranking) in report.rankings.iter().enumerate() {
        let selected = ranking
            .ranked_actions
            .first()
            .ok_or("active exploration returned an empty ranking")?;
        println!(
            "step {}: selected {} (disagreement {:.3}, score {:.3})",
            step + 1,
            selected.candidate_id,
            selected.model_disagreement,
            selected.score
        );
    }
    let final_position = report
        .runtime
        .final_observation
        .payload
        .get("position")
        .and_then(DataValue::as_f64)
        .unwrap_or(0.0);
    println!(
        "executed {} bounded step(s); final position {:.1}; journal records {}",
        report.runtime.steps.len(),
        final_position,
        journal.len()
    );

    drop(journal);
    std::fs::remove_file(path)?;
    Ok(())
}
