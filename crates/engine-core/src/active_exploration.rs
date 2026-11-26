//! Dynamic model-disagreement exploration over fresh runtime observations.

use crate::environment::EnvironmentAdapter;
use crate::experiment_selection::{
    rank_model_disagreement_actions, ExperimentSelectionError, ModelExplorationCandidate,
    ModelExplorationPolicy, ModelExplorationRanking,
};
use crate::experience_journal::ExperienceJournal;
use crate::runtime::{
    run_receding_horizon_with_selector, RuntimeConfig, RuntimeError, RuntimeReport,
};
use crate::value::DataValue;
use crate::world_model::EnsembleWorldModel;
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct ActiveExplorationRequest {
    pub candidate_actions: Vec<ModelExplorationCandidate>,
    pub selection_policy: ModelExplorationPolicy,
    pub runtime: RuntimeConfig,
}

impl ActiveExplorationRequest {
    pub fn validate(&self, model: &EnsembleWorldModel) -> Result<(), ActiveExplorationError> {
        self.runtime
            .validate()
            .map_err(|_| ActiveExplorationError::InvalidRequest)?;
        self.selection_policy
            .validate()
            .map_err(ActiveExplorationError::Selection)?;
        if self.runtime.maximum_steps == 0
            || self.candidate_actions.is_empty()
            || model.len() < 2
        {
            return Err(ActiveExplorationError::InvalidRequest);
        }
        let mut ids = BTreeSet::new();
        for candidate in &self.candidate_actions {
            candidate
                .validate()
                .map_err(ActiveExplorationError::Selection)?;
            if !ids.insert(candidate.id.as_str()) {
                return Err(ActiveExplorationError::Selection(
                    ExperimentSelectionError::DuplicateCandidate,
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ActiveExplorationReport {
    /// Complete ranking snapshot for each state at which an action was selected.
    pub rankings: Vec<ModelExplorationRanking>,
    pub runtime: RuntimeReport,
}

/// Runs a bounded episode, selecting an action from ensemble disagreement at
/// every fresh observation. The runtime journals intent before executing each
/// selected action, re-authorizes it immediately before execution, measures the
/// observed transition, and then invokes the selector again on the new state.
pub fn run_model_disagreement_exploration<A, F>(
    adapter: &mut A,
    model: &EnsembleWorldModel,
    request: ActiveExplorationRequest,
    journal: &mut ExperienceJournal,
    utility: F,
) -> Result<ActiveExplorationReport, ActiveExplorationError>
where
    A: EnvironmentAdapter,
    F: Fn(&DataValue) -> f64,
{
    request.validate(model)?;
    let ActiveExplorationRequest {
        candidate_actions,
        selection_policy,
        runtime,
    } = request;
    let authorizer = runtime.authorizer;
    let mut rankings = Vec::with_capacity(runtime.maximum_steps);
    let mut selection_error = None;

    let execution = run_receding_horizon_with_selector(
        adapter,
        model,
        runtime,
        journal,
        |environment, state| {
            match rank_model_disagreement_actions(
                environment,
                authorizer,
                model,
                state,
                &candidate_actions,
                selection_policy,
                &utility,
            ) {
                Ok(ranking) => {
                    let Some(selected) = ranking.ranked_actions.first() else {
                        return Err("no exploration candidate was eligible".to_owned());
                    };
                    let selected_action = selected.action.clone();
                    rankings.push(ranking);
                    Ok(vec![selected_action])
                }
                Err(error) => {
                    selection_error = Some(error);
                    Err("model-disagreement ranking failed".to_owned())
                }
            }
        },
        &utility,
    );

    let runtime = match execution {
        Ok(report) => report,
        Err(RuntimeError::ActionSelection(_)) => {
            if let Some(error) = selection_error {
                return Err(ActiveExplorationError::Selection(error));
            }
            return Err(ActiveExplorationError::NoEligibleActions);
        }
        Err(error) => return Err(ActiveExplorationError::Runtime(error)),
    };
    Ok(ActiveExplorationReport { rankings, runtime })
}

#[derive(Debug)]
pub enum ActiveExplorationError {
    InvalidRequest,
    NoEligibleActions,
    Selection(ExperimentSelectionError),
    Runtime(RuntimeError),
}

impl std::fmt::Display for ActiveExplorationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest => write!(formatter, "active-exploration request is invalid"),
            Self::NoEligibleActions => write!(formatter, "no active-exploration action was eligible"),
            Self::Selection(error) => write!(formatter, "active-exploration selection failed: {error}"),
            Self::Runtime(error) => write!(formatter, "active-exploration runtime failed: {error}"),
        }
    }
}

impl std::error::Error for ActiveExplorationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{
        ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionProposal, ActionRisk,
        AuthorizedAction,
    };
    use crate::environment::{
        ActionOutcome, EnvironmentDescriptor, EnvironmentError, ObservationFrame,
    };
    use crate::experiment_selection::ModelExplorationCandidate;
    use crate::learning::TransitionSample;
    use crate::planning::PlanningConfig;
    use crate::schema::{FieldSpec, SchemaDefinition, ValueKind};
    use crate::world_model::{ActionDeltaModel, PredictiveModel};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn path() -> std::path::PathBuf {
        let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "5413-active-exploration-{}-{id}.bin",
            std::process::id()
        ))
    }

    struct CounterEnvironment {
        descriptor: EnvironmentDescriptor,
        position: f64,
        sequence: u64,
        executed: usize,
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
                            arguments: SchemaDefinition {
                                id: "empty".to_owned(),
                                version: 1,
                                fields: Vec::new(),
                                allow_additional_fields: false,
                            },
                            risk: ActionRisk::Reversible,
                        },
                        ActionDescriptor {
                            id: "wait".to_owned(),
                            arguments: SchemaDefinition {
                                id: "empty-wait".to_owned(),
                                version: 1,
                                fields: Vec::new(),
                                allow_additional_fields: false,
                            },
                            risk: ActionRisk::ReadOnly,
                        },
                    ],
                    supports_reset: false,
                },
                position: 0.0,
                sequence: 0,
                executed: 0,
            }
        }

        fn frame(&mut self) -> ObservationFrame {
            let frame = ObservationFrame {
                environment_id: "active-counter".to_owned(),
                session_id: "active-session".to_owned(),
                episode_id: "active-episode".to_owned(),
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
            self.executed += 1;
            Ok(ActionOutcome {
                action_id: action.action_id().to_owned(),
                completed: true,
                observations: Vec::new(),
                feedback: DataValue::Null,
            })
        }
    }

    fn model() -> EnsembleWorldModel {
        let sample = |action: &str, before: f64, after: f64| TransitionSample {
            action_id: action.to_owned(),
            before: DataValue::object([("position".to_owned(), DataValue::Number(before))]),
            after: DataValue::object([("position".to_owned(), DataValue::Number(after))]),
        };
        let mut first = ActionDeltaModel::new("active-model-a", 1).expect("first");
        first
            .fit(
                "active-training-a",
                &[
                    sample("advance", 0.0, 1.0),
                    sample("advance", 10.0, 11.0),
                    sample("wait", 0.0, 0.0),
                    sample("wait", 10.0, 10.0),
                ],
            )
            .expect("fit first");
        let mut second = ActionDeltaModel::new("active-model-b", 1).expect("second");
        second
            .fit(
                "active-training-b",
                &[
                    sample("advance", 0.0, 3.0),
                    sample("advance", 10.0, 13.0),
                    sample("wait", 0.0, 0.0),
                    sample("wait", 10.0, 10.0),
                ],
            )
            .expect("fit second");
        EnsembleWorldModel::new(vec![
            Box::new(first) as Box<dyn PredictiveModel>,
            Box::new(second) as Box<dyn PredictiveModel>,
        ])
        .expect("ensemble")
    }

    #[test]
    fn reselects_from_each_fresh_observation_and_executes_within_step_budget() {
        let file_path = path();
        let mut journal = ExperienceJournal::open(&file_path).expect("journal");
        let mut environment = CounterEnvironment::new();
        let request = ActiveExplorationRequest {
            candidate_actions: vec![
                ModelExplorationCandidate {
                    id: "advance".to_owned(),
                    action: ActionProposal {
                        action_id: "advance".to_owned(),
                        arguments: DataValue::object(std::iter::empty()),
                    },
                    resource_cost: 0.0,
                },
                ModelExplorationCandidate {
                    id: "wait".to_owned(),
                    action: ActionProposal {
                        action_id: "wait".to_owned(),
                        arguments: DataValue::object(std::iter::empty()),
                    },
                    resource_cost: 0.0,
                },
            ],
            selection_policy: ModelExplorationPolicy {
                maximum_resource_cost: 1.0,
                maximum_action_risk: ActionRisk::Reversible,
                task_utility_weight: 0.0,
                disagreement_weight: 1.0,
                resource_cost_weight: 0.0,
                risk_weight: 0.0,
            },
            runtime: RuntimeConfig {
                execution_id: "active-exploration-run-1".to_owned(),
                episode_id: "active-episode".to_owned(),
                maximum_steps: 2,
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
        };

        let report = run_model_disagreement_exploration(
            &mut environment,
            &model(),
            request,
            &mut journal,
            |_| 0.0,
        )
        .expect("active exploration");
        assert_eq!(report.runtime.steps.len(), 2);
        assert_eq!(report.rankings.len(), 2);
        assert!(report.rankings.iter().all(|ranking| {
            ranking
                .ranked_actions
                .first()
                .is_some_and(|action| action.candidate_id == "advance")
        }));
        assert_eq!(environment.executed, 2);
        assert_eq!(environment.position, 2.0);
        drop(journal);
        std::fs::remove_file(file_path).expect("cleanup");
    }
}
