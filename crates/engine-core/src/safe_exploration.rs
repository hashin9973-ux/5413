//! Reset-aware, globally budgeted multi-episode active exploration.
//!
//! This supervisor executes one action per runtime invocation so it can account
//! for the selected proposal's declared resource cost before starting another
//! action. It requires an explicit environment reset contract for multi-episode
//! runs and preserves completed results when later work fails.

use crate::active_exploration::{
    run_model_disagreement_exploration, ActiveExplorationReport, ActiveExplorationRequest,
};
use crate::environment::{EnvironmentAdapter, LifecycleOutcome};
use crate::experience::ExperienceEvent;
use crate::experience_journal::ExperienceJournal;
use crate::experiment_selection::{
    ExperimentSelectionError, ModelExplorationCandidate, ModelExplorationPolicy,
};
use crate::runtime::{RuntimeConfig, RuntimeStepReport};
use crate::value::DataValue;
use crate::world_model::EnsembleWorldModel;
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct SafeExplorationRequest {
    pub candidate_actions: Vec<ModelExplorationCandidate>,
    pub maximum_actions_per_plan: usize,
    pub selection_policy: ModelExplorationPolicy,
    /// Supplies the initial episode ID and a base execution ID. Each step gets
    /// a derived unique execution ID; this config's maximum_steps is ignored.
    pub runtime: RuntimeConfig,
    pub maximum_episodes: usize,
    pub maximum_steps_per_episode: usize,
    pub maximum_total_steps: usize,
    pub maximum_total_resource_cost: f64,
}

impl SafeExplorationRequest {
    pub fn validate(
        &self,
        adapter: &impl EnvironmentAdapter,
        model: &EnsembleWorldModel,
    ) -> Result<(), SafeExplorationError> {
        self.runtime
            .validate()
            .map_err(|_| SafeExplorationError::InvalidRequest)?;
        self.selection_policy
            .validate()
            .map_err(SafeExplorationError::Selection)?;
        if self.candidate_actions.is_empty()
            || self.maximum_actions_per_plan == 0
            || self.maximum_episodes == 0
            || self.maximum_steps_per_episode == 0
            || self.maximum_total_steps == 0
            || !self.maximum_total_resource_cost.is_finite()
            || self.maximum_total_resource_cost < 0.0
            || self.runtime.maximum_steps != 1
            || model.len() < 2
        {
            return Err(SafeExplorationError::InvalidRequest);
        }
        let mut candidate_ids = BTreeSet::new();
        for candidate in &self.candidate_actions {
            candidate
                .validate()
                .map_err(SafeExplorationError::Selection)?;
            if !candidate_ids.insert(candidate.id.as_str()) {
                return Err(SafeExplorationError::Selection(
                    ExperimentSelectionError::DuplicateCandidate,
                ));
            }
        }
        if self.maximum_episodes > 1 && !adapter.descriptor().supports_reset {
            return Err(SafeExplorationError::ResetUnsupported);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SafeExplorationTermination {
    EpisodeBudgetReached,
    StepBudgetReached,
    ResourceBudgetReached,
    StoppedAfterFailure,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SafeExplorationFailure {
    ResetFailed(String),
    ResetReturnedEmptyEpisodeId,
    ResetReusedEpisodeId(String),
    DescriptorChangedAfterReset,
    ExplorationFailed(String),
    RuntimeReportShapeMismatch,
    ActionReportedIncomplete,
    MissingActionIntent,
    SelectedActionMissingFromRanking,
    NumericalOverflow,
}

#[derive(Debug)]
pub struct SafeExplorationEpisodeReport {
    pub episode_index: usize,
    pub episode_id: String,
    /// The reset result that created this episode; absent for the initial episode.
    pub reset_before: Option<LifecycleOutcome>,
    pub rankings: Vec<crate::experiment_selection::ModelExplorationRanking>,
    pub steps: Vec<RuntimeStepReport>,
    pub resource_cost: f64,
}

#[derive(Debug)]
pub struct SafeExplorationReport {
    pub episodes: Vec<SafeExplorationEpisodeReport>,
    pub total_steps: usize,
    pub total_resource_cost: f64,
    pub termination: SafeExplorationTermination,
    pub failure: Option<SafeExplorationFailure>,
}

/// Runs multiple bounded exploration episodes under one total step and cost
/// budget. Each step is a separate one-step active-exploration run so the
/// supervisor can account for the exact selected proposal before continuing.
///
/// Reset is called only between episodes and only when the descriptor declares
/// reset support. A reset's returned episode ID is validated and used for the
/// next runtime call. Reset outcomes are reported here; the current journal
/// event schema does not persist lifecycle-reset events.
pub fn run_safe_exploration<A, U>(
    adapter: &mut A,
    model: &EnsembleWorldModel,
    request: SafeExplorationRequest,
    journal: &mut ExperienceJournal,
    utility: U,
) -> Result<SafeExplorationReport, SafeExplorationError>
where
    A: EnvironmentAdapter,
    U: Fn(&DataValue) -> f64,
{
    request.validate(adapter, model)?;
    let initial_descriptor = adapter.descriptor().clone();
    initial_descriptor
        .validate()
        .map_err(|error| SafeExplorationError::InvalidEnvironment(error.to_string()))?;

    let mut episode_ids = BTreeSet::new();
    episode_ids.insert(request.runtime.episode_id.clone());
    let mut current_episode_id = request.runtime.episode_id.clone();
    let mut episodes = Vec::with_capacity(request.maximum_episodes);
    let mut total_steps = 0usize;
    let mut total_resource_cost = 0.0;
    let mut termination = SafeExplorationTermination::EpisodeBudgetReached;
    let mut failure = None;

    for episode_index in 0..request.maximum_episodes {
        let reset_before = if episode_index == 0 {
            None
        } else {
            match adapter.reset() {
                Ok(outcome) => {
                    if outcome.episode_id.trim().is_empty() {
                        failure = Some(SafeExplorationFailure::ResetReturnedEmptyEpisodeId);
                        termination = SafeExplorationTermination::StoppedAfterFailure;
                        break;
                    }
                    if !episode_ids.insert(outcome.episode_id.clone()) {
                        failure = Some(SafeExplorationFailure::ResetReusedEpisodeId(
                            outcome.episode_id,
                        ));
                        termination = SafeExplorationTermination::StoppedAfterFailure;
                        break;
                    }
                    let descriptor = adapter.descriptor();
                    if descriptor != &initial_descriptor {
                        failure = Some(SafeExplorationFailure::DescriptorChangedAfterReset);
                        termination = SafeExplorationTermination::StoppedAfterFailure;
                        break;
                    }
                    current_episode_id = outcome.episode_id.clone();
                    Some(outcome)
                }
                Err(error) => {
                    failure = Some(SafeExplorationFailure::ResetFailed(error.to_string()));
                    termination = SafeExplorationTermination::StoppedAfterFailure;
                    break;
                }
            }
        };

        let mut episode = SafeExplorationEpisodeReport {
            episode_index,
            episode_id: current_episode_id.clone(),
            reset_before,
            rankings: Vec::new(),
            steps: Vec::new(),
            resource_cost: 0.0,
        };

        for step_index in 0..request.maximum_steps_per_episode {
            if total_steps >= request.maximum_total_steps {
                termination = SafeExplorationTermination::StepBudgetReached;
                break;
            }
            let remaining_cost = request.maximum_total_resource_cost - total_resource_cost;
            if remaining_cost < 0.0 {
                termination = SafeExplorationTermination::ResourceBudgetReached;
                break;
            }

            let eligible_candidates: Vec<ModelExplorationCandidate> = request
                .candidate_actions
                .iter()
                .filter(|candidate| {
                    candidate.resource_cost <= remaining_cost
                        && candidate.resource_cost <= request.selection_policy.maximum_resource_cost
                })
                .cloned()
                .collect();
            if eligible_candidates.is_empty() {
                termination = SafeExplorationTermination::ResourceBudgetReached;
                break;
            }

            let mut selection_policy = request.selection_policy;
            selection_policy.maximum_resource_cost =
                selection_policy.maximum_resource_cost.min(remaining_cost);
            let mut runtime = request.runtime.clone();
            runtime.episode_id = current_episode_id.clone();
            runtime.maximum_steps = 1;
            runtime.execution_id = format!(
                "{}/episode/{episode_index}/step/{step_index}",
                request.runtime.execution_id
            );

            let before_journal_len = journal.len();
            let exploration = run_model_disagreement_exploration(
                adapter,
                model,
                ActiveExplorationRequest {
                    candidate_actions: eligible_candidates,
                    maximum_actions_per_plan: request.maximum_actions_per_plan,
                    selection_policy,
                    runtime,
                },
                journal,
                &utility,
            );
            let ActiveExplorationReport { rankings, runtime } = match exploration {
                Ok(report) => report,
                Err(error) => {
                    failure = Some(SafeExplorationFailure::ExplorationFailed(error.to_string()));
                    termination = SafeExplorationTermination::StoppedAfterFailure;
                    break;
                }
            };

            if runtime.steps.len() != 1 || rankings.len() != 1 {
                failure = Some(SafeExplorationFailure::RuntimeReportShapeMismatch);
                termination = SafeExplorationTermination::StoppedAfterFailure;
                break;
            }
            let step = runtime
                .steps
                .into_iter()
                .next()
                .ok_or(SafeExplorationError::InternalInvariant)?;
            let ranking = rankings
                .into_iter()
                .next()
                .ok_or(SafeExplorationError::InternalInvariant)?;

            let action_intent = journal.records()[before_journal_len..]
                .iter()
                .find_map(|record| match &record.event {
                    ExperienceEvent::ActionProposed {
                        action_id,
                        arguments,
                    } if action_id == &step.action_id => Some((action_id, arguments)),
                    _ => None,
                });
            let Some((action_id, arguments)) = action_intent else {
                failure = Some(SafeExplorationFailure::MissingActionIntent);
                termination = SafeExplorationTermination::StoppedAfterFailure;
                break;
            };
            // The planner chooses an ActionProposal, not the candidate ID. If
            // two candidate IDs map to the same proposal, charge the largest
            // declared cost because the selection provenance is ambiguous.
            let selected_cost = ranking
                .ranked_actions
                .iter()
                .filter(|candidate| {
                    candidate.action.action_id == *action_id
                        && candidate.action.arguments == *arguments
                })
                .map(|candidate| candidate.resource_cost)
                .reduce(f64::max);
            let Some(selected_cost) = selected_cost else {
                failure = Some(SafeExplorationFailure::SelectedActionMissingFromRanking);
                termination = SafeExplorationTermination::StoppedAfterFailure;
                break;
            };

            let next_episode_cost = episode.resource_cost + selected_cost;
            let next_total_cost = total_resource_cost + selected_cost;
            if !next_episode_cost.is_finite()
                || !next_total_cost.is_finite()
                || next_total_cost > request.maximum_total_resource_cost
            {
                failure = Some(SafeExplorationFailure::NumericalOverflow);
                termination = SafeExplorationTermination::StoppedAfterFailure;
                break;
            }
            episode.resource_cost = next_episode_cost;
            total_resource_cost = next_total_cost;
            total_steps += 1;
            episode.rankings.push(ranking);
            episode.steps.push(step);

            if episode
                .steps
                .last()
                .is_some_and(|step| !step.outcome.completed)
            {
                failure = Some(SafeExplorationFailure::ActionReportedIncomplete);
                termination = SafeExplorationTermination::StoppedAfterFailure;
                break;
            }
        }

        let episode_has_steps = !episode.steps.is_empty();
        episodes.push(episode);
        if failure.is_some() {
            break;
        }
        if total_steps >= request.maximum_total_steps {
            termination = SafeExplorationTermination::StepBudgetReached;
            break;
        }
        if termination == SafeExplorationTermination::ResourceBudgetReached {
            break;
        }
        if !episode_has_steps && episode_index + 1 < request.maximum_episodes {
            termination = SafeExplorationTermination::StoppedAfterFailure;
            break;
        }
    }

    Ok(SafeExplorationReport {
        episodes,
        total_steps,
        total_resource_cost,
        termination,
        failure,
    })
}

#[derive(Debug)]
pub enum SafeExplorationError {
    InvalidRequest,
    ResetUnsupported,
    InvalidEnvironment(String),
    Selection(crate::experiment_selection::ExperimentSelectionError),
    InternalInvariant,
}

impl std::fmt::Display for SafeExplorationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest => write!(formatter, "safe-exploration request is invalid"),
            Self::ResetUnsupported => write!(
                formatter,
                "multi-episode safe exploration requires declared environment reset support"
            ),
            Self::InvalidEnvironment(message) => write!(
                formatter,
                "safe-exploration environment is invalid: {message}"
            ),
            Self::Selection(error) => write!(
                formatter,
                "safe-exploration selection policy is invalid: {error}"
            ),
            Self::InternalInvariant => write!(
                formatter,
                "safe-exploration runtime report violated an internal invariant"
            ),
        }
    }
}

impl std::error::Error for SafeExplorationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{
        ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionProposal, ActionRisk,
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

    fn path(label: &str) -> std::path::PathBuf {
        let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "5413-safe-exploration-{}-{label}-{id}.bin",
            std::process::id()
        ))
    }

    struct ResettableCounter {
        descriptor: EnvironmentDescriptor,
        position: f64,
        sequence: u64,
        episode_number: usize,
        executions: usize,
    }

    impl ResettableCounter {
        fn new(supports_reset: bool) -> Self {
            Self {
                descriptor: EnvironmentDescriptor {
                    id: "safe-counter".to_owned(),
                    version: "v1".to_owned(),
                    observation_schemas: vec![SchemaDefinition {
                        id: "state".to_owned(),
                        version: 1,
                        fields: vec![FieldSpec {
                            name: "position".to_owned(),
                            kind: ValueKind::Number,
                            required: true,
                        }],
                        allow_additional_fields: false,
                    }],
                    actions: vec![ActionDescriptor {
                        id: "advance".to_owned(),
                        arguments: SchemaDefinition {
                            id: "empty".to_owned(),
                            version: 1,
                            fields: Vec::new(),
                            allow_additional_fields: false,
                        },
                        risk: ActionRisk::Reversible,
                    }],
                    supports_reset,
                },
                position: 0.0,
                sequence: 0,
                episode_number: 0,
                executions: 0,
            }
        }

        fn episode_id(&self) -> String {
            format!("safe-episode-{}", self.episode_number)
        }

        fn frame(&mut self) -> ObservationFrame {
            let frame = ObservationFrame {
                environment_id: "safe-counter".to_owned(),
                session_id: "safe-session".to_owned(),
                episode_id: self.episode_id(),
                sequence: self.sequence,
                observed_at_millis: self.sequence,
                schema_id: "state".to_owned(),
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

    impl EnvironmentAdapter for ResettableCounter {
        fn descriptor(&self) -> &EnvironmentDescriptor {
            &self.descriptor
        }

        fn observe(&mut self) -> Result<ObservationFrame, EnvironmentError> {
            Ok(self.frame())
        }

        fn execute(
            &mut self,
            _action: crate::action::AuthorizedAction,
        ) -> Result<ActionOutcome, EnvironmentError> {
            self.position += 1.0;
            self.executions += 1;
            Ok(ActionOutcome {
                action_id: "advance".to_owned(),
                completed: true,
                observations: Vec::new(),
                feedback: DataValue::Null,
            })
        }

        fn reset(&mut self) -> Result<LifecycleOutcome, EnvironmentError> {
            if !self.descriptor.supports_reset {
                return Err(EnvironmentError::UnsupportedOperation("reset"));
            }
            self.position = 0.0;
            self.sequence = 0;
            self.episode_number += 1;
            Ok(LifecycleOutcome {
                episode_id: self.episode_id(),
            })
        }
    }

    fn model() -> EnsembleWorldModel {
        let mut first = ActionDeltaModel::new("safe-model-a", 1).expect("first");
        first
            .fit(
                "safe-training",
                &[
                    TransitionSample {
                        action_id: "advance".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(0.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(1.0))]),
                    },
                    TransitionSample {
                        action_id: "advance".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(1.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(2.0))]),
                    },
                ],
            )
            .expect("fit first");
        let mut second = ActionDeltaModel::new("safe-model-b", 1).expect("second");
        second
            .fit(
                "safe-training",
                &[
                    TransitionSample {
                        action_id: "advance".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(0.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(1.5))]),
                    },
                    TransitionSample {
                        action_id: "advance".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(1.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(2.5))]),
                    },
                ],
            )
            .expect("fit second");
        EnsembleWorldModel::new(vec![
            Box::new(first) as Box<dyn PredictiveModel>,
            Box::new(second) as Box<dyn PredictiveModel>,
        ])
        .expect("ensemble")
    }

    fn request(episodes: usize, total_steps: usize, total_cost: f64) -> SafeExplorationRequest {
        SafeExplorationRequest {
            candidate_actions: vec![ModelExplorationCandidate {
                id: "advance-once".to_owned(),
                action: ActionProposal {
                    action_id: "advance".to_owned(),
                    arguments: DataValue::object(std::iter::empty()),
                },
                resource_cost: 1.0,
            }],
            maximum_actions_per_plan: 1,
            selection_policy: ModelExplorationPolicy {
                maximum_resource_cost: 100.0,
                maximum_action_risk: ActionRisk::Reversible,
                task_utility_weight: 1.0,
                disagreement_weight: 1.0,
                resource_cost_weight: 0.0,
                risk_weight: 0.0,
            },
            runtime: RuntimeConfig {
                execution_id: "safe-run".to_owned(),
                episode_id: "safe-episode-0".to_owned(),
                maximum_steps: 1,
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
            maximum_episodes: episodes,
            maximum_steps_per_episode: 2,
            maximum_total_steps: total_steps,
            maximum_total_resource_cost: total_cost,
        }
    }

    #[test]
    fn runs_multiple_reset_episodes_with_unique_journal_execution_ids() {
        let journal_path = path("episodes");
        let mut journal = ExperienceJournal::open(&journal_path).expect("journal");
        let mut environment = ResettableCounter::new(true);
        let report = run_safe_exploration(
            &mut environment,
            &model(),
            request(2, 4, 4.0),
            &mut journal,
            |state| {
                state
                    .get("position")
                    .and_then(DataValue::as_f64)
                    .unwrap_or(0.0)
            },
        )
        .expect("safe exploration");
        assert_eq!(report.episodes.len(), 2);
        assert_eq!(report.episodes[0].episode_id, "safe-episode-0");
        assert_eq!(report.episodes[1].episode_id, "safe-episode-1");
        assert!(report.episodes[0].reset_before.is_none());
        assert_eq!(
            report.episodes[1]
                .reset_before
                .as_ref()
                .map(|item| item.episode_id.as_str()),
            Some("safe-episode-1")
        );
        assert_eq!(report.total_steps, 4);
        assert_eq!(report.total_resource_cost, 4.0);
        assert_eq!(environment.executions, 4);
        assert_eq!(
            report.termination,
            SafeExplorationTermination::StepBudgetReached
        );
        drop(journal);
        std::fs::remove_file(journal_path).expect("journal cleanup");
    }

    #[test]
    fn refuses_multi_episode_execution_without_declared_reset_support() {
        let journal_path = path("no-reset");
        let mut journal = ExperienceJournal::open(&journal_path).expect("journal");
        let mut environment = ResettableCounter::new(false);
        let result = run_safe_exploration(
            &mut environment,
            &model(),
            request(2, 4, 4.0),
            &mut journal,
            |_| 0.0,
        );
        assert!(matches!(
            result,
            Err(SafeExplorationError::ResetUnsupported)
        ));
        assert_eq!(environment.executions, 0);
        assert!(journal.is_empty());
        drop(journal);
        std::fs::remove_file(journal_path).expect("journal cleanup");
    }

    #[test]
    fn enforces_one_global_resource_budget_across_steps() {
        let journal_path = path("resource-budget");
        let mut journal = ExperienceJournal::open(&journal_path).expect("journal");
        let mut environment = ResettableCounter::new(true);
        let report = run_safe_exploration(
            &mut environment,
            &model(),
            request(1, 5, 1.0),
            &mut journal,
            |_| 0.0,
        )
        .expect("safe exploration");
        assert_eq!(report.total_steps, 1);
        assert_eq!(report.total_resource_cost, 1.0);
        assert_eq!(
            report.termination,
            SafeExplorationTermination::ResourceBudgetReached
        );
        assert_eq!(environment.executions, 1);
        drop(journal);
        std::fs::remove_file(journal_path).expect("journal cleanup");
    }
}
