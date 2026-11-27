//! Safe, bounded receding-horizon execution over an environment adapter.
//!
//! Each action is planned from fresh state. A durable idempotent intent is
//! appended before invoking the adapter. Reusing an execution ID is blocked:
//! generic adapters cannot promise exactly-once external side effects.

use crate::action::{ActionAuthorizer, ActionProposal};
use crate::environment::{
    ActionOutcome, EnvironmentAdapter, EnvironmentDescriptor, ObservationFrame,
};
use crate::experience::ExperienceEvent;
use crate::experience_journal::{ExperienceJournal, IdempotentAppendOutcome, JournalError};
use crate::planning::{plan, PlanningConfig, PlanningError};
use crate::regression_metrics::RegressionErrorAccumulator;
use crate::value::DataValue;
use crate::world_model::EnsembleWorldModel;

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub execution_id: String,
    pub episode_id: String,
    pub maximum_steps: usize,
    pub authorizer: ActionAuthorizer,
    pub planning: PlanningConfig,
}

impl RuntimeConfig {
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.execution_id.trim().is_empty()
            || self.episode_id.trim().is_empty()
            || self.maximum_steps == 0
        {
            return Err(RuntimeError::InvalidConfiguration);
        }
        if self.planning.horizon == 0
            || self.planning.beam_width == 0
            || !self.planning.uncertainty_penalty.is_finite()
            || self.planning.uncertainty_penalty < 0.0
            || !self.planning.uncalibrated_prediction_penalty.is_finite()
            || self.planning.uncalibrated_prediction_penalty < 0.0
        {
            return Err(RuntimeError::InvalidConfiguration);
        }
        Ok(())
    }

    fn key(&self, suffix: &str) -> String {
        format!("5413/runtime/{}/{}", self.execution_id, suffix)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeTermination {
    StepLimitReached,
    EnvironmentReportedIncomplete,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictionErrorMetrics {
    pub numeric_fields: usize,
    pub mean_absolute_error: Option<f64>,
    pub root_mean_squared_error: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeStepReport {
    pub step_index: usize,
    pub action_id: String,
    pub predicted_state: DataValue,
    pub observed_state: DataValue,
    pub prediction_error: PredictionErrorMetrics,
    pub outcome: ActionOutcome,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeReport {
    pub initial_observation: ObservationFrame,
    pub final_observation: ObservationFrame,
    pub steps: Vec<RuntimeStepReport>,
    pub termination: RuntimeTermination,
}

/// Executes only the first action from each plan and replans from the actual
/// next observation. Action arguments remain caller-supplied.
pub fn run_receding_horizon<A, F>(
    adapter: &mut A,
    model: &EnsembleWorldModel,
    candidate_actions: &[ActionProposal],
    config: RuntimeConfig,
    journal: &mut ExperienceJournal,
    utility: F,
) -> Result<RuntimeReport, RuntimeError>
where
    A: EnvironmentAdapter,
    F: Fn(&DataValue) -> f64,
{
    run_receding_horizon_with_selector(
        adapter,
        model,
        config,
        journal,
        |_, _| Ok(candidate_actions.to_vec()),
        utility,
    )
}

/// Executes a bounded episode while obtaining candidate actions from the latest
/// validated observation. The selector is called again after every action and
/// observation, allowing active exploration to adapt to newly measured state.
pub fn run_receding_horizon_with_selector<A, F, S>(
    adapter: &mut A,
    model: &EnsembleWorldModel,
    config: RuntimeConfig,
    journal: &mut ExperienceJournal,
    mut select_actions: S,
    utility: F,
) -> Result<RuntimeReport, RuntimeError>
where
    A: EnvironmentAdapter,
    F: Fn(&DataValue) -> f64,
    S: FnMut(&EnvironmentDescriptor, &DataValue) -> Result<Vec<ActionProposal>, String>,
{
    config.validate()?;
    let descriptor = adapter.descriptor().clone();
    descriptor
        .validate()
        .map_err(|error| RuntimeError::Environment(error.to_string()))?;

    let start_key = config.key("started");
    if journal.has_idempotency_key(&start_key) {
        return Err(RuntimeError::ExecutionAlreadyStarted);
    }

    let initial_observation = adapter
        .observe()
        .map_err(|error| RuntimeError::Environment(error.to_string()))?;
    validate_observation(&descriptor, &initial_observation, &config.episode_id)?;
    match journal.append_idempotent(
        start_key.as_str(),
        &config.episode_id,
        ExperienceEvent::Observation(initial_observation.clone()),
    )? {
        IdempotentAppendOutcome::Appended { .. } => {}
        IdempotentAppendOutcome::AlreadyPresent { .. } => {
            return Err(RuntimeError::ExecutionAlreadyStarted);
        }
    }

    let mut current_observation = initial_observation.clone();
    let mut steps = Vec::with_capacity(config.maximum_steps);
    let mut termination = RuntimeTermination::StepLimitReached;

    for step_index in 0..config.maximum_steps {
        let current_descriptor = adapter.descriptor();
        if current_descriptor.id != descriptor.id
            || current_descriptor.version != descriptor.version
        {
            return Err(RuntimeError::EnvironmentDescriptorChanged);
        }

        let candidate_actions = select_actions(&descriptor, &current_observation.payload)
            .map_err(RuntimeError::ActionSelection)?;
        if candidate_actions.is_empty() {
            return Err(RuntimeError::ActionSelection(
                "selector returned no eligible action proposals".to_owned(),
            ));
        }
        let selected_plan = plan(
            &descriptor,
            config.authorizer,
            model,
            &current_observation.payload,
            &candidate_actions,
            config.planning,
            &utility,
        )
        .map_err(RuntimeError::Planning)?;
        let selected = selected_plan
            .steps
            .first()
            .ok_or(RuntimeError::PlannerReturnedEmptyPlan)?;
        let proposal = selected.action.clone();
        let predicted_state = selected.predicted_state.clone();

        // Re-authorize immediately before recording intent and executing.
        let authorized = config
            .authorizer
            .authorize(&descriptor, proposal.clone())
            .map_err(|error| RuntimeError::Authorization(error.to_string()))?;

        let intent_key = config.key(&format!("step/{step_index}/intent"));
        match journal.append_idempotent(
            intent_key.as_str(),
            &config.episode_id,
            ExperienceEvent::ActionProposed {
                action_id: proposal.action_id.clone(),
                arguments: proposal.arguments.clone(),
            },
        )? {
            IdempotentAppendOutcome::Appended { .. } => {}
            IdempotentAppendOutcome::AlreadyPresent { .. } => {
                return Err(RuntimeError::DuplicateExecutionIntent { step_index });
            }
        }

        // An adapter error after the intent is durable has ambiguous side-effect status.
        let outcome = adapter
            .execute(authorized)
            .map_err(|error| RuntimeError::ExecutionOutcomeAmbiguous(error.to_string()))?;

        // Persist the raw outcome even when it violates the requested action contract.
        append_once(
            journal,
            &config.key(&format!("step/{step_index}/completed")),
            &config.episode_id,
            ExperienceEvent::ActionCompleted(outcome.clone()),
        )?;
        if outcome.action_id != proposal.action_id {
            return Err(RuntimeError::OutcomeActionMismatch {
                expected: proposal.action_id,
                actual: outcome.action_id,
            });
        }

        for (index, observation) in outcome.observations.iter().enumerate() {
            validate_observation(&descriptor, observation, &config.episode_id)?;
            append_once(
                journal,
                &config.key(&format!("step/{step_index}/adapter-observation/{index}")),
                &config.episode_id,
                ExperienceEvent::Observation(observation.clone()),
            )?;
        }

        let observed = adapter
            .observe()
            .map_err(|error| RuntimeError::Environment(error.to_string()))?;
        validate_observation(&descriptor, &observed, &config.episode_id)?;
        append_once(
            journal,
            &config.key(&format!("step/{step_index}/observation")),
            &config.episode_id,
            ExperienceEvent::Observation(observed.clone()),
        )?;

        let prediction_error = prediction_error(&predicted_state, &observed.payload);
        append_once(
            journal,
            &config.key(&format!("step/{step_index}/prediction-evaluation")),
            &config.episode_id,
            ExperienceEvent::PredictionEvaluated {
                action_id: proposal.action_id.clone(),
                predicted: predicted_state.clone(),
                observed: observed.payload.clone(),
            },
        )?;

        steps.push(RuntimeStepReport {
            step_index,
            action_id: proposal.action_id,
            predicted_state,
            observed_state: observed.payload.clone(),
            prediction_error,
            outcome: outcome.clone(),
        });
        current_observation = observed;

        if !outcome.completed {
            termination = RuntimeTermination::EnvironmentReportedIncomplete;
            break;
        }
    }

    Ok(RuntimeReport {
        initial_observation,
        final_observation: current_observation,
        steps,
        termination,
    })
}

fn validate_observation(
    descriptor: &EnvironmentDescriptor,
    observation: &ObservationFrame,
    expected_episode_id: &str,
) -> Result<(), RuntimeError> {
    descriptor
        .validate_observation(observation)
        .map_err(|error| RuntimeError::Environment(error.to_string()))?;
    if observation.episode_id != expected_episode_id {
        return Err(RuntimeError::EpisodeMismatch {
            expected: expected_episode_id.to_owned(),
            actual: observation.episode_id.clone(),
        });
    }
    observation
        .payload
        .validate_finite()
        .map_err(|_| RuntimeError::InvalidObservation)
}

fn append_once(
    journal: &mut ExperienceJournal,
    key: &str,
    episode_id: &str,
    event: ExperienceEvent,
) -> Result<(), RuntimeError> {
    match journal.append_idempotent(key, episode_id, event)? {
        IdempotentAppendOutcome::Appended { .. }
        | IdempotentAppendOutcome::AlreadyPresent { .. } => Ok(()),
    }
}

fn prediction_error(predicted: &DataValue, observed: &DataValue) -> PredictionErrorMetrics {
    let mut errors = RegressionErrorAccumulator::default();

    if let (Some(predicted), Some(observed)) = (predicted.as_object(), observed.as_object()) {
        for (name, predicted_value) in predicted {
            let Some(predicted_number) = predicted_value.as_f64() else {
                continue;
            };
            let Some(observed_number) = observed.get(name).and_then(DataValue::as_f64) else {
                continue;
            };
            errors.record_difference(predicted_number, observed_number);
        }
    }

    let metrics = errors.finish();
    PredictionErrorMetrics {
        numeric_fields: errors.numeric_fields(),
        mean_absolute_error: metrics.map(|summary| summary.mean_absolute_error),
        root_mean_squared_error: metrics.map(|summary| summary.root_mean_squared_error),
    }
}

#[derive(Debug)]
pub enum RuntimeError {
    InvalidConfiguration,
    ExecutionAlreadyStarted,
    DuplicateExecutionIntent { step_index: usize },
    Environment(String),
    EnvironmentDescriptorChanged,
    Authorization(String),
    ActionSelection(String),
    Planning(PlanningError),
    PlannerReturnedEmptyPlan,
    ExecutionOutcomeAmbiguous(String),
    OutcomeActionMismatch { expected: String, actual: String },
    EpisodeMismatch { expected: String, actual: String },
    InvalidObservation,
    Journal(JournalError),
}

impl From<JournalError> for RuntimeError {
    fn from(error: JournalError) -> Self {
        Self::Journal(error)
    }
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration => write!(formatter, "runtime configuration is invalid"),
            Self::ExecutionAlreadyStarted => write!(formatter, "execution ID has already been used; automatic replay is blocked"),
            Self::DuplicateExecutionIntent { step_index } => write!(formatter, "step {step_index} already has a durable intent; automatic replay is blocked"),
            Self::Environment(message) => write!(formatter, "environment operation failed: {message}"),
            Self::EnvironmentDescriptorChanged => write!(formatter, "environment descriptor changed during execution"),
            Self::Authorization(message) => write!(formatter, "action authorization failed: {message}"),
            Self::ActionSelection(message) => write!(formatter, "action selector failed: {message}"),
            Self::Planning(error) => write!(formatter, "planning failed: {error}"),
            Self::PlannerReturnedEmptyPlan => write!(formatter, "planner returned no first step"),
            Self::ExecutionOutcomeAmbiguous(message) => write!(formatter, "adapter execution failed after durable intent; side-effect status is ambiguous: {message}"),
            Self::OutcomeActionMismatch { expected, actual } => write!(formatter, "adapter completed action {actual}, but runtime requested {expected}"),
            Self::EpisodeMismatch { expected, actual } => write!(formatter, "observation episode mismatch: expected {expected}, received {actual}"),
            Self::InvalidObservation => write!(formatter, "observation contains non-finite values"),
            Self::Journal(error) => write!(formatter, "journal operation failed: {error}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{ActionAuthorizationPolicy, ActionDescriptor, ActionRisk};
    use crate::environment::{EnvironmentDescriptor, EnvironmentError};
    use crate::learning::TransitionSample;
    use crate::schema::{FieldSpec, SchemaDefinition, ValueKind};
    use crate::world_model::{ActionDeltaModel, PredictiveModel};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

    fn test_path() -> std::path::PathBuf {
        let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("5413-runtime-{}-{id}.bin", std::process::id()))
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
                    id: "counter".to_owned(),
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
                    actions: vec![ActionDescriptor {
                        id: "advance".to_owned(),
                        arguments: SchemaDefinition {
                            id: "empty-args".to_owned(),
                            version: 1,
                            fields: Vec::new(),
                            allow_additional_fields: false,
                        },
                        risk: ActionRisk::Reversible,
                    }],
                    supports_reset: false,
                },
                position: 0.0,
                sequence: 0,
                executed: 0,
            }
        }

        fn next_frame(&mut self) -> ObservationFrame {
            let frame = ObservationFrame {
                environment_id: "counter".to_owned(),
                session_id: "session-1".to_owned(),
                episode_id: "episode-1".to_owned(),
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
            Ok(self.next_frame())
        }

        fn execute(
            &mut self,
            _action: crate::action::AuthorizedAction,
        ) -> Result<ActionOutcome, EnvironmentError> {
            self.position += 1.0;
            self.executed += 1;
            Ok(ActionOutcome {
                action_id: "advance".to_owned(),
                completed: true,
                observations: Vec::new(),
                feedback: DataValue::Null,
            })
        }
    }

    fn trained_model() -> EnsembleWorldModel {
        let mut learner = ActionDeltaModel::new("counter-delta", 1).expect("model");
        learner
            .fit(
                "training",
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
                            DataValue::Number(3.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(4.0))]),
                    },
                ],
            )
            .expect("fit");
        EnsembleWorldModel::new(vec![Box::new(learner) as Box<dyn PredictiveModel>])
            .expect("ensemble")
    }

    fn config(execution_id: &str, maximum_steps: usize) -> RuntimeConfig {
        RuntimeConfig {
            execution_id: execution_id.to_owned(),
            episode_id: "episode-1".to_owned(),
            maximum_steps,
            authorizer: ActionAuthorizer::new(ActionAuthorizationPolicy::new(
                ActionRisk::Reversible,
            )),
            planning: PlanningConfig {
                horizon: 2,
                beam_width: 2,
                uncertainty_penalty: 0.0,
                uncalibrated_prediction_penalty: 0.0,
            },
        }
    }

    fn actions() -> [ActionProposal; 1] {
        [ActionProposal {
            action_id: "advance".to_owned(),
            arguments: DataValue::object(std::iter::empty()),
        }]
    }

    #[test]
    fn runtime_replans_and_measures_predictions() {
        let path = test_path();
        let mut journal = ExperienceJournal::open(&path).expect("journal");
        let mut environment = CounterEnvironment::new();
        let report = run_receding_horizon(
            &mut environment,
            &trained_model(),
            &actions(),
            config("run-1", 2),
            &mut journal,
            |state| {
                state
                    .get("position")
                    .and_then(DataValue::as_f64)
                    .unwrap_or(0.0)
            },
        )
        .expect("runtime");
        assert_eq!(report.steps.len(), 2);
        assert_eq!(environment.executed, 2);
        assert_eq!(
            report
                .final_observation
                .payload
                .get("position")
                .and_then(DataValue::as_f64),
            Some(2.0)
        );
        assert!(report.steps.iter().all(|step| {
            step.prediction_error.mean_absolute_error == Some(0.0)
                && step.prediction_error.root_mean_squared_error == Some(0.0)
        }));
        assert_eq!(journal.records().len(), 9);

        let retry = run_receding_horizon(
            &mut environment,
            &trained_model(),
            &actions(),
            config("run-1", 1),
            &mut journal,
            |_| 1.0,
        );
        assert!(matches!(retry, Err(RuntimeError::ExecutionAlreadyStarted)));
        assert_eq!(environment.executed, 2);
        drop(journal);
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn retry_is_blocked_before_observation_or_execution() {
        let path = test_path();
        let mut journal = ExperienceJournal::open(&path).expect("journal");
        let mut environment = CounterEnvironment::new();
        let world = trained_model();
        let cfg = config("blocked", 1);
        run_receding_horizon(
            &mut environment,
            &world,
            &actions(),
            cfg.clone(),
            &mut journal,
            |_| 1.0,
        )
        .expect("first run");
        let sequence_before_retry = environment.sequence;
        let retry = run_receding_horizon(
            &mut environment,
            &world,
            &actions(),
            cfg,
            &mut journal,
            |_| 1.0,
        );
        assert!(matches!(retry, Err(RuntimeError::ExecutionAlreadyStarted)));
        assert_eq!(environment.sequence, sequence_before_retry);
        assert_eq!(environment.executed, 1);
        drop(journal);
        std::fs::remove_file(path).expect("cleanup");
    }
    #[test]
    fn prediction_error_uses_stable_accumulation_for_large_finite_values() {
        let predicted = DataValue::object([
            ("small".to_owned(), DataValue::Number(1.0e200)),
            ("large".to_owned(), DataValue::Number(2.0e200)),
        ]);
        let observed = DataValue::object([
            ("small".to_owned(), DataValue::Number(0.0)),
            ("large".to_owned(), DataValue::Number(0.0)),
        ]);

        let metrics = prediction_error(&predicted, &observed);
        assert_eq!(metrics.numeric_fields, 2);

        let mean_absolute_error = metrics.mean_absolute_error.expect("finite MAE");
        assert!((mean_absolute_error / 1.0e200 - 1.5).abs() < 1.0e-12);

        let root_mean_squared_error = metrics.root_mean_squared_error.expect("finite RMSE");
        assert!((root_mean_squared_error / 1.0e200 - 2.5_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn prediction_error_omits_metrics_when_a_difference_exceeds_f64_range() {
        let predicted = DataValue::object([("position".to_owned(), DataValue::Number(f64::MAX))]);
        let observed = DataValue::object([("position".to_owned(), DataValue::Number(-f64::MAX))]);

        let metrics = prediction_error(&predicted, &observed);
        assert_eq!(metrics.numeric_fields, 1);
        assert_eq!(metrics.mean_absolute_error, None);
        assert_eq!(metrics.root_mean_squared_error, None);
    }
}
