//! Integrated hypothesis experiment and predictive-model improvement cycle.

use crate::autonomous::{run_learning_cycle, LearningCycleConfig, LearningCycleError, LearningCycleReport};
use crate::environment::EnvironmentAdapter;
use crate::experiment_execution::{
    execute_selected_experiment, ExperimentActionMapping, ExperimentExecutionConfig,
    ExperimentExecutionError, ExperimentExecutionReport,
};
use crate::experiment_selection::{ExperimentCandidate, ExperimentSelectionPolicy};
use crate::experience_journal::ExperienceJournal;
use crate::learning::TransitionSample;
use crate::model_artifact::{ModelArtifactError, ModelArtifactStore};
use crate::model_registry::{ModelEvaluation, ModelPromotionPolicy, ModelRegistry, ModelRegistryError};
use crate::reasoning::{HypothesisPolicy, HypothesisRegistry};
use crate::value::DataValue;
use crate::world_model::{EnsembleWorldModel, PredictiveModel, TrainablePredictiveModel};

#[derive(Clone, Debug)]
pub struct AutonomousResearchConfig {
    pub experiment: ExperimentExecutionConfig,
    pub learning: LearningCycleConfig,
}

impl AutonomousResearchConfig {
    pub fn validate(&self) -> Result<(), AutonomousResearchError> {
        if self.experiment.runtime.episode_id != self.learning.episode_id
            || self.experiment.runtime.maximum_steps != 1
            || self.experiment.hypothesis_registry_artifact_ref == self.learning.model_registry_artifact_ref
            || self.experiment.hypothesis_registry_artifact_ref == self.learning.model_artifact_ref
            || self.learning.model_registry_artifact_ref == self.learning.model_artifact_ref
        {
            return Err(AutonomousResearchError::InvalidConfiguration);
        }
        self.learning
            .validate()
            .map_err(|_| AutonomousResearchError::InvalidConfiguration)
    }
}

#[derive(Debug)]
pub struct AutonomousResearchReport {
    pub experiment: ExperimentExecutionReport,
    /// Kept separate so a successfully recorded experiment survives a rejected
    /// or failed candidate training/promotion stage.
    pub learning: Result<LearningCycleReport, LearningCycleError>,
}

/// Select one hypothesis-discriminating experiment, execute one safe action,
/// persist the resulting evidence, then train/evaluate/promote a model from the
/// recorded episode using a separate caller-supplied holdout.
///
/// All external capabilities remain explicit: the action mapping, outcome
/// classifier, validation data, and metric evaluator are supplied by the caller.
pub fn run_autonomous_research_cycle<A, T, U, O, E>(
    adapter: &mut A,
    active_model: &EnsembleWorldModel,
    candidate_model: &mut T,
    hypothesis_registry: &mut HypothesisRegistry,
    hypothesis_id: &str,
    experiments: &[ExperimentCandidate],
    action_mappings: &[ExperimentActionMapping],
    validation_samples: &[TransitionSample],
    selection_policy: ExperimentSelectionPolicy,
    hypothesis_policy: HypothesisPolicy,
    promotion_policy: ModelPromotionPolicy,
    config: AutonomousResearchConfig,
    journal: &mut ExperienceJournal,
    artifact_store: &mut ModelArtifactStore,
    model_registry: &mut ModelRegistry,
    utility: U,
    positive_outcome: O,
    evaluate: E,
) -> Result<AutonomousResearchReport, AutonomousResearchError>
where
    A: EnvironmentAdapter,
    T: TrainablePredictiveModel,
    U: Fn(&DataValue) -> f64,
    O: Fn(&DataValue) -> bool,
    E: FnOnce(&T, &[TransitionSample]) -> Result<ModelEvaluation, String>,
{
    config.validate()?;
    promotion_policy.validate().map_err(AutonomousResearchError::ModelRegistry)?;
    hypothesis_policy.validate().map_err(AutonomousResearchError::Experiment)?;
    if validation_samples.is_empty() {
        return Err(AutonomousResearchError::EmptyValidationSet);
    }

    let descriptor = adapter.descriptor();
    if descriptor.id != config.learning.environment_id
        || descriptor.version != config.learning.environment_version
    {
        return Err(AutonomousResearchError::EnvironmentMismatch);
    }
    let identity = candidate_model.identity();
    if identity.id != config.learning.model_id || identity.version != config.learning.model_version {
        return Err(AutonomousResearchError::ModelIdentityMismatch);
    }
    if model_registry.iter().any(|record| {
        record.spec.id == config.learning.model_id
            && record.spec.version >= config.learning.model_version
    }) {
        return Err(AutonomousResearchError::ModelVersionMustIncrease);
    }
    if artifact_store.get(&config.learning.model_artifact_ref).is_some()
        || artifact_store.get(&config.learning.model_registry_artifact_ref).is_some()
        || artifact_store.get(&config.experiment.hypothesis_registry_artifact_ref).is_some()
    {
        return Err(AutonomousResearchError::ArtifactReferenceAlreadyExists);
    }

    let experiment = execute_selected_experiment(
        adapter,
        active_model,
        hypothesis_registry,
        hypothesis_id,
        experiments,
        action_mappings,
        selection_policy,
        hypothesis_policy,
        config.experiment,
        journal,
        artifact_store,
        &utility,
        positive_outcome,
    )
    .map_err(AutonomousResearchError::Experiment)?;

    let learning = run_learning_cycle(
        candidate_model,
        journal.records(),
        validation_samples,
        config.learning,
        promotion_policy,
        artifact_store,
        model_registry,
        evaluate,
    );

    Ok(AutonomousResearchReport { experiment, learning })
}

#[derive(Debug)]
pub enum AutonomousResearchError {
    InvalidConfiguration,
    EmptyValidationSet,
    EnvironmentMismatch,
    ModelIdentityMismatch,
    ModelVersionMustIncrease,
    ArtifactReferenceAlreadyExists,
    Experiment(ExperimentExecutionError),
    ModelRegistry(ModelRegistryError),
    Artifact(ModelArtifactError),
}

impl std::fmt::Display for AutonomousResearchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration => write!(formatter, "autonomous research configuration is invalid"),
            Self::EmptyValidationSet => write!(formatter, "autonomous research cycle requires held-out validation transitions"),
            Self::EnvironmentMismatch => write!(formatter, "candidate model environment contract does not match the adapter"),
            Self::ModelIdentityMismatch => write!(formatter, "candidate predictive model identity does not match the learning-cycle configuration"),
            Self::ModelVersionMustIncrease => write!(formatter, "candidate model version must be newer than every registered version"),
            Self::ArtifactReferenceAlreadyExists => write!(formatter, "research cycle requires fresh immutable artifact references"),
            Self::Experiment(error) => write!(formatter, "hypothesis experiment failed: {error}"),
            Self::ModelRegistry(error) => write!(formatter, "model registry policy is invalid: {error}"),
            Self::Artifact(error) => write!(formatter, "model artifact operation failed: {error}"),
        }
    }
}

impl std::error::Error for AutonomousResearchError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{
        ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionProposal, ActionRisk,
    };
    use crate::benchmark::ScoreDirection;
    use crate::environment::{
        ActionOutcome, EnvironmentDescriptor, EnvironmentError, ObservationFrame,
    };
    use crate::experiment_execution::ExperimentExecutionConfig;
    use crate::model_registry::ModelStatus;
    use crate::planning::PlanningConfig;
    use crate::reasoning::{HypothesisSpec, HypothesisStatus};
    use crate::schema::{FieldSpec, SchemaDefinition, ValueKind};
    use crate::world_model::ActionDeltaModel;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn path(kind: &str) -> std::path::PathBuf {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("5413-research-{}-{kind}-{id}.bin", std::process::id()))
    }

    struct Counter {
        descriptor: EnvironmentDescriptor,
        position: f64,
        sequence: u64,
    }

    impl Counter {
        fn new() -> Self {
            Self {
                descriptor: EnvironmentDescriptor {
                    id: "research-counter".to_owned(),
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
                    supports_reset: false,
                },
                position: 0.0,
                sequence: 0,
            }
        }

        fn frame(&mut self) -> ObservationFrame {
            let frame = ObservationFrame {
                environment_id: "research-counter".to_owned(),
                session_id: "research-session".to_owned(),
                episode_id: "research-episode".to_owned(),
                sequence: self.sequence,
                observed_at_millis: self.sequence,
                schema_id: "state".to_owned(),
                schema_version: 1,
                payload: DataValue::object([("position".to_owned(), DataValue::Number(self.position))]),
            };
            self.sequence += 1;
            frame
        }
    }

    impl EnvironmentAdapter for Counter {
        fn descriptor(&self) -> &EnvironmentDescriptor { &self.descriptor }

        fn observe(&mut self) -> Result<ObservationFrame, EnvironmentError> {
            Ok(self.frame())
        }

        fn execute(&mut self, _action: crate::action::AuthorizedAction) -> Result<ActionOutcome, EnvironmentError> {
            self.position += 1.0;
            Ok(ActionOutcome {
                action_id: "advance".to_owned(),
                completed: true,
                observations: Vec::new(),
                feedback: DataValue::Null,
            })
        }
    }

    fn active_model() -> EnsembleWorldModel {
        let mut model = ActionDeltaModel::new("research-active", 1).expect("model");
        model.fit("initial-training", &[
            TransitionSample {
                action_id: "advance".to_owned(),
                before: DataValue::object([("position".to_owned(), DataValue::Number(0.0))]),
                after: DataValue::object([("position".to_owned(), DataValue::Number(1.0))]),
            },
            TransitionSample {
                action_id: "advance".to_owned(),
                before: DataValue::object([("position".to_owned(), DataValue::Number(3.0))]),
                after: DataValue::object([("position".to_owned(), DataValue::Number(4.0))]),
            },
        ]).expect("fit");
        EnsembleWorldModel::new(vec![Box::new(model) as Box<dyn PredictiveModel>]).expect("ensemble")
    }

    #[test]
    fn experiment_updates_hypothesis_then_trains_and_registers_new_model() {
        let journal_path = path("journal");
        let artifact_path = path("artifacts");
        let mut journal = ExperienceJournal::open(&journal_path).expect("journal");
        let mut store = ModelArtifactStore::open(&artifact_path).expect("artifacts");
        let mut environment = Counter::new();
        let mut hypotheses = HypothesisRegistry::default();
        let hypothesis_policy = HypothesisPolicy {
            minimum_evidence_count: 1,
            support_probability: 0.8,
            refutation_probability: 0.2,
        };
        hypotheses.insert(HypothesisSpec {
            id: "advance-changes-position".to_owned(),
            claim: DataValue::Text("advance increases position".to_owned()),
            description: "one advance increases the observed position".to_owned(),
            prior_probability: 0.5,
            source_dataset_id: None,
        }, hypothesis_policy).expect("hypothesis");
        let candidate_experiment = ExperimentCandidate {
            id: "advance-once".to_owned(),
            specification: DataValue::Text("execute one reversible advance action".to_owned()),
            observation_probability_if_true: 0.9,
            observation_probability_if_false: 0.1,
            resource_cost: 1.0,
            risk: 0.01,
            irreversible: false,
        };
        let mut candidate_model = ActionDeltaModel::new("research-candidate", 1).expect("candidate");
        let mut model_registry = ModelRegistry::default();
        let report = run_autonomous_research_cycle(
            &mut environment,
            &active_model(),
            &mut candidate_model,
            &mut hypotheses,
            "advance-changes-position",
            &[candidate_experiment],
            &[ExperimentActionMapping {
                experiment_id: "advance-once".to_owned(),
                action: ActionProposal {
                    action_id: "advance".to_owned(),
                    arguments: DataValue::object(std::iter::empty()),
                },
            }],
            &[
                TransitionSample {
                    action_id: "advance".to_owned(),
                    before: DataValue::object([("position".to_owned(), DataValue::Number(20.0))]),
                    after: DataValue::object([("position".to_owned(), DataValue::Number(21.0))]),
                },
                TransitionSample {
                    action_id: "advance".to_owned(),
                    before: DataValue::object([("position".to_owned(), DataValue::Number(30.0))]),
                    after: DataValue::object([("position".to_owned(), DataValue::Number(31.0))]),
                },
            ],
            ExperimentSelectionPolicy {
                maximum_resource_cost: 5.0,
                maximum_risk: 0.1,
                resource_cost_weight: 0.0,
                risk_weight: 0.0,
                allow_irreversible: false,
            },
            hypothesis_policy,
            ModelPromotionPolicy {
                minimum_evaluation_samples: 2,
                minimum_improvement: 0.0,
                resource_cost_weight: 0.0,
                require_calibrated_uncertainty: false,
            },
            AutonomousResearchConfig {
                experiment: ExperimentExecutionConfig {
                    runtime: crate::runtime::RuntimeConfig {
                        execution_id: "research-run-1".to_owned(),
                        episode_id: "research-episode".to_owned(),
                        maximum_steps: 1,
                        authorizer: ActionAuthorizer::new(ActionAuthorizationPolicy::new(ActionRisk::Reversible)),
                        planning: PlanningConfig {
                            horizon: 1,
                            beam_width: 1,
                            uncertainty_penalty: 0.0,
                            uncalibrated_prediction_penalty: 0.0,
                        },
                    },
                    evidence_dataset_id: "research-evidence-1".to_owned(),
                    evidence_rationale: "observed final position after intervention".to_owned(),
                    hypothesis_registry_artifact_ref: "research-hypotheses-v1".to_owned(),
                },
                learning: LearningCycleConfig {
                    episode_id: "research-episode".to_owned(),
                    training_dataset_id: "research-episode-training".to_owned(),
                    validation_dataset_id: "research-holdout-v1".to_owned(),
                    model_id: "research-candidate".to_owned(),
                    model_version: 1,
                    model_artifact_ref: "research-model-v1".to_owned(),
                    model_registry_artifact_ref: "research-model-registry-v1".to_owned(),
                    environment_id: "research-counter".to_owned(),
                    environment_version: "v1".to_owned(),
                },
            },
            &mut journal,
            &mut store,
            &mut model_registry,
            |state| state.get("position").and_then(DataValue::as_f64).unwrap_or(0.0),
            |state| state.get("position").and_then(DataValue::as_f64).unwrap_or(0.0) > 0.0,
            |_model, samples| Ok(ModelEvaluation {
                evaluation_dataset_id: "research-holdout-v1".to_owned(),
                metric_id: "rmse".to_owned(),
                direction: ScoreDirection::LowerIsBetter,
                sample_count: samples.len(),
                score: 0.0,
                resource_cost: 1.0,
                uncertainty_calibrated: true,
            }),
        ).expect("research cycle");

        assert_eq!(report.experiment.updated_hypothesis.status, HypothesisStatus::Supported);
        assert!(report.learning.expect("learning").promoted);
        assert_eq!(model_registry.active("research-candidate").map(|record| record.status), Some(ModelStatus::Active));
        drop(store);
        drop(journal);
        std::fs::remove_file(journal_path).expect("journal cleanup");
        std::fs::remove_file(artifact_path).expect("artifact cleanup");
    }
}
