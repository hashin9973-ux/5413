//! Integrated hypothesis experimentation and predictive-model improvement cycle.
//!
//! External action mappings, validation samples, likelihoods, outcome classifiers,
//! and metric evaluation remain supplied by the environment integration.

use crate::action::ActionProposal;
use crate::environment::EnvironmentAdapter;
use crate::experience_journal::ExperienceJournal;
use crate::experiment_selection::{
    rank_experiments, ExperimentCandidate, ExperimentSelectionError, ExperimentSelectionPolicy,
    RankedExperiment,
};
use crate::learning::{extract_transition_samples, TransitionSample};
use crate::model_artifact::{ModelArtifactError, ModelArtifactStore};
use crate::model_registry::{
    ModelCandidateSpec, ModelEvaluation, ModelPromotionPolicy, ModelRecord, ModelRegistry,
    ModelRegistryError, ModelStatus,
};
use crate::reasoning::{
    EvidenceRecord, HypothesisPolicy, HypothesisRecord, HypothesisRegistry, ReasoningError,
};
use crate::runtime::{run_receding_horizon, RuntimeConfig, RuntimeError, RuntimeReport};
use crate::value::DataValue;
use crate::world_model::{
    EnsembleWorldModel, PredictiveModel, TrainablePredictiveModel, WorldModelError,
};
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct ExperimentActionMapping {
    pub experiment_id: String,
    pub action: ActionProposal,
}

#[derive(Clone, Debug)]
pub struct ExperimentExecutionConfig {
    pub runtime: RuntimeConfig,
    pub evidence_dataset_id: String,
    pub evidence_rationale: String,
    pub hypothesis_registry_artifact_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LearningCycleConfig {
    pub episode_id: String,
    pub training_dataset_id: String,
    pub validation_dataset_id: String,
    pub model_id: String,
    pub model_version: u64,
    pub model_artifact_ref: String,
    pub model_registry_artifact_ref: String,
    pub environment_id: String,
    pub environment_version: String,
}

impl LearningCycleConfig {
    fn validate(&self) -> Result<(), AutonomousResearchError> {
        if self.episode_id.trim().is_empty()
            || self.training_dataset_id.trim().is_empty()
            || self.validation_dataset_id.trim().is_empty()
            || self.training_dataset_id == self.validation_dataset_id
            || self.model_id.trim().is_empty()
            || self.model_version == 0
            || self.model_artifact_ref.trim().is_empty()
            || self.model_registry_artifact_ref.trim().is_empty()
            || self.environment_id.trim().is_empty()
            || self.environment_version.trim().is_empty()
        {
            return Err(AutonomousResearchError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AutonomousResearchConfig {
    pub experiment: ExperimentExecutionConfig,
    pub learning: LearningCycleConfig,
}

impl AutonomousResearchConfig {
    pub fn validate(&self) -> Result<(), AutonomousResearchError> {
        self.learning.validate()?;
        if self.experiment.runtime.episode_id != self.learning.episode_id
            || self.experiment.runtime.maximum_steps != 1
            || self.experiment.evidence_dataset_id.trim().is_empty()
            || self.experiment.evidence_rationale.trim().is_empty()
            || self
                .experiment
                .hypothesis_registry_artifact_ref
                .trim()
                .is_empty()
            || self.experiment.hypothesis_registry_artifact_ref == self.learning.model_artifact_ref
            || self.experiment.hypothesis_registry_artifact_ref
                == self.learning.model_registry_artifact_ref
            || self.learning.model_artifact_ref == self.learning.model_registry_artifact_ref
            || self.experiment.evidence_dataset_id == self.learning.training_dataset_id
            || self.experiment.evidence_dataset_id == self.learning.validation_dataset_id
        {
            return Err(AutonomousResearchError::InvalidConfiguration);
        }
        self.experiment
            .runtime
            .validate()
            .map_err(AutonomousResearchError::Runtime)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LearningCycleReport {
    pub model_id: String,
    pub model_version: u64,
    pub training_samples: usize,
    pub validation_samples: usize,
    pub evaluation: ModelEvaluation,
    pub status: ModelStatus,
    pub promoted: bool,
}

#[derive(Debug)]
pub struct AutonomousResearchReport {
    pub selected: RankedExperiment,
    pub ranked_candidates: Vec<RankedExperiment>,
    pub runtime: RuntimeReport,
    pub evidence: EvidenceRecord,
    pub updated_hypothesis: HypothesisRecord,
    /// Experiment results remain available even when candidate training or
    /// promotion is rejected after the environment effect has occurred.
    pub learning: Result<LearningCycleReport, LearningCycleError>,
}

/// Runs one bounded experiment and then attempts one held-out model improvement.
///
/// The function preflights contracts and immutable references before executing
/// an environment action. The hypothesis update is persisted immediately after
/// the experiment. Model learning is a separate result in the returned report.
pub struct AutonomousResearchContext<'a, A, T> {
    pub adapter: &'a mut A,
    pub active_model: &'a EnsembleWorldModel,
    pub candidate_model: &'a mut T,
    pub hypothesis_registry: &'a mut HypothesisRegistry,
    pub journal: &'a mut ExperienceJournal,
    pub artifact_store: &'a mut ModelArtifactStore,
    pub model_registry: &'a mut ModelRegistry,
}

pub struct AutonomousResearchRequest<'a> {
    pub hypothesis_id: &'a str,
    pub experiments: &'a [ExperimentCandidate],
    pub action_mappings: &'a [ExperimentActionMapping],
    pub validation_samples: &'a [TransitionSample],
    pub selection_policy: ExperimentSelectionPolicy,
    pub hypothesis_policy: HypothesisPolicy,
    pub promotion_policy: ModelPromotionPolicy,
    pub config: AutonomousResearchConfig,
}

pub struct AutonomousResearchCallbacks<U, O, E> {
    pub utility: U,
    pub positive_outcome: O,
    pub evaluate: E,
}

pub fn run_autonomous_research_cycle<A, T, U, O, E>(
    context: AutonomousResearchContext<'_, A, T>,
    request: AutonomousResearchRequest<'_>,
    callbacks: AutonomousResearchCallbacks<U, O, E>,
) -> Result<AutonomousResearchReport, AutonomousResearchError>
where
    A: EnvironmentAdapter,
    T: TrainablePredictiveModel,
    U: Fn(&DataValue) -> f64,
    O: Fn(&DataValue) -> bool,
    E: FnOnce(&T, &[TransitionSample]) -> Result<ModelEvaluation, String>,
{
    let AutonomousResearchContext {
        adapter,
        active_model,
        candidate_model,
        hypothesis_registry,
        journal,
        artifact_store,
        model_registry,
    } = context;
    let AutonomousResearchRequest {
        hypothesis_id,
        experiments,
        action_mappings,
        validation_samples,
        selection_policy,
        hypothesis_policy,
        promotion_policy,
        config,
    } = request;
    let AutonomousResearchCallbacks {
        utility,
        positive_outcome,
        evaluate,
    } = callbacks;
    config.validate()?;
    selection_policy
        .validate()
        .map_err(AutonomousResearchError::Selection)?;
    hypothesis_policy
        .validate()
        .map_err(AutonomousResearchError::Reasoning)?;
    promotion_policy
        .validate()
        .map_err(AutonomousResearchError::ModelRegistry)?;
    if validation_samples.is_empty() {
        return Err(AutonomousResearchError::EmptyValidationSet);
    }

    let descriptor = adapter.descriptor();
    descriptor
        .validate()
        .map_err(|error| AutonomousResearchError::Environment(error.to_string()))?;
    if descriptor.id != config.learning.environment_id
        || descriptor.version != config.learning.environment_version
    {
        return Err(AutonomousResearchError::EnvironmentMismatch);
    }

    let identity = candidate_model.identity();
    if identity.id != config.learning.model_id || identity.version != config.learning.model_version
    {
        return Err(AutonomousResearchError::ModelIdentityMismatch);
    }
    if model_registry.iter().any(|record| {
        record.spec.id == config.learning.model_id
            && record.spec.version >= config.learning.model_version
    }) {
        return Err(AutonomousResearchError::ModelVersionMustIncrease);
    }

    for reference in [
        config.experiment.hypothesis_registry_artifact_ref.as_str(),
        config.learning.model_artifact_ref.as_str(),
        config.learning.model_registry_artifact_ref.as_str(),
    ] {
        if artifact_store.get(reference).is_some() {
            return Err(AutonomousResearchError::ArtifactReferenceAlreadyExists);
        }
    }

    let hypothesis = hypothesis_registry
        .get(hypothesis_id)
        .cloned()
        .ok_or(AutonomousResearchError::HypothesisNotFound)?;
    if hypothesis
        .evidence
        .iter()
        .any(|item| item.evidence_ref == config.experiment.runtime.execution_id)
    {
        return Err(AutonomousResearchError::EvidenceAlreadyRecorded);
    }
    if hypothesis.spec.source_dataset_id.as_deref()
        == Some(config.experiment.evidence_dataset_id.as_str())
        || hypothesis
            .evidence
            .iter()
            .any(|item| item.dataset_id == config.experiment.evidence_dataset_id)
    {
        return Err(AutonomousResearchError::EvidenceDatasetReused);
    }

    let ranked_candidates = rank_experiments(&hypothesis, experiments, selection_policy)
        .map_err(AutonomousResearchError::Selection)?;
    let selected = ranked_candidates
        .iter()
        .find(|candidate| candidate.is_worthwhile())
        .cloned()
        .ok_or(AutonomousResearchError::NoWorthwhileExperiment)?;
    let experiment = experiments
        .iter()
        .find(|candidate| candidate.id == selected.experiment_id)
        .ok_or(AutonomousResearchError::InvalidActionMapping)?;
    if experiment.observation_probability_if_true <= 0.0
        || experiment.observation_probability_if_true >= 1.0
        || experiment.observation_probability_if_false <= 0.0
        || experiment.observation_probability_if_false >= 1.0
    {
        return Err(AutonomousResearchError::InvalidLikelihoodModel);
    }

    let mut mapping_ids = BTreeSet::new();
    for mapping in action_mappings {
        if mapping.experiment_id.trim().is_empty()
            || !mapping_ids.insert(mapping.experiment_id.as_str())
        {
            return Err(AutonomousResearchError::InvalidActionMapping);
        }
    }
    let mapping = action_mappings
        .iter()
        .find(|mapping| mapping.experiment_id == selected.experiment_id)
        .ok_or(AutonomousResearchError::MissingActionMapping)?;

    let runtime = run_receding_horizon(
        adapter,
        active_model,
        std::slice::from_ref(&mapping.action),
        config.experiment.runtime.clone(),
        journal,
        utility,
    )
    .map_err(AutonomousResearchError::Runtime)?;
    let last_step = runtime
        .steps
        .last()
        .ok_or(AutonomousResearchError::NoExecutedAction)?;
    if !last_step.outcome.completed {
        return Err(AutonomousResearchError::ExperimentIncomplete);
    }

    let positive = positive_outcome(&runtime.final_observation.payload);
    let probability_if_true = if positive {
        experiment.observation_probability_if_true
    } else {
        1.0 - experiment.observation_probability_if_true
    };
    let probability_if_false = if positive {
        experiment.observation_probability_if_false
    } else {
        1.0 - experiment.observation_probability_if_false
    };
    let likelihood_ratio = probability_if_true / probability_if_false;
    let log_likelihood_ratio = likelihood_ratio.ln();
    if !likelihood_ratio.is_finite() || !log_likelihood_ratio.is_finite() {
        return Err(AutonomousResearchError::NumericalOverflow);
    }

    let evidence = EvidenceRecord {
        evidence_ref: config.experiment.runtime.execution_id.clone(),
        dataset_id: config.experiment.evidence_dataset_id.clone(),
        log_likelihood_ratio,
        rationale: config.experiment.evidence_rationale.clone(),
    };
    let updated_hypothesis = hypothesis_registry
        .add_evidence(hypothesis_id, evidence.clone(), hypothesis_policy)
        .map_err(AutonomousResearchError::Reasoning)?
        .clone();
    artifact_store
        .put_hypothesis_registry(
            config.experiment.hypothesis_registry_artifact_ref,
            hypothesis_registry,
        )
        .map_err(AutonomousResearchError::Artifact)?;

    let learning = run_learning_cycle(
        candidate_model,
        LearningCycleResources {
            records: journal.records(),
            validation_samples,
            config: &config.learning,
            policy: promotion_policy,
            artifact_store,
            registry: model_registry,
        },
        evaluate,
    );

    Ok(AutonomousResearchReport {
        selected,
        ranked_candidates,
        runtime,
        evidence,
        updated_hypothesis,
        learning,
    })
}

struct LearningCycleResources<'a> {
    records: &'a [crate::experience::ExperienceRecord],
    validation_samples: &'a [TransitionSample],
    config: &'a LearningCycleConfig,
    policy: ModelPromotionPolicy,
    artifact_store: &'a mut ModelArtifactStore,
    registry: &'a mut ModelRegistry,
}

fn run_learning_cycle<T, E>(
    model: &mut T,
    resources: LearningCycleResources<'_>,
    evaluate: E,
) -> Result<LearningCycleReport, LearningCycleError>
where
    T: TrainablePredictiveModel,
    E: FnOnce(&T, &[TransitionSample]) -> Result<ModelEvaluation, String>,
{
    let LearningCycleResources {
        records,
        validation_samples,
        config,
        policy,
        artifact_store,
        registry,
    } = resources;
    policy.validate().map_err(LearningCycleError::Registry)?;
    if config.training_dataset_id == config.validation_dataset_id {
        return Err(LearningCycleError::DatasetLeakage);
    }
    if validation_samples.is_empty() {
        return Err(LearningCycleError::EmptyValidationSet);
    }
    let training_samples = extract_transition_samples(records, &config.episode_id);
    if training_samples.is_empty() {
        return Err(LearningCycleError::WorldModel(
            WorldModelError::NoCompletedTransitions,
        ));
    }
    if training_samples.len() > u32::MAX as usize || validation_samples.len() > u32::MAX as usize {
        return Err(LearningCycleError::DatasetTooLarge);
    }

    model
        .fit_samples(&config.training_dataset_id, &training_samples)
        .map_err(LearningCycleError::WorldModel)?;
    let evaluation =
        evaluate(model, validation_samples).map_err(LearningCycleError::EvaluationFailed)?;
    if evaluation.evaluation_dataset_id != config.validation_dataset_id
        || evaluation.sample_count != validation_samples.len()
    {
        return Err(LearningCycleError::EvaluationProvenanceMismatch);
    }

    artifact_store
        .put_model(&config.model_artifact_ref, &*model as &dyn PredictiveModel)
        .map_err(LearningCycleError::Artifact)?;
    let spec = ModelCandidateSpec {
        id: config.model_id.clone(),
        version: config.model_version,
        model_artifact_ref: config.model_artifact_ref.clone(),
        environment_id: config.environment_id.clone(),
        environment_version: config.environment_version.clone(),
        training_dataset_id: config.training_dataset_id.clone(),
    };
    artifact_store
        .register_model_candidate(registry, spec)
        .map_err(LearningCycleError::Artifact)?;

    if let Err(error) = registry.evaluate_candidate(
        &config.model_id,
        config.model_version,
        evaluation.clone(),
        policy,
    ) {
        persist_model_registry(
            artifact_store,
            registry,
            &config.model_registry_artifact_ref,
        )?;
        if error != ModelRegistryError::InsufficientEvaluationSamples {
            return Err(LearningCycleError::Registry(error));
        }
        return learning_report(
            registry,
            &config.model_id,
            config.model_version,
            training_samples.len(),
            validation_samples.len(),
            evaluation,
            false,
        );
    }

    let promoted = match registry.promote(&config.model_id, config.model_version, policy) {
        Ok(()) => true,
        Err(
            ModelRegistryError::InsufficientImprovement
            | ModelRegistryError::UncalibratedUncertainty,
        ) => false,
        Err(error) => {
            persist_model_registry(
                artifact_store,
                registry,
                &config.model_registry_artifact_ref,
            )?;
            return Err(LearningCycleError::Registry(error));
        }
    };
    persist_model_registry(
        artifact_store,
        registry,
        &config.model_registry_artifact_ref,
    )?;
    learning_report(
        registry,
        &config.model_id,
        config.model_version,
        training_samples.len(),
        validation_samples.len(),
        evaluation,
        promoted,
    )
}

fn persist_model_registry(
    artifact_store: &mut ModelArtifactStore,
    registry: &ModelRegistry,
    artifact_ref: &str,
) -> Result<(), LearningCycleError> {
    artifact_store
        .put_model_registry(artifact_ref, registry)
        .map_err(LearningCycleError::Artifact)?;
    Ok(())
}

fn learning_report(
    registry: &ModelRegistry,
    model_id: &str,
    model_version: u64,
    training_samples: usize,
    validation_samples: usize,
    evaluation: ModelEvaluation,
    promoted: bool,
) -> Result<LearningCycleReport, LearningCycleError> {
    let record: &ModelRecord =
        registry
            .get(model_id, model_version)
            .ok_or(LearningCycleError::Registry(
                ModelRegistryError::ModelNotFound,
            ))?;
    Ok(LearningCycleReport {
        model_id: model_id.to_owned(),
        model_version,
        training_samples,
        validation_samples,
        evaluation,
        status: record.status,
        promoted,
    })
}

#[derive(Debug)]
pub enum LearningCycleError {
    EmptyValidationSet,
    DatasetLeakage,
    DatasetTooLarge,
    EvaluationFailed(String),
    EvaluationProvenanceMismatch,
    WorldModel(WorldModelError),
    Registry(ModelRegistryError),
    Artifact(ModelArtifactError),
}

impl std::fmt::Display for LearningCycleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyValidationSet => write!(
                formatter,
                "learning cycle requires held-out validation transitions"
            ),
            Self::DatasetLeakage => {
                write!(formatter, "training and validation dataset IDs must differ")
            }
            Self::DatasetTooLarge => write!(
                formatter,
                "learning-cycle dataset exceeds supported sample counts"
            ),
            Self::EvaluationFailed(message) => {
                write!(formatter, "candidate evaluation failed: {message}")
            }
            Self::EvaluationProvenanceMismatch => write!(
                formatter,
                "evaluator dataset ID or sample count does not match the supplied holdout"
            ),
            Self::WorldModel(error) => write!(formatter, "candidate fitting failed: {error}"),
            Self::Registry(error) => write!(formatter, "model lifecycle operation failed: {error}"),
            Self::Artifact(error) => write!(formatter, "model artifact operation failed: {error}"),
        }
    }
}
impl std::error::Error for LearningCycleError {}

#[derive(Debug)]
pub enum AutonomousResearchError {
    InvalidConfiguration,
    EmptyValidationSet,
    Environment(String),
    EnvironmentMismatch,
    ModelIdentityMismatch,
    ModelVersionMustIncrease,
    ArtifactReferenceAlreadyExists,
    HypothesisNotFound,
    EvidenceAlreadyRecorded,
    EvidenceDatasetReused,
    NoWorthwhileExperiment,
    InvalidLikelihoodModel,
    InvalidActionMapping,
    MissingActionMapping,
    NoExecutedAction,
    ExperimentIncomplete,
    NumericalOverflow,
    Selection(ExperimentSelectionError),
    Reasoning(ReasoningError),
    Runtime(RuntimeError),
    ModelRegistry(ModelRegistryError),
    Artifact(ModelArtifactError),
}

impl std::fmt::Display for AutonomousResearchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration => write!(formatter, "autonomous research configuration is invalid"),
            Self::EmptyValidationSet => write!(formatter, "autonomous research requires held-out validation transitions"),
            Self::Environment(message) => write!(formatter, "environment descriptor is invalid: {message}"),
            Self::EnvironmentMismatch => write!(formatter, "candidate model environment contract does not match the adapter"),
            Self::ModelIdentityMismatch => write!(formatter, "candidate predictive model identity does not match the learning-cycle configuration"),
            Self::ModelVersionMustIncrease => write!(formatter, "candidate model version must exceed every registered version"),
            Self::ArtifactReferenceAlreadyExists => write!(formatter, "research cycle requires fresh immutable artifact references"),
            Self::HypothesisNotFound => write!(formatter, "target hypothesis was not found"),
            Self::EvidenceAlreadyRecorded => write!(formatter, "execution ID is already recorded as evidence for this hypothesis"),
            Self::EvidenceDatasetReused => write!(formatter, "evidence dataset was already used for this hypothesis"),
            Self::NoWorthwhileExperiment => write!(formatter, "no candidate passes safety constraints with positive adjusted utility"),
            Self::InvalidLikelihoodModel => write!(formatter, "selected experiment requires likelihoods strictly between zero and one"),
            Self::InvalidActionMapping => write!(formatter, "experiment action mapping is empty or duplicated"),
            Self::MissingActionMapping => write!(formatter, "selected experiment has no caller-supplied action mapping"),
            Self::NoExecutedAction => write!(formatter, "runtime returned no executed action"),
            Self::ExperimentIncomplete => write!(formatter, "environment did not complete the selected experiment action"),
            Self::NumericalOverflow => write!(formatter, "experiment likelihood ratio overflowed"),
            Self::Selection(error) => write!(formatter, "experiment selection failed: {error}"),
            Self::Reasoning(error) => write!(formatter, "hypothesis update failed: {error}"),
            Self::Runtime(error) => write!(formatter, "experiment runtime failed: {error}"),
            Self::ModelRegistry(error) => write!(formatter, "model promotion policy is invalid: {error}"),
            Self::Artifact(error) => write!(formatter, "artifact operation failed: {error}"),
        }
    }
}
impl std::error::Error for AutonomousResearchError {}
