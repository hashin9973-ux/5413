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
    pub evidence_commit: EvidenceCommitOutcome,
    /// Experiment results remain available even when candidate training or
    /// promotion is rejected after the environment effect has occurred.
    pub learning: Result<LearningCycleReport, LearningCycleError>,
}

/// Result of updating the in-memory hypothesis and persisting its immutable snapshot.
#[derive(Debug)]
pub enum EvidenceCommitOutcome {
    Persisted {
        hypothesis: HypothesisRecord,
    },
    UpdateRejected {
        error: ReasoningError,
    },
    PersistenceFailed {
        hypothesis: HypothesisRecord,
        error: ModelArtifactError,
    },
}

impl EvidenceCommitOutcome {
    pub fn hypothesis(&self) -> Option<&HypothesisRecord> {
        match self {
            Self::Persisted { hypothesis } | Self::PersistenceFailed { hypothesis, .. } => {
                Some(hypothesis)
            }
            Self::UpdateRejected { .. } => None,
        }
    }

    pub fn is_persisted(&self) -> bool {
        matches!(self, Self::Persisted { .. })
    }
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

    // Calculate both possible log-likelihood ratios before any external effect.
    // Subtracting logarithms avoids overflow from dividing tiny probabilities.
    let positive_log_likelihood_ratio = experiment.observation_probability_if_true.ln()
        - experiment.observation_probability_if_false.ln();
    let negative_log_likelihood_ratio = (-experiment.observation_probability_if_true).ln_1p()
        - (-experiment.observation_probability_if_false).ln_1p();
    let prior_log_odds =
        hypothesis.spec.prior_probability.ln() - (-hypothesis.spec.prior_probability).ln_1p();
    let existing_evidence_log_ratio = hypothesis
        .evidence
        .iter()
        .map(|record| record.log_likelihood_ratio)
        .sum::<f64>();
    if !positive_log_likelihood_ratio.is_finite()
        || !negative_log_likelihood_ratio.is_finite()
        || !existing_evidence_log_ratio.is_finite()
        || !(prior_log_odds + existing_evidence_log_ratio + positive_log_likelihood_ratio)
            .is_finite()
        || !(prior_log_odds + existing_evidence_log_ratio + negative_log_likelihood_ratio)
            .is_finite()
    {
        return Err(AutonomousResearchError::NumericalOverflow);
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
        return Err(AutonomousResearchError::ExperimentIncomplete { runtime });
    }

    let positive = positive_outcome(&runtime.final_observation.payload);
    let log_likelihood_ratio = if positive {
        positive_log_likelihood_ratio
    } else {
        negative_log_likelihood_ratio
    };

    let evidence = EvidenceRecord {
        evidence_ref: config.experiment.runtime.execution_id.clone(),
        dataset_id: config.experiment.evidence_dataset_id.clone(),
        log_likelihood_ratio,
        rationale: config.experiment.evidence_rationale.clone(),
    };
    // The environment effect is already recorded. Preserve its report even if
    // in-memory evidence validation or durable snapshot storage subsequently fails.
    let evidence_commit = match hypothesis_registry
        .add_evidence(hypothesis_id, evidence.clone(), hypothesis_policy)
        .cloned()
    {
        Err(error) => EvidenceCommitOutcome::UpdateRejected { error },
        Ok(hypothesis) => match artifact_store.put_hypothesis_registry(
            config.experiment.hypothesis_registry_artifact_ref,
            hypothesis_registry,
        ) {
            Ok(_) => EvidenceCommitOutcome::Persisted { hypothesis },
            Err(error) => EvidenceCommitOutcome::PersistenceFailed { hypothesis, error },
        },
    };

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
        evidence_commit,
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

    // Lifecycle mutations happen on a private snapshot. The live registry is
    // replaced only after its immutable artifact has been persisted successfully.
    let mut candidate_registry = registry.clone();
    artifact_store
        .register_model_candidate(&mut candidate_registry, spec)
        .map_err(LearningCycleError::Artifact)?;

    if let Err(error) = candidate_registry.evaluate_candidate(
        &config.model_id,
        config.model_version,
        evaluation.clone(),
        policy,
    ) {
        persist_model_registry(
            artifact_store,
            registry,
            candidate_registry,
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

    let promoted = match candidate_registry.promote(&config.model_id, config.model_version, policy)
    {
        Ok(()) => true,
        Err(
            ModelRegistryError::InsufficientImprovement
            | ModelRegistryError::UncalibratedUncertainty,
        ) => false,
        Err(error) => {
            persist_model_registry(
                artifact_store,
                registry,
                candidate_registry,
                &config.model_registry_artifact_ref,
            )?;
            return Err(LearningCycleError::Registry(error));
        }
    };
    persist_model_registry(
        artifact_store,
        registry,
        candidate_registry,
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
    registry: &mut ModelRegistry,
    candidate_registry: ModelRegistry,
    artifact_ref: &str,
) -> Result<(), LearningCycleError> {
    artifact_store
        .put_model_registry(artifact_ref, &candidate_registry)
        .map_err(LearningCycleError::Artifact)?;
    *registry = candidate_registry;
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
    ExperimentIncomplete { runtime: RuntimeReport },
    NumericalOverflow,
    Selection(ExperimentSelectionError),
    Reasoning(ReasoningError),
    Runtime(RuntimeError),
    ModelRegistry(ModelRegistryError),
    Artifact(ModelArtifactError),
}

impl AutonomousResearchError {
    /// Returns the durable execution report when an environment action did not
    /// complete. The report is available for reconciliation and recovery; it
    /// must not be interpreted as completed experimental evidence.
    pub fn runtime_report(&self) -> Option<&RuntimeReport> {
        match self {
            Self::ExperimentIncomplete { runtime } => Some(runtime),
            _ => None,
        }
    }
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
            Self::ExperimentIncomplete { .. } => write!(formatter, "environment did not complete the selected experiment action"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{
        ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionProposal, ActionRisk,
        AuthorizedAction,
    };
    use crate::benchmark::ScoreDirection;
    use crate::environment::{
        ActionOutcome, EnvironmentDescriptor, EnvironmentError, ObservationFrame,
    };
    use crate::model_registry::ModelEvaluation as RegistryEvaluation;
    use crate::planning::PlanningConfig;
    use crate::reasoning::HypothesisSpec;
    use crate::schema::{FieldSpec, SchemaDefinition, ValueKind};
    use crate::world_model::{evaluate_model, ActionDeltaModel};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temporary_path(label: &str) -> std::path::PathBuf {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "5413-research-cycle-{}-{label}-{id}.bin",
            std::process::id()
        ))
    }

    struct CounterEnvironment {
        descriptor: EnvironmentDescriptor,
        position: f64,
        sequence: u64,
        complete_actions: bool,
    }

    impl CounterEnvironment {
        fn new() -> Self {
            Self {
                descriptor: EnvironmentDescriptor {
                    id: "research-counter".to_owned(),
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
                            id: "empty-arguments".to_owned(),
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
                complete_actions: true,
            }
        }

        fn frame(&mut self) -> ObservationFrame {
            let frame = ObservationFrame {
                environment_id: "research-counter".to_owned(),
                session_id: "research-session".to_owned(),
                episode_id: "research-episode".to_owned(),
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
            _action: AuthorizedAction,
        ) -> Result<ActionOutcome, EnvironmentError> {
            self.position += 1.0;
            Ok(ActionOutcome {
                action_id: "advance".to_owned(),
                completed: self.complete_actions,
                observations: Vec::new(),
                feedback: DataValue::Null,
            })
        }
    }

    fn active_model() -> EnsembleWorldModel {
        let mut model = ActionDeltaModel::new("research-active", 1).expect("active model");
        model
            .fit(
                "active-training-v1",
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
                            DataValue::Number(5.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(6.0))]),
                    },
                ],
            )
            .expect("fit active model");
        EnsembleWorldModel::new(vec![Box::new(model)]).expect("active ensemble")
    }

    #[test]
    fn failed_registry_snapshot_does_not_publish_candidate_promotion_in_memory() {
        let path = temporary_path("registry-transaction");
        let mut artifact_store = ModelArtifactStore::open(&path).expect("artifact store");
        let mut incumbent_model =
            ActionDeltaModel::new("transactional-model", 1).expect("incumbent");
        incumbent_model
            .fit(
                "training-v1",
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
                            DataValue::Number(5.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(6.0))]),
                    },
                ],
            )
            .expect("fit incumbent");
        let mut candidate_model =
            ActionDeltaModel::new("transactional-model", 2).expect("candidate");
        candidate_model
            .fit(
                "training-v2",
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
                            DataValue::Number(5.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(6.0))]),
                    },
                ],
            )
            .expect("fit candidate");
        artifact_store
            .put_model("transaction-model-v1", &incumbent_model)
            .expect("persist incumbent");
        artifact_store
            .put_model("transaction-model-v2", &candidate_model)
            .expect("persist candidate");

        let policy = ModelPromotionPolicy {
            minimum_evaluation_samples: 2,
            minimum_improvement: 0.0,
            resource_cost_weight: 0.0,
            require_calibrated_uncertainty: false,
        };
        let mut registry = ModelRegistry::default();
        artifact_store
            .register_model_candidate(
                &mut registry,
                ModelCandidateSpec {
                    id: "transactional-model".to_owned(),
                    version: 1,
                    model_artifact_ref: "transaction-model-v1".to_owned(),
                    environment_id: "counter".to_owned(),
                    environment_version: "v1".to_owned(),
                    training_dataset_id: "training-v1".to_owned(),
                },
            )
            .expect("register incumbent");
        registry
            .evaluate_candidate(
                "transactional-model",
                1,
                RegistryEvaluation {
                    evaluation_dataset_id: "holdout-v1".to_owned(),
                    metric_id: "rmse".to_owned(),
                    direction: ScoreDirection::LowerIsBetter,
                    sample_count: 2,
                    score: 1.0,
                    resource_cost: 0.0,
                    uncertainty_calibrated: true,
                },
                policy,
            )
            .expect("evaluate incumbent");
        registry
            .promote("transactional-model", 1, policy)
            .expect("promote incumbent");

        let mut staged_registry = registry.clone();
        artifact_store
            .register_model_candidate(
                &mut staged_registry,
                ModelCandidateSpec {
                    id: "transactional-model".to_owned(),
                    version: 2,
                    model_artifact_ref: "transaction-model-v2".to_owned(),
                    environment_id: "counter".to_owned(),
                    environment_version: "v1".to_owned(),
                    training_dataset_id: "training-v2".to_owned(),
                },
            )
            .expect("register candidate");
        staged_registry
            .evaluate_candidate(
                "transactional-model",
                2,
                RegistryEvaluation {
                    evaluation_dataset_id: "holdout-v1".to_owned(),
                    metric_id: "rmse".to_owned(),
                    direction: ScoreDirection::LowerIsBetter,
                    sample_count: 2,
                    score: 0.5,
                    resource_cost: 0.0,
                    uncertainty_calibrated: true,
                },
                policy,
            )
            .expect("evaluate candidate");
        staged_registry
            .promote("transactional-model", 2, policy)
            .expect("stage candidate promotion");

        artifact_store
            .put("occupied-registry-reference", DataValue::Null)
            .expect("occupy immutable registry reference");
        let incumbent_snapshot = registry.clone();
        assert!(matches!(
            persist_model_registry(
                &mut artifact_store,
                &mut registry,
                staged_registry,
                "occupied-registry-reference",
            ),
            Err(LearningCycleError::Artifact(
                ModelArtifactError::ReferenceConflict
            ))
        ));
        assert_eq!(registry, incumbent_snapshot);
        assert_eq!(
            registry
                .active("transactional-model")
                .map(|record| record.spec.version),
            Some(1)
        );
        assert_eq!(
            registry
                .get("transactional-model", 1)
                .map(|record| record.status),
            Some(ModelStatus::Active)
        );
        assert_eq!(registry.get("transactional-model", 2), None);

        drop(artifact_store);
        std::fs::remove_file(path).expect("remove artifact store");
    }

    #[test]
    fn research_cycle_preserves_execution_report_when_later_learning_fails() {
        let journal_path = temporary_path("journal");
        let artifact_path = temporary_path("artifacts");
        let mut journal = ExperienceJournal::open(&journal_path).expect("journal");
        let mut artifact_store = ModelArtifactStore::open(&artifact_path).expect("artifacts");
        let mut adapter = CounterEnvironment::new();
        let mut candidate_model =
            ActionDeltaModel::new("research-candidate", 1).expect("candidate model");
        let mut hypothesis_registry = HypothesisRegistry::default();
        let hypothesis_policy = HypothesisPolicy {
            minimum_evidence_count: 1,
            support_probability: 0.8,
            refutation_probability: 0.2,
        };
        hypothesis_registry
            .insert(
                HypothesisSpec {
                    id: "advance-increases-position".to_owned(),
                    claim: DataValue::Text("advance increases position".to_owned()),
                    description: "one advance increases the position observation".to_owned(),
                    prior_probability: 0.5,
                    source_dataset_id: Some("initial-discovery".to_owned()),
                },
                hypothesis_policy,
            )
            .expect("hypothesis");

        let experiments = [ExperimentCandidate {
            id: "advance-once".to_owned(),
            specification: DataValue::Text("apply one reversible advance action".to_owned()),
            observation_probability_if_true: 0.9,
            observation_probability_if_false: 0.1,
            resource_cost: 1.0,
            risk: 0.01,
            irreversible: false,
        }];
        let action_mappings = [ExperimentActionMapping {
            experiment_id: "advance-once".to_owned(),
            action: ActionProposal {
                action_id: "advance".to_owned(),
                arguments: DataValue::Object(Default::default()),
            },
        }];
        let validation_samples = [
            TransitionSample {
                action_id: "advance".to_owned(),
                before: DataValue::object([("position".to_owned(), DataValue::Number(10.0))]),
                after: DataValue::object([("position".to_owned(), DataValue::Number(11.0))]),
            },
            TransitionSample {
                action_id: "advance".to_owned(),
                before: DataValue::object([("position".to_owned(), DataValue::Number(20.0))]),
                after: DataValue::object([("position".to_owned(), DataValue::Number(21.0))]),
            },
        ];
        let promotion_policy = ModelPromotionPolicy {
            minimum_evaluation_samples: 2,
            minimum_improvement: 0.0,
            resource_cost_weight: 0.0,
            require_calibrated_uncertainty: false,
        };

        let mut model_registry = ModelRegistry::default();
        let report = run_autonomous_research_cycle(
            AutonomousResearchContext {
                adapter: &mut adapter,
                active_model: &active_model(),
                candidate_model: &mut candidate_model,
                hypothesis_registry: &mut hypothesis_registry,
                journal: &mut journal,
                artifact_store: &mut artifact_store,
                model_registry: &mut model_registry,
            },
            AutonomousResearchRequest {
                hypothesis_id: "advance-increases-position",
                experiments: &experiments,
                action_mappings: &action_mappings,
                validation_samples: &validation_samples,
                selection_policy: ExperimentSelectionPolicy {
                    maximum_resource_cost: 10.0,
                    maximum_risk: 0.1,
                    resource_cost_weight: 0.0,
                    risk_weight: 0.0,
                    allow_irreversible: false,
                },
                hypothesis_policy,
                promotion_policy,
                config: AutonomousResearchConfig {
                    experiment: ExperimentExecutionConfig {
                        runtime: RuntimeConfig {
                            execution_id: "research-run-1".to_owned(),
                            episode_id: "research-episode".to_owned(),
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
                        evidence_dataset_id: "experiment-evidence-v1".to_owned(),
                        evidence_rationale: "final position observed after intervention".to_owned(),
                        hypothesis_registry_artifact_ref: "hypotheses-after-run-1".to_owned(),
                    },
                    learning: LearningCycleConfig {
                        episode_id: "research-episode".to_owned(),
                        training_dataset_id: "episode-training-v1".to_owned(),
                        validation_dataset_id: "model-holdout-v1".to_owned(),
                        model_id: "research-candidate".to_owned(),
                        model_version: 1,
                        model_artifact_ref: "research-candidate-artifact-v1".to_owned(),
                        model_registry_artifact_ref: "model-registry-after-run-1".to_owned(),
                        environment_id: "research-counter".to_owned(),
                        environment_version: "v1".to_owned(),
                    },
                },
            },
            AutonomousResearchCallbacks {
                utility: |state: &DataValue| {
                    state
                        .get("position")
                        .and_then(DataValue::as_f64)
                        .unwrap_or(0.0)
                },
                positive_outcome: |state: &DataValue| {
                    state
                        .get("position")
                        .and_then(DataValue::as_f64)
                        .is_some_and(|position| position > 0.0)
                },
                evaluate: |model: &ActionDeltaModel, samples: &[TransitionSample]| {
                    let measured = evaluate_model(model, "model-holdout-v1", samples)
                        .map_err(|error| error.to_string())?;
                    Ok(RegistryEvaluation {
                        evaluation_dataset_id: measured.evaluation_dataset_id,
                        metric_id: "rmse".to_owned(),
                        direction: ScoreDirection::LowerIsBetter,
                        sample_count: measured.evaluated_transitions,
                        score: measured.root_mean_squared_error,
                        resource_cost: 1.0,
                        uncertainty_calibrated: measured.calibrated_predictions
                            == measured.predicted_fields,
                    })
                },
            },
        )
        .expect("research cycle");

        assert_eq!(report.runtime.steps.len(), 1);
        assert_eq!(
            report
                .runtime
                .final_observation
                .payload
                .get("position")
                .and_then(DataValue::as_f64),
            Some(1.0)
        );
        let updated_hypothesis = report
            .evidence_commit
            .hypothesis()
            .expect("hypothesis evidence update");
        assert!(report.evidence_commit.is_persisted());
        assert_eq!(
            updated_hypothesis.status,
            crate::reasoning::HypothesisStatus::Supported
        );
        assert!((updated_hypothesis.posterior_probability - 0.9).abs() < 1e-12);
        let learning = report.learning.expect("learning outcome");
        assert!(learning.promoted);
        assert_eq!(learning.training_samples, 1);
        assert_eq!(learning.validation_samples, 2);

        let restored_hypotheses = artifact_store
            .load_hypothesis_registry("hypotheses-after-run-1", hypothesis_policy)
            .expect("restore hypothesis registry");
        assert_eq!(restored_hypotheses, hypothesis_registry);
        let restored_models = artifact_store
            .load_model_registry("model-registry-after-run-1")
            .expect("restore model registry");
        assert_eq!(
            restored_models
                .active("research-candidate")
                .map(|record| record.spec.version),
            Some(1)
        );

        // A successful environment effect must remain reportable when a later
        // candidate-evaluation stage fails. The incumbent registry entry must
        // remain active and no candidate artifact may be published.
        hypothesis_registry
            .insert(
                HypothesisSpec {
                    id: "advance-increases-position-v2".to_owned(),
                    claim: DataValue::Text("advance increases the position observation".to_owned()),
                    description: "independent hypothesis for the injected learning failure"
                        .to_owned(),
                    prior_probability: 0.5,
                    source_dataset_id: Some("independent-discovery-v2".to_owned()),
                },
                hypothesis_policy,
            )
            .expect("second hypothesis");
        let mut second_candidate_model =
            ActionDeltaModel::new("research-candidate", 2).expect("second candidate model");
        let second_report = run_autonomous_research_cycle(
            AutonomousResearchContext {
                adapter: &mut adapter,
                active_model: &active_model(),
                candidate_model: &mut second_candidate_model,
                hypothesis_registry: &mut hypothesis_registry,
                journal: &mut journal,
                artifact_store: &mut artifact_store,
                model_registry: &mut model_registry,
            },
            AutonomousResearchRequest {
                hypothesis_id: "advance-increases-position-v2",
                experiments: &experiments,
                action_mappings: &action_mappings,
                validation_samples: &validation_samples,
                selection_policy: ExperimentSelectionPolicy {
                    maximum_resource_cost: 10.0,
                    maximum_risk: 0.1,
                    resource_cost_weight: 0.0,
                    risk_weight: 0.0,
                    allow_irreversible: false,
                },
                hypothesis_policy,
                promotion_policy,
                config: AutonomousResearchConfig {
                    experiment: ExperimentExecutionConfig {
                        runtime: RuntimeConfig {
                            execution_id: "research-run-2".to_owned(),
                            episode_id: "research-episode".to_owned(),
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
                        evidence_dataset_id: "experiment-evidence-v2".to_owned(),
                        evidence_rationale: "second intervention outcome".to_owned(),
                        hypothesis_registry_artifact_ref: "hypotheses-after-run-2".to_owned(),
                    },
                    learning: LearningCycleConfig {
                        episode_id: "research-episode".to_owned(),
                        training_dataset_id: "episode-training-v2".to_owned(),
                        validation_dataset_id: "model-holdout-v2".to_owned(),
                        model_id: "research-candidate".to_owned(),
                        model_version: 2,
                        model_artifact_ref: "research-candidate-artifact-v2".to_owned(),
                        model_registry_artifact_ref: "model-registry-after-run-2".to_owned(),
                        environment_id: "research-counter".to_owned(),
                        environment_version: "v1".to_owned(),
                    },
                },
            },
            AutonomousResearchCallbacks {
                utility: |state: &DataValue| {
                    state
                        .get("position")
                        .and_then(DataValue::as_f64)
                        .unwrap_or(0.0)
                },
                positive_outcome: |state: &DataValue| {
                    state
                        .get("position")
                        .and_then(DataValue::as_f64)
                        .is_some_and(|position| position > 0.0)
                },
                evaluate: |_model: &ActionDeltaModel, _samples: &[TransitionSample]| {
                    Err("injected held-out evaluation failure".to_owned())
                },
            },
        )
        .expect("execution report survives learning failure");

        assert_eq!(second_report.runtime.steps.len(), 1);
        assert!(second_report.runtime.steps[0].outcome.completed);
        assert_eq!(
            second_report
                .runtime
                .final_observation
                .payload
                .get("position")
                .and_then(DataValue::as_f64),
            Some(2.0)
        );
        assert!(second_report.evidence_commit.is_persisted());
        assert!(matches!(
            second_report.learning,
            Err(LearningCycleError::EvaluationFailed(message))
                if message == "injected held-out evaluation failure"
        ));
        assert_eq!(
            model_registry
                .active("research-candidate")
                .map(|record| record.spec.version),
            Some(1)
        );
        assert_eq!(model_registry.len(), 1);
        assert!(artifact_store
            .get("research-candidate-artifact-v2")
            .is_none());

        // An adapter may report an incomplete action after a real effect has
        // occurred. Preserve its journaled runtime report, but do not classify
        // the observation as evidence or train/promote a candidate from it.
        adapter.complete_actions = false;
        hypothesis_registry
            .insert(
                HypothesisSpec {
                    id: "advance-increases-position-v3".to_owned(),
                    claim: DataValue::Text("advance increases the position observation".to_owned()),
                    description: "hypothesis for an incomplete execution".to_owned(),
                    prior_probability: 0.5,
                    source_dataset_id: Some("independent-discovery-v3".to_owned()),
                },
                hypothesis_policy,
            )
            .expect("third hypothesis");
        let mut third_candidate_model =
            ActionDeltaModel::new("research-candidate", 3).expect("third candidate model");
        let incomplete = run_autonomous_research_cycle(
            AutonomousResearchContext {
                adapter: &mut adapter,
                active_model: &active_model(),
                candidate_model: &mut third_candidate_model,
                hypothesis_registry: &mut hypothesis_registry,
                journal: &mut journal,
                artifact_store: &mut artifact_store,
                model_registry: &mut model_registry,
            },
            AutonomousResearchRequest {
                hypothesis_id: "advance-increases-position-v3",
                experiments: &experiments,
                action_mappings: &action_mappings,
                validation_samples: &validation_samples,
                selection_policy: ExperimentSelectionPolicy {
                    maximum_resource_cost: 10.0,
                    maximum_risk: 0.1,
                    resource_cost_weight: 0.0,
                    risk_weight: 0.0,
                    allow_irreversible: false,
                },
                hypothesis_policy,
                promotion_policy,
                config: AutonomousResearchConfig {
                    experiment: ExperimentExecutionConfig {
                        runtime: RuntimeConfig {
                            execution_id: "research-run-3".to_owned(),
                            episode_id: "research-episode".to_owned(),
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
                        evidence_dataset_id: "experiment-evidence-v3".to_owned(),
                        evidence_rationale: "incomplete intervention must not become evidence".to_owned(),
                        hypothesis_registry_artifact_ref: "hypotheses-after-run-3".to_owned(),
                    },
                    learning: LearningCycleConfig {
                        episode_id: "research-episode".to_owned(),
                        training_dataset_id: "episode-training-v3".to_owned(),
                        validation_dataset_id: "model-holdout-v3".to_owned(),
                        model_id: "research-candidate".to_owned(),
                        model_version: 3,
                        model_artifact_ref: "research-candidate-artifact-v3".to_owned(),
                        model_registry_artifact_ref: "model-registry-after-run-3".to_owned(),
                        environment_id: "research-counter".to_owned(),
                        environment_version: "v1".to_owned(),
                    },
                },
            },
            AutonomousResearchCallbacks {
                utility: |state: &DataValue| {
                    state
                        .get("position")
                        .and_then(DataValue::as_f64)
                        .unwrap_or(0.0)
                },
                positive_outcome: |state: &DataValue| {
                    state
                        .get("position")
                        .and_then(DataValue::as_f64)
                        .is_some_and(|position| position > 0.0)
                },
                evaluate: |_model: &ActionDeltaModel, _samples: &[TransitionSample]| {
                    panic!("incomplete experiments must not reach candidate evaluation")
                },
            },
        )
        .expect_err("incomplete environment outcome must reject the experiment");
        let runtime = incomplete
            .runtime_report()
            .expect("incomplete action retains its runtime report");
        assert_eq!(runtime.steps.len(), 1);
        assert_eq!(
            runtime.termination,
            crate::runtime::RuntimeTermination::EnvironmentReportedIncomplete
        );
        assert!(!runtime.steps[0].outcome.completed);
        assert_eq!(runtime.steps[0].action_id, "advance");
        assert_eq!(
            runtime
                .final_observation
                .payload
                .get("position")
                .and_then(DataValue::as_f64),
            Some(3.0)
        );
        assert!(hypothesis_registry
            .get("advance-increases-position-v3")
            .expect("third hypothesis remains")
            .evidence
            .is_empty());
        assert_eq!(model_registry.len(), 1);
        assert!(artifact_store
            .get("research-candidate-artifact-v3")
            .is_none());

        drop(artifact_store);
        drop(journal);
        std::fs::remove_file(journal_path).expect("remove journal");
        std::fs::remove_file(artifact_path).expect("remove artifacts");
    }
}
