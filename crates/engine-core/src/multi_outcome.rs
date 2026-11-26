//! Categorical-outcome experiment design and evidence-backed execution.
//!
//! Likelihoods and outcome classification are supplied by the environment
//! integration. The selector computes exact expected information gain for that
//! categorical model; it does not claim the declared probabilities are calibrated.

use crate::action::ActionProposal;
use crate::environment::EnvironmentAdapter;
use crate::experience_journal::ExperienceJournal;
use crate::model_artifact::{ModelArtifactError, ModelArtifactStore};
use crate::reasoning::{
    EvidenceRecord, HypothesisPolicy, HypothesisRecord, HypothesisRegistry, ReasoningError,
};
use crate::runtime::{run_receding_horizon, RuntimeConfig, RuntimeError, RuntimeReport};
use crate::value::DataValue;
use crate::world_model::EnsembleWorldModel;
use std::collections::BTreeSet;

const PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeLikelihood {
    pub outcome_id: String,
    pub probability_if_true: f64,
    pub probability_if_false: f64,
}

impl OutcomeLikelihood {
    fn validate(&self) -> Result<(), MultiOutcomeExperimentError> {
        if self.outcome_id.trim().is_empty()
            || !self.probability_if_true.is_finite()
            || self.probability_if_true <= 0.0
            || self.probability_if_true > 1.0
            || !self.probability_if_false.is_finite()
            || self.probability_if_false <= 0.0
            || self.probability_if_false > 1.0
        {
            return Err(MultiOutcomeExperimentError::InvalidLikelihoodModel);
        }
        let ratio = self.probability_if_true / self.probability_if_false;
        if !ratio.is_finite() || !ratio.ln().is_finite() {
            return Err(MultiOutcomeExperimentError::NumericalOverflow);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiOutcomeExperimentCandidate {
    pub id: String,
    pub specification: DataValue,
    pub outcomes: Vec<OutcomeLikelihood>,
    pub resource_cost: f64,
    /// Normalized caller-supplied risk estimate in [0, 1].
    pub risk: f64,
    pub irreversible: bool,
}

impl MultiOutcomeExperimentCandidate {
    pub fn validate(&self) -> Result<(), MultiOutcomeExperimentError> {
        if self.id.trim().is_empty()
            || self.outcomes.len() < 2
            || !self.resource_cost.is_finite()
            || self.resource_cost < 0.0
            || !self.risk.is_finite()
            || !(0.0..=1.0).contains(&self.risk)
        {
            return Err(MultiOutcomeExperimentError::InvalidCandidate);
        }
        self.specification
            .validate_finite()
            .map_err(|_| MultiOutcomeExperimentError::InvalidCandidate)?;

        let mut ids = BTreeSet::new();
        let mut true_total = 0.0;
        let mut false_total = 0.0;
        for outcome in &self.outcomes {
            outcome.validate()?;
            if !ids.insert(outcome.outcome_id.as_str()) {
                return Err(MultiOutcomeExperimentError::DuplicateOutcome);
            }
            true_total += outcome.probability_if_true;
            false_total += outcome.probability_if_false;
        }
        if !true_total.is_finite()
            || !false_total.is_finite()
            || (true_total - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
            || (false_total - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
        {
            return Err(MultiOutcomeExperimentError::InvalidLikelihoodModel);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiOutcomeSelectionPolicy {
    pub maximum_resource_cost: f64,
    pub maximum_risk: f64,
    pub resource_cost_weight: f64,
    pub risk_weight: f64,
    pub allow_irreversible: bool,
}

impl Default for MultiOutcomeSelectionPolicy {
    fn default() -> Self {
        Self {
            maximum_resource_cost: 100.0,
            maximum_risk: 0.25,
            resource_cost_weight: 0.01,
            risk_weight: 1.0,
            allow_irreversible: false,
        }
    }
}

impl MultiOutcomeSelectionPolicy {
    pub fn validate(&self) -> Result<(), MultiOutcomeExperimentError> {
        if !self.maximum_resource_cost.is_finite()
            || self.maximum_resource_cost < 0.0
            || !self.maximum_risk.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_risk)
            || !self.resource_cost_weight.is_finite()
            || self.resource_cost_weight < 0.0
            || !self.risk_weight.is_finite()
            || self.risk_weight < 0.0
        {
            return Err(MultiOutcomeExperimentError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedMultiOutcomeExperiment {
    pub experiment_id: String,
    pub hypothesis_id: String,
    /// Entropy reduction in nats under the supplied categorical likelihood model.
    pub expected_information_gain: f64,
    pub risk_adjusted_utility: f64,
    pub resource_cost: f64,
    pub risk: f64,
    pub irreversible: bool,
}

impl RankedMultiOutcomeExperiment {
    pub fn is_worthwhile(&self) -> bool {
        self.risk_adjusted_utility > 0.0
    }
}

/// Ranks candidate interventions by categorical expected information gain,
/// subtracting declared cost and risk penalties.
pub fn rank_multi_outcome_experiments(
    hypothesis: &HypothesisRecord,
    candidates: &[MultiOutcomeExperimentCandidate],
    policy: MultiOutcomeSelectionPolicy,
) -> Result<Vec<RankedMultiOutcomeExperiment>, MultiOutcomeExperimentError> {
    policy.validate()?;
    let prior = hypothesis.posterior_probability;
    if hypothesis.spec.id.trim().is_empty() || !prior.is_finite() || !(0.0..=1.0).contains(&prior) {
        return Err(MultiOutcomeExperimentError::InvalidHypothesis);
    }

    let mut seen = BTreeSet::new();
    let mut ranked = Vec::new();
    for candidate in candidates {
        candidate.validate()?;
        if !seen.insert(candidate.id.as_str()) {
            return Err(MultiOutcomeExperimentError::DuplicateCandidate);
        }
        if candidate.resource_cost > policy.maximum_resource_cost
            || candidate.risk > policy.maximum_risk
            || (candidate.irreversible && !policy.allow_irreversible)
        {
            continue;
        }
        let information_gain = categorical_expected_information_gain(prior, &candidate.outcomes)?;
        let utility = information_gain
            - policy.resource_cost_weight * candidate.resource_cost
            - policy.risk_weight * candidate.risk;
        if !utility.is_finite() {
            return Err(MultiOutcomeExperimentError::NumericalOverflow);
        }
        ranked.push(RankedMultiOutcomeExperiment {
            experiment_id: candidate.id.clone(),
            hypothesis_id: hypothesis.spec.id.clone(),
            expected_information_gain: information_gain,
            risk_adjusted_utility: utility,
            resource_cost: candidate.resource_cost,
            risk: candidate.risk,
            irreversible: candidate.irreversible,
        });
    }
    ranked.sort_by(|left, right| {
        right
            .risk_adjusted_utility
            .total_cmp(&left.risk_adjusted_utility)
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    Ok(ranked)
}

fn categorical_expected_information_gain(
    prior: f64,
    outcomes: &[OutcomeLikelihood],
) -> Result<f64, MultiOutcomeExperimentError> {
    let prior_entropy = binary_entropy(prior);
    let mut expected_posterior_entropy = 0.0;
    for outcome in outcomes {
        let probability =
            prior * outcome.probability_if_true + (1.0 - prior) * outcome.probability_if_false;
        if !probability.is_finite() || probability < 0.0 {
            return Err(MultiOutcomeExperimentError::NumericalOverflow);
        }
        if probability == 0.0 {
            continue;
        }
        let posterior = prior * outcome.probability_if_true / probability;
        if !posterior.is_finite() || !(0.0..=1.0).contains(&posterior) {
            return Err(MultiOutcomeExperimentError::NumericalOverflow);
        }
        expected_posterior_entropy += probability * binary_entropy(posterior);
    }
    let information_gain = prior_entropy - expected_posterior_entropy;
    if !information_gain.is_finite() {
        return Err(MultiOutcomeExperimentError::NumericalOverflow);
    }
    // Floating-point summation can produce a tiny negative value for an
    // uninformative experiment whose theoretical information gain is zero.
    Ok(information_gain.max(0.0))
}

fn binary_entropy(probability: f64) -> f64 {
    if probability <= 0.0 || probability >= 1.0 {
        0.0
    } else {
        -probability * probability.ln() - (1.0 - probability) * (1.0 - probability).ln()
    }
}

#[derive(Clone, Debug)]
pub struct MultiOutcomeActionMapping {
    pub experiment_id: String,
    pub action: ActionProposal,
}

#[derive(Clone, Debug)]
pub struct MultiOutcomeExecutionConfig {
    pub runtime: RuntimeConfig,
    pub evidence_dataset_id: String,
    pub evidence_rationale: String,
    pub hypothesis_registry_artifact_ref: String,
}

#[derive(Debug)]
pub struct MultiOutcomeExperimentContext<'a, A> {
    pub adapter: &'a mut A,
    pub model: &'a EnsembleWorldModel,
    pub hypotheses: &'a mut HypothesisRegistry,
    pub journal: &'a mut ExperienceJournal,
    pub artifact_store: &'a mut ModelArtifactStore,
}

#[derive(Clone, Debug)]
pub struct MultiOutcomeExperimentRequest<'a> {
    pub hypothesis_id: &'a str,
    pub candidates: &'a [MultiOutcomeExperimentCandidate],
    pub action_mappings: &'a [MultiOutcomeActionMapping],
    pub selection_policy: MultiOutcomeSelectionPolicy,
    pub hypothesis_policy: HypothesisPolicy,
    pub config: MultiOutcomeExecutionConfig,
}

pub struct MultiOutcomeExperimentCallbacks<U, C> {
    pub utility: U,
    /// Returns the ID of the categorical outcome observed in the final state.
    pub classify_outcome: C,
}

#[derive(Debug)]
pub struct MultiOutcomeExperimentReport {
    pub selected: RankedMultiOutcomeExperiment,
    pub ranked_candidates: Vec<RankedMultiOutcomeExperiment>,
    pub runtime: RuntimeReport,
    pub evidence_commit: MultiOutcomeEvidenceCommitOutcome,
}

#[derive(Debug)]
pub enum MultiOutcomeEvidenceCommitOutcome {
    Persisted {
        outcome_id: String,
        evidence: EvidenceRecord,
        hypothesis: HypothesisRecord,
    },
    UpdateRejected {
        outcome_id: String,
        evidence: EvidenceRecord,
        error: ReasoningError,
    },
    PersistenceFailed {
        outcome_id: String,
        evidence: EvidenceRecord,
        hypothesis: HypothesisRecord,
        error: ModelArtifactError,
    },
    ClassificationFailed {
        message: String,
    },
    UnknownObservedOutcome {
        outcome_id: String,
    },
}

impl MultiOutcomeEvidenceCommitOutcome {
    pub fn hypothesis(&self) -> Option<&HypothesisRecord> {
        match self {
            Self::Persisted { hypothesis, .. } | Self::PersistenceFailed { hypothesis, .. } => {
                Some(hypothesis)
            }
            Self::UpdateRejected { .. }
            | Self::ClassificationFailed { .. }
            | Self::UnknownObservedOutcome { .. } => None,
        }
    }

    pub fn is_persisted(&self) -> bool {
        matches!(self, Self::Persisted { .. })
    }
}

/// Selects and executes one intervention, then records evidence for the actual
/// categorical outcome. The runtime report is retained when classification,
/// hypothesis update, or artifact persistence fails after the environment effect.
pub fn run_multi_outcome_experiment<A, U, C>(
    context: MultiOutcomeExperimentContext<'_, A>,
    request: MultiOutcomeExperimentRequest<'_>,
    callbacks: MultiOutcomeExperimentCallbacks<U, C>,
) -> Result<MultiOutcomeExperimentReport, MultiOutcomeExperimentError>
where
    A: EnvironmentAdapter,
    U: Fn(&DataValue) -> f64,
    C: Fn(&DataValue) -> Result<String, String>,
{
    request.selection_policy.validate()?;
    request.hypothesis_policy.validate()?;
    request
        .config
        .runtime
        .validate()
        .map_err(MultiOutcomeExperimentError::Runtime)?;
    if request.config.runtime.maximum_steps != 1
        || request.config.evidence_dataset_id.trim().is_empty()
        || request.config.evidence_rationale.trim().is_empty()
        || request
            .config
            .hypothesis_registry_artifact_ref
            .trim()
            .is_empty()
    {
        return Err(MultiOutcomeExperimentError::InvalidConfiguration);
    }
    if context
        .artifact_store
        .get(&request.config.hypothesis_registry_artifact_ref)
        .is_some()
    {
        return Err(MultiOutcomeExperimentError::ArtifactReferenceAlreadyExists);
    }

    let hypothesis = context
        .hypotheses
        .get(request.hypothesis_id)
        .cloned()
        .ok_or(MultiOutcomeExperimentError::HypothesisNotFound)?;
    if hypothesis
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_ref == request.config.runtime.execution_id)
    {
        return Err(MultiOutcomeExperimentError::EvidenceAlreadyRecorded);
    }
    if hypothesis.spec.source_dataset_id.as_deref()
        == Some(request.config.evidence_dataset_id.as_str())
        || hypothesis
            .evidence
            .iter()
            .any(|evidence| evidence.dataset_id == request.config.evidence_dataset_id)
    {
        return Err(MultiOutcomeExperimentError::EvidenceDatasetReused);
    }

    let ranked_candidates =
        rank_multi_outcome_experiments(&hypothesis, request.candidates, request.selection_policy)?;
    let selected = ranked_candidates
        .iter()
        .find(|candidate| candidate.is_worthwhile())
        .cloned()
        .ok_or(MultiOutcomeExperimentError::NoWorthwhileExperiment)?;
    let experiment = request
        .candidates
        .iter()
        .find(|candidate| candidate.id == selected.experiment_id)
        .ok_or(MultiOutcomeExperimentError::CandidateNotFound)?;

    // Preflight every possible evidence update before any external action.
    for outcome in &experiment.outcomes {
        outcome.validate()?;
    }

    let mut mapping_ids = BTreeSet::new();
    for mapping in request.action_mappings {
        if mapping.experiment_id.trim().is_empty()
            || !mapping_ids.insert(mapping.experiment_id.as_str())
        {
            return Err(MultiOutcomeExperimentError::InvalidActionMapping);
        }
    }
    let mapping = request
        .action_mappings
        .iter()
        .find(|mapping| mapping.experiment_id == selected.experiment_id)
        .ok_or(MultiOutcomeExperimentError::MissingActionMapping)?;

    let runtime = run_receding_horizon(
        context.adapter,
        context.model,
        std::slice::from_ref(&mapping.action),
        request.config.runtime.clone(),
        context.journal,
        callbacks.utility,
    )
    .map_err(MultiOutcomeExperimentError::Runtime)?;
    let last_step = runtime
        .steps
        .last()
        .ok_or(MultiOutcomeExperimentError::NoExecutedAction)?;
    if !last_step.outcome.completed {
        return Err(MultiOutcomeExperimentError::ExperimentIncomplete);
    }

    let evidence_commit = match (callbacks.classify_outcome)(&runtime.final_observation.payload) {
        Err(message) => MultiOutcomeEvidenceCommitOutcome::ClassificationFailed { message },
        Ok(outcome_id) => match experiment
            .outcomes
            .iter()
            .find(|item| item.outcome_id == outcome_id)
        {
            None => MultiOutcomeEvidenceCommitOutcome::UnknownObservedOutcome { outcome_id },
            Some(outcome) => {
                let log_likelihood_ratio =
                    (outcome.probability_if_true / outcome.probability_if_false).ln();
                let evidence = EvidenceRecord {
                    evidence_ref: request.config.runtime.execution_id.clone(),
                    dataset_id: request.config.evidence_dataset_id.clone(),
                    log_likelihood_ratio,
                    rationale: request.config.evidence_rationale.clone(),
                };
                match context.hypotheses.add_evidence(
                    request.hypothesis_id,
                    evidence.clone(),
                    request.hypothesis_policy,
                ) {
                    Err(error) => MultiOutcomeEvidenceCommitOutcome::UpdateRejected {
                        outcome_id,
                        evidence,
                        error,
                    },
                    Ok(updated) => {
                        let hypothesis = updated.clone();
                        match context.artifact_store.put_hypothesis_registry(
                            request.config.hypothesis_registry_artifact_ref,
                            context.hypotheses,
                        ) {
                            Ok(_) => MultiOutcomeEvidenceCommitOutcome::Persisted {
                                outcome_id,
                                evidence,
                                hypothesis,
                            },
                            Err(error) => MultiOutcomeEvidenceCommitOutcome::PersistenceFailed {
                                outcome_id,
                                evidence,
                                hypothesis,
                                error,
                            },
                        }
                    }
                }
            }
        },
    };

    Ok(MultiOutcomeExperimentReport {
        selected,
        ranked_candidates,
        runtime,
        evidence_commit,
    })
}

#[derive(Debug)]
pub enum MultiOutcomeExperimentError {
    InvalidCandidate,
    DuplicateOutcome,
    InvalidLikelihoodModel,
    InvalidPolicy,
    InvalidHypothesis,
    DuplicateCandidate,
    NumericalOverflow,
    InvalidConfiguration,
    ArtifactReferenceAlreadyExists,
    HypothesisNotFound,
    EvidenceAlreadyRecorded,
    EvidenceDatasetReused,
    NoWorthwhileExperiment,
    CandidateNotFound,
    InvalidActionMapping,
    MissingActionMapping,
    NoExecutedAction,
    ExperimentIncomplete,
    Selection(String),
    Runtime(RuntimeError),
    Reasoning(ReasoningError),
}

impl std::fmt::Display for MultiOutcomeExperimentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCandidate => write!(formatter, "multi-outcome experiment candidate is invalid"),
            Self::DuplicateOutcome => write!(formatter, "multi-outcome candidate contains a duplicate outcome ID"),
            Self::InvalidLikelihoodModel => write!(formatter, "categorical outcome probabilities must be positive and sum to one under both hypothesis states"),
            Self::InvalidPolicy => write!(formatter, "multi-outcome selection policy is invalid"),
            Self::InvalidHypothesis => write!(formatter, "multi-outcome target hypothesis is invalid"),
            Self::DuplicateCandidate => write!(formatter, "multi-outcome experiment ID is duplicated"),
            Self::NumericalOverflow => write!(formatter, "multi-outcome information gain or likelihood calculation overflowed"),
            Self::InvalidConfiguration => write!(formatter, "multi-outcome execution configuration is invalid"),
            Self::ArtifactReferenceAlreadyExists => write!(formatter, "hypothesis artifact reference already exists; use a fresh immutable reference"),
            Self::HypothesisNotFound => write!(formatter, "target hypothesis was not found"),
            Self::EvidenceAlreadyRecorded => write!(formatter, "execution ID is already used as evidence for this hypothesis"),
            Self::EvidenceDatasetReused => write!(formatter, "evidence dataset was already used for this hypothesis"),
            Self::NoWorthwhileExperiment => write!(formatter, "no eligible experiment has positive risk-adjusted information value"),
            Self::CandidateNotFound => write!(formatter, "selected multi-outcome candidate was not found"),
            Self::InvalidActionMapping => write!(formatter, "action mappings contain an empty or duplicate experiment ID"),
            Self::MissingActionMapping => write!(formatter, "selected experiment has no caller-supplied action mapping"),
            Self::NoExecutedAction => write!(formatter, "runtime returned no executed action"),
            Self::ExperimentIncomplete => write!(formatter, "environment did not complete the selected experiment action"),
            Self::Selection(message) => write!(formatter, "multi-outcome selection failed: {message}"),
            Self::Runtime(error) => write!(formatter, "multi-outcome runtime failed: {error}"),
            Self::Reasoning(error) => write!(formatter, "multi-outcome evidence update failed: {error}"),
        }
    }
}

impl From<ReasoningError> for MultiOutcomeExperimentError {
    fn from(error: ReasoningError) -> Self {
        Self::Reasoning(error)
    }
}

impl std::error::Error for MultiOutcomeExperimentError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{
        ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionRisk,
    };
    use crate::environment::{
        ActionOutcome, EnvironmentDescriptor, EnvironmentError, ObservationFrame,
    };
    use crate::learning::TransitionSample;
    use crate::planning::PlanningConfig;
    use crate::reasoning::{HypothesisSpec, HypothesisStatus};
    use crate::schema::{FieldSpec, SchemaDefinition, ValueKind};
    use crate::world_model::{ActionDeltaModel, PredictiveModel};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn path(kind: &str) -> std::path::PathBuf {
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "5413-multi-outcome-{}-{kind}-{id}.bin",
            std::process::id()
        ))
    }

    fn candidate() -> MultiOutcomeExperimentCandidate {
        MultiOutcomeExperimentCandidate {
            id: "probe-advance".to_owned(),
            specification: DataValue::Text("apply one reversible advance action".to_owned()),
            outcomes: vec![
                OutcomeLikelihood {
                    outcome_id: "increase".to_owned(),
                    probability_if_true: 0.9,
                    probability_if_false: 0.1,
                },
                OutcomeLikelihood {
                    outcome_id: "unchanged".to_owned(),
                    probability_if_true: 0.1,
                    probability_if_false: 0.9,
                },
            ],
            resource_cost: 1.0,
            risk: 0.01,
            irreversible: false,
        }
    }

    #[test]
    fn categorical_information_gain_distinguishes_informative_and_uninformative_experiments() {
        let hypothesis = HypothesisRecord {
            spec: crate::reasoning::HypothesisSpec {
                id: "increase".to_owned(),
                claim: DataValue::Text("advance increases the state".to_owned()),
                description: "advance changes state".to_owned(),
                prior_probability: 0.5,
                source_dataset_id: None,
            },
            evidence: Vec::new(),
            status: HypothesisStatus::Candidate,
            posterior_probability: 0.5,
        };
        let informative = candidate();
        let uninformative = MultiOutcomeExperimentCandidate {
            id: "unrelated-probe".to_owned(),
            specification: DataValue::Null,
            outcomes: vec![
                OutcomeLikelihood {
                    outcome_id: "one".to_owned(),
                    probability_if_true: 0.5,
                    probability_if_false: 0.5,
                },
                OutcomeLikelihood {
                    outcome_id: "two".to_owned(),
                    probability_if_true: 0.5,
                    probability_if_false: 0.5,
                },
            ],
            resource_cost: 1.0,
            risk: 0.01,
            irreversible: false,
        };
        let ranked = rank_multi_outcome_experiments(
            &hypothesis,
            &[uninformative, informative],
            MultiOutcomeSelectionPolicy {
                maximum_resource_cost: 5.0,
                maximum_risk: 0.1,
                resource_cost_weight: 0.0,
                risk_weight: 0.0,
                allow_irreversible: false,
            },
        )
        .expect("rank");
        assert_eq!(ranked[0].experiment_id, "probe-advance");
        assert!(ranked[0].expected_information_gain > 0.0);
        assert_eq!(ranked[1].expected_information_gain, 0.0);
    }

    struct CounterEnvironment {
        descriptor: EnvironmentDescriptor,
        position: f64,
        sequence: u64,
        executions: usize,
    }

    impl CounterEnvironment {
        fn new() -> Self {
            Self {
                descriptor: EnvironmentDescriptor {
                    id: "multi-outcome-counter".to_owned(),
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
                executions: 0,
            }
        }

        fn frame(&mut self) -> ObservationFrame {
            let frame = ObservationFrame {
                environment_id: "multi-outcome-counter".to_owned(),
                session_id: "multi-outcome-session".to_owned(),
                episode_id: "multi-outcome-episode".to_owned(),
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

    impl EnvironmentAdapter for CounterEnvironment {
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
    }

    fn active_model() -> EnsembleWorldModel {
        let mut model = ActionDeltaModel::new("multi-outcome-model", 1).expect("model");
        model
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
                            DataValue::Number(2.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(3.0))]),
                    },
                ],
            )
            .expect("fit");
        EnsembleWorldModel::new(vec![Box::new(model) as Box<dyn PredictiveModel>])
            .expect("ensemble")
    }

    #[test]
    fn executes_selected_intervention_and_persists_categorical_evidence() {
        let journal_path = path("journal");
        let artifact_path = path("artifacts");
        let mut journal = ExperienceJournal::open(&journal_path).expect("journal");
        let mut artifacts = ModelArtifactStore::open(&artifact_path).expect("artifacts");
        let mut adapter = CounterEnvironment::new();
        let mut hypotheses = HypothesisRegistry::default();
        let hypothesis_policy = HypothesisPolicy {
            minimum_evidence_count: 1,
            support_probability: 0.8,
            refutation_probability: 0.2,
        };
        hypotheses
            .insert(
                HypothesisSpec {
                    id: "advance-increases-position".to_owned(),
                    claim: DataValue::Text("advance increases position".to_owned()),
                    description: "advance increases position".to_owned(),
                    prior_probability: 0.5,
                    source_dataset_id: None,
                },
                hypothesis_policy,
            )
            .expect("hypothesis");
        let request = MultiOutcomeExperimentRequest {
            hypothesis_id: "advance-increases-position",
            candidates: &[candidate()],
            action_mappings: &[MultiOutcomeActionMapping {
                experiment_id: "probe-advance".to_owned(),
                action: ActionProposal {
                    action_id: "advance".to_owned(),
                    arguments: DataValue::object(std::iter::empty()),
                },
            }],
            selection_policy: MultiOutcomeSelectionPolicy {
                maximum_resource_cost: 5.0,
                maximum_risk: 0.1,
                resource_cost_weight: 0.0,
                risk_weight: 0.0,
                allow_irreversible: false,
            },
            hypothesis_policy,
            config: MultiOutcomeExecutionConfig {
                runtime: RuntimeConfig {
                    execution_id: "multi-outcome-run-1".to_owned(),
                    episode_id: "multi-outcome-episode".to_owned(),
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
                evidence_dataset_id: "multi-outcome-evidence-1".to_owned(),
                evidence_rationale: "categorical outcome observed after a reversible intervention"
                    .to_owned(),
                hypothesis_registry_artifact_ref: "multi-outcome-hypotheses-v1".to_owned(),
            },
        };
        let report = run_multi_outcome_experiment(
            MultiOutcomeExperimentContext {
                adapter: &mut adapter,
                model: &active_model(),
                hypotheses: &mut hypotheses,
                journal: &mut journal,
                artifact_store: &mut artifacts,
            },
            request,
            MultiOutcomeExperimentCallbacks {
                utility: |state| {
                    state
                        .get("position")
                        .and_then(DataValue::as_f64)
                        .unwrap_or(0.0)
                },
                classify_outcome: |state| {
                    if state
                        .get("position")
                        .and_then(DataValue::as_f64)
                        .unwrap_or(0.0)
                        > 0.0
                    {
                        Ok("increase".to_owned())
                    } else {
                        Ok("unchanged".to_owned())
                    }
                },
            },
        )
        .expect("experiment");
        assert_eq!(report.selected.experiment_id, "probe-advance");
        assert_eq!(report.runtime.steps.len(), 1);
        assert!(matches!(
            report.evidence_commit,
            MultiOutcomeEvidenceCommitOutcome::Persisted { outcome_id, .. } if outcome_id == "increase"
        ));
        let restored = artifacts
            .load_hypothesis_registry("multi-outcome-hypotheses-v1", hypothesis_policy)
            .expect("restore hypothesis");
        assert_eq!(restored, hypotheses);
        drop(artifacts);
        drop(journal);
        std::fs::remove_file(journal_path).expect("journal cleanup");
        std::fs::remove_file(artifact_path).expect("artifact cleanup");
    }

    #[test]
    fn unknown_outcome_preserves_runtime_without_fabricating_evidence() {
        let journal_path = path("unknown-journal");
        let artifact_path = path("unknown-artifacts");
        let mut journal = ExperienceJournal::open(&journal_path).expect("journal");
        let mut artifacts = ModelArtifactStore::open(&artifact_path).expect("artifacts");
        let mut adapter = CounterEnvironment::new();
        let mut hypotheses = HypothesisRegistry::default();
        let hypothesis_policy = HypothesisPolicy {
            minimum_evidence_count: 1,
            support_probability: 0.8,
            refutation_probability: 0.2,
        };
        hypotheses
            .insert(
                HypothesisSpec {
                    id: "unknown-outcome-hypothesis".to_owned(),
                    claim: DataValue::Bool(true),
                    description: "test".to_owned(),
                    prior_probability: 0.5,
                    source_dataset_id: None,
                },
                hypothesis_policy,
            )
            .expect("hypothesis");
        let candidate = candidate();
        let request = MultiOutcomeExperimentRequest {
            hypothesis_id: "unknown-outcome-hypothesis",
            candidates: &[candidate],
            action_mappings: &[MultiOutcomeActionMapping {
                experiment_id: "probe-advance".to_owned(),
                action: ActionProposal {
                    action_id: "advance".to_owned(),
                    arguments: DataValue::object(std::iter::empty()),
                },
            }],
            selection_policy: MultiOutcomeSelectionPolicy {
                maximum_resource_cost: 5.0,
                maximum_risk: 0.1,
                resource_cost_weight: 0.0,
                risk_weight: 0.0,
                allow_irreversible: false,
            },
            hypothesis_policy,
            config: MultiOutcomeExecutionConfig {
                runtime: RuntimeConfig {
                    execution_id: "multi-outcome-run-unknown".to_owned(),
                    episode_id: "multi-outcome-episode".to_owned(),
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
                evidence_dataset_id: "multi-outcome-unknown-evidence".to_owned(),
                evidence_rationale: "test".to_owned(),
                hypothesis_registry_artifact_ref: "multi-outcome-unknown-hypotheses".to_owned(),
            },
        };
        let report = run_multi_outcome_experiment(
            MultiOutcomeExperimentContext {
                adapter: &mut adapter,
                model: &active_model(),
                hypotheses: &mut hypotheses,
                journal: &mut journal,
                artifact_store: &mut artifacts,
            },
            request,
            MultiOutcomeExperimentCallbacks {
                utility: |_| 0.0,
                classify_outcome: |_| Ok("unmodeled".to_owned()),
            },
        )
        .expect("experiment");
        assert_eq!(report.runtime.steps.len(), 1);
        assert!(matches!(
            report.evidence_commit,
            MultiOutcomeEvidenceCommitOutcome::UnknownObservedOutcome { outcome_id } if outcome_id == "unmodeled"
        ));
        assert!(hypotheses
            .get("unknown-outcome-hypothesis")
            .unwrap()
            .evidence
            .is_empty());
        drop(artifacts);
        drop(journal);
        std::fs::remove_file(journal_path).expect("journal cleanup");
        std::fs::remove_file(artifact_path).expect("artifact cleanup");
    }
}
