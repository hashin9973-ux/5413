//! Expected-information-gain ranking for binary-outcome experiments.
//!
//! The selector is Bayesian and explicit: callers supply P(observation | H)
//! and P(observation | not-H). It ranks informative experiments while applying
//! hard resource/risk/irreversibility constraints. It does not execute actions.

use crate::action::{ActionAuthorizer, ActionProposal, ActionRisk};
use crate::environment::EnvironmentDescriptor;
use crate::reasoning::HypothesisRecord;
use crate::value::DataValue;
use crate::world_model::{EnsembleWorldModel, WorldModelError};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentCandidate {
    pub id: String,
    pub specification: DataValue,
    /// Probability of observing the positive outcome when the hypothesis is true.
    pub observation_probability_if_true: f64,
    /// Probability of observing the positive outcome when the hypothesis is false.
    pub observation_probability_if_false: f64,
    pub resource_cost: f64,
    /// Normalized caller-supplied risk estimate in [0, 1].
    pub risk: f64,
    pub irreversible: bool,
}

impl ExperimentCandidate {
    fn validate(&self) -> Result<(), ExperimentSelectionError> {
        if self.id.trim().is_empty()
            || !self.observation_probability_if_true.is_finite()
            || !(0.0..=1.0).contains(&self.observation_probability_if_true)
            || !self.observation_probability_if_false.is_finite()
            || !(0.0..=1.0).contains(&self.observation_probability_if_false)
            || !self.resource_cost.is_finite()
            || self.resource_cost < 0.0
            || !self.risk.is_finite()
            || !(0.0..=1.0).contains(&self.risk)
        {
            return Err(ExperimentSelectionError::InvalidCandidate);
        }
        self.specification
            .validate_finite()
            .map_err(|_| ExperimentSelectionError::InvalidCandidate)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExperimentSelectionPolicy {
    pub maximum_resource_cost: f64,
    pub maximum_risk: f64,
    pub resource_cost_weight: f64,
    pub risk_weight: f64,
    pub allow_irreversible: bool,
}

impl Default for ExperimentSelectionPolicy {
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

impl ExperimentSelectionPolicy {
    pub fn validate(&self) -> Result<(), ExperimentSelectionError> {
        if !self.maximum_resource_cost.is_finite()
            || self.maximum_resource_cost < 0.0
            || !self.maximum_risk.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_risk)
            || !self.resource_cost_weight.is_finite()
            || self.resource_cost_weight < 0.0
            || !self.risk_weight.is_finite()
            || self.risk_weight < 0.0
        {
            return Err(ExperimentSelectionError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedExperiment {
    pub experiment_id: String,
    pub hypothesis_id: String,
    /// Entropy reduction in nats under the supplied binary likelihood model.
    pub expected_information_gain: f64,
    pub risk_adjusted_utility: f64,
    pub resource_cost: f64,
    pub risk: f64,
    pub irreversible: bool,
}

impl RankedExperiment {
    pub fn is_worthwhile(&self) -> bool {
        self.risk_adjusted_utility > 0.0
    }
}

/// Ranks experiments by expected information gain less declared risk and cost.
///
/// This is exact for the supplied binary observation model, not necessarily for
/// the real environment. Model misspecification and dependent observations can
/// make the information-gain estimate optimistic.
pub fn rank_experiments(
    hypothesis: &HypothesisRecord,
    candidates: &[ExperimentCandidate],
    policy: ExperimentSelectionPolicy,
) -> Result<Vec<RankedExperiment>, ExperimentSelectionError> {
    policy.validate()?;
    let prior = hypothesis.posterior_probability;
    if hypothesis.spec.id.trim().is_empty() || !prior.is_finite() || !(0.0..=1.0).contains(&prior) {
        return Err(ExperimentSelectionError::InvalidHypothesis);
    }

    let mut seen_ids = BTreeSet::new();
    let mut ranked = Vec::new();
    for candidate in candidates {
        candidate.validate()?;
        if !seen_ids.insert(candidate.id.as_str()) {
            return Err(ExperimentSelectionError::DuplicateCandidate);
        }
        if candidate.resource_cost > policy.maximum_resource_cost
            || candidate.risk > policy.maximum_risk
            || (candidate.irreversible && !policy.allow_irreversible)
        {
            continue;
        }

        let information_gain = binary_expected_information_gain(
            prior,
            candidate.observation_probability_if_true,
            candidate.observation_probability_if_false,
        );
        let utility = information_gain
            - policy.resource_cost_weight * candidate.resource_cost
            - policy.risk_weight * candidate.risk;
        if !information_gain.is_finite() || !utility.is_finite() {
            return Err(ExperimentSelectionError::NumericalOverflow);
        }
        ranked.push(RankedExperiment {
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

fn binary_expected_information_gain(
    prior: f64,
    probability_if_true: f64,
    probability_if_false: f64,
) -> f64 {
    let prior_entropy = binary_entropy(prior);
    let probability_positive = prior * probability_if_true + (1.0 - prior) * probability_if_false;
    let posterior_positive = if probability_positive == 0.0 {
        prior
    } else {
        prior * probability_if_true / probability_positive
    };
    let probability_negative = 1.0 - probability_positive;
    let posterior_negative = if probability_negative == 0.0 {
        prior
    } else {
        prior * (1.0 - probability_if_true) / probability_negative
    };
    let expected_posterior_entropy = probability_positive * binary_entropy(posterior_positive)
        + probability_negative * binary_entropy(posterior_negative);
    (prior_entropy - expected_posterior_entropy).max(0.0)
}

fn binary_entropy(probability: f64) -> f64 {
    if probability <= 0.0 || probability >= 1.0 {
        0.0
    } else {
        -probability * probability.ln() - (1.0 - probability) * (1.0 - probability).ln()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelExplorationCandidate {
    pub id: String,
    pub action: ActionProposal,
    pub resource_cost: f64,
}

impl ModelExplorationCandidate {
    fn validate(&self) -> Result<(), ExperimentSelectionError> {
        if self.id.trim().is_empty()
            || self.action.action_id.trim().is_empty()
            || !self.resource_cost.is_finite()
            || self.resource_cost < 0.0
        {
            return Err(ExperimentSelectionError::InvalidCandidate);
        }
        self.action
            .arguments
            .validate_finite()
            .map_err(|_| ExperimentSelectionError::InvalidCandidate)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelExplorationPolicy {
    pub maximum_resource_cost: f64,
    pub maximum_action_risk: ActionRisk,
    pub task_utility_weight: f64,
    pub disagreement_weight: f64,
    pub resource_cost_weight: f64,
    pub risk_weight: f64,
}

impl Default for ModelExplorationPolicy {
    fn default() -> Self {
        Self {
            maximum_resource_cost: 100.0,
            maximum_action_risk: ActionRisk::Reversible,
            task_utility_weight: 1.0,
            disagreement_weight: 1.0,
            resource_cost_weight: 0.01,
            risk_weight: 1.0,
        }
    }
}

impl ModelExplorationPolicy {
    pub fn validate(&self) -> Result<(), ExperimentSelectionError> {
        if !self.maximum_resource_cost.is_finite()
            || self.maximum_resource_cost < 0.0
            || [
                self.task_utility_weight,
                self.disagreement_weight,
                self.resource_cost_weight,
                self.risk_weight,
            ]
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
            || (self.task_utility_weight == 0.0 && self.disagreement_weight == 0.0)
        {
            return Err(ExperimentSelectionError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedExplorationAction {
    pub candidate_id: String,
    pub action: ActionProposal,
    pub predicted_state: DataValue,
    pub predicted_task_utility: f64,
    /// Root-sum-square of ensemble epistemic standard deviations. This is a
    /// scale-dependent disagreement proxy, not a calibrated information gain.
    pub model_disagreement: f64,
    pub resource_cost: f64,
    pub risk: ActionRisk,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelExplorationRanking {
    pub ranked_actions: Vec<RankedExplorationAction>,
    pub rejected_candidates: usize,
}

/// Ranks one-step action proposals by task utility plus ensemble disagreement.
///
/// Only ensembles with at least two models are accepted. The method validates
/// every candidate through the environment descriptor and authorizer, applies
/// hard cost/risk limits, and then ranks predicted utility plus an epistemic
/// disagreement proxy, less declared resource and risk penalties.
pub fn rank_model_disagreement_actions<F>(
    environment: &EnvironmentDescriptor,
    authorizer: ActionAuthorizer,
    model: &EnsembleWorldModel,
    current_state: &DataValue,
    candidates: &[ModelExplorationCandidate],
    policy: ModelExplorationPolicy,
    utility: F,
) -> Result<ModelExplorationRanking, ExperimentSelectionError>
where
    F: Fn(&DataValue) -> f64,
{
    policy.validate()?;
    environment
        .validate()
        .map_err(|_| ExperimentSelectionError::InvalidEnvironment)?;
    current_state
        .validate_finite()
        .map_err(|_| ExperimentSelectionError::InvalidState)?;
    let current_fields = current_state
        .as_object()
        .ok_or(ExperimentSelectionError::InvalidState)?;
    if model.len() < 2 {
        return Err(ExperimentSelectionError::InsufficientEnsembleDiversity);
    }
    if candidates.is_empty() {
        return Err(ExperimentSelectionError::NoEligibleActions);
    }

    let mut seen_ids = BTreeSet::new();
    let mut ranked_actions = Vec::new();
    let mut rejected_candidates = 0usize;
    for candidate in candidates {
        candidate.validate()?;
        if !seen_ids.insert(candidate.id.as_str()) {
            return Err(ExperimentSelectionError::DuplicateCandidate);
        }
        if candidate.resource_cost > policy.maximum_resource_cost {
            rejected_candidates = rejected_candidates
                .checked_add(1)
                .ok_or(ExperimentSelectionError::CounterOverflow)?;
            continue;
        }
        let Some(descriptor) = environment
            .actions
            .iter()
            .find(|action| action.id == candidate.action.action_id)
        else {
            rejected_candidates = rejected_candidates
                .checked_add(1)
                .ok_or(ExperimentSelectionError::CounterOverflow)?;
            continue;
        };
        if descriptor.risk > policy.maximum_action_risk
            || authorizer
                .authorize(environment, candidate.action.clone())
                .is_err()
        {
            rejected_candidates = rejected_candidates
                .checked_add(1)
                .ok_or(ExperimentSelectionError::CounterOverflow)?;
            continue;
        }

        let prediction = model
            .predict(&candidate.action.action_id, current_state)
            .map_err(ExperimentSelectionError::WorldModel)?;
        if prediction.fields.is_empty() {
            rejected_candidates = rejected_candidates
                .checked_add(1)
                .ok_or(ExperimentSelectionError::CounterOverflow)?;
            continue;
        }

        let mut predicted_fields = current_fields.clone();
        let mut disagreement_squared = 0.0;
        for (field, estimate) in prediction.fields {
            if !estimate.mean.is_finite()
                || !estimate.epistemic_std_dev.is_finite()
                || estimate.epistemic_std_dev < 0.0
            {
                return Err(ExperimentSelectionError::NumericalOverflow);
            }
            predicted_fields.insert(field, DataValue::Number(estimate.mean));
            disagreement_squared += estimate.epistemic_std_dev.powi(2);
            if !disagreement_squared.is_finite() {
                return Err(ExperimentSelectionError::NumericalOverflow);
            }
        }
        let predicted_state = DataValue::Object(predicted_fields);
        let predicted_task_utility = utility(&predicted_state);
        let model_disagreement = disagreement_squared.sqrt();
        if !predicted_task_utility.is_finite() || !model_disagreement.is_finite() {
            return Err(ExperimentSelectionError::NumericalOverflow);
        }
        let risk = descriptor.risk;
        let normalized_risk = match risk {
            ActionRisk::ReadOnly => 0.0,
            ActionRisk::Reversible => 0.5,
            ActionRisk::Irreversible => 1.0,
        };
        let score = policy.task_utility_weight * predicted_task_utility
            + policy.disagreement_weight * model_disagreement
            - policy.resource_cost_weight * candidate.resource_cost
            - policy.risk_weight * normalized_risk;
        if !score.is_finite() {
            return Err(ExperimentSelectionError::NumericalOverflow);
        }
        ranked_actions.push(RankedExplorationAction {
            candidate_id: candidate.id.clone(),
            action: candidate.action.clone(),
            predicted_state,
            predicted_task_utility,
            model_disagreement,
            resource_cost: candidate.resource_cost,
            risk,
            score,
        });
    }

    ranked_actions.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    if ranked_actions.is_empty() {
        return Err(ExperimentSelectionError::NoEligibleActions);
    }
    Ok(ModelExplorationRanking {
        ranked_actions,
        rejected_candidates,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExperimentSelectionError {
    InvalidCandidate,
    InvalidPolicy,
    InvalidHypothesis,
    DuplicateCandidate,
    InvalidEnvironment,
    InvalidState,
    InsufficientEnsembleDiversity,
    NoEligibleActions,
    CounterOverflow,
    WorldModel(WorldModelError),
    NumericalOverflow,
}

impl std::fmt::Display for ExperimentSelectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCandidate => write!(formatter, "experiment candidate is invalid"),
            Self::InvalidPolicy => write!(formatter, "experiment selection policy is invalid"),
            Self::InvalidHypothesis => write!(formatter, "hypothesis posterior is invalid"),
            Self::DuplicateCandidate => write!(formatter, "experiment candidate ID is duplicated"),
            Self::InvalidEnvironment => write!(formatter, "environment descriptor is invalid"),
            Self::InvalidState => {
                write!(formatter, "model-exploration state must be a finite object")
            }
            Self::InsufficientEnsembleDiversity => write!(
                formatter,
                "model-disagreement exploration requires at least two predictive models"
            ),
            Self::NoEligibleActions => write!(
                formatter,
                "no model-exploration action passed prediction and safety constraints"
            ),
            Self::CounterOverflow => write!(formatter, "rejected-candidate counter overflowed"),
            Self::WorldModel(error) => {
                write!(formatter, "model-disagreement prediction failed: {error}")
            }
            Self::NumericalOverflow => write!(
                formatter,
                "experiment information-gain calculation overflowed"
            ),
        }
    }
}

impl std::error::Error for ExperimentSelectionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning::{HypothesisSpec, HypothesisStatus};
    use crate::world_model::PredictiveModel;

    fn hypothesis(probability: f64) -> HypothesisRecord {
        HypothesisRecord {
            spec: HypothesisSpec {
                id: "hypothesis-1".to_owned(),
                claim: DataValue::Text("x increases y".to_owned()),
                description: "test hypothesis".to_owned(),
                prior_probability: 0.5,
                source_dataset_id: None,
            },
            evidence: Vec::new(),
            status: HypothesisStatus::Candidate,
            posterior_probability: probability,
        }
    }

    fn candidate(id: &str, if_true: f64, if_false: f64) -> ExperimentCandidate {
        ExperimentCandidate {
            id: id.to_owned(),
            specification: DataValue::object([(
                "action".to_owned(),
                DataValue::Text("vary-x".to_owned()),
            )]),
            observation_probability_if_true: if_true,
            observation_probability_if_false: if_false,
            resource_cost: 1.0,
            risk: 0.01,
            irreversible: false,
        }
    }

    fn exploration_environment() -> EnvironmentDescriptor {
        use crate::action::{ActionDescriptor, ActionRisk};
        use crate::schema::SchemaDefinition;

        let empty_arguments = || SchemaDefinition {
            id: "empty-arguments".to_owned(),
            version: 1,
            fields: Vec::new(),
            allow_additional_fields: false,
        };
        EnvironmentDescriptor {
            id: "exploration-counter".to_owned(),
            version: "v1".to_owned(),
            observation_schemas: Vec::new(),
            actions: vec![
                ActionDescriptor {
                    id: "advance".to_owned(),
                    arguments: empty_arguments(),
                    risk: ActionRisk::Reversible,
                },
                ActionDescriptor {
                    id: "wait".to_owned(),
                    arguments: empty_arguments(),
                    risk: ActionRisk::ReadOnly,
                },
            ],
            supports_reset: false,
        }
    }

    fn exploration_model() -> EnsembleWorldModel {
        use crate::learning::TransitionSample;
        use crate::world_model::{ActionDeltaModel, PredictiveModel};

        let mut first = ActionDeltaModel::new("explore-model-a", 1).expect("first model");
        first
            .fit(
                "explore-train-a",
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
                            DataValue::Number(10.0),
                        )]),
                        after: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(11.0),
                        )]),
                    },
                    TransitionSample {
                        action_id: "wait".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(0.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(0.0))]),
                    },
                    TransitionSample {
                        action_id: "wait".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(10.0),
                        )]),
                        after: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(10.0),
                        )]),
                    },
                ],
            )
            .expect("fit first model");

        let mut second = ActionDeltaModel::new("explore-model-b", 1).expect("second model");
        second
            .fit(
                "explore-train-b",
                &[
                    TransitionSample {
                        action_id: "advance".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(0.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(3.0))]),
                    },
                    TransitionSample {
                        action_id: "advance".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(10.0),
                        )]),
                        after: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(13.0),
                        )]),
                    },
                    TransitionSample {
                        action_id: "wait".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(0.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(0.0))]),
                    },
                    TransitionSample {
                        action_id: "wait".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(10.0),
                        )]),
                        after: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(10.0),
                        )]),
                    },
                ],
            )
            .expect("fit second model");
        EnsembleWorldModel::new(vec![
            Box::new(first) as Box<dyn PredictiveModel>,
            Box::new(second) as Box<dyn PredictiveModel>,
        ])
        .expect("ensemble")
    }

    #[test]
    fn model_disagreement_ranking_prefers_uncertain_actions_under_safety_limits() {
        let environment = exploration_environment();
        let model = exploration_model();
        let state = DataValue::object([("position".to_owned(), DataValue::Number(0.0))]);
        let candidates = vec![
            ModelExplorationCandidate {
                id: "wait".to_owned(),
                action: ActionProposal {
                    action_id: "wait".to_owned(),
                    arguments: DataValue::object(std::iter::empty()),
                },
                resource_cost: 0.0,
            },
            ModelExplorationCandidate {
                id: "advance".to_owned(),
                action: ActionProposal {
                    action_id: "advance".to_owned(),
                    arguments: DataValue::object(std::iter::empty()),
                },
                resource_cost: 0.0,
            },
        ];
        let ranking = rank_model_disagreement_actions(
            &environment,
            ActionAuthorizer::new(crate::action::ActionAuthorizationPolicy::new(
                ActionRisk::Reversible,
            )),
            &model,
            &state,
            &candidates,
            ModelExplorationPolicy {
                maximum_resource_cost: 1.0,
                maximum_action_risk: ActionRisk::Reversible,
                task_utility_weight: 0.0,
                disagreement_weight: 1.0,
                resource_cost_weight: 0.0,
                risk_weight: 0.0,
            },
            |_| 0.0,
        )
        .expect("rank exploration actions");
        assert_eq!(ranking.ranked_actions[0].candidate_id, "advance");
        assert!(ranking.ranked_actions[0].model_disagreement > 0.0);
        assert_eq!(ranking.ranked_actions[1].candidate_id, "wait");
    }

    #[test]
    fn model_disagreement_ranking_rejects_single_model_and_overrisk_actions() {
        let environment = exploration_environment();
        let state = DataValue::object([("position".to_owned(), DataValue::Number(0.0))]);
        let candidate = ModelExplorationCandidate {
            id: "advance".to_owned(),
            action: ActionProposal {
                action_id: "advance".to_owned(),
                arguments: DataValue::object(std::iter::empty()),
            },
            resource_cost: 0.0,
        };
        let mut only = crate::world_model::ActionDeltaModel::new("only-model", 1).expect("model");
        only.fit(
            "only-training",
            &[
                crate::learning::TransitionSample {
                    action_id: "advance".to_owned(),
                    before: state.clone(),
                    after: DataValue::object([("position".to_owned(), DataValue::Number(1.0))]),
                },
                crate::learning::TransitionSample {
                    action_id: "advance".to_owned(),
                    before: state.clone(),
                    after: DataValue::object([("position".to_owned(), DataValue::Number(2.0))]),
                },
            ],
        )
        .expect("fit");
        let one_model = EnsembleWorldModel::new(vec![
            Box::new(only) as Box<dyn crate::world_model::PredictiveModel>
        ])
        .expect("single model ensemble");
        assert_eq!(
            rank_model_disagreement_actions(
                &environment,
                ActionAuthorizer::new(crate::action::ActionAuthorizationPolicy::new(
                    ActionRisk::Reversible
                )),
                &one_model,
                &state,
                std::slice::from_ref(&candidate),
                ModelExplorationPolicy::default(),
                |_| 0.0,
            ),
            Err(ExperimentSelectionError::InsufficientEnsembleDiversity)
        );

        let mut irreversible_environment = environment;
        irreversible_environment
            .actions
            .push(crate::action::ActionDescriptor {
                id: "delete".to_owned(),
                arguments: crate::schema::SchemaDefinition {
                    id: "empty-delete".to_owned(),
                    version: 1,
                    fields: Vec::new(),
                    allow_additional_fields: false,
                },
                risk: ActionRisk::Irreversible,
            });
        let delete = ModelExplorationCandidate {
            id: "delete".to_owned(),
            action: ActionProposal {
                action_id: "delete".to_owned(),
                arguments: DataValue::object(std::iter::empty()),
            },
            resource_cost: 0.0,
        };
        assert_eq!(
            rank_model_disagreement_actions(
                &irreversible_environment,
                ActionAuthorizer::new(crate::action::ActionAuthorizationPolicy::new(
                    ActionRisk::Irreversible
                )),
                &exploration_model(),
                &state,
                &[delete],
                ModelExplorationPolicy {
                    maximum_action_risk: ActionRisk::Reversible,
                    ..ModelExplorationPolicy::default()
                },
                |_| 0.0,
            ),
            Err(ExperimentSelectionError::NoEligibleActions)
        );
    }

    #[test]
    fn prefers_experiments_with_high_expected_information_gain() {
        let ranked = rank_experiments(
            &hypothesis(0.5),
            &[
                candidate("uninformative", 0.5, 0.5),
                candidate("discriminating", 0.95, 0.05),
            ],
            ExperimentSelectionPolicy {
                maximum_resource_cost: 10.0,
                maximum_risk: 0.5,
                resource_cost_weight: 0.0,
                risk_weight: 0.0,
                allow_irreversible: false,
            },
        )
        .expect("rank");
        assert_eq!(ranked[0].experiment_id, "discriminating");
        assert!(ranked[0].expected_information_gain > ranked[1].expected_information_gain);
        assert!(ranked[0].is_worthwhile());
    }

    #[test]
    fn constraints_exclude_unsafe_or_over_budget_experiments() {
        let mut unsafe_action = candidate("unsafe", 1.0, 0.0);
        unsafe_action.irreversible = true;
        let mut expensive = candidate("expensive", 1.0, 0.0);
        expensive.resource_cost = 1000.0;
        let ranked = rank_experiments(
            &hypothesis(0.5),
            &[unsafe_action, expensive, candidate("safe", 0.8, 0.2)],
            ExperimentSelectionPolicy::default(),
        )
        .expect("rank");
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].experiment_id, "safe");
    }

    #[test]
    fn rejects_duplicate_ids_and_invalid_likelihoods() {
        assert_eq!(
            rank_experiments(
                &hypothesis(0.5),
                &[
                    candidate("duplicate", 0.9, 0.1),
                    candidate("duplicate", 0.8, 0.2)
                ],
                ExperimentSelectionPolicy::default(),
            ),
            Err(ExperimentSelectionError::DuplicateCandidate)
        );
        let mut invalid = candidate("bad", 0.9, 0.1);
        invalid.observation_probability_if_true = 2.0;
        assert_eq!(
            rank_experiments(
                &hypothesis(0.5),
                &[invalid],
                ExperimentSelectionPolicy::default()
            ),
            Err(ExperimentSelectionError::InvalidCandidate)
        );
    }
}
