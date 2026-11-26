//! Expected-information-gain ranking for binary-outcome experiments.
//!
//! The selector is Bayesian and explicit: callers supply P(observation | H)
//! and P(observation | not-H). It ranks informative experiments while applying
//! hard resource/risk/irreversibility constraints. It does not execute actions.

use crate::reasoning::HypothesisRecord;
use crate::value::DataValue;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExperimentSelectionError {
    InvalidCandidate,
    InvalidPolicy,
    InvalidHypothesis,
    DuplicateCandidate,
    NumericalOverflow,
}

impl std::fmt::Display for ExperimentSelectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCandidate => write!(formatter, "experiment candidate is invalid"),
            Self::InvalidPolicy => write!(formatter, "experiment selection policy is invalid"),
            Self::InvalidHypothesis => write!(formatter, "hypothesis posterior is invalid"),
            Self::DuplicateCandidate => write!(formatter, "experiment candidate ID is duplicated"),
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
