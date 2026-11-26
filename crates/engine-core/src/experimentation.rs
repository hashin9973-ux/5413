use crate::action::{ActionAuthorizationPolicy, ActionAuthorizer, ActionProposal, ActionRisk};
use crate::environment::EnvironmentDescriptor;
use crate::value::DataValue;
use crate::world_model::{EnsembleWorldModel, WorldModelError};

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentCandidate {
    pub proposal: ActionProposal,
    /// Caller-supplied task utility estimate, independent of information gain.
    pub expected_utility: f64,
    /// Estimated resource cost in caller-defined, consistent units.
    pub estimated_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExperimentPolicy {
    pub maximum_risk: ActionRisk,
    pub information_gain_weight: f64,
    pub utility_weight: f64,
    pub cost_weight: f64,
    pub risk_weight: f64,
}

impl Default for ExperimentPolicy {
    fn default() -> Self {
        Self {
            maximum_risk: ActionRisk::Reversible,
            information_gain_weight: 1.0,
            utility_weight: 1.0,
            cost_weight: 0.1,
            risk_weight: 1.0,
        }
    }
}

impl ExperimentPolicy {
    pub fn validate(&self) -> Result<(), ExperimentError> {
        let weights = [
            self.information_gain_weight,
            self.utility_weight,
            self.cost_weight,
            self.risk_weight,
        ];
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(ExperimentError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentScore {
    pub proposal: ActionProposal,
    /// Disagreement-based information proxy, not a Bayesian expected information gain.
    pub information_gain_proxy: f64,
    pub epistemic_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
    pub expected_utility: f64,
    pub estimated_cost: f64,
    pub risk: ActionRisk,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentRanking {
    pub ranked: Vec<ExperimentScore>,
    pub rejected_candidates: usize,
}

/// Ranks candidate interventions by ensemble disagreement, estimated task utility,
/// resource cost, and declared action risk. Invalid or unauthorized proposals are
/// counted and omitted; ranking never grants execution permission.
pub fn rank_experiments(
    environment: &EnvironmentDescriptor,
    current_state: &DataValue,
    candidates: &[ExperimentCandidate],
    model: &EnsembleWorldModel,
    policy: ExperimentPolicy,
) -> Result<ExperimentRanking, ExperimentError> {
    policy.validate()?;
    environment
        .validate()
        .map_err(|error| ExperimentError::InvalidEnvironment(error.to_string()))?;
    current_state
        .validate_finite()
        .map_err(|_| ExperimentError::InvalidState)?;

    let authorizer = ActionAuthorizer::new(ActionAuthorizationPolicy::new(policy.maximum_risk));
    let mut ranked = Vec::new();
    let mut rejected_candidates = 0usize;

    for candidate in candidates {
        if !candidate.expected_utility.is_finite()
            || !candidate.estimated_cost.is_finite()
            || candidate.estimated_cost < 0.0
        {
            rejected_candidates = rejected_candidates
                .checked_add(1)
                .ok_or(ExperimentError::CounterOverflow)?;
            continue;
        }

        let authorized = match authorizer.authorize(environment, candidate.proposal.clone()) {
            Ok(authorized) => authorized,
            Err(_) => {
                rejected_candidates = rejected_candidates
                    .checked_add(1)
                    .ok_or(ExperimentError::CounterOverflow)?;
                continue;
            }
        };

        let prediction = model
            .predict(&candidate.proposal.action_id, current_state)
            .map_err(ExperimentError::WorldModel)?;
        if prediction.fields.is_empty() {
            rejected_candidates = rejected_candidates
                .checked_add(1)
                .ok_or(ExperimentError::CounterOverflow)?;
            continue;
        }

        let mut epistemic_variance = 0.0;
        let mut aleatoric_variance = 0.0;
        for estimate in prediction.fields.values() {
            epistemic_variance += estimate.epistemic_std_dev.powi(2);
            aleatoric_variance += estimate.aleatoric_std_dev.powi(2);
        }
        let field_count = prediction.fields.len() as f64;
        let epistemic_uncertainty = (epistemic_variance / field_count).sqrt();
        let aleatoric_uncertainty = (aleatoric_variance / field_count).sqrt();
        // A bounded, dimensionless proxy that grows with epistemic disagreement
        // relative to observed outcome noise. It is an action-ranking heuristic.
        let information_gain_proxy =
            0.5 * (epistemic_variance / (aleatoric_variance + 1e-9) + 1.0).ln();

        let risk = authorized.risk();
        let risk_cost = match risk {
            ActionRisk::ReadOnly => 0.0,
            ActionRisk::Reversible => 1.0,
            ActionRisk::Irreversible => 2.0,
        };
        let score = policy.information_gain_weight * information_gain_proxy
            + policy.utility_weight * candidate.expected_utility
            - policy.cost_weight * candidate.estimated_cost
            - policy.risk_weight * risk_cost;
        if !score.is_finite()
            || !information_gain_proxy.is_finite()
            || !epistemic_uncertainty.is_finite()
            || !aleatoric_uncertainty.is_finite()
        {
            return Err(ExperimentError::NonFiniteScore);
        }

        ranked.push(ExperimentScore {
            proposal: candidate.proposal.clone(),
            information_gain_proxy,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            expected_utility: candidate.expected_utility,
            estimated_cost: candidate.estimated_cost,
            risk,
            score,
        });
    }

    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.proposal.action_id.cmp(&right.proposal.action_id))
    });
    Ok(ExperimentRanking {
        ranked,
        rejected_candidates,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExperimentError {
    InvalidPolicy,
    InvalidEnvironment(String),
    InvalidState,
    NonFiniteScore,
    CounterOverflow,
    WorldModel(WorldModelError),
}

impl std::fmt::Display for ExperimentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPolicy => write!(
                formatter,
                "experiment policy weights must be finite and non-negative"
            ),
            Self::InvalidEnvironment(message) => {
                write!(formatter, "invalid environment: {message}")
            }
            Self::InvalidState => write!(formatter, "experiment state must contain finite values"),
            Self::NonFiniteScore => write!(formatter, "experiment score is non-finite"),
            Self::CounterOverflow => write!(formatter, "experiment rejection counter is exhausted"),
            Self::WorldModel(error) => write!(formatter, "world-model prediction failed: {error}"),
        }
    }
}

impl std::error::Error for ExperimentError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{ActionDescriptor, ActionRisk};
    use crate::learning::TransitionSample;
    use crate::schema::SchemaDefinition;
    use crate::world_model::{ActionDeltaModel, EnsembleWorldModel, PredictiveModel};

    fn descriptor() -> EnvironmentDescriptor {
        EnvironmentDescriptor {
            id: "experiment-world".to_owned(),
            version: "1".to_owned(),
            observation_schemas: Vec::new(),
            actions: vec![
                ActionDescriptor {
                    id: "probe".to_owned(),
                    arguments: SchemaDefinition {
                        id: "probe-args".to_owned(),
                        version: 1,
                        fields: Vec::new(),
                        allow_additional_fields: false,
                    },
                    risk: ActionRisk::ReadOnly,
                },
                ActionDescriptor {
                    id: "mutate".to_owned(),
                    arguments: SchemaDefinition {
                        id: "mutate-args".to_owned(),
                        version: 1,
                        fields: Vec::new(),
                        allow_additional_fields: false,
                    },
                    risk: ActionRisk::Reversible,
                },
                ActionDescriptor {
                    id: "erase".to_owned(),
                    arguments: SchemaDefinition {
                        id: "erase-args".to_owned(),
                        version: 1,
                        fields: Vec::new(),
                        allow_additional_fields: false,
                    },
                    risk: ActionRisk::Irreversible,
                },
            ],
            supports_reset: false,
        }
    }

    #[test]
    fn ranks_authorized_candidates_and_rejects_irreversible_actions() {
        let mut first = ActionDeltaModel::new("model-a", 1).expect("model");
        first
            .fit(
                "train-a",
                &[
                    TransitionSample {
                        action_id: "probe".to_owned(),
                        before: DataValue::object([("value".to_owned(), DataValue::Number(0.0))]),
                        after: DataValue::object([("value".to_owned(), DataValue::Number(1.0))]),
                    },
                    TransitionSample {
                        action_id: "probe".to_owned(),
                        before: DataValue::object([("value".to_owned(), DataValue::Number(1.0))]),
                        after: DataValue::object([("value".to_owned(), DataValue::Number(2.0))]),
                    },
                ],
            )
            .expect("fit");
        let mut second = ActionDeltaModel::new("model-b", 1).expect("model");
        second
            .fit(
                "train-b",
                &[
                    TransitionSample {
                        action_id: "probe".to_owned(),
                        before: DataValue::object([("value".to_owned(), DataValue::Number(0.0))]),
                        after: DataValue::object([("value".to_owned(), DataValue::Number(3.0))]),
                    },
                    TransitionSample {
                        action_id: "probe".to_owned(),
                        before: DataValue::object([("value".to_owned(), DataValue::Number(1.0))]),
                        after: DataValue::object([("value".to_owned(), DataValue::Number(4.0))]),
                    },
                ],
            )
            .expect("fit");
        let ensemble =
            EnsembleWorldModel::new(vec![Box::new(first), Box::new(second)]).expect("ensemble");

        let ranking = rank_experiments(
            &descriptor(),
            &DataValue::object([("value".to_owned(), DataValue::Number(5.0))]),
            &[
                ExperimentCandidate {
                    proposal: ActionProposal {
                        action_id: "probe".to_owned(),
                        arguments: DataValue::object(std::iter::empty()),
                    },
                    expected_utility: 0.0,
                    estimated_cost: 1.0,
                },
                ExperimentCandidate {
                    proposal: ActionProposal {
                        action_id: "erase".to_owned(),
                        arguments: DataValue::object(std::iter::empty()),
                    },
                    expected_utility: 100.0,
                    estimated_cost: 0.0,
                },
            ],
            &ensemble,
            ExperimentPolicy::default(),
        )
        .expect("ranking");

        assert_eq!(ranking.ranked.len(), 1);
        assert_eq!(ranking.ranked[0].proposal.action_id, "probe");
        assert_eq!(ranking.rejected_candidates, 1);
        assert!(ranking.ranked[0].information_gain_proxy > 0.0);
    }

    #[test]
    fn rejects_invalid_policy_weights() {
        assert_eq!(
            ExperimentPolicy {
                information_gain_weight: f64::NAN,
                ..ExperimentPolicy::default()
            }
            .validate(),
            Err(ExperimentError::InvalidPolicy)
        );
    }
}
