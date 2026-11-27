use crate::action::{ActionAuthorizer, ActionProposal};
use crate::environment::EnvironmentDescriptor;
use crate::regression_metrics::ScaledSumSquares;
use crate::value::DataValue;
use crate::world_model::{EnsembleWorldModel, WorldModelError};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanningConfig {
    pub horizon: usize,
    pub beam_width: usize,
    pub uncertainty_penalty: f64,
    /// Additional penalty for a prediction whose uncertainty is not calibrated.
    pub uncalibrated_prediction_penalty: f64,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            horizon: 3,
            beam_width: 8,
            uncertainty_penalty: 1.0,
            uncalibrated_prediction_penalty: 1.0,
        }
    }
}

impl PlanningConfig {
    pub fn validate(&self) -> Result<(), PlanningError> {
        if self.horizon == 0
            || self.beam_width == 0
            || !self.uncertainty_penalty.is_finite()
            || self.uncertainty_penalty < 0.0
            || !self.uncalibrated_prediction_penalty.is_finite()
            || self.uncalibrated_prediction_penalty < 0.0
        {
            return Err(PlanningError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanStep {
    pub action: ActionProposal,
    pub predicted_state: DataValue,
    pub prediction_uncertainty: f64,
    pub uncertainty_calibrated: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub steps: Vec<PlanStep>,
    pub expected_utility: f64,
    pub total_uncertainty: f64,
    pub objective_score: f64,
    pub expanded_nodes: u64,
    pub rejected_actions: u64,
}

#[derive(Clone)]
struct BeamNode {
    state: DataValue,
    steps: Vec<PlanStep>,
    cumulative_utility: f64,
    cumulative_uncertainty: f64,
}

impl BeamNode {
    fn score(&self, config: PlanningConfig) -> f64 {
        self.cumulative_utility - config.uncertainty_penalty * self.cumulative_uncertainty
    }
}

/// Performs bounded beam search against the learned ensemble.
///
/// The returned proposals are not execution capabilities. The caller must authorize
/// the selected action again immediately before execution; the environment remains
/// the final authority on current state, permissions, and action validity.
pub fn plan<F>(
    environment: &EnvironmentDescriptor,
    authorizer: ActionAuthorizer,
    model: &EnsembleWorldModel,
    current_state: &DataValue,
    candidate_actions: &[ActionProposal],
    config: PlanningConfig,
    utility: F,
) -> Result<Plan, PlanningError>
where
    F: Fn(&DataValue) -> f64,
{
    config.validate()?;
    environment
        .validate()
        .map_err(|error| PlanningError::InvalidEnvironment(error.to_string()))?;
    if candidate_actions.is_empty() {
        return Err(PlanningError::NoCandidateActions);
    }
    current_state
        .validate_finite()
        .map_err(|_| PlanningError::InvalidState)?;

    let mut frontier = vec![BeamNode {
        state: current_state.clone(),
        steps: Vec::new(),
        cumulative_utility: 0.0,
        cumulative_uncertainty: 0.0,
    }];
    let mut expanded_nodes = 0u64;
    let mut rejected_actions = 0u64;

    for _ in 0..config.horizon {
        let mut expanded = Vec::new();
        for node in &frontier {
            for candidate in candidate_actions {
                if authorizer
                    .authorize(environment, candidate.clone())
                    .is_err()
                {
                    rejected_actions = rejected_actions
                        .checked_add(1)
                        .ok_or(PlanningError::CounterOverflow)?;
                    continue;
                }

                let prediction = model
                    .predict(&candidate.action_id, &node.state)
                    .map_err(PlanningError::WorldModel)?;
                if prediction.fields.is_empty() {
                    continue;
                }

                let mut predicted_state = match &node.state {
                    DataValue::Object(fields) => fields.clone(),
                    _ => return Err(PlanningError::InvalidState),
                };
                let mut uncertainty_rms = ScaledSumSquares::default();
                let mut estimate_count = 0usize;
                let mut calibrated = true;
                for (field, estimate) in &prediction.fields {
                    if !estimate.mean.is_finite() {
                        return Err(PlanningError::InvalidPrediction);
                    }
                    predicted_state.insert(field.clone(), DataValue::Number(estimate.mean));
                    let std_dev = estimate.total_std_dev();
                    if !std_dev.is_finite() || !uncertainty_rms.push_nonnegative(std_dev) {
                        return Err(PlanningError::InvalidPrediction);
                    }
                    estimate_count += 1;
                    calibrated &= estimate.uncertainty_calibrated;
                }
                if estimate_count == 0 {
                    continue;
                }
                let mut prediction_uncertainty = uncertainty_rms
                    .root_mean_square()
                    .ok_or(PlanningError::InvalidPrediction)?;
                if !calibrated {
                    prediction_uncertainty += config.uncalibrated_prediction_penalty;
                }
                if !prediction_uncertainty.is_finite() {
                    return Err(PlanningError::InvalidPrediction);
                }

                let predicted_state = DataValue::Object(predicted_state);
                let step_utility = utility(&predicted_state);
                if !step_utility.is_finite() {
                    return Err(PlanningError::NonFiniteUtility);
                }

                let mut steps = node.steps.clone();
                steps.push(PlanStep {
                    action: candidate.clone(),
                    predicted_state: predicted_state.clone(),
                    prediction_uncertainty,
                    uncertainty_calibrated: calibrated,
                });
                expanded_nodes = expanded_nodes
                    .checked_add(1)
                    .ok_or(PlanningError::CounterOverflow)?;
                let next = BeamNode {
                    state: predicted_state,
                    steps,
                    cumulative_utility: node.cumulative_utility + step_utility,
                    cumulative_uncertainty: node.cumulative_uncertainty + prediction_uncertainty,
                };
                if !next.cumulative_utility.is_finite() {
                    return Err(PlanningError::NonFiniteUtility);
                }
                if !next.cumulative_uncertainty.is_finite() || !next.score(config).is_finite() {
                    return Err(PlanningError::InvalidPrediction);
                }
                expanded.push(next);
            }
        }

        if expanded.is_empty() {
            return Err(PlanningError::NoFeasiblePlan);
        }
        expanded.sort_by(|left, right| {
            right
                .score(config)
                .total_cmp(&left.score(config))
                .then_with(|| last_action_id(left).cmp(last_action_id(right)))
        });
        expanded.truncate(config.beam_width);
        frontier = expanded;
    }

    let best = frontier
        .into_iter()
        .next()
        .ok_or(PlanningError::NoFeasiblePlan)?;
    Ok(Plan {
        objective_score: best.score(config),
        steps: best.steps,
        expected_utility: best.cumulative_utility,
        total_uncertainty: best.cumulative_uncertainty,
        expanded_nodes,
        rejected_actions,
    })
}

fn last_action_id(node: &BeamNode) -> &str {
    node.steps
        .last()
        .map(|step| step.action.action_id.as_str())
        .unwrap_or("")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanningError {
    InvalidConfiguration,
    InvalidEnvironment(String),
    InvalidState,
    InvalidPrediction,
    NonFiniteUtility,
    NoCandidateActions,
    NoFeasiblePlan,
    CounterOverflow,
    WorldModel(WorldModelError),
}

impl std::fmt::Display for PlanningError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration => write!(formatter, "planning configuration is invalid"),
            Self::InvalidEnvironment(message) => {
                write!(formatter, "invalid environment: {message}")
            }
            Self::InvalidState => write!(formatter, "planning state must be a finite object"),
            Self::InvalidPrediction => {
                write!(formatter, "world model returned an invalid prediction")
            }
            Self::NonFiniteUtility => {
                write!(formatter, "utility function returned a non-finite score")
            }
            Self::NoCandidateActions => {
                write!(formatter, "at least one candidate action is required")
            }
            Self::NoFeasiblePlan => write!(formatter, "no feasible model-supported plan was found"),
            Self::CounterOverflow => write!(formatter, "planning counter is exhausted"),
            Self::WorldModel(error) => write!(formatter, "world-model prediction failed: {error}"),
        }
    }
}

impl std::error::Error for PlanningError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{ActionAuthorizationPolicy, ActionDescriptor, ActionRisk};
    use crate::learning::TransitionSample;
    use crate::schema::SchemaDefinition;
    use crate::world_model::{ActionDeltaModel, EnsembleWorldModel, PredictiveModel};

    fn environment() -> EnvironmentDescriptor {
        EnvironmentDescriptor {
            id: "world".to_owned(),
            version: "1".to_owned(),
            observation_schemas: Vec::new(),
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
                    id: "danger".to_owned(),
                    arguments: SchemaDefinition {
                        id: "empty-danger".to_owned(),
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
    fn beam_search_uses_model_predictions_and_filters_unsafe_actions() {
        let mut first = ActionDeltaModel::new("delta-a", 1).expect("model");
        first
            .fit(
                "train-a",
                &[
                    TransitionSample {
                        action_id: "advance".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(0.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(2.0))]),
                    },
                    TransitionSample {
                        action_id: "advance".to_owned(),
                        before: DataValue::object([(
                            "position".to_owned(),
                            DataValue::Number(2.0),
                        )]),
                        after: DataValue::object([("position".to_owned(), DataValue::Number(4.0))]),
                    },
                ],
            )
            .expect("fit");
        let model = EnsembleWorldModel::new(vec![Box::new(first)]).expect("ensemble");
        let environment = environment();
        let plan = plan(
            &environment,
            ActionAuthorizer::new(ActionAuthorizationPolicy::new(ActionRisk::Reversible)),
            &model,
            &DataValue::object([("position".to_owned(), DataValue::Number(0.0))]),
            &[
                ActionProposal {
                    action_id: "advance".to_owned(),
                    arguments: DataValue::object(std::iter::empty()),
                },
                ActionProposal {
                    action_id: "danger".to_owned(),
                    arguments: DataValue::object(std::iter::empty()),
                },
            ],
            PlanningConfig {
                horizon: 2,
                beam_width: 2,
                uncertainty_penalty: 0.0,
                uncalibrated_prediction_penalty: 0.0,
            },
            |state| {
                state
                    .get("position")
                    .and_then(DataValue::as_f64)
                    .unwrap_or(0.0)
            },
        )
        .expect("plan");

        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].action.action_id, "advance");
        assert_eq!(plan.steps[1].action.action_id, "advance");
        assert_eq!(
            plan.steps[1]
                .predicted_state
                .get("position")
                .and_then(DataValue::as_f64),
            Some(4.0)
        );
        assert!(plan.rejected_actions > 0);
    }

    #[test]
    fn rejects_zero_horizon() {
        assert_eq!(
            PlanningConfig {
                horizon: 0,
                ..PlanningConfig::default()
            }
            .validate(),
            Err(PlanningError::InvalidConfiguration)
        );
    }
}
