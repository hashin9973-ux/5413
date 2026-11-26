use engine_core::action::{ActionDescriptor, ActionAuthorizationPolicy, ActionAuthorizer, ActionProposal, ActionRisk};
use engine_core::environment::EnvironmentDescriptor;
use engine_core::experimentation::{
    rank_experiments, ExperimentCandidate, ExperimentPolicy,
};
use engine_core::learning::TransitionSample;
use engine_core::planning::{plan, PlanningConfig};
use engine_core::schema::SchemaDefinition;
use engine_core::value::DataValue;
use engine_core::world_model::{
    evaluate_ensemble, ActionDeltaModel, ContextualLinearModel, EnsembleWorldModel,
    PredictiveModel,
};
use std::error::Error;

fn state(position: f64, velocity: f64) -> DataValue {
    DataValue::object([
        ("position".to_owned(), DataValue::Number(position)),
        ("velocity".to_owned(), DataValue::Number(velocity)),
    ])
}

fn transition(
    action_id: &str,
    position: f64,
    velocity: f64,
    next_position: f64,
    next_velocity: f64,
) -> TransitionSample {
    TransitionSample {
        action_id: action_id.to_owned(),
        before: state(position, velocity),
        after: state(next_position, next_velocity),
    }
}

fn environment() -> EnvironmentDescriptor {
    let action_schema = SchemaDefinition {
        id: "empty-action-arguments".to_owned(),
        version: 1,
        fields: Vec::new(),
        allow_additional_fields: false,
    };
    EnvironmentDescriptor {
        id: "toy-dynamics".to_owned(),
        version: "1".to_owned(),
        observation_schemas: Vec::new(),
        actions: vec![
            ActionDescriptor {
                id: "advance".to_owned(),
                arguments: action_schema.clone(),
                risk: ActionRisk::Reversible,
            },
            ActionDescriptor {
                id: "boost".to_owned(),
                arguments: action_schema,
                risk: ActionRisk::Reversible,
            },
        ],
        supports_reset: false,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let training = vec![
        transition("advance", 0.0, 1.0, 1.0, 1.0),
        transition("advance", 2.0, 2.0, 4.0, 2.0),
        transition("advance", 3.0, 4.0, 7.0, 4.0),
        transition("advance", 7.0, 3.0, 10.0, 3.0),
        transition("boost", 0.0, 1.0, 1.0, 2.0),
        transition("boost", 1.0, 2.0, 3.0, 3.0),
        transition("boost", 3.0, 3.0, 6.0, 4.0),
        transition("boost", 6.0, 4.0, 10.0, 5.0),
    ];

    let mut statistical = ActionDeltaModel::new("action-delta", 1)?;
    statistical.fit("training-v1", &training)?;
    let mut contextual = ContextualLinearModel::new("contextual-ridge", 1, 1e-6)?;
    contextual.fit("training-v1", &training)?;
    let held_out = vec![
        transition("advance", 0.0, 5.0, 5.0, 5.0),
        transition("boost", 0.0, 5.0, 5.0, 6.0),
    ];
    let ensemble = EnsembleWorldModel::new(vec![
        Box::new(statistical) as Box<dyn PredictiveModel>,
        Box::new(contextual) as Box<dyn PredictiveModel>,
    ])?;

    let evaluation = evaluate_ensemble(&ensemble, "validation-v1", &held_out)?;
    println!(
        "held-out RMSE {:.3}; mean predicted stddev {:.3}; calibrated coverage {:?}",
        evaluation.root_mean_squared_error,
        evaluation.mean_predicted_std_dev,
        evaluation.empirical_coverage_95
    );

    let environment = environment();
    let current = state(0.0, 2.0);
    let candidates = vec![
        ActionProposal {
            action_id: "advance".to_owned(),
            arguments: DataValue::object(std::iter::empty()),
        },
        ActionProposal {
            action_id: "boost".to_owned(),
            arguments: DataValue::object(std::iter::empty()),
        },
    ];

    let experiments = rank_experiments(
        &environment,
        &current,
        &[
            ExperimentCandidate {
                proposal: candidates[0].clone(),
                expected_utility: 1.0,
                estimated_cost: 1.0,
            },
            ExperimentCandidate {
                proposal: candidates[1].clone(),
                expected_utility: 1.0,
                estimated_cost: 1.2,
            },
        ],
        &ensemble,
        ExperimentPolicy::default(),
    )?;
    if let Some(best) = experiments.ranked.first() {
        println!(
            "next experiment: {}; disagreement proxy {:.3}; score {:.3}",
            best.proposal.action_id, best.information_gain_proxy, best.score
        );
    }

    let authorizer = ActionAuthorizer::new(ActionAuthorizationPolicy::new(ActionRisk::Reversible));
    let plan = plan(
        &environment,
        authorizer,
        &ensemble,
        &current,
        &candidates,
        PlanningConfig {
            horizon: 3,
            beam_width: 4,
            uncertainty_penalty: 0.2,
            uncalibrated_prediction_penalty: 2.0,
        },
        |predicted| {
            predicted
                .get("position")
                .and_then(DataValue::as_f64)
                .unwrap_or_default()
        },
    )?;
    println!(
        "planned {} steps; expected cumulative utility {:.3}; uncertainty {:.3}; first action {}",
        plan.steps.len(),
        plan.expected_utility,
        plan.total_uncertainty,
        plan.steps
            .first()
            .map(|step| step.action.action_id.as_str())
            .unwrap_or("none")
    );
    Ok(())
}
