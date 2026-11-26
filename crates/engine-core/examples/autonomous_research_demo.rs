//! End-to-end experiment → evidence → model-training → promotion demonstration.

use engine_core::action::{
    ActionAuthorizationPolicy, ActionAuthorizer, ActionDescriptor, ActionProposal, ActionRisk,
    AuthorizedAction,
};
use engine_core::benchmark::ScoreDirection;
use engine_core::environment::{
    ActionOutcome, EnvironmentAdapter, EnvironmentDescriptor, EnvironmentError, ObservationFrame,
};
use engine_core::experience_journal::ExperienceJournal;
use engine_core::experiment_selection::{ExperimentCandidate, ExperimentSelectionPolicy};
use engine_core::learning::TransitionSample;
use engine_core::model_artifact::ModelArtifactStore;
use engine_core::model_registry::{
    ModelEvaluation as RegistryEvaluation, ModelPromotionPolicy, ModelRegistry,
};
use engine_core::planning::PlanningConfig;
use engine_core::reasoning::{HypothesisPolicy, HypothesisRegistry, HypothesisSpec};
use engine_core::research_cycle::{
    run_autonomous_research_cycle, AutonomousResearchCallbacks, AutonomousResearchConfig,
    AutonomousResearchContext, AutonomousResearchRequest, ExperimentActionMapping,
    ExperimentExecutionConfig, LearningCycleConfig,
};
use engine_core::schema::{FieldSpec, SchemaDefinition, ValueKind};
use engine_core::world_model::{
    evaluate_model, ActionDeltaModel, EnsembleWorldModel, PredictiveModel,
};
use engine_core::{DataValue, RuntimeConfig};
use std::error::Error;
use std::path::PathBuf;

struct CounterEnvironment {
    descriptor: EnvironmentDescriptor,
    position: f64,
    sequence: u64,
}

impl CounterEnvironment {
    fn new() -> Self {
        Self {
            descriptor: EnvironmentDescriptor {
                id: "counter-environment".to_owned(),
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
        }
    }

    fn frame(&mut self) -> ObservationFrame {
        let frame = ObservationFrame {
            environment_id: "counter-environment".to_owned(),
            session_id: "counter-session".to_owned(),
            episode_id: "research-episode".to_owned(),
            sequence: self.sequence,
            observed_at_millis: self.sequence,
            schema_id: "counter-state".to_owned(),
            schema_version: 1,
            payload: DataValue::object([("position".to_owned(), DataValue::Number(self.position))]),
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

    fn execute(&mut self, _action: AuthorizedAction) -> Result<ActionOutcome, EnvironmentError> {
        self.position += 1.0;
        Ok(ActionOutcome {
            action_id: "advance".to_owned(),
            completed: true,
            observations: Vec::new(),
            feedback: DataValue::Null,
        })
    }
}

fn active_world_model() -> Result<EnsembleWorldModel, Box<dyn Error>> {
    let mut model = ActionDeltaModel::new("counter-world-model", 1)?;
    model.fit(
        "seed-training-v1",
        &[transition(0.0, 1.0), transition(4.0, 5.0)],
    )?;
    Ok(EnsembleWorldModel::new(vec![
        Box::new(model) as Box<dyn PredictiveModel>
    ])?)
}

fn transition(before: f64, after: f64) -> TransitionSample {
    TransitionSample {
        action_id: "advance".to_owned(),
        before: DataValue::object([("position".to_owned(), DataValue::Number(before))]),
        after: DataValue::object([("position".to_owned(), DataValue::Number(after))]),
    }
}

fn temp_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("5413-{name}-{}.bin", std::process::id()))
}

fn main() -> Result<(), Box<dyn Error>> {
    let journal_path = temp_file("autonomous-research-journal");
    let artifacts_path = temp_file("autonomous-research-artifacts");
    let mut journal = ExperienceJournal::open(&journal_path)?;
    let mut artifacts = ModelArtifactStore::open(&artifacts_path)?;
    let mut environment = CounterEnvironment::new();
    let mut candidate_model = ActionDeltaModel::new("counter-world-model", 2)?;
    let mut hypotheses = HypothesisRegistry::default();
    let mut model_registry = ModelRegistry::default();

    let hypothesis_policy = HypothesisPolicy {
        minimum_evidence_count: 1,
        support_probability: 0.8,
        refutation_probability: 0.2,
    };
    hypotheses.insert(
        HypothesisSpec {
            id: "advance-increases-position".to_owned(),
            claim: DataValue::Text("advance increases position".to_owned()),
            description: "A single advance action increases the observed position".to_owned(),
            prior_probability: 0.5,
            source_dataset_id: Some("initial-discovery-v1".to_owned()),
        },
        hypothesis_policy,
    )?;

    let experiment = ExperimentCandidate {
        id: "advance-once".to_owned(),
        specification: DataValue::Text("execute one reversible advance action".to_owned()),
        observation_probability_if_true: 0.9,
        observation_probability_if_false: 0.1,
        resource_cost: 1.0,
        risk: 0.01,
        irreversible: false,
    };
    let mapping = ExperimentActionMapping {
        experiment_id: experiment.id.clone(),
        action: ActionProposal {
            action_id: "advance".to_owned(),
            arguments: DataValue::Object(Default::default()),
        },
    };
    let validation_samples = [transition(10.0, 11.0), transition(20.0, 21.0)];

    let report = run_autonomous_research_cycle(
        AutonomousResearchContext {
            adapter: &mut environment,
            active_model: &active_world_model()?,
            candidate_model: &mut candidate_model,
            hypothesis_registry: &mut hypotheses,
            journal: &mut journal,
            artifact_store: &mut artifacts,
            model_registry: &mut model_registry,
        },
        AutonomousResearchRequest {
            hypothesis_id: "advance-increases-position",
            experiments: &[experiment],
            action_mappings: &[mapping],
            validation_samples: &validation_samples,
            selection_policy: ExperimentSelectionPolicy {
                maximum_resource_cost: 10.0,
                maximum_risk: 0.1,
                resource_cost_weight: 0.0,
                risk_weight: 0.0,
                allow_irreversible: false,
            },
            hypothesis_policy,
            promotion_policy: ModelPromotionPolicy {
                minimum_evaluation_samples: 2,
                minimum_improvement: 0.0,
                resource_cost_weight: 0.0,
                require_calibrated_uncertainty: false,
            },
            config: AutonomousResearchConfig {
                experiment: ExperimentExecutionConfig {
                    runtime: RuntimeConfig {
                        execution_id: "counter-research-run-1".to_owned(),
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
                    evidence_rationale: "Final position was measured after the intervention"
                        .to_owned(),
                    hypothesis_registry_artifact_ref: "hypothesis-registry-after-run-1".to_owned(),
                },
                learning: LearningCycleConfig {
                    episode_id: "research-episode".to_owned(),
                    training_dataset_id: "episode-training-v1".to_owned(),
                    validation_dataset_id: "model-holdout-v1".to_owned(),
                    model_id: "counter-world-model".to_owned(),
                    model_version: 2,
                    model_artifact_ref: "counter-world-model-v2".to_owned(),
                    model_registry_artifact_ref: "model-registry-after-run-1".to_owned(),
                    environment_id: "counter-environment".to_owned(),
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
                let metrics = evaluate_model(model, "model-holdout-v1", samples)
                    .map_err(|error| error.to_string())?;
                Ok(RegistryEvaluation {
                    evaluation_dataset_id: metrics.evaluation_dataset_id,
                    metric_id: "rmse".to_owned(),
                    direction: ScoreDirection::LowerIsBetter,
                    sample_count: metrics.evaluated_transitions,
                    score: metrics.root_mean_squared_error,
                    resource_cost: 1.0,
                    uncertainty_calibrated: metrics.calibrated_predictions
                        == metrics.predicted_fields,
                })
            },
        },
    )?;

    let learning = report.learning?;
    let restored_hypotheses =
        artifacts.load_hypothesis_registry("hypothesis-registry-after-run-1", hypothesis_policy)?;
    let restored_models = artifacts.load_model_registry("model-registry-after-run-1")?;
    let active_version = restored_models
        .active("counter-world-model")
        .map(|record| record.spec.version)
        .unwrap_or(0);

    println!("selected experiment: {}", report.selected.experiment_id);
    println!(
        "hypothesis posterior: {:.3} ({:?})",
        restored_hypotheses
            .get("advance-increases-position")
            .map(|record| record.posterior_probability)
            .unwrap_or(0.0),
        report.updated_hypothesis.status
    );
    println!(
        "model: v{}; trained on {} episode transition(s); validation RMSE {:.4}",
        learning.model_version, learning.training_samples, learning.evaluation.score
    );
    println!("active model version after restoration: {active_version}");

    drop(artifacts);
    drop(journal);
    std::fs::remove_file(journal_path)?;
    std::fs::remove_file(artifacts_path)?;
    Ok(())
}
