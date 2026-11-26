//! Versioned predictive-model lifecycle and held-out promotion gates.
//!
//! The registry tracks durable artifact references and evaluation provenance.
//! It never executes model artifacts; loading and prediction remain explicit
//! operations through the model-artifact store and trusted model implementations.

use crate::benchmark::ScoreDirection;
use crate::value::DataValue;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelCandidateSpec {
    pub id: String,
    pub version: u64,
    pub model_artifact_ref: String,
    pub environment_id: String,
    pub environment_version: String,
    pub training_dataset_id: String,
}

impl ModelCandidateSpec {
    fn validate(&self) -> Result<(), ModelRegistryError> {
        if self.id.trim().is_empty()
            || self.version == 0
            || self.model_artifact_ref.trim().is_empty()
            || self.environment_id.trim().is_empty()
            || self.environment_version.trim().is_empty()
            || self.training_dataset_id.trim().is_empty()
        {
            return Err(ModelRegistryError::InvalidSpec);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelEvaluation {
    pub evaluation_dataset_id: String,
    pub metric_id: String,
    pub direction: ScoreDirection,
    pub sample_count: usize,
    pub score: f64,
    pub resource_cost: f64,
    pub uncertainty_calibrated: bool,
}

impl ModelEvaluation {
    fn validate(&self) -> Result<(), ModelRegistryError> {
        if self.evaluation_dataset_id.trim().is_empty()
            || self.metric_id.trim().is_empty()
            || self.sample_count == 0
            || !self.score.is_finite()
            || !self.resource_cost.is_finite()
            || self.resource_cost < 0.0
        {
            return Err(ModelRegistryError::InvalidEvaluation);
        }
        Ok(())
    }

    fn utility(&self, policy: ModelPromotionPolicy) -> Result<f64, ModelRegistryError> {
        let directional_score = match self.direction {
            ScoreDirection::HigherIsBetter => self.score,
            ScoreDirection::LowerIsBetter => -self.score,
        };
        let utility = directional_score - policy.resource_cost_weight * self.resource_cost;
        if utility.is_finite() {
            Ok(utility)
        } else {
            Err(ModelRegistryError::NumericalOverflow)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelPromotionPolicy {
    pub minimum_evaluation_samples: usize,
    pub minimum_improvement: f64,
    pub resource_cost_weight: f64,
    pub require_calibrated_uncertainty: bool,
}

impl Default for ModelPromotionPolicy {
    fn default() -> Self {
        Self {
            minimum_evaluation_samples: 20,
            minimum_improvement: 0.01,
            resource_cost_weight: 0.0,
            require_calibrated_uncertainty: false,
        }
    }
}

impl ModelPromotionPolicy {
    pub fn validate(&self) -> Result<(), ModelRegistryError> {
        if self.minimum_evaluation_samples == 0
            || !self.minimum_improvement.is_finite()
            || self.minimum_improvement < 0.0
            || !self.resource_cost_weight.is_finite()
            || self.resource_cost_weight < 0.0
        {
            return Err(ModelRegistryError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelStatus {
    Candidate,
    Active,
    Rejected,
    Retired,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelRecord {
    pub spec: ModelCandidateSpec,
    pub status: ModelStatus,
    pub evaluation: Option<ModelEvaluation>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelRegistry {
    records: BTreeMap<(String, u64), ModelRecord>,
}

impl ModelRegistry {
    pub fn register_candidate(&mut self, spec: ModelCandidateSpec) -> Result<(), ModelRegistryError> {
        spec.validate()?;
        if self.records.contains_key(&(spec.id.clone(), spec.version)) {
            return Err(ModelRegistryError::DuplicateVersion);
        }
        if self.records
            .keys()
            .filter(|(id, _)| id == &spec.id)
            .any(|(_, version)| *version >= spec.version)
        {
            return Err(ModelRegistryError::VersionMustIncrease);
        }
        self.records.insert(
            (spec.id.clone(), spec.version),
            ModelRecord {
                spec,
                status: ModelStatus::Candidate,
                evaluation: None,
            },
        );
        Ok(())
    }

    pub fn evaluate_candidate(
        &mut self,
        id: &str,
        version: u64,
        evaluation: ModelEvaluation,
        policy: ModelPromotionPolicy,
    ) -> Result<(), ModelRegistryError> {
        policy.validate()?;
        evaluation.validate()?;
        let record = self.records
            .get_mut(&(id.to_owned(), version))
            .ok_or(ModelRegistryError::ModelNotFound)?;
        if record.status != ModelStatus::Candidate {
            return Err(ModelRegistryError::InvalidLifecycle);
        }
        if evaluation.evaluation_dataset_id == record.spec.training_dataset_id {
            return Err(ModelRegistryError::EvaluationReusesTrainingData);
        }
        if evaluation.sample_count < policy.minimum_evaluation_samples {
            record.evaluation = Some(evaluation);
            record.status = ModelStatus::Rejected;
            return Err(ModelRegistryError::InsufficientEvaluationSamples);
        }
        record.evaluation = Some(evaluation);
        Ok(())
    }

    pub fn promote(
        &mut self,
        id: &str,
        version: u64,
        policy: ModelPromotionPolicy,
    ) -> Result<(), ModelRegistryError> {
        policy.validate()?;
        let candidate_key = (id.to_owned(), version);
        let candidate = self.records.get(&candidate_key)
            .ok_or(ModelRegistryError::ModelNotFound)?;
        if candidate.status != ModelStatus::Candidate {
            return Err(ModelRegistryError::InvalidLifecycle);
        }
        let candidate_evaluation = candidate.evaluation.as_ref()
            .ok_or(ModelRegistryError::NotEvaluated)?;
        if policy.require_calibrated_uncertainty && !candidate_evaluation.uncertainty_calibrated {
            self.records.get_mut(&candidate_key).ok_or(ModelRegistryError::ModelNotFound)?.status =
                ModelStatus::Rejected;
            return Err(ModelRegistryError::UncalibratedUncertainty);
        }
        if candidate_evaluation.sample_count < policy.minimum_evaluation_samples {
            self.records.get_mut(&candidate_key).ok_or(ModelRegistryError::ModelNotFound)?.status =
                ModelStatus::Rejected;
            return Err(ModelRegistryError::InsufficientEvaluationSamples);
        }
        let candidate_utility = candidate_evaluation.utility(policy)?;
        let incumbent_key = self.records.values()
            .find(|record| record.spec.id == id && record.status == ModelStatus::Active)
            .map(|record| (record.spec.id.clone(), record.spec.version));

        if let Some(key) = incumbent_key.as_ref() {
            let incumbent = self.records.get(key).ok_or(ModelRegistryError::ModelNotFound)?;
            if incumbent.spec.environment_id != candidate.spec.environment_id
                || incumbent.spec.environment_version != candidate.spec.environment_version
            {
                return Err(ModelRegistryError::IncompatibleEnvironment);
            }
            let incumbent_evaluation = incumbent.evaluation.as_ref()
                .ok_or(ModelRegistryError::NotEvaluated)?;
            if incumbent_evaluation.evaluation_dataset_id != candidate_evaluation.evaluation_dataset_id
                || incumbent_evaluation.metric_id != candidate_evaluation.metric_id
                || incumbent_evaluation.direction != candidate_evaluation.direction
            {
                return Err(ModelRegistryError::IncomparableEvaluations);
            }
            let incumbent_utility = incumbent_evaluation.utility(policy)?;
            if candidate_utility < incumbent_utility + policy.minimum_improvement {
                self.records.get_mut(&candidate_key).ok_or(ModelRegistryError::ModelNotFound)?.status =
                    ModelStatus::Rejected;
                return Err(ModelRegistryError::InsufficientImprovement);
            }
        }

        if let Some(key) = incumbent_key {
            self.records.get_mut(&key).ok_or(ModelRegistryError::ModelNotFound)?.status =
                ModelStatus::Retired;
        }
        self.records.get_mut(&candidate_key).ok_or(ModelRegistryError::ModelNotFound)?.status =
            ModelStatus::Active;
        Ok(())
    }

    pub fn active(&self, id: &str) -> Option<&ModelRecord> {
        self.records.values().find(|record| {
            record.spec.id == id && record.status == ModelStatus::Active
        })
    }

    pub fn get(&self, id: &str, version: u64) -> Option<&ModelRecord> {
        self.records.get(&(id.to_owned(), version))
    }

    pub fn iter(&self) -> impl Iterator<Item = &ModelRecord> {
        self.records.values()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "models".to_owned(),
                DataValue::Sequence(self.records.values().map(record_to_value).collect()),
            ),
        ])
    }

    pub fn from_artifact_value(value: &DataValue) -> Result<Self, ModelRegistryError> {
        let fields = value.as_object().ok_or(ModelRegistryError::InvalidSerializedState)?;
        if fields.len() != 2 || fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
            return Err(ModelRegistryError::InvalidSerializedState);
        }
        let values = match fields.get("models") {
            Some(DataValue::Sequence(values)) => values,
            _ => return Err(ModelRegistryError::InvalidSerializedState),
        };
        let mut records = BTreeMap::new();
        let mut active_ids = BTreeSet::new();
        let mut max_versions: BTreeMap<String, u64> = BTreeMap::new();
        for value in values {
            let record = record_from_value(value)?;
            if record.status == ModelStatus::Active && !active_ids.insert(record.spec.id.clone()) {
                return Err(ModelRegistryError::InvalidSerializedState);
            }
            let max = max_versions.entry(record.spec.id.clone()).or_insert(0);
            if record.spec.version <= *max {
                return Err(ModelRegistryError::InvalidSerializedState);
            }
            *max = record.spec.version;
            let key = (record.spec.id.clone(), record.spec.version);
            if records.insert(key, record).is_some() {
                return Err(ModelRegistryError::InvalidSerializedState);
            }
        }
        Ok(Self { records })
    }
}

fn record_to_value(record: &ModelRecord) -> DataValue {
    let evaluation = match &record.evaluation {
        None => DataValue::Null,
        Some(value) => DataValue::object([
            ("evaluation_dataset_id".to_owned(), DataValue::Text(value.evaluation_dataset_id.clone())),
            ("metric_id".to_owned(), DataValue::Text(value.metric_id.clone())),
            ("direction".to_owned(), DataValue::Text(match value.direction {
                ScoreDirection::HigherIsBetter => "higher",
                ScoreDirection::LowerIsBetter => "lower",
            }.to_owned())),
            ("sample_count".to_owned(), DataValue::Text(value.sample_count.to_string())),
            ("score".to_owned(), DataValue::Number(value.score)),
            ("resource_cost".to_owned(), DataValue::Number(value.resource_cost)),
            ("uncertainty_calibrated".to_owned(), DataValue::Bool(value.uncertainty_calibrated)),
        ]),
    };
    DataValue::object([
        ("id".to_owned(), DataValue::Text(record.spec.id.clone())),
        ("version".to_owned(), DataValue::Text(record.spec.version.to_string())),
        ("model_artifact_ref".to_owned(), DataValue::Text(record.spec.model_artifact_ref.clone())),
        ("environment_id".to_owned(), DataValue::Text(record.spec.environment_id.clone())),
        ("environment_version".to_owned(), DataValue::Text(record.spec.environment_version.clone())),
        ("training_dataset_id".to_owned(), DataValue::Text(record.spec.training_dataset_id.clone())),
        ("status".to_owned(), DataValue::Text(status_name(record.status).to_owned())),
        ("evaluation".to_owned(), evaluation),
    ])
}

fn record_from_value(value: &DataValue) -> Result<ModelRecord, ModelRegistryError> {
    let fields = value.as_object().ok_or(ModelRegistryError::InvalidSerializedState)?;
    if fields.len() != 8 {
        return Err(ModelRegistryError::InvalidSerializedState);
    }
    let text = |name: &str| match fields.get(name) {
        Some(DataValue::Text(value)) => Ok(value.clone()),
        _ => Err(ModelRegistryError::InvalidSerializedState),
    };
    let version = text("version")?.parse().map_err(|_| ModelRegistryError::InvalidSerializedState)?;
    let spec = ModelCandidateSpec {
        id: text("id")?,
        version,
        model_artifact_ref: text("model_artifact_ref")?,
        environment_id: text("environment_id")?,
        environment_version: text("environment_version")?,
        training_dataset_id: text("training_dataset_id")?,
    };
    spec.validate()?;
    let status = match text("status")?.as_str() {
        "candidate" => ModelStatus::Candidate,
        "active" => ModelStatus::Active,
        "rejected" => ModelStatus::Rejected,
        "retired" => ModelStatus::Retired,
        _ => return Err(ModelRegistryError::InvalidSerializedState),
    };
    let evaluation = match fields.get("evaluation") {
        Some(DataValue::Null) => None,
        Some(DataValue::Object(values)) if values.len() == 7 => {
            let text_value = |name: &str| match values.get(name) {
                Some(DataValue::Text(value)) => Ok(value.clone()),
                _ => Err(ModelRegistryError::InvalidSerializedState),
            };
            let direction = match text_value("direction")?.as_str() {
                "higher" => ScoreDirection::HigherIsBetter,
                "lower" => ScoreDirection::LowerIsBetter,
                _ => return Err(ModelRegistryError::InvalidSerializedState),
            };
            let sample_count = text_value("sample_count")?.parse().map_err(|_| ModelRegistryError::InvalidSerializedState)?;
            let score = values.get("score").and_then(DataValue::as_f64)
                .filter(|value| value.is_finite()).ok_or(ModelRegistryError::InvalidSerializedState)?;
            let resource_cost = values.get("resource_cost").and_then(DataValue::as_f64)
                .filter(|value| value.is_finite()).ok_or(ModelRegistryError::InvalidSerializedState)?;
            let uncertainty_calibrated = match values.get("uncertainty_calibrated") {
                Some(DataValue::Bool(value)) => *value,
                _ => return Err(ModelRegistryError::InvalidSerializedState),
            };
            let evaluation = ModelEvaluation {
                evaluation_dataset_id: text_value("evaluation_dataset_id")?,
                metric_id: text_value("metric_id")?,
                direction,
                sample_count,
                score,
                resource_cost,
                uncertainty_calibrated,
            };
            evaluation.validate()?;
            if evaluation.evaluation_dataset_id == spec.training_dataset_id {
                return Err(ModelRegistryError::InvalidSerializedState);
            }
            Some(evaluation)
        }
        _ => return Err(ModelRegistryError::InvalidSerializedState),
    };
    if matches!(status, ModelStatus::Active | ModelStatus::Retired | ModelStatus::Rejected)
        && evaluation.is_none()
    {
        return Err(ModelRegistryError::InvalidSerializedState);
    }
    Ok(ModelRecord { spec, status, evaluation })
}

fn status_name(status: ModelStatus) -> &'static str {
    match status {
        ModelStatus::Candidate => "candidate",
        ModelStatus::Active => "active",
        ModelStatus::Rejected => "rejected",
        ModelStatus::Retired => "retired",
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRegistryError {
    InvalidSpec,
    DuplicateVersion,
    VersionMustIncrease,
    InvalidEvaluation,
    InvalidPolicy,
    ModelNotFound,
    InvalidLifecycle,
    EvaluationReusesTrainingData,
    InsufficientEvaluationSamples,
    NotEvaluated,
    UncalibratedUncertainty,
    IncompatibleEnvironment,
    IncomparableEvaluations,
    InsufficientImprovement,
    NumericalOverflow,
    InvalidSerializedState,
}

impl std::fmt::Display for ModelRegistryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSpec => write!(formatter, "predictive model candidate is invalid"),
            Self::DuplicateVersion => write!(formatter, "predictive model version already exists"),
            Self::VersionMustIncrease => write!(formatter, "predictive model versions must increase monotonically"),
            Self::InvalidEvaluation => write!(formatter, "predictive model evaluation is invalid"),
            Self::InvalidPolicy => write!(formatter, "predictive model promotion policy is invalid"),
            Self::ModelNotFound => write!(formatter, "predictive model candidate was not found"),
            Self::InvalidLifecycle => write!(formatter, "predictive model lifecycle transition is invalid"),
            Self::EvaluationReusesTrainingData => write!(formatter, "model evaluation must use data distinct from training"),
            Self::InsufficientEvaluationSamples => write!(formatter, "model evaluation has too few samples"),
            Self::NotEvaluated => write!(formatter, "predictive model has not been evaluated"),
            Self::UncalibratedUncertainty => write!(formatter, "model uncertainty is not calibrated to the required policy"),
            Self::IncompatibleEnvironment => write!(formatter, "model candidates target different environment contracts"),
            Self::IncomparableEvaluations => write!(formatter, "model candidates require the same metric and evaluation dataset"),
            Self::InsufficientImprovement => write!(formatter, "candidate model did not exceed the improvement margin"),
            Self::NumericalOverflow => write!(formatter, "model promotion utility overflowed"),
            Self::InvalidSerializedState => write!(formatter, "serialized model registry is invalid"),
        }
    }
}

impl std::error::Error for ModelRegistryError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(version: u64) -> ModelCandidateSpec {
        ModelCandidateSpec {
            id: "world-model".to_owned(),
            version,
            model_artifact_ref: format!("world-model-v{version}"),
            environment_id: "grid".to_owned(),
            environment_version: "v1".to_owned(),
            training_dataset_id: format!("training-v{version}"),
        }
    }

    fn evaluation(dataset: &str, score: f64, calibrated: bool) -> ModelEvaluation {
        ModelEvaluation {
            evaluation_dataset_id: dataset.to_owned(),
            metric_id: "rmse".to_owned(),
            direction: ScoreDirection::LowerIsBetter,
            sample_count: 100,
            score,
            resource_cost: 2.0,
            uncertainty_calibrated: calibrated,
        }
    }

    fn policy() -> ModelPromotionPolicy {
        ModelPromotionPolicy {
            minimum_evaluation_samples: 20,
            minimum_improvement: 0.01,
            resource_cost_weight: 0.1,
            require_calibrated_uncertainty: true,
        }
    }

    #[test]
    fn promotion_requires_same_holdout_and_calibrated_uncertainty() {
        let mut registry = ModelRegistry::default();
        registry.register_candidate(spec(1)).expect("v1");
        registry.evaluate_candidate("world-model", 1, evaluation("holdout-1", 0.5, true), policy()).expect("evaluate v1");
        registry.promote("world-model", 1, policy()).expect("promote v1");

        registry.register_candidate(spec(2)).expect("v2");
        registry.evaluate_candidate("world-model", 2, evaluation("holdout-1", 0.4, false), policy()).expect("evaluate v2");
        assert_eq!(registry.promote("world-model", 2, policy()), Err(ModelRegistryError::UncalibratedUncertainty));
        assert_eq!(registry.active("world-model").map(|record| record.spec.version), Some(1));
    }

    #[test]
    fn promotion_retires_incumbent_and_registry_round_trips() {
        let mut registry = ModelRegistry::default();
        registry.register_candidate(spec(1)).expect("v1");
        registry.evaluate_candidate("world-model", 1, evaluation("holdout", 0.6, true), policy()).expect("evaluate");
        registry.promote("world-model", 1, policy()).expect("promote");
        registry.register_candidate(spec(2)).expect("v2");
        registry.evaluate_candidate("world-model", 2, evaluation("holdout", 0.4, true), policy()).expect("evaluate");
        registry.promote("world-model", 2, policy()).expect("promote v2");
        assert_eq!(registry.get("world-model", 1).map(|record| record.status), Some(ModelStatus::Retired));
        let restored = ModelRegistry::from_artifact_value(&registry.to_artifact_value()).expect("restore");
        assert_eq!(restored, registry);
    }
}
