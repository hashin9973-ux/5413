use crate::experience::ExperienceRecord;
use crate::learning::{
    extract_transition_samples, LearningError, NumericDeltaModel, TransitionSample,
};
use crate::value::DataValue;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelIdentity {
    pub id: String,
    pub version: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldEstimate {
    pub mean: f64,
    pub aleatoric_std_dev: f64,
    pub epistemic_std_dev: f64,
    pub sample_count: u64,
    /// False means the available data is insufficient to claim calibrated uncertainty.
    pub uncertainty_calibrated: bool,
}

impl FieldEstimate {
    pub fn total_std_dev(&self) -> f64 {
        (self.aleatoric_std_dev.powi(2) + self.epistemic_std_dev.powi(2)).sqrt()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelPrediction {
    pub contributors: Vec<ModelIdentity>,
    pub fields: BTreeMap<String, FieldEstimate>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelFitReport {
    pub samples_seen: u64,
    pub trainable_fields: usize,
}

pub trait PredictiveModel: Send + Sync {
    fn identity(&self) -> &ModelIdentity;

    fn fit(
        &mut self,
        training_dataset_id: &str,
        samples: &[TransitionSample],
    ) -> Result<ModelFitReport, WorldModelError>;

    fn predict(
        &self,
        action_id: &str,
        current_state: &DataValue,
    ) -> Result<ModelPrediction, WorldModelError>;

    fn training_dataset_id(&self) -> Option<&str>;

    fn training_samples(&self) -> u64;
    fn to_artifact_value(&self) -> Result<DataValue, WorldModelError>;
}

/// Statistical baseline that estimates action-conditioned numeric deltas.
pub struct ActionDeltaModel {
    identity: ModelIdentity,
    learner: NumericDeltaModel,
    training_dataset_id: Option<String>,
}

impl ActionDeltaModel {
    pub fn new(id: impl Into<String>, version: u64) -> Result<Self, WorldModelError> {
        let id = id.into();
        validate_identity(&id, version)?;
        Ok(Self {
            identity: ModelIdentity { id, version },
            learner: NumericDeltaModel::default(),
            training_dataset_id: None,
        })
    }
}

impl PredictiveModel for ActionDeltaModel {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn fit(
        &mut self,
        training_dataset_id: &str,
        samples: &[TransitionSample],
    ) -> Result<ModelFitReport, WorldModelError> {
        validate_training_set(training_dataset_id, samples)?;
        let mut learner = NumericDeltaModel::default();
        let mut fields = BTreeSet::new();
        for sample in samples {
            let before = sample
                .before
                .as_object()
                .ok_or(WorldModelError::ExpectedObjectState)?;
            let after = sample
                .after
                .as_object()
                .ok_or(WorldModelError::ExpectedObjectState)?;
            for (name, before_value) in before {
                if before_value.as_f64().is_some()
                    && after.get(name).and_then(DataValue::as_f64).is_some()
                {
                    fields.insert(name.clone());
                }
            }
            learner
                .observe_transition(sample)
                .map_err(map_learning_error)?;
        }

        self.learner = learner;
        self.training_dataset_id = Some(training_dataset_id.to_owned());
        Ok(ModelFitReport {
            samples_seen: to_u64(samples.len())?,
            trainable_fields: fields.len(),
        })
    }

    fn predict(
        &self,
        action_id: &str,
        current_state: &DataValue,
    ) -> Result<ModelPrediction, WorldModelError> {
        ensure_fitted(self.training_dataset_id())?;
        let predictions = self
            .learner
            .predict_next(action_id, current_state)
            .map_err(map_learning_error)?;
        let fields = predictions
            .into_iter()
            .map(|(name, prediction)| {
                let calibrated = prediction.sample_count >= 2;
                let epistemic_std_dev = if calibrated {
                    prediction.delta_standard_deviation / (prediction.sample_count as f64).sqrt()
                } else {
                    0.0
                };
                (
                    name,
                    FieldEstimate {
                        mean: prediction.predicted_value,
                        aleatoric_std_dev: prediction.delta_standard_deviation,
                        epistemic_std_dev,
                        sample_count: prediction.sample_count,
                        uncertainty_calibrated: calibrated,
                    },
                )
            })
            .collect();
        Ok(ModelPrediction {
            contributors: vec![self.identity.clone()],
            fields,
        })
    }

    fn training_dataset_id(&self) -> Option<&str> {
        self.training_dataset_id.as_deref()
    }

    fn training_samples(&self) -> u64 {
        self.learner.transitions_seen()
    }

    fn to_artifact_value(&self) -> Result<DataValue, WorldModelError> {
        let training_dataset_id = self
            .training_dataset_id
            .as_ref()
            .ok_or(WorldModelError::NotFitted)?;
        Ok(DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "model_type".to_owned(),
                DataValue::Text("action_delta".to_owned()),
            ),
            ("id".to_owned(), DataValue::Text(self.identity.id.clone())),
            (
                "model_version".to_owned(),
                DataValue::Text(self.identity.version.to_string()),
            ),
            (
                "training_dataset_id".to_owned(),
                DataValue::Text(training_dataset_id.clone()),
            ),
            ("learner".to_owned(), self.learner.to_artifact_value()),
        ]))
    }
}

/// Batch ridge-regression model with dynamically discovered state and action features.
///
/// Each numeric output is modeled as its previous value plus a learned delta. Missing
/// input features are represented as zero; callers should keep observation schemas
/// stable or explicitly model missingness in their environment representation.
pub struct ContextualLinearModel {
    identity: ModelIdentity,
    regularization: f64,
    feature_names: Vec<String>,
    fields: BTreeMap<String, LinearFieldModel>,
    training_dataset_id: Option<String>,
    training_samples: u64,
}

#[derive(Clone, Debug)]
struct LinearFieldModel {
    coefficients: Vec<f64>,
    residual_std_dev: f64,
    sample_count: u64,
    uncertainty_calibrated: bool,
}

impl ContextualLinearModel {
    pub fn new(
        id: impl Into<String>,
        version: u64,
        regularization: f64,
    ) -> Result<Self, WorldModelError> {
        let id = id.into();
        validate_identity(&id, version)?;
        if !regularization.is_finite() || regularization < 0.0 {
            return Err(WorldModelError::InvalidRegularization);
        }
        Ok(Self {
            identity: ModelIdentity { id, version },
            regularization,
            feature_names: Vec::new(),
            fields: BTreeMap::new(),
            training_dataset_id: None,
            training_samples: 0,
        })
    }
}

impl PredictiveModel for ContextualLinearModel {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn fit(
        &mut self,
        training_dataset_id: &str,
        samples: &[TransitionSample],
    ) -> Result<ModelFitReport, WorldModelError> {
        validate_training_set(training_dataset_id, samples)?;

        let mut feature_set = BTreeSet::new();
        for sample in samples {
            sample
                .before
                .validate_finite()
                .map_err(|_| WorldModelError::NonFiniteTrainingData)?;
            sample
                .after
                .validate_finite()
                .map_err(|_| WorldModelError::NonFiniteTrainingData)?;
            let before = sample
                .before
                .as_object()
                .ok_or(WorldModelError::ExpectedObjectState)?;
            let after = sample
                .after
                .as_object()
                .ok_or(WorldModelError::ExpectedObjectState)?;
            feature_set.insert(action_feature(&sample.action_id));
            for (name, value) in before {
                if value.as_f64().is_some() && after.get(name).and_then(DataValue::as_f64).is_some()
                {
                    feature_set.insert(state_feature(name));
                }
            }
        }

        let feature_names: Vec<String> = feature_set.into_iter().collect();
        let mut rows_by_target: BTreeMap<String, Vec<(Vec<f64>, f64)>> = BTreeMap::new();
        for sample in samples {
            let before = sample
                .before
                .as_object()
                .ok_or(WorldModelError::ExpectedObjectState)?;
            let after = sample
                .after
                .as_object()
                .ok_or(WorldModelError::ExpectedObjectState)?;
            let features = feature_vector(&feature_names, &sample.action_id, &sample.before)?;
            for (name, before_value) in before {
                let Some(before_number) = before_value.as_f64() else {
                    continue;
                };
                let Some(after_number) = after.get(name).and_then(DataValue::as_f64) else {
                    continue;
                };
                let delta = after_number - before_number;
                if !delta.is_finite() {
                    return Err(WorldModelError::NonFiniteTrainingData);
                }
                rows_by_target
                    .entry(name.clone())
                    .or_default()
                    .push((features.clone(), delta));
            }
        }

        let mut fitted_fields = BTreeMap::new();
        for (name, rows) in rows_by_target {
            let fitted = fit_linear_field(&rows, self.regularization)?;
            fitted_fields.insert(name, fitted);
        }
        if fitted_fields.is_empty() {
            return Err(WorldModelError::NoTrainableFields);
        }

        self.feature_names = feature_names;
        self.fields = fitted_fields;
        self.training_dataset_id = Some(training_dataset_id.to_owned());
        self.training_samples = to_u64(samples.len())?;
        Ok(ModelFitReport {
            samples_seen: self.training_samples,
            trainable_fields: self.fields.len(),
        })
    }

    fn predict(
        &self,
        action_id: &str,
        current_state: &DataValue,
    ) -> Result<ModelPrediction, WorldModelError> {
        ensure_fitted(self.training_dataset_id())?;
        current_state
            .validate_finite()
            .map_err(|_| WorldModelError::NonFinitePrediction)?;
        let current = current_state
            .as_object()
            .ok_or(WorldModelError::ExpectedObjectState)?;
        let features = feature_vector(&self.feature_names, action_id, current_state)?;
        let mut predictions = BTreeMap::new();

        for (name, model) in &self.fields {
            let Some(current_value) = current.get(name).and_then(DataValue::as_f64) else {
                continue;
            };
            let delta = dot(&model.coefficients, &features)?;
            let mean = current_value + delta;
            if !mean.is_finite() {
                return Err(WorldModelError::NonFinitePrediction);
            }
            let epistemic_std_dev = if model.sample_count == 0 {
                0.0
            } else {
                model.residual_std_dev / (model.sample_count as f64).sqrt()
            };
            predictions.insert(
                name.clone(),
                FieldEstimate {
                    mean,
                    aleatoric_std_dev: model.residual_std_dev,
                    epistemic_std_dev,
                    sample_count: model.sample_count,
                    uncertainty_calibrated: model.uncertainty_calibrated,
                },
            );
        }

        Ok(ModelPrediction {
            contributors: vec![self.identity.clone()],
            fields: predictions,
        })
    }

    fn training_dataset_id(&self) -> Option<&str> {
        self.training_dataset_id.as_deref()
    }

    fn training_samples(&self) -> u64 {
        self.training_samples
    }

    fn to_artifact_value(&self) -> Result<DataValue, WorldModelError> {
        let training_dataset_id = self
            .training_dataset_id
            .as_ref()
            .ok_or(WorldModelError::NotFitted)?;
        let features = DataValue::Sequence(
            self.feature_names
                .iter()
                .cloned()
                .map(DataValue::Text)
                .collect(),
        );
        let fields = DataValue::Object(
            self.fields
                .iter()
                .map(|(name, field)| {
                    (
                        name.clone(),
                        DataValue::object([
                            (
                                "coefficients".to_owned(),
                                DataValue::Sequence(
                                    field
                                        .coefficients
                                        .iter()
                                        .copied()
                                        .map(DataValue::Number)
                                        .collect(),
                                ),
                            ),
                            (
                                "residual_std_dev".to_owned(),
                                DataValue::Number(field.residual_std_dev),
                            ),
                            (
                                "sample_count".to_owned(),
                                DataValue::Text(field.sample_count.to_string()),
                            ),
                            (
                                "uncertainty_calibrated".to_owned(),
                                DataValue::Bool(field.uncertainty_calibrated),
                            ),
                        ]),
                    )
                })
                .collect(),
        );
        Ok(DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "model_type".to_owned(),
                DataValue::Text("contextual_linear".to_owned()),
            ),
            ("id".to_owned(), DataValue::Text(self.identity.id.clone())),
            (
                "model_version".to_owned(),
                DataValue::Text(self.identity.version.to_string()),
            ),
            (
                "training_dataset_id".to_owned(),
                DataValue::Text(training_dataset_id.clone()),
            ),
            (
                "training_samples".to_owned(),
                DataValue::Text(self.training_samples.to_string()),
            ),
            (
                "regularization".to_owned(),
                DataValue::Number(self.regularization),
            ),
            ("features".to_owned(), features),
            ("fields".to_owned(), fields),
        ]))
    }
}

/// Combines independently trained models. Between-model disagreement contributes
/// to epistemic uncertainty; within-model residual spread contributes to aleatoric
/// uncertainty. This is an empirical ensemble, not a Bayesian posterior.
pub struct EnsembleWorldModel {
    models: Vec<Box<dyn PredictiveModel>>,
}

impl EnsembleWorldModel {
    pub fn new(models: Vec<Box<dyn PredictiveModel>>) -> Result<Self, WorldModelError> {
        if models.is_empty() {
            return Err(WorldModelError::EmptyEnsemble);
        }
        let mut identities = BTreeSet::new();
        for model in &models {
            let identity = model.identity();
            validate_identity(&identity.id, identity.version)?;
            if !identities.insert(identity.clone()) {
                return Err(WorldModelError::DuplicateModelIdentity(identity.id.clone()));
            }
            ensure_fitted(model.training_dataset_id())?;
        }
        Ok(Self { models })
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn predict(
        &self,
        action_id: &str,
        current_state: &DataValue,
    ) -> Result<ModelPrediction, WorldModelError> {
        let predictions: Vec<ModelPrediction> = self
            .models
            .iter()
            .map(|model| model.predict(action_id, current_state))
            .collect::<Result<_, _>>()?;

        let mut contributors = Vec::new();
        let mut estimates_by_field: BTreeMap<String, Vec<FieldEstimate>> = BTreeMap::new();
        for prediction in predictions {
            contributors.extend(prediction.contributors);
            for (name, estimate) in prediction.fields {
                estimates_by_field.entry(name).or_default().push(estimate);
            }
        }

        let mut fields = BTreeMap::new();
        for (name, estimates) in estimates_by_field {
            let count = estimates.len() as f64;
            let mean = estimates.iter().map(|estimate| estimate.mean).sum::<f64>() / count;
            let between_model_variance = estimates
                .iter()
                .map(|estimate| (estimate.mean - mean).powi(2))
                .sum::<f64>()
                / count;
            let aleatoric_variance = estimates
                .iter()
                .map(|estimate| estimate.aleatoric_std_dev.powi(2))
                .sum::<f64>()
                / count;
            let within_epistemic_variance = estimates
                .iter()
                .map(|estimate| estimate.epistemic_std_dev.powi(2))
                .sum::<f64>()
                / count;
            let aleatoric_std_dev = aleatoric_variance.sqrt();
            let epistemic_std_dev = (between_model_variance + within_epistemic_variance).sqrt();
            if !mean.is_finite() || !aleatoric_std_dev.is_finite() || !epistemic_std_dev.is_finite()
            {
                return Err(WorldModelError::NonFinitePrediction);
            }
            fields.insert(
                name,
                FieldEstimate {
                    mean,
                    aleatoric_std_dev,
                    epistemic_std_dev,
                    sample_count: estimates
                        .iter()
                        .map(|estimate| estimate.sample_count)
                        .min()
                        .unwrap_or(0),
                    uncertainty_calibrated: estimates.len() == self.models.len()
                        && estimates
                            .iter()
                            .all(|estimate| estimate.uncertainty_calibrated),
                },
            );
        }

        Ok(ModelPrediction {
            contributors,
            fields,
        })
    }

    pub fn training_dataset_ids(&self) -> Vec<&str> {
        self.models
            .iter()
            .filter_map(|model| model.training_dataset_id())
            .collect()
    }
}

/// Restores a built-in predictive model from a versioned parameter artifact.
pub fn restore_predictive_model(
    artifact: &DataValue,
) -> Result<Box<dyn PredictiveModel>, WorldModelError> {
    let fields = artifact
        .as_object()
        .ok_or(WorldModelError::InvalidArtifact)?;
    if fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
        return Err(WorldModelError::UnsupportedArtifactVersion);
    }
    let model_type = match fields.get("model_type") {
        Some(DataValue::Text(value)) => value.as_str(),
        _ => return Err(WorldModelError::InvalidArtifact),
    };
    let id = match fields.get("id") {
        Some(DataValue::Text(value)) if !value.trim().is_empty() => value.clone(),
        _ => return Err(WorldModelError::InvalidArtifact),
    };
    let version = parse_artifact_u64(fields.get("model_version"))?;
    validate_identity(&id, version)?;
    let training_dataset_id = match fields.get("training_dataset_id") {
        Some(DataValue::Text(value)) if !value.trim().is_empty() => value.clone(),
        _ => return Err(WorldModelError::InvalidArtifact),
    };
    match model_type {
        "action_delta" => {
            if fields.len() != 6 {
                return Err(WorldModelError::InvalidArtifact);
            }
            let learner = NumericDeltaModel::from_artifact_value(
                fields
                    .get("learner")
                    .ok_or(WorldModelError::InvalidArtifact)?,
            )
            .map_err(map_learning_error)?;
            if learner.transitions_seen() == 0 {
                return Err(WorldModelError::InvalidArtifact);
            }
            Ok(Box::new(ActionDeltaModel {
                identity: ModelIdentity { id, version },
                learner,
                training_dataset_id: Some(training_dataset_id),
            }))
        }
        "contextual_linear" => {
            if fields.len() != 9 {
                return Err(WorldModelError::InvalidArtifact);
            }
            let training_samples = parse_artifact_u64(fields.get("training_samples"))?;
            let regularization = fields
                .get("regularization")
                .and_then(DataValue::as_f64)
                .ok_or(WorldModelError::InvalidArtifact)?;
            if training_samples == 0 || !regularization.is_finite() || regularization < 0.0 {
                return Err(WorldModelError::InvalidArtifact);
            }
            let feature_names = match fields.get("features") {
                Some(DataValue::Sequence(features)) => {
                    let mut names = Vec::new();
                    let mut unique = BTreeSet::new();
                    for feature in features {
                        let DataValue::Text(name) = feature else {
                            return Err(WorldModelError::InvalidArtifact);
                        };
                        if name.trim().is_empty() || !unique.insert(name.clone()) {
                            return Err(WorldModelError::InvalidArtifact);
                        }
                        names.push(name.clone());
                    }
                    names
                }
                _ => return Err(WorldModelError::InvalidArtifact),
            };
            let serialized_fields = match fields.get("fields") {
                Some(DataValue::Object(values)) if !values.is_empty() => values,
                _ => return Err(WorldModelError::InvalidArtifact),
            };
            let mut restored_fields = BTreeMap::new();
            for (name, value) in serialized_fields {
                if name.trim().is_empty() {
                    return Err(WorldModelError::InvalidArtifact);
                }
                let values = value.as_object().ok_or(WorldModelError::InvalidArtifact)?;
                if values.len() != 4 {
                    return Err(WorldModelError::InvalidArtifact);
                }
                let coefficients = match values.get("coefficients") {
                    Some(DataValue::Sequence(values))
                        if values.len() == feature_names.len() + 1 =>
                    {
                        values
                            .iter()
                            .map(|value| {
                                value
                                    .as_f64()
                                    .filter(|number| number.is_finite())
                                    .ok_or(WorldModelError::InvalidArtifact)
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    }
                    _ => return Err(WorldModelError::InvalidArtifact),
                };
                let residual_std_dev = values
                    .get("residual_std_dev")
                    .and_then(DataValue::as_f64)
                    .ok_or(WorldModelError::InvalidArtifact)?;
                let sample_count = parse_artifact_u64(values.get("sample_count"))?;
                let uncertainty_calibrated = match values.get("uncertainty_calibrated") {
                    Some(DataValue::Bool(value)) => *value,
                    _ => return Err(WorldModelError::InvalidArtifact),
                };
                if !residual_std_dev.is_finite()
                    || residual_std_dev < 0.0
                    || sample_count == 0
                    || sample_count > training_samples
                {
                    return Err(WorldModelError::InvalidArtifact);
                }
                restored_fields.insert(
                    name.clone(),
                    LinearFieldModel {
                        coefficients,
                        residual_std_dev,
                        sample_count,
                        uncertainty_calibrated,
                    },
                );
            }
            Ok(Box::new(ContextualLinearModel {
                identity: ModelIdentity { id, version },
                regularization,
                feature_names,
                fields: restored_fields,
                training_dataset_id: Some(training_dataset_id),
                training_samples,
            }))
        }
        _ => Err(WorldModelError::UnknownModelType(model_type.to_owned())),
    }
}

fn parse_artifact_u64(value: Option<&DataValue>) -> Result<u64, WorldModelError> {
    match value {
        Some(DataValue::Text(value)) => value.parse().map_err(|_| WorldModelError::InvalidArtifact),
        _ => Err(WorldModelError::InvalidArtifact),
    }
}

/// A predictive model that can be fitted from validated transition samples.
///
/// The default episode helper uses the journal's conservative transition
/// reconstruction rules; incomplete or mismatched actions are not trained on.
pub trait TrainablePredictiveModel: PredictiveModel {
    fn fit_samples(
        &mut self,
        dataset_id: &str,
        samples: &[TransitionSample],
    ) -> Result<ModelFitReport, WorldModelError>;

    fn fit_from_episode(
        &mut self,
        dataset_id: &str,
        records: &[ExperienceRecord],
        episode_id: &str,
    ) -> Result<ModelFitReport, WorldModelError> {
        let samples = extract_transition_samples(records, episode_id);
        if samples.is_empty() {
            return Err(WorldModelError::NoCompletedTransitions);
        }
        self.fit_samples(dataset_id, &samples)
    }
}

impl TrainablePredictiveModel for ActionDeltaModel {
    fn fit_samples(
        &mut self,
        dataset_id: &str,
        samples: &[TransitionSample],
    ) -> Result<ModelFitReport, WorldModelError> {
        ActionDeltaModel::fit(self, dataset_id, samples)
    }
}

impl TrainablePredictiveModel for ContextualLinearModel {
    fn fit_samples(
        &mut self,
        dataset_id: &str,
        samples: &[TransitionSample],
    ) -> Result<ModelFitReport, WorldModelError> {
        ContextualLinearModel::fit(self, dataset_id, samples)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelEvaluation {
    pub evaluation_dataset_id: String,
    pub evaluated_transitions: usize,
    pub predicted_fields: usize,
    pub mean_absolute_error: f64,
    pub root_mean_squared_error: f64,
    pub mean_predicted_std_dev: f64,
    pub calibrated_predictions: usize,
    pub empirical_coverage_95: Option<f64>,
}

pub fn evaluate_model<M: PredictiveModel + ?Sized>(
    model: &M,
    evaluation_dataset_id: &str,
    samples: &[TransitionSample],
) -> Result<ModelEvaluation, WorldModelError> {
    let training_id = model
        .training_dataset_id()
        .ok_or(WorldModelError::NotFitted)?;
    if training_id == evaluation_dataset_id {
        return Err(WorldModelError::EvaluationDatasetLeakage);
    }
    evaluate_predictions(evaluation_dataset_id, samples, |action, state| {
        model.predict(action, state)
    })
}

pub fn evaluate_ensemble(
    ensemble: &EnsembleWorldModel,
    evaluation_dataset_id: &str,
    samples: &[TransitionSample],
) -> Result<ModelEvaluation, WorldModelError> {
    if ensemble
        .training_dataset_ids()
        .contains(&evaluation_dataset_id)
    {
        return Err(WorldModelError::EvaluationDatasetLeakage);
    }
    evaluate_predictions(evaluation_dataset_id, samples, |action, state| {
        ensemble.predict(action, state)
    })
}

fn evaluate_predictions(
    evaluation_dataset_id: &str,
    samples: &[TransitionSample],
    mut predict: impl FnMut(&str, &DataValue) -> Result<ModelPrediction, WorldModelError>,
) -> Result<ModelEvaluation, WorldModelError> {
    if evaluation_dataset_id.trim().is_empty() {
        return Err(WorldModelError::InvalidEvaluationDatasetId);
    }
    if samples.is_empty() {
        return Err(WorldModelError::NoEvaluationSamples);
    }

    let mut absolute_error = 0.0;
    let mut squared_error = 0.0;
    let mut predicted_std_dev = 0.0;
    let mut predicted_fields = 0usize;
    let mut calibrated_predictions = 0usize;
    let mut covered_predictions = 0usize;

    for sample in samples {
        let prediction = predict(&sample.action_id, &sample.before)?;
        for (name, estimate) in prediction.fields {
            let Some(actual) = sample.after.get(&name).and_then(DataValue::as_f64) else {
                continue;
            };
            let error = estimate.mean - actual;
            let std_dev = estimate.total_std_dev();
            if !error.is_finite() || !std_dev.is_finite() {
                return Err(WorldModelError::NonFinitePrediction);
            }
            absolute_error += error.abs();
            squared_error += error * error;
            predicted_std_dev += std_dev;
            predicted_fields += 1;
            if estimate.uncertainty_calibrated {
                calibrated_predictions += 1;
                if error.abs() <= 1.96 * std_dev {
                    covered_predictions += 1;
                }
            }
        }
    }

    if predicted_fields == 0 {
        return Err(WorldModelError::NoEvaluationPredictions);
    }
    let count = predicted_fields as f64;
    Ok(ModelEvaluation {
        evaluation_dataset_id: evaluation_dataset_id.to_owned(),
        evaluated_transitions: samples.len(),
        predicted_fields,
        mean_absolute_error: absolute_error / count,
        root_mean_squared_error: (squared_error / count).sqrt(),
        mean_predicted_std_dev: predicted_std_dev / count,
        calibrated_predictions,
        empirical_coverage_95: if calibrated_predictions == 0 {
            None
        } else {
            Some(covered_predictions as f64 / calibrated_predictions as f64)
        },
    })
}

fn validate_identity(id: &str, version: u64) -> Result<(), WorldModelError> {
    if id.trim().is_empty() || version == 0 {
        return Err(WorldModelError::InvalidModelIdentity);
    }
    Ok(())
}

fn validate_training_set(
    dataset_id: &str,
    samples: &[TransitionSample],
) -> Result<(), WorldModelError> {
    if dataset_id.trim().is_empty() {
        return Err(WorldModelError::InvalidTrainingDatasetId);
    }
    if samples.is_empty() {
        return Err(WorldModelError::NoTrainingSamples);
    }
    Ok(())
}

fn ensure_fitted(dataset_id: Option<&str>) -> Result<(), WorldModelError> {
    if dataset_id.is_none() {
        Err(WorldModelError::NotFitted)
    } else {
        Ok(())
    }
}

fn to_u64(value: usize) -> Result<u64, WorldModelError> {
    u64::try_from(value).map_err(|_| WorldModelError::SampleCountOverflow)
}

fn map_learning_error(error: LearningError) -> WorldModelError {
    WorldModelError::Learning(error.to_string())
}

fn action_feature(action_id: &str) -> String {
    format!("action:{action_id}")
}

fn state_feature(field: &str) -> String {
    format!("state:{field}")
}

fn feature_vector(
    feature_names: &[String],
    action_id: &str,
    state: &DataValue,
) -> Result<Vec<f64>, WorldModelError> {
    let object = state
        .as_object()
        .ok_or(WorldModelError::ExpectedObjectState)?;
    let mut vector = vec![0.0; feature_names.len() + 1];
    vector[0] = 1.0;
    for (index, feature) in feature_names.iter().enumerate() {
        if feature == &action_feature(action_id) {
            vector[index + 1] = 1.0;
        } else if let Some(field) = feature.strip_prefix("state:") {
            if let Some(value) = object.get(field).and_then(DataValue::as_f64) {
                vector[index + 1] = value;
            }
        }
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(WorldModelError::NonFinitePrediction);
    }
    Ok(vector)
}

fn fit_linear_field(
    rows: &[(Vec<f64>, f64)],
    regularization: f64,
) -> Result<LinearFieldModel, WorldModelError> {
    if rows.is_empty() {
        return Err(WorldModelError::NoTrainableFields);
    }
    let dimensions = rows[0].0.len();
    if dimensions == 0
        || rows.iter().any(|(features, target)| {
            features.len() != dimensions
                || !target.is_finite()
                || features.iter().any(|value| !value.is_finite())
        })
    {
        return Err(WorldModelError::InvalidTrainingMatrix);
    }

    let mut gram = vec![vec![0.0; dimensions]; dimensions];
    let mut cross = vec![0.0; dimensions];
    for (features, target) in rows {
        for row in 0..dimensions {
            cross[row] += features[row] * target;
            for column in 0..dimensions {
                gram[row][column] += features[row] * features[column];
            }
        }
    }
    for (diagonal, row) in gram.iter_mut().enumerate().skip(1) {
        row[diagonal] += regularization;
    }

    let coefficients = solve_linear_system(gram, cross)?;
    let mut squared_residuals = 0.0;
    for (features, target) in rows {
        let prediction = dot(&coefficients, features)?;
        squared_residuals += (prediction - target).powi(2);
    }
    let residual_std_dev = (squared_residuals / rows.len() as f64).sqrt();
    if !residual_std_dev.is_finite() {
        return Err(WorldModelError::NonFinitePrediction);
    }

    let sample_count = to_u64(rows.len())?;
    Ok(LinearFieldModel {
        coefficients,
        residual_std_dev,
        sample_count,
        uncertainty_calibrated: rows.len() > dimensions,
    })
}

fn solve_linear_system(
    mut matrix: Vec<Vec<f64>>,
    mut right_hand_side: Vec<f64>,
) -> Result<Vec<f64>, WorldModelError> {
    let dimensions = right_hand_side.len();
    if matrix.len() != dimensions || matrix.iter().any(|row| row.len() != dimensions) {
        return Err(WorldModelError::InvalidTrainingMatrix);
    }

    for pivot in 0..dimensions {
        let best_row = (pivot..dimensions)
            .max_by(|left, right| {
                matrix[*left][pivot]
                    .abs()
                    .total_cmp(&matrix[*right][pivot].abs())
            })
            .ok_or(WorldModelError::SingularTrainingMatrix)?;
        if matrix[best_row][pivot].abs() <= f64::EPSILON {
            return Err(WorldModelError::SingularTrainingMatrix);
        }
        matrix.swap(pivot, best_row);
        right_hand_side.swap(pivot, best_row);

        let scale = matrix[pivot][pivot];
        for value in matrix[pivot].iter_mut().skip(pivot) {
            *value /= scale;
        }
        right_hand_side[pivot] /= scale;

        for row in 0..dimensions {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..dimensions {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            right_hand_side[row] -= factor * right_hand_side[pivot];
        }
    }

    if right_hand_side.iter().any(|value| !value.is_finite()) {
        return Err(WorldModelError::NonFinitePrediction);
    }
    Ok(right_hand_side)
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, WorldModelError> {
    if left.len() != right.len() {
        return Err(WorldModelError::InvalidTrainingMatrix);
    }
    let result = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(WorldModelError::NonFinitePrediction)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorldModelError {
    InvalidModelIdentity,
    InvalidTrainingDatasetId,
    InvalidEvaluationDatasetId,
    InvalidRegularization,
    NoTrainingSamples,
    NoEvaluationSamples,
    NoTrainableFields,
    NoEvaluationPredictions,
    NotFitted,
    ExpectedObjectState,
    NonFiniteTrainingData,
    NonFinitePrediction,
    InvalidTrainingMatrix,
    SingularTrainingMatrix,
    SampleCountOverflow,
    EvaluationDatasetLeakage,
    EmptyEnsemble,
    DuplicateModelIdentity(String),
    Learning(String),
    InvalidArtifact,
    UnsupportedArtifactVersion,
    UnknownModelType(String),
}

impl std::fmt::Display for WorldModelError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidModelIdentity => write!(
                formatter,
                "model identity must have a non-empty id and positive version"
            ),
            Self::InvalidTrainingDatasetId => {
                write!(formatter, "training dataset id must not be empty")
            }
            Self::InvalidEvaluationDatasetId => {
                write!(formatter, "evaluation dataset id must not be empty")
            }
            Self::InvalidRegularization => write!(
                formatter,
                "ridge regularization must be finite and non-negative"
            ),
            Self::NoTrainingSamples => {
                write!(formatter, "at least one training transition is required")
            }
            Self::NoEvaluationSamples => {
                write!(formatter, "at least one held-out transition is required")
            }
            Self::NoTrainableFields => write!(
                formatter,
                "training data contains no numeric transition targets"
            ),
            Self::NoEvaluationPredictions => {
                write!(formatter, "model produced no evaluable numeric predictions")
            }
            Self::NotFitted => write!(formatter, "model has not been fitted"),
            Self::ExpectedObjectState => write!(formatter, "world-model states must be objects"),
            Self::NonFiniteTrainingData => write!(
                formatter,
                "training data contains a non-finite numeric value"
            ),
            Self::NonFinitePrediction => {
                write!(formatter, "model produced a non-finite prediction")
            }
            Self::InvalidTrainingMatrix => write!(
                formatter,
                "training matrix dimensions or values are invalid"
            ),
            Self::SingularTrainingMatrix => write!(
                formatter,
                "training matrix is singular; use positive regularization or more varied data"
            ),
            Self::SampleCountOverflow => {
                write!(formatter, "sample count exceeds the supported range")
            }
            Self::EvaluationDatasetLeakage => write!(
                formatter,
                "evaluation dataset id must differ from the model's training dataset id"
            ),
            Self::EmptyEnsemble => write!(
                formatter,
                "an ensemble must contain at least one fitted model"
            ),
            Self::DuplicateModelIdentity(id) => {
                write!(formatter, "ensemble contains duplicate model identity {id}")
            }
            Self::Learning(message) => write!(formatter, "baseline learner failed: {message}"),
            Self::InvalidArtifact => write!(
                formatter,
                "predictive model artifact is malformed or invalid"
            ),
            Self::UnsupportedArtifactVersion => write!(
                formatter,
                "predictive model artifact version is unsupported"
            ),
            Self::UnknownModelType(model_type) => write!(
                formatter,
                "predictive model type {model_type} is not supported"
            ),
        }
    }
}

impl std::error::Error for WorldModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(action: &str, before: f64, after: f64) -> TransitionSample {
        TransitionSample {
            action_id: action.to_owned(),
            before: DataValue::object([("position".to_owned(), DataValue::Number(before))]),
            after: DataValue::object([("position".to_owned(), DataValue::Number(after))]),
        }
    }

    #[test]
    fn statistical_model_predicts_mean_delta_and_uncertainty() {
        let mut model = ActionDeltaModel::new("delta", 1).expect("identity");
        model
            .fit(
                "training-a",
                &[sample("advance", 0.0, 2.0), sample("advance", 4.0, 6.0)],
            )
            .expect("fit");
        let prediction = model
            .predict(
                "advance",
                &DataValue::object([("position".to_owned(), DataValue::Number(10.0))]),
            )
            .expect("predict");
        let estimate = prediction.fields.get("position").expect("position");
        assert_eq!(estimate.mean, 12.0);
        assert_eq!(estimate.sample_count, 2);
        assert!(estimate.uncertainty_calibrated);
    }

    #[test]
    fn contextual_model_learns_from_state_features() {
        let mut model = ContextualLinearModel::new("contextual", 1, 1e-6).expect("model");
        let samples = vec![
            TransitionSample {
                action_id: "move".to_owned(),
                before: DataValue::object([
                    ("position".to_owned(), DataValue::Number(0.0)),
                    ("velocity".to_owned(), DataValue::Number(1.0)),
                ]),
                after: DataValue::object([
                    ("position".to_owned(), DataValue::Number(1.0)),
                    ("velocity".to_owned(), DataValue::Number(1.0)),
                ]),
            },
            TransitionSample {
                action_id: "move".to_owned(),
                before: DataValue::object([
                    ("position".to_owned(), DataValue::Number(2.0)),
                    ("velocity".to_owned(), DataValue::Number(3.0)),
                ]),
                after: DataValue::object([
                    ("position".to_owned(), DataValue::Number(5.0)),
                    ("velocity".to_owned(), DataValue::Number(3.0)),
                ]),
            },
            TransitionSample {
                action_id: "move".to_owned(),
                before: DataValue::object([
                    ("position".to_owned(), DataValue::Number(8.0)),
                    ("velocity".to_owned(), DataValue::Number(2.0)),
                ]),
                after: DataValue::object([
                    ("position".to_owned(), DataValue::Number(10.0)),
                    ("velocity".to_owned(), DataValue::Number(2.0)),
                ]),
            },
            TransitionSample {
                action_id: "move".to_owned(),
                before: DataValue::object([
                    ("position".to_owned(), DataValue::Number(10.0)),
                    ("velocity".to_owned(), DataValue::Number(4.0)),
                ]),
                after: DataValue::object([
                    ("position".to_owned(), DataValue::Number(14.0)),
                    ("velocity".to_owned(), DataValue::Number(4.0)),
                ]),
            },
        ];
        model.fit("training", &samples).expect("fit");
        let prediction = model
            .predict(
                "move",
                &DataValue::object([
                    ("position".to_owned(), DataValue::Number(20.0)),
                    ("velocity".to_owned(), DataValue::Number(5.0)),
                ]),
            )
            .expect("predict");
        let position = prediction.fields.get("position").expect("position");
        assert!((position.mean - 25.0).abs() < 1e-4);
    }

    #[test]
    fn ensemble_disagreement_is_reported_as_epistemic_uncertainty() {
        let mut first = ActionDeltaModel::new("first", 1).expect("model");
        first
            .fit(
                "training-a",
                &[sample("advance", 0.0, 1.0), sample("advance", 1.0, 2.0)],
            )
            .expect("fit");
        let mut second = ActionDeltaModel::new("second", 1).expect("model");
        second
            .fit(
                "training-b",
                &[sample("advance", 0.0, 3.0), sample("advance", 1.0, 4.0)],
            )
            .expect("fit");

        let ensemble =
            EnsembleWorldModel::new(vec![Box::new(first), Box::new(second)]).expect("ensemble");
        let prediction = ensemble
            .predict(
                "advance",
                &DataValue::object([("position".to_owned(), DataValue::Number(4.0))]),
            )
            .expect("predict");
        let estimate = prediction.fields.get("position").expect("position");
        assert_eq!(estimate.mean, 6.0);
        assert_eq!(estimate.epistemic_std_dev, 1.0);
    }

    #[test]
    fn held_out_evaluation_rejects_training_dataset_reuse() {
        let mut model = ActionDeltaModel::new("delta", 1).expect("model");
        model
            .fit(
                "training",
                &[sample("advance", 0.0, 1.0), sample("advance", 1.0, 2.0)],
            )
            .expect("fit");
        assert_eq!(
            evaluate_model(&model, "training", &[sample("advance", 2.0, 3.0)]),
            Err(WorldModelError::EvaluationDatasetLeakage)
        );
        let report =
            evaluate_model(&model, "validation", &[sample("advance", 2.0, 3.0)]).expect("eval");
        assert_eq!(report.predicted_fields, 1);
        assert_eq!(report.mean_absolute_error, 0.0);
    }
}
