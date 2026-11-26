//! Split-conformal calibration for per-field predictive intervals.
//!
//! Calibration uses held-out absolute residuals and a finite-sample order
//! statistic. Marginal coverage relies on exchangeability between calibration
//! and future examples; dataset IDs are provenance labels, not proof of
//! independence. The intervals are per-field, not simultaneous across fields.

use crate::learning::TransitionSample;
use crate::value::DataValue;
use crate::world_model::{ModelIdentity, ModelPrediction, PredictiveModel};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct FieldCalibration {
    pub calibration_samples: usize,
    pub absolute_error_quantile: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictionInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub target_coverage: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedPrediction {
    pub prediction: ModelPrediction,
    pub intervals: BTreeMap<String, PredictionInterval>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConformalCalibrator {
    pub model_identity: ModelIdentity,
    pub training_dataset_id: String,
    pub calibration_dataset_id: String,
    pub target_coverage: f64,
    pub fields: BTreeMap<String, FieldCalibration>,
}

impl ConformalCalibrator {
    /// Fits per-field absolute-residual quantiles using an independent calibration split.
    ///
    /// For each field, the quantile rank is ceil((n + 1) * coverage), as in
    /// split conformal prediction. Fields with too few residuals are rejected
    /// rather than assigned an infinite or misleading interval.
    pub fn fit(
        model: &dyn PredictiveModel,
        calibration_dataset_id: &str,
        samples: &[TransitionSample],
        target_coverage: f64,
    ) -> Result<Self, ConformalCalibrationError> {
        if calibration_dataset_id.trim().is_empty() {
            return Err(ConformalCalibrationError::InvalidDatasetId);
        }
        if !target_coverage.is_finite() || !(0.0..1.0).contains(&target_coverage) {
            return Err(ConformalCalibrationError::InvalidCoverage);
        }
        if samples.is_empty() {
            return Err(ConformalCalibrationError::NoCalibrationSamples);
        }
        let training_dataset_id = model
            .training_dataset_id()
            .filter(|id| !id.trim().is_empty())
            .ok_or(ConformalCalibrationError::ModelNotFitted)?
            .to_owned();
        if training_dataset_id == calibration_dataset_id {
            return Err(ConformalCalibrationError::DatasetLeakage);
        }

        let mut residuals: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for sample in samples {
            sample
                .before
                .validate_finite()
                .map_err(|_| ConformalCalibrationError::NonFiniteCalibrationData)?;
            sample
                .after
                .validate_finite()
                .map_err(|_| ConformalCalibrationError::NonFiniteCalibrationData)?;
            let prediction = model
                .predict(&sample.action_id, &sample.before)
                .map_err(|error| {
                    ConformalCalibrationError::PredictionFailed(error.to_string())
                })?;
            let observed = sample
                .after
                .as_object()
                .ok_or(ConformalCalibrationError::ExpectedObjectState)?;
            for (field, estimate) in &prediction.fields {
                let Some(actual) = observed.get(field).and_then(DataValue::as_f64) else {
                    continue;
                };
                let residual = (actual - estimate.mean).abs();
                if !residual.is_finite() {
                    return Err(ConformalCalibrationError::NumericalOverflow);
                }
                residuals.entry(field.clone()).or_default().push(residual);
            }
        }
        if residuals.is_empty() {
            return Err(ConformalCalibrationError::NoCalibratableFields);
        }

        let mut fields = BTreeMap::new();
        for (field, mut scores) in residuals {
            let required = minimum_sample_count(target_coverage)?;
            if scores.len() < required {
                return Err(ConformalCalibrationError::InsufficientFieldSamples {
                    field,
                    required,
                    actual: scores.len(),
                });
            }
            scores.sort_by(f64::total_cmp);
            let rank = ((scores.len() as f64 + 1.0) * target_coverage).ceil() as usize;
            if rank == 0 || rank > scores.len() {
                return Err(ConformalCalibrationError::InvalidQuantileRank);
            }
            let absolute_error_quantile = scores[rank - 1];
            if !absolute_error_quantile.is_finite() {
                return Err(ConformalCalibrationError::NumericalOverflow);
            }
            fields.insert(
                field,
                FieldCalibration {
                    calibration_samples: scores.len(),
                    absolute_error_quantile,
                },
            );
        }

        Ok(Self {
            model_identity: model.identity().clone(),
            training_dataset_id,
            calibration_dataset_id: calibration_dataset_id.to_owned(),
            target_coverage,
            fields,
        })
    }

    /// Predicts and attaches marginal per-field conformal intervals.
    pub fn predict(
        &self,
        model: &dyn PredictiveModel,
        action_id: &str,
        current_state: &DataValue,
    ) -> Result<CalibratedPrediction, ConformalCalibrationError> {
        self.validate()?;
        if model.identity() != &self.model_identity
            || model.training_dataset_id() != Some(self.training_dataset_id.as_str())
        {
            return Err(ConformalCalibrationError::ModelIdentityMismatch);
        }
        let prediction = model
            .predict(action_id, current_state)
            .map_err(|error| ConformalCalibrationError::PredictionFailed(error.to_string()))?;
        let mut intervals = BTreeMap::new();
        for (field, estimate) in &prediction.fields {
            let calibration = self
                .fields
                .get(field)
                .ok_or_else(|| ConformalCalibrationError::MissingFieldCalibration(field.clone()))?;
            let lower_bound = estimate.mean - calibration.absolute_error_quantile;
            let upper_bound = estimate.mean + calibration.absolute_error_quantile;
            if !lower_bound.is_finite() || !upper_bound.is_finite() {
                return Err(ConformalCalibrationError::NumericalOverflow);
            }
            intervals.insert(
                field.clone(),
                PredictionInterval {
                    lower_bound,
                    upper_bound,
                    target_coverage: self.target_coverage,
                },
            );
        }
        if intervals.is_empty() {
            return Err(ConformalCalibrationError::NoCalibratableFields);
        }
        Ok(CalibratedPrediction {
            prediction,
            intervals,
        })
    }

    pub fn validate(&self) -> Result<(), ConformalCalibrationError> {
        if self.model_identity.id.trim().is_empty()
            || self.model_identity.version == 0
            || self.training_dataset_id.trim().is_empty()
            || self.calibration_dataset_id.trim().is_empty()
            || self.training_dataset_id == self.calibration_dataset_id
            || !self.target_coverage.is_finite()
            || !(0.0..1.0).contains(&self.target_coverage)
            || self.fields.is_empty()
        {
            return Err(ConformalCalibrationError::InvalidArtifact);
        }
        let required = minimum_sample_count(self.target_coverage)
            .map_err(|_| ConformalCalibrationError::InvalidArtifact)?;
        for (field, calibration) in &self.fields {
            if field.trim().is_empty()
                || calibration.calibration_samples < required
                || !calibration.absolute_error_quantile.is_finite()
                || calibration.absolute_error_quantile < 0.0
            {
                return Err(ConformalCalibrationError::InvalidArtifact);
            }
        }
        Ok(())
    }

    pub fn to_artifact_value(&self) -> DataValue {
        let fields = DataValue::Object(
            self.fields
                .iter()
                .map(|(name, calibration)| {
                    (
                        name.clone(),
                        DataValue::object([
                            (
                                "calibration_samples".to_owned(),
                                DataValue::Text(calibration.calibration_samples.to_string()),
                            ),
                            (
                                "absolute_error_quantile".to_owned(),
                                DataValue::Number(calibration.absolute_error_quantile),
                            ),
                        ]),
                    )
                })
                .collect(),
        );
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            ("model_id".to_owned(), DataValue::Text(self.model_identity.id.clone())),
            (
                "model_version".to_owned(),
                DataValue::Text(self.model_identity.version.to_string()),
            ),
            (
                "training_dataset_id".to_owned(),
                DataValue::Text(self.training_dataset_id.clone()),
            ),
            (
                "calibration_dataset_id".to_owned(),
                DataValue::Text(self.calibration_dataset_id.clone()),
            ),
            ("target_coverage".to_owned(), DataValue::Number(self.target_coverage)),
            ("fields".to_owned(), fields),
        ])
    }

    pub fn from_artifact_value(value: &DataValue) -> Result<Self, ConformalCalibrationError> {
        let object = value
            .as_object()
            .ok_or(ConformalCalibrationError::InvalidArtifact)?;
        if object.len() != 7
            || object.get("artifact_version") != Some(&DataValue::Integer(1))
        {
            return Err(ConformalCalibrationError::InvalidArtifact);
        }
        let text = |name: &str| match object.get(name) {
            Some(DataValue::Text(value)) => Ok(value.clone()),
            _ => Err(ConformalCalibrationError::InvalidArtifact),
        };
        let model_version = text("model_version")?
            .parse()
            .map_err(|_| ConformalCalibrationError::InvalidArtifact)?;
        let target_coverage = object
            .get("target_coverage")
            .and_then(DataValue::as_f64)
            .filter(|value| value.is_finite())
            .ok_or(ConformalCalibrationError::InvalidArtifact)?;
        let encoded_fields = object
            .get("fields")
            .and_then(DataValue::as_object)
            .ok_or(ConformalCalibrationError::InvalidArtifact)?;
        let mut fields = BTreeMap::new();
        for (name, value) in encoded_fields {
            let field = value
                .as_object()
                .ok_or(ConformalCalibrationError::InvalidArtifact)?;
            if field.len() != 2 {
                return Err(ConformalCalibrationError::InvalidArtifact);
            }
            let calibration_samples = match field.get("calibration_samples") {
                Some(DataValue::Text(value)) => value
                    .parse()
                    .map_err(|_| ConformalCalibrationError::InvalidArtifact)?,
                _ => return Err(ConformalCalibrationError::InvalidArtifact),
            };
            let absolute_error_quantile = field
                .get("absolute_error_quantile")
                .and_then(DataValue::as_f64)
                .filter(|value| value.is_finite())
                .ok_or(ConformalCalibrationError::InvalidArtifact)?;
            fields.insert(
                name.clone(),
                FieldCalibration {
                    calibration_samples,
                    absolute_error_quantile,
                },
            );
        }
        let calibrator = Self {
            model_identity: ModelIdentity {
                id: text("model_id")?,
                version: model_version,
            },
            training_dataset_id: text("training_dataset_id")?,
            calibration_dataset_id: text("calibration_dataset_id")?,
            target_coverage,
            fields,
        };
        calibrator.validate()?;
        Ok(calibrator)
    }
}

fn minimum_sample_count(target_coverage: f64) -> Result<usize, ConformalCalibrationError> {
    if !target_coverage.is_finite() || !(0.0..1.0).contains(&target_coverage) {
        return Err(ConformalCalibrationError::InvalidCoverage);
    }
    let required = (target_coverage / (1.0 - target_coverage)).ceil();
    if !required.is_finite() || required > usize::MAX as f64 {
        return Err(ConformalCalibrationError::InvalidCoverage);
    }
    Ok((required as usize).max(1))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConformalCalibrationError {
    InvalidDatasetId,
    InvalidCoverage,
    NoCalibrationSamples,
    ModelNotFitted,
    DatasetLeakage,
    NonFiniteCalibrationData,
    ExpectedObjectState,
    PredictionFailed(String),
    NoCalibratableFields,
    InsufficientFieldSamples {
        field: String,
        required: usize,
        actual: usize,
    },
    InvalidQuantileRank,
    ModelIdentityMismatch,
    MissingFieldCalibration(String),
    NumericalOverflow,
    InvalidArtifact,
}

impl std::fmt::Display for ConformalCalibrationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDatasetId => write!(formatter, "calibration dataset ID must not be empty"),
            Self::InvalidCoverage => write!(formatter, "target coverage must be finite and strictly between zero and one"),
            Self::NoCalibrationSamples => write!(formatter, "at least one calibration transition is required"),
            Self::ModelNotFitted => write!(formatter, "predictive model has no training dataset provenance"),
            Self::DatasetLeakage => write!(formatter, "calibration dataset ID must differ from model training dataset ID"),
            Self::NonFiniteCalibrationData => write!(formatter, "calibration data contains non-finite numeric values"),
            Self::ExpectedObjectState => write!(formatter, "calibration target state must be an object"),
            Self::PredictionFailed(message) => write!(formatter, "model prediction failed during calibration: {message}"),
            Self::NoCalibratableFields => write!(formatter, "calibration contains no numeric fields with observed targets"),
            Self::InsufficientFieldSamples { field, required, actual } => write!(formatter, "field {field} needs at least {required} calibration residuals, received {actual}"),
            Self::InvalidQuantileRank => write!(formatter, "finite-sample conformal quantile rank is invalid"),
            Self::ModelIdentityMismatch => write!(formatter, "calibrator does not match model identity and training dataset"),
            Self::MissingFieldCalibration(field) => write!(formatter, "no calibration residuals are available for predicted field {field}"),
            Self::NumericalOverflow => write!(formatter, "calibration residual or interval overflowed"),
            Self::InvalidArtifact => write!(formatter, "serialized conformal calibrator is invalid"),
        }
    }
}

impl std::error::Error for ConformalCalibrationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_model::ActionDeltaModel;

    fn sample(before: f64, after: f64) -> TransitionSample {
        TransitionSample {
            action_id: "advance".to_owned(),
            before: DataValue::object([("position".to_owned(), DataValue::Number(before))]),
            after: DataValue::object([("position".to_owned(), DataValue::Number(after))]),
        }
    }

    fn fitted_model() -> ActionDeltaModel {
        let mut model = ActionDeltaModel::new("calibrated-delta", 1).expect("model");
        model
            .fit("training-v1", &[sample(0.0, 1.0), sample(5.0, 6.0)])
            .expect("fit");
        model
    }

    #[test]
    fn split_conformal_quantile_produces_finite_sample_interval() {
        let model = fitted_model();
        let calibrator = ConformalCalibrator::fit(
            &model,
            "calibration-v1",
            &[sample(10.0, 11.0), sample(20.0, 22.0), sample(30.0, 33.0), sample(40.0, 44.0)],
            0.75,
        )
        .expect("fit calibrator");
        let calibrated = calibrator
            .predict(
                &model,
                "advance",
                &DataValue::object([("position".to_owned(), DataValue::Number(100.0))]),
            )
            .expect("calibrated prediction");
        let interval = calibrated.intervals.get("position").expect("interval");
        assert_eq!(interval.target_coverage, 0.75);
        assert_eq!(interval.lower_bound, 98.0);
        assert_eq!(interval.upper_bound, 104.0);
    }

    #[test]
    fn rejects_leakage_and_insufficient_finite_sample_count() {
        let model = fitted_model();
        assert_eq!(
            ConformalCalibrator::fit(&model, "training-v1", &[sample(10.0, 11.0)], 0.75),
            Err(ConformalCalibrationError::DatasetLeakage)
        );
        let insufficient: Vec<TransitionSample> =
            (0..18).map(|index| sample(index as f64, index as f64 + 1.0)).collect();
        assert_eq!(
            ConformalCalibrator::fit(
                &model,
                "calibration-v1",
                &insufficient,
                0.95,
            ),
            Err(ConformalCalibrationError::InsufficientFieldSamples {
                field: "position".to_owned(),
                required: 19,
                actual: 18,
            })
        );
    }

    #[test]
    fn calibrator_artifact_round_trips_and_is_bound_to_model_version() {
        let model = fitted_model();
        let calibrator = ConformalCalibrator::fit(
            &model,
            "calibration-v1",
            &[sample(10.0, 11.0), sample(20.0, 22.0), sample(30.0, 33.0), sample(40.0, 44.0)],
            0.75,
        )
        .expect("fit calibrator");
        let restored =
            ConformalCalibrator::from_artifact_value(&calibrator.to_artifact_value())
                .expect("restore calibrator");
        assert_eq!(restored, calibrator);

        let mut other_model = ActionDeltaModel::new("other-model", 1).expect("other");
        other_model
            .fit("training-v1", &[sample(0.0, 1.0), sample(5.0, 6.0)])
            .expect("fit other");
        assert_eq!(
            restored.predict(
                &other_model,
                "advance",
                &DataValue::object([("position".to_owned(), DataValue::Number(10.0))]),
            ),
            Err(ConformalCalibrationError::ModelIdentityMismatch)
        );
    }
}
