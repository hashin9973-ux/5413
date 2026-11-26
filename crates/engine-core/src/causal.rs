//! Paired randomized-intervention effect estimation.
//!
//! This module computes a transparent paired difference-in-means estimator.
//! The caller must attest that assignment was randomized and that the pairs,
//! outcome measurement, and no-interference assumptions are valid. The engine
//! cannot verify those design assumptions from outcome values alone.

use crate::value::DataValue;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignmentDesign {
    Randomized,
    Observational,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairedOutcome {
    pub pair_id: String,
    pub control_outcome: f64,
    pub treatment_outcome: f64,
}

impl PairedOutcome {
    fn validate(&self) -> Result<(), CausalError> {
        if self.pair_id.trim().is_empty()
            || !self.control_outcome.is_finite()
            || !self.treatment_outcome.is_finite()
        {
            return Err(CausalError::InvalidPair);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairedRandomizedStudy {
    pub experiment_id: String,
    pub dataset_id: String,
    pub treatment_label: String,
    pub control_label: String,
    pub outcome_label: String,
    pub assignment_design: AssignmentDesign,
    pub pairs: Vec<PairedOutcome>,
}

impl PairedRandomizedStudy {
    pub fn validate(&self) -> Result<(), CausalError> {
        if self.assignment_design != AssignmentDesign::Randomized {
            return Err(CausalError::RandomizationRequired);
        }
        if self.experiment_id.trim().is_empty()
            || self.dataset_id.trim().is_empty()
            || self.treatment_label.trim().is_empty()
            || self.control_label.trim().is_empty()
            || self.outcome_label.trim().is_empty()
            || self.treatment_label == self.control_label
            || self.pairs.len() < 2
        {
            return Err(CausalError::InvalidStudy);
        }
        let mut pair_ids = BTreeSet::new();
        for pair in &self.pairs {
            pair.validate()?;
            if !pair_ids.insert(pair.pair_id.as_str()) {
                return Err(CausalError::DuplicatePair);
            }
        }
        Ok(())
    }

    /// Estimates the average paired treatment effect: treatment minus control.
    ///
    /// The interval uses a 1.96 normal critical value. It is an approximation,
    /// not an exact small-sample confidence interval or a randomization test.
    pub fn estimate(&self) -> Result<CausalEffectEstimate, CausalError> {
        self.validate()?;
        let count = self.pairs.len() as f64;
        let differences: Vec<f64> = self
            .pairs
            .iter()
            .map(|pair| pair.treatment_outcome - pair.control_outcome)
            .collect();
        let mean_effect = differences.iter().sum::<f64>() / count;
        let treatment_mean = self
            .pairs
            .iter()
            .map(|pair| pair.treatment_outcome)
            .sum::<f64>()
            / count;
        let control_mean = self
            .pairs
            .iter()
            .map(|pair| pair.control_outcome)
            .sum::<f64>()
            / count;
        let sum_squared = differences
            .iter()
            .map(|difference| (difference - mean_effect).powi(2))
            .sum::<f64>();
        let sample_variance = sum_squared / (count - 1.0);
        let standard_error = (sample_variance / count).sqrt();
        let margin = 1.96 * standard_error;
        let lower = mean_effect - margin;
        let upper = mean_effect + margin;
        if !mean_effect.is_finite()
            || !treatment_mean.is_finite()
            || !control_mean.is_finite()
            || !standard_error.is_finite()
            || !lower.is_finite()
            || !upper.is_finite()
        {
            return Err(CausalError::NumericalOverflow);
        }
        Ok(CausalEffectEstimate {
            experiment_id: self.experiment_id.clone(),
            dataset_id: self.dataset_id.clone(),
            treatment_label: self.treatment_label.clone(),
            control_label: self.control_label.clone(),
            outcome_label: self.outcome_label.clone(),
            pair_count: self.pairs.len(),
            treatment_mean,
            control_mean,
            mean_effect,
            standard_error,
            normal_approximation_95_percent_interval: (lower, upper),
        })
    }

    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "experiment_id".to_owned(),
                DataValue::Text(self.experiment_id.clone()),
            ),
            (
                "dataset_id".to_owned(),
                DataValue::Text(self.dataset_id.clone()),
            ),
            (
                "treatment_label".to_owned(),
                DataValue::Text(self.treatment_label.clone()),
            ),
            (
                "control_label".to_owned(),
                DataValue::Text(self.control_label.clone()),
            ),
            (
                "outcome_label".to_owned(),
                DataValue::Text(self.outcome_label.clone()),
            ),
            (
                "assignment_design".to_owned(),
                DataValue::Text(
                    match self.assignment_design {
                        AssignmentDesign::Randomized => "randomized",
                        AssignmentDesign::Observational => "observational",
                    }
                    .to_owned(),
                ),
            ),
            (
                "pairs".to_owned(),
                DataValue::Sequence(
                    self.pairs
                        .iter()
                        .map(|pair| {
                            DataValue::object([
                                ("pair_id".to_owned(), DataValue::Text(pair.pair_id.clone())),
                                (
                                    "control_outcome".to_owned(),
                                    DataValue::Number(pair.control_outcome),
                                ),
                                (
                                    "treatment_outcome".to_owned(),
                                    DataValue::Number(pair.treatment_outcome),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }

    pub fn from_artifact_value(value: &DataValue) -> Result<Self, CausalError> {
        let fields = value
            .as_object()
            .ok_or(CausalError::InvalidSerializedStudy)?;
        if fields.len() != 8 || fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
            return Err(CausalError::InvalidSerializedStudy);
        }
        let text = |name: &str| match fields.get(name) {
            Some(DataValue::Text(value)) => Ok(value.clone()),
            _ => Err(CausalError::InvalidSerializedStudy),
        };
        let assignment_design = match text("assignment_design")?.as_str() {
            "randomized" => AssignmentDesign::Randomized,
            "observational" => AssignmentDesign::Observational,
            _ => return Err(CausalError::InvalidSerializedStudy),
        };
        let pairs = match fields.get("pairs") {
            Some(DataValue::Sequence(values)) => values
                .iter()
                .map(|value| {
                    let fields = value
                        .as_object()
                        .ok_or(CausalError::InvalidSerializedStudy)?;
                    if fields.len() != 3 {
                        return Err(CausalError::InvalidSerializedStudy);
                    }
                    let pair_id = match fields.get("pair_id") {
                        Some(DataValue::Text(value)) => value.clone(),
                        _ => return Err(CausalError::InvalidSerializedStudy),
                    };
                    let control_outcome = fields
                        .get("control_outcome")
                        .and_then(DataValue::as_f64)
                        .filter(|value| value.is_finite())
                        .ok_or(CausalError::InvalidSerializedStudy)?;
                    let treatment_outcome = fields
                        .get("treatment_outcome")
                        .and_then(DataValue::as_f64)
                        .filter(|value| value.is_finite())
                        .ok_or(CausalError::InvalidSerializedStudy)?;
                    Ok(PairedOutcome {
                        pair_id,
                        control_outcome,
                        treatment_outcome,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(CausalError::InvalidSerializedStudy),
        };
        let study = Self {
            experiment_id: text("experiment_id")?,
            dataset_id: text("dataset_id")?,
            treatment_label: text("treatment_label")?,
            control_label: text("control_label")?,
            outcome_label: text("outcome_label")?,
            assignment_design,
            pairs,
        };
        study
            .validate()
            .map_err(|_| CausalError::InvalidSerializedStudy)?;
        Ok(study)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CausalEffectEstimate {
    pub experiment_id: String,
    pub dataset_id: String,
    pub treatment_label: String,
    pub control_label: String,
    pub outcome_label: String,
    pub pair_count: usize,
    pub treatment_mean: f64,
    pub control_mean: f64,
    /// Positive values indicate a higher measured outcome under treatment.
    pub mean_effect: f64,
    pub standard_error: f64,
    pub normal_approximation_95_percent_interval: (f64, f64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CausalError {
    RandomizationRequired,
    InvalidStudy,
    InvalidPair,
    DuplicatePair,
    NumericalOverflow,
    InvalidSerializedStudy,
}

impl std::fmt::Display for CausalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RandomizationRequired => write!(
                formatter,
                "causal estimate requires caller-attested randomized assignment"
            ),
            Self::InvalidStudy => {
                write!(formatter, "paired randomized study is invalid or too small")
            }
            Self::InvalidPair => write!(
                formatter,
                "paired outcome must have a non-empty ID and finite outcomes"
            ),
            Self::DuplicatePair => write!(formatter, "paired outcome ID is duplicated"),
            Self::NumericalOverflow => write!(formatter, "causal estimate overflowed"),
            Self::InvalidSerializedStudy => write!(formatter, "serialized causal study is invalid"),
        }
    }
}

impl std::error::Error for CausalError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn study() -> PairedRandomizedStudy {
        PairedRandomizedStudy {
            experiment_id: "friction-randomized-v1".to_owned(),
            dataset_id: "paired-outcomes-v1".to_owned(),
            treatment_label: "high-friction".to_owned(),
            control_label: "low-friction".to_owned(),
            outcome_label: "deceleration".to_owned(),
            assignment_design: AssignmentDesign::Randomized,
            pairs: vec![
                PairedOutcome {
                    pair_id: "block-1".to_owned(),
                    control_outcome: 1.0,
                    treatment_outcome: 2.0,
                },
                PairedOutcome {
                    pair_id: "block-2".to_owned(),
                    control_outcome: 1.5,
                    treatment_outcome: 2.5,
                },
                PairedOutcome {
                    pair_id: "block-3".to_owned(),
                    control_outcome: 2.0,
                    treatment_outcome: 3.0,
                },
            ],
        }
    }

    #[test]
    fn estimates_paired_effect_and_uncertainty() {
        let estimate = study().estimate().expect("estimate");
        assert_eq!(estimate.pair_count, 3);
        assert!((estimate.mean_effect - 1.0).abs() < 1e-12);
        assert_eq!(estimate.standard_error, 0.0);
        assert_eq!(
            estimate.normal_approximation_95_percent_interval,
            (1.0, 1.0)
        );
    }

    #[test]
    fn rejects_observational_or_malformed_pairs() {
        let mut observational = study();
        observational.assignment_design = AssignmentDesign::Observational;
        assert_eq!(
            observational.estimate(),
            Err(CausalError::RandomizationRequired)
        );

        let mut duplicate = study();
        duplicate.pairs[1].pair_id = duplicate.pairs[0].pair_id.clone();
        assert_eq!(duplicate.estimate(), Err(CausalError::DuplicatePair));

        let mut too_small = study();
        too_small.pairs.truncate(1);
        assert_eq!(too_small.estimate(), Err(CausalError::InvalidStudy));
    }

    #[test]
    fn causal_study_artifact_round_trips_and_checks_design() {
        let original = study();
        let restored = PairedRandomizedStudy::from_artifact_value(&original.to_artifact_value())
            .expect("restore");
        assert_eq!(restored, original);

        let mut observational = original;
        observational.assignment_design = AssignmentDesign::Observational;
        assert_eq!(
            PairedRandomizedStudy::from_artifact_value(&observational.to_artifact_value()),
            Err(CausalError::InvalidSerializedStudy)
        );
    }
}
