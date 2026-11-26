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

#[derive(Clone, Debug, PartialEq)]
pub struct RandomizedArmOutcome {
    pub unit_id: String,
    pub arm_id: String,
    pub outcome: f64,
}

impl RandomizedArmOutcome {
    fn validate(&self) -> Result<(), CausalError> {
        if self.unit_id.trim().is_empty()
            || self.arm_id.trim().is_empty()
            || !self.outcome.is_finite()
        {
            return Err(CausalError::InvalidArmOutcome);
        }
        Ok(())
    }
}

/// Independent-unit randomized multi-arm study.
///
/// The assignment seed, protocol, and planned allocation are caller-attested
/// provenance. Internal consistency can be checked, but the engine cannot prove
/// that allocation was actually randomized or that no interference occurred.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomizedMultiArmStudy {
    pub experiment_id: String,
    pub dataset_id: String,
    pub control_arm_id: String,
    pub outcome_label: String,
    pub assignment_design: AssignmentDesign,
    pub assignment_seed: u64,
    pub randomization_protocol: String,
    /// Planned unit count for every arm, including the control.
    pub allocation_plan: BTreeMap<String, usize>,
    pub outcomes: Vec<RandomizedArmOutcome>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiArmEffectEstimate {
    pub treatment_arm_id: String,
    pub control_arm_id: String,
    pub treatment_count: usize,
    pub control_count: usize,
    pub treatment_mean: f64,
    pub control_mean: f64,
    /// Treatment mean minus control mean.
    pub mean_effect: f64,
    /// Independent-arm Welch-style standard error.
    pub standard_error: f64,
    /// Unadjusted normal-approximation interval.
    pub normal_approximation_95_percent_interval: (f64, f64),
    pub z_score: Option<f64>,
    pub two_sided_normal_approximation_p_value: f64,
    /// Holm-Bonferroni adjusted p-value across the treatment contrasts.
    pub holm_adjusted_p_value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomizedMultiArmEstimate {
    pub experiment_id: String,
    pub dataset_id: String,
    pub control_arm_id: String,
    pub outcome_label: String,
    pub assignment_seed: u64,
    pub randomization_protocol: String,
    pub unit_count: usize,
    pub effects: Vec<MultiArmEffectEstimate>,
}

impl RandomizedMultiArmStudy {
    pub fn validate(&self) -> Result<(), CausalError> {
        if self.assignment_design != AssignmentDesign::Randomized {
            return Err(CausalError::RandomizationRequired);
        }
        if self.experiment_id.trim().is_empty()
            || self.dataset_id.trim().is_empty()
            || self.control_arm_id.trim().is_empty()
            || self.outcome_label.trim().is_empty()
            || self.randomization_protocol.trim().is_empty()
            || self.allocation_plan.len() < 2
            || self.outcomes.is_empty()
            || !self.allocation_plan.contains_key(&self.control_arm_id)
        {
            return Err(CausalError::InvalidMultiArmStudy);
        }

        let mut planned_total = 0usize;
        for (arm_id, count) in &self.allocation_plan {
            if arm_id.trim().is_empty() || *count < 2 {
                return Err(CausalError::InvalidMultiArmAllocation);
            }
            planned_total = planned_total
                .checked_add(*count)
                .ok_or(CausalError::NumericalOverflow)?;
        }
        if planned_total != self.outcomes.len() {
            return Err(CausalError::InvalidMultiArmAllocation);
        }

        let mut unit_ids = BTreeSet::new();
        let mut observed_counts = BTreeMap::<String, usize>::new();
        for observation in &self.outcomes {
            observation.validate()?;
            if !unit_ids.insert(observation.unit_id.as_str()) {
                return Err(CausalError::DuplicateUnit);
            }
            if !self.allocation_plan.contains_key(&observation.arm_id) {
                return Err(CausalError::UnknownArm);
            }
            let count = observed_counts.entry(observation.arm_id.clone()).or_default();
            *count = count
                .checked_add(1)
                .ok_or(CausalError::NumericalOverflow)?;
        }
        for (arm_id, planned_count) in &self.allocation_plan {
            if observed_counts.get(arm_id).copied().unwrap_or(0) != *planned_count {
                return Err(CausalError::AllocationMismatch);
            }
        }
        Ok(())
    }

    /// Estimates each non-control arm against the declared control arm.
    ///
    /// Standard errors use independent-arm sample variances. Intervals and
    /// p-values use a standard-normal approximation. This is not exact
    /// randomization inference or a small-sample Student-t procedure.
    pub fn estimate(&self) -> Result<RandomizedMultiArmEstimate, CausalError> {
        self.validate()?;
        let mut by_arm = BTreeMap::<&str, Vec<f64>>::new();
        for observation in &self.outcomes {
            by_arm
                .entry(observation.arm_id.as_str())
                .or_default()
                .push(observation.outcome);
        }
        let control = by_arm
            .get(self.control_arm_id.as_str())
            .ok_or(CausalError::InvalidMultiArmStudy)?;
        let (control_mean, control_variance) = mean_and_sample_variance(control)?;
        let control_count = control.len();
        let mut effects = Vec::with_capacity(self.allocation_plan.len() - 1);

        for (arm_id, treatment_values) in &by_arm {
            if *arm_id == self.control_arm_id {
                continue;
            }
            let (treatment_mean, treatment_variance) =
                mean_and_sample_variance(treatment_values)?;
            let mean_effect = treatment_mean - control_mean;
            let standard_error = (treatment_variance / treatment_values.len() as f64
                + control_variance / control_count as f64)
                .sqrt();
            let margin = 1.96 * standard_error;
            let lower = mean_effect - margin;
            let upper = mean_effect + margin;
            if !mean_effect.is_finite()
                || !standard_error.is_finite()
                || !lower.is_finite()
                || !upper.is_finite()
            {
                return Err(CausalError::NumericalOverflow);
            }

            let (z_score, p_value) = if standard_error == 0.0 {
                if mean_effect == 0.0 {
                    (Some(0.0), 1.0)
                } else {
                    (None, 0.0)
                }
            } else {
                let z = mean_effect / standard_error;
                if !z.is_finite() {
                    return Err(CausalError::NumericalOverflow);
                }
                let p = (2.0 * (1.0 - standard_normal_cdf(z.abs()))).clamp(0.0, 1.0);
                (Some(z), p)
            };
            effects.push(MultiArmEffectEstimate {
                treatment_arm_id: (*arm_id).to_owned(),
                control_arm_id: self.control_arm_id.clone(),
                treatment_count: treatment_values.len(),
                control_count,
                treatment_mean,
                control_mean,
                mean_effect,
                standard_error,
                normal_approximation_95_percent_interval: (lower, upper),
                z_score,
                two_sided_normal_approximation_p_value: p_value,
                holm_adjusted_p_value: p_value,
            });
        }
        apply_holm_bonferroni(&mut effects);
        Ok(RandomizedMultiArmEstimate {
            experiment_id: self.experiment_id.clone(),
            dataset_id: self.dataset_id.clone(),
            control_arm_id: self.control_arm_id.clone(),
            outcome_label: self.outcome_label.clone(),
            assignment_seed: self.assignment_seed,
            randomization_protocol: self.randomization_protocol.clone(),
            unit_count: self.outcomes.len(),
            effects,
        })
    }

    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            ("experiment_id".to_owned(), DataValue::Text(self.experiment_id.clone())),
            ("dataset_id".to_owned(), DataValue::Text(self.dataset_id.clone())),
            ("control_arm_id".to_owned(), DataValue::Text(self.control_arm_id.clone())),
            ("outcome_label".to_owned(), DataValue::Text(self.outcome_label.clone())),
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
            ("assignment_seed".to_owned(), DataValue::Text(self.assignment_seed.to_string())),
            ("randomization_protocol".to_owned(), DataValue::Text(self.randomization_protocol.clone())),
            (
                "allocation_plan".to_owned(),
                DataValue::Object(
                    self.allocation_plan
                        .iter()
                        .map(|(arm, count)| {
                            (arm.clone(), DataValue::Text(count.to_string()))
                        })
                        .collect(),
                ),
            ),
            (
                "outcomes".to_owned(),
                DataValue::Sequence(
                    self.outcomes
                        .iter()
                        .map(|item| {
                            DataValue::object([
                                ("unit_id".to_owned(), DataValue::Text(item.unit_id.clone())),
                                ("arm_id".to_owned(), DataValue::Text(item.arm_id.clone())),
                                ("outcome".to_owned(), DataValue::Number(item.outcome)),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }

    pub fn from_artifact_value(value: &DataValue) -> Result<Self, CausalError> {
        let fields = value.as_object().ok_or(CausalError::InvalidSerializedStudy)?;
        if fields.len() != 10 || fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
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
        let assignment_seed = text("assignment_seed")?
            .parse()
            .map_err(|_| CausalError::InvalidSerializedStudy)?;
        let allocation_plan = match fields.get("allocation_plan") {
            Some(DataValue::Object(values)) => values
                .iter()
                .map(|(arm, value)| {
                    let count = match value {
                        DataValue::Text(value) => {
                            value.parse().map_err(|_| CausalError::InvalidSerializedStudy)?
                        }
                        _ => return Err(CausalError::InvalidSerializedStudy),
                    };
                    Ok((arm.clone(), count))
                })
                .collect::<Result<BTreeMap<_, _>, CausalError>>()?,
            _ => return Err(CausalError::InvalidSerializedStudy),
        };
        let outcomes = match fields.get("outcomes") {
            Some(DataValue::Sequence(values)) => values
                .iter()
                .map(|value| {
                    let item = value.as_object().ok_or(CausalError::InvalidSerializedStudy)?;
                    if item.len() != 3 {
                        return Err(CausalError::InvalidSerializedStudy);
                    }
                    let item_text = |name: &str| match item.get(name) {
                        Some(DataValue::Text(value)) => Ok(value.clone()),
                        _ => Err(CausalError::InvalidSerializedStudy),
                    };
                    let outcome = item
                        .get("outcome")
                        .and_then(DataValue::as_f64)
                        .filter(|value| value.is_finite())
                        .ok_or(CausalError::InvalidSerializedStudy)?;
                    Ok(RandomizedArmOutcome {
                        unit_id: item_text("unit_id")?,
                        arm_id: item_text("arm_id")?,
                        outcome,
                    })
                })
                .collect::<Result<Vec<_>, CausalError>>()?,
            _ => return Err(CausalError::InvalidSerializedStudy),
        };
        let study = Self {
            experiment_id: text("experiment_id")?,
            dataset_id: text("dataset_id")?,
            control_arm_id: text("control_arm_id")?,
            outcome_label: text("outcome_label")?,
            assignment_design,
            assignment_seed,
            randomization_protocol: text("randomization_protocol")?,
            allocation_plan,
            outcomes,
        };
        study.validate().map_err(|_| CausalError::InvalidSerializedStudy)?;
        Ok(study)
    }
}

fn mean_and_sample_variance(values: &[f64]) -> Result<(f64, f64), CausalError> {
    if values.len() < 2 {
        return Err(CausalError::InvalidMultiArmAllocation);
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let sum_squared = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    let variance = sum_squared / (count - 1.0);
    if !mean.is_finite() || !variance.is_finite() {
        return Err(CausalError::NumericalOverflow);
    }
    Ok((mean, variance))
}

// Abramowitz-Stegun-style normal CDF approximation.
fn standard_normal_cdf(value: f64) -> f64 {
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * x);
    let density = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let tail = density
        * t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let positive_cdf = (1.0 - tail).clamp(0.0, 1.0);
    if value >= 0.0 {
        positive_cdf
    } else {
        1.0 - positive_cdf
    }
}

fn apply_holm_bonferroni(effects: &mut [MultiArmEffectEstimate]) {
    let mut order: Vec<usize> = (0..effects.len()).collect();
    order.sort_by(|left, right| {
        effects[*left]
            .two_sided_normal_approximation_p_value
            .total_cmp(&effects[*right].two_sided_normal_approximation_p_value)
            .then_with(|| {
                effects[*left]
                    .treatment_arm_id
                    .cmp(&effects[*right].treatment_arm_id)
            })
    });
    let comparisons = effects.len();
    let mut adjusted_so_far: f64 = 0.0;
    for (rank, index) in order.into_iter().enumerate() {
        let candidate = ((comparisons - rank) as f64
            * effects[index].two_sided_normal_approximation_p_value)
            .min(1.0);
        adjusted_so_far = adjusted_so_far.max(candidate);
        effects[index].holm_adjusted_p_value = adjusted_so_far;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CausalError {
    RandomizationRequired,
    InvalidStudy,
    InvalidPair,
    DuplicatePair,
    NumericalOverflow,
    InvalidSerializedStudy,
    InvalidMultiArmStudy,
    InvalidMultiArmAllocation,
    InvalidArmOutcome,
    DuplicateUnit,
    UnknownArm,
    AllocationMismatch,
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
            Self::InvalidMultiArmStudy => write!(formatter, "randomized multi-arm study is invalid"),
            Self::InvalidMultiArmAllocation => write!(formatter, "planned multi-arm allocation is invalid or underpowered"),
            Self::InvalidArmOutcome => write!(formatter, "arm outcome requires non-empty unit and arm IDs and a finite value"),
            Self::DuplicateUnit => write!(formatter, "experimental unit appears more than once"),
            Self::UnknownArm => write!(formatter, "outcome references an arm absent from the allocation plan"),
            Self::AllocationMismatch => write!(formatter, "observed arm counts do not match the declared allocation plan"),
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

    fn multi_arm_study() -> RandomizedMultiArmStudy {
        let mut outcomes = Vec::new();
        for (arm_id, values) in [
            ("control", [1.0, 1.5, 2.0, 2.5]),
            ("treatment-a", [2.0, 2.5, 3.0, 3.5]),
            ("treatment-b", [0.0, 0.5, 1.0, 1.5]),
        ] {
            for (index, outcome) in values.into_iter().enumerate() {
                outcomes.push(RandomizedArmOutcome {
                    unit_id: format!("{arm_id}-unit-{index}"),
                    arm_id: arm_id.to_owned(),
                    outcome,
                });
            }
        }
        RandomizedMultiArmStudy {
            experiment_id: "three-arm-v1".to_owned(),
            dataset_id: "three-arm-outcomes-v1".to_owned(),
            control_arm_id: "control".to_owned(),
            outcome_label: "task-score".to_owned(),
            assignment_design: AssignmentDesign::Randomized,
            assignment_seed: 31415,
            randomization_protocol: "block-randomization-v2".to_owned(),
            allocation_plan: BTreeMap::from([
                ("control".to_owned(), 4),
                ("treatment-a".to_owned(), 4),
                ("treatment-b".to_owned(), 4),
            ]),
            outcomes,
        }
    }

    #[test]
    fn multi_arm_estimator_reports_control_contrasts_and_holm_adjustment() {
        let report = multi_arm_study().estimate().expect("estimate");
        assert_eq!(report.unit_count, 12);
        assert_eq!(report.effects.len(), 2);
        let treatment_a = report
            .effects
            .iter()
            .find(|item| item.treatment_arm_id == "treatment-a")
            .expect("treatment a");
        let treatment_b = report
            .effects
            .iter()
            .find(|item| item.treatment_arm_id == "treatment-b")
            .expect("treatment b");
        assert!((treatment_a.mean_effect - 1.0).abs() < 1e-12);
        assert!((treatment_b.mean_effect + 1.0).abs() < 1e-12);
        assert_eq!(treatment_a.control_count, 4);
        assert_eq!(treatment_a.treatment_count, 4);
        assert!(treatment_a.standard_error > 0.0);
        assert!(treatment_a.holm_adjusted_p_value >= treatment_a.two_sided_normal_approximation_p_value);
        assert!(treatment_b.holm_adjusted_p_value >= treatment_b.two_sided_normal_approximation_p_value);
        assert!(treatment_a.normal_approximation_95_percent_interval.0 < treatment_a.mean_effect);
        assert!(treatment_a.normal_approximation_95_percent_interval.1 > treatment_a.mean_effect);
    }

    #[test]
    fn multi_arm_study_requires_randomization_unique_units_and_matching_allocation() {
        let mut observational = multi_arm_study();
        observational.assignment_design = AssignmentDesign::Observational;
        assert_eq!(observational.estimate(), Err(CausalError::RandomizationRequired));

        let mut duplicate_unit = multi_arm_study();
        duplicate_unit.outcomes[1].unit_id = duplicate_unit.outcomes[0].unit_id.clone();
        assert_eq!(duplicate_unit.estimate(), Err(CausalError::DuplicateUnit));

        let mut allocation_mismatch = multi_arm_study();
        allocation_mismatch.outcomes.pop();
        assert_eq!(allocation_mismatch.estimate(), Err(CausalError::InvalidMultiArmAllocation));

        let mut unknown_arm = multi_arm_study();
        unknown_arm.outcomes[0].arm_id = "unplanned".to_owned();
        assert_eq!(unknown_arm.estimate(), Err(CausalError::UnknownArm));
    }

    #[test]
    fn multi_arm_study_artifact_round_trips_and_revalidates_allocation() {
        let original = multi_arm_study();
        let restored = RandomizedMultiArmStudy::from_artifact_value(&original.to_artifact_value())
            .expect("restore");
        assert_eq!(restored, original);

        let mut malformed = original.to_artifact_value();
        if let DataValue::Object(fields) = &mut malformed {
            fields.insert(
                "allocation_plan".to_owned(),
                DataValue::Object(BTreeMap::from([(
                    "control".to_owned(),
                    DataValue::Text("999".to_owned()),
                )])),
            );
        }
        assert_eq!(
            RandomizedMultiArmStudy::from_artifact_value(&malformed),
            Err(CausalError::InvalidSerializedStudy)
        );
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
