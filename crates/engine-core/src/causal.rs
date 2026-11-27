//! Paired randomized-intervention effect estimation.
//!
//! This module computes a transparent paired difference-in-means estimator.
//! The caller must attest that assignment was randomized and that the pairs,
//! outcome measurement, and no-interference assumptions are valid. The engine
//! cannot verify those design assumptions from outcome values alone.

use crate::regression_metrics::{OnlineMean, ScaledSumSquares};
use crate::value::DataValue;
use std::collections::{BTreeMap, BTreeSet};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RandomizationInferencePolicy {
    pub maximum_assignments: u64,
    /// Attests that each pair independently randomized which unit received treatment
    /// with probability one half.
    pub attest_independent_fair_pair_swaps: bool,
}

impl Default for RandomizationInferencePolicy {
    fn default() -> Self {
        Self {
            maximum_assignments: 1_048_576,
            attest_independent_fair_pair_swaps: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairedRandomizationInference {
    pub experiment_id: String,
    pub dataset_id: String,
    pub pair_count: usize,
    pub observed_absolute_mean_effect: f64,
    pub assignment_count: u64,
    pub extreme_assignment_count: u64,
    pub exact_two_sided_p_value: f64,
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
    /// Exhaustively enumerates the paired sign-flip randomization distribution.
    ///
    /// This is exact under the sharp null only when treatment was independently
    /// assigned to either member of every pair with probability one half. That
    /// assignment mechanism must be explicitly attested in the policy. All 2^n
    /// assignments are enumerated, including the observed assignment. The hard
    /// cap is checked before enumeration begins.
    pub fn exact_randomization_inference(
        &self,
        policy: RandomizationInferencePolicy,
    ) -> Result<PairedRandomizationInference, CausalError> {
        self.validate()?;
        if !policy.attest_independent_fair_pair_swaps {
            return Err(CausalError::RandomizationMechanismNotAttested);
        }
        if policy.maximum_assignments == 0 {
            return Err(CausalError::InvalidInferencePolicy);
        }
        let shift =
            u32::try_from(self.pairs.len()).map_err(|_| CausalError::AssignmentCountOverflow)?;
        let assignment_count = 1_u64
            .checked_shl(shift)
            .ok_or(CausalError::AssignmentCountOverflow)?;
        if assignment_count > policy.maximum_assignments {
            return Err(CausalError::AssignmentBudgetExceeded {
                required: assignment_count,
                maximum: policy.maximum_assignments,
            });
        }

        let (scale, differences) = normalized_paired_differences(&self.pairs);
        let mut observed_mean = OnlineMean::default();
        for difference in &differences {
            if !observed_mean.push_finite(*difference) {
                return Err(CausalError::NumericalOverflow);
            }
        }
        let observed_mean = observed_mean
            .mean()
            .ok_or(CausalError::NumericalOverflow)?;
        let observed_statistic = scale * observed_mean.abs();
        if !observed_statistic.is_finite() {
            return Err(CausalError::NumericalOverflow);
        }
        let observed_statistic_normalized = observed_mean.abs();
        let tolerance = if scale == 0.0 {
            f64::INFINITY
        } else {
            8.0 * f64::EPSILON * observed_statistic_normalized.max(scale.recip())
        };
        let mut extreme_assignment_count = 0_u64;

        for assignment in 0..assignment_count {
            let mut signed_mean = OnlineMean::default();
            for (index, difference) in differences.iter().enumerate() {
                let signed_difference = if (assignment >> index) & 1 == 0 {
                    *difference
                } else {
                    -*difference
                };
                if !signed_mean.push_finite(signed_difference) {
                    return Err(CausalError::NumericalOverflow);
                }
            }
            let statistic = signed_mean
                .mean()
                .ok_or(CausalError::NumericalOverflow)?
                .abs();
            if statistic + tolerance >= observed_statistic_normalized {
                extreme_assignment_count = extreme_assignment_count
                    .checked_add(1)
                    .ok_or(CausalError::NumericalOverflow)?;
            }
        }

        Ok(PairedRandomizationInference {
            experiment_id: self.experiment_id.clone(),
            dataset_id: self.dataset_id.clone(),
            pair_count: self.pairs.len(),
            observed_absolute_mean_effect: observed_statistic,
            assignment_count,
            extreme_assignment_count,
            exact_two_sided_p_value: extreme_assignment_count as f64 / assignment_count as f64,
        })
    }

    pub fn estimate(&self) -> Result<CausalEffectEstimate, CausalError> {
        self.validate()?;
        let (scale, differences) = normalized_paired_differences(&self.pairs);
        let mut difference_mean = OnlineMean::default();
        let mut treatment_mean = OnlineMean::default();
        let mut control_mean = OnlineMean::default();
        for (pair, difference) in self.pairs.iter().zip(&differences) {
            if !difference_mean.push_finite(*difference)
                || !treatment_mean.push_finite(pair.treatment_outcome)
                || !control_mean.push_finite(pair.control_outcome)
            {
                return Err(CausalError::NumericalOverflow);
            }
        }
        let normalized_mean_effect = difference_mean
            .mean()
            .ok_or(CausalError::NumericalOverflow)?;
        let mean_effect = scale * normalized_mean_effect;
        let treatment_mean = treatment_mean.mean().ok_or(CausalError::NumericalOverflow)?;
        let control_mean = control_mean.mean().ok_or(CausalError::NumericalOverflow)?;
        let mut deviations = ScaledSumSquares::default();
        for difference in &differences {
            if !deviations.push_nonnegative((*difference - normalized_mean_effect).abs()) {
                return Err(CausalError::NumericalOverflow);
            }
        }
        let normalized_rms = deviations
            .root_mean_square()
            .ok_or(CausalError::NumericalOverflow)?;
        let standard_error = scale * (normalized_rms / ((self.pairs.len() - 1) as f64).sqrt());
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiArmRandomizationInferencePolicy {
    pub maximum_assignments: u64,
    /// Attests that all assignments respecting the fixed allocation were
    /// equally likely under the declared randomization protocol.
    pub attest_complete_random_assignment: bool,
}

impl Default for MultiArmRandomizationInferencePolicy {
    fn default() -> Self {
        Self {
            maximum_assignments: 1_000_000,
            attest_complete_random_assignment: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiArmRandomizationInference {
    pub experiment_id: String,
    pub dataset_id: String,
    pub control_arm_id: String,
    pub assignment_count: u64,
    pub extreme_assignment_count: u64,
    /// Exact omnibus p-value for the maximum absolute treatment-control mean contrast.
    pub exact_two_sided_p_value: f64,
    pub observed_maximum_absolute_effect: f64,
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
            let count = observed_counts
                .entry(observation.arm_id.clone())
                .or_default();
            *count = count.checked_add(1).ok_or(CausalError::NumericalOverflow)?;
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
        let (control_mean, control_scale, control_normalized_rms) =
            mean_and_scaled_sample_spread(control)?;
        let control_count = control.len();
        let mut effects = Vec::with_capacity(self.allocation_plan.len() - 1);

        for (arm_id, treatment_values) in &by_arm {
            if *arm_id == self.control_arm_id {
                continue;
            }
            let (treatment_mean, treatment_scale, treatment_normalized_rms) =
                mean_and_scaled_sample_spread(treatment_values)?;
            let mean_effect = treatment_mean - control_mean;
            let treatment_standard_error = treatment_scale
                * (treatment_normalized_rms / ((treatment_values.len() - 1) as f64).sqrt());
            let control_standard_error = control_scale
                * (control_normalized_rms / ((control_count - 1) as f64).sqrt());
            let standard_error = treatment_standard_error.hypot(control_standard_error);
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

    /// Enumerates every unique fixed-allocation arm assignment for an exact
    /// omnibus randomization test under the sharp null.
    ///
    /// The statistic is the maximum absolute difference between any treatment
    /// arm mean and the declared control mean. This incorporates the family of
    /// treatment-vs-control contrasts into one randomization distribution. The
    /// caller must attest that all assignments respecting the allocation plan
    /// were equally likely. The cap is checked before more than the configured
    /// number of allocations is evaluated.
    pub fn exact_randomization_inference(
        &self,
        policy: MultiArmRandomizationInferencePolicy,
    ) -> Result<MultiArmRandomizationInference, CausalError> {
        self.validate()?;
        if !policy.attest_complete_random_assignment {
            return Err(CausalError::RandomizationMechanismNotAttested);
        }
        if policy.maximum_assignments == 0 {
            return Err(CausalError::InvalidInferencePolicy);
        }

        let arm_ids: Vec<String> = self.allocation_plan.keys().cloned().collect();
        let control_index = arm_ids
            .iter()
            .position(|arm_id| arm_id == &self.control_arm_id)
            .ok_or(CausalError::InvalidMultiArmStudy)?;
        let arm_counts: Vec<usize> = arm_ids
            .iter()
            .map(|arm_id| self.allocation_plan[arm_id])
            .collect();
        let arm_indices: BTreeMap<&str, usize> = arm_ids
            .iter()
            .enumerate()
            .map(|(index, arm_id)| (arm_id.as_str(), index))
            .collect();
        let outcome_values: Vec<f64> = self.outcomes.iter().map(|item| item.outcome).collect();
        let observed_assignment: Vec<usize> = self
            .outcomes
            .iter()
            .map(|item| {
                arm_indices
                    .get(item.arm_id.as_str())
                    .copied()
                    .ok_or(CausalError::UnknownArm)
            })
            .collect::<Result<_, _>>()?;
        let observed_maximum_absolute_effect = multi_arm_maximum_absolute_effect(
            &outcome_values,
            &observed_assignment,
            &arm_counts,
            control_index,
        )?;
        let tolerance = 8.0 * f64::EPSILON * observed_maximum_absolute_effect.max(1.0);

        // Lexicographically sorted multiset assignment is the first unique
        // allocation. The next-permutation routine visits each allocation once.
        let mut assignment = Vec::with_capacity(self.outcomes.len());
        for (arm_index, count) in arm_counts.iter().enumerate() {
            assignment.extend(std::iter::repeat(arm_index).take(*count));
        }
        let mut assignment_count = 0_u64;
        let mut extreme_assignment_count = 0_u64;
        loop {
            if assignment_count == policy.maximum_assignments {
                if next_multiset_permutation(&mut assignment) {
                    return Err(CausalError::AssignmentBudgetExceededAtLeast {
                        required_at_least: assignment_count.saturating_add(1),
                        maximum: policy.maximum_assignments,
                    });
                }
                break;
            }
            assignment_count = assignment_count
                .checked_add(1)
                .ok_or(CausalError::NumericalOverflow)?;
            let statistic = multi_arm_maximum_absolute_effect(
                &outcome_values,
                &assignment,
                &arm_counts,
                control_index,
            )?;
            if statistic + tolerance >= observed_maximum_absolute_effect {
                extreme_assignment_count = extreme_assignment_count
                    .checked_add(1)
                    .ok_or(CausalError::NumericalOverflow)?;
            }
            if !next_multiset_permutation(&mut assignment) {
                break;
            }
        }
        if assignment_count == 0 {
            return Err(CausalError::InvalidMultiArmAllocation);
        }
        Ok(MultiArmRandomizationInference {
            experiment_id: self.experiment_id.clone(),
            dataset_id: self.dataset_id.clone(),
            control_arm_id: self.control_arm_id.clone(),
            assignment_count,
            extreme_assignment_count,
            exact_two_sided_p_value: extreme_assignment_count as f64 / assignment_count as f64,
            observed_maximum_absolute_effect,
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
                "control_arm_id".to_owned(),
                DataValue::Text(self.control_arm_id.clone()),
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
                "assignment_seed".to_owned(),
                DataValue::Text(self.assignment_seed.to_string()),
            ),
            (
                "randomization_protocol".to_owned(),
                DataValue::Text(self.randomization_protocol.clone()),
            ),
            (
                "allocation_plan".to_owned(),
                DataValue::Object(
                    self.allocation_plan
                        .iter()
                        .map(|(arm, count)| (arm.clone(), DataValue::Text(count.to_string())))
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
        let fields = value
            .as_object()
            .ok_or(CausalError::InvalidSerializedStudy)?;
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
                        DataValue::Text(value) => value
                            .parse()
                            .map_err(|_| CausalError::InvalidSerializedStudy)?,
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
                    let item = value
                        .as_object()
                        .ok_or(CausalError::InvalidSerializedStudy)?;
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
        study
            .validate()
            .map_err(|_| CausalError::InvalidSerializedStudy)?;
        Ok(study)
    }
}

fn multi_arm_maximum_absolute_effect(
    outcomes: &[f64],
    assignment: &[usize],
    arm_counts: &[usize],
    control_index: usize,
) -> Result<f64, CausalError> {
    if outcomes.len() != assignment.len()
        || arm_counts.is_empty()
        || control_index >= arm_counts.len()
    {
        return Err(CausalError::InvalidMultiArmStudy);
    }
    let mut means: Vec<OnlineMean> =
        std::iter::repeat_with(OnlineMean::default).take(arm_counts.len()).collect();
    for (outcome, arm_index) in outcomes.iter().zip(assignment.iter()) {
        let Some(mean) = means.get_mut(*arm_index) else {
            return Err(CausalError::UnknownArm);
        };
        if !mean.push_finite(*outcome) {
            return Err(CausalError::NumericalOverflow);
        }
    }
    let control_mean = means[control_index]
        .mean()
        .ok_or(CausalError::NumericalOverflow)?;
    let mut maximum = 0.0_f64;
    for arm_index in 0..arm_counts.len() {
        if arm_index == control_index {
            continue;
        }
        let treatment_mean = means[arm_index]
            .mean()
            .ok_or(CausalError::NumericalOverflow)?;
        let effect = (treatment_mean - control_mean).abs();
        if !effect.is_finite() {
            return Err(CausalError::NumericalOverflow);
        }
        maximum = maximum.max(effect);
    }
    Ok(maximum)
}

/// Advances a sorted multiset permutation without generating duplicates.
fn next_multiset_permutation(values: &mut [usize]) -> bool {
    if values.len() < 2 {
        return false;
    }
    let mut pivot = values.len() - 2;
    loop {
        if values[pivot] < values[pivot + 1] {
            break;
        }
        if pivot == 0 {
            return false;
        }
        pivot -= 1;
    }
    let mut successor = values.len() - 1;
    while values[successor] <= values[pivot] {
        successor -= 1;
    }
    values.swap(pivot, successor);
    values[pivot + 1..].reverse();
    true
}

fn normalized_paired_differences(pairs: &[PairedOutcome]) -> (f64, Vec<f64>) {
    let scale = pairs
        .iter()
        .flat_map(|pair| [pair.control_outcome.abs(), pair.treatment_outcome.abs()])
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return (0.0, vec![0.0; pairs.len()]);
    }

    let differences = pairs
        .iter()
        .map(|pair| {
            let difference = pair.treatment_outcome - pair.control_outcome;
            if difference.is_finite() {
                difference / scale
            } else {
                pair.treatment_outcome / scale - pair.control_outcome / scale
            }
        })
        .collect();
    (scale, differences)
}

fn mean_and_scaled_sample_spread(values: &[f64]) -> Result<(f64, f64, f64), CausalError> {
    if values.len() < 2 {
        return Err(CausalError::InvalidMultiArmAllocation);
    }
    let mut mean = OnlineMean::default();
    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    for value in values {
        if !mean.push_finite(*value) {
            return Err(CausalError::NumericalOverflow);
        }
    }
    let mean = mean.mean().ok_or(CausalError::NumericalOverflow)?;
    let normalized_mean = if scale == 0.0 { 0.0 } else { mean / scale };
    let mut deviations = ScaledSumSquares::default();
    for value in values {
        let normalized_deviation = if scale == 0.0 {
            0.0
        } else {
            let difference = *value - mean;
            if difference.is_finite() {
                difference / scale
            } else {
                *value / scale - normalized_mean
            }
        };
        if !deviations.push_nonnegative(normalized_deviation.abs()) {
            return Err(CausalError::NumericalOverflow);
        }
    }
    let normalized_rms = deviations
        .root_mean_square()
        .ok_or(CausalError::NumericalOverflow)?;
    Ok((mean, scale, normalized_rms))
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
    RandomizationMechanismNotAttested,
    InvalidInferencePolicy,
    AssignmentCountOverflow,
    AssignmentBudgetExceeded {
        required: u64,
        maximum: u64,
    },
    AssignmentBudgetExceededAtLeast {
        required_at_least: u64,
        maximum: u64,
    },
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
            Self::InvalidMultiArmStudy => {
                write!(formatter, "randomized multi-arm study is invalid")
            }
            Self::InvalidMultiArmAllocation => write!(
                formatter,
                "planned multi-arm allocation is invalid or underpowered"
            ),
            Self::InvalidArmOutcome => write!(
                formatter,
                "arm outcome requires non-empty unit and arm IDs and a finite value"
            ),
            Self::DuplicateUnit => write!(formatter, "experimental unit appears more than once"),
            Self::UnknownArm => write!(
                formatter,
                "outcome references an arm absent from the allocation plan"
            ),
            Self::AllocationMismatch => write!(
                formatter,
                "observed arm counts do not match the declared allocation plan"
            ),
            Self::RandomizationMechanismNotAttested => write!(
                formatter,
                "exact paired inference requires an explicit independent fair pair-swap attestation"
            ),
            Self::InvalidInferencePolicy => write!(
                formatter,
                "randomization inference assignment budget must be positive"
            ),
            Self::AssignmentCountOverflow => write!(
                formatter,
                "paired randomization assignment count exceeds the supported range"
            ),
            Self::AssignmentBudgetExceeded { required, maximum } => write!(
                formatter,
                "exact paired inference requires {required} assignments, exceeding the configured cap {maximum}"
            ),
            Self::AssignmentBudgetExceededAtLeast {
                required_at_least,
                maximum,
            } => write!(
                formatter,
                "exact multi-arm inference requires at least {required_at_least} assignments, exceeding the configured cap {maximum}"
            ),
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
    fn paired_estimator_handles_large_finite_differences_without_squared_overflow() {
        let magnitude = f64::MAX * 0.2;
        let mut study = study();
        study.pairs = vec![
            PairedOutcome {
                pair_id: "large-a".to_owned(),
                control_outcome: -magnitude,
                treatment_outcome: magnitude,
            },
            PairedOutcome {
                pair_id: "large-b".to_owned(),
                control_outcome: magnitude,
                treatment_outcome: -magnitude,
            },
        ];

        let estimate = study.estimate().expect("finite paired estimate");
        assert_eq!(estimate.mean_effect, 0.0);
        assert_eq!(estimate.treatment_mean, 0.0);
        assert_eq!(estimate.control_mean, 0.0);
        assert!((estimate.standard_error / (f64::MAX * 0.4) - 1.0).abs() < 1.0e-12);
        assert!(estimate.normal_approximation_95_percent_interval.0.is_finite());
        assert!(estimate.normal_approximation_95_percent_interval.1.is_finite());
    }

    #[test]
    fn paired_exact_inference_handles_overflowing_intermediate_signed_sums() {
        let mut study = study();
        study.pairs = vec![
            PairedOutcome {
                pair_id: "extreme-a".to_owned(),
                control_outcome: -f64::MAX * 0.5,
                treatment_outcome: f64::MAX * 0.5,
            },
            PairedOutcome {
                pair_id: "extreme-b".to_owned(),
                control_outcome: f64::MAX * 0.5,
                treatment_outcome: -f64::MAX * 0.5,
            },
        ];

        let inference = study
            .exact_randomization_inference(RandomizationInferencePolicy {
                maximum_assignments: 4,
                attest_independent_fair_pair_swaps: true,
            })
            .expect("finite exact randomization inference");
        assert_eq!(inference.observed_absolute_mean_effect, 0.0);
        assert_eq!(inference.extreme_assignment_count, 4);
        assert_eq!(inference.exact_two_sided_p_value, 1.0);
    }

    #[test]
    fn exact_paired_randomization_inference_enumerates_every_sign_assignment() {
        let study = PairedRandomizedStudy {
            experiment_id: "exact-paired-v1".to_owned(),
            dataset_id: "exact-paired-data-v1".to_owned(),
            treatment_label: "treatment".to_owned(),
            control_label: "control".to_owned(),
            outcome_label: "score".to_owned(),
            assignment_design: AssignmentDesign::Randomized,
            pairs: vec![
                PairedOutcome {
                    pair_id: "p1".to_owned(),
                    control_outcome: 0.0,
                    treatment_outcome: 1.0,
                },
                PairedOutcome {
                    pair_id: "p2".to_owned(),
                    control_outcome: 0.0,
                    treatment_outcome: 1.0,
                },
                PairedOutcome {
                    pair_id: "p3".to_owned(),
                    control_outcome: 0.0,
                    treatment_outcome: 1.0,
                },
            ],
        };
        let inference = study
            .exact_randomization_inference(RandomizationInferencePolicy {
                maximum_assignments: 8,
                attest_independent_fair_pair_swaps: true,
            })
            .expect("exact inference");
        assert_eq!(inference.assignment_count, 8);
        assert_eq!(inference.extreme_assignment_count, 2);
        assert_eq!(inference.exact_two_sided_p_value, 0.25);
        assert_eq!(inference.observed_absolute_mean_effect, 1.0);
    }

    #[test]
    fn exact_paired_inference_requires_attestation_and_enforces_work_cap() {
        let study = PairedRandomizedStudy {
            experiment_id: "exact-paired-v1".to_owned(),
            dataset_id: "exact-paired-data-v1".to_owned(),
            treatment_label: "treatment".to_owned(),
            control_label: "control".to_owned(),
            outcome_label: "score".to_owned(),
            assignment_design: AssignmentDesign::Randomized,
            pairs: vec![
                PairedOutcome {
                    pair_id: "p1".to_owned(),
                    control_outcome: 0.0,
                    treatment_outcome: 1.0,
                },
                PairedOutcome {
                    pair_id: "p2".to_owned(),
                    control_outcome: 0.0,
                    treatment_outcome: 1.0,
                },
                PairedOutcome {
                    pair_id: "p3".to_owned(),
                    control_outcome: 0.0,
                    treatment_outcome: 1.0,
                },
            ],
        };
        assert_eq!(
            study.exact_randomization_inference(RandomizationInferencePolicy::default()),
            Err(CausalError::RandomizationMechanismNotAttested)
        );
        assert_eq!(
            study.exact_randomization_inference(RandomizationInferencePolicy {
                maximum_assignments: 7,
                attest_independent_fair_pair_swaps: true,
            }),
            Err(CausalError::AssignmentBudgetExceeded {
                required: 8,
                maximum: 7
            })
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
        assert!(
            treatment_a.holm_adjusted_p_value >= treatment_a.two_sided_normal_approximation_p_value
        );
        assert!(
            treatment_b.holm_adjusted_p_value >= treatment_b.two_sided_normal_approximation_p_value
        );
        assert!(treatment_a.normal_approximation_95_percent_interval.0 < treatment_a.mean_effect);
        assert!(treatment_a.normal_approximation_95_percent_interval.1 > treatment_a.mean_effect);
    }

    #[test]
    fn multi_arm_estimator_handles_large_finite_outcomes_without_variance_overflow() {
        let magnitude = f64::MAX;
        let study = RandomizedMultiArmStudy {
            experiment_id: "large-multi-arm-v1".to_owned(),
            dataset_id: "large-multi-arm-data-v1".to_owned(),
            control_arm_id: "control".to_owned(),
            outcome_label: "score".to_owned(),
            assignment_design: AssignmentDesign::Randomized,
            assignment_seed: 91,
            randomization_protocol: "complete-randomization-v1".to_owned(),
            allocation_plan: BTreeMap::from([
                ("control".to_owned(), 2),
                ("treatment".to_owned(), 2),
            ]),
            outcomes: vec![
                RandomizedArmOutcome {
                    unit_id: "c1".to_owned(),
                    arm_id: "control".to_owned(),
                    outcome: -0.1 * magnitude,
                },
                RandomizedArmOutcome {
                    unit_id: "c2".to_owned(),
                    arm_id: "control".to_owned(),
                    outcome: 0.1 * magnitude,
                },
                RandomizedArmOutcome {
                    unit_id: "t1".to_owned(),
                    arm_id: "treatment".to_owned(),
                    outcome: 0.3 * magnitude,
                },
                RandomizedArmOutcome {
                    unit_id: "t2".to_owned(),
                    arm_id: "treatment".to_owned(),
                    outcome: 0.5 * magnitude,
                },
            ],
        };

        let report = study.estimate().expect("finite multi-arm estimate");
        let effect = report.effects.first().expect("treatment effect");
        assert!((effect.mean_effect / (0.4 * magnitude) - 1.0).abs() < 1.0e-12);
        assert!(effect.standard_error.is_finite());
        assert!((effect.standard_error / (0.2_f64.sqrt() * magnitude * 0.1)).is_finite());
        assert!(effect.normal_approximation_95_percent_interval.0.is_finite());
        assert!(effect.normal_approximation_95_percent_interval.1.is_finite());
    }

    #[test]
    fn exact_multi_arm_inference_handles_arm_sum_overflow() {
        let magnitude = f64::MAX;
        let mut outcomes = Vec::new();
        for (arm_id, values) in [
            ("control", [0.7, 0.8]),
            ("treatment-a", [0.6, 0.7]),
            ("treatment-b", [0.5, 0.6]),
        ] {
            for (index, value) in values.into_iter().enumerate() {
                outcomes.push(RandomizedArmOutcome {
                    unit_id: format!("{arm_id}-{index}"),
                    arm_id: arm_id.to_owned(),
                    outcome: value * magnitude,
                });
            }
        }
        let study = RandomizedMultiArmStudy {
            experiment_id: "large-exact-multi-arm-v1".to_owned(),
            dataset_id: "large-exact-multi-arm-data-v1".to_owned(),
            control_arm_id: "control".to_owned(),
            outcome_label: "score".to_owned(),
            assignment_design: AssignmentDesign::Randomized,
            assignment_seed: 19,
            randomization_protocol: "complete-randomization-v1".to_owned(),
            allocation_plan: BTreeMap::from([
                ("control".to_owned(), 2),
                ("treatment-a".to_owned(), 2),
                ("treatment-b".to_owned(), 2),
            ]),
            outcomes,
        };

        let inference = study
            .exact_randomization_inference(MultiArmRandomizationInferencePolicy {
                maximum_assignments: 90,
                attest_complete_random_assignment: true,
            })
            .expect("finite exact multi-arm inference");
        assert_eq!(inference.assignment_count, 90);
        assert!(inference.observed_maximum_absolute_effect.is_finite());
    }

    #[test]
    fn exact_multi_arm_randomization_inference_enumerates_unique_fixed_allocations() {
        let study = RandomizedMultiArmStudy {
            experiment_id: "exact-multi-arm-v1".to_owned(),
            dataset_id: "exact-multi-arm-data-v1".to_owned(),
            control_arm_id: "control".to_owned(),
            outcome_label: "score".to_owned(),
            assignment_design: AssignmentDesign::Randomized,
            assignment_seed: 7,
            randomization_protocol: "complete-randomization-v1".to_owned(),
            allocation_plan: BTreeMap::from([
                ("control".to_owned(), 2),
                ("treatment-a".to_owned(), 2),
                ("treatment-b".to_owned(), 2),
            ]),
            outcomes: vec![
                RandomizedArmOutcome {
                    unit_id: "u1".to_owned(),
                    arm_id: "control".to_owned(),
                    outcome: 1.0,
                },
                RandomizedArmOutcome {
                    unit_id: "u2".to_owned(),
                    arm_id: "control".to_owned(),
                    outcome: 2.0,
                },
                RandomizedArmOutcome {
                    unit_id: "u3".to_owned(),
                    arm_id: "treatment-a".to_owned(),
                    outcome: 2.0,
                },
                RandomizedArmOutcome {
                    unit_id: "u4".to_owned(),
                    arm_id: "treatment-a".to_owned(),
                    outcome: 3.0,
                },
                RandomizedArmOutcome {
                    unit_id: "u5".to_owned(),
                    arm_id: "treatment-b".to_owned(),
                    outcome: 0.0,
                },
                RandomizedArmOutcome {
                    unit_id: "u6".to_owned(),
                    arm_id: "treatment-b".to_owned(),
                    outcome: 1.0,
                },
            ],
        };
        let inference = study
            .exact_randomization_inference(MultiArmRandomizationInferencePolicy {
                maximum_assignments: 90,
                attest_complete_random_assignment: true,
            })
            .expect("exact multi-arm inference");
        assert_eq!(inference.assignment_count, 90);
        assert!(inference.extreme_assignment_count >= 1);
        assert!(inference.extreme_assignment_count <= inference.assignment_count);
        assert_eq!(
            inference.exact_two_sided_p_value,
            inference.extreme_assignment_count as f64 / 90.0
        );
        assert!(inference.observed_maximum_absolute_effect.is_finite());
    }

    #[test]
    fn exact_multi_arm_inference_requires_attestation_and_enforces_assignment_cap() {
        let study = multi_arm_study();
        assert_eq!(
            study.exact_randomization_inference(MultiArmRandomizationInferencePolicy {
                maximum_assignments: 90,
                attest_complete_random_assignment: false,
            }),
            Err(CausalError::RandomizationMechanismNotAttested)
        );
        assert_eq!(
            study.exact_randomization_inference(MultiArmRandomizationInferencePolicy {
                maximum_assignments: 89,
                attest_complete_random_assignment: true,
            }),
            Err(CausalError::AssignmentBudgetExceededAtLeast {
                required_at_least: 90,
                maximum: 89,
            })
        );
    }

    #[test]
    fn multi_arm_study_requires_randomization_unique_units_and_matching_allocation() {
        let mut observational = multi_arm_study();
        observational.assignment_design = AssignmentDesign::Observational;
        assert_eq!(
            observational.estimate(),
            Err(CausalError::RandomizationRequired)
        );

        let mut duplicate_unit = multi_arm_study();
        duplicate_unit.outcomes[1].unit_id = duplicate_unit.outcomes[0].unit_id.clone();
        assert_eq!(duplicate_unit.estimate(), Err(CausalError::DuplicateUnit));

        let mut allocation_mismatch = multi_arm_study();
        allocation_mismatch.outcomes.pop();
        assert_eq!(
            allocation_mismatch.estimate(),
            Err(CausalError::InvalidMultiArmAllocation)
        );

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
