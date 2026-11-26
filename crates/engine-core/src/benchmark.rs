//! Cross-environment knowledge-transfer benchmark records.
//!
//! This module measures supplied baseline and transferred-model evaluations.
//! It does not train, execute, or infer transfer from schema compatibility.

use crate::value::DataValue;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreDirection {
    HigherIsBetter,
    LowerIsBetter,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransferEvaluation {
    pub case_id: String,
    pub source_environment_id: String,
    pub target_environment_id: String,
    pub source_training_dataset_id: String,
    pub target_evaluation_dataset_id: String,
    pub model_artifact_ref: String,
    pub metric_id: String,
    pub direction: ScoreDirection,
    pub baseline_score: f64,
    pub transferred_score: f64,
    pub sample_count: usize,
    pub resource_cost: f64,
}

impl TransferEvaluation {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.case_id.trim().is_empty()
            || self.source_environment_id.trim().is_empty()
            || self.target_environment_id.trim().is_empty()
            || self.source_training_dataset_id.trim().is_empty()
            || self.target_evaluation_dataset_id.trim().is_empty()
            || self.model_artifact_ref.trim().is_empty()
            || self.metric_id.trim().is_empty()
            || self.sample_count == 0
            || !self.baseline_score.is_finite()
            || !self.transferred_score.is_finite()
            || !self.resource_cost.is_finite()
            || self.resource_cost < 0.0
        {
            return Err(BenchmarkError::InvalidEvaluation);
        }
        if self.source_environment_id == self.target_environment_id {
            return Err(BenchmarkError::NotCrossEnvironment);
        }
        if self.source_training_dataset_id == self.target_evaluation_dataset_id {
            return Err(BenchmarkError::DatasetLeakage);
        }
        Ok(())
    }

    pub fn improvement(&self) -> f64 {
        match self.direction {
            ScoreDirection::HigherIsBetter => self.transferred_score - self.baseline_score,
            ScoreDirection::LowerIsBetter => self.baseline_score - self.transferred_score,
        }
    }

    pub fn is_positive_transfer(&self) -> bool {
        self.improvement() > 0.0
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TransferBenchmark {
    evaluations: BTreeMap<String, TransferEvaluation>,
}

impl TransferBenchmark {
    pub fn record(&mut self, evaluation: TransferEvaluation) -> Result<(), BenchmarkError> {
        evaluation.validate()?;
        if self.evaluations.contains_key(&evaluation.case_id) {
            return Err(BenchmarkError::DuplicateCase);
        }
        self.evaluations
            .insert(evaluation.case_id.clone(), evaluation);
        Ok(())
    }

    pub fn get(&self, case_id: &str) -> Option<&TransferEvaluation> {
        self.evaluations.get(case_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &TransferEvaluation> {
        self.evaluations.values()
    }

    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.evaluations.is_empty()
    }

    /// Summarizes cases for one metric without mixing metric definitions or directions.
    pub fn summarize(&self, metric_id: &str) -> Result<TransferSummary, BenchmarkError> {
        if metric_id.trim().is_empty() {
            return Err(BenchmarkError::InvalidMetric);
        }
        let selected: Vec<&TransferEvaluation> = self
            .evaluations
            .values()
            .filter(|evaluation| evaluation.metric_id == metric_id)
            .collect();
        if selected.is_empty() {
            return Err(BenchmarkError::NoMatchingCases);
        }
        let direction = selected[0].direction;
        if selected
            .iter()
            .any(|evaluation| evaluation.direction != direction)
        {
            return Err(BenchmarkError::InconsistentMetricDirection);
        }
        let improvements: Vec<f64> = selected
            .iter()
            .map(|evaluation| evaluation.improvement())
            .collect();
        let count = improvements.len() as f64;
        let mean_improvement = improvements.iter().sum::<f64>() / count;
        let wins = improvements
            .iter()
            .filter(|improvement| **improvement > 0.0)
            .count();
        let sample_variance = if improvements.len() > 1 {
            improvements
                .iter()
                .map(|improvement| (improvement - mean_improvement).powi(2))
                .sum::<f64>()
                / (count - 1.0)
        } else {
            0.0
        };
        let standard_error = if improvements.len() > 1 {
            Some((sample_variance / count).sqrt())
        } else {
            None
        };
        let mean_resource_cost = selected
            .iter()
            .map(|evaluation| evaluation.resource_cost)
            .sum::<f64>()
            / count;
        let total_samples = selected.iter().try_fold(0usize, |sum, evaluation| {
            sum.checked_add(evaluation.sample_count)
                .ok_or(BenchmarkError::NumericalOverflow)
        })?;
        if !mean_improvement.is_finite()
            || !mean_resource_cost.is_finite()
            || standard_error.is_some_and(|value| !value.is_finite())
        {
            return Err(BenchmarkError::NumericalOverflow);
        }
        Ok(TransferSummary {
            metric_id: metric_id.to_owned(),
            direction,
            case_count: improvements.len(),
            total_evaluation_samples: total_samples,
            mean_improvement,
            standard_error,
            positive_transfer_rate: wins as f64 / count,
            mean_resource_cost,
        })
    }

    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "evaluations".to_owned(),
                DataValue::Sequence(self.evaluations.values().map(evaluation_to_value).collect()),
            ),
        ])
    }

    pub fn from_artifact_value(value: &DataValue) -> Result<Self, BenchmarkError> {
        let fields = value
            .as_object()
            .ok_or(BenchmarkError::InvalidSerializedState)?;
        if fields.len() != 2 || fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
            return Err(BenchmarkError::InvalidSerializedState);
        }
        let values = match fields.get("evaluations") {
            Some(DataValue::Sequence(values)) => values,
            _ => return Err(BenchmarkError::InvalidSerializedState),
        };
        let mut benchmark = Self::default();
        for value in values {
            let evaluation = evaluation_from_value(value)?;
            if benchmark
                .evaluations
                .insert(evaluation.case_id.clone(), evaluation)
                .is_some()
            {
                return Err(BenchmarkError::InvalidSerializedState);
            }
        }
        Ok(benchmark)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransferSummary {
    pub metric_id: String,
    pub direction: ScoreDirection,
    pub case_count: usize,
    pub total_evaluation_samples: usize,
    pub mean_improvement: f64,
    /// Unavailable for a single case; otherwise standard error across case-level improvements.
    pub standard_error: Option<f64>,
    pub positive_transfer_rate: f64,
    pub mean_resource_cost: f64,
}

fn direction_name(direction: ScoreDirection) -> &'static str {
    match direction {
        ScoreDirection::HigherIsBetter => "higher",
        ScoreDirection::LowerIsBetter => "lower",
    }
}

fn parse_direction(value: &str) -> Result<ScoreDirection, BenchmarkError> {
    match value {
        "higher" => Ok(ScoreDirection::HigherIsBetter),
        "lower" => Ok(ScoreDirection::LowerIsBetter),
        _ => Err(BenchmarkError::InvalidSerializedState),
    }
}

fn evaluation_to_value(value: &TransferEvaluation) -> DataValue {
    DataValue::object([
        ("case_id".to_owned(), DataValue::Text(value.case_id.clone())),
        (
            "source_environment_id".to_owned(),
            DataValue::Text(value.source_environment_id.clone()),
        ),
        (
            "target_environment_id".to_owned(),
            DataValue::Text(value.target_environment_id.clone()),
        ),
        (
            "source_training_dataset_id".to_owned(),
            DataValue::Text(value.source_training_dataset_id.clone()),
        ),
        (
            "target_evaluation_dataset_id".to_owned(),
            DataValue::Text(value.target_evaluation_dataset_id.clone()),
        ),
        (
            "model_artifact_ref".to_owned(),
            DataValue::Text(value.model_artifact_ref.clone()),
        ),
        (
            "metric_id".to_owned(),
            DataValue::Text(value.metric_id.clone()),
        ),
        (
            "direction".to_owned(),
            DataValue::Text(direction_name(value.direction).to_owned()),
        ),
        (
            "baseline_score".to_owned(),
            DataValue::Number(value.baseline_score),
        ),
        (
            "transferred_score".to_owned(),
            DataValue::Number(value.transferred_score),
        ),
        (
            "sample_count".to_owned(),
            DataValue::Text(value.sample_count.to_string()),
        ),
        (
            "resource_cost".to_owned(),
            DataValue::Number(value.resource_cost),
        ),
    ])
}

fn evaluation_from_value(value: &DataValue) -> Result<TransferEvaluation, BenchmarkError> {
    let fields = value
        .as_object()
        .ok_or(BenchmarkError::InvalidSerializedState)?;
    if fields.len() != 12 {
        return Err(BenchmarkError::InvalidSerializedState);
    }
    let text = |name: &str| match fields.get(name) {
        Some(DataValue::Text(value)) => Ok(value.clone()),
        _ => Err(BenchmarkError::InvalidSerializedState),
    };
    let number = |name: &str| {
        fields
            .get(name)
            .and_then(DataValue::as_f64)
            .filter(|value| value.is_finite())
            .ok_or(BenchmarkError::InvalidSerializedState)
    };
    let sample_count = text("sample_count")?
        .parse()
        .map_err(|_| BenchmarkError::InvalidSerializedState)?;
    let evaluation = TransferEvaluation {
        case_id: text("case_id")?,
        source_environment_id: text("source_environment_id")?,
        target_environment_id: text("target_environment_id")?,
        source_training_dataset_id: text("source_training_dataset_id")?,
        target_evaluation_dataset_id: text("target_evaluation_dataset_id")?,
        model_artifact_ref: text("model_artifact_ref")?,
        metric_id: text("metric_id")?,
        direction: parse_direction(&text("direction")?)?,
        baseline_score: number("baseline_score")?,
        transferred_score: number("transferred_score")?,
        sample_count,
        resource_cost: number("resource_cost")?,
    };
    evaluation
        .validate()
        .map_err(|_| BenchmarkError::InvalidSerializedState)?;
    Ok(evaluation)
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransferAdaptationPoint {
    pub adaptation_samples: usize,
    pub baseline_score: f64,
    pub transferred_score: f64,
    pub resource_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FewShotTransferCurve {
    pub source_environment_id: String,
    pub target_environment_id: String,
    pub source_training_dataset_id: String,
    pub target_evaluation_dataset_id: String,
    pub model_artifact_ref: String,
    pub metric_id: String,
    pub direction: ScoreDirection,
    pub evaluation_sample_count: usize,
    pub points: Vec<TransferAdaptationPoint>,
}

impl FewShotTransferCurve {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.source_environment_id.trim().is_empty()
            || self.target_environment_id.trim().is_empty()
            || self.source_training_dataset_id.trim().is_empty()
            || self.target_evaluation_dataset_id.trim().is_empty()
            || self.model_artifact_ref.trim().is_empty()
            || self.metric_id.trim().is_empty()
            || self.evaluation_sample_count == 0
            || self.points.len() < 2
            || self.source_environment_id == self.target_environment_id
            || self.source_training_dataset_id == self.target_evaluation_dataset_id
        {
            return Err(BenchmarkError::InvalidAdaptationCurve);
        }
        let mut previous_samples = None;
        for point in &self.points {
            if point.adaptation_samples == 0
                || previous_samples.is_some_and(|previous| point.adaptation_samples <= previous)
                || !point.baseline_score.is_finite()
                || !point.transferred_score.is_finite()
                || !point.resource_cost.is_finite()
                || point.resource_cost < 0.0
            {
                return Err(BenchmarkError::InvalidAdaptationCurve);
            }
            previous_samples = Some(point.adaptation_samples);
        }
        Ok(())
    }

    pub fn improvements(&self) -> Result<Vec<f64>, BenchmarkError> {
        self.validate()?;
        let values: Vec<f64> = self
            .points
            .iter()
            .map(|point| match self.direction {
                ScoreDirection::HigherIsBetter => point.transferred_score - point.baseline_score,
                ScoreDirection::LowerIsBetter => point.baseline_score - point.transferred_score,
            })
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(BenchmarkError::NumericalOverflow);
        }
        Ok(values)
    }

    /// Trapezoidal area under the improvement-vs-adaptation-sample curve.
    pub fn area_under_improvement_curve(&self) -> Result<f64, BenchmarkError> {
        let improvements = self.improvements()?;
        let mut area = 0.0;
        for (index, pair) in self.points.windows(2).enumerate() {
            let width = (pair[1].adaptation_samples - pair[0].adaptation_samples) as f64;
            area += width * (improvements[index] + improvements[index + 1]) / 2.0;
            if !area.is_finite() {
                return Err(BenchmarkError::NumericalOverflow);
            }
        }
        Ok(area)
    }

    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "source_environment_id".to_owned(),
                DataValue::Text(self.source_environment_id.clone()),
            ),
            (
                "target_environment_id".to_owned(),
                DataValue::Text(self.target_environment_id.clone()),
            ),
            (
                "source_training_dataset_id".to_owned(),
                DataValue::Text(self.source_training_dataset_id.clone()),
            ),
            (
                "target_evaluation_dataset_id".to_owned(),
                DataValue::Text(self.target_evaluation_dataset_id.clone()),
            ),
            (
                "model_artifact_ref".to_owned(),
                DataValue::Text(self.model_artifact_ref.clone()),
            ),
            (
                "metric_id".to_owned(),
                DataValue::Text(self.metric_id.clone()),
            ),
            (
                "direction".to_owned(),
                DataValue::Text(direction_name(self.direction).to_owned()),
            ),
            (
                "evaluation_sample_count".to_owned(),
                DataValue::Text(self.evaluation_sample_count.to_string()),
            ),
            (
                "points".to_owned(),
                DataValue::Sequence(
                    self.points
                        .iter()
                        .map(|point| {
                            DataValue::object([
                                (
                                    "adaptation_samples".to_owned(),
                                    DataValue::Text(point.adaptation_samples.to_string()),
                                ),
                                (
                                    "baseline_score".to_owned(),
                                    DataValue::Number(point.baseline_score),
                                ),
                                (
                                    "transferred_score".to_owned(),
                                    DataValue::Number(point.transferred_score),
                                ),
                                (
                                    "resource_cost".to_owned(),
                                    DataValue::Number(point.resource_cost),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }

    pub fn from_artifact_value(value: &DataValue) -> Result<Self, BenchmarkError> {
        let fields = value
            .as_object()
            .ok_or(BenchmarkError::InvalidAdaptationCurve)?;
        if fields.len() != 10 || fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
            return Err(BenchmarkError::InvalidAdaptationCurve);
        }
        let text = |name: &str| match fields.get(name) {
            Some(DataValue::Text(value)) => Ok(value.clone()),
            _ => Err(BenchmarkError::InvalidAdaptationCurve),
        };
        let direction = match text("direction")?.as_str() {
            "higher" => ScoreDirection::HigherIsBetter,
            "lower" => ScoreDirection::LowerIsBetter,
            _ => return Err(BenchmarkError::InvalidAdaptationCurve),
        };
        let evaluation_sample_count = text("evaluation_sample_count")?
            .parse()
            .map_err(|_| BenchmarkError::InvalidAdaptationCurve)?;
        let points = match fields.get("points") {
            Some(DataValue::Sequence(points)) => points
                .iter()
                .map(|value| {
                    let point = value
                        .as_object()
                        .ok_or(BenchmarkError::InvalidAdaptationCurve)?;
                    if point.len() != 4 {
                        return Err(BenchmarkError::InvalidAdaptationCurve);
                    }
                    let adaptation_samples = match point.get("adaptation_samples") {
                        Some(DataValue::Text(value)) => value
                            .parse()
                            .map_err(|_| BenchmarkError::InvalidAdaptationCurve)?,
                        _ => return Err(BenchmarkError::InvalidAdaptationCurve),
                    };
                    let number = |name: &str| {
                        point
                            .get(name)
                            .and_then(DataValue::as_f64)
                            .filter(|value| value.is_finite())
                            .ok_or(BenchmarkError::InvalidAdaptationCurve)
                    };
                    Ok(TransferAdaptationPoint {
                        adaptation_samples,
                        baseline_score: number("baseline_score")?,
                        transferred_score: number("transferred_score")?,
                        resource_cost: number("resource_cost")?,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(BenchmarkError::InvalidAdaptationCurve),
        };
        let curve = Self {
            source_environment_id: text("source_environment_id")?,
            target_environment_id: text("target_environment_id")?,
            source_training_dataset_id: text("source_training_dataset_id")?,
            target_evaluation_dataset_id: text("target_evaluation_dataset_id")?,
            model_artifact_ref: text("model_artifact_ref")?,
            metric_id: text("metric_id")?,
            direction,
            evaluation_sample_count,
            points,
        };
        curve
            .validate()
            .map_err(|_| BenchmarkError::InvalidAdaptationCurve)?;
        Ok(curve)
    }
}

/// Aggregated results for one directed source-to-target environment pair.
#[derive(Clone, Debug, PartialEq)]
pub struct TransferMatrixCell {
    pub source_environment_id: String,
    pub target_environment_id: String,
    pub case_count: usize,
    pub evaluation_sample_count: usize,
    pub mean_improvement: f64,
    pub positive_transfer_rate: f64,
    pub negative_transfer_rate: f64,
    pub mean_resource_cost: f64,
    pub standard_error: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransferMatrix {
    pub metric_id: String,
    pub direction: ScoreDirection,
    /// Sorted union of all source and target environment IDs.
    pub environment_ids: Vec<String>,
    /// Sparse cells: a missing directed pair means no case was measured.
    pub cells: Vec<TransferMatrixCell>,
}

impl TransferMatrix {
    /// Builds a directed, sparse matrix from cases for one metric.
    pub fn from_benchmark(
        benchmark: &TransferBenchmark,
        metric_id: &str,
    ) -> Result<Self, BenchmarkError> {
        if metric_id.trim().is_empty() {
            return Err(BenchmarkError::InvalidMetric);
        }
        let selected: Vec<&TransferEvaluation> = benchmark
            .iter()
            .filter(|evaluation| evaluation.metric_id == metric_id)
            .collect();
        if selected.is_empty() {
            return Err(BenchmarkError::NoMatchingCases);
        }
        let direction = selected[0].direction;
        if selected
            .iter()
            .any(|evaluation| evaluation.direction != direction)
        {
            return Err(BenchmarkError::InconsistentMetricDirection);
        }

        let mut environment_ids = std::collections::BTreeSet::new();
        let mut pairs: BTreeMap<(String, String), Vec<&TransferEvaluation>> = BTreeMap::new();
        for evaluation in selected {
            evaluation.validate()?;
            environment_ids.insert(evaluation.source_environment_id.clone());
            environment_ids.insert(evaluation.target_environment_id.clone());
            pairs
                .entry((
                    evaluation.source_environment_id.clone(),
                    evaluation.target_environment_id.clone(),
                ))
                .or_default()
                .push(evaluation);
        }

        let mut cells = Vec::with_capacity(pairs.len());
        for ((source_environment_id, target_environment_id), evaluations) in pairs {
            let mut improvements = Vec::with_capacity(evaluations.len());
            let mut total_evaluation_samples = 0usize;
            let mut total_resource_cost = 0.0;
            for evaluation in &evaluations {
                let improvement = evaluation.improvement();
                if !improvement.is_finite() {
                    return Err(BenchmarkError::NumericalOverflow);
                }
                improvements.push(improvement);
                total_evaluation_samples = total_evaluation_samples
                    .checked_add(evaluation.sample_count)
                    .ok_or(BenchmarkError::NumericalOverflow)?;
                total_resource_cost += evaluation.resource_cost;
            }
            let count = improvements.len() as f64;
            let mean_improvement = improvements.iter().sum::<f64>() / count;
            let positive_count = improvements.iter().filter(|value| **value > 0.0).count();
            let negative_count = improvements.iter().filter(|value| **value < 0.0).count();
            let variance = if improvements.len() > 1 {
                improvements
                    .iter()
                    .map(|value| (value - mean_improvement).powi(2))
                    .sum::<f64>()
                    / (count - 1.0)
            } else {
                0.0
            };
            let standard_error = if improvements.len() > 1 {
                Some((variance / count).sqrt())
            } else {
                None
            };
            let cell = TransferMatrixCell {
                source_environment_id,
                target_environment_id,
                case_count: evaluations.len(),
                evaluation_sample_count: total_evaluation_samples,
                mean_improvement,
                positive_transfer_rate: positive_count as f64 / count,
                negative_transfer_rate: negative_count as f64 / count,
                mean_resource_cost: total_resource_cost / count,
                standard_error,
            };
            if !cell.mean_improvement.is_finite()
                || !cell.positive_transfer_rate.is_finite()
                || !cell.negative_transfer_rate.is_finite()
                || !cell.mean_resource_cost.is_finite()
                || cell.standard_error.is_some_and(|value| !value.is_finite())
            {
                return Err(BenchmarkError::NumericalOverflow);
            }
            cells.push(cell);
        }

        let matrix = Self {
            metric_id: metric_id.to_owned(),
            direction,
            environment_ids: environment_ids.into_iter().collect(),
            cells,
        };
        matrix.validate()?;
        Ok(matrix)
    }

    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.metric_id.trim().is_empty() || self.environment_ids.len() < 2 {
            return Err(BenchmarkError::InvalidTransferMatrix);
        }
        if self.environment_ids.iter().any(|id| id.trim().is_empty())
            || self
                .environment_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self.cells.is_empty()
        {
            return Err(BenchmarkError::InvalidTransferMatrix);
        }
        let known: std::collections::BTreeSet<&str> =
            self.environment_ids.iter().map(String::as_str).collect();
        let mut previous_pair: Option<(&str, &str)> = None;
        for cell in &self.cells {
            let pair = (
                cell.source_environment_id.as_str(),
                cell.target_environment_id.as_str(),
            );
            if cell.source_environment_id.trim().is_empty()
                || cell.target_environment_id.trim().is_empty()
                || cell.source_environment_id == cell.target_environment_id
                || !known.contains(pair.0)
                || !known.contains(pair.1)
                || previous_pair.is_some_and(|previous| previous >= pair)
                || cell.case_count == 0
                || cell.evaluation_sample_count == 0
                || !cell.mean_improvement.is_finite()
                || !unit_interval(cell.positive_transfer_rate)
                || !unit_interval(cell.negative_transfer_rate)
                || cell.positive_transfer_rate + cell.negative_transfer_rate > 1.0 + f64::EPSILON
                || !cell.mean_resource_cost.is_finite()
                || cell.mean_resource_cost < 0.0
                || cell
                    .standard_error
                    .is_some_and(|value| !value.is_finite() || value < 0.0)
            {
                return Err(BenchmarkError::InvalidTransferMatrix);
            }
            previous_pair = Some(pair);
        }
        Ok(())
    }

    pub fn cell(
        &self,
        source_environment_id: &str,
        target_environment_id: &str,
    ) -> Option<&TransferMatrixCell> {
        self.cells.iter().find(|cell| {
            cell.source_environment_id == source_environment_id
                && cell.target_environment_id == target_environment_id
        })
    }

    /// Finds directed pairs with a sufficiently high observed negative-case rate.
    ///
    /// A diagnosis is a benchmark warning, not proof that transfer caused the
    /// degradation. Small sample groups are excluded by the caller's minimum.
    pub fn diagnose_negative_transfer(
        &self,
        policy: NegativeTransferPolicy,
    ) -> Result<Vec<NegativeTransferDiagnosis>, BenchmarkError> {
        self.validate()?;
        policy.validate()?;
        Ok(self
            .cells
            .iter()
            .filter(|cell| {
                cell.case_count >= policy.minimum_cases
                    && cell.negative_transfer_rate >= policy.minimum_negative_rate
            })
            .map(|cell| NegativeTransferDiagnosis {
                source_environment_id: cell.source_environment_id.clone(),
                target_environment_id: cell.target_environment_id.clone(),
                case_count: cell.case_count,
                negative_transfer_rate: cell.negative_transfer_rate,
                mean_improvement: cell.mean_improvement,
                mean_improvement_is_negative: cell.mean_improvement < 0.0,
            })
            .collect())
    }

    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "metric_id".to_owned(),
                DataValue::Text(self.metric_id.clone()),
            ),
            (
                "direction".to_owned(),
                DataValue::Text(direction_name(self.direction).to_owned()),
            ),
            (
                "environment_ids".to_owned(),
                DataValue::Sequence(
                    self.environment_ids
                        .iter()
                        .cloned()
                        .map(DataValue::Text)
                        .collect(),
                ),
            ),
            (
                "cells".to_owned(),
                DataValue::Sequence(
                    self.cells
                        .iter()
                        .map(|cell| {
                            DataValue::object([
                                (
                                    "source".to_owned(),
                                    DataValue::Text(cell.source_environment_id.clone()),
                                ),
                                (
                                    "target".to_owned(),
                                    DataValue::Text(cell.target_environment_id.clone()),
                                ),
                                (
                                    "case_count".to_owned(),
                                    DataValue::Text(cell.case_count.to_string()),
                                ),
                                (
                                    "evaluation_sample_count".to_owned(),
                                    DataValue::Text(cell.evaluation_sample_count.to_string()),
                                ),
                                (
                                    "mean_improvement".to_owned(),
                                    DataValue::Number(cell.mean_improvement),
                                ),
                                (
                                    "positive_transfer_rate".to_owned(),
                                    DataValue::Number(cell.positive_transfer_rate),
                                ),
                                (
                                    "negative_transfer_rate".to_owned(),
                                    DataValue::Number(cell.negative_transfer_rate),
                                ),
                                (
                                    "mean_resource_cost".to_owned(),
                                    DataValue::Number(cell.mean_resource_cost),
                                ),
                                (
                                    "standard_error".to_owned(),
                                    cell.standard_error
                                        .map(DataValue::Number)
                                        .unwrap_or(DataValue::Null),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }

    pub fn from_artifact_value(value: &DataValue) -> Result<Self, BenchmarkError> {
        let fields = value
            .as_object()
            .ok_or(BenchmarkError::InvalidTransferMatrix)?;
        if fields.len() != 5 || fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
            return Err(BenchmarkError::InvalidTransferMatrix);
        }
        let text = |object: &BTreeMap<String, DataValue>, name: &str| match object.get(name) {
            Some(DataValue::Text(value)) => Ok(value.clone()),
            _ => Err(BenchmarkError::InvalidTransferMatrix),
        };
        let number = |object: &BTreeMap<String, DataValue>, name: &str| {
            object
                .get(name)
                .and_then(DataValue::as_f64)
                .filter(|value| value.is_finite())
                .ok_or(BenchmarkError::InvalidTransferMatrix)
        };
        let direction = match text(fields, "direction")?.as_str() {
            "higher" => ScoreDirection::HigherIsBetter,
            "lower" => ScoreDirection::LowerIsBetter,
            _ => return Err(BenchmarkError::InvalidTransferMatrix),
        };
        let environment_ids = match fields.get("environment_ids") {
            Some(DataValue::Sequence(values)) => values
                .iter()
                .map(|value| match value {
                    DataValue::Text(value) => Ok(value.clone()),
                    _ => Err(BenchmarkError::InvalidTransferMatrix),
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(BenchmarkError::InvalidTransferMatrix),
        };
        let cells = match fields.get("cells") {
            Some(DataValue::Sequence(values)) => values
                .iter()
                .map(|value| {
                    let cell = value
                        .as_object()
                        .ok_or(BenchmarkError::InvalidTransferMatrix)?;
                    if cell.len() != 9 {
                        return Err(BenchmarkError::InvalidTransferMatrix);
                    }
                    let parse_usize = |name: &str| {
                        text(cell, name)?
                            .parse::<usize>()
                            .map_err(|_| BenchmarkError::InvalidTransferMatrix)
                    };
                    let standard_error = match cell.get("standard_error") {
                        Some(DataValue::Null) => None,
                        Some(DataValue::Number(value)) if value.is_finite() => Some(*value),
                        Some(DataValue::Integer(value)) => Some(*value as f64),
                        _ => return Err(BenchmarkError::InvalidTransferMatrix),
                    };
                    Ok(TransferMatrixCell {
                        source_environment_id: text(cell, "source")?,
                        target_environment_id: text(cell, "target")?,
                        case_count: parse_usize("case_count")?,
                        evaluation_sample_count: parse_usize("evaluation_sample_count")?,
                        mean_improvement: number(cell, "mean_improvement")?,
                        positive_transfer_rate: number(cell, "positive_transfer_rate")?,
                        negative_transfer_rate: number(cell, "negative_transfer_rate")?,
                        mean_resource_cost: number(cell, "mean_resource_cost")?,
                        standard_error,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(BenchmarkError::InvalidTransferMatrix),
        };
        let matrix = Self {
            metric_id: text(fields, "metric_id")?,
            direction,
            environment_ids,
            cells,
        };
        matrix.validate()?;
        Ok(matrix)
    }
}

fn unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegativeTransferPolicy {
    pub minimum_cases: usize,
    pub minimum_negative_rate: f64,
}

impl NegativeTransferPolicy {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.minimum_cases == 0 || !unit_interval(self.minimum_negative_rate) {
            return Err(BenchmarkError::InvalidNegativeTransferPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NegativeTransferDiagnosis {
    pub source_environment_id: String,
    pub target_environment_id: String,
    pub case_count: usize,
    pub negative_transfer_rate: f64,
    pub mean_improvement: f64,
    pub mean_improvement_is_negative: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BenchmarkError {
    InvalidEvaluation,
    NotCrossEnvironment,
    DatasetLeakage,
    DuplicateCase,
    InvalidMetric,
    NoMatchingCases,
    InconsistentMetricDirection,
    NumericalOverflow,
    InvalidSerializedState,
    InvalidAdaptationCurve,
    InvalidTransferMatrix,
    InvalidNegativeTransferPolicy,
}

impl std::fmt::Display for BenchmarkError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidEvaluation => write!(formatter, "transfer evaluation is invalid"),
            Self::NotCrossEnvironment => write!(
                formatter,
                "transfer evaluation must use distinct source and target environments"
            ),
            Self::DatasetLeakage => write!(
                formatter,
                "source training and target evaluation datasets must differ"
            ),
            Self::DuplicateCase => write!(formatter, "transfer benchmark case ID is duplicated"),
            Self::InvalidMetric => write!(formatter, "benchmark metric ID is invalid"),
            Self::NoMatchingCases => write!(formatter, "benchmark has no cases for this metric"),
            Self::InconsistentMetricDirection => {
                write!(formatter, "benchmark metric direction is inconsistent")
            }
            Self::NumericalOverflow => write!(formatter, "benchmark aggregate overflowed"),
            Self::InvalidSerializedState => {
                write!(formatter, "serialized transfer benchmark is invalid")
            }
            Self::InvalidAdaptationCurve => {
                write!(formatter, "few-shot transfer adaptation curve is invalid")
            }
            Self::InvalidTransferMatrix => {
                write!(
                    formatter,
                    "serialized or aggregated transfer matrix is invalid"
                )
            }
            Self::InvalidNegativeTransferPolicy => {
                write!(formatter, "negative-transfer diagnosis policy is invalid")
            }
        }
    }
}

impl std::error::Error for BenchmarkError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluation(
        case_id: &str,
        source: &str,
        target: &str,
        baseline: f64,
        transferred: f64,
        direction: ScoreDirection,
    ) -> TransferEvaluation {
        TransferEvaluation {
            case_id: case_id.to_owned(),
            source_environment_id: source.to_owned(),
            target_environment_id: target.to_owned(),
            source_training_dataset_id: format!("{source}-train"),
            target_evaluation_dataset_id: format!("{target}-holdout"),
            model_artifact_ref: "model-sha-1".to_owned(),
            metric_id: "prediction-error".to_owned(),
            direction,
            baseline_score: baseline,
            transferred_score: transferred,
            sample_count: 20,
            resource_cost: 4.0,
        }
    }

    #[test]
    fn transfer_summary_reports_gain_win_rate_and_uncertainty() {
        let mut benchmark = TransferBenchmark::default();
        benchmark
            .record(evaluation(
                "case-1",
                "physics",
                "grid",
                0.5,
                0.3,
                ScoreDirection::LowerIsBetter,
            ))
            .expect("case 1");
        benchmark
            .record(evaluation(
                "case-2",
                "workflow",
                "grid",
                0.6,
                0.4,
                ScoreDirection::LowerIsBetter,
            ))
            .expect("case 2");
        benchmark
            .record(evaluation(
                "case-3",
                "physics",
                "robotics",
                0.4,
                0.5,
                ScoreDirection::LowerIsBetter,
            ))
            .expect("case 3");
        let summary = benchmark.summarize("prediction-error").expect("summary");
        assert_eq!(summary.case_count, 3);
        assert_eq!(summary.total_evaluation_samples, 60);
        assert!((summary.mean_improvement - 0.1).abs() < 1e-12);
        assert!((summary.positive_transfer_rate - 2.0 / 3.0).abs() < 1e-12);
        assert!(summary.standard_error.is_some());
    }

    #[test]
    fn rejects_same_environment_or_reused_dataset() {
        let mut benchmark = TransferBenchmark::default();
        assert_eq!(
            benchmark.record(evaluation(
                "same",
                "grid",
                "grid",
                0.2,
                0.1,
                ScoreDirection::LowerIsBetter
            )),
            Err(BenchmarkError::NotCrossEnvironment)
        );
        let mut leaked = evaluation(
            "leak",
            "physics",
            "robotics",
            0.2,
            0.1,
            ScoreDirection::LowerIsBetter,
        );
        leaked.target_evaluation_dataset_id = leaked.source_training_dataset_id.clone();
        assert_eq!(
            benchmark.record(leaked),
            Err(BenchmarkError::DatasetLeakage)
        );
    }

    #[test]
    fn few_shot_transfer_curve_validates_and_measures_area() {
        let curve = FewShotTransferCurve {
            source_environment_id: "physics".to_owned(),
            target_environment_id: "robotics".to_owned(),
            source_training_dataset_id: "physics-train".to_owned(),
            target_evaluation_dataset_id: "robotics-holdout".to_owned(),
            model_artifact_ref: "transferred-model-v1".to_owned(),
            metric_id: "rmse".to_owned(),
            direction: ScoreDirection::LowerIsBetter,
            evaluation_sample_count: 100,
            points: vec![
                TransferAdaptationPoint {
                    adaptation_samples: 1,
                    baseline_score: 1.0,
                    transferred_score: 0.8,
                    resource_cost: 1.0,
                },
                TransferAdaptationPoint {
                    adaptation_samples: 5,
                    baseline_score: 1.0,
                    transferred_score: 0.6,
                    resource_cost: 2.0,
                },
                TransferAdaptationPoint {
                    adaptation_samples: 10,
                    baseline_score: 1.0,
                    transferred_score: 0.5,
                    resource_cost: 3.0,
                },
            ],
        };
        let improvements = curve.improvements().expect("improvements");
        assert_eq!(improvements.len(), 3);
        assert!((improvements[0] - 0.2).abs() < 1e-12);
        assert!((improvements[1] - 0.4).abs() < 1e-12);
        assert!((improvements[2] - 0.5).abs() < 1e-12);
        assert!((curve.area_under_improvement_curve().expect("area") - 3.45).abs() < 1e-12);
        assert_eq!(
            FewShotTransferCurve::from_artifact_value(&curve.to_artifact_value()).expect("restore"),
            curve
        );
    }

    #[test]
    fn few_shot_transfer_curve_rejects_non_monotonic_shots() {
        let curve = FewShotTransferCurve {
            source_environment_id: "source".to_owned(),
            target_environment_id: "target".to_owned(),
            source_training_dataset_id: "train".to_owned(),
            target_evaluation_dataset_id: "holdout".to_owned(),
            model_artifact_ref: "model".to_owned(),
            metric_id: "rmse".to_owned(),
            direction: ScoreDirection::LowerIsBetter,
            evaluation_sample_count: 2,
            points: vec![
                TransferAdaptationPoint {
                    adaptation_samples: 4,
                    baseline_score: 1.0,
                    transferred_score: 0.8,
                    resource_cost: 1.0,
                },
                TransferAdaptationPoint {
                    adaptation_samples: 2,
                    baseline_score: 1.0,
                    transferred_score: 0.7,
                    resource_cost: 1.0,
                },
            ],
        };
        assert_eq!(
            curve.validate(),
            Err(BenchmarkError::InvalidAdaptationCurve)
        );
    }

    #[test]
    fn transfer_matrix_groups_directed_pairs_and_flags_negative_transfer() {
        let mut benchmark = TransferBenchmark::default();
        for evaluation in [
            evaluation(
                "p-grid-1",
                "physics",
                "grid",
                1.0,
                0.7,
                ScoreDirection::LowerIsBetter,
            ),
            evaluation(
                "p-grid-2",
                "physics",
                "grid",
                1.0,
                1.2,
                ScoreDirection::LowerIsBetter,
            ),
            evaluation(
                "p-grid-3",
                "physics",
                "grid",
                1.0,
                1.4,
                ScoreDirection::LowerIsBetter,
            ),
            evaluation(
                "w-grid-1",
                "workflow",
                "grid",
                1.0,
                0.8,
                ScoreDirection::LowerIsBetter,
            ),
            evaluation(
                "grid-w-1",
                "grid",
                "workflow",
                1.0,
                0.7,
                ScoreDirection::LowerIsBetter,
            ),
        ] {
            benchmark.record(evaluation).expect("record");
        }
        let matrix =
            TransferMatrix::from_benchmark(&benchmark, "prediction-error").expect("matrix");
        assert_eq!(
            matrix.environment_ids,
            vec![
                "grid".to_owned(),
                "physics".to_owned(),
                "workflow".to_owned()
            ]
        );
        assert_eq!(matrix.cells.len(), 3);
        let physics_to_grid = matrix.cell("physics", "grid").expect("pair");
        assert_eq!(physics_to_grid.case_count, 3);
        assert_eq!(physics_to_grid.evaluation_sample_count, 60);
        assert!((physics_to_grid.negative_transfer_rate - (2.0 / 3.0)).abs() < 1e-12);
        assert!(physics_to_grid.mean_improvement < 0.0);
        assert!(matrix.cell("grid", "physics").is_none());

        let diagnoses = matrix
            .diagnose_negative_transfer(NegativeTransferPolicy {
                minimum_cases: 3,
                minimum_negative_rate: 0.6,
            })
            .expect("diagnoses");
        assert_eq!(diagnoses.len(), 1);
        assert_eq!(diagnoses[0].source_environment_id, "physics");
        assert_eq!(diagnoses[0].target_environment_id, "grid");
        assert!(diagnoses[0].mean_improvement_is_negative);
        assert_eq!(
            TransferMatrix::from_artifact_value(&matrix.to_artifact_value()).expect("restore"),
            matrix
        );
    }

    #[test]
    fn transfer_matrix_rejects_corruption_and_invalid_diagnosis_policy() {
        let mut benchmark = TransferBenchmark::default();
        benchmark
            .record(evaluation(
                "one",
                "source",
                "target",
                0.5,
                0.4,
                ScoreDirection::LowerIsBetter,
            ))
            .expect("record");
        let matrix =
            TransferMatrix::from_benchmark(&benchmark, "prediction-error").expect("matrix");
        assert_eq!(
            matrix.diagnose_negative_transfer(NegativeTransferPolicy {
                minimum_cases: 0,
                minimum_negative_rate: 0.5,
            }),
            Err(BenchmarkError::InvalidNegativeTransferPolicy)
        );
        let mut value = matrix.to_artifact_value();
        let object = match &mut value {
            DataValue::Object(fields) => fields,
            _ => unreachable!(),
        };
        object.insert("unexpected".to_owned(), DataValue::Null);
        assert_eq!(
            TransferMatrix::from_artifact_value(&value),
            Err(BenchmarkError::InvalidTransferMatrix)
        );
    }

    #[test]
    fn benchmark_artifact_round_trips() {
        let mut benchmark = TransferBenchmark::default();
        benchmark
            .record(evaluation(
                "case-1",
                "physics",
                "grid",
                0.5,
                0.3,
                ScoreDirection::LowerIsBetter,
            ))
            .expect("case");
        let restored = TransferBenchmark::from_artifact_value(&benchmark.to_artifact_value())
            .expect("restore");
        assert_eq!(restored, benchmark);
    }
}
