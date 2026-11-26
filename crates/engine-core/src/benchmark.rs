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

/// One repeated-seed comparison of a learning strategy against a reference
/// strategy at increasing training-sample budgets.
#[derive(Clone, Debug, PartialEq)]
pub struct LearningEfficiencyPoint {
    pub training_samples: usize,
    pub learner_score: f64,
    pub reference_score: f64,
    pub learner_resource_cost: f64,
    pub reference_resource_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LearningEfficiencyCurve {
    pub curve_id: String,
    pub environment_id: String,
    pub environment_version: String,
    pub task_id: String,
    pub learner_strategy_id: String,
    pub reference_strategy_id: String,
    pub learner_artifact_ref: String,
    pub training_dataset_id: String,
    pub evaluation_dataset_id: String,
    pub metric_id: String,
    pub direction: ScoreDirection,
    pub seed: u64,
    pub evaluation_sample_count: usize,
    pub points: Vec<LearningEfficiencyPoint>,
}

impl LearningEfficiencyCurve {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.curve_id.trim().is_empty()
            || self.environment_id.trim().is_empty()
            || self.environment_version.trim().is_empty()
            || self.task_id.trim().is_empty()
            || self.learner_strategy_id.trim().is_empty()
            || self.reference_strategy_id.trim().is_empty()
            || self.learner_strategy_id == self.reference_strategy_id
            || self.learner_artifact_ref.trim().is_empty()
            || self.training_dataset_id.trim().is_empty()
            || self.evaluation_dataset_id.trim().is_empty()
            || self.training_dataset_id == self.evaluation_dataset_id
            || self.metric_id.trim().is_empty()
            || self.evaluation_sample_count == 0
            || self.points.len() < 2
        {
            return Err(BenchmarkError::InvalidLearningEfficiencyCurve);
        }
        let mut previous_samples = None;
        for point in &self.points {
            if previous_samples.is_some_and(|previous| point.training_samples <= previous)
                || !point.learner_score.is_finite()
                || !point.reference_score.is_finite()
                || !point.learner_resource_cost.is_finite()
                || !point.reference_resource_cost.is_finite()
                || point.learner_resource_cost < 0.0
                || point.reference_resource_cost < 0.0
            {
                return Err(BenchmarkError::InvalidLearningEfficiencyCurve);
            }
            previous_samples = Some(point.training_samples);
        }
        if self
            .improvements_unchecked()
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(BenchmarkError::NumericalOverflow);
        }
        Ok(())
    }

    fn improvements_unchecked(&self) -> Vec<f64> {
        self.points
            .iter()
            .map(|point| match self.direction {
                ScoreDirection::HigherIsBetter => point.learner_score - point.reference_score,
                ScoreDirection::LowerIsBetter => point.reference_score - point.learner_score,
            })
            .collect()
    }

    pub fn improvements(&self) -> Result<Vec<f64>, BenchmarkError> {
        self.validate()?;
        Ok(self.improvements_unchecked())
    }

    /// Raw trapezoidal area under improvement versus training samples.
    pub fn area_under_improvement_curve(&self) -> Result<f64, BenchmarkError> {
        let improvements = self.improvements()?;
        let mut area = 0.0;
        for (index, pair) in self.points.windows(2).enumerate() {
            let width = (pair[1].training_samples - pair[0].training_samples) as f64;
            area += width * (improvements[index] + improvements[index + 1]) / 2.0;
            if !area.is_finite() {
                return Err(BenchmarkError::NumericalOverflow);
            }
        }
        Ok(area)
    }

    /// Returns the first training budget at which the learner reaches a
    /// caller-defined improvement threshold against the reference strategy.
    pub fn samples_to_target_improvement(
        &self,
        target_improvement: f64,
    ) -> Result<Option<usize>, BenchmarkError> {
        if !target_improvement.is_finite() || target_improvement < 0.0 {
            return Err(BenchmarkError::InvalidLearningEfficiencyTarget);
        }
        let improvements = self.improvements()?;
        Ok(self
            .points
            .iter()
            .zip(improvements)
            .find_map(|(point, improvement)| {
                (improvement >= target_improvement).then_some(point.training_samples)
            }))
    }

    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "curve_id".to_owned(),
                DataValue::Text(self.curve_id.clone()),
            ),
            (
                "environment_id".to_owned(),
                DataValue::Text(self.environment_id.clone()),
            ),
            (
                "environment_version".to_owned(),
                DataValue::Text(self.environment_version.clone()),
            ),
            ("task_id".to_owned(), DataValue::Text(self.task_id.clone())),
            (
                "learner_strategy_id".to_owned(),
                DataValue::Text(self.learner_strategy_id.clone()),
            ),
            (
                "reference_strategy_id".to_owned(),
                DataValue::Text(self.reference_strategy_id.clone()),
            ),
            (
                "learner_artifact_ref".to_owned(),
                DataValue::Text(self.learner_artifact_ref.clone()),
            ),
            (
                "training_dataset_id".to_owned(),
                DataValue::Text(self.training_dataset_id.clone()),
            ),
            (
                "evaluation_dataset_id".to_owned(),
                DataValue::Text(self.evaluation_dataset_id.clone()),
            ),
            (
                "metric_id".to_owned(),
                DataValue::Text(self.metric_id.clone()),
            ),
            (
                "direction".to_owned(),
                DataValue::Text(direction_name(self.direction).to_owned()),
            ),
            ("seed".to_owned(), DataValue::Text(self.seed.to_string())),
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
                                    "training_samples".to_owned(),
                                    DataValue::Text(point.training_samples.to_string()),
                                ),
                                (
                                    "learner_score".to_owned(),
                                    DataValue::Number(point.learner_score),
                                ),
                                (
                                    "reference_score".to_owned(),
                                    DataValue::Number(point.reference_score),
                                ),
                                (
                                    "learner_resource_cost".to_owned(),
                                    DataValue::Number(point.learner_resource_cost),
                                ),
                                (
                                    "reference_resource_cost".to_owned(),
                                    DataValue::Number(point.reference_resource_cost),
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
            .ok_or(BenchmarkError::InvalidLearningEfficiencyCurve)?;
        if fields.len() != 15 || fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
            return Err(BenchmarkError::InvalidLearningEfficiencyCurve);
        }
        let text = |name: &str| match fields.get(name) {
            Some(DataValue::Text(value)) => Ok(value.clone()),
            _ => Err(BenchmarkError::InvalidLearningEfficiencyCurve),
        };
        let parse_usize = |name: &str| {
            text(name)?
                .parse::<usize>()
                .map_err(|_| BenchmarkError::InvalidLearningEfficiencyCurve)
        };
        let direction = match text("direction")?.as_str() {
            "higher" => ScoreDirection::HigherIsBetter,
            "lower" => ScoreDirection::LowerIsBetter,
            _ => return Err(BenchmarkError::InvalidLearningEfficiencyCurve),
        };
        let points = match fields.get("points") {
            Some(DataValue::Sequence(values)) => values
                .iter()
                .map(|value| {
                    let point = value
                        .as_object()
                        .ok_or(BenchmarkError::InvalidLearningEfficiencyCurve)?;
                    if point.len() != 5 {
                        return Err(BenchmarkError::InvalidLearningEfficiencyCurve);
                    }
                    let text = |name: &str| match point.get(name) {
                        Some(DataValue::Text(value)) => Ok(value.clone()),
                        _ => Err(BenchmarkError::InvalidLearningEfficiencyCurve),
                    };
                    let number = |name: &str| {
                        point
                            .get(name)
                            .and_then(DataValue::as_f64)
                            .filter(|value| value.is_finite())
                            .ok_or(BenchmarkError::InvalidLearningEfficiencyCurve)
                    };
                    Ok(LearningEfficiencyPoint {
                        training_samples: text("training_samples")?
                            .parse()
                            .map_err(|_| BenchmarkError::InvalidLearningEfficiencyCurve)?,
                        learner_score: number("learner_score")?,
                        reference_score: number("reference_score")?,
                        learner_resource_cost: number("learner_resource_cost")?,
                        reference_resource_cost: number("reference_resource_cost")?,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(BenchmarkError::InvalidLearningEfficiencyCurve),
        };
        let curve = Self {
            curve_id: text("curve_id")?,
            environment_id: text("environment_id")?,
            environment_version: text("environment_version")?,
            task_id: text("task_id")?,
            learner_strategy_id: text("learner_strategy_id")?,
            reference_strategy_id: text("reference_strategy_id")?,
            learner_artifact_ref: text("learner_artifact_ref")?,
            training_dataset_id: text("training_dataset_id")?,
            evaluation_dataset_id: text("evaluation_dataset_id")?,
            metric_id: text("metric_id")?,
            direction,
            seed: text("seed")?
                .parse()
                .map_err(|_| BenchmarkError::InvalidLearningEfficiencyCurve)?,
            evaluation_sample_count: parse_usize("evaluation_sample_count")?,
            points,
        };
        curve.validate()?;
        Ok(curve)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LearningEfficiencyBenchmark {
    curves: BTreeMap<String, LearningEfficiencyCurve>,
}

impl LearningEfficiencyBenchmark {
    pub fn record(&mut self, curve: LearningEfficiencyCurve) -> Result<(), BenchmarkError> {
        curve.validate()?;
        if self.curves.contains_key(&curve.curve_id) {
            return Err(BenchmarkError::DuplicateLearningCurve);
        }
        let key = learning_efficiency_group_key(&curve);
        if self
            .curves
            .values()
            .any(|existing| learning_efficiency_group_key(existing) == key && existing.seed == curve.seed)
        {
            return Err(BenchmarkError::DuplicateLearningSeed);
        }
        self.curves.insert(curve.curve_id.clone(), curve);
        Ok(())
    }

    pub fn get(&self, curve_id: &str) -> Option<&LearningEfficiencyCurve> {
        self.curves.get(curve_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &LearningEfficiencyCurve> {
        self.curves.values()
    }

    pub fn len(&self) -> usize {
        self.curves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.curves.is_empty()
    }

    /// Aggregates only repeated runs with exactly matching task, strategies,
    /// metric direction, evaluation size, and training-sample budgets.
    pub fn summarize(
        &self,
        metric_id: &str,
        target_improvement: f64,
    ) -> Result<Vec<LearningEfficiencySummary>, BenchmarkError> {
        if metric_id.trim().is_empty() {
            return Err(BenchmarkError::InvalidMetric);
        }
        if !target_improvement.is_finite() || target_improvement < 0.0 {
            return Err(BenchmarkError::InvalidLearningEfficiencyTarget);
        }
        let mut groups: BTreeMap<LearningEfficiencyGroupKey, Vec<&LearningEfficiencyCurve>> =
            BTreeMap::new();
        for curve in self.curves.values().filter(|curve| curve.metric_id == metric_id) {
            groups
                .entry(learning_efficiency_group_key(curve))
                .or_default()
                .push(curve);
        }
        if groups.is_empty() {
            return Err(BenchmarkError::NoMatchingCases);
        }

        let mut summaries = Vec::with_capacity(groups.len());
        for (key, curves) in groups {
            let first = curves
                .first()
                .copied()
                .ok_or(BenchmarkError::InvalidLearningEfficiencyCurve)?;
            let budgets: Vec<usize> = first
                .points
                .iter()
                .map(|point| point.training_samples)
                .collect();
            let mut mean_improvement_by_budget = Vec::with_capacity(budgets.len());
            let mut standard_error_by_budget = Vec::with_capacity(budgets.len());
            for index in 0..budgets.len() {
                let values: Vec<f64> = curves
                    .iter()
                    .map(|curve| curve.improvements_unchecked()[index])
                    .collect();
                let (mean, standard_error) = mean_and_standard_error(&values)?;
                mean_improvement_by_budget.push(mean);
                standard_error_by_budget.push(standard_error);
            }
            let areas = curves
                .iter()
                .map(|curve| curve.area_under_improvement_curve())
                .collect::<Result<Vec<_>, _>>()?;
            let (mean_area_under_curve, standard_error_area_under_curve) =
                mean_and_standard_error(&areas)?;
            let target_budgets: Vec<usize> = curves
                .iter()
                .filter_map(|curve| {
                    curve
                        .samples_to_target_improvement(target_improvement)
                        .ok()
                        .flatten()
                })
                .collect();
            let target_reach_rate = target_budgets.len() as f64 / curves.len() as f64;
            let mean_samples_to_target = if target_budgets.is_empty() {
                None
            } else {
                Some(
                    target_budgets.iter().map(|value| *value as f64).sum::<f64>()
                        / target_budgets.len() as f64,
                )
            };
            summaries.push(LearningEfficiencySummary {
                environment_id: key.environment_id,
                environment_version: key.environment_version,
                task_id: key.task_id,
                learner_strategy_id: key.learner_strategy_id,
                reference_strategy_id: key.reference_strategy_id,
                metric_id: key.metric_id,
                direction: first.direction,
                evaluation_sample_count: key.evaluation_sample_count,
                seed_count: curves.len(),
                sample_budgets: budgets,
                mean_improvement_by_budget,
                standard_error_by_budget,
                mean_area_under_curve,
                standard_error_area_under_curve,
                target_improvement,
                target_reach_rate,
                mean_samples_to_target,
            });
        }
        Ok(summaries)
    }

    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "curves".to_owned(),
                DataValue::Sequence(
                    self.curves
                        .values()
                        .map(LearningEfficiencyCurve::to_artifact_value)
                        .collect(),
                ),
            ),
        ])
    }

    pub fn from_artifact_value(value: &DataValue) -> Result<Self, BenchmarkError> {
        let fields = value
            .as_object()
            .ok_or(BenchmarkError::InvalidLearningEfficiencyBenchmark)?;
        if fields.len() != 2
            || fields.get("artifact_version") != Some(&DataValue::Integer(1))
        {
            return Err(BenchmarkError::InvalidLearningEfficiencyBenchmark);
        }
        let encoded = match fields.get("curves") {
            Some(DataValue::Sequence(values)) => values,
            _ => return Err(BenchmarkError::InvalidLearningEfficiencyBenchmark),
        };
        let mut benchmark = Self::default();
        for value in encoded {
            let curve = LearningEfficiencyCurve::from_artifact_value(value)
                .map_err(|_| BenchmarkError::InvalidLearningEfficiencyBenchmark)?;
            benchmark.record(curve)
                .map_err(|_| BenchmarkError::InvalidLearningEfficiencyBenchmark)?;
        }
        Ok(benchmark)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LearningEfficiencyGroupKey {
    environment_id: String,
    environment_version: String,
    task_id: String,
    learner_strategy_id: String,
    reference_strategy_id: String,
    metric_id: String,
    direction: String,
    evaluation_sample_count: usize,
    sample_budgets: Vec<usize>,
}

fn learning_efficiency_group_key(curve: &LearningEfficiencyCurve) -> LearningEfficiencyGroupKey {
    LearningEfficiencyGroupKey {
        environment_id: curve.environment_id.clone(),
        environment_version: curve.environment_version.clone(),
        task_id: curve.task_id.clone(),
        learner_strategy_id: curve.learner_strategy_id.clone(),
        reference_strategy_id: curve.reference_strategy_id.clone(),
        metric_id: curve.metric_id.clone(),
        direction: direction_name(curve.direction).to_owned(),
        evaluation_sample_count: curve.evaluation_sample_count,
        sample_budgets: curve.points.iter().map(|point| point.training_samples).collect(),
    }
}

fn mean_and_standard_error(values: &[f64]) -> Result<(f64, Option<f64>), BenchmarkError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(BenchmarkError::NumericalOverflow);
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    if !mean.is_finite() {
        return Err(BenchmarkError::NumericalOverflow);
    }
    if values.len() < 2 {
        return Ok((mean, None));
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (count - 1.0);
    let standard_error = (variance / count).sqrt();
    if !variance.is_finite() || !standard_error.is_finite() {
        return Err(BenchmarkError::NumericalOverflow);
    }
    Ok((mean, Some(standard_error)))
}

#[derive(Clone, Debug, PartialEq)]
pub struct LearningEfficiencySummary {
    pub environment_id: String,
    pub environment_version: String,
    pub task_id: String,
    pub learner_strategy_id: String,
    pub reference_strategy_id: String,
    pub metric_id: String,
    pub direction: ScoreDirection,
    pub evaluation_sample_count: usize,
    pub seed_count: usize,
    pub sample_budgets: Vec<usize>,
    pub mean_improvement_by_budget: Vec<f64>,
    pub standard_error_by_budget: Vec<Option<f64>>,
    pub mean_area_under_curve: f64,
    pub standard_error_area_under_curve: Option<f64>,
    pub target_improvement: f64,
    pub target_reach_rate: f64,
    /// Mean sample budget among runs that reached the target; None if no run did.
    pub mean_samples_to_target: Option<f64>,
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
    InvalidLearningEfficiencyCurve,
    InvalidLearningEfficiencyTarget,
    DuplicateLearningCurve,
    DuplicateLearningSeed,
    InvalidLearningEfficiencyBenchmark,
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
            Self::InvalidLearningEfficiencyCurve => {
                write!(formatter, "learning-efficiency curve is invalid")
            }
            Self::InvalidLearningEfficiencyTarget => {
                write!(
                    formatter,
                    "learning-efficiency target must be finite and non-negative"
                )
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
    fn learning_efficiency_curve_compares_strategies_and_finds_sample_threshold() {
        let curve = LearningEfficiencyCurve {
            curve_id: "active-versus-random-seed-7".to_owned(),
            environment_id: "grid".to_owned(),
            environment_version: "v2".to_owned(),
            task_id: "reach-goal".to_owned(),
            learner_strategy_id: "active-learning".to_owned(),
            reference_strategy_id: "random-exploration".to_owned(),
            learner_artifact_ref: "model-active-v3".to_owned(),
            training_dataset_id: "grid-training-seed-7".to_owned(),
            evaluation_dataset_id: "grid-heldout-seed-7".to_owned(),
            metric_id: "goal-success-rate".to_owned(),
            direction: ScoreDirection::HigherIsBetter,
            seed: 7,
            evaluation_sample_count: 100,
            points: vec![
                LearningEfficiencyPoint {
                    training_samples: 0,
                    learner_score: 0.1,
                    reference_score: 0.1,
                    learner_resource_cost: 0.0,
                    reference_resource_cost: 0.0,
                },
                LearningEfficiencyPoint {
                    training_samples: 10,
                    learner_score: 0.4,
                    reference_score: 0.2,
                    learner_resource_cost: 2.0,
                    reference_resource_cost: 2.0,
                },
                LearningEfficiencyPoint {
                    training_samples: 20,
                    learner_score: 0.7,
                    reference_score: 0.3,
                    learner_resource_cost: 4.0,
                    reference_resource_cost: 4.0,
                },
            ],
        };
        let improvements = curve.improvements().expect("improvements");
        assert_eq!(improvements.len(), 3);
        assert!((improvements[0] - 0.0).abs() < 1e-12);
        assert!((improvements[1] - 0.2).abs() < 1e-12);
        assert!((improvements[2] - 0.4).abs() < 1e-12);
        assert!((curve.area_under_improvement_curve().expect("area") - 3.0).abs() < 1e-12);
        assert_eq!(
            curve
                .samples_to_target_improvement(0.25)
                .expect("threshold"),
            Some(20)
        );
        assert_eq!(
            curve.samples_to_target_improvement(0.5).expect("threshold"),
            None
        );
        assert_eq!(
            LearningEfficiencyCurve::from_artifact_value(&curve.to_artifact_value())
                .expect("restore"),
            curve
        );
    }

    fn learning_curve(curve_id: &str, seed: u64, learner_scores: [f64; 3]) -> LearningEfficiencyCurve {
        LearningEfficiencyCurve {
            curve_id: curve_id.to_owned(),
            environment_id: "grid".to_owned(),
            environment_version: "v1".to_owned(),
            task_id: "reach-goal".to_owned(),
            learner_strategy_id: "active".to_owned(),
            reference_strategy_id: "random".to_owned(),
            learner_artifact_ref: format!("active-model-{seed}"),
            training_dataset_id: format!("train-{seed}"),
            evaluation_dataset_id: format!("holdout-{seed}"),
            metric_id: "success-rate".to_owned(),
            direction: ScoreDirection::HigherIsBetter,
            seed,
            evaluation_sample_count: 100,
            points: [0usize, 10, 20]
                .into_iter()
                .zip(learner_scores)
                .map(|(training_samples, learner_score)| LearningEfficiencyPoint {
                    training_samples,
                    learner_score,
                    reference_score: 0.2,
                    learner_resource_cost: training_samples as f64,
                    reference_resource_cost: training_samples as f64,
                })
                .collect(),
        }
    }

    #[test]
    fn repeated_seed_benchmark_aggregates_only_comparable_curves() {
        let mut benchmark = LearningEfficiencyBenchmark::default();
        benchmark.record(learning_curve("seed-1", 1, [0.2, 0.4, 0.6])).expect("seed 1");
        benchmark.record(learning_curve("seed-2", 2, [0.2, 0.5, 0.7])).expect("seed 2");
        assert_eq!(
            benchmark.record(learning_curve("duplicate-seed", 1, [0.2, 0.5, 0.7])),
            Err(BenchmarkError::DuplicateLearningSeed)
        );
        let summaries = benchmark.summarize("success-rate", 0.35).expect("summaries");
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];
        assert_eq!(summary.seed_count, 2);
        assert_eq!(summary.sample_budgets, vec![0, 10, 20]);
        assert!((summary.mean_improvement_by_budget[0] - 0.0).abs() < 1e-12);
        assert!((summary.mean_improvement_by_budget[1] - 0.25).abs() < 1e-12);
        assert!((summary.mean_improvement_by_budget[2] - 0.45).abs() < 1e-12);
        assert!(summary.standard_error_by_budget[1].is_some());
        assert!((summary.target_reach_rate - 0.5).abs() < 1e-12);
        assert_eq!(summary.mean_samples_to_target, Some(20.0));
        let restored = LearningEfficiencyBenchmark::from_artifact_value(&benchmark.to_artifact_value())
            .expect("restore");
        assert_eq!(restored, benchmark);
    }

    #[test]
    fn repeated_seed_benchmark_keeps_different_budgets_in_separate_groups() {
        let mut benchmark = LearningEfficiencyBenchmark::default();
        let mut first = learning_curve("first", 1, [0.2, 0.5, 0.7]);
        let mut second = learning_curve("second", 2, [0.2, 0.5, 0.7]);
        second.points.pop();
        benchmark.record(first.clone()).expect("first");
        benchmark.record(second).expect("second");
        let summaries = benchmark.summarize("success-rate", 0.1).expect("summaries");
        assert_eq!(summaries.len(), 2);
        assert!(summaries.iter().all(|summary| summary.seed_count == 1));
        first.seed = 3;
        first.curve_id = "third".to_owned();
        first.environment_version = "v2".to_owned();
        benchmark.record(first).expect("different group");
    }

    #[test]
    fn learning_efficiency_curve_rejects_leakage_and_invalid_sample_budgets() {
        let mut curve = LearningEfficiencyCurve {
            curve_id: "curve".to_owned(),
            environment_id: "environment".to_owned(),
            environment_version: "v1".to_owned(),
            task_id: "task".to_owned(),
            learner_strategy_id: "learner".to_owned(),
            reference_strategy_id: "random".to_owned(),
            learner_artifact_ref: "model-v1".to_owned(),
            training_dataset_id: "same".to_owned(),
            evaluation_dataset_id: "same".to_owned(),
            metric_id: "rmse".to_owned(),
            direction: ScoreDirection::LowerIsBetter,
            seed: 0,
            evaluation_sample_count: 10,
            points: vec![
                LearningEfficiencyPoint {
                    training_samples: 5,
                    learner_score: 0.9,
                    reference_score: 1.0,
                    learner_resource_cost: 1.0,
                    reference_resource_cost: 1.0,
                },
                LearningEfficiencyPoint {
                    training_samples: 5,
                    learner_score: 0.8,
                    reference_score: 1.0,
                    learner_resource_cost: 2.0,
                    reference_resource_cost: 2.0,
                },
            ],
        };
        assert_eq!(
            curve.validate(),
            Err(BenchmarkError::InvalidLearningEfficiencyCurve)
        );
        curve.training_dataset_id = "train".to_owned();
        curve.evaluation_dataset_id = "holdout".to_owned();
        assert_eq!(
            curve.validate(),
            Err(BenchmarkError::InvalidLearningEfficiencyCurve)
        );
        curve.points[1].training_samples = 10;
        assert_eq!(curve.validate(), Ok(()));
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
