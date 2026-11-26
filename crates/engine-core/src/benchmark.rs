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
        self.evaluations.insert(evaluation.case_id.clone(), evaluation);
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
        if selected.iter().any(|evaluation| evaluation.direction != direction) {
            return Err(BenchmarkError::InconsistentMetricDirection);
        }
        let improvements: Vec<f64> = selected.iter().map(|evaluation| evaluation.improvement()).collect();
        let count = improvements.len() as f64;
        let mean_improvement = improvements.iter().sum::<f64>() / count;
        let wins = improvements.iter().filter(|improvement| **improvement > 0.0).count();
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
        let mean_resource_cost = selected.iter().map(|evaluation| evaluation.resource_cost).sum::<f64>() / count;
        let total_samples = selected.iter().try_fold(0usize, |sum, evaluation| {
            sum.checked_add(evaluation.sample_count).ok_or(BenchmarkError::NumericalOverflow)
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
        let fields = value.as_object().ok_or(BenchmarkError::InvalidSerializedState)?;
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
            if benchmark.evaluations.insert(evaluation.case_id.clone(), evaluation).is_some() {
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
        ("source_environment_id".to_owned(), DataValue::Text(value.source_environment_id.clone())),
        ("target_environment_id".to_owned(), DataValue::Text(value.target_environment_id.clone())),
        ("source_training_dataset_id".to_owned(), DataValue::Text(value.source_training_dataset_id.clone())),
        ("target_evaluation_dataset_id".to_owned(), DataValue::Text(value.target_evaluation_dataset_id.clone())),
        ("model_artifact_ref".to_owned(), DataValue::Text(value.model_artifact_ref.clone())),
        ("metric_id".to_owned(), DataValue::Text(value.metric_id.clone())),
        ("direction".to_owned(), DataValue::Text(direction_name(value.direction).to_owned())),
        ("baseline_score".to_owned(), DataValue::Number(value.baseline_score)),
        ("transferred_score".to_owned(), DataValue::Number(value.transferred_score)),
        ("sample_count".to_owned(), DataValue::Text(value.sample_count.to_string())),
        ("resource_cost".to_owned(), DataValue::Number(value.resource_cost)),
    ])
}

fn evaluation_from_value(value: &DataValue) -> Result<TransferEvaluation, BenchmarkError> {
    let fields = value.as_object().ok_or(BenchmarkError::InvalidSerializedState)?;
    if fields.len() != 12 {
        return Err(BenchmarkError::InvalidSerializedState);
    }
    let text = |name: &str| match fields.get(name) {
        Some(DataValue::Text(value)) => Ok(value.clone()),
        _ => Err(BenchmarkError::InvalidSerializedState),
    };
    let number = |name: &str| fields.get(name)
        .and_then(DataValue::as_f64)
        .filter(|value| value.is_finite())
        .ok_or(BenchmarkError::InvalidSerializedState);
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
    evaluation.validate().map_err(|_| BenchmarkError::InvalidSerializedState)?;
    Ok(evaluation)
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
}

impl std::fmt::Display for BenchmarkError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidEvaluation => write!(formatter, "transfer evaluation is invalid"),
            Self::NotCrossEnvironment => write!(formatter, "transfer evaluation must use distinct source and target environments"),
            Self::DatasetLeakage => write!(formatter, "source training and target evaluation datasets must differ"),
            Self::DuplicateCase => write!(formatter, "transfer benchmark case ID is duplicated"),
            Self::InvalidMetric => write!(formatter, "benchmark metric ID is invalid"),
            Self::NoMatchingCases => write!(formatter, "benchmark has no cases for this metric"),
            Self::InconsistentMetricDirection => write!(formatter, "benchmark metric direction is inconsistent"),
            Self::NumericalOverflow => write!(formatter, "benchmark aggregate overflowed"),
            Self::InvalidSerializedState => write!(formatter, "serialized transfer benchmark is invalid"),
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
        benchmark.record(evaluation("case-1", "physics", "grid", 0.5, 0.3, ScoreDirection::LowerIsBetter)).expect("case 1");
        benchmark.record(evaluation("case-2", "workflow", "grid", 0.6, 0.4, ScoreDirection::LowerIsBetter)).expect("case 2");
        benchmark.record(evaluation("case-3", "physics", "robotics", 0.4, 0.5, ScoreDirection::LowerIsBetter)).expect("case 3");
        let summary = benchmark.summarize("prediction-error").expect("summary");
        assert_eq!(summary.case_count, 3);
        assert_eq!(summary.total_evaluation_samples, 60);
        assert!((summary.mean_improvement - 0.1 / 3.0).abs() < 1e-12);
        assert!((summary.positive_transfer_rate - 2.0 / 3.0).abs() < 1e-12);
        assert!(summary.standard_error.is_some());
    }

    #[test]
    fn rejects_same_environment_or_reused_dataset() {
        let mut benchmark = TransferBenchmark::default();
        assert_eq!(
            benchmark.record(evaluation("same", "grid", "grid", 0.2, 0.1, ScoreDirection::LowerIsBetter)),
            Err(BenchmarkError::NotCrossEnvironment)
        );
        let mut leaked = evaluation("leak", "physics", "robotics", 0.2, 0.1, ScoreDirection::LowerIsBetter);
        leaked.target_evaluation_dataset_id = leaked.source_training_dataset_id.clone();
        assert_eq!(benchmark.record(leaked), Err(BenchmarkError::DatasetLeakage));
    }

    #[test]
    fn benchmark_artifact_round_trips() {
        let mut benchmark = TransferBenchmark::default();
        benchmark.record(evaluation("case-1", "physics", "grid", 0.5, 0.3, ScoreDirection::LowerIsBetter)).expect("case");
        let restored = TransferBenchmark::from_artifact_value(&benchmark.to_artifact_value()).expect("restore");
        assert_eq!(restored, benchmark);
    }
}
