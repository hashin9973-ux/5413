//! Executable, paired, multi-seed learning-efficiency benchmark protocols.
//!
//! The runner controls trial ordering, provenance checks, metric consistency,
//! resource accounting, aggregation, and persistence. Actual strategy training
//! and evaluation are delegated to a caller-supplied executor; dataset IDs are
//! declared provenance labels and do not prove physical disjointness.

use crate::benchmark::{
    BenchmarkError, LearningEfficiencyBenchmark, LearningEfficiencyCurve,
    LearningEfficiencyPoint, LearningEfficiencySummary, ScoreDirection,
};
use crate::model_artifact::{ModelArtifactError, ModelArtifactStore};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkSeedPlan {
    pub seed: u64,
    pub training_dataset_id: String,
    pub evaluation_dataset_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkProtocol {
    pub protocol_id: String,
    pub environment_id: String,
    pub environment_version: String,
    pub task_id: String,
    pub learner_strategy_id: String,
    pub reference_strategy_id: String,
    pub metric_id: String,
    pub direction: ScoreDirection,
    pub evaluation_sample_count: usize,
    /// Strictly increasing sample budgets. A zero-shot budget is supported.
    pub training_budgets: Vec<usize>,
    /// Every seed has its own training and holdout dataset identity.
    pub seeds: Vec<BenchmarkSeedPlan>,
    pub target_improvement: f64,
    pub maximum_total_evaluations: usize,
    pub maximum_total_resource_cost: f64,
    pub benchmark_artifact_ref: String,
}

impl BenchmarkProtocol {
    pub fn validate(&self) -> Result<(), BenchmarkProtocolError> {
        if self.protocol_id.trim().is_empty()
            || self.environment_id.trim().is_empty()
            || self.environment_version.trim().is_empty()
            || self.task_id.trim().is_empty()
            || self.learner_strategy_id.trim().is_empty()
            || self.reference_strategy_id.trim().is_empty()
            || self.learner_strategy_id == self.reference_strategy_id
            || self.metric_id.trim().is_empty()
            || self.evaluation_sample_count == 0
            || self.training_budgets.len() < 2
            || self.seeds.len() < 2
            || !self.target_improvement.is_finite()
            || self.target_improvement < 0.0
            || self.maximum_total_evaluations == 0
            || !self.maximum_total_resource_cost.is_finite()
            || self.maximum_total_resource_cost < 0.0
            || self.benchmark_artifact_ref.trim().is_empty()
        {
            return Err(BenchmarkProtocolError::InvalidProtocol);
        }
        if self
            .training_budgets
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
        {
            return Err(BenchmarkProtocolError::InvalidTrainingBudgets);
        }

        let trial_count = self
            .training_budgets
            .len()
            .checked_mul(self.seeds.len())
            .and_then(|value| value.checked_mul(2))
            .ok_or(BenchmarkProtocolError::EvaluationCountOverflow)?;
        if trial_count > self.maximum_total_evaluations {
            return Err(BenchmarkProtocolError::EvaluationBudgetExceeded {
                required: trial_count,
                maximum: self.maximum_total_evaluations,
            });
        }

        let mut seeds = BTreeSet::new();
        let mut training_datasets = BTreeSet::new();
        let mut evaluation_datasets = BTreeSet::new();
        for seed in &self.seeds {
            if seed.training_dataset_id.trim().is_empty()
                || seed.evaluation_dataset_id.trim().is_empty()
            {
                return Err(BenchmarkProtocolError::InvalidDatasetIdentity);
            }
            if !seeds.insert(seed.seed) {
                return Err(BenchmarkProtocolError::DuplicateSeed);
            }
            if seed.training_dataset_id == seed.evaluation_dataset_id {
                return Err(BenchmarkProtocolError::DatasetLeakage);
            }
            if !training_datasets.insert(seed.training_dataset_id.as_str()) {
                return Err(BenchmarkProtocolError::DuplicateTrainingDataset);
            }
            if !evaluation_datasets.insert(seed.evaluation_dataset_id.as_str()) {
                return Err(BenchmarkProtocolError::DuplicateEvaluationDataset);
            }
            if training_datasets.contains(seed.evaluation_dataset_id.as_str())
                || evaluation_datasets.contains(seed.training_dataset_id.as_str())
            {
                return Err(BenchmarkProtocolError::DatasetLeakage);
            }
        }
        Ok(())
    }

    pub fn expected_trial_count(&self) -> Result<usize, BenchmarkProtocolError> {
        self.training_budgets
            .len()
            .checked_mul(self.seeds.len())
            .and_then(|value| value.checked_mul(2))
            .ok_or(BenchmarkProtocolError::EvaluationCountOverflow)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkStrategyRole {
    Learner,
    Reference,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkTrialRequest<'a> {
    pub protocol_id: &'a str,
    pub environment_id: &'a str,
    pub environment_version: &'a str,
    pub task_id: &'a str,
    pub strategy_id: &'a str,
    pub role: BenchmarkStrategyRole,
    pub metric_id: &'a str,
    pub direction: ScoreDirection,
    pub seed: u64,
    pub training_samples: usize,
    pub training_dataset_id: &'a str,
    pub evaluation_dataset_id: &'a str,
    pub evaluation_sample_count: usize,
}

/// Evidence returned by one isolated strategy evaluation.
///
/// The executor must echo the requested identity and provenance. The runner
/// checks those declarations, but cannot prove that an executor actually used
/// the declared dataset or random seed.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkTrialResult {
    pub environment_id: String,
    pub task_id: String,
    pub strategy_id: String,
    pub metric_id: String,
    pub direction: ScoreDirection,
    pub seed: u64,
    pub training_samples: usize,
    pub training_dataset_id: String,
    pub evaluation_dataset_id: String,
    pub evaluation_sample_count: usize,
    pub score: f64,
    pub resource_cost: f64,
    /// Required for learner trials; ignored for reference trials.
    pub model_artifact_ref: Option<String>,
}

pub trait BenchmarkProtocolExecutor {
    fn evaluate(
        &mut self,
        request: BenchmarkTrialRequest<'_>,
    ) -> Result<BenchmarkTrialResult, String>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkTrialRecord {
    pub role: BenchmarkStrategyRole,
    pub seed: u64,
    pub training_samples: usize,
    pub strategy_id: String,
    pub score: f64,
    pub resource_cost: f64,
    pub model_artifact_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkProtocolReport {
    pub protocol_id: String,
    pub artifact_ref: String,
    pub trial_count: usize,
    pub total_resource_cost: f64,
    pub benchmark: LearningEfficiencyBenchmark,
    pub summaries: Vec<LearningEfficiencySummary>,
    pub trials: Vec<BenchmarkTrialRecord>,
}

pub fn run_benchmark_protocol<E: BenchmarkProtocolExecutor>(
    protocol: &BenchmarkProtocol,
    executor: &mut E,
    artifact_store: &mut ModelArtifactStore,
) -> Result<BenchmarkProtocolReport, BenchmarkProtocolError> {
    protocol.validate()?;
    if artifact_store.get(&protocol.benchmark_artifact_ref).is_some() {
        return Err(BenchmarkProtocolError::ArtifactReferenceAlreadyExists);
    }

    let mut benchmark = LearningEfficiencyBenchmark::default();
    let mut trials = Vec::with_capacity(protocol.expected_trial_count()?);
    let mut total_resource_cost = 0.0;

    for seed_plan in &protocol.seeds {
        let mut points = Vec::with_capacity(protocol.training_budgets.len());
        let mut learner_artifact_ref = None;
        for &training_samples in &protocol.training_budgets {
            let learner = evaluate_one(
                protocol,
                seed_plan,
                training_samples,
                BenchmarkStrategyRole::Learner,
                &protocol.learner_strategy_id,
                executor,
            )?;
            let reference = evaluate_one(
                protocol,
                seed_plan,
                training_samples,
                BenchmarkStrategyRole::Reference,
                &protocol.reference_strategy_id,
                executor,
            )?;
            total_resource_cost += learner.resource_cost + reference.resource_cost;
            if !total_resource_cost.is_finite() {
                return Err(BenchmarkProtocolError::NumericalOverflow);
            }
            if total_resource_cost > protocol.maximum_total_resource_cost {
                return Err(BenchmarkProtocolError::ResourceBudgetExceeded {
                    observed: total_resource_cost,
                    maximum: protocol.maximum_total_resource_cost,
                });
            }

            if training_samples == *protocol.training_budgets.last().ok_or(
                BenchmarkProtocolError::InvalidTrainingBudgets,
            )? {
                learner_artifact_ref = learner.model_artifact_ref.clone();
            }
            points.push(LearningEfficiencyPoint {
                training_samples,
                learner_score: learner.score,
                reference_score: reference.score,
                learner_resource_cost: learner.resource_cost,
                reference_resource_cost: reference.resource_cost,
            });
            trials.push(BenchmarkTrialRecord {
                role: BenchmarkStrategyRole::Learner,
                seed: seed_plan.seed,
                training_samples,
                strategy_id: protocol.learner_strategy_id.clone(),
                score: learner.score,
                resource_cost: learner.resource_cost,
                model_artifact_ref: learner.model_artifact_ref,
            });
            trials.push(BenchmarkTrialRecord {
                role: BenchmarkStrategyRole::Reference,
                seed: seed_plan.seed,
                training_samples,
                strategy_id: protocol.reference_strategy_id.clone(),
                score: reference.score,
                resource_cost: reference.resource_cost,
                model_artifact_ref: reference.model_artifact_ref,
            });
        }

        let curve = LearningEfficiencyCurve {
            curve_id: format!("{}-seed-{}", protocol.protocol_id, seed_plan.seed),
            environment_id: protocol.environment_id.clone(),
            environment_version: protocol.environment_version.clone(),
            task_id: protocol.task_id.clone(),
            learner_strategy_id: protocol.learner_strategy_id.clone(),
            reference_strategy_id: protocol.reference_strategy_id.clone(),
            learner_artifact_ref: learner_artifact_ref
                .ok_or(BenchmarkProtocolError::MissingLearnerArtifact)?,
            training_dataset_id: seed_plan.training_dataset_id.clone(),
            evaluation_dataset_id: seed_plan.evaluation_dataset_id.clone(),
            metric_id: protocol.metric_id.clone(),
            direction: protocol.direction,
            seed: seed_plan.seed,
            evaluation_sample_count: protocol.evaluation_sample_count,
            points,
        };
        benchmark
            .record(curve)
            .map_err(BenchmarkProtocolError::Benchmark)?;
    }

    let summaries = benchmark
        .summarize(&protocol.metric_id, protocol.target_improvement)
        .map_err(BenchmarkProtocolError::Benchmark)?;
    artifact_store
        .put_learning_efficiency_benchmark(&protocol.benchmark_artifact_ref, &benchmark)
        .map_err(BenchmarkProtocolError::Artifact)?;

    Ok(BenchmarkProtocolReport {
        protocol_id: protocol.protocol_id.clone(),
        artifact_ref: protocol.benchmark_artifact_ref.clone(),
        trial_count: trials.len(),
        total_resource_cost,
        benchmark,
        summaries,
        trials,
    })
}

fn evaluate_one<E: BenchmarkProtocolExecutor>(
    protocol: &BenchmarkProtocol,
    seed_plan: &BenchmarkSeedPlan,
    training_samples: usize,
    role: BenchmarkStrategyRole,
    strategy_id: &str,
    executor: &mut E,
) -> Result<BenchmarkTrialResult, BenchmarkProtocolError> {
    let request = BenchmarkTrialRequest {
        protocol_id: &protocol.protocol_id,
        environment_id: &protocol.environment_id,
        environment_version: &protocol.environment_version,
        task_id: &protocol.task_id,
        strategy_id,
        role,
        metric_id: &protocol.metric_id,
        direction: protocol.direction,
        seed: seed_plan.seed,
        training_samples,
        training_dataset_id: &seed_plan.training_dataset_id,
        evaluation_dataset_id: &seed_plan.evaluation_dataset_id,
        evaluation_sample_count: protocol.evaluation_sample_count,
    };
    let result = executor
        .evaluate(request)
        .map_err(|message| BenchmarkProtocolError::EvaluationFailed {
            strategy_id: strategy_id.to_owned(),
            seed: seed_plan.seed,
            training_samples,
            message,
        })?;
    if result.environment_id != protocol.environment_id
        || result.task_id != protocol.task_id
        || result.strategy_id != strategy_id
        || result.metric_id != protocol.metric_id
        || result.direction != protocol.direction
        || result.seed != seed_plan.seed
        || result.training_samples != training_samples
        || result.training_dataset_id != seed_plan.training_dataset_id
        || result.evaluation_dataset_id != seed_plan.evaluation_dataset_id
        || result.evaluation_sample_count != protocol.evaluation_sample_count
        || !result.score.is_finite()
        || !result.resource_cost.is_finite()
        || result.resource_cost < 0.0
    {
        return Err(BenchmarkProtocolError::ProvenanceMismatch {
            strategy_id: strategy_id.to_owned(),
            seed: seed_plan.seed,
            training_samples,
        });
    }
    if role == BenchmarkStrategyRole::Learner
        && result
            .model_artifact_ref
            .as_deref()
            .map_or(true, |reference| reference.trim().is_empty())
    {
        return Err(BenchmarkProtocolError::MissingLearnerArtifact);
    }
    Ok(result)
}

#[derive(Debug)]
pub enum BenchmarkProtocolError {
    InvalidProtocol,
    InvalidTrainingBudgets,
    EvaluationCountOverflow,
    EvaluationBudgetExceeded { required: usize, maximum: usize },
    ResourceBudgetExceeded { observed: f64, maximum: f64 },
    InvalidDatasetIdentity,
    DuplicateSeed,
    DuplicateTrainingDataset,
    DuplicateEvaluationDataset,
    DatasetLeakage,
    ArtifactReferenceAlreadyExists,
    MissingLearnerArtifact,
    EvaluationFailed {
        strategy_id: String,
        seed: u64,
        training_samples: usize,
        message: String,
    },
    ProvenanceMismatch {
        strategy_id: String,
        seed: u64,
        training_samples: usize,
    },
    NumericalOverflow,
    Benchmark(BenchmarkError),
    Artifact(ModelArtifactError),
}

impl From<BenchmarkError> for BenchmarkProtocolError {
    fn from(error: BenchmarkError) -> Self {
        Self::Benchmark(error)
    }
}

impl std::fmt::Display for BenchmarkProtocolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProtocol => write!(formatter, "benchmark protocol is invalid"),
            Self::InvalidTrainingBudgets => write!(formatter, "training budgets must contain at least two strictly increasing sample counts"),
            Self::EvaluationCountOverflow => write!(formatter, "benchmark evaluation count overflowed"),
            Self::EvaluationBudgetExceeded { required, maximum } => write!(formatter, "benchmark requires {required} evaluations, exceeding the configured limit {maximum}"),
            Self::ResourceBudgetExceeded { observed, maximum } => write!(formatter, "benchmark resource cost {observed} exceeds the configured limit {maximum}"),
            Self::InvalidDatasetIdentity => write!(formatter, "benchmark dataset identity must not be empty"),
            Self::DuplicateSeed => write!(formatter, "benchmark seed is duplicated"),
            Self::DuplicateTrainingDataset => write!(formatter, "training dataset identity is reused across seeds"),
            Self::DuplicateEvaluationDataset => write!(formatter, "evaluation dataset identity is reused across seeds"),
            Self::DatasetLeakage => write!(formatter, "training and evaluation dataset identities overlap"),
            Self::ArtifactReferenceAlreadyExists => write!(formatter, "benchmark artifact reference is already in use"),
            Self::MissingLearnerArtifact => write!(formatter, "learner trial did not return an artifact reference"),
            Self::EvaluationFailed { strategy_id, seed, training_samples, message } => write!(formatter, "benchmark evaluation failed for {strategy_id}, seed {seed}, budget {training_samples}: {message}"),
            Self::ProvenanceMismatch { strategy_id, seed, training_samples } => write!(formatter, "benchmark executor returned incompatible provenance for {strategy_id}, seed {seed}, budget {training_samples}"),
            Self::NumericalOverflow => write!(formatter, "benchmark resource aggregation overflowed"),
            Self::Benchmark(error) => write!(formatter, "learning-efficiency benchmark failed: {error}"),
            Self::Artifact(error) => write!(formatter, "benchmark artifact persistence failed: {error}"),
        }
    }
}

impl std::error::Error for BenchmarkProtocolError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_artifact::ModelArtifactStore;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn path(label: &str) -> std::path::PathBuf {
        let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "5413-benchmark-protocol-{}-{label}-{id}.bin",
            std::process::id()
        ))
    }

    fn protocol() -> BenchmarkProtocol {
        BenchmarkProtocol {
            protocol_id: "active-vs-random-grid".to_owned(),
            environment_id: "grid".to_owned(),
            environment_version: "v3".to_owned(),
            task_id: "reach-goal".to_owned(),
            learner_strategy_id: "active-learning".to_owned(),
            reference_strategy_id: "random-exploration".to_owned(),
            metric_id: "success-rate".to_owned(),
            direction: ScoreDirection::HigherIsBetter,
            evaluation_sample_count: 100,
            training_budgets: vec![0, 10, 20],
            seeds: vec![
                BenchmarkSeedPlan {
                    seed: 7,
                    training_dataset_id: "grid-train-7".to_owned(),
                    evaluation_dataset_id: "grid-holdout-7".to_owned(),
                },
                BenchmarkSeedPlan {
                    seed: 9,
                    training_dataset_id: "grid-train-9".to_owned(),
                    evaluation_dataset_id: "grid-holdout-9".to_owned(),
                },
            ],
            target_improvement: 0.15,
            maximum_total_evaluations: 12,
            maximum_total_resource_cost: 100.0,
            benchmark_artifact_ref: "benchmarks/grid/active-vs-random/v1".to_owned(),
        }
    }

    struct DeterministicExecutor {
        calls: usize,
        mismatch_evaluation: bool,
    }

    impl BenchmarkProtocolExecutor for DeterministicExecutor {
        fn evaluate(
            &mut self,
            request: BenchmarkTrialRequest<'_>,
        ) -> Result<BenchmarkTrialResult, String> {
            self.calls += 1;
            let is_learner = request.role == BenchmarkStrategyRole::Learner;
            let score = if is_learner {
                0.2 + request.training_samples as f64 / 100.0
            } else {
                0.1
            };
            Ok(BenchmarkTrialResult {
                environment_id: request.environment_id.to_owned(),
                task_id: request.task_id.to_owned(),
                strategy_id: request.strategy_id.to_owned(),
                metric_id: request.metric_id.to_owned(),
                direction: request.direction,
                seed: request.seed,
                training_samples: request.training_samples,
                training_dataset_id: request.training_dataset_id.to_owned(),
                evaluation_dataset_id: if self.mismatch_evaluation {
                    "wrong-holdout".to_owned()
                } else {
                    request.evaluation_dataset_id.to_owned()
                },
                evaluation_sample_count: request.evaluation_sample_count,
                score,
                resource_cost: 1.0,
                model_artifact_ref: is_learner.then(|| {
                    format!(
                        "models/{}-seed-{}-samples-{}",
                        request.strategy_id, request.seed, request.training_samples
                    )
                }),
            })
        }
    }

    #[test]
    fn executes_paired_multiseed_protocol_and_persists_aggregate_artifact() {
        let path = path("store");
        let mut store = ModelArtifactStore::open(&path).expect("artifact store");
        let protocol = protocol();
        let mut executor = DeterministicExecutor {
            calls: 0,
            mismatch_evaluation: false,
        };
        let report = run_benchmark_protocol(&protocol, &mut executor, &mut store)
            .expect("protocol");
        assert_eq!(executor.calls, 12);
        assert_eq!(report.trial_count, 12);
        assert_eq!(report.benchmark.len(), 2);
        assert_eq!(report.summaries.len(), 1);
        assert_eq!(report.summaries[0].seed_count, 2);
        assert_eq!(report.summaries[0].sample_budgets, vec![0, 10, 20]);
        assert_eq!(report.total_resource_cost, 12.0);
        let restored = store
            .load_learning_efficiency_benchmark(&protocol.benchmark_artifact_ref)
            .expect("restore");
        assert_eq!(restored, report.benchmark);
        drop(store);
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn rejects_protocol_budget_and_dataset_reuse_before_invoking_executor() {
        let mut invalid = protocol();
        invalid.maximum_total_evaluations = 11;
        let path = path("budget");
        let mut store = ModelArtifactStore::open(&path).expect("store");
        let mut executor = DeterministicExecutor {
            calls: 0,
            mismatch_evaluation: false,
        };
        assert!(matches!(
            run_benchmark_protocol(&invalid, &mut executor, &mut store),
            Err(BenchmarkProtocolError::EvaluationBudgetExceeded { .. })
        ));
        assert_eq!(executor.calls, 0);

        let mut invalid = protocol();
        invalid.seeds[1].training_dataset_id = invalid.seeds[0].training_dataset_id.clone();
        assert!(matches!(
            run_benchmark_protocol(&invalid, &mut executor, &mut store),
            Err(BenchmarkProtocolError::DuplicateTrainingDataset)
        ));
        assert_eq!(executor.calls, 0);
        drop(store);
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn rejects_executor_provenance_mismatch_without_persisting_partial_benchmark() {
        let path = path("provenance");
        let mut store = ModelArtifactStore::open(&path).expect("store");
        let protocol = protocol();
        let mut executor = DeterministicExecutor {
            calls: 0,
            mismatch_evaluation: true,
        };
        assert!(matches!(
            run_benchmark_protocol(&protocol, &mut executor, &mut store),
            Err(BenchmarkProtocolError::ProvenanceMismatch { .. })
        ));
        assert_eq!(executor.calls, 1);
        assert!(store.get(&protocol.benchmark_artifact_ref).is_none());
        drop(store);
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn enforces_total_resource_cost_as_trials_complete() {
        let path = path("resource");
        let mut store = ModelArtifactStore::open(&path).expect("store");
        let mut protocol = protocol();
        protocol.maximum_total_resource_cost = 2.0;
        let mut executor = DeterministicExecutor {
            calls: 0,
            mismatch_evaluation: false,
        };
        assert!(matches!(
            run_benchmark_protocol(&protocol, &mut executor, &mut store),
            Err(BenchmarkProtocolError::ResourceBudgetExceeded { .. })
        ));
        assert!(store.get(&protocol.benchmark_artifact_ref).is_none());
        drop(store);
        std::fs::remove_file(path).expect("cleanup");
    }
}
