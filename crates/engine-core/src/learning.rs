use crate::experience::{ExperienceEvent, ExperienceRecord};
use crate::value::DataValue;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct TransitionSample {
    pub action_id: String,
    pub before: DataValue,
    pub after: DataValue,
}

#[derive(Clone, Debug)]
struct PendingTransition {
    action_id: String,
    before: DataValue,
    completed: bool,
    observed_after: Option<DataValue>,
}

/// Reconstructs completed state transitions from one episode's ordered journal
/// records. For a completed action, the last subsequent valid observation before
/// the next proposal is used as its observed next state. Incomplete or mismatched
/// action outcomes are excluded.
pub fn extract_transition_samples(
    records: &[ExperienceRecord],
    episode_id: &str,
) -> Vec<TransitionSample> {
    if episode_id.trim().is_empty() {
        return Vec::new();
    }

    let mut current_state: Option<DataValue> = None;
    let mut pending: Option<PendingTransition> = None;
    let mut samples = Vec::new();

    for record in records
        .iter()
        .filter(|record| record.episode_id == episode_id)
    {
        match &record.event {
            ExperienceEvent::Observation(observation) => {
                if observation.payload.validate_finite().is_err() {
                    current_state = None;
                    if let Some(active) = pending.as_mut() {
                        active.observed_after = None;
                    }
                    continue;
                }
                if let Some(active) = pending.as_mut() {
                    if active.completed {
                        active.observed_after = Some(observation.payload.clone());
                    }
                }
                current_state = Some(observation.payload.clone());
            }
            ExperienceEvent::ActionProposed { action_id, .. } => {
                if let Some(previous) = pending.take() {
                    append_completed_transition(previous, &mut samples);
                }
                pending = current_state.clone().map(|before| PendingTransition {
                    action_id: action_id.clone(),
                    before,
                    completed: false,
                    observed_after: None,
                });
            }
            ExperienceEvent::ActionCompleted(outcome) => {
                if let Some(active) = pending.as_mut() {
                    active.completed = outcome.completed && outcome.action_id == active.action_id;
                    active.observed_after = None;
                }
            }
            ExperienceEvent::PredictionEvaluated { .. } => {}
            ExperienceEvent::EnvironmentResetStarted { .. }
            | ExperienceEvent::EnvironmentResetCompleted { .. } => {
                if let Some(previous) = pending.take() {
                    append_completed_transition(previous, &mut samples);
                }
                current_state = None;
            }
        }
    }

    if let Some(last) = pending {
        append_completed_transition(last, &mut samples);
    }
    samples
}

fn append_completed_transition(pending: PendingTransition, samples: &mut Vec<TransitionSample>) {
    if !pending.completed {
        return;
    }
    let Some(after) = pending.observed_after else {
        return;
    };
    samples.push(TransitionSample {
        action_id: pending.action_id,
        before: pending.before,
        after,
    });
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldPrediction {
    pub predicted_value: f64,
    pub mean_delta: f64,
    pub delta_standard_deviation: f64,
    pub sample_count: u64,
}

#[derive(Clone, Debug, Default)]
struct DeltaStatistics {
    count: u64,
    scale: f64,
    normalized_mean: f64,
    normalized_sum_squared_deviations: f64,
}

impl DeltaStatistics {
    fn updated(&self, value: f64, field: &str) -> Result<Self, LearningError> {
        if !value.is_finite() {
            return Err(LearningError::NonFiniteDelta(field.to_owned()));
        }

        let count = self
            .count
            .checked_add(1)
            .ok_or(LearningError::SampleCountExhausted)?;
        let scale = self.scale.max(value.abs());
        let previous_scale_ratio = if scale == 0.0 {
            0.0
        } else {
            self.scale / scale
        };
        let previous_mean = self.normalized_mean * previous_scale_ratio;
        let previous_sum_squared_deviations = self.normalized_sum_squared_deviations
            * previous_scale_ratio
            * previous_scale_ratio;
        let normalized_value = if scale == 0.0 { 0.0 } else { value / scale };

        // All values are normalized by the largest magnitude seen so far. The
        // Welford update therefore avoids both overflowing deltas and raw squares.
        let delta = normalized_value - previous_mean;
        let normalized_mean = previous_mean + delta / count as f64;
        let delta_after_update = normalized_value - normalized_mean;
        let normalized_sum_squared_deviations = previous_sum_squared_deviations
            + delta * delta_after_update;

        if !scale.is_finite()
            || !normalized_mean.is_finite()
            || !normalized_sum_squared_deviations.is_finite()
        {
            return Err(LearningError::NonFiniteDelta(field.to_owned()));
        }

        Ok(Self {
            count,
            scale,
            normalized_mean,
            normalized_sum_squared_deviations: normalized_sum_squared_deviations.max(0.0),
        })
    }

    fn mean(&self) -> f64 {
        self.scale * self.normalized_mean
    }

    fn standard_deviation(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.scale
                * (self.normalized_sum_squared_deviations / (self.count - 1) as f64).sqrt()
        }
    }
}

/// A deliberately small baseline model: estimates per-action mean deltas
/// for numeric fields shared by consecutive object-shaped observations.
///
/// This is a generic starting point, not a general-purpose world model.
#[derive(Clone, Debug, Default)]
pub struct NumericDeltaModel {
    statistics: BTreeMap<(String, String), DeltaStatistics>,
    transitions_seen: u64,
}

impl NumericDeltaModel {
    pub fn observe_transition(
        &mut self,
        sample: &TransitionSample,
    ) -> Result<usize, LearningError> {
        sample
            .before
            .validate_finite()
            .map_err(|_| LearningError::NonFiniteObservation)?;
        sample
            .after
            .validate_finite()
            .map_err(|_| LearningError::NonFiniteObservation)?;

        let before = sample
            .before
            .as_object()
            .ok_or(LearningError::ExpectedObjectState)?;
        let after = sample
            .after
            .as_object()
            .ok_or(LearningError::ExpectedObjectState)?;

        let mut updates = Vec::new();
        for (field, before_value) in before {
            let Some(after_value) = after.get(field) else {
                continue;
            };
            let (Some(before_number), Some(after_number)) =
                (before_value.as_f64(), after_value.as_f64())
            else {
                continue;
            };

            let delta = after_number - before_number;
            let key = (sample.action_id.clone(), field.clone());
            let current = self.statistics.get(&key).cloned().unwrap_or_default();
            updates.push((key, current.updated(delta, field)?));
        }

        let next_transition_count = self
            .transitions_seen
            .checked_add(1)
            .ok_or(LearningError::SampleCountExhausted)?;

        for (key, statistics) in &updates {
            self.statistics.insert(key.clone(), statistics.clone());
        }
        self.transitions_seen = next_transition_count;

        Ok(updates.len())
    }

    pub fn predict_next(
        &self,
        action_id: &str,
        current_state: &DataValue,
    ) -> Result<BTreeMap<String, FieldPrediction>, LearningError> {
        current_state
            .validate_finite()
            .map_err(|_| LearningError::NonFiniteObservation)?;
        let current = current_state
            .as_object()
            .ok_or(LearningError::ExpectedObjectState)?;

        let mut predictions = BTreeMap::new();
        for (field, value) in current {
            let Some(current_number) = value.as_f64() else {
                continue;
            };
            let Some(statistics) = self.statistics.get(&(action_id.to_owned(), field.clone()))
            else {
                continue;
            };

            let mean_delta = statistics.mean();
            let predicted_value = current_number + mean_delta;
            let deviation = statistics.standard_deviation();
            if !mean_delta.is_finite() || !predicted_value.is_finite() || !deviation.is_finite() {
                return Err(LearningError::NonFinitePrediction(field.clone()));
            }

            predictions.insert(
                field.clone(),
                FieldPrediction {
                    predicted_value,
                    mean_delta,
                    delta_standard_deviation: deviation,
                    sample_count: statistics.count,
                },
            );
        }
        Ok(predictions)
    }

    pub fn transitions_seen(&self) -> u64 {
        self.transitions_seen
    }

    /// Exports versioned sufficient statistics in a deterministic data form.
    pub fn to_artifact_value(&self) -> DataValue {
        let statistics = self
            .statistics
            .iter()
            .map(|((action_id, field), stats)| {
                DataValue::object([
                    ("action_id".to_owned(), DataValue::Text(action_id.clone())),
                    ("field".to_owned(), DataValue::Text(field.clone())),
                    ("count".to_owned(), DataValue::Text(stats.count.to_string())),
                    ("scale".to_owned(), DataValue::Number(stats.scale)),
                    (
                        "normalized_mean".to_owned(),
                        DataValue::Number(stats.normalized_mean),
                    ),
                    (
                        "normalized_sum_squared_deviations".to_owned(),
                        DataValue::Number(stats.normalized_sum_squared_deviations),
                    ),
                ])
            })
            .collect();
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(2)),
            (
                "transitions_seen".to_owned(),
                DataValue::Text(self.transitions_seen.to_string()),
            ),
            ("statistics".to_owned(), DataValue::Sequence(statistics)),
        ])
    }

    /// Restores current artifacts and migrates the original unversioned format.
    pub fn from_artifact_value(value: &DataValue) -> Result<Self, LearningError> {
        let fields = value
            .as_object()
            .ok_or(LearningError::InvalidSerializedState)?;
        let legacy_format = match fields.get("artifact_version") {
            None if fields.len() == 2 => true,
            Some(DataValue::Integer(2)) if fields.len() == 3 => false,
            _ => return Err(LearningError::InvalidSerializedState),
        };
        let transitions_seen = match fields.get("transitions_seen") {
            Some(DataValue::Text(value)) => value
                .parse::<u64>()
                .map_err(|_| LearningError::InvalidSerializedState)?,
            _ => return Err(LearningError::InvalidSerializedState),
        };
        let values = match fields.get("statistics") {
            Some(DataValue::Sequence(values)) => values,
            _ => return Err(LearningError::InvalidSerializedState),
        };

        let mut statistics = BTreeMap::new();
        for entry in values {
            let entry = entry
                .as_object()
                .ok_or(LearningError::InvalidSerializedState)?;
            let action_id = match entry.get("action_id") {
                Some(DataValue::Text(value)) if !value.trim().is_empty() => value.clone(),
                _ => return Err(LearningError::InvalidSerializedState),
            };
            let field = match entry.get("field") {
                Some(DataValue::Text(value)) if !value.trim().is_empty() => value.clone(),
                _ => return Err(LearningError::InvalidSerializedState),
            };
            let count = match entry.get("count") {
                Some(DataValue::Text(value)) => value
                    .parse::<u64>()
                    .map_err(|_| LearningError::InvalidSerializedState)?,
                _ => return Err(LearningError::InvalidSerializedState),
            };
            if count == 0 || count > transitions_seen {
                return Err(LearningError::InvalidSerializedState);
            }

            let restored = if legacy_format {
                if entry.len() != 5 {
                    return Err(LearningError::InvalidSerializedState);
                }
                let mean = entry
                    .get("mean")
                    .and_then(DataValue::as_f64)
                    .filter(|value| value.is_finite())
                    .ok_or(LearningError::InvalidSerializedState)?;
                let sum_squared_deviations = entry
                    .get("sum_squared_deviations")
                    .and_then(DataValue::as_f64)
                    .filter(|value| value.is_finite() && *value >= 0.0)
                    .ok_or(LearningError::InvalidSerializedState)?;
                let root_sum_squared_deviations = sum_squared_deviations.sqrt();
                let magnitude = mean.abs().max(root_sum_squared_deviations);
                // Leave headroom so all historical observations remain bounded
                // after migration: |x| <= |mean| + sqrt(M2).
                let scale = if magnitude == 0.0 {
                    0.0
                } else if magnitude > f64::MAX / 2.0 {
                    f64::MAX
                } else {
                    magnitude * 2.0
                };
                let normalized_mean = if scale == 0.0 { 0.0 } else { mean / scale };
                let normalized_root = if scale == 0.0 {
                    0.0
                } else {
                    root_sum_squared_deviations / scale
                };
                DeltaStatistics {
                    count,
                    scale,
                    normalized_mean,
                    normalized_sum_squared_deviations: normalized_root * normalized_root,
                }
            } else {
                if entry.len() != 6 {
                    return Err(LearningError::InvalidSerializedState);
                }
                let scale = entry
                    .get("scale")
                    .and_then(DataValue::as_f64)
                    .filter(|value| value.is_finite() && *value >= 0.0)
                    .ok_or(LearningError::InvalidSerializedState)?;
                let normalized_mean = entry
                    .get("normalized_mean")
                    .and_then(DataValue::as_f64)
                    .filter(|value| value.is_finite() && value.abs() <= 1.0 + 1.0e-12)
                    .ok_or(LearningError::InvalidSerializedState)?;
                let normalized_sum_squared_deviations = entry
                    .get("normalized_sum_squared_deviations")
                    .and_then(DataValue::as_f64)
                    .filter(|value| value.is_finite() && *value >= 0.0)
                    .ok_or(LearningError::InvalidSerializedState)?;
                if scale == 0.0
                    && (normalized_mean != 0.0 || normalized_sum_squared_deviations != 0.0)
                {
                    return Err(LearningError::InvalidSerializedState);
                }
                DeltaStatistics {
                    count,
                    scale,
                    normalized_mean,
                    normalized_sum_squared_deviations,
                }
            };

            if statistics
                .insert((action_id, field), restored)
                .is_some()
            {
                return Err(LearningError::InvalidSerializedState);
            }
        }

        Ok(Self {
            statistics,
            transitions_seen,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LearningError {
    ExpectedObjectState,
    NonFiniteObservation,
    NonFiniteDelta(String),
    NonFinitePrediction(String),
    SampleCountExhausted,
    InvalidSerializedState,
}

impl std::fmt::Display for LearningError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExpectedObjectState => write!(formatter, "transition state must be an object"),
            Self::NonFiniteObservation => {
                write!(formatter, "observation contains a non-finite number")
            }
            Self::NonFiniteDelta(field) => {
                write!(formatter, "non-finite transition delta for field {field}")
            }
            Self::NonFinitePrediction(field) => {
                write!(formatter, "non-finite prediction for field {field}")
            }
            Self::SampleCountExhausted => write!(formatter, "learning sample count is exhausted"),
            Self::InvalidSerializedState => {
                write!(formatter, "serialized learner state is invalid")
            }
        }
    }
}

impl std::error::Error for LearningError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn state(position: f64, label: &str) -> DataValue {
        DataValue::Object(BTreeMap::from([
            ("position".to_owned(), DataValue::Number(position)),
            ("label".to_owned(), DataValue::Text(label.to_owned())),
        ]))
    }

    #[test]
    fn learns_action_conditioned_numeric_deltas() {
        let mut model = NumericDeltaModel::default();
        model
            .observe_transition(&TransitionSample {
                action_id: "advance".to_owned(),
                before: state(1.0, "a"),
                after: state(3.0, "b"),
            })
            .expect("first sample");
        model
            .observe_transition(&TransitionSample {
                action_id: "advance".to_owned(),
                before: state(3.0, "b"),
                after: state(5.0, "c"),
            })
            .expect("second sample");

        let prediction = model
            .predict_next("advance", &state(5.0, "d"))
            .expect("prediction");
        let position = prediction.get("position").expect("position prediction");

        assert_eq!(position.predicted_value, 7.0);
        assert_eq!(position.sample_count, 2);
        assert_eq!(position.delta_standard_deviation, 0.0);
    }

    #[test]
    fn keeps_action_models_separate() {
        let mut model = NumericDeltaModel::default();
        model
            .observe_transition(&TransitionSample {
                action_id: "increase".to_owned(),
                before: state(0.0, "a"),
                after: state(2.0, "b"),
            })
            .expect("sample");

        let prediction = model
            .predict_next("decrease", &state(2.0, "c"))
            .expect("prediction");
        assert!(!prediction.contains_key("position"));
    }

    #[test]
    fn learns_extreme_finite_deltas_without_overflowing_variance() {
        let magnitude = f64::MAX / 2.0;
        let mut model = NumericDeltaModel::default();
        for delta in [magnitude, -magnitude] {
            model
                .observe_transition(&TransitionSample {
                    action_id: "advance".to_owned(),
                    before: state(0.0, "before"),
                    after: state(delta, "after"),
                })
                .expect("finite delta should update normalized statistics");
        }

        let prediction = model
            .predict_next("advance", &state(0.0, "current"))
            .expect("representable mean and standard deviation");
        let estimate = prediction.get("position").expect("position prediction");
        assert!(estimate.mean_delta.abs() <= f64::EPSILON * magnitude);
        assert!((estimate.delta_standard_deviation / f64::MAX
            - std::f64::consts::FRAC_1_SQRT_2)
            .abs()
            < 1.0e-12);

        let restored = NumericDeltaModel::from_artifact_value(&model.to_artifact_value())
            .expect("versioned learner artifact restores");
        assert_eq!(
            restored
                .predict_next("advance", &state(0.0, "current"))
                .expect("restored prediction"),
            prediction
        );
    }

    #[test]
    fn migrates_legacy_numeric_delta_artifacts() {
        let legacy = DataValue::object([
            (
                "transitions_seen".to_owned(),
                DataValue::Text("2".to_owned()),
            ),
            (
                "statistics".to_owned(),
                DataValue::Sequence(vec![DataValue::object([
                    ("action_id".to_owned(), DataValue::Text("advance".to_owned())),
                    ("field".to_owned(), DataValue::Text("position".to_owned())),
                    ("count".to_owned(), DataValue::Text("2".to_owned())),
                    ("mean".to_owned(), DataValue::Number(0.0)),
                    (
                        "sum_squared_deviations".to_owned(),
                        DataValue::Number(2.0),
                    ),
                ])]),
            ),
        ]);
        let mut model =
            NumericDeltaModel::from_artifact_value(&legacy).expect("legacy artifact migrates");
        model
            .observe_transition(&TransitionSample {
                action_id: "advance".to_owned(),
                before: state(0.0, "before"),
                after: state(4.0, "after"),
            })
            .expect("migrated model remains trainable");

        let prediction = model
            .predict_next("advance", &state(0.0, "current"))
            .expect("prediction after migration");
        let estimate = prediction.get("position").expect("position prediction");
        assert!((estimate.mean_delta - (4.0 / 3.0)).abs() < 1.0e-12);
        assert!((estimate.delta_standard_deviation - (19.0_f64 / 3.0).sqrt()).abs() < 1.0e-12);
    }
}

#[cfg(test)]
mod transition_extraction_tests {
    use super::*;
    use crate::environment::{ActionOutcome, ObservationFrame};
    use crate::value::DataValue;

    fn observation(sequence: u64, position: f64) -> ExperienceEvent {
        ExperienceEvent::Observation(ObservationFrame {
            environment_id: "counter".to_owned(),
            session_id: "session".to_owned(),
            episode_id: "episode-a".to_owned(),
            sequence,
            observed_at_millis: sequence,
            schema_id: "state".to_owned(),
            schema_version: 1,
            payload: DataValue::object([("position".to_owned(), DataValue::Number(position))]),
        })
    }

    fn proposal(action_id: &str) -> ExperienceEvent {
        ExperienceEvent::ActionProposed {
            action_id: action_id.to_owned(),
            arguments: DataValue::Null,
        }
    }

    fn outcome(action_id: &str, completed: bool) -> ExperienceEvent {
        ExperienceEvent::ActionCompleted(ActionOutcome {
            action_id: action_id.to_owned(),
            completed,
            observations: Vec::new(),
            feedback: DataValue::Null,
        })
    }

    fn record(sequence: u64, event: ExperienceEvent, episode_id: &str) -> ExperienceRecord {
        ExperienceRecord {
            sequence,
            episode_id: episode_id.to_owned(),
            event,
        }
    }

    #[test]
    fn extracts_only_completed_episode_scoped_transitions() {
        let records = vec![
            record(0, observation(0, 0.0), "episode-a"),
            record(1, proposal("advance"), "episode-a"),
            record(2, outcome("advance", true), "episode-a"),
            record(3, observation(1, 1.0), "episode-a"),
            record(4, proposal("advance"), "episode-a"),
            record(5, outcome("advance", false), "episode-a"),
            record(6, observation(2, 1.0), "episode-a"),
            record(7, observation(0, 50.0), "episode-b"),
        ];

        let samples = extract_transition_samples(&records, "episode-a");
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].action_id, "advance");
        assert_eq!(
            samples[0]
                .before
                .get("position")
                .and_then(DataValue::as_f64),
            Some(0.0)
        );
        assert_eq!(
            samples[0].after.get("position").and_then(DataValue::as_f64),
            Some(1.0)
        );
        assert!(extract_transition_samples(&records, "").is_empty());
    }

    #[test]
    fn uses_latest_observation_after_completed_action() {
        let records = vec![
            record(0, observation(0, 0.0), "episode-a"),
            record(1, proposal("advance"), "episode-a"),
            record(2, outcome("advance", true), "episode-a"),
            record(3, observation(1, 0.5), "episode-a"),
            record(4, observation(2, 1.0), "episode-a"),
        ];
        let samples = extract_transition_samples(&records, "episode-a");
        assert_eq!(samples.len(), 1);
        assert_eq!(
            samples[0].after.get("position").and_then(DataValue::as_f64),
            Some(1.0)
        );
    }

    #[test]
    fn ignores_mismatched_action_outcomes() {
        let records = vec![
            record(0, observation(0, 0.0), "episode-a"),
            record(1, proposal("advance"), "episode-a"),
            record(2, outcome("different-action", true), "episode-a"),
            record(3, observation(1, 1.0), "episode-a"),
        ];
        assert!(extract_transition_samples(&records, "episode-a").is_empty());
    }
}
