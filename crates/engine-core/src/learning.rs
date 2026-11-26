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
            ExperienceEvent::ActionProposed(proposal) => {
                if let Some(previous) = pending.take() {
                    append_completed_transition(previous, &mut samples);
                }
                pending = current_state.clone().map(|before| PendingTransition {
                    action_id: proposal.action_id.clone(),
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
    mean: f64,
    sum_squared_deviations: f64,
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
        let delta = value - self.mean;
        let mean = self.mean + delta / count as f64;
        let delta_after_update = value - mean;
        let sum_squared_deviations = self.sum_squared_deviations + delta * delta_after_update;

        if !mean.is_finite() || !sum_squared_deviations.is_finite() {
            return Err(LearningError::NonFiniteDelta(field.to_owned()));
        }

        Ok(Self {
            count,
            mean,
            sum_squared_deviations,
        })
    }

    fn standard_deviation(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            (self.sum_squared_deviations / (self.count - 1) as f64)
                .max(0.0)
                .sqrt()
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

            let predicted_value = current_number + statistics.mean;
            let deviation = statistics.standard_deviation();
            if !predicted_value.is_finite() || !deviation.is_finite() {
                return Err(LearningError::NonFinitePrediction(field.clone()));
            }

            predictions.insert(
                field.clone(),
                FieldPrediction {
                    predicted_value,
                    mean_delta: statistics.mean,
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

    /// Exports sufficient statistics in a deterministic, versionable data form.
    pub fn to_artifact_value(&self) -> DataValue {
        let statistics = self
            .statistics
            .iter()
            .map(|((action_id, field), stats)| {
                DataValue::object([
                    ("action_id".to_owned(), DataValue::Text(action_id.clone())),
                    ("field".to_owned(), DataValue::Text(field.clone())),
                    ("count".to_owned(), DataValue::Text(stats.count.to_string())),
                    ("mean".to_owned(), DataValue::Number(stats.mean)),
                    (
                        "sum_squared_deviations".to_owned(),
                        DataValue::Number(stats.sum_squared_deviations),
                    ),
                ])
            })
            .collect();
        DataValue::object([
            (
                "transitions_seen".to_owned(),
                DataValue::Text(self.transitions_seen.to_string()),
            ),
            ("statistics".to_owned(), DataValue::Sequence(statistics)),
        ])
    }

    /// Restores sufficient statistics after strict shape, range, and finiteness checks.
    pub fn from_artifact_value(value: &DataValue) -> Result<Self, LearningError> {
        let fields = value
            .as_object()
            .ok_or(LearningError::InvalidSerializedState)?;
        if fields.len() != 2 {
            return Err(LearningError::InvalidSerializedState);
        }
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
            if entry.len() != 5 {
                return Err(LearningError::InvalidSerializedState);
            }
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
            let mean = entry
                .get("mean")
                .and_then(DataValue::as_f64)
                .ok_or(LearningError::InvalidSerializedState)?;
            let sum_squared_deviations = entry
                .get("sum_squared_deviations")
                .and_then(DataValue::as_f64)
                .ok_or(LearningError::InvalidSerializedState)?;
            if count == 0
                || count > transitions_seen
                || !mean.is_finite()
                || !sum_squared_deviations.is_finite()
                || sum_squared_deviations < 0.0
            {
                return Err(LearningError::InvalidSerializedState);
            }
            if statistics
                .insert(
                    (action_id, field),
                    DeltaStatistics {
                        count,
                        mean,
                        sum_squared_deviations,
                    },
                )
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
}

#[cfg(test)]
mod transition_extraction_tests {
    use super::*;
    use crate::action::ActionProposal;
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
        ExperienceEvent::ActionProposed(ActionProposal {
            action_id: action_id.to_owned(),
            arguments: DataValue::Null,
        })
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
