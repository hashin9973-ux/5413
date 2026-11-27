use crate::experience::{ExperienceEvent, ExperienceRecord};
use crate::value::DataValue;
use std::collections::{BTreeMap, VecDeque};

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
        let previous_sum_squared_deviations =
            self.normalized_sum_squared_deviations * previous_scale_ratio * previous_scale_ratio;
        let normalized_value = if scale == 0.0 { 0.0 } else { value / scale };

        // All values are normalized by the largest magnitude seen so far. The
        // Welford update therefore avoids both overflowing deltas and raw squares.
        let delta = normalized_value - previous_mean;
        let normalized_mean = previous_mean + delta / count as f64;
        let delta_after_update = normalized_value - normalized_mean;
        let normalized_sum_squared_deviations =
            previous_sum_squared_deviations + delta * delta_after_update;

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
            self.scale * (self.normalized_sum_squared_deviations / (self.count - 1) as f64).sqrt()
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

            if statistics.insert((action_id, field), restored).is_some() {
                return Err(LearningError::InvalidSerializedState);
            }
        }

        Ok(Self {
            statistics,
            transitions_seen,
        })
    }
}

/// Online action-delta learner that retains only the most recent transitions.
///
/// A bounded recent window adapts when an environment's dynamics drift or an
/// adversary changes strategy. Updates are transactional: invalid samples leave
/// the retained history, predictions, and lifetime transition count unchanged.
#[derive(Clone, Debug)]
pub struct WindowedNumericDeltaModel {
    window_size: usize,
    history: VecDeque<TransitionSample>,
    model: NumericDeltaModel,
    transitions_seen: u64,
}

impl WindowedNumericDeltaModel {
    pub fn new(window_size: usize) -> Result<Self, LearningError> {
        if window_size == 0 {
            return Err(LearningError::InvalidWindowSize);
        }
        Ok(Self {
            window_size,
            history: VecDeque::new(),
            model: NumericDeltaModel::default(),
            transitions_seen: 0,
        })
    }

    /// Adds one transition and rebuilds sufficient statistics from the retained
    /// window. The lifetime counter is independent of the window's sample count.
    pub fn observe_transition(
        &mut self,
        sample: &TransitionSample,
    ) -> Result<usize, LearningError> {
        let next_transitions_seen = self
            .transitions_seen
            .checked_add(1)
            .ok_or(LearningError::SampleCountExhausted)?;
        let mut history = self.history.clone();
        history.push_back(sample.clone());
        while history.len() > self.window_size {
            history.pop_front();
        }

        let mut model = NumericDeltaModel::default();
        let mut updated_fields = 0;
        for retained in &history {
            updated_fields = model.observe_transition(retained)?;
        }

        self.history = history;
        self.model = model;
        self.transitions_seen = next_transitions_seen;
        Ok(updated_fields)
    }

    pub fn predict_next(
        &self,
        action_id: &str,
        current_state: &DataValue,
    ) -> Result<BTreeMap<String, FieldPrediction>, LearningError> {
        self.model.predict_next(action_id, current_state)
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn retained_transitions(&self) -> usize {
        self.history.len()
    }

    pub fn transitions_seen(&self) -> u64 {
        self.transitions_seen
    }

    fn resize_window(&mut self, window_size: usize) -> Result<(), LearningError> {
        if window_size == 0 {
            return Err(LearningError::InvalidWindowSize);
        }
        let mut history = self.history.clone();
        while history.len() > window_size {
            history.pop_front();
        }
        let mut model = NumericDeltaModel::default();
        for retained in &history {
            model.observe_transition(retained)?;
        }
        self.window_size = window_size;
        self.history = history;
        self.model = model;
        Ok(())
    }
}

/// Data-driven policy for detecting prediction-error shifts and adapting memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveLearningPolicy {
    pub min_window_size: usize,
    pub max_window_size: usize,
    pub stable_observations_to_expand: usize,
    pub minimum_model_samples: u64,
    pub minimum_residual_samples: u64,
    pub consecutive_outliers: usize,
    pub change_threshold_sigma: f64,
}

impl AdaptiveLearningPolicy {
    fn validate(self) -> Result<Self, LearningError> {
        if self.min_window_size == 0
            || self.max_window_size < self.min_window_size
            || self.stable_observations_to_expand == 0
            || self.minimum_model_samples == 0
            || self.minimum_residual_samples < 2
            || self.consecutive_outliers == 0
            || !self.change_threshold_sigma.is_finite()
            || self.change_threshold_sigma <= 0.0
        {
            return Err(LearningError::InvalidAdaptivePolicy);
        }
        Ok(self)
    }
}

/// Diagnostics returned after one adaptive learner update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveLearningUpdate {
    pub change_detected: bool,
    pub window_size: usize,
    pub prediction_errors_observed: usize,
    pub change_points_detected: u64,
}

#[derive(Clone, Debug, Default)]
struct ResidualMonitor {
    statistics: DeltaStatistics,
    consecutive_outliers: usize,
}

impl ResidualMonitor {
    fn observe(
        &mut self,
        residual: f64,
        policy: AdaptiveLearningPolicy,
    ) -> Result<bool, LearningError> {
        if !residual.is_finite() {
            return Err(LearningError::NonFinitePrediction(
                "prediction residual".to_owned(),
            ));
        }

        let mut is_outlier = false;
        if self.statistics.count >= policy.minimum_residual_samples {
            // Compare in normalized coordinates so even extreme finite residuals
            // do not overflow while computing a standard deviation or z-score.
            let scale = self.statistics.scale.max(residual.abs()).max(1.0);
            let previous_scale_ratio = self.statistics.scale / scale;
            let normalized_center = self.statistics.normalized_mean * previous_scale_ratio;
            let normalized_residual = residual / scale - normalized_center;
            let normalized_spread = (self.statistics.normalized_sum_squared_deviations
                / (self.statistics.count - 1) as f64)
                .sqrt()
                * previous_scale_ratio;
            let threshold = policy.change_threshold_sigma * normalized_spread;
            is_outlier = if normalized_spread == 0.0 {
                normalized_residual.abs() > 8.0 * f64::EPSILON
            } else {
                normalized_residual.abs() > threshold
            };
        }

        if is_outlier {
            self.consecutive_outliers = self.consecutive_outliers.saturating_add(1);
            return Ok(self.consecutive_outliers >= policy.consecutive_outliers);
        }

        self.consecutive_outliers = 0;
        self.statistics = self.statistics.updated(residual, "prediction residual")?;
        Ok(false)
    }
}

/// Action-conditioned online learner with change detection and adaptive memory.
///
/// Prediction residuals are monitored per action/field. A confirmed shift shrinks
/// memory to the configured minimum; stable evidence then expands it gradually.
/// Updates are transactional, including detector state and memory resizing.
#[derive(Clone, Debug)]
pub struct AdaptiveNumericDeltaModel {
    policy: AdaptiveLearningPolicy,
    model: WindowedNumericDeltaModel,
    residual_monitors: BTreeMap<(String, String), ResidualMonitor>,
    stable_observations: usize,
    change_points_detected: u64,
}

impl AdaptiveNumericDeltaModel {
    pub fn new(policy: AdaptiveLearningPolicy) -> Result<Self, LearningError> {
        let policy = policy.validate()?;
        Ok(Self {
            model: WindowedNumericDeltaModel::new(policy.min_window_size)?,
            policy,
            residual_monitors: BTreeMap::new(),
            stable_observations: 0,
            change_points_detected: 0,
        })
    }

    pub fn observe_transition(
        &mut self,
        sample: &TransitionSample,
    ) -> Result<AdaptiveLearningUpdate, LearningError> {
        let mut candidate = self.clone();
        let update = candidate.observe_transition_inner(sample)?;
        *self = candidate;
        Ok(update)
    }

    fn observe_transition_inner(
        &mut self,
        sample: &TransitionSample,
    ) -> Result<AdaptiveLearningUpdate, LearningError> {
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

        let predictions = self.model.predict_next(&sample.action_id, &sample.before)?;
        let mut prediction_errors_observed = 0;
        let mut change_detected = false;
        for (field, before_value) in before {
            let Some(after_value) = after.get(field) else {
                continue;
            };
            let (Some(before_number), Some(after_number)) =
                (before_value.as_f64(), after_value.as_f64())
            else {
                continue;
            };
            let Some(prediction) = predictions.get(field) else {
                continue;
            };
            if prediction.sample_count < self.policy.minimum_model_samples {
                continue;
            }

            let actual_delta = after_number - before_number;
            if !actual_delta.is_finite() {
                return Err(LearningError::NonFiniteDelta(field.clone()));
            }
            let residual = actual_delta - prediction.mean_delta;
            if !residual.is_finite() {
                return Err(LearningError::NonFinitePrediction(field.clone()));
            }
            prediction_errors_observed += 1;
            let monitor = self
                .residual_monitors
                .entry((sample.action_id.clone(), field.clone()))
                .or_default();
            if monitor.observe(residual, self.policy)? {
                change_detected = true;
            }
        }

        if change_detected {
            self.change_points_detected = self
                .change_points_detected
                .checked_add(1)
                .ok_or(LearningError::SampleCountExhausted)?;
            self.model.resize_window(self.policy.min_window_size)?;
            self.residual_monitors.clear();
            self.stable_observations = 0;
        } else if prediction_errors_observed > 0 {
            self.stable_observations = self.stable_observations.saturating_add(1);
            if self.stable_observations >= self.policy.stable_observations_to_expand
                && self.model.window_size < self.policy.max_window_size
            {
                self.model
                    .resize_window(self.model.window_size.saturating_add(1))?;
                self.stable_observations = 0;
            }
        }

        self.model.observe_transition(sample)?;
        Ok(AdaptiveLearningUpdate {
            change_detected,
            window_size: self.model.window_size(),
            prediction_errors_observed,
            change_points_detected: self.change_points_detected,
        })
    }

    pub fn predict_next(
        &self,
        action_id: &str,
        current_state: &DataValue,
    ) -> Result<BTreeMap<String, FieldPrediction>, LearningError> {
        self.model.predict_next(action_id, current_state)
    }

    pub fn window_size(&self) -> usize {
        self.model.window_size()
    }

    pub fn retained_transitions(&self) -> usize {
        self.model.retained_transitions()
    }

    pub fn transitions_seen(&self) -> u64 {
        self.model.transitions_seen()
    }

    pub fn change_points_detected(&self) -> u64 {
        self.change_points_detected
    }

    pub fn policy(&self) -> AdaptiveLearningPolicy {
        self.policy
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
    InvalidWindowSize,
    InvalidAdaptivePolicy,
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
            Self::InvalidWindowSize => write!(formatter, "learner window size must be positive"),
            Self::InvalidAdaptivePolicy => write!(formatter, "adaptive learning policy is invalid"),
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
    fn opposite_sign_maximum_deltas_keep_a_finite_mean() {
        let mut model = NumericDeltaModel::default();
        for delta in [f64::MAX, -f64::MAX] {
            model
                .observe_transition(&TransitionSample {
                    action_id: "advance".to_owned(),
                    before: state(0.0, "before"),
                    after: state(delta, "after"),
                })
                .expect("each delta is finite");
        }

        let statistics = model
            .statistics
            .get(&("advance".to_owned(), "position".to_owned()))
            .expect("position statistics");
        assert_eq!(statistics.mean(), 0.0);
        assert!(!statistics.standard_deviation().is_finite());
        assert_eq!(
            model.predict_next("advance", &state(0.0, "current")),
            Err(LearningError::NonFinitePrediction("position".to_owned()))
        );
    }

    #[test]
    fn rejects_unrepresentable_delta_without_partially_updating_other_fields() {
        let before = DataValue::object([
            ("a_safe".to_owned(), DataValue::Number(0.0)),
            ("z_overflow".to_owned(), DataValue::Number(-f64::MAX)),
        ]);
        let after = DataValue::object([
            ("a_safe".to_owned(), DataValue::Number(1.0)),
            ("z_overflow".to_owned(), DataValue::Number(f64::MAX)),
        ]);
        let mut model = NumericDeltaModel::default();

        assert_eq!(
            model.observe_transition(&TransitionSample {
                action_id: "advance".to_owned(),
                before,
                after,
            }),
            Err(LearningError::NonFiniteDelta("z_overflow".to_owned()))
        );
        assert!(model.statistics.is_empty());
        assert_eq!(model.transitions_seen(), 0);
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
        assert!(
            (estimate.delta_standard_deviation / f64::MAX - std::f64::consts::FRAC_1_SQRT_2).abs()
                < 1.0e-12
        );

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
    fn windowed_delta_model_forgets_stale_dynamics_and_tracks_lifetime_samples() {
        let mut cumulative = NumericDeltaModel::default();
        let mut recent = WindowedNumericDeltaModel::new(2).expect("positive window");
        for delta in [1.0, 1.0, -1.0, -1.0] {
            let sample = TransitionSample {
                action_id: "adapt".to_owned(),
                before: state(0.0, "before"),
                after: state(delta, "after"),
            };
            cumulative
                .observe_transition(&sample)
                .expect("cumulative update");
            recent.observe_transition(&sample).expect("recent update");
        }

        let current = state(0.0, "current");
        let cumulative_prediction = cumulative
            .predict_next("adapt", &current)
            .expect("cumulative prediction");
        let recent_prediction = recent
            .predict_next("adapt", &current)
            .expect("recent prediction");
        assert_eq!(
            cumulative_prediction
                .get("position")
                .expect("cumulative field")
                .mean_delta,
            0.0
        );
        assert_eq!(
            recent_prediction
                .get("position")
                .expect("recent field")
                .mean_delta,
            -1.0
        );
        assert_eq!(recent.window_size(), 2);
        assert_eq!(recent.retained_transitions(), 2);
        assert_eq!(recent.transitions_seen(), 4);
    }

    #[test]
    fn windowed_delta_model_rejects_invalid_updates_transactionally() {
        let mut model = WindowedNumericDeltaModel::new(3).expect("positive window");
        let valid = TransitionSample {
            action_id: "advance".to_owned(),
            before: state(0.0, "before"),
            after: state(1.0, "after"),
        };
        model.observe_transition(&valid).expect("valid update");
        let before = model
            .predict_next("advance", &state(0.0, "current"))
            .expect("prediction before invalid update");
        let invalid = TransitionSample {
            action_id: "advance".to_owned(),
            before: DataValue::object([("position".to_owned(), DataValue::Number(-f64::MAX))]),
            after: DataValue::object([("position".to_owned(), DataValue::Number(f64::MAX))]),
        };

        assert_eq!(
            model.observe_transition(&invalid),
            Err(LearningError::NonFiniteDelta("position".to_owned()))
        );
        assert_eq!(model.transitions_seen(), 1);
        assert_eq!(model.retained_transitions(), 1);
        assert_eq!(
            model
                .predict_next("advance", &state(0.0, "current"))
                .expect("prediction remains unchanged"),
            before
        );
    }

    #[test]
    fn rejects_zero_sized_recent_window() {
        assert!(matches!(
            WindowedNumericDeltaModel::new(0),
            Err(LearningError::InvalidWindowSize)
        ));
    }

    fn adaptive_policy() -> AdaptiveLearningPolicy {
        AdaptiveLearningPolicy {
            min_window_size: 4,
            max_window_size: 16,
            stable_observations_to_expand: 8,
            minimum_model_samples: 2,
            minimum_residual_samples: 8,
            consecutive_outliers: 2,
            change_threshold_sigma: 3.0,
        }
    }

    fn numeric_sample(delta: f64) -> TransitionSample {
        TransitionSample {
            action_id: "adapt".to_owned(),
            before: DataValue::object([("position".to_owned(), DataValue::Number(0.0))]),
            after: DataValue::object([("position".to_owned(), DataValue::Number(delta))]),
        }
    }

    #[test]
    fn adaptive_learner_expands_during_stability_and_shrinks_after_a_change() {
        let mut model =
            AdaptiveNumericDeltaModel::new(adaptive_policy()).expect("valid adaptive policy");
        for _ in 0..100 {
            model
                .observe_transition(&numeric_sample(1.0))
                .expect("stable transition");
        }
        assert_eq!(model.window_size(), adaptive_policy().max_window_size);
        assert_eq!(model.change_points_detected(), 0);

        let mut detection_step = None;
        for step in 1..=6 {
            let update = model
                .observe_transition(&numeric_sample(-1.0))
                .expect("changed transition");
            if update.change_detected {
                detection_step = Some(step);
                assert_eq!(update.window_size, adaptive_policy().min_window_size);
                break;
            }
        }
        assert!(
            detection_step.is_some_and(|step| step <= 3),
            "the regime shift should be detected within three observations"
        );
        assert_eq!(model.change_points_detected(), 1);
    }

    #[test]
    fn adaptive_learner_avoids_false_alarms_under_stationary_noise() {
        let mut model =
            AdaptiveNumericDeltaModel::new(adaptive_policy()).expect("valid adaptive policy");
        let mut random = 0xD1B5_4A32_D192_ED03_u64;
        for _ in 0..5_000 {
            let mut noise = 0.0;
            for _ in 0..12 {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                noise += ((random >> 11) as f64 / ((1_u64 << 53) as f64)) - 0.5;
            }
            let delta = 1.0 + noise / 12.0_f64.sqrt() * 0.7;
            model
                .observe_transition(&numeric_sample(delta))
                .expect("finite stationary noisy transition");
        }
        assert_eq!(
            model.change_points_detected(),
            0,
            "stationary noise must not be mistaken for a persistent regime change"
        );
    }

    #[test]
    fn adaptive_learner_rejects_invalid_updates_transactionally() {
        let mut model =
            AdaptiveNumericDeltaModel::new(adaptive_policy()).expect("valid adaptive policy");
        for _ in 0..40 {
            model
                .observe_transition(&numeric_sample(1.0))
                .expect("valid transition");
        }
        let before_prediction = model
            .predict_next(
                "adapt",
                &DataValue::object([("position".to_owned(), DataValue::Number(0.0))]),
            )
            .expect("prediction before invalid update");
        let before_window = model.window_size();
        let before_count = model.transitions_seen();
        let before_changes = model.change_points_detected();
        let invalid = TransitionSample {
            action_id: "adapt".to_owned(),
            before: DataValue::object([("position".to_owned(), DataValue::Number(-f64::MAX))]),
            after: DataValue::object([("position".to_owned(), DataValue::Number(f64::MAX))]),
        };
        assert_eq!(
            model.observe_transition(&invalid),
            Err(LearningError::NonFiniteDelta("position".to_owned()))
        );
        assert_eq!(model.window_size(), before_window);
        assert_eq!(model.transitions_seen(), before_count);
        assert_eq!(model.change_points_detected(), before_changes);
        assert_eq!(
            model
                .predict_next(
                    "adapt",
                    &DataValue::object([("position".to_owned(), DataValue::Number(0.0))]),
                )
                .expect("prediction remains unchanged"),
            before_prediction
        );
    }

    #[test]
    fn rejects_invalid_adaptive_learning_policies() {
        let mut policy = adaptive_policy();
        policy.max_window_size = 0;
        assert!(matches!(
            AdaptiveNumericDeltaModel::new(policy),
            Err(LearningError::InvalidAdaptivePolicy)
        ));
        policy = adaptive_policy();
        policy.change_threshold_sigma = f64::NAN;
        assert!(matches!(
            AdaptiveNumericDeltaModel::new(policy),
            Err(LearningError::InvalidAdaptivePolicy)
        ));
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
                    (
                        "action_id".to_owned(),
                        DataValue::Text("advance".to_owned()),
                    ),
                    ("field".to_owned(), DataValue::Text("position".to_owned())),
                    ("count".to_owned(), DataValue::Text("2".to_owned())),
                    ("mean".to_owned(), DataValue::Number(0.0)),
                    ("sum_squared_deviations".to_owned(), DataValue::Number(2.0)),
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
