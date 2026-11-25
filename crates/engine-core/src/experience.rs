use crate::environment::{ActionOutcome, ObservationFrame};
use crate::value::DataValue;

#[derive(Clone, Debug, PartialEq)]
pub enum ExperienceEvent {
    Observation(ObservationFrame),
    ActionProposed {
        action_id: String,
        arguments: DataValue,
    },
    ActionCompleted(ActionOutcome),
    PredictionEvaluated {
        action_id: String,
        predicted: DataValue,
        observed: DataValue,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperienceRecord {
    pub sequence: u64,
    pub episode_id: String,
    pub event: ExperienceEvent,
}

#[derive(Clone, Debug, Default)]
pub struct ExperienceLedger {
    next_sequence: u64,
    records: Vec<ExperienceRecord>,
}

impl ExperienceLedger {
    pub fn append(
        &mut self,
        episode_id: impl Into<String>,
        event: ExperienceEvent,
    ) -> Result<u64, LedgerError> {
        let sequence = self.next_sequence;
        self.append_record(ExperienceRecord {
            sequence,
            episode_id: episode_id.into(),
            event,
        })?;
        Ok(sequence)
    }

    pub fn records(&self) -> &[ExperienceRecord] {
        &self.records
    }

    pub fn records_for_episode<'a>(
        &'a self,
        episode_id: &'a str,
    ) -> impl Iterator<Item = &'a ExperienceRecord> + 'a {
        self.records
            .iter()
            .filter(move |record| record.episode_id == episode_id)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub(crate) fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    pub(crate) fn append_record(
        &mut self,
        record: ExperienceRecord,
    ) -> Result<(), LedgerError> {
        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(LedgerError::SequenceExhausted)?;

        if record.sequence != self.next_sequence {
            return Err(LedgerError::UnexpectedSequence {
                expected: self.next_sequence,
                actual: record.sequence,
            });
        }

        self.records.push(record);
        self.next_sequence = next_sequence;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerError {
    SequenceExhausted,
    UnexpectedSequence { expected: u64, actual: u64 },
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SequenceExhausted => write!(formatter, "experience sequence is exhausted"),
            Self::UnexpectedSequence { expected, actual } => write!(
                formatter,
                "experience sequence mismatch: expected {expected}, received {actual}"
            ),
        }
    }
}

impl std::error::Error for LedgerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::DataValue;

    #[test]
    fn appends_records_with_monotonic_sequences() {
        let mut ledger = ExperienceLedger::default();
        let first = ledger
            .append(
                "episode-a",
                ExperienceEvent::PredictionEvaluated {
                    action_id: "wait".to_owned(),
                    predicted: DataValue::Integer(1),
                    observed: DataValue::Integer(1),
                },
            )
            .expect("first append should succeed");
        let second = ledger
            .append(
                "episode-a",
                ExperienceEvent::PredictionEvaluated {
                    action_id: "wait".to_owned(),
                    predicted: DataValue::Integer(2),
                    observed: DataValue::Integer(3),
                },
            )
            .expect("second append should succeed");

        assert_eq!((first, second), (0, 1));
        assert_eq!(ledger.len(), 2);
    }

    #[test]
    fn episode_replay_preserves_global_sequence_order() {
        let mut ledger = ExperienceLedger::default();
        for episode_id in ["episode-a", "episode-b", "episode-a"] {
            ledger
                .append(
                    episode_id,
                    ExperienceEvent::PredictionEvaluated {
                        action_id: "observe".to_owned(),
                        predicted: DataValue::Null,
                        observed: DataValue::Null,
                    },
                )
                .expect("append should succeed");
        }

        let sequences: Vec<_> = ledger
            .records_for_episode("episode-a")
            .map(|record| record.sequence)
            .collect();
        assert_eq!(sequences, vec![0, 2]);
    }
}
