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
        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(LedgerError::SequenceExhausted)?;
        let sequence = self.next_sequence;

        self.records.push(ExperienceRecord {
            sequence,
            episode_id: episode_id.into(),
            event,
        });
        self.next_sequence = next_sequence;
        Ok(sequence)
    }

    pub fn records(&self) -> &[ExperienceRecord] {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerError {
    SequenceExhausted,
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SequenceExhausted => write!(formatter, "experience sequence is exhausted"),
        }
    }
}

impl std::error::Error for LedgerError {}

#[cfg(test)]
mod tests {
    use super::*;

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
}
