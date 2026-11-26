//! Append-only durable checkpoints and deterministic journal restoration.
//!
//! Checkpoints are stored in their own single-writer, checksummed append-only
//! file. This avoids relying on cross-platform replacement semantics for a
//! mutable latest-checkpoint file. A torn final frame is truncated on reopen;
//! complete-frame corruption fails closed.

use crate::experience::ExperienceRecord;
use crate::experience_journal::{
    crc32, decode_data_value, encode_data_value, MAX_FRAME_PAYLOAD_BYTES,
};
use crate::value::DataValue;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

const CHECKPOINT_MAGIC: [u8; 8] = *b"5413CKPT";
const CHECKPOINT_FORMAT_VERSION: u16 = 1;
const CHECKPOINT_HEADER_SIZE: usize = 18;

#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointSnapshot {
    pub environment_id: String,
    pub environment_version: String,
    pub state_schema_id: String,
    pub state_schema_version: u64,
    /// The last journal sequence incorporated into state. None means no
    /// experience record has yet been applied.
    pub last_applied_sequence: Option<u64>,
    pub state: DataValue,
    /// References to separately persisted, versioned model artifacts.
    pub model_artifact_refs: BTreeMap<String, String>,
}

impl CheckpointSnapshot {
    pub fn validate(&self) -> Result<(), CheckpointError> {
        if self.environment_id.trim().is_empty()
            || self.environment_version.trim().is_empty()
            || self.state_schema_id.trim().is_empty()
            || self.state_schema_version == 0
        {
            return Err(CheckpointError::InvalidSnapshot);
        }
        self.state
            .validate_finite()
            .map_err(|_| CheckpointError::InvalidSnapshot)?;
        if self
            .model_artifact_refs
            .iter()
            .any(|(model_id, artifact_ref)| {
                model_id.trim().is_empty() || artifact_ref.trim().is_empty()
            })
        {
            return Err(CheckpointError::InvalidSnapshot);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointAppendOutcome {
    Appended,
    AlreadyPresent,
}

pub struct CheckpointStore {
    file: File,
    snapshots: Vec<CheckpointSnapshot>,
    poisoned: bool,
}

impl CheckpointStore {
    /// Opens a checkpoint log and reconstructs its ordered snapshots.
    ///
    /// This store assumes exactly one active writer. It repairs only an
    /// incomplete final frame; complete-frame corruption is returned as error.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CheckpointError> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        let snapshots = scan_and_recover(&mut file)?;
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            snapshots,
            poisoned: false,
        })
    }

    /// Persists a snapshot. Repeating the latest identical snapshot is
    /// idempotent; same-sequence conflicts and sequence regressions are rejected.
    pub fn append(
        &mut self,
        snapshot: CheckpointSnapshot,
    ) -> Result<CheckpointAppendOutcome, CheckpointError> {
        if self.poisoned {
            return Err(CheckpointError::Poisoned);
        }
        snapshot.validate()?;

        if let Some(latest) = self.snapshots.last() {
            match compare_sequence(snapshot.last_applied_sequence, latest.last_applied_sequence) {
                std::cmp::Ordering::Less => return Err(CheckpointError::SequenceRegression),
                std::cmp::Ordering::Equal => {
                    if snapshot == *latest {
                        return Ok(CheckpointAppendOutcome::AlreadyPresent);
                    }
                    return Err(CheckpointError::SequenceConflict);
                }
                std::cmp::Ordering::Greater => {}
            }
        }

        let frame = encode_frame(&snapshot)?;
        if let Err(error) = self
            .file
            .seek(SeekFrom::End(0))
            .and_then(|_| self.file.write_all(&frame))
            .and_then(|_| self.file.sync_all())
        {
            self.poisoned = true;
            return Err(CheckpointError::Io(error));
        }

        self.snapshots.push(snapshot);
        Ok(CheckpointAppendOutcome::Appended)
    }

    pub fn latest(&self) -> Option<&CheckpointSnapshot> {
        self.snapshots.last()
    }

    pub fn snapshots(&self) -> &[CheckpointSnapshot] {
        &self.snapshots
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

fn compare_sequence(left: Option<u64>, right: Option<u64>) -> std::cmp::Ordering {
    match (left, right) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some(left), Some(right)) => left.cmp(&right),
    }
}

fn encode_frame(snapshot: &CheckpointSnapshot) -> Result<Vec<u8>, CheckpointError> {
    let payload = encode_data_value(&snapshot_to_value(snapshot)?)
        .map_err(|error| CheckpointError::Encoding(error.to_string()))?;
    if payload.len() > MAX_FRAME_PAYLOAD_BYTES {
        return Err(CheckpointError::FrameTooLarge {
            size: payload.len(),
            maximum: MAX_FRAME_PAYLOAD_BYTES,
        });
    }
    let payload_length =
        u32::try_from(payload.len()).map_err(|_| CheckpointError::FrameTooLarge {
            size: payload.len(),
            maximum: MAX_FRAME_PAYLOAD_BYTES,
        })?;
    let mut frame = Vec::with_capacity(CHECKPOINT_HEADER_SIZE + payload.len());
    frame.extend_from_slice(&CHECKPOINT_MAGIC);
    frame.extend_from_slice(&CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    frame.extend_from_slice(&payload_length.to_le_bytes());
    frame.extend_from_slice(&crc32(&payload).to_le_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

fn scan_and_recover(file: &mut File) -> Result<Vec<CheckpointSnapshot>, CheckpointError> {
    file.seek(SeekFrom::Start(0))?;
    let mut snapshots: Vec<CheckpointSnapshot> = Vec::new();
    let mut offset = 0_u64;

    loop {
        let mut header = [0_u8; CHECKPOINT_HEADER_SIZE];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                truncate_incomplete_tail(file, offset)?;
                break;
            }
            Err(error) => return Err(CheckpointError::Io(error)),
        }

        if header[..8] != CHECKPOINT_MAGIC {
            return Err(CheckpointError::CorruptFrame {
                offset,
                reason: "invalid checkpoint frame marker".to_owned(),
            });
        }
        let version = u16::from_le_bytes([header[8], header[9]]);
        if version != CHECKPOINT_FORMAT_VERSION {
            return Err(CheckpointError::UnsupportedVersion(version));
        }
        let payload_length =
            u32::from_le_bytes([header[10], header[11], header[12], header[13]]) as usize;
        if payload_length > MAX_FRAME_PAYLOAD_BYTES {
            return Err(CheckpointError::FrameTooLarge {
                size: payload_length,
                maximum: MAX_FRAME_PAYLOAD_BYTES,
            });
        }
        let expected_checksum =
            u32::from_le_bytes([header[14], header[15], header[16], header[17]]);
        let mut payload = vec![0_u8; payload_length];
        match file.read_exact(&mut payload) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                truncate_incomplete_tail(file, offset)?;
                break;
            }
            Err(error) => return Err(CheckpointError::Io(error)),
        }
        if crc32(&payload) != expected_checksum {
            return Err(CheckpointError::CorruptFrame {
                offset,
                reason: "CRC-32 checksum mismatch".to_owned(),
            });
        }

        let value = decode_data_value(&payload).map_err(|error| CheckpointError::CorruptFrame {
            offset,
            reason: error.to_string(),
        })?;
        let snapshot = snapshot_from_value(value)
            .map_err(|reason| CheckpointError::CorruptFrame { offset, reason })?;
        if let Some(previous) = snapshots.last() {
            if compare_sequence(
                snapshot.last_applied_sequence,
                previous.last_applied_sequence,
            ) != std::cmp::Ordering::Greater
            {
                return Err(CheckpointError::CorruptFrame {
                    offset,
                    reason: "checkpoint sequence is not strictly increasing".to_owned(),
                });
            }
        }
        snapshots.push(snapshot);

        offset = offset
            .checked_add(CHECKPOINT_HEADER_SIZE as u64)
            .and_then(|value| value.checked_add(payload_length as u64))
            .ok_or_else(|| CheckpointError::CorruptFrame {
                offset,
                reason: "checkpoint offset overflow".to_owned(),
            })?;
    }

    Ok(snapshots)
}

fn truncate_incomplete_tail(file: &mut File, valid_length: u64) -> Result<(), CheckpointError> {
    file.set_len(valid_length)?;
    file.sync_all()?;
    Ok(())
}

fn snapshot_to_value(snapshot: &CheckpointSnapshot) -> Result<DataValue, CheckpointError> {
    snapshot.validate()?;
    let artifact_refs = DataValue::Object(
        snapshot
            .model_artifact_refs
            .iter()
            .map(|(model_id, artifact_ref)| {
                (model_id.clone(), DataValue::Text(artifact_ref.clone()))
            })
            .collect(),
    );
    Ok(DataValue::object([
        (
            "environment_id".to_owned(),
            DataValue::Text(snapshot.environment_id.clone()),
        ),
        (
            "environment_version".to_owned(),
            DataValue::Text(snapshot.environment_version.clone()),
        ),
        (
            "state_schema_id".to_owned(),
            DataValue::Text(snapshot.state_schema_id.clone()),
        ),
        (
            "state_schema_version".to_owned(),
            DataValue::Text(snapshot.state_schema_version.to_string()),
        ),
        (
            "last_applied_sequence".to_owned(),
            DataValue::Text(
                snapshot
                    .last_applied_sequence
                    .map(|sequence| sequence.to_string())
                    .unwrap_or_default(),
            ),
        ),
        ("state".to_owned(), snapshot.state.clone()),
        ("model_artifact_refs".to_owned(), artifact_refs),
    ]))
}

fn snapshot_from_value(value: DataValue) -> Result<CheckpointSnapshot, String> {
    let DataValue::Object(mut fields) = value else {
        return Err("checkpoint payload must be an object".to_owned());
    };
    if fields.len() != 7 {
        return Err("checkpoint payload contains missing or unknown fields".to_owned());
    }

    fn take_text(fields: &mut BTreeMap<String, DataValue>, key: &str) -> Result<String, String> {
        match fields.remove(key) {
            Some(DataValue::Text(value)) => Ok(value),
            _ => Err(format!("checkpoint field {key} must be text")),
        }
    }

    let environment_id = take_text(&mut fields, "environment_id")?;
    let environment_version = take_text(&mut fields, "environment_version")?;
    let state_schema_id = take_text(&mut fields, "state_schema_id")?;
    let state_schema_version = take_text(&mut fields, "state_schema_version")?
        .parse::<u64>()
        .map_err(|_| "checkpoint schema version is invalid".to_owned())?;
    let sequence_text = take_text(&mut fields, "last_applied_sequence")?;
    let last_applied_sequence = if sequence_text.is_empty() {
        None
    } else {
        Some(
            sequence_text
                .parse::<u64>()
                .map_err(|_| "checkpoint journal sequence is invalid".to_owned())?,
        )
    };
    let state = fields
        .remove("state")
        .ok_or_else(|| "checkpoint state is missing".to_owned())?;
    let artifact_values = match fields.remove("model_artifact_refs") {
        Some(DataValue::Object(values)) => values,
        _ => return Err("checkpoint artifact references must be an object".to_owned()),
    };
    let mut model_artifact_refs = BTreeMap::new();
    for (model_id, value) in artifact_values {
        let DataValue::Text(artifact_ref) = value else {
            return Err("model artifact reference must be text".to_owned());
        };
        model_artifact_refs.insert(model_id, artifact_ref);
    }

    let snapshot = CheckpointSnapshot {
        environment_id,
        environment_version,
        state_schema_id,
        state_schema_version,
        last_applied_sequence,
        state,
        model_artifact_refs,
    };
    snapshot
        .validate()
        .map_err(|error| format!("invalid checkpoint snapshot: {error}"))?;
    Ok(snapshot)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestoreTarget<'a> {
    pub environment_id: &'a str,
    pub environment_version: &'a str,
    pub state_schema_id: &'a str,
    pub state_schema_version: u64,
}

impl RestoreTarget<'_> {
    fn validate(&self) -> bool {
        !self.environment_id.trim().is_empty()
            && !self.environment_version.trim().is_empty()
            && !self.state_schema_id.trim().is_empty()
            && self.state_schema_version > 0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RestorationReport {
    pub state: DataValue,
    pub restored_from_sequence: Option<u64>,
    pub last_applied_sequence: Option<u64>,
    pub replayed_records: usize,
    pub model_artifact_refs: BTreeMap<String, String>,
}

/// Restores state from a checkpoint, then applies every later journal record.
/// The apply callback is domain-specific and must be deterministic for recovery.
/// This function verifies environment/schema identity, journal continuity, and
/// that the checkpoint high-water mark exists in the supplied journal.
pub fn restore_state<E, F>(
    target: RestoreTarget<'_>,
    initial_state: DataValue,
    checkpoint: Option<&CheckpointSnapshot>,
    journal_records: &[ExperienceRecord],
    mut apply: F,
) -> Result<RestorationReport, RestoreError<E>>
where
    F: FnMut(&mut DataValue, &ExperienceRecord) -> Result<(), E>,
{
    if !target.validate() {
        return Err(RestoreError::InvalidTarget);
    }
    initial_state
        .validate_finite()
        .map_err(|_| RestoreError::InvalidInitialState)?;

    let mut last_record_sequence = None;
    for (index, record) in journal_records.iter().enumerate() {
        let expected = u64::try_from(index).map_err(|_| RestoreError::SequenceOverflow)?;
        if record.sequence != expected {
            return Err(RestoreError::JournalSequenceGap {
                expected,
                actual: record.sequence,
            });
        }
        last_record_sequence = Some(record.sequence);
    }

    let (
        mut state,
        restored_from_sequence,
        mut last_applied_sequence,
        model_artifact_refs,
        replay_start,
    ) = match checkpoint {
        Some(snapshot) => {
            snapshot
                .validate()
                .map_err(|_| RestoreError::InvalidCheckpoint)?;
            if snapshot.environment_id != target.environment_id
                || snapshot.environment_version != target.environment_version
            {
                return Err(RestoreError::EnvironmentMismatch);
            }
            if snapshot.state_schema_id != target.state_schema_id
                || snapshot.state_schema_version != target.state_schema_version
            {
                return Err(RestoreError::SchemaMismatch);
            }
            let replay_start = match snapshot.last_applied_sequence {
                None => 0,
                Some(sequence) => {
                    if Some(sequence) > last_record_sequence {
                        return Err(RestoreError::CheckpointAheadOfJournal);
                    }
                    let index = journal_records
                        .iter()
                        .position(|record| record.sequence == sequence)
                        .ok_or(RestoreError::CheckpointSequenceMissing(sequence))?;
                    index.checked_add(1).ok_or(RestoreError::SequenceOverflow)?
                }
            };
            (
                snapshot.state.clone(),
                snapshot.last_applied_sequence,
                snapshot.last_applied_sequence,
                snapshot.model_artifact_refs.clone(),
                replay_start,
            )
        }
        None => (initial_state, None, None, BTreeMap::new(), 0),
    };

    let mut replayed_records = 0usize;
    for record in journal_records.iter().skip(replay_start) {
        apply(&mut state, record).map_err(RestoreError::Apply)?;
        state
            .validate_finite()
            .map_err(|_| RestoreError::AppliedInvalidState)?;
        replayed_records = replayed_records
            .checked_add(1)
            .ok_or(RestoreError::SequenceOverflow)?;
        last_applied_sequence = Some(record.sequence);
    }

    Ok(RestorationReport {
        state,
        restored_from_sequence,
        last_applied_sequence,
        replayed_records,
        model_artifact_refs,
    })
}

#[derive(Debug)]
pub enum RestoreError<E> {
    InvalidTarget,
    InvalidInitialState,
    AppliedInvalidState,
    InvalidCheckpoint,
    EnvironmentMismatch,
    SchemaMismatch,
    JournalSequenceGap { expected: u64, actual: u64 },
    CheckpointAheadOfJournal,
    CheckpointSequenceMissing(u64),
    SequenceOverflow,
    Apply(E),
}

impl<E: std::fmt::Display> std::fmt::Display for RestoreError<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTarget => write!(formatter, "restore target is invalid"),
            Self::InvalidInitialState => {
                write!(formatter, "initial state contains a non-finite value")
            }
            Self::AppliedInvalidState => {
                write!(formatter, "replayed event produced a non-finite state")
            }
            Self::InvalidCheckpoint => write!(formatter, "checkpoint snapshot is invalid"),
            Self::EnvironmentMismatch => {
                write!(formatter, "checkpoint environment identity does not match")
            }
            Self::SchemaMismatch => write!(formatter, "checkpoint state schema does not match"),
            Self::JournalSequenceGap { expected, actual } => {
                write!(
                    formatter,
                    "journal sequence gap: expected {expected}, found {actual}"
                )
            }
            Self::CheckpointAheadOfJournal => {
                write!(
                    formatter,
                    "checkpoint sequence is ahead of the experience journal"
                )
            }
            Self::CheckpointSequenceMissing(sequence) => {
                write!(
                    formatter,
                    "checkpoint sequence {sequence} is absent from the experience journal"
                )
            }
            Self::SequenceOverflow => write!(formatter, "restore sequence counter is exhausted"),
            Self::Apply(error) => write!(formatter, "replaying experience event failed: {error}"),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for RestoreError<E> {}

#[derive(Debug)]
pub enum CheckpointError {
    Io(io::Error),
    Encoding(String),
    InvalidSnapshot,
    UnsupportedVersion(u16),
    FrameTooLarge { size: usize, maximum: usize },
    CorruptFrame { offset: u64, reason: String },
    SequenceRegression,
    SequenceConflict,
    Poisoned,
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "checkpoint I/O error: {error}"),
            Self::Encoding(message) => write!(formatter, "checkpoint encoding failed: {message}"),
            Self::InvalidSnapshot => write!(formatter, "checkpoint snapshot is invalid"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported checkpoint version: {version}")
            }
            Self::FrameTooLarge { size, maximum } => {
                write!(
                    formatter,
                    "checkpoint frame size {size} exceeds limit {maximum}"
                )
            }
            Self::CorruptFrame { offset, reason } => {
                write!(
                    formatter,
                    "corrupt checkpoint frame at byte {offset}: {reason}"
                )
            }
            Self::SequenceRegression => {
                write!(formatter, "checkpoint sequence cannot move backwards")
            }
            Self::SequenceConflict => {
                write!(
                    formatter,
                    "checkpoint sequence was reused for different state"
                )
            }
            Self::Poisoned => {
                write!(
                    formatter,
                    "checkpoint store must be reopened after an I/O failure"
                )
            }
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<io::Error> for CheckpointError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experience::ExperienceEvent;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

    fn test_path() -> std::path::PathBuf {
        let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("5413-checkpoints-{}-{id}.bin", std::process::id()))
    }

    fn snapshot(sequence: Option<u64>, count: i64) -> CheckpointSnapshot {
        CheckpointSnapshot {
            environment_id: "environment-a".to_owned(),
            environment_version: "v1".to_owned(),
            state_schema_id: "state".to_owned(),
            state_schema_version: 1,
            last_applied_sequence: sequence,
            state: DataValue::object([("count".to_owned(), DataValue::Integer(count))]),
            model_artifact_refs: BTreeMap::from([(
                "model-a".to_owned(),
                "artifact:sha256:abc".to_owned(),
            )]),
        }
    }

    fn record(sequence: u64) -> ExperienceRecord {
        ExperienceRecord {
            sequence,
            episode_id: "episode-a".to_owned(),
            event: ExperienceEvent::PredictionEvaluated {
                action_id: "increment".to_owned(),
                predicted: DataValue::Null,
                observed: DataValue::Null,
            },
        }
    }

    fn target() -> RestoreTarget<'static> {
        RestoreTarget {
            environment_id: "environment-a",
            environment_version: "v1",
            state_schema_id: "state",
            state_schema_version: 1,
        }
    }

    #[test]
    fn checkpoints_are_idempotent_and_survive_reopen() {
        let path = test_path();
        let first = snapshot(None, 0);
        let second = snapshot(Some(1), 2);
        {
            let mut store = CheckpointStore::open(&path).expect("open");
            assert_eq!(
                store.append(first.clone()).expect("append first"),
                CheckpointAppendOutcome::Appended
            );
            assert_eq!(
                store.append(first).expect("duplicate first"),
                CheckpointAppendOutcome::AlreadyPresent
            );
            store.append(second.clone()).expect("append second");
        }
        {
            let store = CheckpointStore::open(&path).expect("reopen");
            assert_eq!(store.len(), 2);
            assert_eq!(store.latest(), Some(&second));
            assert_eq!(
                store.latest().expect("latest").model_artifact_refs["model-a"],
                "artifact:sha256:abc"
            );
        }
        std::fs::remove_file(path).expect("remove checkpoint file");
    }

    #[test]
    fn rejects_sequence_regression_and_conflicting_reuse() {
        let path = test_path();
        let mut store = CheckpointStore::open(&path).expect("open");
        store.append(snapshot(Some(4), 4)).expect("append");
        assert!(matches!(
            store.append(snapshot(Some(3), 3)),
            Err(CheckpointError::SequenceRegression)
        ));
        assert!(matches!(
            store.append(snapshot(Some(4), 9)),
            Err(CheckpointError::SequenceConflict)
        ));
        drop(store);
        std::fs::remove_file(path).expect("remove checkpoint file");
    }

    #[test]
    fn restore_replays_only_records_after_checkpoint() {
        let records = vec![record(0), record(1), record(2)];
        let checkpoint = snapshot(Some(1), 2);
        let report = restore_state(
            target(),
            DataValue::object([("count".to_owned(), DataValue::Integer(0))]),
            Some(&checkpoint),
            &records,
            |state, _record| {
                let count = state
                    .get("count")
                    .and_then(DataValue::as_f64)
                    .unwrap_or(0.0);
                *state =
                    DataValue::object([("count".to_owned(), DataValue::Integer(count as i64 + 1))]);
                Ok::<(), &'static str>(())
            },
        )
        .expect("restore");
        assert_eq!(report.replayed_records, 1);
        assert_eq!(report.last_applied_sequence, Some(2));
        assert_eq!(report.state.get("count"), Some(&DataValue::Integer(3)));
    }

    #[test]
    fn restore_rejects_checkpoint_ahead_or_wrong_environment() {
        let records = vec![record(0)];
        let ahead = snapshot(Some(2), 3);
        assert!(matches!(
            restore_state(
                target(),
                DataValue::Null,
                Some(&ahead),
                &records,
                |_, _| Ok::<(), &'static str>(())
            ),
            Err(RestoreError::CheckpointAheadOfJournal)
        ));

        let mut wrong_environment = snapshot(Some(0), 1);
        wrong_environment.environment_id = "other".to_owned();
        assert!(matches!(
            restore_state(
                target(),
                DataValue::Null,
                Some(&wrong_environment),
                &records,
                |_, _| Ok::<(), &'static str>(())
            ),
            Err(RestoreError::EnvironmentMismatch)
        ));
    }

    #[test]
    fn truncated_final_checkpoint_frame_is_recovered() {
        let path = test_path();
        let first = snapshot(Some(0), 1);
        let second = snapshot(Some(1), 2);
        let first_length;
        {
            let mut store = CheckpointStore::open(&path).expect("open");
            store.append(first.clone()).expect("append first");
            first_length = store.file.metadata().expect("metadata").len();
            store.append(second).expect("append second");
        }
        {
            let file = OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("open raw");
            file.set_len(first_length + 3)
                .expect("truncate final header");
            file.sync_all().expect("sync");
        }
        {
            let store = CheckpointStore::open(&path).expect("recover");
            assert_eq!(store.len(), 1);
            assert_eq!(store.latest(), Some(&first));
        }
        std::fs::remove_file(path).expect("remove checkpoint file");
    }

    #[test]
    fn complete_frame_corruption_fails_closed() {
        let path = test_path();
        {
            let mut store = CheckpointStore::open(&path).expect("open");
            store.append(snapshot(Some(0), 1)).expect("append");
        }
        {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open raw");
            file.seek(SeekFrom::Start(CHECKPOINT_HEADER_SIZE as u64))
                .expect("seek");
            let mut byte = [0_u8; 1];
            file.read_exact(&mut byte).expect("read");
            file.seek(SeekFrom::Start(CHECKPOINT_HEADER_SIZE as u64))
                .expect("rewind");
            file.write_all(&[byte[0] ^ 0x01]).expect("corrupt");
            file.sync_all().expect("sync");
        }
        assert!(matches!(
            CheckpointStore::open(&path),
            Err(CheckpointError::CorruptFrame { .. })
        ));
        std::fs::remove_file(path).expect("remove checkpoint file");
    }
}
