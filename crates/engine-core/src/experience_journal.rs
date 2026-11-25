//! Dependency-free, versioned append-only persistence for experience records.
//!
//! Each frame contains a magic marker, format version, payload length, CRC-32,
//! and a deterministic binary encoding of one record. CRC-32 detects accidental
//! corruption; it is not an authenticity or tamper-resistance mechanism.
//!
//! A journal is intended for a single writer. Every successful append calls
//! File::sync_all before updating the in-memory view. Filesystem and hardware
//! durability guarantees still depend on the host platform and storage stack.

use crate::environment::{ActionOutcome, ObservationFrame};
use crate::experience::{ExperienceEvent, ExperienceLedger, ExperienceRecord, LedgerError};
use crate::value::DataValue;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

const FRAME_MAGIC: [u8; 4] = *b"5413";
const FORMAT_VERSION: u16 = 1;
const HEADER_SIZE: usize = 14;
const MAX_FRAME_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
const MAX_VALUE_NESTING: usize = 128;
const MAX_COLLECTION_ITEMS: usize = 100_000;

pub struct ExperienceJournal {
    file: File,
    ledger: ExperienceLedger,
    poisoned: bool,
}

impl ExperienceJournal {
    /// Opens or creates a journal, rebuilding the in-memory ledger from its frames.
    ///
    /// An incomplete final header or payload is treated as a torn tail and
    /// truncated. A complete frame with invalid magic, checksum, encoding, or
    /// sequence is reported as corruption and is never silently discarded.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, JournalError> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;

        let ledger = scan_and_recover(&mut file)?;
        file.seek(SeekFrom::End(0))?;

        Ok(Self {
            file,
            ledger,
            poisoned: false,
        })
    }

    /// Appends and synchronizes one record before making it visible in memory.
    ///
    /// If writing or synchronization fails, this instance is poisoned and must
    /// be reopened. The outcome may be ambiguous after a synchronization error:
    /// reopening determines whether the frame reached the journal.
    pub fn append(
        &mut self,
        episode_id: impl Into<String>,
        event: ExperienceEvent,
    ) -> Result<u64, JournalError> {
        if self.poisoned {
            return Err(JournalError::Poisoned);
        }

        let sequence = self.ledger.next_sequence();
        sequence
            .checked_add(1)
            .ok_or(JournalError::Ledger(LedgerError::SequenceExhausted))?;

        let record = ExperienceRecord {
            sequence,
            episode_id: episode_id.into(),
            event,
        };
        let frame = encode_frame(&record)?;

        if let Err(error) = self
            .file
            .seek(SeekFrom::End(0))
            .and_then(|_| self.file.write_all(&frame))
            .and_then(|_| self.file.sync_all())
        {
            self.poisoned = true;
            return Err(JournalError::Io(error));
        }

        if let Err(error) = self.ledger.append_record(record) {
            self.poisoned = true;
            return Err(JournalError::Ledger(error));
        }

        Ok(sequence)
    }

    pub fn records(&self) -> &[ExperienceRecord] {
        self.ledger.records()
    }

    /// Returns events for one episode in their original global append order.
    pub fn replay_episode<'a>(
        &'a self,
        episode_id: &'a str,
    ) -> impl Iterator<Item = &'a ExperienceRecord> + 'a {
        self.ledger.records_for_episode(episode_id)
    }

    pub fn len(&self) -> usize {
        self.ledger.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ledger.is_empty()
    }

    pub fn ledger(&self) -> &ExperienceLedger {
        &self.ledger
    }
}

fn scan_and_recover(file: &mut File) -> Result<ExperienceLedger, JournalError> {
    file.seek(SeekFrom::Start(0))?;
    let mut ledger = ExperienceLedger::default();
    let mut offset = 0_u64;

    loop {
        let mut header = [0_u8; HEADER_SIZE];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                truncate_incomplete_tail(file, offset)?;
                break;
            }
            Err(error) => return Err(JournalError::Io(error)),
        }

        if header[..4] != FRAME_MAGIC {
            return Err(JournalError::CorruptFrame {
                offset,
                reason: "invalid frame marker".to_owned(),
            });
        }

        let version = u16::from_le_bytes([header[4], header[5]]);
        if version != FORMAT_VERSION {
            return Err(JournalError::UnsupportedVersion(version));
        }

        let payload_length =
            u32::from_le_bytes([header[6], header[7], header[8], header[9]]) as usize;
        if payload_length > MAX_FRAME_PAYLOAD_BYTES {
            return Err(JournalError::FrameTooLarge {
                size: payload_length,
                maximum: MAX_FRAME_PAYLOAD_BYTES,
            });
        }

        let expected_checksum =
            u32::from_le_bytes([header[10], header[11], header[12], header[13]]);
        let mut payload = vec![0_u8; payload_length];
        match file.read_exact(&mut payload) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                truncate_incomplete_tail(file, offset)?;
                break;
            }
            Err(error) => return Err(JournalError::Io(error)),
        }

        if crc32(&payload) != expected_checksum {
            return Err(JournalError::CorruptFrame {
                offset,
                reason: "CRC-32 checksum mismatch".to_owned(),
            });
        }

        let record = decode_record(&payload)
            .map_err(|reason| JournalError::CorruptFrame { offset, reason })?;
        ledger.append_record(record).map_err(JournalError::Ledger)?;

        offset = offset
            .checked_add(HEADER_SIZE as u64)
            .and_then(|value| value.checked_add(payload_length as u64))
            .ok_or_else(|| JournalError::CorruptFrame {
                offset,
                reason: "journal offset overflow".to_owned(),
            })?;
    }

    Ok(ledger)
}

fn truncate_incomplete_tail(file: &mut File, valid_length: u64) -> Result<(), JournalError> {
    file.set_len(valid_length)?;
    file.sync_all()?;
    Ok(())
}

fn encode_frame(record: &ExperienceRecord) -> Result<Vec<u8>, JournalError> {
    let mut encoder = Encoder::default();
    encoder.put_u64(record.sequence);
    encoder.put_string(&record.episode_id)?;
    encoder.put_event(&record.event, 0)?;

    let payload = encoder.bytes;
    if payload.len() > MAX_FRAME_PAYLOAD_BYTES {
        return Err(JournalError::FrameTooLarge {
            size: payload.len(),
            maximum: MAX_FRAME_PAYLOAD_BYTES,
        });
    }

    let payload_length = u32::try_from(payload.len()).map_err(|_| JournalError::FrameTooLarge {
        size: payload.len(),
        maximum: MAX_FRAME_PAYLOAD_BYTES,
    })?;
    let mut frame = Vec::with_capacity(HEADER_SIZE + payload.len());
    frame.extend_from_slice(&FRAME_MAGIC);
    frame.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    frame.extend_from_slice(&payload_length.to_le_bytes());
    frame.extend_from_slice(&crc32(&payload).to_le_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

fn decode_record(payload: &[u8]) -> Result<ExperienceRecord, String> {
    let mut decoder = Decoder::new(payload);
    let sequence = decoder.read_u64()?;
    let episode_id = decoder.read_string()?;
    let event = decoder.read_event(0)?;
    if decoder.remaining() != 0 {
        return Err("record contains trailing payload bytes".to_owned());
    }

    Ok(ExperienceRecord {
        sequence,
        episode_id,
        event,
    })
}

#[derive(Default)]
struct Encoder {
    bytes: Vec<u8>,
    encoded_collection_items: usize,
}

impl Encoder {
    fn put_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn put_bool(&mut self, value: bool) {
        self.put_u8(u8::from(value));
    }

    fn put_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_f64(&mut self, value: f64) -> Result<(), JournalError> {
        if !value.is_finite() {
            return Err(JournalError::InvalidNumber);
        }
        self.put_u64(value.to_bits());
        Ok(())
    }

    fn put_length(&mut self, length: usize) -> Result<(), JournalError> {
        let length = u32::try_from(length).map_err(|_| JournalError::FrameTooLarge {
            size: length,
            maximum: MAX_FRAME_PAYLOAD_BYTES,
        })?;
        self.put_u32(length);
        Ok(())
    }

    fn put_count(&mut self, count: usize) -> Result<(), JournalError> {
        let total = self.encoded_collection_items.checked_add(count).ok_or(
            JournalError::CollectionTooLarge {
                size: usize::MAX,
                maximum: MAX_COLLECTION_ITEMS,
            },
        )?;
        if count > MAX_COLLECTION_ITEMS || total > MAX_COLLECTION_ITEMS {
            return Err(JournalError::CollectionTooLarge {
                size: total.max(count),
                maximum: MAX_COLLECTION_ITEMS,
            });
        }
        self.encoded_collection_items = total;
        self.put_length(count)
    }

    fn put_string(&mut self, value: &str) -> Result<(), JournalError> {
        let projected_size = self
            .bytes
            .len()
            .checked_add(4)
            .and_then(|size| size.checked_add(value.len()))
            .ok_or(JournalError::FrameTooLarge {
                size: usize::MAX,
                maximum: MAX_FRAME_PAYLOAD_BYTES,
            })?;
        if projected_size > MAX_FRAME_PAYLOAD_BYTES {
            return Err(JournalError::FrameTooLarge {
                size: projected_size,
                maximum: MAX_FRAME_PAYLOAD_BYTES,
            });
        }
        self.put_length(value.len())?;
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn put_observation(
        &mut self,
        value: &ObservationFrame,
        depth: usize,
    ) -> Result<(), JournalError> {
        self.put_string(&value.environment_id)?;
        self.put_string(&value.session_id)?;
        self.put_string(&value.episode_id)?;
        self.put_u64(value.sequence);
        self.put_u64(value.observed_at_millis);
        self.put_string(&value.schema_id)?;
        self.put_u64(value.schema_version);
        self.put_value(&value.payload, depth + 1)
    }

    fn put_event(&mut self, value: &ExperienceEvent, depth: usize) -> Result<(), JournalError> {
        match value {
            ExperienceEvent::Observation(observation) => {
                self.put_u8(0);
                self.put_observation(observation, depth)?;
            }
            ExperienceEvent::ActionProposed {
                action_id,
                arguments,
            } => {
                self.put_u8(1);
                self.put_string(action_id)?;
                self.put_value(arguments, depth + 1)?;
            }
            ExperienceEvent::ActionCompleted(outcome) => {
                self.put_u8(2);
                self.put_string(&outcome.action_id)?;
                self.put_bool(outcome.completed);
                self.put_count(outcome.observations.len())?;
                for observation in &outcome.observations {
                    self.put_observation(observation, depth)?;
                }
                self.put_value(&outcome.feedback, depth + 1)?;
            }
            ExperienceEvent::PredictionEvaluated {
                action_id,
                predicted,
                observed,
            } => {
                self.put_u8(3);
                self.put_string(action_id)?;
                self.put_value(predicted, depth + 1)?;
                self.put_value(observed, depth + 1)?;
            }
        }
        Ok(())
    }

    fn put_value(&mut self, value: &DataValue, depth: usize) -> Result<(), JournalError> {
        if depth > MAX_VALUE_NESTING {
            return Err(JournalError::NestingLimitExceeded);
        }

        match value {
            DataValue::Null => self.put_u8(0),
            DataValue::Bool(value) => {
                self.put_u8(1);
                self.put_bool(*value);
            }
            DataValue::Integer(value) => {
                self.put_u8(2);
                self.put_i64(*value);
            }
            DataValue::Number(value) => {
                self.put_u8(3);
                self.put_f64(*value)?;
            }
            DataValue::Text(value) => {
                self.put_u8(4);
                self.put_string(value)?;
            }
            DataValue::Sequence(values) => {
                self.put_u8(5);
                self.put_count(values.len())?;
                for value in values {
                    self.put_value(value, depth + 1)?;
                }
            }
            DataValue::Object(values) => {
                self.put_u8(6);
                self.put_count(values.len())?;
                for (key, value) in values {
                    self.put_string(key)?;
                    self.put_value(value, depth + 1)?;
                }
            }
            DataValue::BlobRef(value) => {
                self.put_u8(7);
                self.put_string(value)?;
            }
        }
        Ok(())
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
    decoded_collection_items: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            offset: 0,
            decoded_collection_items: 0,
        }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| "payload offset overflow".to_owned())?;
        if end > self.bytes.len() {
            return Err("unexpected end of record payload".to_owned());
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn read_bool(&mut self) -> Result<bool, String> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err("invalid boolean tag".to_owned()),
        }
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        let bytes = self.take(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_count(&mut self, minimum_item_bytes: usize) -> Result<usize, String> {
        let count = self.read_u32()? as usize;
        if count > MAX_COLLECTION_ITEMS {
            return Err("collection length exceeds configured item limit".to_owned());
        }
        let maximum_count = self.remaining() / minimum_item_bytes;
        if count > maximum_count {
            return Err("collection length exceeds remaining payload".to_owned());
        }
        let total = self
            .decoded_collection_items
            .checked_add(count)
            .ok_or_else(|| "decoded collection item count overflow".to_owned())?;
        if total > MAX_COLLECTION_ITEMS {
            return Err("record exceeds configured total collection item limit".to_owned());
        }
        self.decoded_collection_items = total;
        Ok(count)
    }

    fn read_string(&mut self) -> Result<String, String> {
        let length = self.read_u32()? as usize;
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| "string is not valid UTF-8".to_owned())
    }

    fn read_observation(&mut self, depth: usize) -> Result<ObservationFrame, String> {
        Ok(ObservationFrame {
            environment_id: self.read_string()?,
            session_id: self.read_string()?,
            episode_id: self.read_string()?,
            sequence: self.read_u64()?,
            observed_at_millis: self.read_u64()?,
            schema_id: self.read_string()?,
            schema_version: self.read_u64()?,
            payload: self.read_value(depth + 1)?,
        })
    }

    fn read_event(&mut self, depth: usize) -> Result<ExperienceEvent, String> {
        match self.read_u8()? {
            0 => Ok(ExperienceEvent::Observation(self.read_observation(depth)?)),
            1 => Ok(ExperienceEvent::ActionProposed {
                action_id: self.read_string()?,
                arguments: self.read_value(depth + 1)?,
            }),
            2 => {
                let action_id = self.read_string()?;
                let completed = self.read_bool()?;
                let count = self.read_count(41)?;
                let mut observations = Vec::with_capacity(count);
                for _ in 0..count {
                    observations.push(self.read_observation(depth)?);
                }
                let feedback = self.read_value(depth + 1)?;
                Ok(ExperienceEvent::ActionCompleted(ActionOutcome {
                    action_id,
                    completed,
                    observations,
                    feedback,
                }))
            }
            3 => Ok(ExperienceEvent::PredictionEvaluated {
                action_id: self.read_string()?,
                predicted: self.read_value(depth + 1)?,
                observed: self.read_value(depth + 1)?,
            }),
            _ => Err("unknown experience event tag".to_owned()),
        }
    }

    fn read_value(&mut self, depth: usize) -> Result<DataValue, String> {
        if depth > MAX_VALUE_NESTING {
            return Err("value nesting limit exceeded".to_owned());
        }

        match self.read_u8()? {
            0 => Ok(DataValue::Null),
            1 => Ok(DataValue::Bool(self.read_bool()?)),
            2 => Ok(DataValue::Integer(self.read_i64()?)),
            3 => {
                let value = f64::from_bits(self.read_u64()?);
                if !value.is_finite() {
                    return Err("non-finite number in record".to_owned());
                }
                Ok(DataValue::Number(value))
            }
            4 => Ok(DataValue::Text(self.read_string()?)),
            5 => {
                let count = self.read_count(1)?;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.read_value(depth + 1)?);
                }
                Ok(DataValue::Sequence(values))
            }
            6 => {
                let count = self.read_count(5)?;
                let mut values = BTreeMap::new();
                for _ in 0..count {
                    let key = self.read_string()?;
                    let value = self.read_value(depth + 1)?;
                    if values.insert(key, value).is_some() {
                        return Err("object contains a duplicate key".to_owned());
                    }
                }
                Ok(DataValue::Object(values))
            }
            7 => Ok(DataValue::BlobRef(self.read_string()?)),
            _ => Err("unknown data value tag".to_owned()),
        }
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[derive(Debug)]
pub enum JournalError {
    Io(io::Error),
    Ledger(LedgerError),
    UnsupportedVersion(u16),
    FrameTooLarge { size: usize, maximum: usize },
    CorruptFrame { offset: u64, reason: String },
    InvalidNumber,
    CollectionTooLarge { size: usize, maximum: usize },
    NestingLimitExceeded,
    Poisoned,
}

impl std::fmt::Display for JournalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "journal I/O error: {error}"),
            Self::Ledger(error) => write!(formatter, "journal ledger error: {error}"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported journal format version: {version}")
            }
            Self::FrameTooLarge { size, maximum } => {
                write!(
                    formatter,
                    "journal frame size {size} exceeds limit {maximum}"
                )
            }
            Self::CorruptFrame { offset, reason } => {
                write!(
                    formatter,
                    "corrupt journal frame at byte {offset}: {reason}"
                )
            }
            Self::InvalidNumber => write!(formatter, "journal cannot encode a non-finite number"),
            Self::CollectionTooLarge { size, maximum } => write!(
                formatter,
                "journal collection size {size} exceeds limit {maximum}"
            ),
            Self::NestingLimitExceeded => {
                write!(formatter, "journal value nesting limit exceeded")
            }
            Self::Poisoned => write!(
                formatter,
                "journal is unusable after a write failure; reopen it to recover"
            ),
        }
    }
}

impl std::error::Error for JournalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Ledger(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for JournalError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

    fn test_path() -> std::path::PathBuf {
        let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "5413-experience-journal-{}-{id}.bin",
            std::process::id()
        ))
    }

    fn sample_observation(episode_id: &str) -> ObservationFrame {
        ObservationFrame {
            environment_id: "test-world".to_owned(),
            session_id: "session-1".to_owned(),
            episode_id: episode_id.to_owned(),
            sequence: 9,
            observed_at_millis: 1234,
            schema_id: "state".to_owned(),
            schema_version: 2,
            payload: DataValue::object([
                (
                    "nested".to_owned(),
                    DataValue::Sequence(vec![
                        DataValue::Integer(-7),
                        DataValue::Number(0.25),
                        DataValue::BlobRef("blob-1".to_owned()),
                    ]),
                ),
                ("ready".to_owned(), DataValue::Bool(true)),
            ]),
        }
    }

    #[test]
    fn all_event_variants_round_trip_after_reopen() {
        let path = test_path();
        let expected = vec![
            ExperienceRecord {
                sequence: 0,
                episode_id: "episode-a".to_owned(),
                event: ExperienceEvent::Observation(sample_observation("episode-a")),
            },
            ExperienceRecord {
                sequence: 1,
                episode_id: "episode-b".to_owned(),
                event: ExperienceEvent::ActionProposed {
                    action_id: "move".to_owned(),
                    arguments: DataValue::object([(
                        "delta".to_owned(),
                        DataValue::Sequence(vec![DataValue::Number(1.5), DataValue::Integer(2)]),
                    )]),
                },
            },
            ExperienceRecord {
                sequence: 2,
                episode_id: "episode-a".to_owned(),
                event: ExperienceEvent::ActionCompleted(ActionOutcome {
                    action_id: "move".to_owned(),
                    completed: true,
                    observations: vec![sample_observation("episode-a")],
                    feedback: DataValue::Text("ok".to_owned()),
                }),
            },
            ExperienceRecord {
                sequence: 3,
                episode_id: "episode-a".to_owned(),
                event: ExperienceEvent::PredictionEvaluated {
                    action_id: "move".to_owned(),
                    predicted: DataValue::Number(2.0),
                    observed: DataValue::Number(2.25),
                },
            },
        ];

        {
            let mut journal = ExperienceJournal::open(&path).expect("journal should open");
            for record in &expected {
                let sequence = journal
                    .append(record.episode_id.clone(), record.event.clone())
                    .expect("append should succeed");
                assert_eq!(sequence, record.sequence);
            }
        }

        {
            let journal = ExperienceJournal::open(&path).expect("journal should reopen");
            assert_eq!(journal.records(), expected.as_slice());
            let episode_sequences: Vec<_> = journal
                .replay_episode("episode-a")
                .map(|record| record.sequence)
                .collect();
            assert_eq!(episode_sequences, vec![0, 2, 3]);
        }

        fs::remove_file(path).expect("test journal should be removed");
    }

    #[test]
    fn incomplete_final_frame_is_truncated_and_next_append_reuses_sequence() {
        let path = test_path();
        {
            let mut journal = ExperienceJournal::open(&path).expect("journal should open");
            journal
                .append(
                    "episode-a",
                    ExperienceEvent::PredictionEvaluated {
                        action_id: "a".to_owned(),
                        predicted: DataValue::Integer(1),
                        observed: DataValue::Integer(1),
                    },
                )
                .expect("first append should succeed");
            journal
                .append(
                    "episode-a",
                    ExperienceEvent::PredictionEvaluated {
                        action_id: "b".to_owned(),
                        predicted: DataValue::Integer(2),
                        observed: DataValue::Integer(2),
                    },
                )
                .expect("second append should succeed");
        }

        let original_length = fs::metadata(&path).expect("metadata should exist").len();
        let file = OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("journal should open for truncation");
        file.set_len(original_length - 3)
            .expect("tail should be truncated");
        drop(file);

        {
            let mut journal = ExperienceJournal::open(&path).expect("torn tail should recover");
            assert_eq!(journal.len(), 1);
            let sequence = journal
                .append(
                    "episode-a",
                    ExperienceEvent::PredictionEvaluated {
                        action_id: "recovered".to_owned(),
                        predicted: DataValue::Integer(3),
                        observed: DataValue::Integer(3),
                    },
                )
                .expect("append after recovery should succeed");
            assert_eq!(sequence, 1);
        }

        let journal = ExperienceJournal::open(&path).expect("recovered journal should reopen");
        assert_eq!(journal.records().len(), 2);
        assert_eq!(journal.records()[1].sequence, 1);
        drop(journal);
        fs::remove_file(path).expect("test journal should be removed");
    }

    #[test]
    fn complete_frame_checksum_corruption_is_not_silently_recovered() {
        let path = test_path();
        {
            let mut journal = ExperienceJournal::open(&path).expect("journal should open");
            journal
                .append(
                    "episode-a",
                    ExperienceEvent::PredictionEvaluated {
                        action_id: "a".to_owned(),
                        predicted: DataValue::Integer(1),
                        observed: DataValue::Integer(1),
                    },
                )
                .expect("append should succeed");
        }

        let mut bytes = fs::read(&path).expect("journal should be readable");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        fs::write(&path, bytes).expect("corruption should be written");

        let result = ExperienceJournal::open(&path);
        assert!(matches!(result, Err(JournalError::CorruptFrame { .. })));
        fs::remove_file(path).expect("test journal should be removed");
    }

    #[test]
    fn unsupported_format_versions_fail_closed() {
        let path = test_path();
        {
            let mut journal = ExperienceJournal::open(&path).expect("journal should open");
            journal
                .append(
                    "episode-a",
                    ExperienceEvent::PredictionEvaluated {
                        action_id: "a".to_owned(),
                        predicted: DataValue::Integer(1),
                        observed: DataValue::Integer(1),
                    },
                )
                .expect("append should succeed");
        }

        let mut bytes = fs::read(&path).expect("journal should be readable");
        bytes[4] = 2;
        fs::write(&path, bytes).expect("version should be changed");

        assert!(matches!(
            ExperienceJournal::open(&path),
            Err(JournalError::UnsupportedVersion(2))
        ));
        fs::remove_file(path).expect("test journal should be removed");
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn rejects_non_finite_values_before_writing() {
        let path = test_path();
        let mut journal = ExperienceJournal::open(&path).expect("journal should open");
        let result = journal.append(
            "episode-a",
            ExperienceEvent::PredictionEvaluated {
                action_id: "a".to_owned(),
                predicted: DataValue::Number(f64::NAN),
                observed: DataValue::Null,
            },
        );
        assert!(matches!(result, Err(JournalError::InvalidNumber)));
        assert!(journal.is_empty());
        drop(journal);
        fs::remove_file(path).expect("test journal should be removed");
    }

    #[test]
    fn stress_round_trips_many_records_across_repeated_reopens() {
        let path = test_path();
        let mut expected = Vec::with_capacity(1_200);
        let mut journal = ExperienceJournal::open(&path).expect("journal should open");

        for index in 0..1_200_u64 {
            let episode_id = format!("episode-{}", index % 23);
            let event = ExperienceEvent::PredictionEvaluated {
                action_id: format!("operator-{}", index % 11),
                predicted: DataValue::object([
                    ("index".to_owned(), DataValue::Integer(index as i64)),
                    (
                        "features".to_owned(),
                        DataValue::Sequence(
                            (0..8)
                                .map(|feature| {
                                    DataValue::Number(index as f64 + feature as f64 / 10.0)
                                })
                                .collect(),
                        ),
                    ),
                ]),
                observed: DataValue::object([
                    ("accepted".to_owned(), DataValue::Bool(index % 3 == 0)),
                    (
                        "samples".to_owned(),
                        DataValue::Sequence(
                            (0..8)
                                .map(|sample| DataValue::Integer((index + sample) as i64))
                                .collect(),
                        ),
                    ),
                ]),
            };

            let sequence = journal
                .append(episode_id.clone(), event.clone())
                .expect("stress append should succeed");
            assert_eq!(sequence, index);
            expected.push(ExperienceRecord {
                sequence,
                episode_id,
                event,
            });

            if (index + 1) % 173 == 0 {
                drop(journal);
                journal =
                    ExperienceJournal::open(&path).expect("journal should survive repeated reopen");
                assert_eq!(journal.records(), expected.as_slice());
            }
        }

        drop(journal);
        let journal = ExperienceJournal::open(&path).expect("final reopen should succeed");
        assert_eq!(journal.records(), expected.as_slice());
        assert_eq!(journal.len(), 1_200);

        for episode_number in 0..23 {
            let episode_id = format!("episode-{episode_number}");
            let expected_episode: Vec<_> = expected
                .iter()
                .filter(|record| record.episode_id == episode_id)
                .collect();
            let replayed: Vec<_> = journal.replay_episode(&episode_id).collect();
            assert_eq!(replayed, expected_episode);
        }

        drop(journal);
        fs::remove_file(path).expect("stress journal should be removed");
    }

    #[test]
    fn stress_recovers_every_incomplete_final_frame_prefix() {
        let path = test_path();
        {
            let mut journal = ExperienceJournal::open(&path).expect("journal should open");
            journal
                .append(
                    "episode-base",
                    ExperienceEvent::PredictionEvaluated {
                        action_id: "base".to_owned(),
                        predicted: DataValue::Integer(1),
                        observed: DataValue::Integer(1),
                    },
                )
                .expect("base append should succeed");
        }
        let base_bytes = fs::read(&path).expect("base journal should be readable");

        {
            let mut journal = ExperienceJournal::open(&path).expect("base journal should reopen");
            journal
                .append(
                    "episode-tail",
                    ExperienceEvent::PredictionEvaluated {
                        action_id: "tail".to_owned(),
                        predicted: DataValue::Integer(2),
                        observed: DataValue::Integer(2),
                    },
                )
                .expect("tail append should succeed");
        }
        let complete_bytes = fs::read(&path).expect("complete journal should be readable");
        assert!(complete_bytes.len() > base_bytes.len());

        for cut in base_bytes.len()..complete_bytes.len() {
            let truncated_path = test_path();
            fs::write(&truncated_path, &complete_bytes[..cut])
                .expect("truncated prefix should be written");

            let recovered =
                ExperienceJournal::open(&truncated_path).expect("incomplete tail should recover");
            assert_eq!(recovered.len(), 1, "unexpected record count at byte {cut}");
            assert_eq!(recovered.records()[0].episode_id, "episode-base");
            drop(recovered);

            let recovered_bytes =
                fs::read(&truncated_path).expect("recovered journal should be readable");
            assert_eq!(
                recovered_bytes.len() as u64,
                base_bytes.len() as u64,
                "wrong truncation boundary at byte {cut}"
            );
            fs::remove_file(truncated_path).expect("truncated journal should be removed");
        }

        fs::remove_file(path).expect("source journal should be removed");
    }

    #[test]
    fn decoder_rejects_a_single_collection_over_the_configured_limit() {
        let mut encoder = Encoder::default();
        encoder.put_u64(0);
        encoder.put_string("").expect("empty string should encode");
        encoder.put_u8(1);
        encoder
            .put_string("")
            .expect("empty action id should encode");
        encoder.put_u8(5);
        encoder.put_u32((MAX_COLLECTION_ITEMS as u32) + 1);

        let error = decode_record(&encoder.bytes).expect_err("oversized count must be rejected");
        assert!(error.contains("configured item limit"));
    }

    #[test]
    fn decoder_rejects_many_nested_collections_over_the_aggregate_limit() {
        let mut encoder = Encoder::default();
        encoder.put_u64(0);
        encoder.put_string("").expect("empty string should encode");
        encoder.put_u8(1);
        encoder
            .put_string("")
            .expect("empty action id should encode");
        encoder.put_u8(5);
        encoder.put_u32(50_000);

        for _ in 0..50_000 {
            encoder.put_u8(5);
            encoder.put_u32(2);
            encoder.put_u8(0);
            encoder.put_u8(0);
        }

        let error = decode_record(&encoder.bytes)
            .expect_err("aggregate nested item count must be rejected");
        assert!(error.contains("total collection item limit"));
    }

    #[test]
    fn rejects_oversized_string_before_copying_into_the_frame_buffer() {
        let path = test_path();
        let mut journal = ExperienceJournal::open(&path).expect("journal should open");
        let oversized_episode_id = "x".repeat(MAX_FRAME_PAYLOAD_BYTES);

        let result = journal.append(
            oversized_episode_id,
            ExperienceEvent::PredictionEvaluated {
                action_id: "a".to_owned(),
                predicted: DataValue::Null,
                observed: DataValue::Null,
            },
        );

        assert!(matches!(result, Err(JournalError::FrameTooLarge { .. })));
        assert!(journal.is_empty());
        drop(journal);
        assert_eq!(
            fs::metadata(&path)
                .expect("journal file should exist")
                .len(),
            0
        );
        fs::remove_file(path).expect("test journal should be removed");
    }

    #[test]
    fn encoder_rejects_aggregate_collection_limit_before_writing_record() {
        let mut encoder = Encoder::default();
        encoder
            .put_count(60_000)
            .expect("first collection should fit the budget");
        let result = encoder.put_count(40_001);
        assert!(matches!(
            result,
            Err(JournalError::CollectionTooLarge {
                maximum: MAX_COLLECTION_ITEMS,
                ..
            })
        ));
    }
}
