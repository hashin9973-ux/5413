//! Immutable, append-only persistence for learned predictive-model artifacts.
//!
//! References are caller-supplied stable identifiers, not hashes. CRC-32 detects
//! accidental corruption; it does not provide authenticity or tamper resistance.

use crate::experience_journal::{
    crc32, decode_data_value, encode_data_value, MAX_FRAME_PAYLOAD_BYTES,
};
use crate::value::DataValue;
use crate::world_model::{restore_predictive_model, PredictiveModel, WorldModelError};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: [u8; 8] = *b"5413MART";
const FORMAT_VERSION: u16 = 1;
const HEADER_SIZE: usize = 18;

pub struct ModelArtifactStore {
    file: File,
    artifacts: BTreeMap<String, DataValue>,
    poisoned: bool,
}

impl ModelArtifactStore {
    /// Opens or creates the single-writer artifact log and rebuilds its index.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ModelArtifactError> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        let artifacts = scan_and_recover(&mut file)?;
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            artifacts,
            poisoned: false,
        })
    }

    /// Persists an immutable artifact under a stable, caller-supplied reference.
    /// Identical retries are idempotent; conflicting reuse is rejected.
    pub fn put(
        &mut self,
        artifact_ref: impl Into<String>,
        artifact: DataValue,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        if self.poisoned {
            return Err(ModelArtifactError::Poisoned);
        }
        let artifact_ref = artifact_ref.into();
        if artifact_ref.trim().is_empty() {
            return Err(ModelArtifactError::InvalidReference);
        }
        artifact
            .validate_finite()
            .map_err(|_| ModelArtifactError::InvalidArtifact)?;

        if let Some(existing) = self.artifacts.get(&artifact_ref) {
            return if existing == &artifact {
                Ok(ArtifactPutOutcome::AlreadyPresent)
            } else {
                Err(ModelArtifactError::ReferenceConflict)
            };
        }

        let payload = encode_data_value(&DataValue::object([
            ("reference".to_owned(), DataValue::Text(artifact_ref.clone())),
            ("artifact".to_owned(), artifact.clone()),
        ]))
        .map_err(|error| ModelArtifactError::Encoding(error.to_string()))?;
        let payload_length =
            u32::try_from(payload.len()).map_err(|_| ModelArtifactError::FrameTooLarge {
                size: payload.len(),
                maximum: MAX_FRAME_PAYLOAD_BYTES,
            })?;
        let mut frame = Vec::with_capacity(HEADER_SIZE + payload.len());
        frame.extend_from_slice(&MAGIC);
        frame.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        frame.extend_from_slice(&payload_length.to_le_bytes());
        frame.extend_from_slice(&crc32(&payload).to_le_bytes());
        frame.extend_from_slice(&payload);

        if let Err(error) = self
            .file
            .seek(SeekFrom::End(0))
            .and_then(|_| self.file.write_all(&frame))
            .and_then(|_| self.file.sync_all())
        {
            self.poisoned = true;
            return Err(ModelArtifactError::Io(error));
        }
        self.artifacts.insert(artifact_ref, artifact);
        Ok(ArtifactPutOutcome::Stored)
    }

    /// Serializes and persists a fitted built-in model, including its parameters.
    pub fn put_model(
        &mut self,
        artifact_ref: impl Into<String>,
        model: &dyn PredictiveModel,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        let artifact = model
            .to_artifact_value()
            .map_err(ModelArtifactError::Model)?;
        self.put(artifact_ref, artifact)
    }

    pub fn get(&self, artifact_ref: &str) -> Option<&DataValue> {
        self.artifacts.get(artifact_ref)
    }

    /// Reconstructs a supported built-in model from its persisted artifact.
    pub fn load_model(
        &self,
        artifact_ref: &str,
    ) -> Result<Box<dyn PredictiveModel>, ModelArtifactError> {
        let artifact = self
            .artifacts
            .get(artifact_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        restore_predictive_model(artifact).map_err(ModelArtifactError::Model)
    }

    pub fn references(&self) -> impl Iterator<Item = &str> {
        self.artifacts.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.artifacts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.artifacts.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactPutOutcome {
    Stored,
    AlreadyPresent,
}

fn scan_and_recover(file: &mut File) -> Result<BTreeMap<String, DataValue>, ModelArtifactError> {
    file.seek(SeekFrom::Start(0))?;
    let mut artifacts = BTreeMap::new();
    let mut offset = 0_u64;

    loop {
        let mut header = [0_u8; HEADER_SIZE];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                truncate_incomplete_tail(file, offset)?;
                break;
            }
            Err(error) => return Err(ModelArtifactError::Io(error)),
        }

        if header[..8] != MAGIC {
            return Err(ModelArtifactError::CorruptFrame {
                offset,
                reason: "invalid model-artifact frame marker".to_owned(),
            });
        }
        let version = u16::from_le_bytes([header[8], header[9]]);
        if version != FORMAT_VERSION {
            return Err(ModelArtifactError::UnsupportedVersion(version));
        }
        let payload_length =
            u32::from_le_bytes([header[10], header[11], header[12], header[13]]) as usize;
        if payload_length > MAX_FRAME_PAYLOAD_BYTES {
            return Err(ModelArtifactError::FrameTooLarge {
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
            Err(error) => return Err(ModelArtifactError::Io(error)),
        }
        if crc32(&payload) != expected_checksum {
            return Err(ModelArtifactError::CorruptFrame {
                offset,
                reason: "CRC-32 checksum mismatch".to_owned(),
            });
        }

        let value =
            decode_data_value(&payload).map_err(|error| ModelArtifactError::CorruptFrame {
                offset,
                reason: error.to_string(),
            })?;
        let mut fields = match value {
            DataValue::Object(fields) if fields.len() == 2 => fields,
            _ => {
                return Err(ModelArtifactError::CorruptFrame {
                    offset,
                    reason: "artifact envelope must contain exactly two fields".to_owned(),
                })
            }
        };
        let artifact_ref = match fields.remove("reference") {
            Some(DataValue::Text(reference)) if !reference.trim().is_empty() => reference,
            _ => {
                return Err(ModelArtifactError::CorruptFrame {
                    offset,
                    reason: "artifact reference is missing or invalid".to_owned(),
                })
            }
        };
        let artifact = fields
            .remove("artifact")
            .ok_or_else(|| ModelArtifactError::CorruptFrame {
                offset,
                reason: "artifact payload is missing".to_owned(),
            })?;
        artifact
            .validate_finite()
            .map_err(|_| ModelArtifactError::CorruptFrame {
                offset,
                reason: "artifact contains non-finite data".to_owned(),
            })?;
        if artifacts.insert(artifact_ref, artifact).is_some() {
            return Err(ModelArtifactError::CorruptFrame {
                offset,
                reason: "duplicate immutable artifact reference".to_owned(),
            });
        }

        offset = offset
            .checked_add(HEADER_SIZE as u64)
            .and_then(|value| value.checked_add(payload_length as u64))
            .ok_or_else(|| ModelArtifactError::CorruptFrame {
                offset,
                reason: "artifact log offset overflow".to_owned(),
            })?;
    }

    Ok(artifacts)
}

fn truncate_incomplete_tail(file: &mut File, valid_length: u64) -> Result<(), ModelArtifactError> {
    file.set_len(valid_length)?;
    file.sync_all()?;
    Ok(())
}

#[derive(Debug)]
pub enum ModelArtifactError {
    Io(io::Error),
    Encoding(String),
    Model(WorldModelError),
    InvalidReference,
    InvalidArtifact,
    NotFound,
    ReferenceConflict,
    UnsupportedVersion(u16),
    FrameTooLarge { size: usize, maximum: usize },
    CorruptFrame { offset: u64, reason: String },
    Poisoned,
}

impl std::fmt::Display for ModelArtifactError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "model-artifact I/O error: {error}"),
            Self::Encoding(message) => write!(formatter, "model-artifact encoding failed: {message}"),
            Self::Model(error) => write!(formatter, "model-artifact conversion failed: {error}"),
            Self::InvalidReference => write!(formatter, "artifact reference must not be empty"),
            Self::InvalidArtifact => write!(formatter, "artifact data is invalid"),
            Self::NotFound => write!(formatter, "model artifact was not found"),
            Self::ReferenceConflict => write!(formatter, "immutable artifact reference already has different content"),
            Self::UnsupportedVersion(version) => write!(formatter, "unsupported artifact-store version: {version}"),
            Self::FrameTooLarge { size, maximum } => write!(formatter, "artifact frame size {size} exceeds limit {maximum}"),
            Self::CorruptFrame { offset, reason } => write!(formatter, "corrupt model-artifact frame at byte {offset}: {reason}"),
            Self::Poisoned => write!(formatter, "model-artifact store must be reopened after an I/O failure"),
        }
    }
}

impl std::error::Error for ModelArtifactError {}

impl From<io::Error> for ModelArtifactError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::TransitionSample;
    use crate::world_model::{ActionDeltaModel, ContextualLinearModel, PredictiveModel};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

    fn test_path() -> std::path::PathBuf {
        let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("5413-model-artifacts-{}-{id}.bin", std::process::id()))
    }

    fn sample(action: &str, before: f64, after: f64) -> TransitionSample {
        TransitionSample {
            action_id: action.to_owned(),
            before: DataValue::object([("position".to_owned(), DataValue::Number(before))]),
            after: DataValue::object([("position".to_owned(), DataValue::Number(after))]),
        }
    }

    #[test]
    fn persists_and_restores_statistical_model_predictions() {
        let path = test_path();
        let mut model = ActionDeltaModel::new("delta", 2).expect("model");
        model
            .fit("training-a", &[sample("advance", 0.0, 2.0), sample("advance", 4.0, 6.0)])
            .expect("fit");
        let state = DataValue::object([("position".to_owned(), DataValue::Number(10.0))]);
        let expected = model.predict("advance", &state).expect("prediction");
        {
            let mut store = ModelArtifactStore::open(&path).expect("open");
            assert_eq!(
                store.put_model("model-artifact-1", &model).expect("put"),
                ArtifactPutOutcome::Stored
            );
            assert_eq!(
                store.put_model("model-artifact-1", &model).expect("duplicate"),
                ArtifactPutOutcome::AlreadyPresent
            );
        }
        {
            let store = ModelArtifactStore::open(&path).expect("reopen");
            let restored = store.load_model("model-artifact-1").expect("restore");
            assert_eq!(restored.identity().id, "delta");
            assert_eq!(restored.identity().version, 2);
            assert_eq!(restored.predict("advance", &state).expect("predict"), expected);
        }
        std::fs::remove_file(path).expect("remove artifact file");
    }

    #[test]
    fn persists_and_restores_contextual_model_coefficients() {
        let path = test_path();
        let samples = vec![
            TransitionSample {
                action_id: "move".to_owned(),
                before: DataValue::object([
                    ("position".to_owned(), DataValue::Number(0.0)),
                    ("velocity".to_owned(), DataValue::Number(1.0)),
                ]),
                after: DataValue::object([
                    ("position".to_owned(), DataValue::Number(1.0)),
                    ("velocity".to_owned(), DataValue::Number(1.0)),
                ]),
            },
            TransitionSample {
                action_id: "move".to_owned(),
                before: DataValue::object([
                    ("position".to_owned(), DataValue::Number(2.0)),
                    ("velocity".to_owned(), DataValue::Number(3.0)),
                ]),
                after: DataValue::object([
                    ("position".to_owned(), DataValue::Number(5.0)),
                    ("velocity".to_owned(), DataValue::Number(3.0)),
                ]),
            },
            TransitionSample {
                action_id: "move".to_owned(),
                before: DataValue::object([
                    ("position".to_owned(), DataValue::Number(8.0)),
                    ("velocity".to_owned(), DataValue::Number(2.0)),
                ]),
                after: DataValue::object([
                    ("position".to_owned(), DataValue::Number(10.0)),
                    ("velocity".to_owned(), DataValue::Number(2.0)),
                ]),
            },
            TransitionSample {
                action_id: "move".to_owned(),
                before: DataValue::object([
                    ("position".to_owned(), DataValue::Number(10.0)),
                    ("velocity".to_owned(), DataValue::Number(4.0)),
                ]),
                after: DataValue::object([
                    ("position".to_owned(), DataValue::Number(14.0)),
                    ("velocity".to_owned(), DataValue::Number(4.0)),
                ]),
            },
        ];
        let mut model = ContextualLinearModel::new("context", 1, 1e-6).expect("model");
        model.fit("training", &samples).expect("fit");
        let state = DataValue::object([
            ("position".to_owned(), DataValue::Number(20.0)),
            ("velocity".to_owned(), DataValue::Number(5.0)),
        ]);
        let expected = model.predict("move", &state).expect("prediction");
        {
            let mut store = ModelArtifactStore::open(&path).expect("open");
            store.put_model("context-artifact", &model).expect("put");
        }
        {
            let store = ModelArtifactStore::open(&path).expect("reopen");
            let restored = store.load_model("context-artifact").expect("restore");
            assert_eq!(restored.predict("move", &state).expect("predict"), expected);
        }
        std::fs::remove_file(path).expect("remove artifact file");
    }

    #[test]
    fn rejects_conflicting_reference_and_recovers_torn_tail() {
        let path = test_path();
        let artifact = DataValue::object([("opaque".to_owned(), DataValue::Text("value".to_owned()))]);
        let first_length;
        {
            let mut store = ModelArtifactStore::open(&path).expect("open");
            store.put("opaque-v1", artifact.clone()).expect("put");
            first_length = store.file.metadata().expect("metadata").len();
            assert!(matches!(
                store.put("opaque-v1", DataValue::Null),
                Err(ModelArtifactError::ReferenceConflict)
            ));
            store.put("opaque-v2", DataValue::Integer(2)).expect("put second");
        }
        {
            let mut file = OpenOptions::new().write(true).open(&path).expect("open raw");
            file.set_len(first_length + 2).expect("truncate tail");
            file.sync_all().expect("sync");
        }
        {
            let store = ModelArtifactStore::open(&path).expect("recover");
            assert_eq!(store.len(), 1);
            assert_eq!(store.get("opaque-v1"), Some(&artifact));
        }
        std::fs::remove_file(path).expect("remove artifact file");
    }
}
