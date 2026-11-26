//! Immutable, append-only persistence for learned predictive-model artifacts.
//!
//! References are caller-supplied stable identifiers, not hashes. CRC-32 detects
//! accidental corruption; it does not provide authenticity or tamper resistance.

use crate::benchmark::{
    BenchmarkError, LearningEfficiencyBenchmark, LearningEfficiencyCurve, TransferBenchmark,
    TransferMatrix,
};
use crate::calibration::{ConformalCalibrationError, ConformalCalibrator};
use crate::causal::{CausalError, PairedRandomizedStudy};
use crate::experience_journal::{
    crc32, decode_data_value, encode_data_value, MAX_FRAME_PAYLOAD_BYTES,
};
use crate::meta_learning::{OperatorCatalogue, StrategyError, StrategyRegistry};
use crate::model_registry::{ModelCandidateSpec, ModelRegistry, ModelRegistryError};
use crate::reasoning::{HypothesisPolicy, HypothesisRegistry, ReasoningError};
use crate::representation::{RegistryError, RepresentationRegistry};
use crate::value::DataValue;
use crate::world_model::{
    restore_predictive_model, EnsembleWorldModel, PredictiveModel, WorldModelError,
};
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
            (
                "reference".to_owned(),
                DataValue::Text(artifact_ref.clone()),
            ),
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

    /// Persists a complete immutable snapshot of a representation registry.
    pub fn put_representation_registry(
        &mut self,
        artifact_ref: impl Into<String>,
        registry: &RepresentationRegistry,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        self.put(artifact_ref, registry.to_artifact_value())
    }

    /// Restores a representation registry snapshot from its immutable artifact.
    pub fn load_representation_registry(
        &self,
        artifact_ref: &str,
    ) -> Result<RepresentationRegistry, ModelArtifactError> {
        let artifact = self
            .artifacts
            .get(artifact_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        RepresentationRegistry::from_artifact_value(artifact).map_err(ModelArtifactError::Registry)
    }

    /// Persists an auditable hypothesis registry snapshot.
    pub fn put_hypothesis_registry(
        &mut self,
        artifact_ref: impl Into<String>,
        registry: &HypothesisRegistry,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        self.put(artifact_ref, registry.to_artifact_value())
    }

    pub fn load_hypothesis_registry(
        &self,
        artifact_ref: &str,
        policy: HypothesisPolicy,
    ) -> Result<HypothesisRegistry, ModelArtifactError> {
        let artifact = self
            .artifacts
            .get(artifact_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        HypothesisRegistry::from_artifact_value(artifact, policy)
            .map_err(ModelArtifactError::Reasoning)
    }

    /// Persists a typed, declarative strategy-registry snapshot.
    pub fn put_strategy_registry(
        &mut self,
        artifact_ref: impl Into<String>,
        registry: &StrategyRegistry,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        self.put(artifact_ref, registry.to_artifact_value())
    }

    /// Restores strategy metadata and validates every graph against a trusted catalogue.
    pub fn load_strategy_registry(
        &self,
        artifact_ref: &str,
        catalogue: &OperatorCatalogue,
    ) -> Result<StrategyRegistry, ModelArtifactError> {
        let artifact = self
            .artifacts
            .get(artifact_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        StrategyRegistry::from_artifact_value(artifact, catalogue)
            .map_err(ModelArtifactError::Strategy)
    }

    /// Persists an attested paired randomized study as an immutable artifact.
    pub fn put_causal_study(
        &mut self,
        artifact_ref: impl Into<String>,
        study: &PairedRandomizedStudy,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        study.validate().map_err(ModelArtifactError::Causal)?;
        self.put(artifact_ref, study.to_artifact_value())
    }

    pub fn load_causal_study(
        &self,
        artifact_ref: &str,
    ) -> Result<PairedRandomizedStudy, ModelArtifactError> {
        let artifact = self
            .artifacts
            .get(artifact_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        PairedRandomizedStudy::from_artifact_value(artifact).map_err(ModelArtifactError::Causal)
    }

    /// Persists a cross-environment transfer benchmark as an immutable artifact.
    pub fn put_transfer_benchmark(
        &mut self,
        artifact_ref: impl Into<String>,
        benchmark: &TransferBenchmark,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        self.put(artifact_ref, benchmark.to_artifact_value())
    }

    pub fn load_transfer_benchmark(
        &self,
        artifact_ref: &str,
    ) -> Result<TransferBenchmark, ModelArtifactError> {
        let artifact = self
            .artifacts
            .get(artifact_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        TransferBenchmark::from_artifact_value(artifact).map_err(ModelArtifactError::Benchmark)
    }

    /// Persists a derived source-to-target transfer matrix.
    pub fn put_transfer_matrix(
        &mut self,
        artifact_ref: impl Into<String>,
        matrix: &TransferMatrix,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        matrix.validate().map_err(ModelArtifactError::Benchmark)?;
        self.put(artifact_ref, matrix.to_artifact_value())
    }

    pub fn load_transfer_matrix(
        &self,
        artifact_ref: &str,
    ) -> Result<TransferMatrix, ModelArtifactError> {
        let artifact = self
            .artifacts
            .get(artifact_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        TransferMatrix::from_artifact_value(artifact).map_err(ModelArtifactError::Benchmark)
    }

    /// Persists a sample-efficiency comparison curve.
    pub fn put_learning_efficiency_curve(
        &mut self,
        artifact_ref: impl Into<String>,
        curve: &LearningEfficiencyCurve,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        curve.validate().map_err(ModelArtifactError::Benchmark)?;
        self.put(artifact_ref, curve.to_artifact_value())
    }

    pub fn load_learning_efficiency_curve(
        &self,
        artifact_ref: &str,
    ) -> Result<LearningEfficiencyCurve, ModelArtifactError> {
        let artifact = self
            .artifacts
            .get(artifact_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        LearningEfficiencyCurve::from_artifact_value(artifact)
            .map_err(ModelArtifactError::Benchmark)
    }

    /// Persists a repeated-seed sample-efficiency benchmark.
    pub fn put_learning_efficiency_benchmark(
        &mut self,
        artifact_ref: impl Into<String>,
        benchmark: &LearningEfficiencyBenchmark,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        self.put(artifact_ref, benchmark.to_artifact_value())
    }

    pub fn load_learning_efficiency_benchmark(
        &self,
        artifact_ref: &str,
    ) -> Result<LearningEfficiencyBenchmark, ModelArtifactError> {
        let artifact = self
            .artifacts
            .get(artifact_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        LearningEfficiencyBenchmark::from_artifact_value(artifact)
            .map_err(ModelArtifactError::Benchmark)
    }

    /// Registers only a candidate whose artifact loads and matches its declared identity.
    pub fn register_model_candidate(
        &self,
        registry: &mut ModelRegistry,
        spec: ModelCandidateSpec,
    ) -> Result<(), ModelArtifactError> {
        let model = self.load_model(&spec.model_artifact_ref)?;
        if model.identity().id != spec.id || model.identity().version != spec.version {
            return Err(ModelArtifactError::ModelIdentityMismatch);
        }
        registry
            .register_candidate(spec)
            .map_err(ModelArtifactError::ModelRegistry)
    }

    pub fn put_model_registry(
        &mut self,
        artifact_ref: impl Into<String>,
        registry: &ModelRegistry,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        for record in registry.iter() {
            let model = self.load_model(&record.spec.model_artifact_ref)?;
            if model.identity().id != record.spec.id
                || model.identity().version != record.spec.version
                || model.training_dataset_id() != Some(record.spec.training_dataset_id.as_str())
            {
                return Err(ModelArtifactError::ModelIdentityMismatch);
            }
        }
        self.put(artifact_ref, registry.to_artifact_value())
    }

    pub fn load_model_registry(
        &self,
        artifact_ref: &str,
    ) -> Result<ModelRegistry, ModelArtifactError> {
        let artifact = self
            .artifacts
            .get(artifact_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        let registry = ModelRegistry::from_artifact_value(artifact)
            .map_err(ModelArtifactError::ModelRegistry)?;
        for record in registry.iter() {
            let model = self.load_model(&record.spec.model_artifact_ref)?;
            if model.identity().id != record.spec.id
                || model.identity().version != record.spec.version
                || model.training_dataset_id() != Some(record.spec.training_dataset_id.as_str())
            {
                return Err(ModelArtifactError::ModelIdentityMismatch);
            }
        }
        Ok(registry)
    }

    /// Reconstructs the active predictive-model ensemble for one environment contract.
    ///
    /// Only registry records explicitly marked Active and targeting the requested
    /// environment ID/version are loaded. Each artifact is checked against its
    /// registry identity, version, and training-dataset provenance.
    pub fn load_active_ensemble(
        &self,
        registry: &ModelRegistry,
        environment_id: &str,
        environment_version: &str,
    ) -> Result<EnsembleWorldModel, ModelArtifactError> {
        if environment_id.trim().is_empty() || environment_version.trim().is_empty() {
            return Err(ModelArtifactError::InvalidReference);
        }
        let mut models: Vec<Box<dyn PredictiveModel>> = Vec::new();
        for record in registry.iter().filter(|record| {
            record.status == crate::model_registry::ModelStatus::Active
                && record.spec.environment_id == environment_id
                && record.spec.environment_version == environment_version
        }) {
            let model = self.load_model(&record.spec.model_artifact_ref)?;
            if model.identity().id != record.spec.id
                || model.identity().version != record.spec.version
                || model.training_dataset_id() != Some(record.spec.training_dataset_id.as_str())
            {
                return Err(ModelArtifactError::ModelIdentityMismatch);
            }
            models.push(model);
        }
        EnsembleWorldModel::new(models).map_err(ModelArtifactError::Model)
    }

    /// Loads a persisted registry snapshot and reconstructs its active ensemble.
    pub fn load_active_ensemble_from_registry(
        &self,
        registry_artifact_ref: &str,
        environment_id: &str,
        environment_version: &str,
    ) -> Result<EnsembleWorldModel, ModelArtifactError> {
        let registry = self.load_model_registry(registry_artifact_ref)?;
        self.load_active_ensemble(&registry, environment_id, environment_version)
    }

    /// Persists a conformal calibrator only when it matches the referenced model artifact.
    pub fn put_conformal_calibrator(
        &mut self,
        calibrator_ref: impl Into<String>,
        model_artifact_ref: &str,
        calibrator: &ConformalCalibrator,
    ) -> Result<ArtifactPutOutcome, ModelArtifactError> {
        calibrator
            .validate()
            .map_err(ModelArtifactError::Calibration)?;
        let model = self.load_model(model_artifact_ref)?;
        if model.identity() != &calibrator.model_identity
            || model.training_dataset_id() != Some(calibrator.training_dataset_id.as_str())
        {
            return Err(ModelArtifactError::CalibrationModelMismatch);
        }
        self.put(calibrator_ref, calibrator.to_artifact_value())
    }

    /// Restores a conformal calibrator and verifies its model/version provenance.
    pub fn load_conformal_calibrator(
        &self,
        calibrator_ref: &str,
        model_artifact_ref: &str,
    ) -> Result<ConformalCalibrator, ModelArtifactError> {
        let value = self
            .artifacts
            .get(calibrator_ref)
            .ok_or(ModelArtifactError::NotFound)?;
        let calibrator = ConformalCalibrator::from_artifact_value(value)
            .map_err(ModelArtifactError::Calibration)?;
        let model = self.load_model(model_artifact_ref)?;
        if model.identity() != &calibrator.model_identity
            || model.training_dataset_id() != Some(calibrator.training_dataset_id.as_str())
        {
            return Err(ModelArtifactError::CalibrationModelMismatch);
        }
        Ok(calibrator)
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
        let artifact =
            fields
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
    Calibration(ConformalCalibrationError),
    CalibrationModelMismatch,
    Registry(RegistryError),
    Reasoning(ReasoningError),
    Strategy(StrategyError),
    Causal(CausalError),
    Benchmark(BenchmarkError),
    ModelRegistry(ModelRegistryError),
    ModelIdentityMismatch,
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
            Self::Encoding(message) => {
                write!(formatter, "model-artifact encoding failed: {message}")
            }
            Self::Model(error) => write!(formatter, "model-artifact conversion failed: {error}"),
            Self::Calibration(error) => write!(
                formatter,
                "conformal calibration artifact is invalid: {error}"
            ),
            Self::CalibrationModelMismatch => write!(
                formatter,
                "conformal calibrator does not match the referenced model artifact"
            ),
            Self::Registry(error) => write!(
                formatter,
                "representation-registry artifact is invalid: {error}"
            ),
            Self::Reasoning(error) => write!(
                formatter,
                "hypothesis registry artifact is invalid: {error}"
            ),
            Self::Strategy(error) => {
                write!(formatter, "strategy registry artifact is invalid: {error}")
            }
            Self::Causal(error) => write!(formatter, "causal study artifact is invalid: {error}"),
            Self::Benchmark(error) => {
                write!(formatter, "transfer benchmark artifact is invalid: {error}")
            }
            Self::ModelRegistry(error) => {
                write!(formatter, "model registry artifact is invalid: {error}")
            }
            Self::ModelIdentityMismatch => write!(
                formatter,
                "model artifact identity does not match its registry specification"
            ),
            Self::InvalidReference => write!(formatter, "artifact reference must not be empty"),
            Self::InvalidArtifact => write!(formatter, "artifact data is invalid"),
            Self::NotFound => write!(formatter, "model artifact was not found"),
            Self::ReferenceConflict => write!(
                formatter,
                "immutable artifact reference already has different content"
            ),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported artifact-store version: {version}")
            }
            Self::FrameTooLarge { size, maximum } => write!(
                formatter,
                "artifact frame size {size} exceeds limit {maximum}"
            ),
            Self::CorruptFrame { offset, reason } => write!(
                formatter,
                "corrupt model-artifact frame at byte {offset}: {reason}"
            ),
            Self::Poisoned => write!(
                formatter,
                "model-artifact store must be reopened after an I/O failure"
            ),
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
        std::env::temp_dir().join(format!(
            "5413-model-artifacts-{}-{id}.bin",
            std::process::id()
        ))
    }

    fn sample(action: &str, before: f64, after: f64) -> TransitionSample {
        TransitionSample {
            action_id: action.to_owned(),
            before: DataValue::object([("position".to_owned(), DataValue::Number(before))]),
            after: DataValue::object([("position".to_owned(), DataValue::Number(after))]),
        }
    }

    #[test]
    fn persists_conformal_calibration_bound_to_model_artifact() {
        use crate::calibration::ConformalCalibrator;

        let path = test_path();
        let mut model = ActionDeltaModel::new("calibrated-delta", 3).expect("model");
        model
            .fit(
                "training-v1",
                &[sample("advance", 0.0, 1.0), sample("advance", 5.0, 6.0)],
            )
            .expect("fit model");
        let calibration_samples = vec![
            sample("advance", 10.0, 11.0),
            sample("advance", 20.0, 22.0),
            sample("advance", 30.0, 33.0),
            sample("advance", 40.0, 44.0),
        ];
        let calibrator =
            ConformalCalibrator::fit(&model, "calibration-v1", &calibration_samples, 0.75)
                .expect("fit conformal calibrator");
        {
            let mut store = ModelArtifactStore::open(&path).expect("open artifact store");
            store
                .put_model("model-calibrated-v3", &model)
                .expect("persist model");
            store
                .put_conformal_calibrator(
                    "calibrator-for-model-v3",
                    "model-calibrated-v3",
                    &calibrator,
                )
                .expect("persist calibrator");
        }
        {
            let mut store = ModelArtifactStore::open(&path).expect("reopen artifact store");
            let restored = store
                .load_conformal_calibrator("calibrator-for-model-v3", "model-calibrated-v3")
                .expect("restore calibrator");
            assert_eq!(restored, calibrator);

            let mut other = ActionDeltaModel::new("calibrated-delta", 4).expect("other model");
            other
                .fit(
                    "training-v1",
                    &[sample("advance", 0.0, 1.0), sample("advance", 5.0, 6.0)],
                )
                .expect("fit other version");
            store
                .load_conformal_calibrator("calibrator-for-model-v3", "model-calibrated-v3")
                .expect("stored model reference remains valid");
            // A caller cannot substitute an unrelated model artifact reference.
            assert!(matches!(
                store.put_conformal_calibrator(
                    "calibrator-for-other-model",
                    "model-calibrated-v3",
                    &ConformalCalibrator::fit(
                        &other,
                        "calibration-v1",
                        &calibration_samples,
                        0.75,
                    )
                    .expect("fit other calibrator"),
                ),
                Err(ModelArtifactError::CalibrationModelMismatch)
            ));
        }
        std::fs::remove_file(path).expect("remove artifact store");
    }

    #[test]
    fn persists_and_restores_statistical_model_predictions() {
        let path = test_path();
        let mut model = ActionDeltaModel::new("delta", 2).expect("model");
        model
            .fit(
                "training-a",
                &[sample("advance", 0.0, 2.0), sample("advance", 4.0, 6.0)],
            )
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
                store
                    .put_model("model-artifact-1", &model)
                    .expect("duplicate"),
                ArtifactPutOutcome::AlreadyPresent
            );
        }
        {
            let store = ModelArtifactStore::open(&path).expect("reopen");
            let restored = store.load_model("model-artifact-1").expect("restore");
            assert_eq!(restored.identity().id, "delta");
            assert_eq!(restored.identity().version, 2);
            assert_eq!(
                restored.predict("advance", &state).expect("predict"),
                expected
            );
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
    fn persists_and_restores_representation_registry_snapshot() {
        use crate::representation::{
            EvaluationPolicy, RepresentationKind, RepresentationRegistry, RepresentationSpec,
            RepresentationStatus,
        };
        use crate::schema::{FieldSpec, SchemaDefinition, ValueKind};

        let path = test_path();
        let mut registry = RepresentationRegistry::default();
        registry
            .register_candidate(RepresentationSpec {
                id: "state-vocabulary".to_owned(),
                version: 1,
                kind: RepresentationKind::Relational,
                schema: SchemaDefinition {
                    id: "state-schema".to_owned(),
                    version: 1,
                    fields: vec![FieldSpec {
                        name: "position".to_owned(),
                        kind: ValueKind::Number,
                        required: true,
                    }],
                    allow_additional_fields: false,
                },
                description: "learned state representation".to_owned(),
                training_set_id: "train-v1".to_owned(),
                evidence_refs: vec!["episode:1".to_owned()],
            })
            .expect("register");
        registry
            .evaluate_candidate(
                "state-vocabulary",
                1,
                "validation-v1",
                &[DataValue::object([(
                    "position".to_owned(),
                    DataValue::Number(2.0),
                )])],
                EvaluationPolicy {
                    minimum_validation_samples: 1,
                    minimum_valid_fraction: 1.0,
                },
            )
            .expect("evaluate");
        registry.promote("state-vocabulary", 1).expect("promote");

        {
            let mut store = ModelArtifactStore::open(&path).expect("open");
            store
                .put_representation_registry("registry-snapshot-v1", &registry)
                .expect("persist registry");
        }
        {
            let store = ModelArtifactStore::open(&path).expect("reopen");
            let restored = store
                .load_representation_registry("registry-snapshot-v1")
                .expect("restore registry");
            let active = restored.active("state-vocabulary").expect("active");
            assert_eq!(active.spec.version, 1);
            assert_eq!(active.status, RepresentationStatus::Active);
            assert_eq!(active.spec.evidence_refs, vec!["episode:1"]);
            assert_eq!(
                active
                    .evaluation
                    .as_ref()
                    .map(|item| item.evaluation_set_id.as_str()),
                Some("validation-v1")
            );
        }
        std::fs::remove_file(path).expect("remove artifact file");
    }

    #[test]
    fn persists_and_restores_hypothesis_evidence() {
        use crate::reasoning::{
            EvidenceRecord, HypothesisPolicy, HypothesisRegistry, HypothesisSpec, HypothesisStatus,
        };

        let path = test_path();
        let policy = HypothesisPolicy {
            minimum_evidence_count: 1,
            support_probability: 0.9,
            refutation_probability: 0.1,
        };
        let mut registry = HypothesisRegistry::default();
        registry
            .insert(
                HypothesisSpec {
                    id: "relation-a".to_owned(),
                    claim: DataValue::object([(
                        "relation".to_owned(),
                        DataValue::Text("x->y".to_owned()),
                    )]),
                    description: "candidate relation".to_owned(),
                    prior_probability: 0.5,
                    source_dataset_id: Some("discovery".to_owned()),
                },
                policy,
            )
            .expect("insert");
        registry
            .add_evidence(
                "relation-a",
                EvidenceRecord {
                    evidence_ref: "trial-1".to_owned(),
                    dataset_id: "validation-1".to_owned(),
                    log_likelihood_ratio: 2.5,
                    rationale: "predeclared trial evidence".to_owned(),
                },
                policy,
            )
            .expect("evidence");
        assert_eq!(
            registry.get("relation-a").map(|record| record.status),
            Some(HypothesisStatus::Supported)
        );

        {
            let mut store = ModelArtifactStore::open(&path).expect("open");
            store
                .put_hypothesis_registry("hypotheses-v1", &registry)
                .expect("persist");
        }
        {
            let store = ModelArtifactStore::open(&path).expect("reopen");
            let restored = store
                .load_hypothesis_registry("hypotheses-v1", policy)
                .expect("restore");
            assert_eq!(restored, registry);
        }
        std::fs::remove_file(path).expect("remove artifact file");
    }

    #[test]
    fn persists_and_restores_typed_strategy_registry() {
        use crate::meta_learning::{
            InputBinding, OperatorCatalogue, OperatorSpec, PortSpec, StrategyEvaluation,
            StrategyGraph, StrategyNode, StrategyPolicy, StrategyRegistry, StrategySpec,
            StrategyStatus,
        };
        use crate::schema::ValueKind;
        use std::collections::BTreeMap;

        let path = test_path();
        let mut catalogue = OperatorCatalogue::default();
        catalogue
            .register(OperatorSpec {
                id: "numeric-normalizer".to_owned(),
                inputs: vec![PortSpec {
                    name: "raw".to_owned(),
                    kind: ValueKind::Number,
                }],
                outputs: vec![PortSpec {
                    name: "normalized".to_owned(),
                    kind: ValueKind::Number,
                }],
                cost_units: 3,
                deterministic: true,
            })
            .expect("register operator");

        let graph = StrategyGraph {
            external_inputs: BTreeMap::from([("observation".to_owned(), ValueKind::Number)]),
            nodes: vec![StrategyNode {
                id: "normalize".to_owned(),
                operator_id: "numeric-normalizer".to_owned(),
                inputs: BTreeMap::from([(
                    "raw".to_owned(),
                    InputBinding::External("observation".to_owned()),
                )]),
            }],
            outputs: BTreeMap::from([(
                "features".to_owned(),
                InputBinding::NodeOutput {
                    node_id: "normalize".to_owned(),
                    port: "normalized".to_owned(),
                },
            )]),
        };
        let policy = StrategyPolicy {
            minimum_evaluation_samples: 5,
            minimum_improvement: 0.01,
            resource_cost_weight: 0.1,
        };
        let mut registry = StrategyRegistry::default();
        registry
            .register_candidate(
                StrategySpec {
                    id: "pipeline".to_owned(),
                    version: 1,
                    training_dataset_id: "training-v1".to_owned(),
                    graph,
                },
                &catalogue,
            )
            .expect("register candidate");
        registry
            .evaluate_candidate(
                "pipeline",
                1,
                StrategyEvaluation {
                    evaluation_dataset_id: "holdout-v1".to_owned(),
                    metric_id: "accuracy".to_owned(),
                    direction: crate::benchmark::ScoreDirection::HigherIsBetter,
                    sample_count: 20,
                    primary_score: 0.85,
                    resource_cost: 2.0,
                },
                policy,
            )
            .expect("evaluate");
        registry.promote("pipeline", 1, policy).expect("promote");
        assert_eq!(
            registry.active("pipeline").map(|record| record.status),
            Some(StrategyStatus::Active)
        );

        {
            let mut store = ModelArtifactStore::open(&path).expect("open");
            store
                .put_strategy_registry("strategies-v1", &registry)
                .expect("persist");
        }
        {
            let store = ModelArtifactStore::open(&path).expect("reopen");
            let restored = store
                .load_strategy_registry("strategies-v1", &catalogue)
                .expect("restore");
            assert_eq!(restored, registry);
        }
        std::fs::remove_file(path).expect("remove artifact file");
    }

    #[test]
    fn persists_and_restores_paired_randomized_causal_study() {
        use crate::causal::{AssignmentDesign, PairedOutcome, PairedRandomizedStudy};
        let path = test_path();
        let study = PairedRandomizedStudy {
            experiment_id: "intervention-v1".to_owned(),
            dataset_id: "paired-set-v1".to_owned(),
            treatment_label: "treatment".to_owned(),
            control_label: "control".to_owned(),
            outcome_label: "outcome".to_owned(),
            assignment_design: AssignmentDesign::Randomized,
            pairs: vec![
                PairedOutcome {
                    pair_id: "block-a".to_owned(),
                    control_outcome: 1.0,
                    treatment_outcome: 2.0,
                },
                PairedOutcome {
                    pair_id: "block-b".to_owned(),
                    control_outcome: 1.5,
                    treatment_outcome: 2.8,
                },
            ],
        };
        {
            let mut store = ModelArtifactStore::open(&path).expect("open");
            store
                .put_causal_study("causal-study-v1", &study)
                .expect("persist");
        }
        {
            let store = ModelArtifactStore::open(&path).expect("reopen");
            let restored = store.load_causal_study("causal-study-v1").expect("restore");
            assert_eq!(restored, study);
            let estimate = restored.estimate().expect("estimate");
            assert_eq!(estimate.pair_count, 2);
            assert!(estimate.mean_effect > 0.0);
        }
        std::fs::remove_file(path).expect("remove artifact file");
    }

    #[test]
    fn persists_and_restores_transfer_benchmark() {
        use crate::benchmark::{ScoreDirection, TransferBenchmark, TransferEvaluation};
        let path = test_path();
        let mut benchmark = TransferBenchmark::default();
        benchmark
            .record(TransferEvaluation {
                case_id: "transfer-1".to_owned(),
                source_environment_id: "physics".to_owned(),
                target_environment_id: "grid".to_owned(),
                source_training_dataset_id: "physics-train".to_owned(),
                target_evaluation_dataset_id: "grid-holdout".to_owned(),
                model_artifact_ref: "model-v1".to_owned(),
                metric_id: "prediction-rmse".to_owned(),
                direction: ScoreDirection::LowerIsBetter,
                baseline_score: 0.5,
                transferred_score: 0.3,
                sample_count: 100,
                resource_cost: 12.0,
            })
            .expect("record transfer");
        {
            let mut store = ModelArtifactStore::open(&path).expect("open");
            store
                .put_transfer_benchmark("transfer-benchmark-v1", &benchmark)
                .expect("persist");
        }
        {
            let store = ModelArtifactStore::open(&path).expect("reopen");
            let restored = store
                .load_transfer_benchmark("transfer-benchmark-v1")
                .expect("restore");
            assert_eq!(restored, benchmark);
            assert!(
                (restored
                    .summarize("prediction-rmse")
                    .expect("summary")
                    .mean_improvement
                    - 0.2)
                    .abs()
                    < 1e-12
            );
        }
        std::fs::remove_file(path).expect("remove artifact file");
    }

    #[test]
    fn model_registry_requires_matching_persisted_model_artifacts() {
        use crate::model_registry::{
            ModelCandidateSpec, ModelEvaluation, ModelPromotionPolicy, ModelRegistry, ModelStatus,
        };
        let path = test_path();
        let mut model = ActionDeltaModel::new("registered-model", 1).expect("model");
        model
            .fit(
                "training-v1",
                &[sample("advance", 0.0, 1.0), sample("advance", 2.0, 3.0)],
            )
            .expect("fit");
        let mut store = ModelArtifactStore::open(&path).expect("open");
        store
            .put_model("registered-model-v1", &model)
            .expect("put model");
        let mut registry = ModelRegistry::default();
        store
            .register_model_candidate(
                &mut registry,
                ModelCandidateSpec {
                    id: "registered-model".to_owned(),
                    version: 1,
                    model_artifact_ref: "registered-model-v1".to_owned(),
                    environment_id: "counter".to_owned(),
                    environment_version: "v1".to_owned(),
                    training_dataset_id: "training-v1".to_owned(),
                },
            )
            .expect("register candidate");
        let policy = ModelPromotionPolicy {
            minimum_evaluation_samples: 2,
            minimum_improvement: 0.0,
            resource_cost_weight: 0.0,
            require_calibrated_uncertainty: false,
        };
        registry
            .evaluate_candidate(
                "registered-model",
                1,
                ModelEvaluation {
                    evaluation_dataset_id: "holdout-v1".to_owned(),
                    metric_id: "rmse".to_owned(),
                    direction: crate::benchmark::ScoreDirection::LowerIsBetter,
                    sample_count: 5,
                    score: 0.0,
                    resource_cost: 1.0,
                    uncertainty_calibrated: true,
                },
                policy,
            )
            .expect("evaluate");
        registry
            .promote("registered-model", 1, policy)
            .expect("promote");
        store
            .put_model_registry("model-registry-v1", &registry)
            .expect("persist registry");
        let restored = store
            .load_model_registry("model-registry-v1")
            .expect("restore registry");
        assert_eq!(restored, registry);
        assert_eq!(
            restored
                .active("registered-model")
                .map(|record| record.status),
            Some(ModelStatus::Active)
        );
        let restored_ensemble = store
            .load_active_ensemble_from_registry("model-registry-v1", "counter", "v1")
            .expect("restore active ensemble");
        assert_eq!(restored_ensemble.len(), 1);
        let prediction = restored_ensemble
            .predict(
                "advance",
                &DataValue::object([("position".to_owned(), DataValue::Number(10.0))]),
            )
            .expect("predict with restored active model");
        assert_eq!(
            prediction
                .fields
                .get("position")
                .map(|estimate| estimate.mean),
            Some(11.0)
        );
        assert!(matches!(
            store.load_active_ensemble(&restored, "different-environment", "v1"),
            Err(ModelArtifactError::Model(WorldModelError::EmptyEnsemble))
        ));
        std::fs::remove_file(path).expect("remove artifact file");
    }

    #[test]
    fn rejects_conflicting_reference_and_recovers_torn_tail() {
        let path = test_path();
        let artifact =
            DataValue::object([("opaque".to_owned(), DataValue::Text("value".to_owned()))]);
        let first_length;
        {
            let mut store = ModelArtifactStore::open(&path).expect("open");
            store.put("opaque-v1", artifact.clone()).expect("put");
            first_length = store.file.metadata().expect("metadata").len();
            assert!(matches!(
                store.put("opaque-v1", DataValue::Null),
                Err(ModelArtifactError::ReferenceConflict)
            ));
            store
                .put("opaque-v2", DataValue::Integer(2))
                .expect("put second");
        }
        {
            let file = OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("open raw");
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
