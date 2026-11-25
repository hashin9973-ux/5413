use crate::schema::{SchemaDefinition, SchemaDefinitionError};
use crate::value::DataValue;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepresentationKind {
    Relational,
    Temporal,
    Symbolic,
    Statistical,
    Vector,
    Procedure,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationSpec {
    pub id: String,
    pub version: u64,
    pub kind: RepresentationKind,
    pub schema: SchemaDefinition,
    pub description: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepresentationStatus {
    Candidate,
    Active,
    Rejected,
    Retired,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvaluationPolicy {
    pub minimum_validation_samples: usize,
    pub minimum_valid_fraction: f64,
}

impl EvaluationPolicy {
    pub fn validate(&self) -> Result<(), RegistryError> {
        if self.minimum_validation_samples == 0
            || !self.minimum_valid_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_valid_fraction)
        {
            return Err(RegistryError::InvalidEvaluationPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepresentationEvaluation {
    pub evaluation_set_id: String,
    pub evaluated_samples: usize,
    pub valid_samples: usize,
    pub valid_fraction: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepresentationRecord {
    pub spec: RepresentationSpec,
    pub status: RepresentationStatus,
    pub evaluation: Option<RepresentationEvaluation>,
}

#[derive(Clone, Debug, Default)]
pub struct RepresentationRegistry {
    records: std::collections::BTreeMap<(String, u64), RepresentationRecord>,
}

impl RepresentationRegistry {
    pub fn register_candidate(&mut self, spec: RepresentationSpec) -> Result<(), RegistryError> {
        if spec.id.trim().is_empty() {
            return Err(RegistryError::EmptyId);
        }
        if spec.version == 0 {
            return Err(RegistryError::ZeroVersion);
        }
        spec.schema
            .validate_definition()
            .map_err(RegistryError::InvalidSchema)?;

        let key = (spec.id.clone(), spec.version);
        if self.records.contains_key(&key) {
            return Err(RegistryError::VersionAlreadyExists {
                id: spec.id,
                version: spec.version,
            });
        }

        self.records.insert(
            key,
            RepresentationRecord {
                spec,
                status: RepresentationStatus::Candidate,
                evaluation: None,
            },
        );
        Ok(())
    }

    pub fn evaluate_candidate(
        &mut self,
        id: &str,
        version: u64,
        evaluation_set_id: impl Into<String>,
        validation_samples: &[DataValue],
        policy: EvaluationPolicy,
    ) -> Result<RepresentationEvaluation, RegistryError> {
        policy.validate()?;
        let evaluation_set_id = evaluation_set_id.into();
        if evaluation_set_id.trim().is_empty() {
            return Err(RegistryError::EmptyEvaluationSetId);
        }

        let key = (id.to_owned(), version);
        let schema = {
            let record = self.records.get(&key).ok_or_else(|| RegistryError::NotFound {
                id: id.to_owned(),
                version,
            })?;
            if record.status != RepresentationStatus::Candidate {
                return Err(RegistryError::NotCandidate {
                    id: id.to_owned(),
                    version,
                });
            }
            record.spec.schema.clone()
        };

        let valid_samples = validation_samples
            .iter()
            .filter(|sample| schema.validate(sample).is_ok())
            .count();
        let evaluated_samples = validation_samples.len();
        let valid_fraction = if evaluated_samples == 0 {
            0.0
        } else {
            valid_samples as f64 / evaluated_samples as f64
        };
        let passed = evaluated_samples >= policy.minimum_validation_samples
            && valid_fraction >= policy.minimum_valid_fraction;

        let evaluation = RepresentationEvaluation {
            evaluation_set_id,
            evaluated_samples,
            valid_samples,
            valid_fraction,
            passed,
        };
        if let Some(record) = self.records.get_mut(&key) {
            record.evaluation = Some(evaluation.clone());
        }
        Ok(evaluation)
    }

    pub fn promote(&mut self, id: &str, version: u64) -> Result<(), RegistryError> {
        let key = (id.to_owned(), version);
        let target = self
            .records
            .get(&key)
            .ok_or_else(|| RegistryError::NotFound {
                id: id.to_owned(),
                version,
            })?;

        if target.status != RepresentationStatus::Candidate {
            return Err(RegistryError::NotCandidate {
                id: id.to_owned(),
                version,
            });
        }
        match &target.evaluation {
            None => return Err(RegistryError::MissingEvaluation),
            Some(evaluation) if !evaluation.passed => {
                return Err(RegistryError::EvaluationFailed);
            }
            Some(_) => {}
        }

        for ((record_id, _), record) in &mut self.records {
            if record_id == id && record.status == RepresentationStatus::Active {
                record.status = RepresentationStatus::Retired;
            }
        }

        if let Some(record) = self.records.get_mut(&key) {
            record.status = RepresentationStatus::Active;
        }
        Ok(())
    }

    pub fn reject(&mut self, id: &str, version: u64) -> Result<(), RegistryError> {
        let record = self
            .records
            .get_mut(&(id.to_owned(), version))
            .ok_or_else(|| RegistryError::NotFound {
                id: id.to_owned(),
                version,
            })?;

        if record.status != RepresentationStatus::Candidate {
            return Err(RegistryError::NotCandidate {
                id: id.to_owned(),
                version,
            });
        }
        record.status = RepresentationStatus::Rejected;
        Ok(())
    }

    pub fn get(&self, id: &str, version: u64) -> Option<&RepresentationRecord> {
        self.records.get(&(id.to_owned(), version))
    }

    pub fn active(&self, id: &str) -> Option<&RepresentationRecord> {
        self.records
            .iter()
            .find(|((record_id, _), record)| {
                record_id == id && record.status == RepresentationStatus::Active
            })
            .map(|(_, record)| record)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryError {
    EmptyId,
    ZeroVersion,
    InvalidSchema(SchemaDefinitionError),
    VersionAlreadyExists { id: String, version: u64 },
    NotFound { id: String, version: u64 },
    NotCandidate { id: String, version: u64 },
    EmptyEvaluationSetId,
    InvalidEvaluationPolicy,
    MissingEvaluation,
    EvaluationFailed,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyId => write!(formatter, "representation id must not be empty"),
            Self::ZeroVersion => {
                write!(
                    formatter,
                    "representation version must be greater than zero"
                )
            }
            Self::InvalidSchema(error) => {
                write!(formatter, "invalid representation schema: {error}")
            }
            Self::VersionAlreadyExists { id, version } => {
                write!(formatter, "representation {id}@{version} already exists")
            }
            Self::NotFound { id, version } => {
                write!(formatter, "representation {id}@{version} was not found")
            }
            Self::NotCandidate { id, version } => {
                write!(
                    formatter,
                    "representation {id}@{version} is not a candidate"
                )
            }
            Self::EmptyEvaluationSetId => write!(formatter, "evaluation set id must not be empty"),
            Self::InvalidEvaluationPolicy => write!(formatter, "evaluation policy is invalid"),
            Self::MissingEvaluation => write!(formatter, "candidate has not been evaluated"),
            Self::EvaluationFailed => write!(formatter, "candidate did not pass evaluation"),
        }
    }
}

impl std::error::Error for RegistryError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{FieldSpec, ValueKind};

    fn spec(version: u64) -> RepresentationSpec {
        RepresentationSpec {
            id: "state-model".to_owned(),
            version,
            kind: RepresentationKind::Relational,
            schema: SchemaDefinition {
                id: format!("state-model-schema-v{version}"),
                version,
                fields: vec![FieldSpec {
                    name: "state".to_owned(),
                    kind: ValueKind::Object,
                    required: true,
                }],
                allow_additional_fields: false,
            },
            description: "A candidate state representation".to_owned(),
            evidence_refs: Vec::new(),
        }
    }

    #[test]
    fn promotion_requires_held_out_evaluation() {
        let mut registry = RepresentationRegistry::default();
        registry.register_candidate(spec(1)).expect("candidate");
        assert_eq!(
            registry.promote("state-model", 1),
            Err(RegistryError::MissingEvaluation)
        );
    }

    #[test]
    fn failed_evaluation_blocks_promotion() {
        let mut registry = RepresentationRegistry::default();
        registry.register_candidate(spec(1)).expect("candidate");
        let evaluation = registry
            .evaluate_candidate(
                "state-model",
                1,
                "held-out-invalid",
                &[DataValue::object([(
                    "unexpected".to_owned(),
                    DataValue::Bool(true),
                )])],
                EvaluationPolicy {
                    minimum_validation_samples: 1,
                    minimum_valid_fraction: 1.0,
                },
            )
            .expect("evaluation should be recorded");

        assert!(!evaluation.passed);
        assert_eq!(evaluation.valid_samples, 0);
        assert_eq!(
            registry.promote("state-model", 1),
            Err(RegistryError::EvaluationFailed)
        );
    }

    #[test]
    fn insufficient_validation_samples_cannot_pass() {
        let mut registry = RepresentationRegistry::default();
        registry.register_candidate(spec(1)).expect("candidate");
        let evaluation = registry
            .evaluate_candidate(
                "state-model",
                1,
                "held-out-too-small",
                &[DataValue::object([(
                    "state".to_owned(),
                    DataValue::Object(std::collections::BTreeMap::new()),
                )])],
                EvaluationPolicy {
                    minimum_validation_samples: 2,
                    minimum_valid_fraction: 1.0,
                },
            )
            .expect("evaluation should be recorded");

        assert_eq!(evaluation.valid_fraction, 1.0);
        assert!(!evaluation.passed);
    }

    #[test]
    fn promotion_retires_previous_active_version() {
        let mut registry = RepresentationRegistry::default();
        registry.register_candidate(spec(1)).expect("candidate v1");
        registry
            .evaluate_candidate(
                "state-model",
                1,
                "held-out-a",
                &[DataValue::object([(
                    "state".to_owned(),
                    DataValue::Object(std::collections::BTreeMap::new()),
                )])],
                EvaluationPolicy {
                    minimum_validation_samples: 1,
                    minimum_valid_fraction: 1.0,
                },
            )
            .expect("evaluate v1");
        registry.promote("state-model", 1).expect("promote v1");
        registry.register_candidate(spec(2)).expect("candidate v2");
        registry
            .evaluate_candidate(
                "state-model",
                2,
                "held-out-b",
                &[DataValue::object([(
                    "state".to_owned(),
                    DataValue::Object(std::collections::BTreeMap::new()),
                )])],
                EvaluationPolicy {
                    minimum_validation_samples: 1,
                    minimum_valid_fraction: 1.0,
                },
            )
            .expect("evaluate v2");
        registry.promote("state-model", 2).expect("promote v2");

        assert_eq!(
            registry.get("state-model", 1).map(|record| record.status),
            Some(RepresentationStatus::Retired)
        );
        assert_eq!(
            registry
                .active("state-model")
                .map(|record| record.spec.version),
            Some(2)
        );
    }
}
