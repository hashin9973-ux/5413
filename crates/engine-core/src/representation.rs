use crate::schema::{SchemaDefinition, SchemaDefinitionError};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationRecord {
    pub spec: RepresentationSpec,
    pub status: RepresentationStatus,
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
            },
        );
        Ok(())
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
    fn promotion_retires_previous_active_version() {
        let mut registry = RepresentationRegistry::default();
        registry.register_candidate(spec(1)).expect("candidate v1");
        registry.promote("state-model", 1).expect("promote v1");
        registry.register_candidate(spec(2)).expect("candidate v2");
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
