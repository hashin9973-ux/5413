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
    pub training_set_id: String,
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
    pub policy: EvaluationPolicy,
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
        if spec.training_set_id.trim().is_empty() {
            return Err(RegistryError::EmptyTrainingSetId);
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
        let (schema, training_set_id) = {
            let record = self
                .records
                .get(&key)
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
            (
                record.spec.schema.clone(),
                record.spec.training_set_id.clone(),
            )
        };
        if training_set_id == evaluation_set_id {
            return Err(RegistryError::EvaluationSetLeakage);
        }

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
            policy,
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

    /// Encodes registry records and evaluation evidence into a versioned data value.
    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "records".to_owned(),
                DataValue::Sequence(self.records.values().map(record_to_value).collect()),
            ),
        ])
    }

    /// Restores all registry records while checking lifecycle and evaluation invariants.
    pub fn from_artifact_value(value: &DataValue) -> Result<Self, RegistryError> {
        let fields = value.as_object().ok_or(RegistryError::InvalidSerializedState)?;
        if fields.len() != 2 || fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
            return Err(RegistryError::InvalidSerializedState);
        }
        let encoded_records = match fields.get("records") {
            Some(DataValue::Sequence(records)) => records,
            _ => return Err(RegistryError::InvalidSerializedState),
        };

        let mut records = std::collections::BTreeMap::new();
        let mut active_ids = std::collections::BTreeSet::new();
        for encoded in encoded_records {
            let record = record_from_value(encoded)?;
            if record.status == RepresentationStatus::Active
                && !active_ids.insert(record.spec.id.clone())
            {
                return Err(RegistryError::InvalidSerializedState);
            }
            let key = (record.spec.id.clone(), record.spec.version);
            if records.insert(key, record).is_some() {
                return Err(RegistryError::InvalidSerializedState);
            }
        }
        Ok(Self { records })
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


fn record_to_value(record: &RepresentationRecord) -> DataValue {
    let kind = match &record.spec.kind {
        RepresentationKind::Relational => "relational".to_owned(),
        RepresentationKind::Temporal => "temporal".to_owned(),
        RepresentationKind::Symbolic => "symbolic".to_owned(),
        RepresentationKind::Statistical => "statistical".to_owned(),
        RepresentationKind::Vector => "vector".to_owned(),
        RepresentationKind::Procedure => "procedure".to_owned(),
        RepresentationKind::Custom(value) => format!("custom:{value}"),
    };
    let schema = DataValue::object([
        ("id".to_owned(), DataValue::Text(record.spec.schema.id.clone())),
        ("version".to_owned(), DataValue::Text(record.spec.schema.version.to_string())),
        (
            "allow_additional_fields".to_owned(),
            DataValue::Bool(record.spec.schema.allow_additional_fields),
        ),
        (
            "fields".to_owned(),
            DataValue::Sequence(
                record.spec.schema.fields.iter().map(|field| {
                    DataValue::object([
                        ("name".to_owned(), DataValue::Text(field.name.clone())),
                        ("kind".to_owned(), DataValue::Text(value_kind_name(field.kind).to_owned())),
                        ("required".to_owned(), DataValue::Bool(field.required)),
                    ])
                }).collect(),
            ),
        ),
    ]);
    let evaluation = match &record.evaluation {
        None => DataValue::Null,
        Some(value) => DataValue::object([
            ("evaluation_set_id".to_owned(), DataValue::Text(value.evaluation_set_id.clone())),
            ("minimum_validation_samples".to_owned(), DataValue::Text(value.policy.minimum_validation_samples.to_string())),
            ("minimum_valid_fraction".to_owned(), DataValue::Number(value.policy.minimum_valid_fraction)),
            ("evaluated_samples".to_owned(), DataValue::Text(value.evaluated_samples.to_string())),
            ("valid_samples".to_owned(), DataValue::Text(value.valid_samples.to_string())),
            ("valid_fraction".to_owned(), DataValue::Number(value.valid_fraction)),
            ("passed".to_owned(), DataValue::Bool(value.passed)),
        ]),
    };
    DataValue::object([
        ("id".to_owned(), DataValue::Text(record.spec.id.clone())),
        ("version".to_owned(), DataValue::Text(record.spec.version.to_string())),
        ("kind".to_owned(), DataValue::Text(kind)),
        ("schema".to_owned(), schema),
        ("description".to_owned(), DataValue::Text(record.spec.description.clone())),
        ("training_set_id".to_owned(), DataValue::Text(record.spec.training_set_id.clone())),
        (
            "evidence_refs".to_owned(),
            DataValue::Sequence(record.spec.evidence_refs.iter().cloned().map(DataValue::Text).collect()),
        ),
        ("status".to_owned(), DataValue::Text(status_name(record.status).to_owned())),
        ("evaluation".to_owned(), evaluation),
    ])
}

fn status_name(status: RepresentationStatus) -> &'static str {
    match status {
        RepresentationStatus::Candidate => "candidate",
        RepresentationStatus::Active => "active",
        RepresentationStatus::Rejected => "rejected",
        RepresentationStatus::Retired => "retired",
    }
}

fn value_kind_name(kind: crate::schema::ValueKind) -> &'static str {
    use crate::schema::ValueKind::*;
    match kind {
        Any => "any",
        Null => "null",
        Boolean => "boolean",
        Integer => "integer",
        Number => "number",
        Text => "text",
        Sequence => "sequence",
        Object => "object",
        BlobRef => "blob_ref",
    }
}

fn parse_value_kind(value: &str) -> Result<crate::schema::ValueKind, RegistryError> {
    use crate::schema::ValueKind::*;
    match value {
        "any" => Ok(Any),
        "null" => Ok(Null),
        "boolean" => Ok(Boolean),
        "integer" => Ok(Integer),
        "number" => Ok(Number),
        "text" => Ok(Text),
        "sequence" => Ok(Sequence),
        "object" => Ok(Object),
        "blob_ref" => Ok(BlobRef),
        _ => Err(RegistryError::InvalidSerializedState),
    }
}

fn object_fields(value: &DataValue) -> Result<&std::collections::BTreeMap<String, DataValue>, RegistryError> {
    value.as_object().ok_or(RegistryError::InvalidSerializedState)
}

fn text_field(
    fields: &std::collections::BTreeMap<String, DataValue>,
    name: &str,
) -> Result<String, RegistryError> {
    match fields.get(name) {
        Some(DataValue::Text(value)) => Ok(value.clone()),
        _ => Err(RegistryError::InvalidSerializedState),
    }
}

fn u64_field(
    fields: &std::collections::BTreeMap<String, DataValue>,
    name: &str,
) -> Result<u64, RegistryError> {
    text_field(fields, name)?
        .parse()
        .map_err(|_| RegistryError::InvalidSerializedState)
}

fn bool_field(
    fields: &std::collections::BTreeMap<String, DataValue>,
    name: &str,
) -> Result<bool, RegistryError> {
    match fields.get(name) {
        Some(DataValue::Bool(value)) => Ok(*value),
        _ => Err(RegistryError::InvalidSerializedState),
    }
}

fn record_from_value(value: &DataValue) -> Result<RepresentationRecord, RegistryError> {
    use crate::schema::{FieldSpec, SchemaDefinition};
    let fields = object_fields(value)?;
    if fields.len() != 9 {
        return Err(RegistryError::InvalidSerializedState);
    }
    let id = text_field(fields, "id")?;
    let version = u64_field(fields, "version")?;
    let kind_text = text_field(fields, "kind")?;
    let kind = match kind_text.as_str() {
        "relational" => RepresentationKind::Relational,
        "temporal" => RepresentationKind::Temporal,
        "symbolic" => RepresentationKind::Symbolic,
        "statistical" => RepresentationKind::Statistical,
        "vector" => RepresentationKind::Vector,
        "procedure" => RepresentationKind::Procedure,
        value if value.starts_with("custom:") => RepresentationKind::Custom(value[7..].to_owned()),
        _ => return Err(RegistryError::InvalidSerializedState),
    };

    let schema_fields = object_fields(fields.get("schema").ok_or(RegistryError::InvalidSerializedState)?)?;
    if schema_fields.len() != 4 {
        return Err(RegistryError::InvalidSerializedState);
    }
    let schema_id = text_field(schema_fields, "id")?;
    let schema_version = u64_field(schema_fields, "version")?;
    let allow_additional_fields = bool_field(schema_fields, "allow_additional_fields")?;
    let encoded_fields = match schema_fields.get("fields") {
        Some(DataValue::Sequence(values)) => values,
        _ => return Err(RegistryError::InvalidSerializedState),
    };
    let mut decoded_fields = Vec::with_capacity(encoded_fields.len());
    for encoded in encoded_fields {
        let field = object_fields(encoded)?;
        if field.len() != 3 {
            return Err(RegistryError::InvalidSerializedState);
        }
        decoded_fields.push(FieldSpec {
            name: text_field(field, "name")?,
            kind: parse_value_kind(&text_field(field, "kind")?)?,
            required: bool_field(field, "required")?,
        });
    }
    let schema = SchemaDefinition {
        id: schema_id,
        version: schema_version,
        fields: decoded_fields,
        allow_additional_fields,
    };
    schema.validate_definition().map_err(RegistryError::InvalidSchema)?;

    let description = text_field(fields, "description")?;
    let training_set_id = text_field(fields, "training_set_id")?;
    let evidence_refs = match fields.get("evidence_refs") {
        Some(DataValue::Sequence(values)) => values.iter().map(|value| match value {
            DataValue::Text(value) => Ok(value.clone()),
            _ => Err(RegistryError::InvalidSerializedState),
        }).collect::<Result<Vec<_>, _>>()?,
        _ => return Err(RegistryError::InvalidSerializedState),
    };
    let status = match text_field(fields, "status")?.as_str() {
        "candidate" => RepresentationStatus::Candidate,
        "active" => RepresentationStatus::Active,
        "rejected" => RepresentationStatus::Rejected,
        "retired" => RepresentationStatus::Retired,
        _ => return Err(RegistryError::InvalidSerializedState),
    };
    let evaluation = match fields.get("evaluation") {
        Some(DataValue::Null) => None,
        Some(DataValue::Object(values)) if values.len() == 7 => {
            let evaluation_set_id = text_field(values, "evaluation_set_id")?;
            let policy = EvaluationPolicy {
                minimum_validation_samples: usize::try_from(u64_field(values, "minimum_validation_samples")?)
                    .map_err(|_| RegistryError::InvalidSerializedState)?,
                minimum_valid_fraction: values.get("minimum_valid_fraction")
                    .and_then(DataValue::as_f64)
                    .ok_or(RegistryError::InvalidSerializedState)?,
            };
            policy.validate()?;
            let evaluated_samples = usize::try_from(u64_field(values, "evaluated_samples")?)
                .map_err(|_| RegistryError::InvalidSerializedState)?;
            let valid_samples = usize::try_from(u64_field(values, "valid_samples")?)
                .map_err(|_| RegistryError::InvalidSerializedState)?;
            let valid_fraction = values.get("valid_fraction")
                .and_then(DataValue::as_f64)
                .ok_or(RegistryError::InvalidSerializedState)?;
            let passed = bool_field(values, "passed")?;
            let calculated_fraction = if evaluated_samples == 0 { 0.0 } else { valid_samples as f64 / evaluated_samples as f64 };
            let calculated_passed = evaluated_samples >= policy.minimum_validation_samples
                && calculated_fraction >= policy.minimum_valid_fraction;
            if evaluation_set_id.trim().is_empty()
                || evaluation_set_id == training_set_id
                || valid_samples > evaluated_samples
                || !valid_fraction.is_finite()
                || (valid_fraction - calculated_fraction).abs() > f64::EPSILON
                || passed != calculated_passed
            {
                return Err(RegistryError::InvalidSerializedState);
            }
            Some(RepresentationEvaluation {
                evaluation_set_id,
                policy,
                evaluated_samples,
                valid_samples,
                valid_fraction,
                passed,
            })
        }
        _ => return Err(RegistryError::InvalidSerializedState),
    };

    if id.trim().is_empty()
        || version == 0
        || training_set_id.trim().is_empty()
        || matches!(status, RepresentationStatus::Active | RepresentationStatus::Retired)
            && !evaluation.as_ref().is_some_and(|value| value.passed)
    {
        return Err(RegistryError::InvalidSerializedState);
    }

    Ok(RepresentationRecord {
        spec: RepresentationSpec {
            id,
            version,
            kind,
            schema,
            description,
            training_set_id,
            evidence_refs,
        },
        status,
        evaluation,
    })
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
    EmptyTrainingSetId,
    EvaluationSetLeakage,
    InvalidEvaluationPolicy,
    MissingEvaluation,
    EvaluationFailed,
    InvalidSerializedState,
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
            Self::EmptyTrainingSetId => write!(formatter, "training set id must not be empty"),
            Self::EvaluationSetLeakage => {
                write!(formatter, "evaluation set must differ from training set")
            }
            Self::InvalidEvaluationPolicy => write!(formatter, "evaluation policy is invalid"),
            Self::MissingEvaluation => write!(formatter, "candidate has not been evaluated"),
            Self::EvaluationFailed => write!(formatter, "candidate did not pass evaluation"),
            Self::InvalidSerializedState => write!(formatter, "serialized representation registry is invalid"),
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
            training_set_id: "training-set".to_owned(),
            evidence_refs: Vec::new(),
        }
    }

    #[test]
    fn registry_artifact_round_trip_preserves_status_and_evaluation() {
        let mut registry = RepresentationRegistry::default();
        registry.register_candidate(spec(1)).expect("candidate v1");
        registry
            .evaluate_candidate(
                "state-model",
                1,
                "held-out-v1",
                &[DataValue::object([(
                    "state".to_owned(),
                    DataValue::Object(std::collections::BTreeMap::new()),
                )])],
                EvaluationPolicy {
                    minimum_validation_samples: 1,
                    minimum_valid_fraction: 1.0,
                },
            )
            .expect("evaluate");
        registry.promote("state-model", 1).expect("promote");
        registry.register_candidate(spec(2)).expect("candidate v2");

        let restored = RepresentationRegistry::from_artifact_value(&registry.to_artifact_value())
            .expect("restore");
        assert_eq!(
            restored.active("state-model").map(|record| record.spec.version),
            Some(1)
        );
        assert_eq!(
            restored.get("state-model", 2).map(|record| record.status),
            Some(RepresentationStatus::Candidate)
        );
        assert_eq!(
            restored.active("state-model").and_then(|record| record.evaluation.as_ref())
                .map(|evaluation| evaluation.evaluation_set_id.as_str()),
            Some("held-out-v1")
        );
    }

    #[test]
    fn registry_artifact_rejects_unknown_version() {
        let value = DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(9)),
            ("records".to_owned(), DataValue::Sequence(Vec::new())),
        ]);
        assert!(matches!(
            RepresentationRegistry::from_artifact_value(&value),
            Err(RegistryError::InvalidSerializedState)
        ));
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
    fn rejects_evaluation_on_training_dataset() {
        let mut registry = RepresentationRegistry::default();
        registry.register_candidate(spec(1)).expect("candidate");

        assert_eq!(
            registry.evaluate_candidate(
                "state-model",
                1,
                "training-set",
                &[],
                EvaluationPolicy {
                    minimum_validation_samples: 1,
                    minimum_valid_fraction: 1.0,
                },
            ),
            Err(RegistryError::EvaluationSetLeakage)
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
