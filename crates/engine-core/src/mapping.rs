use crate::representation::RepresentationSpec;
use crate::value::DataValue;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationRef {
    pub id: String,
    pub version: u64,
}

impl RepresentationRef {
    pub fn from_spec(spec: &RepresentationSpec) -> Self {
        Self {
            id: spec.id.clone(),
            version: spec.version,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MappingTransform {
    Copy,
    NumericAffine { scale: f64, offset: f64 },
    Constant(DataValue),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldMapping {
    /// None is only valid for a constant mapping.
    pub source_field: Option<String>,
    pub target_field: String,
    pub transform: MappingTransform,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepresentationMapping {
    pub id: String,
    pub version: u64,
    pub source: RepresentationRef,
    pub target: RepresentationRef,
    pub fields: Vec<FieldMapping>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MappingReport {
    pub mapped_source_fields: usize,
    pub input_source_fields: usize,
    pub dropped_source_fields: usize,
    pub source_coverage: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MappingResult {
    pub value: DataValue,
    pub report: MappingReport,
}

impl RepresentationMapping {
    pub fn apply(
        &self,
        source_spec: &RepresentationSpec,
        target_spec: &RepresentationSpec,
        source_value: &DataValue,
    ) -> Result<MappingResult, MappingError> {
        if self.id.trim().is_empty() || self.version == 0 {
            return Err(MappingError::InvalidMappingIdentity);
        }
        if self.source != RepresentationRef::from_spec(source_spec) {
            return Err(MappingError::SourceRepresentationMismatch);
        }
        if self.target != RepresentationRef::from_spec(target_spec) {
            return Err(MappingError::TargetRepresentationMismatch);
        }

        source_spec
            .schema
            .validate(source_value)
            .map_err(MappingError::InvalidSource)?;
        let source_object = source_value
            .as_object()
            .ok_or(MappingError::ExpectedSourceObject)?;

        let target_fields: BTreeSet<&str> = target_spec
            .schema
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        let mut target_values = BTreeMap::new();
        let mut mapped_sources = BTreeSet::new();

        for field_mapping in &self.fields {
            if !target_fields.contains(field_mapping.target_field.as_str()) {
                return Err(MappingError::UnknownTargetField(
                    field_mapping.target_field.clone(),
                ));
            }
            if target_values.contains_key(&field_mapping.target_field) {
                return Err(MappingError::DuplicateTargetField(
                    field_mapping.target_field.clone(),
                ));
            }

            let mapped_value = match (&field_mapping.source_field, &field_mapping.transform) {
                (Some(source_field), MappingTransform::Copy) => {
                    let value = source_object
                        .get(source_field)
                        .ok_or_else(|| MappingError::MissingSourceField(source_field.clone()))?;
                    mapped_sources.insert(source_field.clone());
                    value.clone()
                }
                (
                    Some(source_field),
                    MappingTransform::NumericAffine { scale, offset },
                ) => {
                    if !scale.is_finite() || !offset.is_finite() {
                        return Err(MappingError::InvalidTransform);
                    }
                    let value = source_object
                        .get(source_field)
                        .ok_or_else(|| MappingError::MissingSourceField(source_field.clone()))?;
                    let number = value
                        .as_f64()
                        .ok_or_else(|| MappingError::ExpectedNumericSource(source_field.clone()))?;
                    let transformed = number * scale + offset;
                    if !transformed.is_finite() {
                        return Err(MappingError::NonFiniteResult(
                            field_mapping.target_field.clone(),
                        ));
                    }
                    mapped_sources.insert(source_field.clone());
                    DataValue::Number(transformed)
                }
                (None, MappingTransform::Constant(value)) => {
                    value
                        .validate_finite()
                        .map_err(|_| MappingError::InvalidConstant)?;
                    value.clone()
                }
                _ => return Err(MappingError::InvalidSourceTransformCombination),
            };

            target_values.insert(field_mapping.target_field.clone(), mapped_value);
        }

        let mapped_source_fields = mapped_sources.len();
        let input_source_fields = source_object.len();
        let source_coverage = if input_source_fields == 0 {
            1.0
        } else {
            mapped_source_fields as f64 / input_source_fields as f64
        };
        let report = MappingReport {
            mapped_source_fields,
            input_source_fields,
            dropped_source_fields: input_source_fields.saturating_sub(mapped_source_fields),
            source_coverage,
        };

        let value = DataValue::Object(target_values);
        target_spec
            .schema
            .validate(&value)
            .map_err(MappingError::InvalidTarget)?;

        Ok(MappingResult { value, report })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MappingError {
    InvalidMappingIdentity,
    SourceRepresentationMismatch,
    TargetRepresentationMismatch,
    ExpectedSourceObject,
    InvalidSource(crate::schema::SchemaValidationError),
    InvalidTarget(crate::schema::SchemaValidationError),
    UnknownTargetField(String),
    DuplicateTargetField(String),
    MissingSourceField(String),
    ExpectedNumericSource(String),
    InvalidTransform,
    InvalidConstant,
    InvalidSourceTransformCombination,
    NonFiniteResult(String),
}

impl std::fmt::Display for MappingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMappingIdentity => write!(formatter, "mapping identity is invalid"),
            Self::SourceRepresentationMismatch => write!(formatter, "source representation mismatch"),
            Self::TargetRepresentationMismatch => write!(formatter, "target representation mismatch"),
            Self::ExpectedSourceObject => write!(formatter, "source value must be an object"),
            Self::InvalidSource(error) => write!(formatter, "source value is invalid: {error}"),
            Self::InvalidTarget(error) => write!(formatter, "mapped target is invalid: {error}"),
            Self::UnknownTargetField(field) => write!(formatter, "unknown target field: {field}"),
            Self::DuplicateTargetField(field) => write!(formatter, "target field mapped more than once: {field}"),
            Self::MissingSourceField(field) => write!(formatter, "missing source field: {field}"),
            Self::ExpectedNumericSource(field) => write!(formatter, "source field is not numeric: {field}"),
            Self::InvalidTransform => write!(formatter, "mapping transform contains non-finite parameters"),
            Self::InvalidConstant => write!(formatter, "mapping constant contains invalid data"),
            Self::InvalidSourceTransformCombination => write!(formatter, "mapping source and transform combination is invalid"),
            Self::NonFiniteResult(field) => write!(formatter, "mapping produced a non-finite value for {field}"),
        }
    }
}

impl std::error::Error for MappingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::representation::RepresentationKind;
    use crate::schema::{FieldSpec, SchemaDefinition, ValueKind};

    fn spec(id: &str, fields: Vec<(&str, ValueKind)>) -> RepresentationSpec {
        RepresentationSpec {
            id: id.to_owned(),
            version: 1,
            kind: RepresentationKind::Relational,
            schema: SchemaDefinition {
                id: format!("{id}-schema"),
                version: 1,
                fields: fields
                    .into_iter()
                    .map(|(name, kind)| FieldSpec {
                        name: name.to_owned(),
                        kind,
                        required: true,
                    })
                    .collect(),
                allow_additional_fields: false,
            },
            description: id.to_owned(),
            evidence_refs: Vec::new(),
        }
    }

    #[test]
    fn maps_between_distinct_representation_schemas() {
        let source = spec(
            "source-state",
            vec![("x", ValueKind::Number), ("unused", ValueKind::Text)],
        );
        let target = spec(
            "target-state",
            vec![("position", ValueKind::Number), ("kind", ValueKind::Text)],
        );
        let mapping = RepresentationMapping {
            id: "source-to-target".to_owned(),
            version: 1,
            source: RepresentationRef::from_spec(&source),
            target: RepresentationRef::from_spec(&target),
            fields: vec![
                FieldMapping {
                    source_field: Some("x".to_owned()),
                    target_field: "position".to_owned(),
                    transform: MappingTransform::NumericAffine {
                        scale: 2.0,
                        offset: 1.0,
                    },
                },
                FieldMapping {
                    source_field: None,
                    target_field: "kind".to_owned(),
                    transform: MappingTransform::Constant(DataValue::Text("point".to_owned())),
                },
            ],
        };
        let source_value = DataValue::object([
            ("x".to_owned(), DataValue::Number(3.0)),
            ("unused".to_owned(), DataValue::Text("evidence".to_owned())),
        ]);

        let result = mapping
            .apply(&source, &target, &source_value)
            .expect("mapping should be valid");

        assert_eq!(
            result.value.get("position"),
            Some(&DataValue::Number(7.0))
        );
        assert_eq!(
            result.value.get("kind"),
            Some(&DataValue::Text("point".to_owned()))
        );
        assert_eq!(result.report.dropped_source_fields, 1);
        assert_eq!(result.report.source_coverage, 0.5);
    }

    #[test]
    fn rejects_invalid_target_schema_after_mapping() {
        let source = spec("source", vec![("x", ValueKind::Number)]);
        let target = spec(
            "target",
            vec![("required", ValueKind::Text)],
        );
        let mapping = RepresentationMapping {
            id: "bad-mapping".to_owned(),
            version: 1,
            source: RepresentationRef::from_spec(&source),
            target: RepresentationRef::from_spec(&target),
            fields: Vec::new(),
        };
        let source_value = DataValue::object([("x".to_owned(), DataValue::Number(1.0))]);

        assert!(matches!(
            mapping.apply(&source, &target, &source_value),
            Err(MappingError::InvalidTarget(_))
        ));
    }
}
