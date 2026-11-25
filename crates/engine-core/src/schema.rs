use crate::value::{DataValue, DataValueError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Any,
    Null,
    Boolean,
    Integer,
    Number,
    Text,
    Sequence,
    Object,
    BlobRef,
}

impl ValueKind {
    fn accepts(self, value: &DataValue) -> bool {
        match self {
            Self::Any => true,
            Self::Null => matches!(value, DataValue::Null),
            Self::Boolean => matches!(value, DataValue::Bool(_)),
            Self::Integer => matches!(value, DataValue::Integer(_)),
            Self::Number => matches!(value, DataValue::Integer(_) | DataValue::Number(_)),
            Self::Text => matches!(value, DataValue::Text(_)),
            Self::Sequence => matches!(value, DataValue::Sequence(_)),
            Self::Object => matches!(value, DataValue::Object(_)),
            Self::BlobRef => matches!(value, DataValue::BlobRef(_)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldSpec {
    pub name: String,
    pub kind: ValueKind,
    pub required: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaDefinition {
    pub id: String,
    pub version: u64,
    pub fields: Vec<FieldSpec>,
    pub allow_additional_fields: bool,
}

impl SchemaDefinition {
    pub fn validate_definition(&self) -> Result<(), SchemaDefinitionError> {
        if self.id.trim().is_empty() {
            return Err(SchemaDefinitionError::EmptyId);
        }
        if self.version == 0 {
            return Err(SchemaDefinitionError::ZeroVersion);
        }

        let mut names = std::collections::BTreeSet::new();
        for field in &self.fields {
            if field.name.trim().is_empty() {
                return Err(SchemaDefinitionError::EmptyFieldName);
            }
            if !names.insert(field.name.as_str()) {
                return Err(SchemaDefinitionError::DuplicateField(field.name.clone()));
            }
        }
        Ok(())
    }

    pub fn validate(&self, value: &DataValue) -> Result<(), SchemaValidationError> {
        self.validate_definition()
            .map_err(SchemaValidationError::InvalidDefinition)?;
        value
            .validate_finite()
            .map_err(SchemaValidationError::InvalidValue)?;

        let object = value
            .as_object()
            .ok_or(SchemaValidationError::ExpectedObject)?;

        for field in &self.fields {
            match object.get(&field.name) {
                Some(value) if !field.kind.accepts(value) => {
                    return Err(SchemaValidationError::TypeMismatch {
                        field: field.name.clone(),
                        expected: field.kind,
                        actual: value.kind_name(),
                    });
                }
                None if field.required => {
                    return Err(SchemaValidationError::MissingRequiredField(
                        field.name.clone(),
                    ));
                }
                _ => {}
            }
        }

        if !self.allow_additional_fields {
            for name in object.keys() {
                if !self.fields.iter().any(|field| field.name == *name) {
                    return Err(SchemaValidationError::UnknownField(name.clone()));
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchemaDefinitionError {
    EmptyId,
    ZeroVersion,
    EmptyFieldName,
    DuplicateField(String),
}

impl std::fmt::Display for SchemaDefinitionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyId => write!(formatter, "schema id must not be empty"),
            Self::ZeroVersion => write!(formatter, "schema version must be greater than zero"),
            Self::EmptyFieldName => write!(formatter, "schema field name must not be empty"),
            Self::DuplicateField(name) => write!(formatter, "duplicate schema field: {name}"),
        }
    }
}

impl std::error::Error for SchemaDefinitionError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchemaValidationError {
    InvalidDefinition(SchemaDefinitionError),
    InvalidValue(DataValueError),
    ExpectedObject,
    MissingRequiredField(String),
    UnknownField(String),
    TypeMismatch {
        field: String,
        expected: ValueKind,
        actual: &'static str,
    },
}

impl std::fmt::Display for SchemaValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDefinition(error) => write!(formatter, "invalid schema: {error}"),
            Self::InvalidValue(error) => write!(formatter, "invalid value: {error}"),
            Self::ExpectedObject => write!(formatter, "schema payload must be an object"),
            Self::MissingRequiredField(field) => {
                write!(formatter, "missing required field: {field}")
            }
            Self::UnknownField(field) => write!(formatter, "unknown field: {field}"),
            Self::TypeMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "field {field} expected {expected:?}, received {actual}"
            ),
        }
    }
}

impl std::error::Error for SchemaValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::DataValue;
    use std::collections::BTreeMap;

    fn schema() -> SchemaDefinition {
        SchemaDefinition {
            id: "sample-state".to_owned(),
            version: 1,
            fields: vec![
                FieldSpec {
                    name: "count".to_owned(),
                    kind: ValueKind::Integer,
                    required: true,
                },
                FieldSpec {
                    name: "label".to_owned(),
                    kind: ValueKind::Text,
                    required: false,
                },
            ],
            allow_additional_fields: false,
        }
    }

    #[test]
    fn validates_dynamic_object_payload() {
        let payload = DataValue::Object(BTreeMap::from([(
            "count".to_owned(),
            DataValue::Integer(3),
        )]));
        assert!(schema().validate(&payload).is_ok());
    }

    #[test]
    fn rejects_unknown_fields_when_schema_is_closed() {
        let payload = DataValue::Object(BTreeMap::from([
            ("count".to_owned(), DataValue::Integer(3)),
            ("unexpected".to_owned(), DataValue::Bool(true)),
        ]));
        assert_eq!(
            schema().validate(&payload),
            Err(SchemaValidationError::UnknownField("unexpected".to_owned()))
        );
    }

    #[test]
    fn rejects_non_finite_nested_numbers() {
        let payload = DataValue::Object(BTreeMap::from([(
            "count".to_owned(),
            DataValue::Sequence(vec![DataValue::Number(f64::NAN)]),
        )]));
        assert!(matches!(
            schema().validate(&payload),
            Err(SchemaValidationError::InvalidValue(_))
        ));
    }
}
