use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum DataValue {
    Null,
    Bool(bool),
    Integer(i64),
    Number(f64),
    Text(String),
    Sequence(Vec<DataValue>),
    Object(BTreeMap<String, DataValue>),
    BlobRef(String),
}

impl DataValue {
    pub fn object(entries: impl IntoIterator<Item = (String, DataValue)>) -> Self {
        Self::Object(entries.into_iter().collect())
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, DataValue>> {
        match self {
            Self::Object(values) => Some(values),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&DataValue> {
        self.as_object()?.get(key)
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Integer(value) => Some(*value as f64),
            Self::Number(value) => Some(*value),
            _ => None,
        }
    }

    pub fn validate_finite(&self) -> Result<(), DataValueError> {
        self.validate_finite_at("$")
    }

    fn validate_finite_at(&self, path: &str) -> Result<(), DataValueError> {
        match self {
            Self::Number(value) if !value.is_finite() => {
                Err(DataValueError::NonFiniteNumber {
                    path: path.to_owned(),
                })
            }
            Self::Sequence(values) => {
                for (index, value) in values.iter().enumerate() {
                    value.validate_finite_at(&format!("{path}[{index}]"))?;
                }
                Ok(())
            }
            Self::Object(values) => {
                for (key, value) in values {
                    value.validate_finite_at(&format!("{path}.{key}"))?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "boolean",
            Self::Integer(_) => "integer",
            Self::Number(_) => "number",
            Self::Text(_) => "text",
            Self::Sequence(_) => "sequence",
            Self::Object(_) => "object",
            Self::BlobRef(_) => "blob reference",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DataValueError {
    NonFiniteNumber { path: String },
}

impl std::fmt::Display for DataValueError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteNumber { path } => {
                write!(formatter, "non-finite number at {path}")
            }
        }
    }
}

impl std::error::Error for DataValueError {}
