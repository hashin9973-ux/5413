use crate::representation::{RepresentationKind, RepresentationSpec};
use crate::schema::{FieldSpec, SchemaDefinition, SchemaDefinitionError, ValueKind};
use crate::value::DataValue;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscoverySummary {
    pub sample_count: usize,
    pub discovered_field_count: usize,
    pub optional_field_count: usize,
    pub generalized_field_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationProposal {
    pub spec: RepresentationSpec,
    pub summary: DiscoverySummary,
}

pub fn propose_representation(
    id: impl Into<String>,
    version: u64,
    kind: RepresentationKind,
    description: impl Into<String>,
    training_set_id: impl Into<String>,
    evidence_refs: Vec<String>,
    samples: &[DataValue],
) -> Result<RepresentationProposal, DiscoveryError> {
    let id = id.into();
    let training_set_id = training_set_id.into();
    if id.trim().is_empty() {
        return Err(DiscoveryError::EmptyRepresentationId);
    }
    if version == 0 {
        return Err(DiscoveryError::ZeroVersion);
    }
    if training_set_id.trim().is_empty() {
        return Err(DiscoveryError::EmptyTrainingSetId);
    }
    if samples.is_empty() {
        return Err(DiscoveryError::NoSamples);
    }

    let mut observations: BTreeMap<String, (ValueKind, usize)> = BTreeMap::new();

    for (index, sample) in samples.iter().enumerate() {
        sample
            .validate_finite()
            .map_err(|_| DiscoveryError::InvalidNumericValue {
                sample_index: index,
            })?;
        let object = sample
            .as_object()
            .ok_or(DiscoveryError::ExpectedObjectSample {
                sample_index: index,
            })?;

        for (field_name, value) in object {
            let observed_kind = infer_kind(value);
            observations
                .entry(field_name.clone())
                .and_modify(|(kind, count)| {
                    *kind = merge_kinds(*kind, observed_kind);
                    *count += 1;
                })
                .or_insert((observed_kind, 1));
        }
    }

    let mut optional_field_count = 0;
    let mut generalized_field_count = 0;
    let fields = observations
        .into_iter()
        .map(|(name, (kind, observed_count))| {
            let required = observed_count == samples.len();
            if !required {
                optional_field_count += 1;
            }
            if kind == ValueKind::Any {
                generalized_field_count += 1;
            }
            FieldSpec {
                name,
                kind,
                required,
            }
        })
        .collect::<Vec<_>>();

    let schema = SchemaDefinition {
        id: format!("{id}-schema"),
        version,
        fields,
        allow_additional_fields: false,
    };
    schema
        .validate_definition()
        .map_err(DiscoveryError::InvalidSchema)?;

    let discovered_field_count = schema.fields.len();
    let spec = RepresentationSpec {
        id,
        version,
        kind,
        schema,
        description: description.into(),
        training_set_id,
        evidence_refs,
    };

    Ok(RepresentationProposal {
        spec,
        summary: DiscoverySummary {
            sample_count: samples.len(),
            discovered_field_count,
            optional_field_count,
            generalized_field_count,
        },
    })
}

fn infer_kind(value: &DataValue) -> ValueKind {
    match value {
        DataValue::Null => ValueKind::Null,
        DataValue::Bool(_) => ValueKind::Boolean,
        DataValue::Integer(_) => ValueKind::Integer,
        DataValue::Number(_) => ValueKind::Number,
        DataValue::Text(_) => ValueKind::Text,
        DataValue::Sequence(_) => ValueKind::Sequence,
        DataValue::Object(_) => ValueKind::Object,
        DataValue::BlobRef(_) => ValueKind::BlobRef,
    }
}

fn merge_kinds(left: ValueKind, right: ValueKind) -> ValueKind {
    if left == right {
        return left;
    }
    match (left, right) {
        (ValueKind::Integer, ValueKind::Number) | (ValueKind::Number, ValueKind::Integer) => {
            ValueKind::Number
        }
        _ => ValueKind::Any,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscoveryError {
    EmptyRepresentationId,
    ZeroVersion,
    EmptyTrainingSetId,
    NoSamples,
    ExpectedObjectSample { sample_index: usize },
    InvalidNumericValue { sample_index: usize },
    InvalidSchema(SchemaDefinitionError),
}

impl std::fmt::Display for DiscoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRepresentationId => write!(formatter, "representation id must not be empty"),
            Self::ZeroVersion => write!(
                formatter,
                "representation version must be greater than zero"
            ),
            Self::EmptyTrainingSetId => write!(formatter, "training set id must not be empty"),
            Self::NoSamples => write!(formatter, "at least one sample is required"),
            Self::ExpectedObjectSample { sample_index } => {
                write!(formatter, "sample {sample_index} is not an object")
            }
            Self::InvalidNumericValue { sample_index } => {
                write!(
                    formatter,
                    "sample {sample_index} contains non-finite numeric data"
                )
            }
            Self::InvalidSchema(error) => {
                write!(formatter, "inferred schema is invalid: {error}")
            }
        }
    }
}

impl std::error::Error for DiscoveryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_union_of_fields_and_marks_missing_fields_optional() {
        let samples = vec![
            DataValue::object([
                ("position".to_owned(), DataValue::Integer(2)),
                ("label".to_owned(), DataValue::Text("A".to_owned())),
            ]),
            DataValue::object([("position".to_owned(), DataValue::Number(3.5))]),
        ];

        let proposal = propose_representation(
            "observed-state",
            1,
            RepresentationKind::Relational,
            "Inferred from observations",
            "training-set-a",
            vec!["episode:1".to_owned(), "episode:2".to_owned()],
            &samples,
        )
        .expect("schema proposal should succeed");

        let position = proposal
            .spec
            .schema
            .fields
            .iter()
            .find(|field| field.name == "position")
            .expect("position field");
        let label = proposal
            .spec
            .schema
            .fields
            .iter()
            .find(|field| field.name == "label")
            .expect("label field");

        assert_eq!(position.kind, ValueKind::Number);
        assert!(position.required);
        assert_eq!(label.kind, ValueKind::Text);
        assert!(!label.required);
        assert_eq!(proposal.summary.optional_field_count, 1);
        assert_eq!(proposal.summary.sample_count, 2);
        assert!(proposal.spec.schema.validate(&samples[0]).is_ok());
        assert!(proposal.spec.schema.validate(&samples[1]).is_ok());
    }

    #[test]
    fn generalizes_incompatible_observed_types_to_any() {
        let samples = vec![
            DataValue::object([("state".to_owned(), DataValue::Integer(1))]),
            DataValue::object([("state".to_owned(), DataValue::Text("ready".to_owned()))]),
        ];
        let proposal = propose_representation(
            "mixed-state",
            1,
            RepresentationKind::Custom("discovered".to_owned()),
            "Mixed state samples",
            "training-set-b",
            Vec::new(),
            &samples,
        )
        .expect("proposal should generalize");

        assert_eq!(proposal.spec.schema.fields[0].kind, ValueKind::Any);
        assert_eq!(proposal.summary.generalized_field_count, 1);
    }

    #[test]
    fn rejects_non_object_samples_instead_of_silently_ignoring_them() {
        let samples = vec![DataValue::Text("not a state object".to_owned())];
        assert_eq!(
            propose_representation(
                "invalid",
                1,
                RepresentationKind::Relational,
                "invalid",
                "training-set-invalid",
                Vec::new(),
                &samples
            ),
            Err(DiscoveryError::ExpectedObjectSample { sample_index: 0 })
        );
    }
}
