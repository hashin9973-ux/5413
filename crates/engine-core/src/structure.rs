use crate::value::DataValue;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiscoveryLimits {
    pub maximum_depth: usize,
    pub maximum_nodes: usize,
}

impl Default for DiscoveryLimits {
    fn default() -> Self {
        Self {
            maximum_depth: 128,
            maximum_nodes: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscoveredKind {
    Unknown,
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

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct StructureNode {
    /// Number of values represented by this node. For sequence elements, this is
    /// the total number of observed elements rather than the number of sequences.
    pub observations: u64,
    pub null_observations: u64,
    pub kind: Option<DiscoveredKind>,
    pub fields: BTreeMap<String, StructureNode>,
    pub sequence_element: Option<Box<StructureNode>>,
}

impl StructureNode {
    pub fn is_optional_within(&self, parent_observations: u64) -> bool {
        self.observations < parent_observations
    }

    pub fn field_is_required(&self, name: &str) -> Option<bool> {
        self.fields
            .get(name)
            .map(|field| field.observations == self.observations)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructureProfile {
    pub sample_count: u64,
    pub visited_nodes: usize,
    pub root: StructureNode,
}

/// Infers a recursive structural profile from heterogeneous observations.
///
/// Object fields are unioned across samples, missing fields remain visible through
/// their lower observation count, number/integer mixtures widen to Number, and
/// incompatible types widen to Any. This describes structure, not semantics.
pub fn infer_recursive_structure(
    samples: &[DataValue],
    limits: DiscoveryLimits,
) -> Result<StructureProfile, StructureDiscoveryError> {
    if samples.is_empty() {
        return Err(StructureDiscoveryError::NoSamples);
    }
    if limits.maximum_depth == 0 || limits.maximum_nodes == 0 {
        return Err(StructureDiscoveryError::InvalidLimits);
    }

    let mut root = StructureNode::default();
    let mut visited_nodes = 0usize;
    for (sample_index, sample) in samples.iter().enumerate() {
        sample
            .validate_finite()
            .map_err(|_| StructureDiscoveryError::NonFiniteSample { sample_index })?;
        observe(
            &mut root,
            sample,
            0,
            limits,
            &mut visited_nodes,
        )?;
    }

    Ok(StructureProfile {
        sample_count: u64::try_from(samples.len())
            .map_err(|_| StructureDiscoveryError::CountOverflow)?,
        visited_nodes,
        root,
    })
}

fn observe(
    node: &mut StructureNode,
    value: &DataValue,
    depth: usize,
    limits: DiscoveryLimits,
    visited_nodes: &mut usize,
) -> Result<(), StructureDiscoveryError> {
    if depth > limits.maximum_depth {
        return Err(StructureDiscoveryError::MaximumDepthExceeded {
            maximum: limits.maximum_depth,
        });
    }
    *visited_nodes = visited_nodes
        .checked_add(1)
        .ok_or(StructureDiscoveryError::CountOverflow)?;
    if *visited_nodes > limits.maximum_nodes {
        return Err(StructureDiscoveryError::MaximumNodesExceeded {
            maximum: limits.maximum_nodes,
        });
    }

    node.observations = node
        .observations
        .checked_add(1)
        .ok_or(StructureDiscoveryError::CountOverflow)?;
    let observed_kind = kind_of(value);
    node.kind = Some(match node.kind {
        None => observed_kind,
        Some(previous) => merge_kinds(previous, observed_kind),
    });

    match value {
        DataValue::Null => {
            node.null_observations = node
                .null_observations
                .checked_add(1)
                .ok_or(StructureDiscoveryError::CountOverflow)?;
        }
        DataValue::Object(fields) => {
            for (name, field_value) in fields {
                observe(
                    node.fields.entry(name.clone()).or_default(),
                    field_value,
                    depth + 1,
                    limits,
                    visited_nodes,
                )?;
            }
        }
        DataValue::Sequence(values) => {
            if !values.is_empty() {
                let element = node
                    .sequence_element
                    .get_or_insert_with(|| Box::new(StructureNode::default()));
                for item in values {
                    observe(element, item, depth + 1, limits, visited_nodes)?;
                }
            }
        }
        DataValue::Bool(_)
        | DataValue::Integer(_)
        | DataValue::Number(_)
        | DataValue::Text(_)
        | DataValue::BlobRef(_) => {}
    }
    Ok(())
}

fn kind_of(value: &DataValue) -> DiscoveredKind {
    match value {
        DataValue::Null => DiscoveredKind::Null,
        DataValue::Bool(_) => DiscoveredKind::Boolean,
        DataValue::Integer(_) => DiscoveredKind::Integer,
        DataValue::Number(_) => DiscoveredKind::Number,
        DataValue::Text(_) => DiscoveredKind::Text,
        DataValue::Sequence(_) => DiscoveredKind::Sequence,
        DataValue::Object(_) => DiscoveredKind::Object,
        DataValue::BlobRef(_) => DiscoveredKind::BlobRef,
    }
}

fn merge_kinds(left: DiscoveredKind, right: DiscoveredKind) -> DiscoveredKind {
    use DiscoveredKind::*;
    match (left, right) {
        (Unknown, kind) | (kind, Unknown) => kind,
        (Any, _) | (_, Any) => Any,
        (Null, kind) | (kind, Null) => kind,
        (Integer, Number) | (Number, Integer) => Number,
        (left, right) if left == right => left,
        _ => Any,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructureDiscoveryError {
    NoSamples,
    InvalidLimits,
    MaximumDepthExceeded { maximum: usize },
    MaximumNodesExceeded { maximum: usize },
    NonFiniteSample { sample_index: usize },
    CountOverflow,
}

impl std::fmt::Display for StructureDiscoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSamples => write!(formatter, "at least one observation is required"),
            Self::InvalidLimits => write!(formatter, "discovery limits must be greater than zero"),
            Self::MaximumDepthExceeded { maximum } => {
                write!(formatter, "observation nesting exceeds maximum depth {maximum}")
            }
            Self::MaximumNodesExceeded { maximum } => {
                write!(formatter, "observation structure exceeds maximum node count {maximum}")
            }
            Self::NonFiniteSample { sample_index } => {
                write!(formatter, "sample {sample_index} contains a non-finite number")
            }
            Self::CountOverflow => write!(formatter, "structure discovery counter is exhausted"),
        }
    }
}

impl std::error::Error for StructureDiscoveryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_nested_objects_optional_fields_and_numeric_widening() {
        let samples = vec![
            DataValue::object([
                (
                    "agent".to_owned(),
                    DataValue::object([
                        ("position".to_owned(), DataValue::Integer(4)),
                        ("label".to_owned(), DataValue::Text("ready".to_owned())),
                    ]),
                ),
                (
                    "sensors".to_owned(),
                    DataValue::Sequence(vec![DataValue::Number(1.0), DataValue::Number(2.0)]),
                ),
            ]),
            DataValue::object([
                (
                    "agent".to_owned(),
                    DataValue::object([("position".to_owned(), DataValue::Number(4.5))]),
                ),
                ("sensors".to_owned(), DataValue::Sequence(vec![DataValue::Number(3.0)])),
            ]),
        ];

        let profile =
            infer_recursive_structure(&samples, DiscoveryLimits::default()).expect("profile");
        let agent = profile.root.fields.get("agent").expect("agent");
        let position = agent.fields.get("position").expect("position");
        let label = agent.fields.get("label").expect("label");
        assert_eq!(position.kind, Some(DiscoveredKind::Number));
        assert_eq!(label.kind, Some(DiscoveredKind::Text));
        assert!(agent.field_is_required("position").expect("position required"));
        assert!(!agent.field_is_required("label").expect("label optional"));
        let sensors = profile.root.fields.get("sensors").expect("sensors");
        assert_eq!(
            sensors.sequence_element.as_deref().and_then(|node| node.kind),
            Some(DiscoveredKind::Number)
        );
    }

    #[test]
    fn incompatible_types_generalize_to_any_and_null_is_tracked() {
        let profile = infer_recursive_structure(
            &[
                DataValue::object([("value".to_owned(), DataValue::Null)]),
                DataValue::object([("value".to_owned(), DataValue::Text("present".to_owned()))]),
                DataValue::object([("value".to_owned(), DataValue::Bool(true))]),
            ],
            DiscoveryLimits::default(),
        )
        .expect("profile");
        let value = profile.root.fields.get("value").expect("value");
        assert_eq!(value.kind, Some(DiscoveredKind::Any));
        assert_eq!(value.null_observations, 1);
    }

    #[test]
    fn enforces_depth_and_node_limits() {
        let nested = DataValue::object([(
            "outer".to_owned(),
            DataValue::object([("inner".to_owned(), DataValue::Integer(1))]),
        )]);
        assert!(matches!(
            infer_recursive_structure(
                &[nested],
                DiscoveryLimits {
                    maximum_depth: 1,
                    maximum_nodes: 100,
                }
            ),
            Err(StructureDiscoveryError::MaximumDepthExceeded { .. })
        ));
        assert!(matches!(
            infer_recursive_structure(
                &[DataValue::Sequence(vec![
                    DataValue::Integer(1),
                    DataValue::Integer(2),
                ])],
                DiscoveryLimits {
                    maximum_depth: 8,
                    maximum_nodes: 2,
                }
            ),
            Err(StructureDiscoveryError::MaximumNodesExceeded { .. })
        ));
    }
}
