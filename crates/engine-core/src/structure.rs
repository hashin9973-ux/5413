use crate::value::DataValue;
use std::collections::{BTreeMap, BTreeSet};

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
        observe(&mut root, sample, 0, limits, &mut visited_nodes)?;
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

#[derive(Clone, Debug, PartialEq)]
pub struct TemporalFieldProfile {
    pub field: String,
    pub transition_count: usize,
    pub missing_transition_count: usize,
    pub mean_delta: f64,
    /// Population variance across the observed consecutive deltas.
    pub variance: f64,
    pub minimum_delta: f64,
    pub maximum_delta: f64,
    /// Fraction of adjacent non-zero deltas whose signs differ.
    pub direction_change_rate: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TemporalProfile {
    pub observation_count: usize,
    pub fields: Vec<TemporalFieldProfile>,
}

/// Summarizes consecutive numeric changes in a caller-ordered observation stream.
///
/// The input order is treated as temporal order. This function does not infer
/// timestamps, resample irregular intervals, or interpolate missing fields.
pub fn analyze_temporal_sequence(
    observations: &[DataValue],
) -> Result<TemporalProfile, StructureDiscoveryError> {
    if observations.len() < 2 {
        return Err(StructureDiscoveryError::InsufficientTemporalSamples);
    }
    for (sample_index, observation) in observations.iter().enumerate() {
        observation
            .validate_finite()
            .map_err(|_| StructureDiscoveryError::NonFiniteSample { sample_index })?;
        if observation.as_object().is_none() {
            return Err(StructureDiscoveryError::ExpectedObjectObservation { sample_index });
        }
    }

    let mut deltas: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for pair in observations.windows(2) {
        let before = pair[0]
            .as_object()
            .ok_or(StructureDiscoveryError::ExpectedObjectObservation { sample_index: 0 })?;
        let after = pair[1]
            .as_object()
            .ok_or(StructureDiscoveryError::ExpectedObjectObservation { sample_index: 1 })?;
        for (name, before_value) in before {
            let (Some(before_number), Some(after_number)) = (
                before_value.as_f64(),
                after.get(name).and_then(DataValue::as_f64),
            ) else {
                continue;
            };
            let delta = after_number - before_number;
            if !delta.is_finite() {
                return Err(StructureDiscoveryError::NumericalOverflow);
            }
            deltas.entry(name.clone()).or_default().push(delta);
        }
    }
    if deltas.is_empty() {
        return Err(StructureDiscoveryError::NoNumericTemporalFields);
    }

    let total_transitions = observations.len() - 1;
    let mut fields = Vec::with_capacity(deltas.len());
    for (name, values) in deltas {
        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / count;
        let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let signs: Vec<i8> = values
            .iter()
            .filter_map(|value| {
                if *value > 0.0 {
                    Some(1)
                } else if *value < 0.0 {
                    Some(-1)
                } else {
                    None
                }
            })
            .collect();
        let direction_changes = signs.windows(2).filter(|pair| pair[0] != pair[1]).count();
        let direction_change_rate = if signs.len() < 2 {
            0.0
        } else {
            direction_changes as f64 / (signs.len() - 1) as f64
        };
        if !mean.is_finite()
            || !variance.is_finite()
            || !minimum.is_finite()
            || !maximum.is_finite()
            || !direction_change_rate.is_finite()
        {
            return Err(StructureDiscoveryError::NumericalOverflow);
        }
        let transition_count = values.len();
        fields.push(TemporalFieldProfile {
            field: name,
            transition_count,
            missing_transition_count: total_transitions - transition_count,
            mean_delta: mean,
            variance,
            minimum_delta: minimum,
            maximum_delta: maximum,
            direction_change_rate,
        });
    }
    Ok(TemporalProfile {
        observation_count: observations.len(),
        fields,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnresolvedReferencePolicy {
    AllowExternal,
    RequireKnownEntity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationEdge {
    pub source_id: String,
    pub relation_field: String,
    pub target_id: String,
    pub target_is_known: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationshipGraph {
    pub entity_ids: Vec<String>,
    pub edges: Vec<RelationEdge>,
}

/// Extracts directed relationships from caller-declared identity and reference fields.
///
/// Identity/reference values may be non-empty text, integer, or blob references;
/// reference fields may also contain a sequence of such identifiers. Field names
/// are never guessed from suffixes or domain conventions.
pub fn discover_relationships(
    entities: &[DataValue],
    identity_field: &str,
    relation_fields: &[String],
    unresolved_policy: UnresolvedReferencePolicy,
) -> Result<RelationshipGraph, StructureDiscoveryError> {
    if entities.is_empty() || identity_field.trim().is_empty() || relation_fields.is_empty() {
        return Err(StructureDiscoveryError::InvalidRelationConfiguration);
    }
    let mut relation_names = BTreeSet::new();
    if relation_fields
        .iter()
        .any(|field| field.trim().is_empty() || !relation_names.insert(field.as_str()))
    {
        return Err(StructureDiscoveryError::InvalidRelationConfiguration);
    }

    let mut identities = BTreeSet::new();
    let mut parsed = Vec::with_capacity(entities.len());
    for (sample_index, entity) in entities.iter().enumerate() {
        let object = entity
            .as_object()
            .ok_or(StructureDiscoveryError::InvalidEntity { sample_index })?;
        let identity_value =
            object
                .get(identity_field)
                .ok_or(StructureDiscoveryError::MissingIdentity {
                    sample_index,
                    field: identity_field.to_owned(),
                })?;
        let identity =
            identifier_value(identity_value).ok_or(StructureDiscoveryError::InvalidIdentity {
                sample_index,
                field: identity_field.to_owned(),
            })?;
        if !identities.insert(identity.clone()) {
            return Err(StructureDiscoveryError::DuplicateIdentity(identity));
        }
        parsed.push((identity, object));
    }

    let mut edges = Vec::new();
    for (source_id, object) in &parsed {
        for relation_field in relation_fields {
            let Some(value) = object.get(relation_field) else {
                continue;
            };
            let references: Vec<&DataValue> = match value {
                DataValue::Null => Vec::new(),
                DataValue::Sequence(values) => values.iter().collect(),
                scalar => vec![scalar],
            };
            for reference in references {
                let target_id = identifier_value(reference).ok_or(
                    StructureDiscoveryError::InvalidReference {
                        source_id: source_id.clone(),
                        field: relation_field.clone(),
                    },
                )?;
                let target_is_known = identities.contains(&target_id);
                if !target_is_known
                    && unresolved_policy == UnresolvedReferencePolicy::RequireKnownEntity
                {
                    return Err(StructureDiscoveryError::UnresolvedReference {
                        source_id: source_id.clone(),
                        field: relation_field.clone(),
                        target_id,
                    });
                }
                edges.push(RelationEdge {
                    source_id: source_id.clone(),
                    relation_field: relation_field.clone(),
                    target_id,
                    target_is_known,
                });
            }
        }
    }
    edges.sort_by(|left, right| {
        (&left.source_id, &left.relation_field, &left.target_id).cmp(&(
            &right.source_id,
            &right.relation_field,
            &right.target_id,
        ))
    });
    Ok(RelationshipGraph {
        entity_ids: identities.into_iter().collect(),
        edges,
    })
}

fn identifier_value(value: &DataValue) -> Option<String> {
    match value {
        DataValue::Text(value) if !value.trim().is_empty() => Some(value.clone()),
        DataValue::Integer(value) => Some(value.to_string()),
        DataValue::BlobRef(value) if !value.trim().is_empty() => Some(value.clone()),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldProfilePolicy {
    pub minimum_samples: usize,
    pub identifier_uniqueness_threshold: f64,
    pub maximum_categorical_cardinality: usize,
}

impl Default for FieldProfilePolicy {
    fn default() -> Self {
        Self {
            minimum_samples: 10,
            identifier_uniqueness_threshold: 0.95,
            maximum_categorical_cardinality: 32,
        }
    }
}

impl FieldProfilePolicy {
    pub fn validate(&self) -> Result<(), StructureDiscoveryError> {
        if self.minimum_samples == 0
            || !self.identifier_uniqueness_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.identifier_uniqueness_threshold)
            || self.maximum_categorical_cardinality == 0
        {
            return Err(StructureDiscoveryError::InvalidFieldProfilePolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldRole {
    Unknown,
    NumericMeasure,
    BooleanFlag,
    Categorical,
    IdentifierCandidate,
    Structured,
    Heterogeneous,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldProfile {
    pub name: String,
    pub sample_count: usize,
    /// Number of samples containing the field, including explicit null values.
    pub observed_count: usize,
    /// Number of samples where the field was absent.
    pub missing_count: usize,
    pub null_count: usize,
    pub unique_value_count: usize,
    pub uniqueness_ratio: f64,
    pub type_counts: BTreeMap<String, usize>,
    pub role: FieldRole,
    pub numeric_mean: Option<f64>,
    pub numeric_variance: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticProfile {
    pub sample_count: usize,
    pub fields: Vec<FieldProfile>,
}

/// Produces statistical field-role candidates without consulting field names.
///
/// The results are heuristics over observed distributions, not semantic labels
/// or guarantees that a text field is truly an identifier.
pub fn profile_fields(
    samples: &[DataValue],
    policy: FieldProfilePolicy,
) -> Result<SemanticProfile, StructureDiscoveryError> {
    policy.validate()?;
    if samples.len() < policy.minimum_samples {
        return Err(StructureDiscoveryError::InsufficientSemanticSamples {
            minimum: policy.minimum_samples,
            actual: samples.len(),
        });
    }

    let mut field_values: BTreeMap<String, Vec<Option<DataValue>>> = BTreeMap::new();
    for (sample_index, sample) in samples.iter().enumerate() {
        sample
            .validate_finite()
            .map_err(|_| StructureDiscoveryError::NonFiniteSample { sample_index })?;
        let object = sample
            .as_object()
            .ok_or(StructureDiscoveryError::ExpectedObjectSample { sample_index })?;
        for values in field_values.values_mut() {
            values.push(None);
        }
        for (name, value) in object {
            let values = field_values
                .entry(name.clone())
                .or_insert_with(|| vec![None; sample_index + 1]);
            if values.len() != sample_index + 1 {
                return Err(StructureDiscoveryError::CountOverflow);
            }
            values[sample_index] = Some(value.clone());
        }
    }

    let mut fields = Vec::with_capacity(field_values.len());
    for (name, values) in field_values {
        let mut unique_values: Vec<DataValue> = Vec::new();
        let mut type_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut numeric_values = Vec::new();
        let mut non_null_count = 0usize;
        let mut null_count = 0usize;
        let mut observed_count = 0usize;
        let mut non_null_unique_count = 0usize;

        for value in values.iter().flatten() {
            observed_count += 1;
            let type_name = match value {
                DataValue::Null => {
                    null_count += 1;
                    "null"
                }
                DataValue::Bool(_) => {
                    non_null_count += 1;
                    "boolean"
                }
                DataValue::Integer(_) => {
                    non_null_count += 1;
                    "integer"
                }
                DataValue::Number(_) => {
                    non_null_count += 1;
                    "number"
                }
                DataValue::Text(_) => {
                    non_null_count += 1;
                    "text"
                }
                DataValue::Sequence(_) => {
                    non_null_count += 1;
                    "sequence"
                }
                DataValue::Object(_) => {
                    non_null_count += 1;
                    "object"
                }
                DataValue::BlobRef(_) => {
                    non_null_count += 1;
                    "blob_ref"
                }
            };
            *type_counts.entry(type_name.to_owned()).or_insert(0) += 1;
            if !unique_values.iter().any(|existing| existing == value) {
                unique_values.push(value.clone());
                if !matches!(value, DataValue::Null) {
                    non_null_unique_count += 1;
                }
            }
            if let Some(number) = value.as_f64() {
                numeric_values.push(number);
            }
        }

        let missing_count = samples.len() - observed_count;
        let uniqueness_ratio = if non_null_count == 0 {
            0.0
        } else {
            non_null_unique_count as f64 / non_null_count as f64
        };
        let kinds: BTreeSet<&str> = type_counts
            .keys()
            .map(String::as_str)
            .filter(|kind| *kind != "null")
            .collect();
        let role = if kinds.is_empty() {
            FieldRole::Unknown
        } else if kinds
            .iter()
            .all(|kind| *kind == "integer" || *kind == "number")
        {
            FieldRole::NumericMeasure
        } else if kinds.len() == 1 && kinds.contains("boolean") {
            FieldRole::BooleanFlag
        } else if kinds.len() == 1 && kinds.contains("text") {
            if uniqueness_ratio >= policy.identifier_uniqueness_threshold
                && non_null_unique_count > policy.maximum_categorical_cardinality
            {
                FieldRole::IdentifierCandidate
            } else {
                FieldRole::Categorical
            }
        } else if kinds
            .iter()
            .all(|kind| *kind == "object" || *kind == "sequence")
        {
            FieldRole::Structured
        } else {
            FieldRole::Heterogeneous
        };

        let (numeric_mean, numeric_variance) =
            if role == FieldRole::NumericMeasure && !numeric_values.is_empty() {
                let count = numeric_values.len() as f64;
                let mean = numeric_values.iter().sum::<f64>() / count;
                let variance = numeric_values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / count;
                if !mean.is_finite() || !variance.is_finite() {
                    return Err(StructureDiscoveryError::NumericalOverflow);
                }
                (Some(mean), Some(variance))
            } else {
                (None, None)
            };

        fields.push(FieldProfile {
            name,
            sample_count: samples.len(),
            observed_count,
            missing_count,
            null_count,
            unique_value_count: unique_values.len(),
            uniqueness_ratio,
            type_counts,
            role,
            numeric_mean,
            numeric_variance,
        });
    }
    Ok(SemanticProfile {
        sample_count: samples.len(),
        fields,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructureDiscoveryError {
    NoSamples,
    InvalidLimits,
    MaximumDepthExceeded {
        maximum: usize,
    },
    MaximumNodesExceeded {
        maximum: usize,
    },
    NonFiniteSample {
        sample_index: usize,
    },
    CountOverflow,
    InsufficientTemporalSamples,
    ExpectedObjectObservation {
        sample_index: usize,
    },
    NoNumericTemporalFields,
    NumericalOverflow,
    InvalidRelationConfiguration,
    InvalidEntity {
        sample_index: usize,
    },
    MissingIdentity {
        sample_index: usize,
        field: String,
    },
    InvalidIdentity {
        sample_index: usize,
        field: String,
    },
    DuplicateIdentity(String),
    InvalidReference {
        source_id: String,
        field: String,
    },
    UnresolvedReference {
        source_id: String,
        field: String,
        target_id: String,
    },
    InvalidFieldProfilePolicy,
    InsufficientSemanticSamples {
        minimum: usize,
        actual: usize,
    },
    ExpectedObjectSample {
        sample_index: usize,
    },
}

impl std::fmt::Display for StructureDiscoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSamples => write!(formatter, "at least one observation is required"),
            Self::InvalidLimits => write!(formatter, "discovery limits must be greater than zero"),
            Self::MaximumDepthExceeded { maximum } => {
                write!(
                    formatter,
                    "observation nesting exceeds maximum depth {maximum}"
                )
            }
            Self::MaximumNodesExceeded { maximum } => {
                write!(
                    formatter,
                    "observation structure exceeds maximum node count {maximum}"
                )
            }
            Self::NonFiniteSample { sample_index } => {
                write!(
                    formatter,
                    "sample {sample_index} contains a non-finite number"
                )
            }
            Self::CountOverflow => write!(formatter, "structure discovery counter is exhausted"),
            Self::InsufficientTemporalSamples => write!(
                formatter,
                "temporal profiling requires at least two ordered observations"
            ),
            Self::ExpectedObjectObservation { sample_index } => write!(
                formatter,
                "temporal observation {sample_index} must be an object"
            ),
            Self::NoNumericTemporalFields => write!(
                formatter,
                "no numeric fields are shared by consecutive observations"
            ),
            Self::NumericalOverflow => write!(formatter, "discovery statistic overflowed"),
            Self::InvalidRelationConfiguration => {
                write!(formatter, "relationship discovery configuration is invalid")
            }
            Self::InvalidEntity { sample_index } => {
                write!(formatter, "entity {sample_index} must be an object")
            }
            Self::MissingIdentity {
                sample_index,
                field,
            } => write!(
                formatter,
                "entity {sample_index} is missing identity field {field}"
            ),
            Self::InvalidIdentity {
                sample_index,
                field,
            } => write!(
                formatter,
                "entity {sample_index} has an invalid identity in field {field}"
            ),
            Self::DuplicateIdentity(id) => write!(formatter, "entity identity {id} is duplicated"),
            Self::InvalidReference { source_id, field } => write!(
                formatter,
                "entity {source_id} has an invalid reference in field {field}"
            ),
            Self::UnresolvedReference {
                source_id,
                field,
                target_id,
            } => write!(
                formatter,
                "entity {source_id} field {field} references unknown entity {target_id}"
            ),
            Self::InvalidFieldProfilePolicy => {
                write!(formatter, "field profiling policy is invalid")
            }
            Self::InsufficientSemanticSamples { minimum, actual } => write!(
                formatter,
                "field profiling needs at least {minimum} samples; received {actual}"
            ),
            Self::ExpectedObjectSample { sample_index } => write!(
                formatter,
                "field profiling sample {sample_index} must be an object"
            ),
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
                (
                    "sensors".to_owned(),
                    DataValue::Sequence(vec![DataValue::Number(3.0)]),
                ),
            ]),
        ];

        let profile =
            infer_recursive_structure(&samples, DiscoveryLimits::default()).expect("profile");
        let agent = profile.root.fields.get("agent").expect("agent");
        let position = agent.fields.get("position").expect("position");
        let label = agent.fields.get("label").expect("label");
        assert_eq!(position.kind, Some(DiscoveredKind::Number));
        assert_eq!(label.kind, Some(DiscoveredKind::Text));
        assert!(agent
            .field_is_required("position")
            .expect("position required"));
        assert!(!agent.field_is_required("label").expect("label optional"));
        let sensors = profile.root.fields.get("sensors").expect("sensors");
        assert_eq!(
            sensors
                .sequence_element
                .as_deref()
                .and_then(|node| node.kind),
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
    fn temporal_profile_measures_numeric_deltas_and_missing_transitions() {
        let observations = vec![
            DataValue::object([
                ("position".to_owned(), DataValue::Number(0.0)),
                ("temperature".to_owned(), DataValue::Number(10.0)),
            ]),
            DataValue::object([
                ("position".to_owned(), DataValue::Number(1.0)),
                ("temperature".to_owned(), DataValue::Number(11.0)),
            ]),
            DataValue::object([("position".to_owned(), DataValue::Number(3.0))]),
            DataValue::object([
                ("position".to_owned(), DataValue::Number(2.0)),
                ("temperature".to_owned(), DataValue::Number(14.0)),
            ]),
            DataValue::object([("position".to_owned(), DataValue::Number(2.5))]),
        ];
        let profile = analyze_temporal_sequence(&observations).expect("temporal profile");
        let position = profile
            .fields
            .iter()
            .find(|field| field.field == "position")
            .unwrap();
        assert_eq!(position.transition_count, 4);
        assert_eq!(position.missing_transition_count, 0);
        assert!((position.mean_delta - 0.625).abs() < 1e-12);
        assert!((position.direction_change_rate - (2.0 / 3.0)).abs() < 1e-12);

        let temperature = profile
            .fields
            .iter()
            .find(|field| field.field == "temperature")
            .unwrap();
        assert_eq!(temperature.transition_count, 1);
        assert_eq!(temperature.missing_transition_count, 3);
    }

    #[test]
    fn relational_discovery_uses_declared_fields_and_unresolved_policy() {
        let entities = vec![
            DataValue::object([
                ("key".to_owned(), DataValue::Text("agent-a".to_owned())),
                (
                    "links".to_owned(),
                    DataValue::Sequence(vec![
                        DataValue::Text("agent-b".to_owned()),
                        DataValue::Text("external-resource".to_owned()),
                    ]),
                ),
            ]),
            DataValue::object([
                ("key".to_owned(), DataValue::Text("agent-b".to_owned())),
                ("links".to_owned(), DataValue::Null),
            ]),
        ];
        let graph = discover_relationships(
            &entities,
            "key",
            &["links".to_owned()],
            UnresolvedReferencePolicy::AllowExternal,
        )
        .expect("relationship graph");
        assert_eq!(
            graph.entity_ids,
            vec!["agent-a".to_owned(), "agent-b".to_owned()]
        );
        assert_eq!(graph.edges.len(), 2);
        assert!(graph.edges.iter().any(|edge| edge.target_is_known));
        assert!(graph.edges.iter().any(|edge| !edge.target_is_known));

        assert!(matches!(
            discover_relationships(
                &entities,
                "key",
                &["links".to_owned()],
                UnresolvedReferencePolicy::RequireKnownEntity,
            ),
            Err(StructureDiscoveryError::UnresolvedReference { .. })
        ));
    }

    #[test]
    fn field_profiling_identifies_distributional_roles_and_missingness() {
        let samples: Vec<DataValue> = (0..20)
            .map(|index| {
                let mut fields = BTreeMap::new();
                fields.insert("measure".to_owned(), DataValue::Number(index as f64));
                fields.insert("flag".to_owned(), DataValue::Bool(index % 2 == 0));
                fields.insert(
                    "entity-key".to_owned(),
                    DataValue::Text(format!("entity-{index}")),
                );
                fields.insert(
                    "category".to_owned(),
                    DataValue::Text(if index % 2 == 0 { "red" } else { "blue" }.to_owned()),
                );
                if index % 2 == 0 {
                    fields.insert("optional".to_owned(), DataValue::Text("present".to_owned()));
                }
                DataValue::Object(fields)
            })
            .collect();
        let profile = profile_fields(
            &samples,
            FieldProfilePolicy {
                minimum_samples: 10,
                identifier_uniqueness_threshold: 0.95,
                maximum_categorical_cardinality: 4,
            },
        )
        .expect("semantic profile");
        let role = |name: &str| {
            profile
                .fields
                .iter()
                .find(|field| field.name == name)
                .unwrap()
                .role
        };
        assert_eq!(role("measure"), FieldRole::NumericMeasure);
        assert_eq!(role("flag"), FieldRole::BooleanFlag);
        assert_eq!(role("entity-key"), FieldRole::IdentifierCandidate);
        assert_eq!(role("category"), FieldRole::Categorical);
        let optional = profile
            .fields
            .iter()
            .find(|field| field.name == "optional")
            .unwrap();
        assert_eq!(optional.missing_count, 10);
        assert_eq!(optional.observed_count, 10);
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
