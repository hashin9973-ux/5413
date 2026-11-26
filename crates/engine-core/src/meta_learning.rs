//! Typed, declarative learning-strategy graphs and guarded promotion.
//!
//! This module describes operator composition; it does not execute arbitrary
//! code. A runtime may only execute operators implemented in its trusted,
//! explicitly registered operator catalogue.

use crate::benchmark::ScoreDirection;
use crate::schema::ValueKind;
use crate::value::DataValue;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortSpec {
    pub name: String,
    pub kind: ValueKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorSpec {
    pub id: String,
    pub inputs: Vec<PortSpec>,
    pub outputs: Vec<PortSpec>,
    pub cost_units: u64,
    pub deterministic: bool,
}

impl OperatorSpec {
    pub fn validate(&self) -> Result<(), StrategyError> {
        if self.id.trim().is_empty() || self.outputs.is_empty() {
            return Err(StrategyError::InvalidOperator);
        }
        let mut inputs = BTreeSet::new();
        let mut outputs = BTreeSet::new();
        if self
            .inputs
            .iter()
            .any(|port| port.name.trim().is_empty() || !inputs.insert(&port.name))
            || self
                .outputs
                .iter()
                .any(|port| port.name.trim().is_empty() || !outputs.insert(&port.name))
        {
            return Err(StrategyError::InvalidOperator);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperatorCatalogue {
    operators: BTreeMap<String, OperatorSpec>,
}

impl OperatorCatalogue {
    pub fn register(&mut self, operator: OperatorSpec) -> Result<(), StrategyError> {
        operator.validate()?;
        let id = operator.id.clone();
        if self.operators.contains_key(&id) {
            return Err(StrategyError::DuplicateOperator);
        }
        self.operators.insert(id, operator);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&OperatorSpec> {
        self.operators.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &OperatorSpec> {
        self.operators.values()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputBinding {
    External(String),
    NodeOutput { node_id: String, port: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrategyNode {
    pub id: String,
    pub operator_id: String,
    pub inputs: BTreeMap<String, InputBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrategyGraph {
    pub external_inputs: BTreeMap<String, ValueKind>,
    pub nodes: Vec<StrategyNode>,
    pub outputs: BTreeMap<String, InputBinding>,
}

impl StrategyGraph {
    /// Validates exact port bindings, type compatibility, acyclicity, and output reachability.
    pub fn validate(&self, catalogue: &OperatorCatalogue) -> Result<(), StrategyError> {
        if self.nodes.is_empty() || self.outputs.is_empty() {
            return Err(StrategyError::EmptyGraph);
        }
        if self
            .external_inputs
            .keys()
            .any(|name| name.trim().is_empty())
            || self.outputs.keys().any(|name| name.trim().is_empty())
        {
            return Err(StrategyError::InvalidGraph);
        }

        let mut nodes = BTreeMap::new();
        for node in &self.nodes {
            if node.id.trim().is_empty() || nodes.insert(node.id.as_str(), node).is_some() {
                return Err(StrategyError::DuplicateNode);
            }
        }

        let mut indegree: BTreeMap<&str, usize> = nodes.keys().map(|id| (*id, 0usize)).collect();
        let mut outgoing: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        let mut reverse: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();

        for node in &self.nodes {
            let operator = catalogue
                .get(&node.operator_id)
                .ok_or(StrategyError::UnknownOperator)?;
            let expected: BTreeMap<&str, ValueKind> = operator
                .inputs
                .iter()
                .map(|port| (port.name.as_str(), port.kind))
                .collect();
            if node.inputs.len() != expected.len()
                || node
                    .inputs
                    .keys()
                    .any(|name| !expected.contains_key(name.as_str()))
            {
                return Err(StrategyError::InputPortMismatch);
            }

            for (port_name, binding) in &node.inputs {
                let expected_kind = expected
                    .get(port_name.as_str())
                    .ok_or(StrategyError::InputPortMismatch)?;
                match binding {
                    InputBinding::External(name) => {
                        let actual = self
                            .external_inputs
                            .get(name)
                            .ok_or(StrategyError::UnknownExternalInput)?;
                        if actual != expected_kind {
                            return Err(StrategyError::PortTypeMismatch);
                        }
                    }
                    InputBinding::NodeOutput { node_id, port } => {
                        let source = nodes
                            .get(node_id.as_str())
                            .ok_or(StrategyError::UnknownNode)?;
                        let source_operator = catalogue
                            .get(&source.operator_id)
                            .ok_or(StrategyError::UnknownOperator)?;
                        let actual = source_operator
                            .outputs
                            .iter()
                            .find(|candidate| candidate.name == *port)
                            .ok_or(StrategyError::UnknownOutputPort)?;
                        if actual.kind != *expected_kind {
                            return Err(StrategyError::PortTypeMismatch);
                        }
                        if node_id == &node.id {
                            return Err(StrategyError::CycleDetected);
                        }
                        *indegree
                            .get_mut(node.id.as_str())
                            .ok_or(StrategyError::InvalidGraph)? += 1;
                        outgoing
                            .entry(node_id.as_str())
                            .or_default()
                            .push(node.id.as_str());
                        reverse
                            .entry(node.id.as_str())
                            .or_default()
                            .insert(node_id.as_str());
                    }
                }
            }
        }

        for binding in self.outputs.values() {
            match binding {
                InputBinding::External(name) => {
                    if !self.external_inputs.contains_key(name) {
                        return Err(StrategyError::UnknownExternalInput);
                    }
                }
                InputBinding::NodeOutput { node_id, port } => {
                    let source = nodes
                        .get(node_id.as_str())
                        .ok_or(StrategyError::UnknownNode)?;
                    let operator = catalogue
                        .get(&source.operator_id)
                        .ok_or(StrategyError::UnknownOperator)?;
                    if !operator
                        .outputs
                        .iter()
                        .any(|candidate| candidate.name == *port)
                    {
                        return Err(StrategyError::UnknownOutputPort);
                    }
                }
            }
        }

        let mut queue: VecDeque<&str> = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
            .collect();
        let mut visited = 0usize;
        while let Some(node_id) = queue.pop_front() {
            visited += 1;
            if let Some(destinations) = outgoing.get(node_id) {
                for destination in destinations {
                    let degree = indegree
                        .get_mut(destination)
                        .ok_or(StrategyError::InvalidGraph)?;
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(destination);
                    }
                }
            }
        }
        if visited != nodes.len() {
            return Err(StrategyError::CycleDetected);
        }

        let mut reachable = BTreeSet::new();
        let mut pending: Vec<&str> = self
            .outputs
            .values()
            .filter_map(|binding| match binding {
                InputBinding::NodeOutput { node_id, .. } => Some(node_id.as_str()),
                InputBinding::External(_) => None,
            })
            .collect();
        while let Some(node_id) = pending.pop() {
            if reachable.insert(node_id) {
                if let Some(sources) = reverse.get(node_id) {
                    pending.extend(sources.iter().copied());
                }
            }
        }
        if reachable.len() != nodes.len() {
            return Err(StrategyError::UnreachableNode);
        }
        Ok(())
    }

    /// Stable aggregate cost of the trusted operators referenced by this graph.
    pub fn estimated_cost(&self, catalogue: &OperatorCatalogue) -> Result<u64, StrategyError> {
        self.nodes.iter().try_fold(0_u64, |total, node| {
            let cost = catalogue
                .get(&node.operator_id)
                .ok_or(StrategyError::UnknownOperator)?
                .cost_units;
            total.checked_add(cost).ok_or(StrategyError::CostOverflow)
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrategySpec {
    pub id: String,
    pub version: u64,
    pub training_dataset_id: String,
    pub graph: StrategyGraph,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrategyPolicy {
    pub minimum_evaluation_samples: usize,
    pub minimum_improvement: f64,
    pub resource_cost_weight: f64,
}

impl Default for StrategyPolicy {
    fn default() -> Self {
        Self {
            minimum_evaluation_samples: 10,
            minimum_improvement: 0.01,
            resource_cost_weight: 0.0,
        }
    }
}

impl StrategyPolicy {
    pub fn validate(&self) -> Result<(), StrategyError> {
        if self.minimum_evaluation_samples == 0
            || !self.minimum_improvement.is_finite()
            || self.minimum_improvement < 0.0
            || !self.resource_cost_weight.is_finite()
            || self.resource_cost_weight < 0.0
        {
            return Err(StrategyError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrategyEvaluation {
    pub evaluation_dataset_id: String,
    pub metric_id: String,
    pub direction: ScoreDirection,
    pub sample_count: usize,
    /// The value of the declared held-out metric.
    pub primary_score: f64,
    /// Measured or consistently estimated resource units consumed.
    pub resource_cost: f64,
}

impl StrategyEvaluation {
    fn validate(&self) -> Result<(), StrategyError> {
        if self.evaluation_dataset_id.trim().is_empty()
            || self.metric_id.trim().is_empty()
            || self.sample_count == 0
            || !self.primary_score.is_finite()
            || !self.resource_cost.is_finite()
            || self.resource_cost < 0.0
        {
            return Err(StrategyError::InvalidEvaluation);
        }
        Ok(())
    }

    pub fn utility(&self, policy: StrategyPolicy) -> f64 {
        let directional_score = match self.direction {
            ScoreDirection::HigherIsBetter => self.primary_score,
            ScoreDirection::LowerIsBetter => -self.primary_score,
        };
        directional_score - policy.resource_cost_weight * self.resource_cost
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyStatus {
    Candidate,
    Active,
    Rejected,
    Retired,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrategyRecord {
    pub spec: StrategySpec,
    pub status: StrategyStatus,
    pub evaluation: Option<StrategyEvaluation>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrategyRegistry {
    records: BTreeMap<(String, u64), StrategyRecord>,
}

impl StrategyRegistry {
    pub fn register_candidate(
        &mut self,
        spec: StrategySpec,
        catalogue: &OperatorCatalogue,
    ) -> Result<(), StrategyError> {
        if spec.id.trim().is_empty()
            || spec.version == 0
            || spec.training_dataset_id.trim().is_empty()
        {
            return Err(StrategyError::InvalidSpec);
        }
        spec.graph.validate(catalogue)?;
        let key = (spec.id.clone(), spec.version);
        if self.records.contains_key(&key) {
            return Err(StrategyError::DuplicateStrategy);
        }
        self.records.insert(
            key,
            StrategyRecord {
                spec,
                status: StrategyStatus::Candidate,
                evaluation: None,
            },
        );
        Ok(())
    }

    pub fn evaluate_candidate(
        &mut self,
        id: &str,
        version: u64,
        evaluation: StrategyEvaluation,
        policy: StrategyPolicy,
    ) -> Result<(), StrategyError> {
        policy.validate()?;
        evaluation.validate()?;
        let record = self
            .records
            .get_mut(&(id.to_owned(), version))
            .ok_or(StrategyError::StrategyNotFound)?;
        if record.status != StrategyStatus::Candidate {
            return Err(StrategyError::InvalidLifecycle);
        }
        if evaluation.evaluation_dataset_id == record.spec.training_dataset_id {
            return Err(StrategyError::EvaluationReusesTrainingData);
        }
        if evaluation.sample_count < policy.minimum_evaluation_samples {
            record.status = StrategyStatus::Rejected;
            record.evaluation = Some(evaluation);
            return Err(StrategyError::InsufficientEvaluationSamples);
        }
        record.evaluation = Some(evaluation);
        Ok(())
    }

    /// Promotes only after a same-dataset comparison with the current strategy.
    /// An incumbent is retired atomically when a candidate passes the margin.
    pub fn promote(
        &mut self,
        id: &str,
        version: u64,
        policy: StrategyPolicy,
    ) -> Result<(), StrategyError> {
        policy.validate()?;
        let candidate_key = (id.to_owned(), version);
        let candidate = self
            .records
            .get(&candidate_key)
            .ok_or(StrategyError::StrategyNotFound)?;
        if candidate.status != StrategyStatus::Candidate {
            return Err(StrategyError::InvalidLifecycle);
        }
        let evaluation = candidate
            .evaluation
            .as_ref()
            .ok_or(StrategyError::NotEvaluated)?;
        let candidate_utility = evaluation.utility(policy);
        let incumbent_key = self
            .records
            .values()
            .find(|record| record.spec.id == id && record.status == StrategyStatus::Active)
            .map(|record| (record.spec.id.clone(), record.spec.version));

        if let Some(key) = incumbent_key.as_ref() {
            let incumbent = self
                .records
                .get(key)
                .ok_or(StrategyError::StrategyNotFound)?;
            let incumbent_evaluation = incumbent
                .evaluation
                .as_ref()
                .ok_or(StrategyError::NotEvaluated)?;
            if incumbent_evaluation.evaluation_dataset_id != evaluation.evaluation_dataset_id
                || incumbent_evaluation.metric_id != evaluation.metric_id
                || incumbent_evaluation.direction != evaluation.direction
            {
                return Err(StrategyError::IncomparableEvaluations);
            }
            if candidate_utility < incumbent_evaluation.utility(policy) + policy.minimum_improvement
            {
                self.records
                    .get_mut(&candidate_key)
                    .ok_or(StrategyError::StrategyNotFound)?
                    .status = StrategyStatus::Rejected;
                return Err(StrategyError::InsufficientImprovement);
            }
        }

        if let Some(key) = incumbent_key {
            self.records
                .get_mut(&key)
                .ok_or(StrategyError::StrategyNotFound)?
                .status = StrategyStatus::Retired;
        }
        self.records
            .get_mut(&candidate_key)
            .ok_or(StrategyError::StrategyNotFound)?
            .status = StrategyStatus::Active;
        Ok(())
    }

    pub fn active(&self, id: &str) -> Option<&StrategyRecord> {
        self.records
            .values()
            .find(|record| record.spec.id == id && record.status == StrategyStatus::Active)
    }

    pub fn get(&self, id: &str, version: u64) -> Option<&StrategyRecord> {
        self.records.get(&(id.to_owned(), version))
    }

    pub fn iter(&self) -> impl Iterator<Item = &StrategyRecord> {
        self.records.values()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Exports registry metadata and evaluation provenance.
    ///
    /// The operator catalogue is a separately versioned dependency and must be
    /// supplied again to validate/execute the graph after restoring this artifact.
    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "strategies".to_owned(),
                DataValue::Sequence(self.records.values().map(strategy_to_value).collect()),
            ),
        ])
    }

    pub fn from_artifact_value(
        value: &DataValue,
        catalogue: &OperatorCatalogue,
    ) -> Result<Self, StrategyError> {
        let fields = value
            .as_object()
            .ok_or(StrategyError::InvalidSerializedState)?;
        if fields.len() != 2 || fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
            return Err(StrategyError::InvalidSerializedState);
        }
        let values = match fields.get("strategies") {
            Some(DataValue::Sequence(values)) => values,
            _ => return Err(StrategyError::InvalidSerializedState),
        };
        let mut records = BTreeMap::new();
        let mut active_ids = BTreeSet::new();
        for value in values {
            let record = strategy_from_value(value, catalogue)?;
            if record.status == StrategyStatus::Active && !active_ids.insert(record.spec.id.clone())
            {
                return Err(StrategyError::InvalidSerializedState);
            }
            let key = (record.spec.id.clone(), record.spec.version);
            if records.insert(key, record).is_some() {
                return Err(StrategyError::InvalidSerializedState);
            }
        }
        Ok(Self { records })
    }
}

fn strategy_to_value(record: &StrategyRecord) -> DataValue {
    let inputs = DataValue::Object(
        record
            .spec
            .graph
            .external_inputs
            .iter()
            .map(|(name, kind)| {
                (
                    name.clone(),
                    DataValue::Text(value_kind_name(*kind).to_owned()),
                )
            })
            .collect(),
    );
    let nodes = DataValue::Sequence(
        record
            .spec
            .graph
            .nodes
            .iter()
            .map(|node| {
                DataValue::object([
                    ("id".to_owned(), DataValue::Text(node.id.clone())),
                    (
                        "operator_id".to_owned(),
                        DataValue::Text(node.operator_id.clone()),
                    ),
                    ("inputs".to_owned(), bindings_to_value(&node.inputs)),
                ])
            })
            .collect(),
    );
    let outputs = bindings_to_value(&record.spec.graph.outputs);
    let evaluation = record
        .evaluation
        .as_ref()
        .map(|item| {
            DataValue::object([
                (
                    "evaluation_dataset_id".to_owned(),
                    DataValue::Text(item.evaluation_dataset_id.clone()),
                ),
                (
                    "metric_id".to_owned(),
                    DataValue::Text(item.metric_id.clone()),
                ),
                (
                    "direction".to_owned(),
                    DataValue::Text(
                        match item.direction {
                            ScoreDirection::HigherIsBetter => "higher",
                            ScoreDirection::LowerIsBetter => "lower",
                        }
                        .to_owned(),
                    ),
                ),
                (
                    "sample_count".to_owned(),
                    DataValue::Text(item.sample_count.to_string()),
                ),
                (
                    "primary_score".to_owned(),
                    DataValue::Number(item.primary_score),
                ),
                (
                    "resource_cost".to_owned(),
                    DataValue::Number(item.resource_cost),
                ),
            ])
        })
        .unwrap_or(DataValue::Null);
    DataValue::object([
        ("id".to_owned(), DataValue::Text(record.spec.id.clone())),
        (
            "version".to_owned(),
            DataValue::Text(record.spec.version.to_string()),
        ),
        (
            "training_dataset_id".to_owned(),
            DataValue::Text(record.spec.training_dataset_id.clone()),
        ),
        ("external_inputs".to_owned(), inputs),
        ("nodes".to_owned(), nodes),
        ("outputs".to_owned(), outputs),
        (
            "status".to_owned(),
            DataValue::Text(status_name(record.status).to_owned()),
        ),
        ("evaluation".to_owned(), evaluation),
    ])
}

fn bindings_to_value(bindings: &BTreeMap<String, InputBinding>) -> DataValue {
    DataValue::Object(
        bindings
            .iter()
            .map(|(name, binding)| {
                let value = match binding {
                    InputBinding::External(source) => DataValue::object([
                        ("kind".to_owned(), DataValue::Text("external".to_owned())),
                        ("name".to_owned(), DataValue::Text(source.clone())),
                    ]),
                    InputBinding::NodeOutput { node_id, port } => DataValue::object([
                        ("kind".to_owned(), DataValue::Text("node".to_owned())),
                        ("node_id".to_owned(), DataValue::Text(node_id.clone())),
                        ("port".to_owned(), DataValue::Text(port.clone())),
                    ]),
                };
                (name.clone(), value)
            })
            .collect(),
    )
}

fn value_to_bindings(value: &DataValue) -> Result<BTreeMap<String, InputBinding>, StrategyError> {
    let values = value
        .as_object()
        .ok_or(StrategyError::InvalidSerializedState)?;
    let mut result = BTreeMap::new();
    for (name, value) in values {
        let fields = value
            .as_object()
            .ok_or(StrategyError::InvalidSerializedState)?;
        let binding = match fields.get("kind") {
            Some(DataValue::Text(kind)) if kind == "external" && fields.len() == 2 => {
                match fields.get("name") {
                    Some(DataValue::Text(name)) => InputBinding::External(name.clone()),
                    _ => return Err(StrategyError::InvalidSerializedState),
                }
            }
            Some(DataValue::Text(kind)) if kind == "node" && fields.len() == 3 => {
                let node_id = match fields.get("node_id") {
                    Some(DataValue::Text(value)) => value.clone(),
                    _ => return Err(StrategyError::InvalidSerializedState),
                };
                let port = match fields.get("port") {
                    Some(DataValue::Text(value)) => value.clone(),
                    _ => return Err(StrategyError::InvalidSerializedState),
                };
                InputBinding::NodeOutput { node_id, port }
            }
            _ => return Err(StrategyError::InvalidSerializedState),
        };
        result.insert(name.clone(), binding);
    }
    Ok(result)
}

fn value_kind_name(kind: ValueKind) -> &'static str {
    use ValueKind::*;
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

fn parse_value_kind(value: &str) -> Result<ValueKind, StrategyError> {
    use ValueKind::*;
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
        _ => Err(StrategyError::InvalidSerializedState),
    }
}

fn text_field(fields: &BTreeMap<String, DataValue>, name: &str) -> Result<String, StrategyError> {
    match fields.get(name) {
        Some(DataValue::Text(value)) => Ok(value.clone()),
        _ => Err(StrategyError::InvalidSerializedState),
    }
}

fn number_field(fields: &BTreeMap<String, DataValue>, name: &str) -> Result<f64, StrategyError> {
    fields
        .get(name)
        .and_then(DataValue::as_f64)
        .filter(|value| value.is_finite())
        .ok_or(StrategyError::InvalidSerializedState)
}

fn strategy_from_value(
    value: &DataValue,
    catalogue: &OperatorCatalogue,
) -> Result<StrategyRecord, StrategyError> {
    let fields = value
        .as_object()
        .ok_or(StrategyError::InvalidSerializedState)?;
    if fields.len() != 8 {
        return Err(StrategyError::InvalidSerializedState);
    }
    let id = text_field(fields, "id")?;
    let version = text_field(fields, "version")?
        .parse()
        .map_err(|_| StrategyError::InvalidSerializedState)?;
    let training_dataset_id = text_field(fields, "training_dataset_id")?;
    let external_inputs = match fields.get("external_inputs") {
        Some(DataValue::Object(values)) => values
            .iter()
            .map(|(name, value)| {
                let DataValue::Text(kind) = value else {
                    return Err(StrategyError::InvalidSerializedState);
                };
                Ok((name.clone(), parse_value_kind(kind)?))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?,
        _ => return Err(StrategyError::InvalidSerializedState),
    };
    let nodes = match fields.get("nodes") {
        Some(DataValue::Sequence(nodes)) => nodes
            .iter()
            .map(|value| {
                let node = value
                    .as_object()
                    .ok_or(StrategyError::InvalidSerializedState)?;
                if node.len() != 3 {
                    return Err(StrategyError::InvalidSerializedState);
                }
                Ok(StrategyNode {
                    id: text_field(node, "id")?,
                    operator_id: text_field(node, "operator_id")?,
                    inputs: value_to_bindings(
                        node.get("inputs")
                            .ok_or(StrategyError::InvalidSerializedState)?,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(StrategyError::InvalidSerializedState),
    };
    let outputs = value_to_bindings(
        fields
            .get("outputs")
            .ok_or(StrategyError::InvalidSerializedState)?,
    )?;
    let status = match text_field(fields, "status")?.as_str() {
        "candidate" => StrategyStatus::Candidate,
        "active" => StrategyStatus::Active,
        "rejected" => StrategyStatus::Rejected,
        "retired" => StrategyStatus::Retired,
        _ => return Err(StrategyError::InvalidSerializedState),
    };
    let evaluation = match fields.get("evaluation") {
        Some(DataValue::Null) => None,
        Some(DataValue::Object(values)) if values.len() == 6 => Some(StrategyEvaluation {
            evaluation_dataset_id: text_field(values, "evaluation_dataset_id")?,
            metric_id: text_field(values, "metric_id")?,
            direction: match text_field(values, "direction")?.as_str() {
                "higher" => ScoreDirection::HigherIsBetter,
                "lower" => ScoreDirection::LowerIsBetter,
                _ => return Err(StrategyError::InvalidSerializedState),
            },
            sample_count: text_field(values, "sample_count")?
                .parse()
                .map_err(|_| StrategyError::InvalidSerializedState)?,
            primary_score: number_field(values, "primary_score")?,
            resource_cost: number_field(values, "resource_cost")?,
        }),
        _ => return Err(StrategyError::InvalidSerializedState),
    };
    if id.trim().is_empty() || version == 0 || training_dataset_id.trim().is_empty() {
        return Err(StrategyError::InvalidSerializedState);
    }
    if let Some(item) = &evaluation {
        item.validate()?;
        if item.evaluation_dataset_id == training_dataset_id {
            return Err(StrategyError::InvalidSerializedState);
        }
    }
    if matches!(
        status,
        StrategyStatus::Active | StrategyStatus::Retired | StrategyStatus::Rejected
    ) && evaluation.is_none()
    {
        return Err(StrategyError::InvalidSerializedState);
    }
    let spec = StrategySpec {
        id,
        version,
        training_dataset_id,
        graph: StrategyGraph {
            external_inputs,
            nodes,
            outputs,
        },
    };
    spec.graph.validate(catalogue)?;
    Ok(StrategyRecord {
        spec,
        status,
        evaluation,
    })
}

fn status_name(status: StrategyStatus) -> &'static str {
    match status {
        StrategyStatus::Candidate => "candidate",
        StrategyStatus::Active => "active",
        StrategyStatus::Rejected => "rejected",
        StrategyStatus::Retired => "retired",
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrategySynthesisPolicy {
    pub input_name: String,
    pub output_name: String,
    pub input_kind: ValueKind,
    pub output_kind: ValueKind,
    pub maximum_depth: usize,
    pub maximum_cost_units: u64,
    pub maximum_candidates: usize,
    pub maximum_expansions: usize,
    pub allow_nondeterministic: bool,
}

impl StrategySynthesisPolicy {
    pub fn validate(&self) -> Result<(), StrategySynthesisError> {
        if self.input_name.trim().is_empty()
            || self.output_name.trim().is_empty()
            || self.maximum_depth == 0
            || self.maximum_candidates == 0
            || self.maximum_expansions == 0
        {
            return Err(StrategySynthesisError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SynthesizedStrategy {
    pub id: String,
    pub graph: StrategyGraph,
    pub cost_units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrategySynthesisResult {
    pub candidates: Vec<SynthesizedStrategy>,
    pub expanded_states: usize,
    pub truncated: bool,
}

#[derive(Clone)]
struct StrategySynthesisState {
    chain: Vec<OperatorSpec>,
    used_operator_ids: BTreeSet<String>,
    output_kind: ValueKind,
    cost_units: u64,
}

/// Enumerates type-compatible unary operator chains from one declared input
/// to one declared output. Operators are selected from the trusted catalogue;
/// no source code is generated or compiled.
///
/// Search is breadth-first and deterministic. It is bounded by depth, total
/// declared cost, expanded states, and returned candidate count. Each generated
/// graph is passed through the normal strategy graph validator.
pub fn synthesize_unary_strategy_chains(
    catalogue: &OperatorCatalogue,
    policy: StrategySynthesisPolicy,
) -> Result<StrategySynthesisResult, StrategySynthesisError> {
    policy.validate()?;
    let mut queue = VecDeque::from([StrategySynthesisState {
        chain: Vec::new(),
        used_operator_ids: BTreeSet::new(),
        output_kind: policy.input_kind,
        cost_units: 0,
    }]);
    let mut candidates = Vec::new();
    let mut expanded_states = 0usize;
    let mut truncated = false;

    while let Some(state) = queue.pop_front() {
        if expanded_states >= policy.maximum_expansions {
            truncated = true;
            break;
        }
        expanded_states += 1;
        if state.chain.len() >= policy.maximum_depth {
            continue;
        }

        for operator in catalogue.iter() {
            if operator.inputs.len() != 1
                || operator.outputs.len() != 1
                || operator.inputs[0].kind != state.output_kind
                || state.used_operator_ids.contains(&operator.id)
                || (!policy.allow_nondeterministic && !operator.deterministic)
            {
                continue;
            }

            let cost_units = state
                .cost_units
                .checked_add(operator.cost_units)
                .ok_or(StrategySynthesisError::CostOverflow)?;
            if cost_units > policy.maximum_cost_units {
                continue;
            }

            let mut next = state.clone();
            next.chain.push(operator.clone());
            next.used_operator_ids.insert(operator.id.clone());
            next.output_kind = operator.outputs[0].kind;
            next.cost_units = cost_units;

            if next.output_kind == policy.output_kind {
                let graph = graph_from_unary_chain(&next.chain, &policy)?;
                graph
                    .validate(catalogue)
                    .map_err(StrategySynthesisError::Graph)?;
                let id = format!(
                    "synthesized-{}",
                    next.chain
                        .iter()
                        .map(|item| item.id.as_str())
                        .collect::<Vec<_>>()
                        .join("-")
                );
                candidates.push(SynthesizedStrategy {
                    id,
                    graph,
                    cost_units,
                });
                if candidates.len() >= policy.maximum_candidates {
                    return Ok(StrategySynthesisResult {
                        candidates,
                        expanded_states,
                        truncated: true,
                    });
                }
            }

            if next.chain.len() < policy.maximum_depth {
                queue.push_back(next);
            }
        }
    }

    Ok(StrategySynthesisResult {
        candidates,
        expanded_states,
        truncated,
    })
}

fn graph_from_unary_chain(
    chain: &[OperatorSpec],
    policy: &StrategySynthesisPolicy,
) -> Result<StrategyGraph, StrategySynthesisError> {
    if chain.is_empty() {
        return Err(StrategySynthesisError::EmptyChain);
    }

    let mut nodes = Vec::with_capacity(chain.len());
    for (index, operator) in chain.iter().enumerate() {
        let input_binding = if index == 0 {
            InputBinding::External(policy.input_name.clone())
        } else {
            InputBinding::NodeOutput {
                node_id: format!("step-{}", index - 1),
                port: chain[index - 1].outputs[0].name.clone(),
            }
        };
        nodes.push(StrategyNode {
            id: format!("step-{index}"),
            operator_id: operator.id.clone(),
            inputs: BTreeMap::from([(operator.inputs[0].name.clone(), input_binding)]),
        });
    }

    let last_index = chain.len() - 1;
    let last_operator = chain.last().ok_or(StrategySynthesisError::EmptyChain)?;
    Ok(StrategyGraph {
        external_inputs: BTreeMap::from([(policy.input_name.clone(), policy.input_kind)]),
        nodes,
        outputs: BTreeMap::from([(
            policy.output_name.clone(),
            InputBinding::NodeOutput {
                node_id: format!("step-{last_index}"),
                port: last_operator.outputs[0].name.clone(),
            },
        )]),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrategySynthesisError {
    InvalidPolicy,
    CostOverflow,
    EmptyChain,
    Graph(StrategyError),
}

impl std::fmt::Display for StrategySynthesisError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPolicy => write!(formatter, "strategy synthesis policy is invalid"),
            Self::CostOverflow => write!(formatter, "strategy synthesis cost overflowed"),
            Self::EmptyChain => write!(
                formatter,
                "cannot synthesize a graph from an empty operator chain"
            ),
            Self::Graph(error) => {
                write!(formatter, "synthesized strategy graph is invalid: {error}")
            }
        }
    }
}

impl std::error::Error for StrategySynthesisError {}

/// Trusted implementation boundary for declarative strategy operators.
///
/// The engine never compiles or evaluates generated source. A runtime supplies
/// this executor, which dispatches only to explicitly trusted operator code.
pub trait StrategyOperatorExecutor {
    fn execute(
        &self,
        operator: &OperatorSpec,
        inputs: &BTreeMap<String, DataValue>,
    ) -> Result<BTreeMap<String, DataValue>, String>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrategyExecutionReport {
    pub outputs: BTreeMap<String, DataValue>,
    pub executed_nodes: Vec<String>,
    pub declared_cost_units: u64,
}

/// Executes a validated strategy DAG through a caller-supplied trusted executor.
///
/// The interpreter enforces exact external input sets, runtime port types,
/// exact output sets, finite values, deterministic topological order, and a
/// total declared operator-cost budget. The executor remains responsible for
/// its own internal resource and side-effect restrictions.
pub fn execute_strategy_graph<E: StrategyOperatorExecutor>(
    graph: &StrategyGraph,
    catalogue: &OperatorCatalogue,
    external_values: &BTreeMap<String, DataValue>,
    executor: &E,
    maximum_cost_units: u64,
) -> Result<StrategyExecutionReport, StrategyExecutionError> {
    graph
        .validate(catalogue)
        .map_err(StrategyExecutionError::Graph)?;
    let declared_cost_units = graph
        .estimated_cost(catalogue)
        .map_err(StrategyExecutionError::Graph)?;
    if declared_cost_units > maximum_cost_units {
        return Err(StrategyExecutionError::CostBudgetExceeded {
            estimated: declared_cost_units,
            maximum: maximum_cost_units,
        });
    }

    if external_values.len() != graph.external_inputs.len()
        || graph
            .external_inputs
            .keys()
            .any(|name| !external_values.contains_key(name))
    {
        return Err(StrategyExecutionError::ExternalInputSetMismatch);
    }
    for (name, kind) in &graph.external_inputs {
        let value = external_values
            .get(name)
            .ok_or(StrategyExecutionError::ExternalInputSetMismatch)?;
        if !runtime_value_matches(value, *kind) {
            return Err(StrategyExecutionError::ExternalInputTypeMismatch(
                name.clone(),
            ));
        }
        value
            .validate_finite()
            .map_err(|_| StrategyExecutionError::NonFiniteExternalInput(name.clone()))?;
    }

    let nodes: BTreeMap<&str, &StrategyNode> = graph
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect();
    let mut indegree: BTreeMap<String, usize> = graph
        .nodes
        .iter()
        .map(|node| (node.id.clone(), 0))
        .collect();
    let mut outgoing: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for node in &graph.nodes {
        for binding in node.inputs.values() {
            if let InputBinding::NodeOutput { node_id, .. } = binding {
                let degree = indegree
                    .get_mut(&node.id)
                    .ok_or(StrategyExecutionError::InvalidGraphState)?;
                *degree = degree
                    .checked_add(1)
                    .ok_or(StrategyExecutionError::InvalidGraphState)?;
                outgoing
                    .entry(node_id.clone())
                    .or_default()
                    .push(node.id.clone());
            }
        }
    }

    // BTreeSet makes the order deterministic when multiple nodes are ready.
    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter_map(|(node_id, degree)| (*degree == 0).then_some(node_id.clone()))
        .collect();
    let mut resolved: BTreeMap<String, BTreeMap<String, DataValue>> = BTreeMap::new();
    let mut executed_nodes = Vec::with_capacity(graph.nodes.len());

    while let Some(node_id) = ready.pop_first() {
        let node = nodes
            .get(node_id.as_str())
            .ok_or(StrategyExecutionError::InvalidGraphState)?;
        let operator = catalogue
            .get(&node.operator_id)
            .ok_or(StrategyExecutionError::InvalidGraphState)?;
        let mut inputs = BTreeMap::new();

        for input_port in &operator.inputs {
            let binding = node
                .inputs
                .get(&input_port.name)
                .ok_or(StrategyExecutionError::InvalidGraphState)?;
            let value = match binding {
                InputBinding::External(name) => external_values.get(name),
                InputBinding::NodeOutput { node_id, port } => {
                    resolved.get(node_id).and_then(|outputs| outputs.get(port))
                }
            }
            .ok_or_else(|| StrategyExecutionError::UpstreamValueMissing {
                node_id: node_id.clone(),
                port: input_port.name.clone(),
            })?;

            if !runtime_value_matches(value, input_port.kind) {
                return Err(StrategyExecutionError::RuntimeInputTypeMismatch {
                    node_id: node_id.clone(),
                    port: input_port.name.clone(),
                });
            }
            value
                .validate_finite()
                .map_err(|_| StrategyExecutionError::NonFiniteInput {
                    node_id: node_id.clone(),
                    port: input_port.name.clone(),
                })?;
            inputs.insert(input_port.name.clone(), value.clone());
        }

        let outputs = executor.execute(operator, &inputs).map_err(|message| {
            StrategyExecutionError::OperatorFailed {
                operator_id: operator.id.clone(),
                message,
            }
        })?;
        if outputs.len() != operator.outputs.len() {
            return Err(StrategyExecutionError::OutputPortSetMismatch(
                operator.id.clone(),
            ));
        }
        for output_port in &operator.outputs {
            let value = outputs.get(&output_port.name).ok_or_else(|| {
                StrategyExecutionError::OutputPortSetMismatch(operator.id.clone())
            })?;
            value
                .validate_finite()
                .map_err(|_| StrategyExecutionError::NonFiniteOutput {
                    operator_id: operator.id.clone(),
                    port: output_port.name.clone(),
                })?;
            if !runtime_value_matches(value, output_port.kind) {
                return Err(StrategyExecutionError::RuntimeOutputTypeMismatch {
                    operator_id: operator.id.clone(),
                    port: output_port.name.clone(),
                });
            }
        }

        executed_nodes.push(node_id.clone());
        resolved.insert(node_id.clone(), outputs);

        if let Some(destinations) = outgoing.get(&node_id) {
            for destination in destinations {
                let degree = indegree
                    .get_mut(destination)
                    .ok_or(StrategyExecutionError::InvalidGraphState)?;
                *degree = degree
                    .checked_sub(1)
                    .ok_or(StrategyExecutionError::InvalidGraphState)?;
                if *degree == 0 {
                    ready.insert(destination.clone());
                }
            }
        }
    }

    if executed_nodes.len() != graph.nodes.len() {
        return Err(StrategyExecutionError::Graph(StrategyError::CycleDetected));
    }

    let mut outputs = BTreeMap::new();
    for (output_name, binding) in &graph.outputs {
        let value = match binding {
            InputBinding::External(name) => external_values.get(name),
            InputBinding::NodeOutput { node_id, port } => {
                resolved.get(node_id).and_then(|values| values.get(port))
            }
        }
        .ok_or_else(|| StrategyExecutionError::StrategyOutputMissing(output_name.clone()))?;
        value
            .validate_finite()
            .map_err(|_| StrategyExecutionError::NonFiniteStrategyOutput(output_name.clone()))?;
        outputs.insert(output_name.clone(), value.clone());
    }

    Ok(StrategyExecutionReport {
        outputs,
        executed_nodes,
        declared_cost_units,
    })
}

fn runtime_value_matches(value: &DataValue, kind: ValueKind) -> bool {
    match kind {
        ValueKind::Any => true,
        ValueKind::Null => matches!(value, DataValue::Null),
        ValueKind::Boolean => matches!(value, DataValue::Bool(_)),
        ValueKind::Integer => matches!(value, DataValue::Integer(_)),
        ValueKind::Number => matches!(value, DataValue::Integer(_) | DataValue::Number(_)),
        ValueKind::Text => matches!(value, DataValue::Text(_)),
        ValueKind::Sequence => matches!(value, DataValue::Sequence(_)),
        ValueKind::Object => matches!(value, DataValue::Object(_)),
        ValueKind::BlobRef => matches!(value, DataValue::BlobRef(_)),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrategyExecutionError {
    Graph(StrategyError),
    CostBudgetExceeded {
        estimated: u64,
        maximum: u64,
    },
    ExternalInputSetMismatch,
    ExternalInputTypeMismatch(String),
    NonFiniteExternalInput(String),
    UpstreamValueMissing {
        node_id: String,
        port: String,
    },
    RuntimeInputTypeMismatch {
        node_id: String,
        port: String,
    },
    NonFiniteInput {
        node_id: String,
        port: String,
    },
    OperatorFailed {
        operator_id: String,
        message: String,
    },
    OutputPortSetMismatch(String),
    NonFiniteOutput {
        operator_id: String,
        port: String,
    },
    RuntimeOutputTypeMismatch {
        operator_id: String,
        port: String,
    },
    StrategyOutputMissing(String),
    NonFiniteStrategyOutput(String),
    InvalidGraphState,
}

impl std::fmt::Display for StrategyExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Graph(error) => write!(formatter, "strategy graph validation failed: {error}"),
            Self::CostBudgetExceeded { estimated, maximum } => {
                write!(
                    formatter,
                    "strategy cost {estimated} exceeds budget {maximum}"
                )
            }
            Self::ExternalInputSetMismatch => {
                write!(
                    formatter,
                    "external input set does not match the strategy declaration"
                )
            }
            Self::ExternalInputTypeMismatch(name) => {
                write!(
                    formatter,
                    "external input {name} has an incompatible value type"
                )
            }
            Self::NonFiniteExternalInput(name) => {
                write!(
                    formatter,
                    "external input {name} contains a non-finite number"
                )
            }
            Self::UpstreamValueMissing { node_id, port } => {
                write!(formatter, "upstream value missing for {node_id}.{port}")
            }
            Self::RuntimeInputTypeMismatch { node_id, port } => {
                write!(formatter, "runtime input type mismatch at {node_id}.{port}")
            }
            Self::NonFiniteInput { node_id, port } => {
                write!(
                    formatter,
                    "runtime input {node_id}.{port} contains a non-finite number"
                )
            }
            Self::OperatorFailed {
                operator_id,
                message,
            } => write!(
                formatter,
                "trusted operator {operator_id} failed: {message}"
            ),
            Self::OutputPortSetMismatch(operator_id) => {
                write!(
                    formatter,
                    "operator {operator_id} returned missing or unexpected outputs"
                )
            }
            Self::NonFiniteOutput { operator_id, port } => write!(
                formatter,
                "operator {operator_id}.{port} returned a non-finite number"
            ),
            Self::RuntimeOutputTypeMismatch { operator_id, port } => write!(
                formatter,
                "operator {operator_id}.{port} returned an incompatible type"
            ),
            Self::StrategyOutputMissing(name) => {
                write!(formatter, "strategy output {name} could not be resolved")
            }
            Self::NonFiniteStrategyOutput(name) => {
                write!(
                    formatter,
                    "strategy output {name} contains a non-finite number"
                )
            }
            Self::InvalidGraphState => {
                write!(
                    formatter,
                    "validated strategy graph entered an invalid execution state"
                )
            }
        }
    }
}

impl std::error::Error for StrategyExecutionError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrategyError {
    InvalidOperator,
    DuplicateOperator,
    EmptyGraph,
    InvalidGraph,
    DuplicateNode,
    UnknownOperator,
    InputPortMismatch,
    UnknownExternalInput,
    PortTypeMismatch,
    UnknownNode,
    UnknownOutputPort,
    CycleDetected,
    UnreachableNode,
    CostOverflow,
    InvalidPolicy,
    InvalidEvaluation,
    InvalidSpec,
    DuplicateStrategy,
    StrategyNotFound,
    InvalidLifecycle,
    EvaluationReusesTrainingData,
    InsufficientEvaluationSamples,
    NotEvaluated,
    IncomparableEvaluations,
    InsufficientImprovement,
    InvalidSerializedState,
}

impl std::fmt::Display for StrategyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOperator => write!(formatter, "operator specification is invalid"),
            Self::DuplicateOperator => write!(formatter, "operator ID is already registered"),
            Self::EmptyGraph => write!(formatter, "strategy graph must contain nodes and outputs"),
            Self::InvalidGraph => write!(formatter, "strategy graph is invalid"),
            Self::DuplicateNode => write!(formatter, "strategy node ID is duplicated"),
            Self::UnknownOperator => write!(formatter, "strategy references an unknown operator"),
            Self::InputPortMismatch => write!(
                formatter,
                "operator input bindings do not match its declared ports"
            ),
            Self::UnknownExternalInput => {
                write!(formatter, "strategy references an unknown external input")
            }
            Self::PortTypeMismatch => {
                write!(formatter, "strategy graph connects incompatible port types")
            }
            Self::UnknownNode => write!(formatter, "strategy references an unknown node"),
            Self::UnknownOutputPort => {
                write!(formatter, "strategy references an unknown output port")
            }
            Self::CycleDetected => write!(formatter, "strategy graph contains a cycle"),
            Self::UnreachableNode => write!(
                formatter,
                "strategy contains nodes that do not contribute to outputs"
            ),
            Self::CostOverflow => write!(formatter, "strategy aggregate cost overflowed"),
            Self::InvalidPolicy => write!(formatter, "strategy evaluation policy is invalid"),
            Self::InvalidEvaluation => write!(formatter, "strategy evaluation metrics are invalid"),
            Self::InvalidSpec => write!(formatter, "strategy specification is invalid"),
            Self::DuplicateStrategy => write!(formatter, "strategy version already exists"),
            Self::StrategyNotFound => write!(formatter, "strategy was not found"),
            Self::InvalidLifecycle => write!(formatter, "strategy lifecycle transition is invalid"),
            Self::EvaluationReusesTrainingData => write!(
                formatter,
                "evaluation dataset must differ from strategy training data"
            ),
            Self::InsufficientEvaluationSamples => {
                write!(formatter, "strategy evaluation has too few samples")
            }
            Self::NotEvaluated => write!(formatter, "strategy candidate has not been evaluated"),
            Self::IncomparableEvaluations => write!(
                formatter,
                "candidate and incumbent must share the same evaluation dataset"
            ),
            Self::InsufficientImprovement => write!(
                formatter,
                "candidate did not exceed the configured improvement margin"
            ),
            Self::InvalidSerializedState => {
                write!(formatter, "serialized strategy registry is invalid")
            }
        }
    }
}

impl std::error::Error for StrategyError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalogue() -> OperatorCatalogue {
        let mut catalogue = OperatorCatalogue::default();
        catalogue
            .register(OperatorSpec {
                id: "normalize".to_owned(),
                inputs: vec![PortSpec {
                    name: "raw".to_owned(),
                    kind: ValueKind::Number,
                }],
                outputs: vec![PortSpec {
                    name: "normalized".to_owned(),
                    kind: ValueKind::Number,
                }],
                cost_units: 2,
                deterministic: true,
            })
            .expect("register normalize");
        catalogue
            .register(OperatorSpec {
                id: "predict".to_owned(),
                inputs: vec![PortSpec {
                    name: "features".to_owned(),
                    kind: ValueKind::Number,
                }],
                outputs: vec![PortSpec {
                    name: "score".to_owned(),
                    kind: ValueKind::Number,
                }],
                cost_units: 5,
                deterministic: true,
            })
            .expect("register predict");
        catalogue
    }

    fn graph() -> StrategyGraph {
        StrategyGraph {
            external_inputs: BTreeMap::from([("observation".to_owned(), ValueKind::Number)]),
            nodes: vec![
                StrategyNode {
                    id: "transform".to_owned(),
                    operator_id: "normalize".to_owned(),
                    inputs: BTreeMap::from([(
                        "raw".to_owned(),
                        InputBinding::External("observation".to_owned()),
                    )]),
                },
                StrategyNode {
                    id: "model".to_owned(),
                    operator_id: "predict".to_owned(),
                    inputs: BTreeMap::from([(
                        "features".to_owned(),
                        InputBinding::NodeOutput {
                            node_id: "transform".to_owned(),
                            port: "normalized".to_owned(),
                        },
                    )]),
                },
            ],
            outputs: BTreeMap::from([(
                "prediction".to_owned(),
                InputBinding::NodeOutput {
                    node_id: "model".to_owned(),
                    port: "score".to_owned(),
                },
            )]),
        }
    }

    fn spec(version: u64) -> StrategySpec {
        StrategySpec {
            id: "default-learning-pipeline".to_owned(),
            version,
            training_dataset_id: format!("training-{version}"),
            graph: graph(),
        }
    }

    fn policy() -> StrategyPolicy {
        StrategyPolicy {
            minimum_evaluation_samples: 10,
            minimum_improvement: 0.02,
            resource_cost_weight: 0.1,
        }
    }

    #[test]
    fn validates_typed_graph_cost_and_reachability() {
        let catalogue = catalogue();
        let strategy = graph();
        strategy.validate(&catalogue).expect("valid graph");
        assert_eq!(strategy.estimated_cost(&catalogue), Ok(7));
    }

    #[test]
    fn rejects_type_mismatch_cycle_and_disconnected_operator() {
        let catalogue = catalogue();
        let mut wrong_type = graph();
        wrong_type
            .external_inputs
            .insert("observation".to_owned(), ValueKind::Text);
        assert_eq!(
            wrong_type.validate(&catalogue),
            Err(StrategyError::PortTypeMismatch)
        );

        let mut cycle = graph();
        cycle.nodes[0].inputs.insert(
            "raw".to_owned(),
            InputBinding::NodeOutput {
                node_id: "model".to_owned(),
                port: "score".to_owned(),
            },
        );
        assert_eq!(
            cycle.validate(&catalogue),
            Err(StrategyError::CycleDetected)
        );

        let mut disconnected = graph();
        disconnected.nodes.push(StrategyNode {
            id: "unused".to_owned(),
            operator_id: "normalize".to_owned(),
            inputs: BTreeMap::from([(
                "raw".to_owned(),
                InputBinding::External("observation".to_owned()),
            )]),
        });
        assert_eq!(
            disconnected.validate(&catalogue),
            Err(StrategyError::UnreachableNode)
        );
    }

    struct DoubleTrustedOperator;

    impl StrategyOperatorExecutor for DoubleTrustedOperator {
        fn execute(
            &self,
            operator: &OperatorSpec,
            inputs: &BTreeMap<String, DataValue>,
        ) -> Result<BTreeMap<String, DataValue>, String> {
            let input = operator
                .inputs
                .first()
                .ok_or_else(|| "operator has no input".to_owned())?;
            let output = operator
                .outputs
                .first()
                .ok_or_else(|| "operator has no output".to_owned())?;
            let number = inputs
                .get(&input.name)
                .and_then(DataValue::as_f64)
                .ok_or_else(|| "expected numeric value".to_owned())?;
            Ok(BTreeMap::from([(
                output.name.clone(),
                DataValue::Number(number * 2.0),
            )]))
        }
    }

    struct WrongTypeTrustedOperator;

    impl StrategyOperatorExecutor for WrongTypeTrustedOperator {
        fn execute(
            &self,
            operator: &OperatorSpec,
            _inputs: &BTreeMap<String, DataValue>,
        ) -> Result<BTreeMap<String, DataValue>, String> {
            let output = operator
                .outputs
                .first()
                .ok_or_else(|| "no output".to_owned())?;
            Ok(BTreeMap::from([(
                output.name.clone(),
                DataValue::Text("wrong type".to_owned()),
            )]))
        }
    }

    #[test]
    fn executes_trusted_operators_in_deterministic_order_with_type_and_cost_checks() {
        let catalogue = catalogue();
        let graph = graph();
        let external_values = BTreeMap::from([("observation".to_owned(), DataValue::Number(3.0))]);
        let report = execute_strategy_graph(
            &graph,
            &catalogue,
            &external_values,
            &DoubleTrustedOperator,
            7,
        )
        .expect("execute strategy");
        assert_eq!(
            report.outputs.get("prediction"),
            Some(&DataValue::Number(12.0))
        );
        assert_eq!(report.executed_nodes, vec!["transform", "model"]);
        assert_eq!(report.declared_cost_units, 7);

        assert_eq!(
            execute_strategy_graph(
                &graph,
                &catalogue,
                &external_values,
                &DoubleTrustedOperator,
                6,
            ),
            Err(StrategyExecutionError::CostBudgetExceeded {
                estimated: 7,
                maximum: 6,
            })
        );
        assert_eq!(
            execute_strategy_graph(
                &graph,
                &catalogue,
                &BTreeMap::new(),
                &DoubleTrustedOperator,
                7,
            ),
            Err(StrategyExecutionError::ExternalInputSetMismatch)
        );
        assert!(matches!(
            execute_strategy_graph(
                &graph,
                &catalogue,
                &external_values,
                &WrongTypeTrustedOperator,
                7,
            ),
            Err(StrategyExecutionError::RuntimeOutputTypeMismatch { .. })
        ));
    }

    #[test]
    fn synthesizes_bounded_type_compatible_unary_chains() {
        let mut catalogue = catalogue();
        catalogue
            .register(OperatorSpec {
                id: "textualize".to_owned(),
                inputs: vec![PortSpec {
                    name: "number".to_owned(),
                    kind: ValueKind::Number,
                }],
                outputs: vec![PortSpec {
                    name: "text".to_owned(),
                    kind: ValueKind::Text,
                }],
                cost_units: 1,
                deterministic: true,
            })
            .expect("register textualize");

        let result = synthesize_unary_strategy_chains(
            &catalogue,
            StrategySynthesisPolicy {
                input_name: "observation".to_owned(),
                output_name: "prediction".to_owned(),
                input_kind: ValueKind::Number,
                output_kind: ValueKind::Number,
                maximum_depth: 2,
                maximum_cost_units: 7,
                maximum_candidates: 10,
                maximum_expansions: 100,
                allow_nondeterministic: false,
            },
        )
        .expect("synthesize");
        assert_eq!(result.candidates.len(), 4);
        assert!(!result.truncated);
        for candidate in &result.candidates {
            candidate
                .graph
                .validate(&catalogue)
                .expect("generated graph validates");
            assert!(candidate.cost_units <= 7);
        }
    }

    #[test]
    fn synthesis_respects_candidate_count_and_cost_budgets() {
        let catalogue = catalogue();
        let limited = synthesize_unary_strategy_chains(
            &catalogue,
            StrategySynthesisPolicy {
                input_name: "input".to_owned(),
                output_name: "output".to_owned(),
                input_kind: ValueKind::Number,
                output_kind: ValueKind::Number,
                maximum_depth: 3,
                maximum_cost_units: 6,
                maximum_candidates: 1,
                maximum_expansions: 100,
                allow_nondeterministic: false,
            },
        )
        .expect("bounded synthesis");
        assert_eq!(limited.candidates.len(), 1);
        assert!(limited.truncated);

        let no_candidate = synthesize_unary_strategy_chains(
            &catalogue,
            StrategySynthesisPolicy {
                input_name: "input".to_owned(),
                output_name: "output".to_owned(),
                input_kind: ValueKind::Text,
                output_kind: ValueKind::Number,
                maximum_depth: 3,
                maximum_cost_units: 100,
                maximum_candidates: 10,
                maximum_expansions: 100,
                allow_nondeterministic: false,
            },
        )
        .expect("empty compatible search");
        assert!(no_candidate.candidates.is_empty());
    }

    #[test]
    fn promotion_requires_comparable_held_out_improvement_and_round_trips() {
        let catalogue = catalogue();
        let mut registry = StrategyRegistry::default();
        registry
            .register_candidate(spec(1), &catalogue)
            .expect("candidate v1");
        registry
            .evaluate_candidate(
                "default-learning-pipeline",
                1,
                StrategyEvaluation {
                    evaluation_dataset_id: "holdout-v1".to_owned(),
                    metric_id: "score".to_owned(),
                    direction: ScoreDirection::HigherIsBetter,
                    sample_count: 20,
                    primary_score: 0.8,
                    resource_cost: 2.0,
                },
                policy(),
            )
            .expect("evaluate v1");
        registry
            .promote("default-learning-pipeline", 1, policy())
            .expect("promote v1");

        registry
            .register_candidate(spec(2), &catalogue)
            .expect("candidate v2");
        registry
            .evaluate_candidate(
                "default-learning-pipeline",
                2,
                StrategyEvaluation {
                    evaluation_dataset_id: "holdout-v1".to_owned(),
                    metric_id: "score".to_owned(),
                    direction: ScoreDirection::HigherIsBetter,
                    sample_count: 20,
                    primary_score: 0.9,
                    resource_cost: 2.0,
                },
                policy(),
            )
            .expect("evaluate v2");
        registry
            .promote("default-learning-pipeline", 2, policy())
            .expect("promote v2");
        assert_eq!(
            registry
                .active("default-learning-pipeline")
                .map(|r| r.spec.version),
            Some(2)
        );
        assert_eq!(
            registry
                .get("default-learning-pipeline", 1)
                .map(|r| r.status),
            Some(StrategyStatus::Retired)
        );

        let artifact = registry.to_artifact_value();
        let restored =
            StrategyRegistry::from_artifact_value(&artifact, &catalogue).expect("restore");
        assert_eq!(restored, registry);
    }

    #[test]
    fn candidate_cannot_promote_on_different_or_tiny_validation_set() {
        let catalogue = catalogue();
        let mut registry = StrategyRegistry::default();
        registry
            .register_candidate(spec(1), &catalogue)
            .expect("v1");
        registry
            .evaluate_candidate(
                "default-learning-pipeline",
                1,
                StrategyEvaluation {
                    evaluation_dataset_id: "holdout".to_owned(),
                    metric_id: "score".to_owned(),
                    direction: ScoreDirection::HigherIsBetter,
                    sample_count: 20,
                    primary_score: 0.7,
                    resource_cost: 1.0,
                },
                policy(),
            )
            .expect("eval");
        registry
            .promote("default-learning-pipeline", 1, policy())
            .expect("promote");

        registry
            .register_candidate(spec(2), &catalogue)
            .expect("v2");
        registry
            .evaluate_candidate(
                "default-learning-pipeline",
                2,
                StrategyEvaluation {
                    evaluation_dataset_id: "another-holdout".to_owned(),
                    metric_id: "score".to_owned(),
                    direction: ScoreDirection::HigherIsBetter,
                    sample_count: 20,
                    primary_score: 0.99,
                    resource_cost: 0.0,
                },
                policy(),
            )
            .expect("eval v2");
        assert_eq!(
            registry.promote("default-learning-pipeline", 2, policy()),
            Err(StrategyError::IncomparableEvaluations)
        );
        assert_eq!(
            registry
                .active("default-learning-pipeline")
                .map(|r| r.spec.version),
            Some(1)
        );
    }
}
