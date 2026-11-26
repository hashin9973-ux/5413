//! Evidence-backed hypotheses with explicit, auditable belief updates.
//!
//! Evidence contributes caller-supplied log Bayes factors. Combining factors
//! assumes conditional independence given the hypothesis; this crate does not
//! discover causal effects or verify that evidence is independent.

use crate::value::DataValue;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HypothesisPolicy {
    pub minimum_evidence_count: usize,
    pub support_probability: f64,
    pub refutation_probability: f64,
}

impl Default for HypothesisPolicy {
    fn default() -> Self {
        Self {
            minimum_evidence_count: 2,
            support_probability: 0.95,
            refutation_probability: 0.05,
        }
    }
}

impl HypothesisPolicy {
    pub fn validate(&self) -> Result<(), ReasoningError> {
        if self.minimum_evidence_count == 0
            || !self.support_probability.is_finite()
            || !self.refutation_probability.is_finite()
            || self.refutation_probability <= 0.0
            || self.support_probability >= 1.0
            || self.refutation_probability >= self.support_probability
        {
            return Err(ReasoningError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HypothesisSpec {
    pub id: String,
    /// A typed, domain-independent claim. Interpretation belongs to the caller.
    pub claim: DataValue,
    pub description: String,
    pub prior_probability: f64,
    pub source_dataset_id: Option<String>,
}

impl HypothesisSpec {
    pub fn validate(&self) -> Result<(), ReasoningError> {
        if self.id.trim().is_empty()
            || self.description.trim().is_empty()
            || !self.prior_probability.is_finite()
            || self.prior_probability <= 0.0
            || self.prior_probability >= 1.0
            || self.source_dataset_id.as_ref().is_some_and(|id| id.trim().is_empty())
        {
            return Err(ReasoningError::InvalidHypothesis);
        }
        self.claim
            .validate_finite()
            .map_err(|_| ReasoningError::InvalidHypothesis)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceRecord {
    pub evidence_ref: String,
    pub dataset_id: String,
    pub log_likelihood_ratio: f64,
    pub rationale: String,
}

impl EvidenceRecord {
    pub fn validate(&self) -> Result<(), ReasoningError> {
        if self.evidence_ref.trim().is_empty()
            || self.dataset_id.trim().is_empty()
            || self.rationale.trim().is_empty()
            || !self.log_likelihood_ratio.is_finite()
        {
            return Err(ReasoningError::InvalidEvidence);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HypothesisStatus {
    Candidate,
    Supported,
    Refuted,
    Inconclusive,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HypothesisRecord {
    pub spec: HypothesisSpec,
    pub evidence: Vec<EvidenceRecord>,
    pub status: HypothesisStatus,
    pub posterior_probability: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HypothesisRegistry {
    records: BTreeMap<String, HypothesisRecord>,
}

impl HypothesisRegistry {
    pub fn insert(
        &mut self,
        spec: HypothesisSpec,
        policy: HypothesisPolicy,
    ) -> Result<(), ReasoningError> {
        spec.validate()?;
        policy.validate()?;
        if self.records.contains_key(&spec.id) {
            return Err(ReasoningError::DuplicateHypothesis);
        }
        let posterior_probability = spec.prior_probability;
        let record = HypothesisRecord {
            spec,
            evidence: Vec::new(),
            status: HypothesisStatus::Candidate,
            posterior_probability,
        };
        self.records.insert(record.spec.id.clone(), record);
        Ok(())
    }

    /// Adds one unique evidence item and recomputes the posterior and status.
    ///
    /// Evidence is combined in log-odds space. Callers must avoid double-counting
    /// dependent observations or evidence already represented by the prior.
    pub fn add_evidence(
        &mut self,
        hypothesis_id: &str,
        evidence: EvidenceRecord,
        policy: HypothesisPolicy,
    ) -> Result<&HypothesisRecord, ReasoningError> {
        policy.validate()?;
        evidence.validate()?;
        let record = self
            .records
            .get_mut(hypothesis_id)
            .ok_or(ReasoningError::HypothesisNotFound)?;
        if record
            .evidence
            .iter()
            .any(|existing| existing.evidence_ref == evidence.evidence_ref)
        {
            return Err(ReasoningError::DuplicateEvidence);
        }
        if record.spec.source_dataset_id.as_deref() == Some(evidence.dataset_id.as_str()) {
            return Err(ReasoningError::EvidenceReusesSourceDataset);
        }

        let prior_log_odds = logit(record.spec.prior_probability);
        let evidence_log_ratio: f64 = record
            .evidence
            .iter()
            .map(|item| item.log_likelihood_ratio)
            .sum::<f64>()
            + evidence.log_likelihood_ratio;
        let posterior_log_odds = prior_log_odds + evidence_log_ratio;
        if !posterior_log_odds.is_finite() {
            return Err(ReasoningError::NumericalOverflow);
        }
        let posterior_probability = logistic(posterior_log_odds);

        record.evidence.push(evidence);
        record.posterior_probability = posterior_probability;
        record.status = classify(
            posterior_probability,
            record.evidence.len(),
            policy,
        );
        Ok(record)
    }

    pub fn get(&self, id: &str) -> Option<&HypothesisRecord> {
        self.records.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &HypothesisRecord> {
        self.records.values()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn to_artifact_value(&self) -> DataValue {
        DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            (
                "hypotheses".to_owned(),
                DataValue::Sequence(self.records.values().map(record_to_value).collect()),
            ),
        ])
    }

    pub fn from_artifact_value(
        value: &DataValue,
        policy: HypothesisPolicy,
    ) -> Result<Self, ReasoningError> {
        policy.validate()?;
        let fields = value
            .as_object()
            .ok_or(ReasoningError::InvalidSerializedState)?;
        if fields.len() != 2 || fields.get("artifact_version") != Some(&DataValue::Integer(1)) {
            return Err(ReasoningError::InvalidSerializedState);
        }
        let hypotheses = match fields.get("hypotheses") {
            Some(DataValue::Sequence(values)) => values,
            _ => return Err(ReasoningError::InvalidSerializedState),
        };
        let mut records = BTreeMap::new();
        for value in hypotheses {
            let record = record_from_value(value, policy)?;
            if records.insert(record.spec.id.clone(), record).is_some() {
                return Err(ReasoningError::InvalidSerializedState);
            }
        }
        Ok(Self { records })
    }
}

fn classify(
    probability: f64,
    evidence_count: usize,
    policy: HypothesisPolicy,
) -> HypothesisStatus {
    if evidence_count == 0 {
        HypothesisStatus::Candidate
    } else if evidence_count < policy.minimum_evidence_count {
        HypothesisStatus::Inconclusive
    } else if probability >= policy.support_probability {
        HypothesisStatus::Supported
    } else if probability <= policy.refutation_probability {
        HypothesisStatus::Refuted
    } else {
        HypothesisStatus::Inconclusive
    }
}

fn logit(probability: f64) -> f64 {
    (probability / (1.0 - probability)).ln()
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn record_to_value(record: &HypothesisRecord) -> DataValue {
    let source_dataset = record
        .spec
        .source_dataset_id
        .as_ref()
        .map(|value| DataValue::Text(value.clone()))
        .unwrap_or(DataValue::Null);
    let evidence = DataValue::Sequence(
        record
            .evidence
            .iter()
            .map(|item| {
                DataValue::object([
                    ("evidence_ref".to_owned(), DataValue::Text(item.evidence_ref.clone())),
                    ("dataset_id".to_owned(), DataValue::Text(item.dataset_id.clone())),
                    ("log_likelihood_ratio".to_owned(), DataValue::Number(item.log_likelihood_ratio)),
                    ("rationale".to_owned(), DataValue::Text(item.rationale.clone())),
                ])
            })
            .collect(),
    );
    DataValue::object([
        ("id".to_owned(), DataValue::Text(record.spec.id.clone())),
        ("claim".to_owned(), record.spec.claim.clone()),
        ("description".to_owned(), DataValue::Text(record.spec.description.clone())),
        ("prior_probability".to_owned(), DataValue::Number(record.spec.prior_probability)),
        ("source_dataset_id".to_owned(), source_dataset),
        ("evidence".to_owned(), evidence),
        ("posterior_probability".to_owned(), DataValue::Number(record.posterior_probability)),
        ("status".to_owned(), DataValue::Text(status_name(record.status).to_owned())),
    ])
}

fn status_name(status: HypothesisStatus) -> &'static str {
    match status {
        HypothesisStatus::Candidate => "candidate",
        HypothesisStatus::Supported => "supported",
        HypothesisStatus::Refuted => "refuted",
        HypothesisStatus::Inconclusive => "inconclusive",
    }
}

fn text_field(
    fields: &BTreeMap<String, DataValue>,
    name: &str,
) -> Result<String, ReasoningError> {
    match fields.get(name) {
        Some(DataValue::Text(value)) => Ok(value.clone()),
        _ => Err(ReasoningError::InvalidSerializedState),
    }
}

fn numeric_field(
    fields: &BTreeMap<String, DataValue>,
    name: &str,
) -> Result<f64, ReasoningError> {
    fields
        .get(name)
        .and_then(DataValue::as_f64)
        .filter(|value| value.is_finite())
        .ok_or(ReasoningError::InvalidSerializedState)
}

fn record_from_value(
    value: &DataValue,
    policy: HypothesisPolicy,
) -> Result<HypothesisRecord, ReasoningError> {
    let fields = value
        .as_object()
        .ok_or(ReasoningError::InvalidSerializedState)?;
    if fields.len() != 8 {
        return Err(ReasoningError::InvalidSerializedState);
    }
    let id = text_field(fields, "id")?;
    let claim = fields
        .get("claim")
        .cloned()
        .ok_or(ReasoningError::InvalidSerializedState)?;
    let description = text_field(fields, "description")?;
    let prior_probability = numeric_field(fields, "prior_probability")?;
    let source_dataset_id = match fields.get("source_dataset_id") {
        Some(DataValue::Null) => None,
        Some(DataValue::Text(value)) if !value.trim().is_empty() => Some(value.clone()),
        _ => return Err(ReasoningError::InvalidSerializedState),
    };
    let spec = HypothesisSpec {
        id,
        claim,
        description,
        prior_probability,
        source_dataset_id,
    };
    spec.validate()?;

    let evidence_values = match fields.get("evidence") {
        Some(DataValue::Sequence(values)) => values,
        _ => return Err(ReasoningError::InvalidSerializedState),
    };
    let mut evidence = Vec::with_capacity(evidence_values.len());
    let mut seen_refs = BTreeSet::new();
    let mut log_likelihood_sum = 0.0;
    for value in evidence_values {
        let item = value
            .as_object()
            .ok_or(ReasoningError::InvalidSerializedState)?;
        if item.len() != 4 {
            return Err(ReasoningError::InvalidSerializedState);
        }
        let record = EvidenceRecord {
            evidence_ref: text_field(item, "evidence_ref")?,
            dataset_id: text_field(item, "dataset_id")?,
            log_likelihood_ratio: numeric_field(item, "log_likelihood_ratio")?,
            rationale: text_field(item, "rationale")?,
        };
        record.validate()?;
        if !seen_refs.insert(record.evidence_ref.clone())
            || spec.source_dataset_id.as_deref() == Some(record.dataset_id.as_str())
        {
            return Err(ReasoningError::InvalidSerializedState);
        }
        log_likelihood_sum += record.log_likelihood_ratio;
        evidence.push(record);
    }
    let expected_posterior = logistic(logit(spec.prior_probability) + log_likelihood_sum);
    let posterior_probability = numeric_field(fields, "posterior_probability")?;
    if (expected_posterior - posterior_probability).abs() > 1e-10 {
        return Err(ReasoningError::InvalidSerializedState);
    }
    let status = match text_field(fields, "status")?.as_str() {
        "candidate" => HypothesisStatus::Candidate,
        "supported" => HypothesisStatus::Supported,
        "refuted" => HypothesisStatus::Refuted,
        "inconclusive" => HypothesisStatus::Inconclusive,
        _ => return Err(ReasoningError::InvalidSerializedState),
    };
    if status != classify(posterior_probability, evidence.len(), policy) {
        return Err(ReasoningError::InvalidSerializedState);
    }
    Ok(HypothesisRecord {
        spec,
        evidence,
        status,
        posterior_probability,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReasoningError {
    InvalidPolicy,
    InvalidHypothesis,
    InvalidEvidence,
    DuplicateHypothesis,
    DuplicateEvidence,
    EvidenceReusesSourceDataset,
    HypothesisNotFound,
    NumericalOverflow,
    InvalidSerializedState,
}

impl std::fmt::Display for ReasoningError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPolicy => write!(formatter, "hypothesis decision policy is invalid"),
            Self::InvalidHypothesis => write!(formatter, "hypothesis specification is invalid"),
            Self::InvalidEvidence => write!(formatter, "evidence record is invalid"),
            Self::DuplicateHypothesis => write!(formatter, "hypothesis ID already exists"),
            Self::DuplicateEvidence => write!(formatter, "evidence reference was already applied"),
            Self::EvidenceReusesSourceDataset => write!(formatter, "evidence reuses the hypothesis source dataset"),
            Self::HypothesisNotFound => write!(formatter, "hypothesis was not found"),
            Self::NumericalOverflow => write!(formatter, "evidence accumulation overflowed"),
            Self::InvalidSerializedState => write!(formatter, "serialized hypothesis registry is invalid"),
        }
    }
}

impl std::error::Error for ReasoningError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> HypothesisPolicy {
        HypothesisPolicy {
            minimum_evidence_count: 2,
            support_probability: 0.9,
            refutation_probability: 0.1,
        }
    }

    fn spec() -> HypothesisSpec {
        HypothesisSpec {
            id: "friction-increases-deceleration".to_owned(),
            claim: DataValue::object([
                ("relation".to_owned(), DataValue::Text("friction -> deceleration".to_owned())),
                ("direction".to_owned(), DataValue::Text("positive".to_owned())),
            ]),
            description: "Higher friction is associated with greater deceleration in this environment".to_owned(),
            prior_probability: 0.5,
            source_dataset_id: Some("discovery-set".to_owned()),
        }
    }

    fn evidence(reference: &str, dataset: &str, log_bf: f64) -> EvidenceRecord {
        EvidenceRecord {
            evidence_ref: reference.to_owned(),
            dataset_id: dataset.to_owned(),
            log_likelihood_ratio: log_bf,
            rationale: "Pre-registered comparison outcome".to_owned(),
        }
    }

    #[test]
    fn independent_evidence_updates_posterior_and_decision() {
        let mut registry = HypothesisRegistry::default();
        registry.insert(spec(), policy()).expect("insert");
        let first = registry.add_evidence(
            "friction-increases-deceleration",
            evidence("trial-1", "validation-1", 2.0),
            policy(),
        ).expect("evidence");
        assert_eq!(first.status, HypothesisStatus::Inconclusive);
        let second = registry.add_evidence(
            "friction-increases-deceleration",
            evidence("trial-2", "validation-2", 2.0),
            policy(),
        ).expect("second evidence");
        assert_eq!(second.status, HypothesisStatus::Supported);
        assert!(second.posterior_probability > 0.9);
    }

    #[test]
    fn duplicate_or_training_set_evidence_is_rejected() {
        let mut registry = HypothesisRegistry::default();
        registry.insert(spec(), policy()).expect("insert");
        registry.add_evidence(
            "friction-increases-deceleration",
            evidence("trial-1", "validation-1", 1.0),
            policy(),
        ).expect("first");
        assert_eq!(
            registry.add_evidence(
                "friction-increases-deceleration",
                evidence("trial-1", "validation-2", 1.0),
                policy(),
            ),
            Err(ReasoningError::DuplicateEvidence)
        );
        assert_eq!(
            registry.add_evidence(
                "friction-increases-deceleration",
                evidence("trial-2", "discovery-set", 1.0),
                policy(),
            ),
            Err(ReasoningError::EvidenceReusesSourceDataset)
        );
    }

    #[test]
    fn registry_artifact_round_trips_and_rejects_corrupt_posterior() {
        let mut registry = HypothesisRegistry::default();
        registry.insert(spec(), policy()).expect("insert");
        registry.add_evidence(
            "friction-increases-deceleration",
            evidence("trial-1", "validation-1", 2.0),
            policy(),
        ).expect("evidence");
        let artifact = registry.to_artifact_value();
        let restored = HypothesisRegistry::from_artifact_value(&artifact, policy()).expect("restore");
        assert_eq!(restored, registry);

        let corrupt = DataValue::object([
            ("artifact_version".to_owned(), DataValue::Integer(1)),
            ("hypotheses".to_owned(), DataValue::Sequence(Vec::new())),
            ("unexpected".to_owned(), DataValue::Null),
        ]);
        assert_eq!(
            HypothesisRegistry::from_artifact_value(&corrupt, policy()),
            Err(ReasoningError::InvalidSerializedState)
        );
    }
}
