use engine_core::{
    propose_representation, DataValue, EvaluationPolicy, FieldMapping, FieldSpec, MappingTransform,
    RepresentationKind, RepresentationMapping, RepresentationRef, RepresentationRegistry,
    RepresentationSpec, SchemaDefinition, ValueKind,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let training_samples = vec![
        DataValue::object([
            ("distance".to_owned(), DataValue::Integer(2)),
            ("category".to_owned(), DataValue::Text("near".to_owned())),
        ]),
        DataValue::object([(
            "distance".to_owned(),
            DataValue::Number(4.0),
        )]),
    ];

    let proposal = propose_representation(
        "observed-environment-state",
        1,
        RepresentationKind::Relational,
        "A schema inferred from environment observations",
        "training-observation-batch",
        vec!["episode:alpha".to_owned(), "episode:beta".to_owned()],
        &training_samples,
    )?;

    let mut registry = RepresentationRegistry::default();
    registry.register_candidate(proposal.spec.clone())?;

    let held_out_samples = vec![DataValue::object([(
        "distance".to_owned(),
        DataValue::Number(6.0),
    )])];
    let evaluation = registry.evaluate_candidate(
        "observed-environment-state",
        1,
        "held-out-observation-batch",
        &held_out_samples,
        EvaluationPolicy {
            minimum_validation_samples: 1,
            minimum_valid_fraction: 1.0,
        },
    )?;
    if !evaluation.passed {
        return Err("discovered representation failed held-out evaluation".into());
    }
    registry.promote("observed-environment-state", 1)?;

    let target = RepresentationSpec {
        id: "normalized-state".to_owned(),
        version: 1,
        kind: RepresentationKind::Statistical,
        schema: SchemaDefinition {
            id: "normalized-state-schema".to_owned(),
            version: 1,
            fields: vec![
                FieldSpec {
                    name: "scaled_distance".to_owned(),
                    kind: ValueKind::Number,
                    required: true,
                },
                FieldSpec {
                    name: "context".to_owned(),
                    kind: ValueKind::Text,
                    required: true,
                },
            ],
            allow_additional_fields: false,
        },
        description: "A second representation with normalized numeric data".to_owned(),
        training_set_id: "normalization-training-batch".to_owned(),
        evidence_refs: vec!["normalization-config:1".to_owned()],
    };

    let mapping = RepresentationMapping {
        id: "state-to-normalized-state".to_owned(),
        version: 1,
        source: RepresentationRef::from_spec(&proposal.spec),
        target: RepresentationRef::from_spec(&target),
        fields: vec![
            FieldMapping {
                source_field: Some("distance".to_owned()),
                target_field: "scaled_distance".to_owned(),
                transform: MappingTransform::NumericAffine {
                    scale: 0.1,
                    offset: 0.0,
                },
            },
            FieldMapping {
                source_field: None,
                target_field: "context".to_owned(),
                transform: MappingTransform::Constant(DataValue::Text(
                    "normalized".to_owned(),
                )),
            },
        ],
    };

    let mapped = mapping.apply(&proposal.spec, &target, &held_out_samples[0])?;
    println!(
        "discovered {} fields from {} samples ({} optional); held-out validity {:.0}%",
        proposal.summary.discovered_field_count,
        proposal.summary.sample_count,
        proposal.summary.optional_field_count,
        evaluation.valid_fraction * 100.0
    );
    println!(
        "promoted representation {}@{} and mapped held-out data to {:?} (source coverage {:.0}%)",
        proposal.spec.id,
        proposal.spec.version,
        mapped.value,
        mapped.report.source_coverage * 100.0
    );
    Ok(())
}
