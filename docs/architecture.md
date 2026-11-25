# Initial architecture contract

## Purpose

The engine core is environment-independent. Environment-specific protocols and domain knowledge belong in adapters and optional extensions, not in the core crate.

The first implementation is a deliberately small foundation. It does not claim general intelligence, causal discovery, durable memory, or autonomous code modification.

## Boundaries

1. **Environment adapter:** describes observations, actions, lifecycle capabilities, and feedback.
2. **Experience ledger:** records observations and action outcomes as historical evidence.
3. **Representation registry:** manages versioned candidate and active representations.
4. **Cognitive and learning components:** consume typed inputs and produce proposals or model outputs.
5. **Executive runtime:** chooses work, enforces budgets, and authorizes environment actions.
6. **Evaluation:** compares candidate behavior with baselines on held-out tasks.

## Invariants

- No physics-specific types in the core API.
- Observations are preserved independently of derived representations.
- Runtime observations are validated against declared, versioned schemas.
- Environment actions require an opaque authorized-action value.
- Candidate representations are not active until explicitly promoted.
- At most one representation version per representation ID is active in the registry.
- Learning algorithms return explicit errors for invalid numeric data.
- Strategy and algorithm changes require evaluation before promotion.

## Implemented in this foundation

- A recursive, extensible DataValue tree.
- Dynamic object schemas and validation.
- Environment and action descriptors.
- Action-risk authorization.
- A minimal in-memory append-only experience ledger.
- Candidate/active/rejected/retired representation lifecycle with held-out schema validation before promotion.
- Structural first-level schema inference from object-shaped observations.
- Explicit, schema-validated field mappings with identity, affine numeric, and constant transforms; mapping reports quantify dropped source fields.
- A per-action numeric mean-delta learner.
- A one-dimensional particle simulator and a finite-state workflow example using the same generic adapter contract.
- An end-to-end representation discovery, held-out evaluation, promotion, and mapping example.

## Deliberately not implemented yet

- Durable event storage and schema migrations.
- Async/streaming adapters and multiple concurrent sessions.
- Tensor/image/audio payload storage and modality-specific encoders.
- General neural or symbolic reasoning implementations.
- Causal discovery and intervention planning.
- General-purpose dynamic graph or program execution.
- Autonomous goals and long-running executive scheduling.
- Learning-strategy synthesis and executable self-modification.

These are future stages, not implied properties of the current code.

## Dependency rules

- The core types are centralized initially; extraction into separate crates requires demonstrated reuse.
- Environment adapters depend on the environment API, never the reverse.
- Core learning and representation modules do not depend on a concrete environment adapter.
- Environment actions are executed only after authorization.
- Candidate models and representations are evaluated outside the trusted path before promotion.

## Representation lifecycle

A representation proposal records its source training dataset ID and evidence references. A candidate is evaluated on a separately identified validation dataset; the registry rejects reuse of the same dataset ID and promotion is blocked until the candidate meets the minimum sample and valid-fraction policy. The evaluation record retains the policy and measured counts for auditability. This is an explicit provenance guard, not cryptographic proof that datasets are disjoint: the caller remains responsible for supplying genuinely held-out samples. Promoting a newer version retires the previously active version without deleting its record. Raw experience remains the historical source of truth.

## Discovery and mapping scope

The initial discovery routine infers a top-level object schema from observations. It unions field names, marks fields absent from some samples optional, widens integer/number mixtures to number, and uses Any for incompatible observed types. It does not yet infer recursive nested schemas, semantic concepts, or causal relations. A mapping specifies source and target representation versions, field transforms, and target validation. Its coverage metric reports how many observed source fields are represented in the output; it is not a semantic equivalence score.

## Numeric baseline learner

NumericDeltaModel estimates mean field deltas conditioned on action ID for numeric fields present in both before/after object observations. It is useful for proving the data path and evaluation mechanics; it is not a general transition model and must not be treated as one.

## Action safety

The default authorization policy permits read-only actions. The simulator and finite-state examples opt into reversible actions. An authorized action is bound to the environment ID and version, and adapters validate that binding, the declared action, its risk, and its argument schema before applying effects. Irreversible operations require a policy that explicitly permits them; external-action support must add capability, permission, and environment-specific safety checks before execution.
