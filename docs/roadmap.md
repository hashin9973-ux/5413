# Development roadmap

## Milestone 1 — Environment-independent foundation

- [x] Typed environment descriptors and observation contracts
- [x] Dynamic observation schemas
- [x] Action proposals and authorization boundary
- [x] In-memory experience ledger
- [x] Versioned representation registry
- [x] Baseline numeric transition learner
- [x] Simulator adapter example
- [x] CI execution result verified on GitHub Actions
- [x] Second, structurally different environment adapter

## Milestone 2 — Reliable experience and replay

- [x] Durable append-only file journal with synchronized writes and incomplete-tail recovery
- [x] Episode replay in original append order
- [x] Versioned record encoding and corruption detection
- [ ] Checkpoint and restore contract
- [ ] Idempotency keys and duplicate observation handling
- [ ] Journal format migration routines

## Milestone 3 — Dynamic representation learning

- [x] Candidate representation generation from observed object structures
- [x] Evidence-backed schema validity scoring on separately identified held-out data
- [x] Versioned field mapping and transformation contracts
- [x] Basic source-field loss/coverage reporting and target schema validation
- [ ] Recursive nested-structure discovery
- [ ] Semantic representation quality scoring beyond schema validity
- [ ] Automated mapping discovery and held-out rollback experiments

## Milestone 4 — World models and reasoning

- Interchangeable prediction model interface
- Temporal and relational models
- Symbolic and statistical inference strategies
- Competing hypotheses and uncertainty
- Intervention-aware causal evaluation

## Milestone 5 — Autonomous experimentation

- Goal and attention management
- Experiment design and resource budgets
- Action planning and authorization
- Curiosity and expected information gain
- Safe exploration policies

## Milestone 6 — Meta-learning

- Strategy definition and versioning
- Strategy execution as constrained operator graphs
- Strategy selection and composition
- Held-out learning-efficiency benchmarks
- Candidate strategy promotion and rollback

## Milestone 7 — Controlled algorithm improvement

- Operator proposals
- Sandboxed implementation experiments
- Reproducible build and benchmark pipeline
- Security review and regression gates
- Explicit promotion to trusted operator registry

## Exit criteria for generality

A new environment must be integrated through an adapter and its declared schemas/actions. It must not require physics-specific or other domain-specific changes to the intelligence core. Transfer must be measured against independent-learning baselines, not inferred from shared interface compatibility alone.
