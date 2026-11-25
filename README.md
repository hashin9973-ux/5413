# 5413

5413 is an environment-independent adaptive learning runtime.

Its first environment is a simulator, but the core architecture is designed to work with different observation/action protocols. The initial implementation focuses on:

- typed, domain-neutral environment contracts;
- schema-validated dynamic observations;
- explicit action authorization;
- append-only in-memory experience capture;
- versioned, dynamically registered representations;
- a small online transition learner;
- physics and finite-state workflow examples that use the same generic environment API.

The first implementation is intentionally dependency-free. Persistence, advanced neural models, planner synthesis, and executable self-modification are future stages—not assumed capabilities of this foundation.

## Development

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo run -p engine-core --example simulator_demo
```

See [docs/architecture.md](docs/architecture.md) for the initial engineering boundaries and [docs/roadmap.md](docs/roadmap.md) for staged development.

## Quality checks

The Rust CI workflow runs formatting checks, Clippy with warnings denied, the workspace test suite, and the simulator example on pushes and pull requests.
