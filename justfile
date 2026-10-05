# Lists all available recipes
default:
    just --list

# Builds the project in development mode
build:
    cargo build

# Tests with all features enabled
test:
    cargo test --all-features

# Runs only the dependency closure planner tests (small, deterministic)
test-closure:
    cargo test -p petgraph --test dependency_closure

# Runs the dependency closure planner example (six-node sharing demo)
example-closure:
    cargo run -p petgraph --example closure_planner

# Miri with all tests (this might take very long)
miri:
    cargo miri nextest run

# Fmt with the same configuration as in CI
fmt:
    cargo fmt --all -- --check

# Clippy with the same configuration as in CI
clippy:
    cargo clippy --all-features --lib --bins --examples --tests -- -D warnings

# Runs all linting checks that are run in CI
lint: fmt clippy

# Runs all tests and linting that are run in CI
ci: fmt clippy test

# Checks if no-std is working same as in CI. Requires the wasm32v1-none target to be installed
check-no-std:
    cargo check --no-default-features -p petgraph --target wasm32v1-none --features graphmap,serde-1,stable_graph,matrix_graph,generate,unstable
