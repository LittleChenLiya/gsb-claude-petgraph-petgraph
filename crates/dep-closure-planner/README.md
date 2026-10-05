# dep-closure-planner

Dependency closure planner: given **target nodes** and optional
**requirement groups** per node (an AND of ORs — every group picks exactly
one available provider predecessor), compute the **required node set with
the fewest nodes**, together with the selection rationale and a topological
execution plan.

Properties:

* Shared dependencies are counted once — the planner searches whole
  assignments, never greedy per-target minimal branches.
* Cycles are reported as explicit, non-executable diagnostics; a cyclic
  branch is never a valid satisfaction.
* Small graphs run under an explicit search budget; exceeding it returns
  `PlanOutcome::Unknown` (a distinct variant), never a false `Unsatisfiable`.
* Planning takes `&self`: the input problem graph is never mutated.

## Local cargo commands

```text
cargo test -p dep-closure-planner                 # all tests
cargo test -p dep-closure-planner --test planner  # acceptance tests only
cargo run  -p dep-closure-planner --example six_node
```

## Six-node example

`examples/six_node.rs` builds six nodes — `app`, `worker`, `postgres`,
`sqlite`, `metrics`, `cache` — where both targets (`app`, `worker`) can
share one database engine and the `metrics` node. The joint plan needs
5 nodes, strictly less than the 3 + 4 = 7 nodes of planning each target
independently, and prints the per-group selection rationale plus the
topological execution order.
