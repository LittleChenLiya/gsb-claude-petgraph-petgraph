//! Six-node dependency-closure example.
//!
//! ```text
//! cargo run -p dep-closure-planner --example six_node
//! ```
//!
//! Nodes (6): `app`, `worker`, `postgres`, `sqlite`, `metrics`, `cache`.
//!
//! * `app` needs one of {postgres, sqlite} (group "database") and
//!   `metrics` (group "observability").
//! * `worker` needs one of {postgres, sqlite} (group "database") and
//!   `cache` (group "cache").
//! * `cache` needs `metrics` (group "observability").
//!
//! Targets: `app` and `worker`. Sharing `postgres` (or `sqlite`) and
//! `metrics` across both targets yields a 5-node closure, strictly smaller
//! than the 3 + 4 = 7 nodes of the two independent plans.

use dep_closure_planner::{PlanOutcome, Planner};

fn build_problem() -> (Planner, usize, usize) {
    let mut planner = Planner::new();
    let app = planner.add_node("app");
    let worker = planner.add_node("worker");
    let postgres = planner.add_node("postgres");
    let sqlite = planner.add_node("sqlite");
    let metrics = planner.add_node("metrics");
    let cache = planner.add_node("cache");

    planner.add_group(app, "database", &[postgres, sqlite]);
    planner.add_group(app, "observability", &[metrics]);
    planner.add_group(worker, "database", &[postgres, sqlite]);
    planner.add_group(worker, "cache", &[cache]);
    planner.add_group(cache, "observability", &[metrics]);

    (planner, app, worker)
}

fn main() {
    let (problem, app, worker) = build_problem();
    let name = |id: usize| problem.node_name(id).to_string();

    // Joint plan for both targets: shared dependencies counted once.
    let joint = match problem.plan(&[app, worker]) {
        PlanOutcome::Planned(plan) => plan,
        other => panic!("expected a plan, got {other:?}"),
    };

    println!("== joint plan for targets [app, worker] ==");
    println!(
        "required set ({} nodes): {:?}",
        joint.size(),
        joint.nodes.iter().map(|&id| name(id)).collect::<Vec<_>>()
    );
    println!("selection rationale:");
    for choice in &joint.choices {
        println!("  - {}", choice.rationale);
    }
    println!(
        "topological execution order: {:?}",
        joint
            .execution_order
            .iter()
            .map(|&id| name(id))
            .collect::<Vec<_>>()
    );

    // Independent plans, one target at a time.
    let plan_one = |target: usize| match problem.plan(&[target]) {
        PlanOutcome::Planned(plan) => plan,
        other => panic!("expected a plan, got {other:?}"),
    };
    let app_only = plan_one(app);
    let worker_only = plan_one(worker);
    let independent_total = app_only.size() + worker_only.size();

    println!("\n== independent plans ==");
    println!("app alone:    {} nodes", app_only.size());
    println!("worker alone: {} nodes", worker_only.size());
    println!("independent sum: {independent_total} nodes");
    println!(
        "\nshared plan saves {} node(s) vs independent planning",
        independent_total - joint.size()
    );

    assert!(joint.size() < independent_total);
    assert_eq!(joint.size(), 5);
    assert_eq!(problem.node_count(), 6);
}
