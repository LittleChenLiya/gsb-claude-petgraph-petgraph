//! Acceptance tests for the dependency closure planner.
//!
//! ```text
//! cargo test -p dep-closure-planner --test planner
//! ```

use dep_closure_planner::{
    Diagnostic, DiagnosticKind, NodeId, Plan, PlanOutcome, Planner, UnknownReason,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn planned(outcome: PlanOutcome) -> Plan {
    match outcome {
        PlanOutcome::Planned(plan) => plan,
        other => panic!("expected PlanOutcome::Planned, got {other:?}"),
    }
}

/// Structural validation of a plan against its problem.
fn assert_plan_valid(problem: &Planner, targets: &[NodeId], plan: &Plan) {
    // Targets are included; node set is sorted and deduplicated.
    for &target in targets {
        assert!(
            plan.nodes.contains(&target),
            "target {target} missing from plan"
        );
    }
    let mut sorted = plan.nodes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted, plan.nodes, "plan nodes must be sorted and unique");

    // Every group of every selected node is satisfied exactly once, by a
    // provider that is also in the plan.
    for &node in &plan.nodes {
        for (group_index, group) in problem.groups(node).iter().enumerate() {
            let matches: Vec<_> = plan
                .choices
                .iter()
                .filter(|choice| choice.node == node && choice.group == group_index)
                .collect();
            assert_eq!(
                matches.len(),
                1,
                "group '{}' of node {} must be satisfied exactly once",
                group.name(),
                problem.node_name(node)
            );
            let provider = matches[0].provider;
            assert!(
                group.providers().contains(&provider),
                "chosen provider must come from the group"
            );
            assert!(
                plan.nodes.contains(&provider),
                "chosen provider must be in the required set"
            );
        }
    }

    // The execution order is a permutation of the required set and respects
    // chosen edges (providers before dependents).
    let mut order = plan.execution_order.clone();
    order.sort_unstable();
    assert_eq!(order, plan.nodes, "execution order must cover the plan");
    let position = |id: NodeId| {
        plan.execution_order
            .iter()
            .position(|&node| node == id)
            .expect("node in execution order")
    };
    for choice in &plan.choices {
        assert!(
            position(choice.provider) < position(choice.node),
            "provider {} must execute before dependent {}",
            problem.node_name(choice.provider),
            problem.node_name(choice.node)
        );
    }
}

fn has_cycle_diagnostic(diagnostics: &[Diagnostic]) -> bool {
    diagnostics
        .iter()
        .any(|diagnostic| matches!(diagnostic.kind, DiagnosticKind::Cycle { .. }))
}

fn cycle_loop_path(diagnostics: &[Diagnostic]) -> Vec<NodeId> {
    diagnostics
        .iter()
        .find_map(|diagnostic| match &diagnostic.kind {
            DiagnosticKind::Cycle { loop_path } => Some(loop_path.clone()),
            DiagnosticKind::EmptyGroup { .. } => None,
        })
        .expect("cycle diagnostic")
}

// ---------------------------------------------------------------------------
// Acceptance: shared dependencies beat independent planning
// ---------------------------------------------------------------------------

#[test]
fn shared_dependency_beats_independent_plans() {
    let mut planner = Planner::new();
    let t1 = planner.add_node("t1");
    let t2 = planner.add_node("t2");
    let shared = planner.add_node("shared");
    let base = planner.add_node("base");
    let own1 = planner.add_node("own1");
    let own2 = planner.add_node("own2");

    planner.add_group(t1, "backend", &[shared, own1]);
    planner.add_group(t2, "backend", &[shared, own2]);
    planner.add_group(shared, "base", &[base]);
    // The per-target alternatives are as expensive as the shared path, so
    // each target alone costs 3 nodes either way.
    planner.add_group(own1, "base", &[base]);
    planner.add_group(own2, "base", &[base]);

    let joint = planned(planner.plan(&[t1, t2]));
    assert_plan_valid(&planner, &[t1, t2], &joint);

    // Shared closure: {t1, t2, shared, base}; shared dependency counted once.
    assert_eq!(joint.size(), 4);
    assert_eq!(joint.nodes, vec![t1, t2, shared, base]);

    let alone1 = planned(planner.plan(&[t1]));
    let alone2 = planned(planner.plan(&[t2]));
    assert_eq!(alone1.size(), 3);
    assert_eq!(alone2.size(), 3);
    assert!(
        joint.size() < alone1.size() + alone2.size(),
        "shared plan must beat the sum of independent plans"
    );

    // The joint plan reuses the shared node instead of adding new ones.
    assert!(
        joint.choices.iter().any(|choice| choice.reused),
        "expected at least one shared (reused) choice, got {:?}",
        joint.choices
    );
    // Rationales are emitted for every satisfied group.
    assert_eq!(joint.choices.len(), 3);
    assert!(
        joint
            .choices
            .iter()
            .all(|choice| !choice.rationale.is_empty())
    );
}

#[test]
fn greedy_per_target_minimum_is_not_used() {
    // The planner must explore *all* assignments and keep the global minimum,
    // whether or not sharing happens to win.
    let mut planner = Planner::new();
    let t1 = planner.add_node("t1");
    let t2 = planner.add_node("t2");
    let cheap1 = planner.add_node("cheap1"); // leaf, looks optimal for t1 alone
    let cheap2 = planner.add_node("cheap2"); // leaf, looks optimal for t2 alone
    let shared = planner.add_node("shared");
    let base = planner.add_node("base");

    planner.add_group(t1, "dep", &[cheap1, shared]);
    planner.add_group(t2, "dep", &[cheap2, shared]);
    planner.add_group(shared, "base", &[base]);

    let joint = planned(planner.plan(&[t1, t2]));
    assert_plan_valid(&planner, &[t1, t2], &joint);
    // Here the separate leaves tie with the shared path (both 4 nodes).
    assert_eq!(joint.size(), 4);
    assert_eq!(joint.nodes, vec![t1, t2, cheap1, cheap2]);

    // Now make sharing strictly better and check the planner switches.
    let mut planner = Planner::new();
    let t1 = planner.add_node("t1");
    let t2 = planner.add_node("t2");
    let own1 = planner.add_node("own1");
    let own2 = planner.add_node("own2");
    let shared = planner.add_node("shared");
    planner.add_group(t1, "dep", &[own1, shared]);
    planner.add_group(t2, "dep", &[own2, shared]);
    let joint = planned(planner.plan(&[t1, t2]));
    assert_eq!(joint.nodes, vec![t1, t2, shared]);
}

// ---------------------------------------------------------------------------
// Acceptance: cycles are explicit non-executable diagnostics
// ---------------------------------------------------------------------------

#[test]
fn cyclic_branch_is_rejected_and_diagnosed() {
    let mut planner = Planner::new();
    let node_a = planner.add_node("a");
    let node_b = planner.add_node("b");
    let node_c = planner.add_node("c");

    planner.add_group(node_a, "dep", &[node_b, node_c]);
    planner.add_group(node_b, "dep", &[node_a]); // choosing b closes a -> b -> a

    let plan = planned(planner.plan(&[node_a]));
    assert_plan_valid(&planner, &[node_a], &plan);
    // The cyclic branch (b) cannot be a valid satisfaction; c is used.
    assert_eq!(plan.nodes, vec![node_a, node_c]);
    assert!(
        has_cycle_diagnostic(&plan.diagnostics),
        "expected a cycle diagnostic, got {:?}",
        plan.diagnostics
    );
    assert_eq!(
        cycle_loop_path(&plan.diagnostics),
        vec![node_a, node_b, node_a]
    );
    assert!(
        plan.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("cycle"))
    );
}

#[test]
fn pure_cycle_is_unsatisfiable() {
    let mut planner = Planner::new();
    let node_a = planner.add_node("a");
    let node_b = planner.add_node("b");
    planner.add_group(node_a, "dep", &[node_b]);
    planner.add_group(node_b, "dep", &[node_a]);

    match planner.plan(&[node_a]) {
        PlanOutcome::Unsatisfiable { diagnostics } => {
            assert!(has_cycle_diagnostic(&diagnostics));
        }
        other => panic!("expected Unsatisfiable, got {other:?}"),
    }
}

#[test]
fn self_cycle_is_unsatisfiable() {
    let mut planner = Planner::new();
    let node_a = planner.add_node("a");
    planner.add_group(node_a, "dep", &[node_a]);

    match planner.plan(&[node_a]) {
        PlanOutcome::Unsatisfiable { diagnostics } => {
            assert!(has_cycle_diagnostic(&diagnostics));
            assert_eq!(cycle_loop_path(&diagnostics), vec![node_a, node_a]);
        }
        other => panic!("expected Unsatisfiable, got {other:?}"),
    }
}

#[test]
fn empty_group_is_unsatisfiable_with_diagnostic() {
    let mut planner = Planner::new();
    let node_a = planner.add_node("a");
    planner.add_group(node_a, "missing", &[]);

    match planner.plan(&[node_a]) {
        PlanOutcome::Unsatisfiable { diagnostics } => {
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| matches!(diagnostic.kind, DiagnosticKind::EmptyGroup { .. }))
            );
        }
        other => panic!("expected Unsatisfiable, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Acceptance: explicit budget, Unknown is distinct from Unsatisfiable
// ---------------------------------------------------------------------------

/// Full binary "or" tree of depth `depth`: node i needs one of {2i+1, 2i+2}.
fn binary_or_tree(depth: usize) -> (Planner, NodeId) {
    let mut planner = Planner::new();
    let count = (1usize << (depth + 1)) - 1;
    for index in 0..count {
        planner.add_node(format!("n{index}"));
    }
    for index in 0..(count - 1) / 2 {
        planner.add_group(index, "child", &[2 * index + 1, 2 * index + 2]);
    }
    (planner, 0)
}

#[test]
fn budget_exhaustion_returns_unknown_not_unsatisfiable() {
    let (planner, root) = binary_or_tree(12);

    let outcome = planner.plan_with_budget(&[root], 64);
    match &outcome {
        PlanOutcome::Unknown {
            reason,
            best_so_far,
            ..
        } => {
            assert_eq!(*reason, UnknownReason::SearchBudgetExhausted);
            // A best-effort plan may already exist, but it is not proven
            // minimal — hence Unknown, never Unsatisfiable.
            if let Some(plan) = best_so_far {
                assert!(plan.size() >= 13);
            }
        }
        other => panic!("expected Unknown, got {other:?}"),
    }
    // Unknown is a distinct outcome from Unsatisfiable.
    assert!(!matches!(outcome, PlanOutcome::Unsatisfiable { .. }));
    assert!(!outcome.is_planned());

    // With enough budget the same instance is decidable: a chain of 13 nodes.
    let plan = planned(planner.plan_with_budget(&[root], 1_000_000));
    assert_eq!(plan.size(), 13);
    assert_plan_valid(&planner, &[root], &plan);
}

#[test]
fn zero_budget_is_unknown_even_for_trivial_instances() {
    let mut planner = Planner::new();
    let node_a = planner.add_node("a");
    let node_b = planner.add_node("b");
    planner.add_group(node_a, "dep", &[node_b]);

    assert!(matches!(
        planner.plan_with_budget(&[node_a], 0),
        PlanOutcome::Unknown { .. }
    ));
}

// ---------------------------------------------------------------------------
// Acceptance: the input problem is immutable
// ---------------------------------------------------------------------------

#[test]
fn planning_does_not_mutate_the_problem() {
    let mut planner = Planner::new();
    let node_a = planner.add_node("a");
    let node_b = planner.add_node("b");
    let node_c = planner.add_node("c");
    planner.add_group(node_a, "dep", &[node_b, node_c]);
    planner.add_group(node_b, "dep", &[node_a]); // cycle branch

    let before = planner.clone();
    let _ = planner.plan(&[node_a]);
    let _ = planner.plan_with_budget(&[node_a, node_b], 10);
    let _ = planner.plan(&[node_b]);
    assert_eq!(
        before, planner,
        "planning must not mutate the problem graph"
    );
}

#[test]
fn targets_without_requirements_plan_trivially() {
    let mut planner = Planner::new();
    let node_a = planner.add_node("a");
    let node_b = planner.add_node("b");
    let plan = planned(planner.plan(&[node_a, node_b, node_a])); // duplicates collapse
    assert_eq!(plan.nodes, vec![node_a, node_b]);
    assert!(plan.choices.is_empty());
    assert_eq!(plan.execution_order.len(), 2);
}

// ---------------------------------------------------------------------------
// Acceptance: small graphs match brute-force enumeration
// ---------------------------------------------------------------------------

/// Deterministic xorshift64* PRNG so the test is reproducible.
struct Rng(u64);

impl Rng {
    const fn next(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.0 = state;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    const fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn below_usize(&mut self, bound: usize) -> usize {
        usize::try_from(self.below(u64::try_from(bound).expect("usize fits u64")))
            .expect("value below usize bound")
    }
}

/// Exact minimum closure size by enumerating every subset and every provider
/// assignment, independent of the planner's search implementation.
fn brute_force_min(problem: &Planner, targets: &[NodeId]) -> Option<usize> {
    let node_total = problem.node_count();
    let mut base = vec![false; node_total];
    for &target in targets {
        base[target] = true;
    }
    let min_size = base.iter().filter(|&&flag| flag).count();
    for size in min_size..=node_total {
        let mut subset = base.clone();
        if enumerate_subsets(problem, &mut subset, 0, size - min_size) {
            return Some(size);
        }
    }
    None
}

fn enumerate_subsets(
    problem: &Planner,
    subset: &mut [bool],
    start: usize,
    remaining: usize,
) -> bool {
    if remaining == 0 {
        return feasible(problem, subset);
    }
    for index in start..problem.node_count() {
        if subset[index] {
            continue;
        }
        subset[index] = true;
        if enumerate_subsets(problem, subset, index + 1, remaining - 1) {
            return true;
        }
        subset[index] = false;
    }
    false
}

/// Is there an acyclic provider assignment for `subset`?
fn feasible(problem: &Planner, subset: &[bool]) -> bool {
    let mut requirements = Vec::new();
    for node in 0..problem.node_count() {
        if subset[node] {
            for (group_index, group) in problem.groups(node).iter().enumerate() {
                if !group.providers().iter().any(|&provider| subset[provider]) {
                    return false;
                }
                requirements.push((node, group_index));
            }
        }
    }
    let mut deps: Vec<Vec<NodeId>> = vec![Vec::new(); problem.node_count()];
    assign(problem, subset, &requirements, 0, &mut deps)
}

fn assign(
    problem: &Planner,
    subset: &[bool],
    requirements: &[(NodeId, usize)],
    index: usize,
    deps: &mut [Vec<NodeId>],
) -> bool {
    if index == requirements.len() {
        return true;
    }
    let (node, group_index) = requirements[index];
    for &provider in problem.groups(node)[group_index].providers() {
        if !subset[provider] || provider == node || reaches(deps, provider, node) {
            continue; // not in subset, self-cycle, or would close a cycle
        }
        deps[node].push(provider);
        if assign(problem, subset, requirements, index + 1, deps) {
            return true;
        }
        deps[node].pop();
    }
    false
}

fn reaches(deps: &[Vec<NodeId>], from: NodeId, to: NodeId) -> bool {
    from == to || deps[from].iter().any(|&next| reaches(deps, next, to))
}

#[test]
fn matches_brute_force_on_random_small_graphs() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for case in 0..200 {
        let node_total = 2 + rng.below_usize(5); // 2..=6 nodes
        let mut planner = Planner::new();
        for index in 0..node_total {
            planner.add_node(format!("n{index}"));
        }
        for node in 0..node_total {
            let group_total = rng.below_usize(3); // 0..=2 groups
            for group_nr in 0..group_total {
                let arity = 1 + rng.below_usize(3); // 1..=3 providers
                let providers: Vec<NodeId> =
                    (0..arity).map(|_| rng.below_usize(node_total)).collect();
                planner.add_group(node, format!("g{group_nr}"), &providers);
            }
        }
        let target_total = 1 + rng.below_usize(2);
        let mut targets: Vec<NodeId> = Vec::new();
        while targets.len() < target_total {
            let target = rng.below_usize(node_total);
            if !targets.contains(&target) {
                targets.push(target);
            }
        }

        let outcome = planner.plan_with_budget(&targets, 10_000_000);
        let expected = brute_force_min(&planner, &targets);
        match (expected, &outcome) {
            (Some(size), PlanOutcome::Planned(plan)) => {
                assert_eq!(
                    plan.size(),
                    size,
                    "case {case}: planner size differs from brute force\n{planner:#?}"
                );
                assert_plan_valid(&planner, &targets, plan);
            }
            (None, PlanOutcome::Unsatisfiable { .. }) => {}
            (expected, outcome) => panic!(
                "case {case}: brute force = {expected:?}, planner = {outcome:?}\n{planner:#?}"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Acceptance: the six-node example scenario
// ---------------------------------------------------------------------------

#[test]
fn six_node_scenario() {
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

    let joint = planned(planner.plan(&[app, worker]));
    assert_plan_valid(&planner, &[app, worker], &joint);
    assert_eq!(joint.size(), 5);
    // Exactly one database engine is selected, shared by both targets.
    let uses_postgres = joint.nodes.contains(&postgres);
    assert_ne!(uses_postgres, joint.nodes.contains(&sqlite));
    assert!(joint.nodes.contains(&metrics));
    assert!(joint.nodes.contains(&cache));

    let app_only = planned(planner.plan(&[app]));
    let worker_only = planned(planner.plan(&[worker]));
    assert_eq!(app_only.size(), 3);
    assert_eq!(worker_only.size(), 4);
    assert!(joint.size() < app_only.size() + worker_only.size());

    // Both database choices name the same shared provider for both targets.
    let db_choices: Vec<_> = joint
        .choices
        .iter()
        .filter(|choice| choice.group_name == "database")
        .collect();
    assert_eq!(db_choices.len(), 2);
    assert_eq!(db_choices[0].provider, db_choices[1].provider);
    assert!(db_choices.iter().any(|choice| choice.reused));
}
