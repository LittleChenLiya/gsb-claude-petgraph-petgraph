//! Tests for the dependency closure planner.
//!
//! These tests cover the acceptance criteria:
//! * sharing beats independent optimization (six-node example),
//! * the planner matches brute-force enumeration on small graphs,
//! * cyclic branches are rejected as non-executable,
//! * budget exhaustion yields `Unknown` (distinct from `NoSolution`),
//! * the global optimum beats greedy per-target optimization.

use hashbrown::{HashMap, HashSet};
use petgraph::algo::{PlannerResult, plan_dependency_closure};

/// Builds the six-node example graph.
///
/// ```text
///   a       b
///  / \     / \
/// c   d   d   e
///  \ /     \ /
///   f       (none)
/// ```
///
/// Targets: `a` and `b`.
/// * `a` can be satisfied by `c` or `d`.
/// * `b` can be satisfied by `d` or `e`.
/// * `c` and `d` each require `f`.
/// * `e` and `f` have no dependencies.
///
/// The shared solution (`a -> d`, `b -> d`) requires only `{a, b, d, f}`
/// (4 nodes), while independent optimization picks `c` for `a` and `e` for
/// `b`, requiring 5 nodes.
fn six_node_graph() -> HashMap<&'static str, Vec<Vec<&'static str>>> {
    let mut groups: HashMap<&str, Vec<Vec<&str>>> = HashMap::new();
    groups.insert("a", vec![vec!["c", "d"]]);
    groups.insert("b", vec![vec!["d", "e"]]);
    groups.insert("c", vec![vec!["f"]]);
    groups.insert("d", vec![vec!["f"]]);
    groups.insert("e", vec![]);
    groups.insert("f", vec![]);
    groups
}

#[test]
fn six_node_sharing_beats_independent() {
    let groups = six_node_graph();
    let result = plan_dependency_closure(&["a", "b"], &groups, 100);

    let plan = match result {
        PlannerResult::Solution(p) => p,
        other => panic!("expected solution, got {:?}", other),
    };

    // The shared solution requires exactly 4 nodes: {a, b, d, f}.
    assert_eq!(plan.required.len(), 4, "required set: {:?}", plan.required);
    let required: HashSet<&&str> = plan.required.iter().collect();
    assert!(required.contains(&"a"));
    assert!(required.contains(&"b"));
    assert!(required.contains(&"d"));
    assert!(required.contains(&"f"));
    assert!(!required.contains(&"c"));
    assert!(!required.contains(&"e"));

    // Both targets choose `d` (the shared predecessor).
    let a_choice = plan
        .choices
        .iter()
        .find(|c| c.node == "a" && c.group_index == 0)
        .expect("a has a choice");
    assert_eq!(a_choice.chosen, "d");
    let b_choice = plan
        .choices
        .iter()
        .find(|c| c.node == "b" && c.group_index == 0)
        .expect("b has a choice");
    assert_eq!(b_choice.chosen, "d");

    // The build order has dependencies first: f before d, d before a and b.
    let pos = |n: &str| plan.build_order.iter().position(|&x| x == n).unwrap();
    assert!(pos("f") < pos("d"), "f must come before d");
    assert!(pos("d") < pos("a"), "d must come before a");
    assert!(pos("d") < pos("b"), "d must come before b");
}

#[test]
fn six_node_matches_enumeration() {
    let groups = six_node_graph();
    let result = plan_dependency_closure(&["a", "b"], &groups, 100);
    let plan = match result {
        PlannerResult::Solution(p) => p,
        other => panic!("expected solution, got {:?}", other),
    };

    let brute = brute_force_minimum(&["a", "b"], &groups);
    assert_eq!(brute, Some(plan.required.len()));
    assert_eq!(brute, Some(4));
}

#[test]
fn greedy_per_target_fails_but_global_optimal_succeeds() {
    // The six-node example: greedy per-target picks `c` for `a` and `e` for
    // `b` (5 nodes), but the global optimum picks `d` for both (4 nodes).
    let groups = six_node_graph();
    let result = plan_dependency_closure(&["a", "b"], &groups, 100);
    let plan = match result {
        PlannerResult::Solution(p) => p,
        other => panic!("expected solution, got {:?}", other),
    };

    // Greedy per-target: a picks c (closure size 2) or d (closure size 2),
    // b picks e (closure size 1). Independent union = 5.
    let greedy_size = greedy_per_target(&["a", "b"], &groups);
    assert_eq!(greedy_size, 5, "greedy per-target should need 5 nodes");

    // Global optimum: 4 nodes (sharing d and f).
    assert_eq!(plan.required.len(), 4);
    assert!(plan.required.len() < greedy_size);
}

#[test]
fn cyclic_branch_is_no_solution() {
    // a -> b -> a: a cycle. No valid execution plan exists.
    let mut groups: HashMap<&str, Vec<Vec<&str>>> = HashMap::new();
    groups.insert("a", vec![vec!["b"]]);
    groups.insert("b", vec![vec!["a"]]);

    let result = plan_dependency_closure(&["a"], &groups, 100);
    assert_eq!(result, PlannerResult::NoSolution);
}

#[test]
fn cyclic_branch_rejected_but_alternative_accepted() {
    // a can be satisfied by b (cyclic: b -> a) or c (acyclic).
    // The planner should reject the cyclic branch and pick c.
    let mut groups: HashMap<&str, Vec<Vec<&str>>> = HashMap::new();
    groups.insert("a", vec![vec!["b", "c"]]);
    groups.insert("b", vec![vec!["a"]]); // cyclic
    groups.insert("c", vec![]); // acyclic

    let result = plan_dependency_closure(&["a"], &groups, 100);
    let plan = match result {
        PlannerResult::Solution(p) => p,
        other => panic!("expected solution, got {:?}", other),
    };
    assert_eq!(plan.required.len(), 2); // {a, c}
    let a_choice = plan
        .choices
        .iter()
        .find(|c| c.node == "a")
        .expect("a has a choice");
    assert_eq!(a_choice.chosen, "c");
}

#[test]
fn budget_exceeded_is_unknown_not_no_solution() {
    // A graph with many independent choices (2^10 = 1024 combinations).
    // A small budget (5) cannot reach a complete solution, so the result is
    // `Unknown` (distinct from `NoSolution`).
    let mut groups: HashMap<&str, Vec<Vec<&str>>> = HashMap::new();
    let target_names: [&str; 10] = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"];
    let targets: Vec<&str> = target_names.to_vec();
    for (i, t) in targets.iter().enumerate() {
        let b = Box::leak(format!("b{}", i).into_boxed_str());
        let c = Box::leak(format!("c{}", i).into_boxed_str());
        groups.insert(*t, vec![vec![b, c]]);
        groups.insert(b, vec![]);
        groups.insert(c, vec![]);
    }

    let result = plan_dependency_closure(&targets, &groups, 5);
    assert_eq!(result, PlannerResult::Unknown);

    // With a large enough budget, the solution is found.
    let result_large = plan_dependency_closure(&targets, &groups, 10_000);
    match result_large {
        PlannerResult::Solution(plan) => assert_eq!(plan.required.len(), 20),
        other => panic!("expected solution with large budget, got {:?}", other),
    }
}

#[test]
fn empty_targets_is_no_solution() {
    let groups: HashMap<&str, Vec<Vec<&str>>> = HashMap::new();
    let result = plan_dependency_closure(&[], &groups, 100);
    assert_eq!(result, PlannerResult::NoSolution);
}

#[test]
fn single_target_no_deps() {
    let mut groups: HashMap<&str, Vec<Vec<&str>>> = HashMap::new();
    groups.insert("a", vec![]);
    let result = plan_dependency_closure(&["a"], &groups, 100);
    let plan = match result {
        PlannerResult::Solution(p) => p,
        other => panic!("expected solution, got {:?}", other),
    };
    assert_eq!(plan.required.len(), 1);
    assert_eq!(plan.build_order, vec!["a"]);
}

// --- Test helpers ---

/// Brute-force enumeration of all choice combinations.
///
/// This is a separate implementation from the planner for cross-validation.
/// It enumerates all combinations of choices for all nodes, builds the
/// required set (the closure of the targets), checks for cycles, and returns
/// the minimum required set size.
fn brute_force_minimum(targets: &[&str], groups: &HashMap<&str, Vec<Vec<&str>>>) -> Option<usize> {
    // Enumerate all combinations of choices (Cartesian product of all groups).
    let nodes: Vec<&&str> = groups.keys().collect();
    let mut current: Vec<HashMap<&str, Vec<&str>>> = vec![HashMap::new()];
    for &node in &nodes {
        let node_groups = &groups[*node];
        if node_groups.is_empty() {
            for combo in current.iter_mut() {
                combo.insert(*node, Vec::new());
            }
            continue;
        }
        let mut next = Vec::new();
        for combo in &current {
            for choice in cartesian_product(node_groups) {
                let mut c = combo.clone();
                c.insert(*node, choice);
                next.push(c);
            }
        }
        current = next;
    }

    let mut min_size: Option<usize> = None;
    for choices in &current {
        // Build the required set (closure of targets).
        let mut required: HashSet<&str> = targets.iter().copied().collect();
        let mut stack: Vec<&str> = targets.to_vec();
        while let Some(n) = stack.pop() {
            if let Some(preds) = choices.get(n) {
                for &p in preds {
                    if required.insert(p) {
                        stack.push(p);
                    }
                }
            }
        }

        // Check acyclic.
        if is_acyclic(&required, choices) {
            let size = required.len();
            if min_size.is_none_or(|m| size < m) {
                min_size = Some(size);
            }
        }
    }
    min_size
}

/// Cartesian product of a list of groups (each group is a list of choices).
fn cartesian_product<'a>(groups: &[Vec<&'a str>]) -> Vec<Vec<&'a str>> {
    let mut result = vec![Vec::new()];
    for group in groups {
        let mut next = Vec::new();
        for combo in &result {
            for &item in group {
                let mut c = combo.clone();
                c.push(item);
                next.push(c);
            }
        }
        result = next;
    }
    result
}

/// Returns `true` if the chosen dependency graph is acyclic.
fn is_acyclic(required: &HashSet<&str>, choices: &HashMap<&str, Vec<&str>>) -> bool {
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for &n in required {
        in_degree.insert(n, 0);
    }
    for (&x, preds) in choices {
        if !required.contains(x) {
            continue;
        }
        for &y in preds {
            adj.entry(y).or_insert_with(Vec::new).push(x);
            *in_degree.entry(x).or_insert(0) += 1;
        }
    }
    let mut queue: Vec<&str> = in_degree
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&n, _)| n)
        .collect();
    let mut count = 0;
    while let Some(n) = queue.pop() {
        count += 1;
        if let Some(deps) = adj.get(n) {
            for &d in deps {
                let deg = in_degree.get_mut(d).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(d);
                }
            }
        }
    }
    count == required.len()
}

/// Greedy per-target optimization: for each target, pick the predecessor with
/// the smallest closure size (counting only that target's closure).
fn greedy_per_target(targets: &[&str], groups: &HashMap<&str, Vec<Vec<&str>>>) -> usize {
    let mut total: HashSet<&str> = HashSet::new();
    for &t in targets {
        let closure = best_closure(t, groups);
        total.extend(closure);
    }
    total.len()
}

/// Returns the closure of the predecessor with the smallest closure size.
fn best_closure<'a>(
    node: &'a str,
    groups: &HashMap<&'a str, Vec<Vec<&'a str>>>,
) -> HashSet<&'a str> {
    let node_groups = groups.get(node).map(|g| g.as_slice()).unwrap_or(&[]);
    if node_groups.is_empty() {
        let mut s = HashSet::new();
        s.insert(node);
        return s;
    }
    // For each group, pick the predecessor with the smallest closure.
    let mut result: HashSet<&str> = HashSet::new();
    result.insert(node);
    for group in node_groups {
        let mut best: Option<HashSet<&str>> = None;
        for &pred in group {
            let closure = best_closure(pred, groups);
            if best.as_ref().is_none_or(|b| closure.len() < b.len()) {
                best = Some(closure);
            }
        }
        result.extend(best.unwrap());
    }
    result
}
