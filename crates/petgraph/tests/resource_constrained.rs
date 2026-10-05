//! Tests for `petgraph::algo::resource_constrained_shortest_path`.
//!
//! Run with:
//!     cargo test -p petgraph --test resource_constrained
//!     cargo test -p petgraph --doc resource_constrained
//!
//! State limit: the resource budget is a hard cap -- candidate states whose
//! resource consumption exceeds the budget are never created, so the search
//! space is bounded by `budget + 1` attainable resource values per node.

use petgraph::{
    Graph,
    algo::{ResourceConstrainedError, resource_constrained_shortest_path},
    prelude::*,
};

type Metrics = (u64, u64);

/// Build the acceptance-scenario graph: a cheap but resource-hungry prefix
/// and an expensive but frugal prefix between `s` and `t`.
///
/// ```text
///        (1, 10)     (1, 10)
///     s -------> a --------> t
///     |                       ^
///     | (5, 1)      (5, 1)    |
///     +-------> b ------------+
/// ```
fn two_prefix_graph() -> (
    Graph<(), Metrics>,
    NodeIndex,
    NodeIndex,
    NodeIndex,
    NodeIndex,
) {
    let mut g: Graph<(), Metrics> = Graph::new();
    let s = g.add_node(());
    let a = g.add_node(());
    let b = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, a, (1, 10));
    g.add_edge(a, t, (1, 10));
    g.add_edge(s, b, (5, 1));
    g.add_edge(b, t, (5, 1));
    (g, s, a, b, t)
}

#[test]
fn cheap_heavy_prefix_does_not_eliminate_expensive_frugal_prefix() {
    let (g, s, _a, b, t) = two_prefix_graph();

    // Budget 20: both routes feasible, the cheap heavy one (cost 2) wins.
    let path = resource_constrained_shortest_path(&g, s, t, 20, |e| *e.weight())
        .unwrap()
        .unwrap();
    assert_eq!(path.cost, 2);
    assert_eq!(path.resource, 20);

    // Budget 11: the cheap prefix (resource 20) is infeasible. Only the
    // expensive frugal prefix reaches `t`; a per-node "cheapest only" search
    // would have discarded it and reported no route.
    let path = resource_constrained_shortest_path(&g, s, t, 11, |e| *e.weight())
        .unwrap()
        .unwrap();
    assert_eq!(path.cost, 10);
    assert_eq!(path.resource, 2);
    assert_eq!(path.nodes, vec![s, b, t]);
}

#[test]
fn reduced_budget_means_no_route() {
    let (g, s, _a, _b, t) = two_prefix_graph();

    // Budget 1: neither route fits, so there is explicitly no route.
    assert!(
        resource_constrained_shortest_path(&g, s, t, 1, |e| *e.weight())
            .unwrap()
            .is_none()
    );
    // Budget 0 likewise.
    assert!(
        resource_constrained_shortest_path(&g, s, t, 0, |e| *e.weight())
            .unwrap()
            .is_none()
    );
}

#[test]
fn dominance_keeps_all_non_dominated_states() {
    // Three prefixes into `x` with mutually non-dominated states
    // (cost, resource) = (1, 6), (3, 3), (5, 1), plus a dominated one (4, 4)
    // which is dominated by (3, 3) and must be pruned.
    let mut g: Graph<(), Metrics> = Graph::new();
    let s = g.add_node(());
    let p1 = g.add_node(());
    let p2 = g.add_node(());
    let p3 = g.add_node(());
    let p4 = g.add_node(());
    let x = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, p1, (1, 6));
    g.add_edge(p1, x, (0, 0));
    g.add_edge(s, p2, (3, 3));
    g.add_edge(p2, x, (0, 0));
    g.add_edge(s, p3, (5, 1));
    g.add_edge(p3, x, (0, 0));
    g.add_edge(s, p4, (4, 4));
    g.add_edge(p4, x, (0, 0));
    g.add_edge(x, t, (1, 4));

    // Budget 8: feasible arrivals at `t` are (4, 7) via p2 and (6, 5) via p3;
    // (2, 10) via p1 is over budget and (5, 8) via p4 is dominated by (4, 7).
    let path = resource_constrained_shortest_path(&g, s, t, 8, |e| *e.weight())
        .unwrap()
        .unwrap();
    assert_eq!((path.cost, path.resource), (4, 7));
    assert_eq!(path.nodes, vec![s, p2, x, t]);

    // Budget 6: only (6, 5) via p3 survives -- the cheapest-at-`x` state
    // (1, 6) is a dead end and must not eliminate it.
    let path = resource_constrained_shortest_path(&g, s, t, 6, |e| *e.weight())
        .unwrap()
        .unwrap();
    assert_eq!((path.cost, path.resource), (6, 5));
    assert_eq!(path.nodes, vec![s, p3, x, t]);

    // Budget 4: no route fits any more.
    assert!(
        resource_constrained_shortest_path(&g, s, t, 4, |e| *e.weight())
            .unwrap()
            .is_none()
    );
}

#[test]
fn parallel_edges_preserve_edge_ids_and_report_real_metrics() {
    // Two parallel edges s -> t with different (cost, resource); the algorithm
    // must pick the cheaper feasible one and report its actual EdgeId.
    let mut g: Graph<(), Metrics> = Graph::new();
    let s = g.add_node(());
    let t = g.add_node(());
    let e_expensive = g.add_edge(s, t, (9, 1));
    let e_cheap = g.add_edge(s, t, (3, 2));

    let path = resource_constrained_shortest_path(&g, s, t, 5, |e| *e.weight())
        .unwrap()
        .unwrap();
    assert_eq!(path.edges, vec![e_cheap]);
    assert_eq!(path.cost, 3);
    assert_eq!(path.resource, 2);
    assert_eq!(path.nodes, vec![s, t]);

    // Tighten the budget below the resource of the cheap edge: now the
    // expensive edge (resource 1) is the only feasible one.
    let path = resource_constrained_shortest_path(&g, s, t, 1, |e| *e.weight())
        .unwrap()
        .unwrap();
    assert_eq!(path.edges, vec![e_expensive]);
    assert_eq!((path.cost, path.resource), (9, 1));
}

#[test]
fn start_equals_goal_is_empty_path() {
    let mut g: Graph<(), Metrics> = Graph::new();
    let s = g.add_node(());
    let path = resource_constrained_shortest_path(&g, s, s, 0, |e| *e.weight())
        .unwrap()
        .unwrap();
    assert_eq!((path.cost, path.resource), (0, 0));
    assert_eq!(path.nodes, vec![s]);
    assert!(path.edges.is_empty());
}

#[test]
fn negative_cost_is_rejected() {
    let mut g: Graph<(), (i64, u64)> = Graph::new();
    let s = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, t, (-1, 0));
    assert_eq!(
        resource_constrained_shortest_path(&g, s, t, 10, |e| *e.weight()),
        Err(ResourceConstrainedError::NegativeCost)
    );
}

#[test]
fn negative_resource_is_rejected() {
    let mut g: Graph<(), (u64, i64)> = Graph::new();
    let s = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, t, (1, -5));
    assert_eq!(
        resource_constrained_shortest_path(&g, s, t, 10, |e| *e.weight()),
        Err(ResourceConstrainedError::NegativeResource)
    );
}

#[test]
fn cost_overflow_is_rejected() {
    let mut g: Graph<(), Metrics> = Graph::new();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, m, (u64::MAX, 0));
    g.add_edge(m, t, (1, 0));
    assert_eq!(
        resource_constrained_shortest_path(&g, s, t, 10, |e| *e.weight()),
        Err(ResourceConstrainedError::CostOverflow)
    );
}

#[test]
fn resource_overflow_is_rejected() {
    let mut g: Graph<(), Metrics> = Graph::new();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, m, (0, u64::MAX));
    g.add_edge(m, t, (0, 1));
    assert_eq!(
        resource_constrained_shortest_path(&g, s, t, u64::MAX, |e| *e.weight()),
        Err(ResourceConstrainedError::ResourceOverflow)
    );
}

#[test]
fn graph_is_not_modified() {
    let (g, s, _a, _b, t) = two_prefix_graph();
    let (nodes_before, edges_before) = (g.node_count(), g.edge_count());
    let _ = resource_constrained_shortest_path(&g, s, t, 11, |e| *e.weight());
    assert_eq!(g.node_count(), nodes_before);
    assert_eq!(g.edge_count(), edges_before);
}

// ---------------------------------------------------------------------------
// Brute-force reference: enumerate all simple paths and take the cheapest
// one within the budget.
// ---------------------------------------------------------------------------

fn brute_force_min_cost(
    g: &Graph<(), Metrics>,
    start: NodeIndex,
    goal: NodeIndex,
    budget: u64,
) -> Option<u64> {
    #[allow(clippy::too_many_arguments)]
    fn dfs(
        g: &Graph<(), Metrics>,
        node: NodeIndex,
        goal: NodeIndex,
        budget: u64,
        visited: &mut [bool],
        cost: u64,
        resource: u64,
        best: &mut Option<u64>,
    ) {
        if node == goal {
            *best = Some(best.map_or(cost, |b| b.min(cost)));
            return;
        }
        for edge in g.edges(node) {
            let &(c, r) = edge.weight();
            let next_resource = resource + r;
            if next_resource > budget {
                continue;
            }
            let next_cost = cost + c;
            if best.is_some_and(|b| next_cost >= b) {
                continue;
            }
            let target = edge.target();
            if visited[target.index()] {
                continue;
            }
            visited[target.index()] = true;
            dfs(
                g,
                target,
                goal,
                budget,
                visited,
                next_cost,
                next_resource,
                best,
            );
            visited[target.index()] = false;
        }
    }

    let mut visited = vec![false; g.node_count()];
    visited[start.index()] = true;
    let mut best = None;
    dfs(g, start, goal, budget, &mut visited, 0, 0, &mut best);
    best
}

/// Check that a returned path is a real s-t path of the graph whose reported
/// cost and resource match the actual edge weights, and that its cost equals
/// the brute-force optimum.
fn check_path_against_brute_force(
    g: &Graph<(), Metrics>,
    start: NodeIndex,
    goal: NodeIndex,
    budget: u64,
) {
    let result =
        resource_constrained_shortest_path(g, start, goal, budget, |e| *e.weight()).unwrap();
    let expected = brute_force_min_cost(g, start, goal, budget);
    match (result, expected) {
        (None, None) => {}
        (Some(path), Some(best_cost)) => {
            assert_eq!(path.cost, best_cost, "cost must match brute-force optimum");
            assert!(path.resource <= budget, "path must respect the budget");
            assert_eq!(path.nodes.len(), path.edges.len() + 1);
            assert_eq!(path.nodes.first(), Some(&start));
            assert_eq!(path.nodes.last(), Some(&goal));
            // Replay the returned EdgeIds: they must chain from start to goal
            // and sum up to the reported cost and resource consumption.
            let mut cost = 0;
            let mut resource = 0;
            let mut node = start;
            for &edge_id in &path.edges {
                let (from, to) = g.edge_endpoints(edge_id).unwrap();
                assert_eq!(from, node, "edges must form a connected path");
                let &(c, r) = g.edge_weight(edge_id).unwrap();
                cost += c;
                resource += r;
                node = to;
            }
            assert_eq!(node, goal);
            assert_eq!(cost, path.cost, "reported cost must be the real cost");
            assert_eq!(
                resource, path.resource,
                "reported resource must be the real consumption"
            );
        }
        (got, want) => panic!(
            "disagreement with brute force: got {:?}, brute force says {:?}",
            got.map(|p| (p.cost, p.resource)),
            want
        ),
    }
}

/// Exhaustive check over *all* 3-node directed graphs where each of the six
/// possible edges is either absent or present with metrics (1, 2) or (2, 1),
/// for every budget in 0..=4: 3^6 * 5 = 3645 (graph, budget) combinations.
#[test]
fn exhaustive_small_graphs() {
    let choices: [Option<Metrics>; 3] = [None, Some((1, 2)), Some((2, 1))];
    let pairs = [(0u32, 1u32), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];

    for mask in 0..3u32.pow(6) {
        let mut g: Graph<(), Metrics> = Graph::new();
        let nodes: Vec<_> = (0..3).map(|_| g.add_node(())).collect();
        let mut m = mask;
        for &(from, to) in &pairs {
            let choice = choices[(m % 3) as usize];
            m /= 3;
            if let Some(metrics) = choice {
                g.add_edge(nodes[from as usize], nodes[to as usize], metrics);
            }
        }
        for budget in 0..=4u64 {
            check_path_against_brute_force(&g, nodes[0], nodes[2], budget);
        }
    }
}

/// Deterministic pseudo-random multigraphs (with parallel edges and loops)
/// checked against the brute-force reference.
#[test]
fn random_multigraphs_match_brute_force() {
    // xorshift64* PRNG, fixed seed: fully deterministic.
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545F4914F6CDD1D)
    };

    for _case in 0..300 {
        let mut g: Graph<(), Metrics> = Graph::new();
        let nodes: Vec<_> = (0..5).map(|_| g.add_node(())).collect();
        for &from in &nodes {
            for &to in &nodes {
                // 0..=2 parallel edges per ordered pair (including loops).
                for _ in 0..next() % 3 {
                    if next() % 2 == 0 {
                        continue;
                    }
                    let cost = next() % 6;
                    let resource = next() % 6;
                    g.add_edge(from, to, (cost, resource));
                }
            }
        }
        let budget = next() % 13;
        check_path_against_brute_force(&g, nodes[0], nodes[4], budget);
    }
}
