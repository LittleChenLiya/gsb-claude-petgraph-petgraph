use petgraph::{
    Graph,
    algo::{ParetoError, pareto_shortest_paths},
    graph::{EdgeIndex, NodeIndex},
    stable_graph::StableGraph,
    visit::EdgeRef,
};

/// Graph with three s->t paths labelled (2, 8), (5, 3) and (6, 9).
fn three_path_graph() -> (Graph<(), (u64, u64)>, NodeIndex, NodeIndex) {
    let mut g: Graph<(), (u64, u64)> = Graph::new();
    let s = g.add_node(());
    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());
    let t = g.add_node(());
    g.extend_with_edges([
        (s, a, (1, 4)),
        (a, t, (1, 4)), // s -> a -> t: (2, 8)
        (s, b, (3, 2)),
        (b, t, (2, 1)), // s -> b -> t: (5, 3)
        (s, c, (3, 5)),
        (c, t, (3, 4)), // s -> c -> t: (6, 9), dominated by (5, 3)
    ]);
    (g, s, t)
}

fn labels(frontier: &petgraph::algo::ParetoFrontier<NodeIndex, EdgeIndex, u64>) -> Vec<(u64, u64)> {
    frontier
        .paths
        .iter()
        .map(|p| (p.cost, p.resource))
        .collect()
}

#[test]
fn three_paths_budget_10() {
    let (g, s, t) = three_path_graph();
    let frontier = pareto_shortest_paths(&g, s, t, 10, 1024, |e| *e.weight()).unwrap();
    assert_eq!(labels(&frontier), vec![(2, 8), (5, 3)]);
}

#[test]
fn three_paths_budget_5() {
    let (g, s, t) = three_path_graph();
    let frontier = pareto_shortest_paths(&g, s, t, 5, 1024, |e| *e.weight()).unwrap();
    assert_eq!(labels(&frontier), vec![(5, 3)]);
}

#[test]
fn budget_boundary_is_inclusive() {
    let (g, s, t) = three_path_graph();
    // Resource exactly on the budget is kept.
    let frontier = pareto_shortest_paths(&g, s, t, 8, 1024, |e| *e.weight()).unwrap();
    assert_eq!(labels(&frontier), vec![(2, 8), (5, 3)]);
    // One below the budget of the (2, 8) path drops it.
    let frontier = pareto_shortest_paths(&g, s, t, 7, 1024, |e| *e.weight()).unwrap();
    assert_eq!(labels(&frontier), vec![(5, 3)]);
}

#[test]
fn zero_weight_cycle_terminates() {
    let (mut g, s, t) = three_path_graph();
    let a = NodeIndex::new(1);
    // Zero-weight cycle s -> a -> z -> a reachable from the source.
    let z = g.add_node(());
    g.add_edge(a, z, (0, 0));
    g.add_edge(z, a, (0, 0));
    let frontier = pareto_shortest_paths(&g, s, t, 10, 1024, |e| *e.weight()).unwrap();
    assert_eq!(labels(&frontier), vec![(2, 8), (5, 3)]);
}

#[test]
fn label_limit_returns_dedicated_error() {
    let (g, s, t) = three_path_graph();
    let result = pareto_shortest_paths(&g, s, t, 10, 2, |e| *e.weight());
    match result {
        Err(ParetoError::LabelLimitExceeded(stats)) => {
            assert_eq!(stats.max_labels, 2);
            assert_eq!(stats.labels_created, 3);
        }
        other => panic!("expected LabelLimitExceeded, got {:?}", other),
    }
}

#[test]
fn missing_source_is_an_error() {
    let (g, _s, t) = three_path_graph();
    let ghost = NodeIndex::new(999);
    let result = pareto_shortest_paths(&g, ghost, t, 10, 1024, |e| *e.weight());
    assert_eq!(result, Err(ParetoError::SourceNotFound(ghost)));
}

#[test]
fn source_equals_target() {
    let (g, s, _t) = three_path_graph();
    let frontier = pareto_shortest_paths(&g, s, s, 0, 1024, |e| *e.weight()).unwrap();
    assert_eq!(frontier.paths.len(), 1);
    assert_eq!((frontier.paths[0].cost, frontier.paths[0].resource), (0, 0));
    assert!(frontier.paths[0].edges.is_empty());
}

#[test]
fn integer_overflow_is_an_error() {
    let mut g: Graph<(), (u64, u64)> = Graph::new();
    let s = g.add_node(());
    let a = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, a, (u64::MAX, 0));
    g.add_edge(a, t, (1, 0));
    let result = pareto_shortest_paths(&g, s, t, u64::MAX, 1024, |e| *e.weight());
    assert_eq!(result, Err(ParetoError::Overflow));

    let mut g: Graph<(), (u64, u64)> = Graph::new();
    let s = g.add_node(());
    let a = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, a, (0, u64::MAX));
    g.add_edge(a, t, (0, 1));
    let result = pareto_shortest_paths(&g, s, t, u64::MAX, 1024, |e| *e.weight());
    assert_eq!(result, Err(ParetoError::Overflow));
}

#[test]
fn parallel_edges_keep_edge_identity() {
    let mut g: Graph<(), (u64, u64)> = Graph::new();
    let s = g.add_node(());
    let t = g.add_node(());
    let e1 = g.add_edge(s, t, (1, 5));
    let e2 = g.add_edge(s, t, (2, 2));
    let _e3 = g.add_edge(s, t, (3, 6)); // dominated by both
    let frontier = pareto_shortest_paths(&g, s, t, 10, 1024, |e| *e.weight()).unwrap();
    assert_eq!(frontier.paths.len(), 2);
    assert_eq!(labels(&frontier), vec![(1, 5), (2, 2)]);
    // The two labels are only distinguishable through their edge identities.
    assert_eq!(frontier.paths[0].edges, vec![e1]);
    assert_eq!(frontier.paths[1].edges, vec![e2]);
}

#[test]
fn stable_graph_with_index_holes() {
    let mut g = StableGraph::<(), (u64, u64)>::new();
    let s = g.add_node(());
    let a = g.add_node(());
    let b = g.add_node(());
    let x = g.add_node(());
    let t = g.add_node(());
    // Holes in both node and edge indices.
    g.remove_node(x);
    let tmp = g.add_edge(a, b, (9, 9));
    g.remove_edge(tmp);
    g.add_edge(s, a, (1, 1));
    g.add_edge(a, t, (2, 2)); // s -> a -> t: (3, 3)
    g.add_edge(s, b, (1, 4));
    g.add_edge(b, t, (1, 2)); // s -> b -> t: (2, 6)
    let frontier = pareto_shortest_paths(&g, s, t, 10, 1024, |e| *e.weight()).unwrap();
    assert_eq!(labels(&frontier), vec![(2, 6), (3, 3)]);
}

/// Enumerate all simple s->t paths and compute the non-dominated frontier
/// directly; used as an independent oracle. For equal labels the
/// lexicographically smallest edge index sequence is kept.
fn oracle_frontier(
    g: &Graph<(), (u64, u64)>,
    source: NodeIndex,
    target: NodeIndex,
    budget: u64,
) -> Vec<(u64, u64, Vec<EdgeIndex>)> {
    let mut all: Vec<(u64, u64, Vec<EdgeIndex>)> = Vec::new();
    let mut stack = vec![(source, 0u64, 0u64, Vec::<EdgeIndex>::new(), vec![source])];
    while let Some((node, cost, resource, edges, nodes)) = stack.pop() {
        if node == target {
            if resource <= budget {
                all.push((cost, resource, edges));
            }
            continue;
        }
        for edge in g.edges(node) {
            let (edge_cost, edge_resource) = *edge.weight();
            let next = edge.target();
            if nodes.contains(&next) {
                continue;
            }
            let mut next_edges = edges.clone();
            next_edges.push(edge.id());
            let mut next_nodes = nodes.clone();
            next_nodes.push(next);
            stack.push((
                next,
                cost + edge_cost,
                resource + edge_resource,
                next_edges,
                next_nodes,
            ));
        }
    }

    let mut frontier: Vec<(u64, u64, Vec<EdgeIndex>)> = Vec::new();
    'candidates: for (i, p) in all.iter().enumerate() {
        for (j, q) in all.iter().enumerate() {
            if i != j && q.0 <= p.0 && q.1 <= p.1 && (q.0 < p.0 || q.1 < p.1) {
                continue 'candidates;
            }
        }
        match frontier.iter_mut().find(|f| f.0 == p.0 && f.1 == p.1) {
            Some(existing) => {
                if p.2 < existing.2 {
                    *existing = p.clone();
                }
            }
            None => frontier.push(p.clone()),
        }
    }
    frontier.sort_by_key(|p| (p.0, p.1));
    frontier
}

#[test]
fn matches_simple_path_enumeration_oracle() {
    let mut g: Graph<(), (u64, u64)> = Graph::new();
    let n: Vec<_> = (0..6).map(|_| g.add_node(())).collect();
    let (s, t) = (n[0], n[5]);
    g.extend_with_edges([
        (n[0], n[1], (2, 3)),
        (n[0], n[2], (1, 5)),
        (n[0], n[3], (8, 1)),
        (n[1], n[2], (1, 1)),
        (n[1], n[3], (4, 2)),
        (n[1], n[4], (5, 5)),
        (n[2], n[3], (2, 2)),
        (n[2], n[4], (3, 1)),
        (n[3], n[4], (1, 4)),
        (n[3], n[5], (2, 3)),
        (n[4], n[5], (2, 2)),
        (n[4], n[1], (1, 1)), // cycle: 1 -> 4 -> 1
    ]);

    for budget in [0, 3, 7, 14, 100] {
        let frontier = pareto_shortest_paths(&g, s, t, budget, 1_000_000, |e| *e.weight())
            .unwrap_or_else(|err| panic!("budget {}: {:?}", budget, err));
        let got: Vec<(u64, u64, Vec<EdgeIndex>)> = frontier
            .paths
            .iter()
            .map(|p| (p.cost, p.resource, p.edges.clone()))
            .collect();
        assert_eq!(got, oracle_frontier(&g, s, t, budget), "budget {}", budget);
    }
}
