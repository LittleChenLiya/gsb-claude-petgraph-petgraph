use std::collections::HashSet;

use petgraph::{Graph, algo::k_edge_disjoint_paths, prelude::*, stable_graph::StableGraph};

/// All simple paths from `start` to `goal`, each as a sequence of edge ids.
fn all_simple_edge_paths(
    graph: &Graph<(), u64>,
    start: NodeIndex,
    goal: NodeIndex,
) -> Vec<Vec<petgraph::graph::EdgeIndex>> {
    fn dfs(
        graph: &Graph<(), u64>,
        node: NodeIndex,
        goal: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        path: &mut Vec<petgraph::graph::EdgeIndex>,
        paths: &mut Vec<Vec<petgraph::graph::EdgeIndex>>,
    ) {
        if node == goal {
            paths.push(path.clone());
            return;
        }
        for edge in graph.edges(node) {
            let target = edge.target();
            if visited.contains(&target) {
                continue;
            }
            visited.insert(target);
            path.push(edge.id());
            dfs(graph, target, goal, visited, path, paths);
            path.pop();
            visited.remove(&target);
        }
    }
    let mut paths = Vec::new();
    let mut visited = HashSet::new();
    visited.insert(start);
    dfs(
        graph,
        start,
        goal,
        &mut visited,
        &mut Vec::new(),
        &mut paths,
    );
    paths
}

/// Optimal total cost of `k` pairwise edge-disjoint simple paths, computed
/// by brute-force enumeration of all legal path combinations.
fn brute_force_optimum(
    graph: &Graph<(), u64>,
    start: NodeIndex,
    goal: NodeIndex,
    k: usize,
) -> Option<u64> {
    let paths = all_simple_edge_paths(graph, start, goal);
    let costs: Vec<u64> = paths
        .iter()
        .map(|path| path.iter().map(|&e| graph[e]).sum())
        .collect();
    let mut best: Option<u64> = None;
    let mut used_edges: HashSet<petgraph::graph::EdgeIndex> = HashSet::new();
    fn search(
        candidates: &[(Vec<petgraph::graph::EdgeIndex>, u64)],
        next: usize,
        remaining: usize,
        used_edges: &mut HashSet<petgraph::graph::EdgeIndex>,
        total: u64,
        best: &mut Option<u64>,
    ) {
        if remaining == 0 {
            if best.is_none_or(|b| total < b) {
                *best = Some(total);
            }
            return;
        }
        for i in next..candidates.len() {
            let (path, cost) = &candidates[i];
            if path.iter().any(|e| used_edges.contains(e)) {
                continue;
            }
            for &e in path {
                used_edges.insert(e);
            }
            search(
                candidates,
                i + 1,
                remaining - 1,
                used_edges,
                total + cost,
                best,
            );
            for &e in path {
                used_edges.remove(&e);
            }
        }
    }
    let candidates: Vec<_> = paths.into_iter().zip(costs).collect();
    search(&candidates, 0, k, &mut used_edges, 0, &mut best);
    best
}

/// Check that the returned paths are legal: each walks from `start` to
/// `goal` without revisiting nodes, no `EdgeId` is used twice, and the
/// reported total cost matches the sum of the used edge weights.
fn verify_paths(
    graph: &Graph<(), u64>,
    start: NodeIndex,
    goal: NodeIndex,
    total_cost: u64,
    paths: &[Vec<petgraph::graph::EdgeIndex>],
) {
    let mut all_edges = HashSet::new();
    let mut recomputed = 0u64;
    for path in paths {
        let mut node = start;
        let mut seen = HashSet::new();
        seen.insert(start);
        for &edge_id in path {
            assert!(
                all_edges.insert(edge_id),
                "edge {edge_id:?} used more than once"
            );
            let (source, target) = graph.edge_endpoints(edge_id).unwrap();
            assert_eq!(source, node, "path is not contiguous");
            assert!(seen.insert(target), "path revisits node {target:?}");
            recomputed += graph[edge_id];
            node = target;
        }
        assert_eq!(node, goal, "path does not end at the goal");
    }
    assert_eq!(recomputed, total_cost, "reported total cost is wrong");
}

#[test]
fn greedy_shortest_path_would_block_the_optimum() {
    // The unique shortest s-t path s -> a -> b -> t (cost 0) uses the
    // "bridge" a -> b and blocks every second path. A greedy algorithm that
    // removes the shortest path cannot find two paths; the optimum reroutes
    // to the two edge-disjoint paths s -> a -> t and s -> b -> t.
    let mut graph: Graph<(), u64> = Graph::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let b = graph.add_node(());
    let t = graph.add_node(());
    graph.extend_with_edges([(s, a, 0), (a, b, 0), (b, t, 0), (s, b, 5), (a, t, 5)]);

    let expected = brute_force_optimum(&graph, s, t, 2);
    assert_eq!(expected, Some(10));

    let (total_cost, paths) = k_edge_disjoint_paths(&graph, s, t, 2, |e| *e.weight()).unwrap();
    assert_eq!(total_cost, 10);
    assert_eq!(Some(total_cost), expected);
    verify_paths(&graph, s, t, total_cost, &paths);
}

#[test]
fn matches_brute_force_on_a_larger_graph() {
    let mut graph: Graph<(), u64> = Graph::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    let t = graph.add_node(());
    graph.extend_with_edges([
        (s, a, 1),
        (s, b, 4),
        (s, c, 2),
        (a, b, 1),
        (a, d, 3),
        (b, c, 1),
        (b, t, 6),
        (c, a, 1),
        (c, d, 1),
        (d, b, 2),
        (d, t, 2),
        (a, t, 8),
    ]);
    for k in 1..=3 {
        let expected = brute_force_optimum(&graph, s, t, k);
        let result = k_edge_disjoint_paths(&graph, s, t, k, |e| *e.weight());
        assert_eq!(result.as_ref().map(|r| r.0), expected, "k = {k}");
        if let Some((total_cost, paths)) = result {
            assert_eq!(paths.len(), k);
            verify_paths(&graph, s, t, total_cost, &paths);
        }
    }
}

#[test]
fn parallel_edges_are_counted_by_edge_id() {
    let mut graph: Graph<(), u64> = Graph::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    let cheap = graph.add_edge(s, t, 3);
    let expensive = graph.add_edge(s, t, 7);

    let (total_cost, paths) = k_edge_disjoint_paths(&graph, s, t, 2, |e| *e.weight()).unwrap();
    assert_eq!(total_cost, 10);
    let used: HashSet<_> = paths.iter().flatten().copied().collect();
    assert_eq!(used.len(), 2);
    assert!(used.contains(&cheap));
    assert!(used.contains(&expensive));
    verify_paths(&graph, s, t, total_cost, &paths);

    // Only two parallel edges exist, so a third path is impossible.
    assert!(k_edge_disjoint_paths(&graph, s, t, 3, |e| *e.weight()).is_none());
}

#[test]
fn zero_cost_edges() {
    let mut graph: Graph<(), u64> = Graph::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let t = graph.add_node(());
    graph.extend_with_edges([(s, a, 0), (a, t, 0), (s, t, 0)]);

    let (total_cost, paths) = k_edge_disjoint_paths(&graph, s, t, 2, |e| *e.weight()).unwrap();
    assert_eq!(total_cost, 0);
    assert_eq!(paths.len(), 2);
    verify_paths(&graph, s, t, total_cost, &paths);
}

#[test]
fn directed_cycles_keep_paths_simple() {
    // Back edges form directed cycles; the optimum must not route flow
    // around them, and every returned path must stay simple.
    let mut graph: Graph<(), u64> = Graph::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let b = graph.add_node(());
    let t = graph.add_node(());
    graph.extend_with_edges([
        (s, a, 1),
        (a, t, 1),
        (s, b, 1),
        (b, t, 1),
        (a, b, 0),
        (b, a, 0),
        (t, s, 0),
    ]);

    let expected = brute_force_optimum(&graph, s, t, 2);
    assert_eq!(expected, Some(4));

    let (total_cost, paths) = k_edge_disjoint_paths(&graph, s, t, 2, |e| *e.weight()).unwrap();
    assert_eq!(Some(total_cost), expected);
    verify_paths(&graph, s, t, total_cost, &paths);
}

#[test]
fn disconnected_graph_has_no_paths() {
    let mut graph: Graph<(), u64> = Graph::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    let u = graph.add_node(());
    graph.add_edge(s, u, 1);

    assert!(k_edge_disjoint_paths(&graph, s, t, 1, |e| *e.weight()).is_none());

    // A single available path cannot satisfy k = 2.
    graph.add_edge(u, t, 1);
    assert!(k_edge_disjoint_paths(&graph, s, t, 1, |e| *e.weight()).is_some());
    assert!(k_edge_disjoint_paths(&graph, s, t, 2, |e| *e.weight()).is_none());
}

#[test]
fn huge_costs_do_not_overflow() {
    let mut graph: Graph<(), u64> = Graph::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let b = graph.add_node(());
    let t = graph.add_node(());
    graph.extend_with_edges([(s, a, u64::MAX), (a, t, 0), (s, b, u64::MAX), (b, t, 1)]);

    // A single path with cost u64::MAX + 1 would overflow, so the optimum
    // picks the cheaper route instead.
    let (total_cost, paths) = k_edge_disjoint_paths(&graph, s, t, 1, |e| *e.weight()).unwrap();
    assert_eq!(total_cost, u64::MAX);
    verify_paths(&graph, s, t, total_cost, &paths);

    // Two paths force a total cost of 2 * u64::MAX + 1, which overflows
    // u64; the function reports this instead of wrapping around.
    assert!(k_edge_disjoint_paths(&graph, s, t, 2, |e| *e.weight()).is_none());
}

#[test]
fn original_graph_is_not_modified() {
    let mut graph: Graph<(), u64> = Graph::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let t = graph.add_node(());
    graph.extend_with_edges([(s, a, 2), (a, t, 2), (s, t, 5)]);
    let before = graph.clone();

    let _ = k_edge_disjoint_paths(&graph, s, t, 2, |e| *e.weight()).unwrap();

    assert_eq!(graph.node_count(), before.node_count());
    assert_eq!(graph.edge_count(), before.edge_count());
    for edge in before.edge_indices() {
        assert_eq!(graph.edge_weight(edge), before.edge_weight(edge));
        assert_eq!(graph.edge_endpoints(edge), before.edge_endpoints(edge));
    }
}

#[test]
fn works_with_stable_graph_and_edge_closure() {
    // The edge cost closure does not have to read the edge weight.
    let mut graph: Graph<(), ()> = Graph::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let t = graph.add_node(());
    graph.extend_with_edges([(s, a), (a, t), (s, t)]);

    let (total_cost, paths) = k_edge_disjoint_paths(&graph, s, t, 2, |_| 1).unwrap();
    assert_eq!(total_cost, 3);
    assert_eq!(paths.len(), 2);
}

#[test]
fn k_zero_and_start_equals_goal() {
    let mut graph: Graph<(), u64> = Graph::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    graph.add_edge(s, t, 1);

    assert_eq!(
        k_edge_disjoint_paths(&graph, s, t, 0, |e| *e.weight()),
        Some((0, Vec::new()))
    );
    let (total_cost, paths) = k_edge_disjoint_paths(&graph, s, s, 3, |e| *e.weight()).unwrap();
    assert_eq!(total_cost, 0);
    assert_eq!(paths.len(), 3);
}

#[test]
fn stable_graph_with_removed_nodes() {
    // Node removal leaves holes in the index space; the algorithm must
    // still handle the sparse indices correctly.
    let mut graph: StableGraph<(), u64> = StableGraph::new();
    let s = graph.add_node(());
    let junk = graph.add_node(());
    let a = graph.add_node(());
    let t = graph.add_node(());
    graph.extend_with_edges([(s, a, 1), (a, t, 1), (s, t, 3), (junk, junk, 9)]);
    graph.remove_node(junk);

    let (total_cost, paths) = k_edge_disjoint_paths(&graph, s, t, 2, |e| *e.weight()).unwrap();
    assert_eq!(total_cost, 5);
    assert_eq!(paths.len(), 2);
}
