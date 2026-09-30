use core::cell::Cell;

use petgraph::{
    Direction::Outgoing,
    algo::k_shortest_simple_paths,
    graph::{DiGraph, EdgeIndex, NodeIndex},
    visit::EdgeRef,
};

/// Exhaustively enumerate all simple paths from `from` to `to` and return
/// them sorted by total cost, then by edge id sequence. This is the
/// reference implementation the lazy iterator is checked against.
fn brute_force_simple_paths(
    graph: &DiGraph<(), u32>,
    from: NodeIndex,
    to: NodeIndex,
) -> Vec<(u64, Vec<EdgeIndex>)> {
    fn dfs(
        graph: &DiGraph<(), u32>,
        node: NodeIndex,
        to: NodeIndex,
        visited: &mut [bool],
        edges: &mut Vec<EdgeIndex>,
        cost: u64,
        out: &mut Vec<(u64, Vec<EdgeIndex>)>,
    ) {
        if node == to {
            out.push((cost, edges.clone()));
            return;
        }
        for edge in graph.edges_directed(node, Outgoing) {
            let target = edge.target();
            if visited[target.index()] {
                continue;
            }
            visited[target.index()] = true;
            edges.push(edge.id());
            dfs(
                graph,
                target,
                to,
                visited,
                edges,
                cost + u64::from(*edge.weight()),
                out,
            );
            edges.pop();
            visited[target.index()] = false;
        }
    }

    let mut out = Vec::new();
    let mut visited = vec![false; graph.node_count()];
    visited[from.index()] = true;
    dfs(
        graph,
        from,
        to,
        &mut visited,
        &mut Vec::new(),
        0,
        &mut out,
    );
    out.sort();
    out
}

/// A small graph exercising parallel edges, a zero-cost cycle and shared
/// prefixes between paths.
fn small_graph() -> (DiGraph<(), u32>, NodeIndex, NodeIndex) {
    let mut graph = DiGraph::<(), u32>::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let t = graph.add_node(());

    graph.extend_with_edges([
        (s, a, 1),  // e0
        (s, a, 2),  // e1, parallel to e0
        (a, b, 1),  // e2
        (a, b, 1),  // e3, parallel to e2 with the same cost
        (b, t, 5),  // e4
        (a, c, 0),  // e5
        (c, a, 0),  // e6, closes a zero-cost cycle a <-> c
        (c, t, 2),  // e7
        (b, c, 0),  // e8
        (s, t, 10), // e9
    ]);

    (graph, s, t)
}

#[test]
fn matches_brute_force_ordering() {
    let (graph, s, t) = small_graph();

    let expected = brute_force_simple_paths(&graph, s, t);
    // The graph is small enough to enumerate, but has enough paths to make
    // a first-5 comparison meaningful.
    assert!(expected.len() > 5);

    let actual: Vec<(u64, Vec<EdgeIndex>)> = k_shortest_simple_paths(&graph, s, t, |edge| *edge.weight())
        .map(|(cost, edges)| (u64::from(cost), edges))
        .collect();

    // The full sequence matches the sorted exhaustive enumeration ...
    assert_eq!(actual, expected);

    // ... and so do the first 5 paths on their own.
    let first_five: Vec<(u64, Vec<EdgeIndex>)> =
        k_shortest_simple_paths(&graph, s, t, |edge| *edge.weight())
            .take(5)
            .map(|(cost, edges)| (u64::from(cost), edges))
            .collect();
    assert_eq!(first_five, expected[..5].to_vec());

    // Spot-check the cheapest paths: zero-cost detours come first, and the
    // parallel edges e2/e3 tie in cost and are ordered by edge id.
    let e = EdgeIndex::new;
    assert_eq!(
        first_five,
        vec![
            (3, vec![e(0), e(5), e(7)]),       // s-a-c-t
            (4, vec![e(0), e(2), e(8), e(7)]), // s-a-b-c-t via e2
            (4, vec![e(0), e(3), e(8), e(7)]), // s-a-b-c-t via e3
            (4, vec![e(1), e(5), e(7)]),       // s-a-c-t via e1
            (5, vec![e(1), e(2), e(8), e(7)]), // s-a-b-c-t via e1, e2
        ]
    );
}

#[test]
fn ties_are_broken_by_edge_id_sequence() {
    let mut graph = DiGraph::<(), u32>::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let t = graph.add_node(());

    graph.extend_with_edges([
        (s, a, 1), // e0
        (a, t, 1), // e1
        (s, t, 2), // e2, same total cost as s-a-t
        (s, t, 1), // e3, parallel to e2 but cheaper
    ]);

    let paths: Vec<_> = k_shortest_simple_paths(&graph, s, t, |edge| *edge.weight()).collect();

    let e = EdgeIndex::new;
    assert_eq!(
        paths,
        vec![
            (1, vec![e(3)]),
            // Both cost 2: the two-edge sequence [0, 1] sorts before [2].
            (2, vec![e(0), e(1)]),
            (2, vec![e(2)]),
        ]
    );
}

#[test]
fn no_path_yields_nothing() {
    let mut graph = DiGraph::<(), u32>::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    let u = graph.add_node(());
    graph.extend_with_edges([(s, u, 1), (t, u, 1)]);

    let mut iter = k_shortest_simple_paths(&graph, s, t, |edge| *edge.weight());
    assert_eq!(iter.next(), None);
    assert_eq!(iter.count(), 0);
}

#[test]
fn same_start_and_goal_yields_empty_path() {
    let mut graph = DiGraph::<(), u32>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    // A cycle through `a`; revisiting `a` would repeat a node, so the only
    // simple path from `a` to `a` is the empty one.
    graph.extend_with_edges([(a, b, 1), (b, a, 1)]);

    let paths: Vec<_> = k_shortest_simple_paths(&graph, a, a, |edge| *edge.weight()).collect();
    assert_eq!(paths, vec![(0, Vec::new())]);
}

#[test]
fn is_lazy_and_stops_early() {
    // A layered dag with two parallel edges between consecutive layers.
    // It has 4^LAYERS * 2 simple paths from start to goal, so enumerating
    // all of them is infeasible; taking the first few must stay cheap.
    const LAYERS: usize = 40;

    let mut graph = DiGraph::<(), u32>::new();
    let start = graph.add_node(());
    let mut layer = vec![start];
    for _ in 0..LAYERS {
        let x = graph.add_node(());
        let y = graph.add_node(());
        for &prev in &layer {
            graph.extend_with_edges([(prev, x, 0), (prev, x, 0), (prev, y, 0), (prev, y, 0)]);
        }
        layer = vec![x, y];
    }
    let goal = graph.add_node(());
    for &prev in &layer {
        graph.extend_with_edges([(prev, goal, 0)]);
    }

    let cost_calls = Cell::new(0_usize);
    let mut iter = k_shortest_simple_paths(&graph, start, goal, |edge| {
        cost_calls.set(cost_calls.get() + 1);
        *edge.weight()
    });

    let first: Vec<_> = iter.by_ref().take(5).collect();
    assert_eq!(first.len(), 5);
    assert!(
        first
            .iter()
            .all(|(cost, edges)| *cost == 0 && edges.len() == LAYERS + 1)
    );

    // Pulling the first paths explored only a tiny fraction of the graph:
    // full enumeration would require on the order of 4^40 edge visits.
    assert!(cost_calls.get() < 100_000, "{}", cost_calls.get());
}

#[test]
fn overflowing_paths_are_skipped() {
    let mut graph = DiGraph::<(), u8>::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let b = graph.add_node(());
    let t = graph.add_node(());

    graph.extend_with_edges([
        (s, a, 200_u8), // e0
        (a, t, 100),    // e1: 200 + 100 overflows u8
        (s, b, 255),    // e2
        (b, t, 0),      // e3: exactly u8::MAX, no overflow
        (s, t, 250),    // e4
    ]);

    let paths: Vec<_> = k_shortest_simple_paths(&graph, s, t, |edge| *edge.weight()).collect();

    let e = EdgeIndex::new;
    assert_eq!(paths, vec![(250, vec![e(4)]), (255, vec![e(2), e(3)])]);
}

#[test]
fn graph_is_not_modified() {
    let (graph, s, t) = small_graph();
    let snapshot = |graph: &DiGraph<(), u32>| {
        let nodes: Vec<()> = graph.node_weights().copied().collect();
        let edges: Vec<(usize, usize, u32)> = graph
            .edge_references()
            .map(|edge| (edge.source().index(), edge.target().index(), *edge.weight()))
            .collect();
        (nodes, edges)
    };
    let before = snapshot(&graph);

    let paths: Vec<_> = k_shortest_simple_paths(&graph, s, t, |edge| *edge.weight()).collect();
    assert!(!paths.is_empty());

    assert_eq!(snapshot(&graph), before);
}
