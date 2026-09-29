use core::cell::Cell;

use petgraph::{algo::simple_shortest_paths, prelude::*};

type TestGraph = DiGraph<u32, u32>;

/// Exhaustively enumerate every simple path from `start` to `goal`, recording
/// `(total cost, edge ids)` pairs, sorted the same way as the lazy iterator:
/// by total cost and then by lexicographic edge identifier sequence.
fn brute_force_simple_paths(
    graph: &TestGraph,
    start: NodeIndex,
    goal: NodeIndex,
) -> Vec<(u32, Vec<EdgeIndex>)> {
    struct BruteForce<'a> {
        graph: &'a TestGraph,
        goal: NodeIndex,
        path_edges: Vec<EdgeIndex>,
        visited: Vec<bool>,
        results: Vec<(u32, Vec<EdgeIndex>)>,
    }

    impl<'a> BruteForce<'a> {
        fn dfs(&mut self, node: NodeIndex, cost: u32) {
            if node == self.goal {
                self.results.push((cost, self.path_edges.clone()));
                return;
            }
            for edge in self.graph.edges(node) {
                let target = edge.target();
                if self.visited[target.index()] {
                    continue;
                }
                self.visited[target.index()] = true;
                self.path_edges.push(edge.id());
                self.dfs(target, cost + edge.weight());
                self.path_edges.pop();
                self.visited[target.index()] = false;
            }
        }
    }

    let mut search = BruteForce {
        graph,
        goal,
        path_edges: Vec::new(),
        visited: vec![false; graph.node_count()],
        results: Vec::new(),
    };
    search.visited[start.index()] = true;
    search.dfs(start, 0);
    search
        .results
        .sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    search.results
}

/// Small directed graph with parallel edges, a zero-cost cycle (1 -> 3 -> 1)
/// and a prefix (0 -> 1) shared by several paths.
fn fixture_graph() -> (TestGraph, NodeIndex, NodeIndex) {
    let mut graph = DiGraph::new();
    let nodes: Vec<_> = (0..6).map(|weight| graph.add_node(weight)).collect();
    let s = nodes[0];
    let t = nodes[5];

    // edge ids are assigned in insertion order
    graph.extend_with_edges([
        (s, nodes[1], 1),        // 0: shared prefix
        (s, nodes[2], 2),        // 1: direct-ish
        (s, t, 3),               // 2: direct edge, cost 3
        (nodes[1], nodes[3], 1), // 3
        (nodes[1], nodes[4], 1), // 4
        (nodes[1], t, 2),        // 5
        (nodes[2], nodes[4], 1), // 6
        (nodes[3], t, 0),        // 7: zero-cost finishing edge
        (nodes[3], nodes[1], 0), // 8: zero-cost cycle, must never be reused
        (nodes[2], t, 1),        // 9: completes another cost-3 route [1, 9]
        (nodes[4], t, 1),        // 10
    ]);

    (graph, s, t)
}

#[test]
fn first_five_match_exhaustive_order() {
    let (graph, start, goal) = fixture_graph();

    let expected = brute_force_simple_paths(&graph, start, goal);
    assert!(expected.len() >= 5, "fixture should offer at least 5 paths");

    let actual: Vec<(u32, Vec<EdgeIndex>)> =
        simple_shortest_paths(&graph, start, goal, |edge| *edge.weight())
            .take(5)
            .map(|path| (path.cost, path.edges))
            .collect();

    assert_eq!(actual, expected[..5]);
}

#[test]
fn full_iteration_matches_exhaustive_simple_paths() {
    let (graph, start, goal) = fixture_graph();

    let expected = brute_force_simple_paths(&graph, start, goal);
    let actual: Vec<(u32, Vec<EdgeIndex>)> =
        simple_shortest_paths(&graph, start, goal, |edge| *edge.weight())
            .map(|path| (path.cost, path.edges))
            .collect();

    assert_eq!(actual, expected);
}

#[test]
fn parallel_edges_are_distinguished_by_edge_id() {
    let mut graph = DiGraph::<(), u32>::new();
    let start = graph.add_node(());
    let middle = graph.add_node(());
    let goal = graph.add_node(());
    // two parallel edges with identical cost, ids 0 and 1
    graph.add_edge(start, middle, 1);
    graph.add_edge(start, middle, 1);
    graph.add_edge(middle, goal, 1);

    let paths: Vec<_> = simple_shortest_paths(&graph, start, goal, |edge| *edge.weight())
        .map(|path| path.edges)
        .collect();

    assert_eq!(paths.len(), 2);
    assert_eq!(paths[0], vec![EdgeIndex::new(0), EdgeIndex::new(2)]);
    assert_eq!(paths[1], vec![EdgeIndex::new(1), EdgeIndex::new(2)]);
}

#[test]
fn equal_costs_are_tie_broken_by_edge_id_sequence() {
    let (graph, start, goal) = fixture_graph();

    let costs: Vec<u32> = simple_shortest_paths(&graph, start, goal, |edge| *edge.weight())
        .map(|path| path.cost)
        .collect();

    let mut sorted_costs = costs.clone();
    sorted_costs.sort_unstable();
    assert_eq!(costs, sorted_costs);

    // the cost-3 paths ordered lexicographically by edge id sequence
    let cost_three: Vec<_> = simple_shortest_paths(&graph, start, goal, |edge| *edge.weight())
        .filter(|path| path.cost == 3)
        .map(|path| path.edges)
        .collect();
    assert_eq!(
        cost_three[0],
        vec![EdgeIndex::new(0), EdgeIndex::new(4), EdgeIndex::new(10)]
    );
    assert_eq!(cost_three[1], vec![EdgeIndex::new(0), EdgeIndex::new(5)]);
    assert_eq!(cost_three[2], vec![EdgeIndex::new(1), EdgeIndex::new(9)]);
    assert_eq!(cost_three[3], vec![EdgeIndex::new(2)]);
}

#[test]
fn no_path_returns_no_results() {
    let (mut graph, start, _goal) = fixture_graph();
    let unreachable = graph.add_node(99);
    let other = graph.add_node(100);
    graph.add_edge(unreachable, other, 0);

    let count = simple_shortest_paths(&graph, start, unreachable, |edge| *edge.weight()).count();
    assert_eq!(count, 0);
}

#[test]
fn start_equals_goal_is_empty_path() {
    let mut graph = DiGraph::<(), u32>::new();
    let node = graph.add_node(());
    let other = graph.add_node(());
    // self loop and a zero-cost round trip must not produce further paths
    graph.add_edge(node, node, 0);
    graph.add_edge(node, other, 0);
    graph.add_edge(other, node, 0);

    let paths: Vec<_> = simple_shortest_paths(&graph, node, node, |edge| *edge.weight()).collect();

    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].cost, 0);
    assert!(paths[0].edges.is_empty());
    assert_eq!(paths[0].nodes, vec![node]);
}

#[test]
fn early_stop_does_not_pull_more_paths() {
    let (graph, start, goal) = fixture_graph();

    let mut iterator = simple_shortest_paths(&graph, start, goal, |edge| *edge.weight());
    let first = iterator.next().expect("a cheapest path exists");
    assert_eq!(first.cost, 2);
    // Dropping the iterator here must be safe and must not force a full
    // enumeration of all simple paths.
    drop(iterator);
}

#[test]
fn first_path_is_lazy() {
    // One direct zero-cost edge plus many long chains of zero-cost edges.
    // Pulling only the direct path must evaluate just the edges leaving the
    // start node, without walking the chains.
    let mut graph: DiGraph<(), u8> = DiGraph::new();
    let start = graph.add_node(());
    let goal = graph.add_node(());
    graph.add_edge(start, goal, 0);

    for _ in 0..8 {
        let chain_start = graph.add_node(());
        graph.add_edge(start, chain_start, 0);
        let mut current = chain_start;
        for _ in 0..12 {
            let next = graph.add_node(());
            graph.add_edge(current, next, 0);
            current = next;
        }
        graph.add_edge(current, goal, 0);
    }

    let evaluated = Cell::new(0usize);
    let mut iterator = simple_shortest_paths(&graph, start, goal, |edge| {
        evaluated.set(evaluated.get() + 1);
        *edge.weight()
    });

    let first = iterator.next().expect("direct path exists");
    assert_eq!(first.edges, vec![EdgeIndex::new(0)]);
    assert_eq!(
        evaluated.get(),
        9,
        "only the 9 edges leaving start are expanded"
    );
    drop(iterator);
}

#[test]
fn overflowing_costs_are_skipped_not_wrapped() {
    let mut graph: DiGraph<(), u8> = DiGraph::new();
    let start = graph.add_node(());
    let middle = graph.add_node(());
    let goal = graph.add_node(());
    // edge 0: 200 then edge 2: 200 -> 400 overflows u8 and must be skipped
    graph.add_edge(start, middle, 200);
    // edge 1: 55 then edge 2: 200 -> 255, the only representable route
    graph.add_edge(start, middle, 55);
    graph.add_edge(middle, goal, 200);

    let paths: Vec<_> = simple_shortest_paths(&graph, start, goal, |edge| *edge.weight())
        .map(|path| (path.cost, path.edges))
        .collect();

    assert_eq!(
        paths,
        vec![(255u8, vec![EdgeIndex::new(1), EdgeIndex::new(2)])]
    );
}

#[test]
fn graph_is_unchanged_before_and_after_iteration() {
    let (graph, start, goal) = fixture_graph();
    let assert_graph_unchanged = |graph: &TestGraph, snapshot: &[(u32, u32, u32)]| {
        let actual: Vec<(u32, u32, u32)> = graph
            .edge_references()
            .map(|edge| (graph[edge.source()], graph[edge.target()], *edge.weight()))
            .collect();
        assert_eq!(actual, snapshot);
    };
    let snapshot: Vec<(u32, u32, u32)> = graph
        .edge_references()
        .map(|edge| (graph[edge.source()], graph[edge.target()], *edge.weight()))
        .collect();

    let mut iterator = simple_shortest_paths(&graph, start, goal, |edge| *edge.weight());
    assert_graph_unchanged(&graph, &snapshot);
    let first = iterator.next();
    assert!(first.is_some());
    assert_graph_unchanged(&graph, &snapshot);
    let rest: Vec<_> = iterator.collect();
    assert!(!rest.is_empty());
    assert_graph_unchanged(&graph, &snapshot);
}

#[test]
fn reported_node_and_edge_sequences_are_consistent() {
    let (graph, start, goal) = fixture_graph();

    for path in simple_shortest_paths(&graph, start, goal, |edge| *edge.weight()) {
        assert_eq!(path.edges.len() + 1, path.nodes.len());
        assert_eq!(*path.nodes.first().unwrap(), start);
        assert_eq!(*path.nodes.last().unwrap(), goal);

        // reconstruct the cost and the node sequence from the graph
        let mut reconstructed_cost = 0u32;
        let mut previous = start;
        let mut unique_nodes = std::collections::HashSet::new();
        assert!(unique_nodes.insert(previous));
        for (&node, &edge_id) in path.nodes.iter().skip(1).zip(&path.edges) {
            let edge = graph.edge_endpoints(edge_id).unwrap();
            assert_eq!(edge.0, previous);
            assert_eq!(edge.1, node);
            reconstructed_cost += graph[edge_id];
            assert!(unique_nodes.insert(node), "path repeats a node");
            previous = node;
        }
        assert_eq!(reconstructed_cost, path.cost);
    }
}
