//! Tests for DFA-constrained shortest paths.

use petgraph::{
    Graph,
    algo::dfa_shortest_path::{Dfa, DfaError, DfaPath, DfaShortestPathError, dfa_shortest_path},
    prelude::*,
    stable_graph::StableGraph,
};

/// A DFA accepting exactly the label sequence "a then b":
/// q0 --a--> q1 --b--> q2 (accepting).
fn a_then_b_dfa() -> Dfa<char> {
    Dfa::new(3, 0, [2], [(0, 'a', 1), (1, 'b', 2)]).unwrap()
}

#[test]
fn prefers_legal_two_edge_path_over_cheap_illegal_direct_edge() {
    let dfa = a_then_b_dfa();

    let mut graph: Graph<(), (char, u64)> = Graph::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    let u = graph.add_node(());
    // Cheap but illegal direct edge: 'b' has no transition from q0.
    graph.add_edge(s, t, ('b', 1));
    // The legal "a then b" detour is more expensive.
    let e_a = graph.add_edge(s, u, ('a', 2));
    let e_b = graph.add_edge(u, t, ('b', 2));

    let path = dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight())
        .unwrap()
        .expect("a valid path exists");

    assert_eq!(path.edges, vec![e_a, e_b]);
    assert_eq!(path.cost, 4);

    // Re-accumulate the cost of the returned edge ids and cross-check.
    let accumulated: u64 = path
        .edges
        .iter()
        .map(|&edge_id| graph.edge_weight(edge_id).unwrap().1)
        .sum();
    assert_eq!(accumulated, path.cost);
}

#[test]
fn zero_weight_self_loop_terminates() {
    // q0 loops on 'c', so the zero-cost self-loop in the graph keeps the
    // search in the same DFA state; the search must still terminate.
    let dfa = Dfa::new(
        3,
        0,
        [2],
        [(0, 'c', 0), (0, 'a', 1), (1, 'c', 1), (1, 'b', 2)],
    )
    .unwrap();

    let mut graph: Graph<(), (char, u64)> = Graph::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    graph.add_edge(s, s, ('c', 0)); // zero-weight self-loop
    graph.add_edge(s, t, ('a', 0)); // zero-weight chain
    graph.add_edge(t, t, ('c', 0));
    graph.add_edge(t, s, ('b', 0)); // zero-weight cycle back to s

    let path = dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight())
        .unwrap()
        .expect("a valid path exists");

    assert_eq!(path.cost, 0);
    // The path must end in the accepting state q2.
    let mut state = dfa.start_state();
    for &edge_id in &path.edges {
        let (label, _) = *graph.edge_weight(edge_id).unwrap();
        state = dfa.transition(state, &label).expect("valid transition");
    }
    assert!(dfa.is_accepting(state));
}

#[test]
fn unreachable_accepting_state_returns_none() {
    let dfa = a_then_b_dfa();

    let mut graph: Graph<(), (char, u64)> = Graph::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    // Only 'a' edges: the accepting state q2 is unreachable.
    graph.add_edge(s, t, ('a', 1));

    let path = dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight()).unwrap();
    assert_eq!(path, None);
}

#[test]
fn empty_path_when_start_state_is_accepting() {
    // The start state q0 is accepting, so the empty path is valid.
    let dfa = Dfa::new(2, 0, [0], [(0, 'a', 1)]).unwrap();

    let mut graph: Graph<(), (char, u64)> = Graph::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    graph.add_edge(s, t, ('a', 5));

    let path = dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight())
        .unwrap()
        .expect("the empty path is valid");

    assert_eq!(
        path,
        DfaPath {
            edges: vec![],
            cost: 0
        }
    );
}

#[test]
fn stable_graph_with_deleted_slots() {
    let dfa = a_then_b_dfa();

    let mut graph: StableGraph<(), (char, u64)> = StableGraph::new();
    let s = graph.add_node(());
    let gone = graph.add_node(());
    let u = graph.add_node(());
    let t = graph.add_node(());
    graph.add_edge(s, gone, ('a', 1));
    graph.add_edge(s, u, ('a', 2));
    graph.add_edge(u, t, ('b', 3));
    // Remove a node, leaving a vacant slot in the middle of the index space.
    graph.remove_node(gone);

    let path = dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight())
        .unwrap()
        .expect("a valid path exists");
    assert_eq!(path.edges.len(), 2);
    assert_eq!(path.cost, 5);
    let accumulated: u64 = path
        .edges
        .iter()
        .map(|&edge_id| graph.edge_weight(edge_id).unwrap().1)
        .sum();
    assert_eq!(accumulated, path.cost);

    // Starting from the deleted node is an explicit error.
    let result = dfa_shortest_path(&graph, gone, &dfa, |edge| *edge.weight());
    assert_eq!(result, Err(DfaShortestPathError::InvalidStartNode(gone)));
}

#[test]
fn parallel_edges_keep_their_identity() {
    let dfa = a_then_b_dfa();

    let mut graph: Graph<(), (char, u64)> = Graph::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    let u = graph.add_node(());
    // Two parallel edges s -> u, both labeled 'a'; the second is cheaper.
    let expensive = graph.add_edge(s, u, ('a', 9));
    let cheap = graph.add_edge(s, u, ('a', 1));
    let e_b = graph.add_edge(u, t, ('b', 1));

    let path = dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight())
        .unwrap()
        .expect("a valid path exists");

    assert_eq!(path.edges, vec![cheap, e_b]);
    assert_ne!(path.edges[0], expensive);
    assert_eq!(path.cost, 2);
}

#[test]
fn cost_overflow_is_reported() {
    let dfa = a_then_b_dfa();

    let mut graph: Graph<(), (char, u64)> = Graph::new();
    let s = graph.add_node(());
    let u = graph.add_node(());
    let t = graph.add_node(());
    graph.add_edge(s, u, ('a', u64::MAX));
    graph.add_edge(u, t, ('b', 1));

    let result = dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight());
    assert_eq!(result, Err(DfaShortestPathError::CostOverflow));
}

#[test]
fn invalid_start_node_is_reported() {
    let dfa = a_then_b_dfa();

    let mut graph: Graph<(), (char, u64)> = Graph::new();
    let s = graph.add_node(());
    let missing = NodeIndex::new(42);

    let result = dfa_shortest_path(&graph, missing, &dfa, |edge| *edge.weight());
    assert_eq!(result, Err(DfaShortestPathError::InvalidStartNode(missing)));

    // A valid start node still works.
    assert!(
        dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight())
            .unwrap()
            .is_none()
    );
}

#[test]
fn dfa_construction_errors_are_explicit() {
    assert_eq!(Dfa::new(0, 0, [], [(0, 'a', 0)]), Err(DfaError::NoStates));
    assert_eq!(
        Dfa::new(2, 5, [], [(0, 'a', 1)]),
        Err(DfaError::StartStateOutOfBounds {
            state: 5,
            num_states: 2
        })
    );
    assert_eq!(
        Dfa::new(2, 0, [7], [(0, 'a', 1)]),
        Err(DfaError::AcceptingStateOutOfBounds {
            state: 7,
            num_states: 2
        })
    );
    assert_eq!(
        Dfa::new(2, 0, [], [(3, 'a', 1)]),
        Err(DfaError::TransitionSourceOutOfBounds {
            state: 3,
            num_states: 2
        })
    );
    // A transition pointing at a nonexistent state is rejected.
    assert_eq!(
        Dfa::new(2, 0, [], [(0, 'a', 9)]),
        Err(DfaError::TransitionTargetOutOfBounds {
            state: 9,
            num_states: 2
        })
    );
}

#[test]
fn dfa_is_not_modified_by_the_search() {
    let dfa = a_then_b_dfa();
    let before = dfa.clone();

    let mut graph: Graph<(), (char, u64)> = Graph::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    graph.add_edge(s, t, ('a', 1));

    // The traversal callback only gets edge references; the DFA stays an
    // immutable shared borrow for the whole search.
    let _ = dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight()).unwrap();

    assert_eq!(dfa, before);
    assert_eq!(dfa.num_states(), 3);
    assert_eq!(dfa.start_state(), 0);
    assert!(dfa.is_accepting(2));
    assert!(!dfa.is_accepting(0));
    assert_eq!(dfa.transition(0, &'a'), Some(1));
    assert_eq!(dfa.transition(0, &'b'), None);
}

/// A tiny deterministic PRNG (xorshift64) so the exhaustive comparison does
/// not depend on external crates.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Exhaustively compute the minimum cost of any accepted path by DFS over
/// (node, dfa state) pairs. Since costs are non-negative, a cheapest
/// accepted walk never needs to revisit a pair, so enumerating pair-simple
/// paths is sufficient.
fn brute_force_min_cost(
    edges: &[(usize, usize, char, u64)],
    dfa: &Dfa<char>,
    start: usize,
) -> Option<u64> {
    fn dfs(
        edges: &[(usize, usize, char, u64)],
        dfa: &Dfa<char>,
        node: usize,
        state: usize,
        cost: u64,
        visited: &mut Vec<(usize, usize)>,
        best: &mut Option<u64>,
    ) {
        if dfa.is_accepting(state) {
            *best = Some(best.map_or(cost, |b| b.min(cost)));
            // A cheapest accepted path stops at the first accepted pair.
            return;
        }
        for &(from, to, label, edge_cost) in edges {
            if from != node {
                continue;
            }
            let Some(next_state) = dfa.transition(state, &label) else {
                continue;
            };
            let next = (to, next_state);
            if visited.contains(&next) {
                continue;
            }
            visited.push(next);
            dfs(edges, dfa, to, next_state, cost + edge_cost, visited, best);
            visited.pop();
        }
    }

    let mut best = None;
    let mut visited = vec![(start, dfa.start_state())];
    dfs(
        edges,
        dfa,
        start,
        dfa.start_state(),
        0,
        &mut visited,
        &mut best,
    );
    best
}

#[test]
fn matches_brute_force_on_small_graphs() {
    let labels = ['a', 'b', 'c'];
    let mut rng = Rng(0x9E3779B97F4A7C15);

    for _case in 0..300 {
        let num_nodes = 2 + rng.below(4) as usize;
        let num_edges = rng.below(8) as usize;
        let edges: Vec<(usize, usize, char, u64)> = (0..num_edges)
            .map(|_| {
                (
                    rng.below(num_nodes as u64) as usize,
                    rng.below(num_nodes as u64) as usize,
                    labels[rng.below(3) as usize],
                    rng.below(5),
                )
            })
            .collect();

        let num_states = 1 + rng.below(3) as usize;
        let accepting: Vec<usize> = (0..num_states).filter(|_| rng.below(2) == 0).collect();
        let num_transitions = rng.below(6) as usize;
        let transitions: Vec<(usize, char, usize)> = (0..num_transitions)
            .map(|_| {
                (
                    rng.below(num_states as u64) as usize,
                    labels[rng.below(3) as usize],
                    rng.below(num_states as u64) as usize,
                )
            })
            .collect();
        let dfa = Dfa::new(num_states, 0, accepting, transitions).unwrap();

        let mut graph: Graph<(), (char, u64)> = Graph::new();
        let nodes: Vec<NodeIndex> = (0..num_nodes).map(|_| graph.add_node(())).collect();
        for &(from, to, label, cost) in &edges {
            graph.add_edge(nodes[from], nodes[to], (label, cost));
        }
        let start = nodes[rng.below(num_nodes as u64) as usize];
        let start_raw = start.index();

        let result = dfa_shortest_path(&graph, start, &dfa, |edge| *edge.weight()).unwrap();
        let expected = brute_force_min_cost(&edges, &dfa, start_raw);

        assert_eq!(
            result.as_ref().map(|path| path.cost),
            expected,
            "cost mismatch for edges {edges:?} and dfa {dfa:?}"
        );

        if let Some(path) = result {
            // Verify the returned edge id sequence: it must form a chain
            // from `start`, be accepted by the DFA, and its re-accumulated
            // cost must equal the reported total.
            let mut node = start;
            let mut state = dfa.start_state();
            let mut accumulated = 0;
            for &edge_id in &path.edges {
                let (source, target) = graph.edge_endpoints(edge_id).unwrap();
                assert_eq!(source, node, "path is not a chain");
                let &(label, cost) = graph.edge_weight(edge_id).unwrap();
                state = dfa.transition(state, &label).expect("valid transition");
                accumulated += cost;
                node = target;
            }
            assert!(dfa.is_accepting(state), "path is not accepted");
            assert_eq!(accumulated, path.cost, "reported cost mismatch");
        }
    }
}
