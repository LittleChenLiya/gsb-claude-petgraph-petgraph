use std::collections::HashMap as StdHashMap;
use std::collections::HashSet as StdHashSet;

use petgraph::Graph;
use petgraph::algo::dfa_shortest_path::{Dfa, DfaPathError, dfa_shortest_path};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableGraph;
use petgraph::visit::EdgeRef;

/// Automaton accepting sequences "first a, then b": 'x' self-loops in the
/// start state, one or more 'a's, then a 'b' reaches the accepting state.
fn a_then_b_dfa() -> Dfa<u32, char> {
    let mut dfa = Dfa::new(0);
    dfa.add_transition(0, 'x', 0);
    dfa.add_transition(0, 'a', 1);
    dfa.add_transition(1, 'a', 1);
    dfa.add_transition(1, 'b', 2);
    dfa.add_accepting(2);
    dfa
}

/// Graph with a cheap but illegal direct edge s -> t labelled 'x' (cost 1)
/// and a legal two-edge path s -> m -> t labelled 'a', 'b' (cost 5 each).
fn illegal_shortcut_graph() -> (
    Graph<(), char>,
    NodeIndex,
    NodeIndex,
    NodeIndex,
    EdgeIndex,
    EdgeIndex,
) {
    let mut g: Graph<(), char> = Graph::new();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    // Illegal: 'x' only self-loops the automaton start state, so this edge
    // can never be part of an accepted path.
    g.add_edge(s, t, 'x');
    let e1 = g.add_edge(s, m, 'a');
    let e2 = g.add_edge(m, t, 'b');
    (g, s, m, t, e1, e2)
}

fn label_cost(label: char) -> u64 {
    match label {
        'x' => 1,
        _ => 5,
    }
}

#[test]
fn prefers_legal_two_edge_path_over_cheap_illegal_direct_edge() {
    let (g, s, _m, _t, e1, e2) = illegal_shortcut_graph();
    let dfa = a_then_b_dfa();

    let path = dfa_shortest_path(&g, s, &dfa, |e| label_cost(*e.weight()))
        .unwrap()
        .expect("a legal path exists");

    // The direct edge is cheaper (1) but its label sequence is rejected by
    // the automaton, so the result must be the legal two-edge path.
    assert_eq!(path.edges, vec![e1, e2]);
    assert_eq!(path.cost, 10);

    // Re-accumulate the cost from the returned edge ids and check it matches.
    let mut total = 0;
    for &eid in &path.edges {
        total += label_cost(*g.edge_weight(eid).unwrap());
    }
    assert_eq!(total, path.cost);
}

#[test]
fn parallel_edges_keep_their_identity() {
    let mut g: Graph<(), char> = Graph::new();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    // Two parallel 'a' edges with different costs; the cheaper one (added
    // second) must be identifiable in the result.
    let expensive_a = g.add_edge(s, m, 'a');
    let cheap_a = g.add_edge(s, m, 'a');
    let b = g.add_edge(m, t, 'b');
    let costs = |e: petgraph::graph::EdgeReference<'_, char>| {
        if e.id() == cheap_a { 1u64 } else { 7 }
    };

    let path = dfa_shortest_path(&g, s, &a_then_b_dfa(), costs)
        .unwrap()
        .unwrap();
    assert_eq!(path.edges, vec![cheap_a, b]);
    assert_ne!(path.edges[0], expensive_a);
    assert_eq!(path.cost, 8);
}

#[test]
fn zero_weight_self_loop_terminates() {
    let mut g: Graph<(), char> = Graph::new();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    // Zero-cost self loops that keep the automaton in its current state.
    g.add_edge(s, s, 'x');
    g.add_edge(m, m, 'x');
    let e1 = g.add_edge(s, m, 'a');
    let e2 = g.add_edge(m, t, 'b');

    let path = dfa_shortest_path(&g, s, &a_then_b_dfa(), |e| match *e.weight() {
        'x' => 0u64,
        _ => 3,
    })
    .unwrap()
    .unwrap();
    assert_eq!(path.edges, vec![e1, e2]);
    assert_eq!(path.cost, 6);
}

#[test]
fn unreachable_accepting_state_returns_none() {
    let mut g: Graph<(), char> = Graph::new();
    let s = g.add_node(());
    let t = g.add_node(());
    // Only 'a' edges: the automaton can never reach the accepting state.
    g.add_edge(s, t, 'a');
    g.add_edge(t, s, 'a');

    let res = dfa_shortest_path(&g, s, &a_then_b_dfa(), |_| 1u64).unwrap();
    assert_eq!(res, None);
}

#[test]
fn empty_path_when_start_state_is_accepting() {
    let mut dfa = Dfa::new(0);
    dfa.add_accepting(0);

    let mut g: Graph<(), char> = Graph::new();
    let s = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, t, 'a');

    let path = dfa_shortest_path(&g, s, &dfa, |_| 1u64).unwrap().unwrap();
    assert!(path.edges.is_empty());
    assert_eq!(path.cost, 0);
}

#[test]
fn invalid_start_node_is_an_error() {
    let (g, _s, _m, _t, _e1, _e2) = illegal_shortcut_graph();
    let missing = NodeIndex::new(999);
    let res = dfa_shortest_path(&g, missing, &a_then_b_dfa(), |_| 1u64);
    assert_eq!(
        res.unwrap_err(),
        DfaPathError::InvalidStart {
            node: missing,
            state: 0
        }
    );
}

#[test]
fn cost_overflow_is_an_error() {
    let mut g: Graph<(), char> = Graph::new();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, m, 'a');
    g.add_edge(m, t, 'b');

    let res = dfa_shortest_path(&g, s, &a_then_b_dfa(), |_| 200u8);
    assert!(matches!(res, Err(DfaPathError::CostOverflow { .. })));
}

#[test]
fn stable_graph_with_deleted_slots() {
    let mut g: StableGraph<(), char> = StableGraph::new();
    let s = g.add_node(());
    let dead = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    let e1 = g.add_edge(s, m, 'a');
    let e2 = g.add_edge(m, t, 'b');
    // Edges through the soon-to-be-deleted node, then delete it to leave a
    // vacant slot in the graph.
    g.add_edge(s, dead, 'a');
    g.add_edge(dead, t, 'b');
    g.remove_node(dead);

    let path = dfa_shortest_path(&g, s, &a_then_b_dfa(), |_| 5u64)
        .unwrap()
        .unwrap();
    assert_eq!(path.edges, vec![e1, e2]);
    assert_eq!(path.cost, 10);

    // Starting from the deleted slot is an explicit error.
    let res = dfa_shortest_path(&g, dead, &a_then_b_dfa(), |_| 5u64);
    assert_eq!(
        res.unwrap_err(),
        DfaPathError::InvalidStart {
            node: dead,
            state: 0
        }
    );
}

/// A graph whose edges point outside the node set, to exercise the
/// MissingTransitionTarget error.
#[allow(clippy::std_instead_of_alloc)]
mod broken_graph {
    use petgraph::visit::{
        Data, EdgeRef, GraphBase, GraphRef, IntoEdgeReferences, IntoEdges, IntoNeighbors,
        IntoNodeIdentifiers, Visitable,
    };

    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct N(pub u32);

    #[derive(Copy, Clone, Debug)]
    pub struct E {
        pub source: N,
        pub target: N,
        pub label: char,
        pub id: u32,
    }

    impl EdgeRef for E {
        type NodeId = N;
        type EdgeId = u32;
        type Weight = char;
        fn source(&self) -> N {
            self.source
        }
        fn target(&self) -> N {
            self.target
        }
        fn weight(&self) -> &char {
            &self.label
        }
        fn id(&self) -> u32 {
            self.id
        }
    }

    /// One node N(0) with a single edge to N(99), which is not a node.
    #[derive(Copy, Clone)]
    pub struct Broken;

    impl GraphBase for Broken {
        type NodeId = N;
        type EdgeId = u32;
    }

    impl Data for Broken {
        type NodeWeight = ();
        type EdgeWeight = char;
    }

    impl GraphRef for Broken {}

    impl IntoEdgeReferences for Broken {
        type EdgeRef = E;
        type EdgeReferences = std::vec::IntoIter<E>;
        fn edge_references(self) -> Self::EdgeReferences {
            vec![E {
                source: N(0),
                target: N(99),
                label: 'a',
                id: 0,
            }]
            .into_iter()
        }
    }

    impl IntoNeighbors for Broken {
        type Neighbors = std::vec::IntoIter<N>;
        fn neighbors(self, _a: N) -> Self::Neighbors {
            vec![N(99)].into_iter()
        }
    }

    impl Visitable for Broken {
        type Map = std::collections::HashSet<N>;
        fn visit_map(&self) -> Self::Map {
            std::collections::HashSet::new()
        }
        fn reset_map(&self, map: &mut Self::Map) {
            map.clear();
        }
    }

    impl IntoEdges for Broken {
        type Edges = std::vec::IntoIter<E>;
        fn edges(self, _a: N) -> Self::Edges {
            self.edge_references()
        }
    }

    impl IntoNodeIdentifiers for Broken {
        type NodeIdentifiers = std::vec::IntoIter<N>;
        fn node_identifiers(self) -> Self::NodeIdentifiers {
            vec![N(0)].into_iter()
        }
    }
}

#[test]
fn missing_transition_target_is_an_error() {
    use broken_graph::{Broken, N};

    let res = dfa_shortest_path(Broken, N(0), &a_then_b_dfa(), |_| 1u64);
    assert_eq!(
        res.unwrap_err(),
        DfaPathError::MissingTransitionTarget { node: N(99) }
    );
}

/// Brute-force minimum cost over all accepted paths, enumerating simple
/// paths in the product space (node, automaton state) with backtracking.
fn brute_force_min_cost(
    g: &Graph<(), char>,
    start: NodeIndex,
    dfa: &Dfa<u32, char>,
    costs: &StdHashMap<EdgeIndex, u64>,
) -> Option<u64> {
    struct Ctx<'a> {
        g: &'a Graph<(), char>,
        dfa: &'a Dfa<u32, char>,
        costs: &'a StdHashMap<EdgeIndex, u64>,
    }

    fn dfs(
        ctx: &Ctx<'_>,
        node: NodeIndex,
        state: u32,
        total: u64,
        on_path: &mut StdHashSet<(NodeIndex, u32)>,
        best: &mut Option<u64>,
    ) {
        if ctx.dfa.is_accepting(&state) {
            *best = Some(best.map_or(total, |b| b.min(total)));
        }
        for edge in ctx.g.edges(node) {
            let Some(&next_state) = ctx.dfa.transition(&state, edge.weight()) else {
                continue;
            };
            let key = (edge.target(), next_state);
            if on_path.insert(key) {
                dfs(
                    ctx,
                    edge.target(),
                    next_state,
                    total + ctx.costs[&edge.id()],
                    on_path,
                    best,
                );
                on_path.remove(&key);
            }
        }
    }

    let mut best = None;
    let mut on_path = StdHashSet::new();
    on_path.insert((start, *dfa.start()));
    let ctx = Ctx { g, dfa, costs };
    dfs(&ctx, start, *dfa.start(), 0, &mut on_path, &mut best);
    best
}

#[test]
fn matches_brute_force_on_small_graphs() {
    // Deterministic pseudo-random small graphs (splitmix64) compared against
    // exhaustive enumeration of accepted product-space paths.
    let labels = ['a', 'b', 'x'];
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rand = move || {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };

    for _case in 0..100 {
        let node_count = 2 + (rand() % 4) as usize; // 2..=5 nodes
        let mut g: Graph<(), char> = Graph::new();
        let nodes: Vec<_> = (0..node_count).map(|_| g.add_node(())).collect();
        let mut costs: StdHashMap<EdgeIndex, u64> = StdHashMap::new();
        for _ in 0..(rand() % 8) {
            let a = nodes[(rand() as usize) % node_count];
            let b = nodes[(rand() as usize) % node_count];
            let label = labels[(rand() as usize) % labels.len()];
            let e = g.add_edge(a, b, label);
            costs.insert(e, rand() % 5); // includes zero-cost edges
        }

        let dfa = a_then_b_dfa();
        let result = dfa_shortest_path(&g, nodes[0], &dfa, |e| costs[&e.id()]).unwrap();
        let expected = brute_force_min_cost(&g, nodes[0], &dfa, &costs);

        assert_eq!(
            result.as_ref().map(|p| p.cost),
            expected,
            "min cost mismatch on graph {g:?}"
        );

        if let Some(path) = result {
            // Replay the returned edge ids: labels must be accepted by the
            // DFA and the re-accumulated cost must match.
            let mut state = *dfa.start();
            let mut total = 0;
            let mut node = nodes[0];
            for &eid in &path.edges {
                let (src, dst) = g.edge_endpoints(eid).unwrap();
                assert_eq!(src, node, "path edges must be contiguous");
                let label = g.edge_weight(eid).unwrap();
                state = *dfa.transition(&state, label).expect("labels accepted");
                total += costs[&eid];
                node = dst;
            }
            assert!(dfa.is_accepting(&state));
            assert_eq!(total, path.cost);
        }
    }
}
