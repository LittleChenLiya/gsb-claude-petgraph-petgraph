//! Tests for the `petgraph::transpose` module (transpose-copy utilities).

use petgraph::Directed;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex, UnGraph};
use petgraph::stable_graph::{StableDiGraph, StableUnGraph};
use petgraph::transpose::{
    TransposeError, transpose_copy, transpose_stable_copy, try_transpose_copy,
    try_transpose_stable_copy,
};
use petgraph::visit::{EdgeIndexable, NodeIndexable};

/// a -> b twice (parallel edges) plus a self-loop on b: after transposing,
/// both b -> a edges and the self-loop must be present, and every edge must
/// be traceable through the returned edge map.
#[test]
fn transpose_graph_parallel_edges_and_self_loop() {
    let mut g = DiGraph::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let e0 = g.add_edge(a, b, "ab-1");
    let e1 = g.add_edge(a, b, "ab-2");
    let e2 = g.add_edge(b, b, "bb-loop");

    let t = transpose_copy(&g).unwrap();

    // Node mapping: every original node has exactly one image.
    assert_eq!(t.node_map.len(), 2);
    let na = t.node_map[a.index()];
    let nb = t.node_map[b.index()];
    assert_ne!(na, nb);
    assert_eq!(t.graph[na], "a");
    assert_eq!(t.graph[nb], "b");

    // Edge mapping, printed one by one, and checked edge by edge.
    assert_eq!(t.edge_map.len(), 3);
    for e in g.edge_indices() {
        let (os, ot) = g.edge_endpoints(e).unwrap();
        let ne = t.edge_map[e.index()];
        let (ns, nt) = t.graph.edge_endpoints(ne).unwrap();
        println!(
            "edge {:?}: {:?} -> {:?} (weight {:?})  =>  {:?}: {:?} -> {:?} (weight {:?})",
            e, os, ot, g[e], ne, ns, nt, t.graph[ne],
        );
        // The edge was reversed and keeps its (cloned) weight.
        assert_eq!((ns, nt), (t.node_map[ot.index()], t.node_map[os.index()]));
        assert_eq!(t.graph[ne], g[e]);
    }

    // Both parallel edges are now b -> a, the self-loop is still on b.
    let (s0, t0) = t.graph.edge_endpoints(t.edge_map[e0.index()]).unwrap();
    let (s1, t1) = t.graph.edge_endpoints(t.edge_map[e1.index()]).unwrap();
    let (s2, t2) = t.graph.edge_endpoints(t.edge_map[e2.index()]).unwrap();
    assert_eq!((s0, t0), (nb, na));
    assert_eq!((s1, t1), (nb, na));
    assert_eq!((s2, t2), (nb, nb));
    assert_ne!(t.edge_map[e0.index()], t.edge_map[e1.index()]);

    // Nothing was merged: 3 edges, 2 of them parallel b -> a, 1 self-loop.
    assert_eq!(t.graph.edge_count(), 3);
    assert_eq!(t.graph.edges_connecting(nb, na).count(), 2);
    assert_eq!(t.graph.edges_connecting(nb, nb).count(), 1);
    assert_eq!(t.graph.edges_connecting(na, nb).count(), 0);

    // The original graph is unchanged.
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.edge_count(), 3);
    assert_eq!(g.edge_endpoints(e0), Some((a, b)));
    assert_eq!(g.edge_endpoints(e1), Some((a, b)));
    assert_eq!(g.edge_endpoints(e2), Some((b, b)));
    assert_eq!(g[e0], "ab-1");
    assert_eq!(g[a], "a");
}

/// Remove a node from a StableGraph, then transpose: the removed node must
/// not appear in the node map (its slot maps to `None`) and must not be
/// treated as a live node in the transposed graph.
#[test]
fn transpose_stable_graph_after_node_removal() {
    let mut g = StableDiGraph::new();
    let a = g.add_node("a");
    let b = g.add_node("b"); // will be removed
    let c = g.add_node("c");
    let e0 = g.add_edge(a, b, "ab");
    let e1 = g.add_edge(b, c, "bc");
    let e2 = g.add_edge(a, c, "ac");
    let e3 = g.add_edge(c, c, "cc-loop");

    g.remove_node(b);
    assert!(!g.contains_node(b));

    let t = transpose_stable_copy(&g).unwrap();

    // The map has one entry per slot of the source graph, and the removed
    // node's slot maps to `None`: the deleted node never appears in the map.
    assert_eq!(t.node_map.len(), g.node_bound());
    assert_eq!(t.node_map[b.index()], None);
    // Exactly the two live nodes appear in the map; the vacant slot does
    // not contribute any entry.
    assert_eq!(t.node_map.iter().filter(|m| m.is_some()).count(), 2);
    // Live nodes are mapped and their weights were cloned.
    let na = t.node_map[a.index()].expect("live node must be mapped");
    let nc = t.node_map[c.index()].expect("live node must be mapped");
    assert_ne!(na, nc);
    assert_eq!(t.graph[na], "a");
    assert_eq!(t.graph[nc], "c");
    assert_eq!(t.graph.node_count(), 2);

    // Edges that died with node `b` map to `None`; surviving edges are
    // reversed and traceable.
    assert_eq!(t.edge_map.len(), g.edge_bound());
    assert_eq!(t.edge_map[e0.index()], None);
    assert_eq!(t.edge_map[e1.index()], None);
    let ne2 = t.edge_map[e2.index()].expect("live edge must be mapped");
    let ne3 = t.edge_map[e3.index()].expect("live edge must be mapped");
    assert_eq!(t.graph.edge_endpoints(ne2), Some((nc, na)));
    assert_eq!(t.graph.edge_endpoints(ne3), Some((nc, nc)));
    assert_eq!(t.graph[ne2], "ac");
    assert_eq!(t.graph[ne3], "cc-loop");
    assert_eq!(t.graph.edge_count(), 2);

    // No node in the transposed graph corresponds to the removed node.
    for n in t.graph.node_indices() {
        assert!(t.node_map.contains(&Some(n)));
        assert_ne!(t.graph[n], "b");
    }

    // The original graph is unchanged.
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.edge_count(), 2);
    assert!(!g.contains_node(b));
    assert_eq!(g.edge_endpoints(e2), Some((a, c)));
}

/// The clone closures are called exactly once per live element; vacant
/// slots of a StableGraph are never cloned.
#[test]
fn transpose_stable_graph_clone_calls_skip_vacant_slots() {
    use core::cell::RefCell;

    let mut g = StableDiGraph::new();
    let a = g.add_node(10);
    let b = g.add_node(20);
    let c = g.add_node(30);
    g.add_edge(a, b, 1);
    g.add_edge(b, c, 2);
    g.add_edge(a, c, 3);
    g.remove_node(b);

    let cloned_nodes = RefCell::new(Vec::new());
    let cloned_edges = RefCell::new(Vec::new());
    let t = try_transpose_stable_copy(
        &g,
        |w| {
            cloned_nodes.borrow_mut().push(*w);
            Ok::<_, core::convert::Infallible>(*w)
        },
        |w| {
            cloned_edges.borrow_mut().push(*w);
            Ok::<_, core::convert::Infallible>(*w)
        },
    )
    .unwrap();

    // Only the two live nodes and the one live edge were cloned.
    assert_eq!(cloned_nodes.into_inner(), vec![10, 30]);
    assert_eq!(cloned_edges.into_inner(), vec![3]);
    assert_eq!(t.graph.node_count(), 2);
    assert_eq!(t.graph.edge_count(), 1);
}

/// A failing node or edge clone must abort the whole operation: an error is
/// returned and no (partial) graph is produced. The source graph stays
/// untouched.
#[test]
fn transpose_clone_failure_returns_no_partial_graph() {
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CloneFailed;

    impl core::fmt::Display for CloneFailed {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "clone failed")
        }
    }
    impl core::error::Error for CloneFailed {}

    let mut g = DiGraph::new();
    let a = g.add_node(0);
    let b = g.add_node(1);
    let c = g.add_node(2);
    g.add_edge(a, b, 10);
    g.add_edge(b, c, 20);
    g.add_edge(c, a, 30);

    // Node clone failure on node index 1.
    let result = try_transpose_copy(
        &g,
        |w| if *w == 1 { Err(CloneFailed) } else { Ok(*w) },
        |e| Ok::<_, CloneFailed>(*e),
    );
    match result {
        Err(TransposeError::NodeClone { index, error }) => {
            assert_eq!(index, 1);
            assert_eq!(error, CloneFailed);
        }
        other => panic!("expected NodeClone error, got {other:?}"),
    }

    // Edge clone failure on edge index 1.
    let result = try_transpose_copy(
        &g,
        |n| Ok::<_, CloneFailed>(*n),
        |e| if *e == 20 { Err(CloneFailed) } else { Ok(*e) },
    );
    match result {
        Err(TransposeError::EdgeClone { index, error }) => {
            assert_eq!(index, 1);
            assert_eq!(error, CloneFailed);
        }
        other => panic!("expected EdgeClone error, got {other:?}"),
    }

    // Same for StableGraph.
    let mut sg = StableDiGraph::new();
    let x = sg.add_node(0);
    let y = sg.add_node(1);
    sg.add_edge(x, y, 5);
    let result = try_transpose_stable_copy(
        &sg,
        |n| Ok::<_, CloneFailed>(*n),
        |_| Err::<i32, CloneFailed>(CloneFailed),
    );
    assert!(matches!(
        result,
        Err(TransposeError::EdgeClone { index: 0, .. })
    ));

    // The source graphs are unchanged.
    assert_eq!(g.node_count(), 3);
    assert_eq!(g.edge_count(), 3);
    assert_eq!(g[EdgeIndex::new(1)], 20);
    assert_eq!(g[NodeIndex::new(1)], 1);
    assert_eq!(sg.node_count(), 2);
    assert_eq!(sg.edge_count(), 1);
}

/// Transposing an empty graph yields an empty graph with empty maps.
#[test]
fn transpose_empty_graph() {
    let g: DiGraph<(), ()> = DiGraph::new();
    let t = transpose_copy(&g).unwrap();
    assert_eq!(t.graph.node_count(), 0);
    assert_eq!(t.graph.edge_count(), 0);
    assert!(t.node_map.is_empty());
    assert!(t.edge_map.is_empty());

    let sg: StableDiGraph<(), ()> = StableDiGraph::new();
    let t = transpose_stable_copy(&sg).unwrap();
    assert_eq!(t.graph.node_count(), 0);
    assert_eq!(t.graph.edge_count(), 0);
    assert!(t.node_map.is_empty());
    assert!(t.edge_map.is_empty());
}

/// Undirected graphs are rejected at runtime with
/// [`TransposeError::Undirected`].
#[test]
fn transpose_undirected_graph_is_rejected() {
    let mut g: UnGraph<i32, i32> = UnGraph::new_undirected();
    let a = g.add_node(1);
    let b = g.add_node(2);
    g.add_edge(a, b, 3);

    let result = transpose_copy(&g);
    assert!(matches!(result, Err(TransposeError::Undirected)));

    let result = try_transpose_copy(
        &g,
        |n| Ok::<_, core::convert::Infallible>(*n),
        |e| Ok::<_, core::convert::Infallible>(*e),
    );
    assert!(matches!(result, Err(TransposeError::Undirected)));

    let mut sg: StableUnGraph<i32, i32> = StableUnGraph::default();
    let a = sg.add_node(1);
    let b = sg.add_node(2);
    sg.add_edge(a, b, 3);

    let result = transpose_stable_copy(&sg);
    assert!(matches!(result, Err(TransposeError::Undirected)));

    // The undirected source graphs are unchanged.
    assert_eq!(g.edge_count(), 1);
    assert_eq!(sg.edge_count(), 1);
}

/// The transposed graph of a directed graph is itself directed.
#[test]
fn transposed_graph_is_directed() {
    let mut g = DiGraph::new();
    let a = g.add_node(1);
    let b = g.add_node(2);
    g.add_edge(a, b, ());

    let t = transpose_copy(&g).unwrap();
    let _: &petgraph::graph::Graph<i32, (), Directed> = &t.graph;
    assert!(t.graph.is_directed());
}
