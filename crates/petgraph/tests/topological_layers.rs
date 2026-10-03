//! Tests for [`petgraph::algo::topological_layers`].

use petgraph::{
    algo::{TopologicalCycle, topological_layers},
    prelude::*,
    stable_graph::StableGraph,
};

/// a and b both point to c, c points to d: layers [a,b], [c], [d].
#[test]
fn diamond_layers() {
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.extend_with_edges([(a, c), (b, c), (c, d)]);

    let layers: Vec<_> = topological_layers(&graph).collect();
    assert_eq!(layers.len(), 3);

    let l0 = layers[0].as_ref().unwrap();
    let l1 = layers[1].as_ref().unwrap();
    let l2 = layers[2].as_ref().unwrap();

    // Same layer: original node-index order.
    assert_eq!(l0, &[a, b]);
    assert_eq!(l1, &[c]);
    assert_eq!(l2, &[d]);
}

/// Two parallel edges a -> c must not change the layer count or order.
#[test]
fn parallel_edges_do_not_change_layers() {
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let a = graph.add_node(());
    let c = graph.add_node(());
    graph.extend_with_edges([(a, c), (a, c)]);

    let layers: Vec<_> = topological_layers(&graph).collect();
    assert_eq!(layers.len(), 2);
    assert_eq!(layers[0].as_ref().unwrap(), &[a]);
    assert_eq!(layers[1].as_ref().unwrap(), &[c]);

    // Three parallel edges, and a longer chain, still layer by distinct
    // predecessor.
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.extend_with_edges([(a, b), (a, b), (a, b), (b, c)]);
    let layers: Vec<_> = topological_layers(&graph).collect();
    assert_eq!(layers.len(), 3);
    assert_eq!(layers[0].as_ref().unwrap(), &[a]);
    assert_eq!(layers[1].as_ref().unwrap(), &[b]);
    assert_eq!(layers[2].as_ref().unwrap(), &[c]);
}

/// Empty graph yields no layers.
#[test]
fn empty_graph() {
    let graph: Graph<(), (), Directed> = Graph::new();
    let layers: Vec<_> = topological_layers(&graph).collect();
    assert!(layers.is_empty());
}

/// A single node is its own layer.
#[test]
fn single_node() {
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let a = graph.add_node(());
    let layers: Vec<_> = topological_layers(&graph).collect();
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].as_ref().unwrap(), &[a]);
}

/// Self-loop is a cycle: the witness is a real closed loop.
#[test]
fn self_loop_cycle_witness() {
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let a = graph.add_node(());
    graph.add_edge(a, a, ());

    let mut iter = topological_layers(&graph);
    let first = iter.next();
    assert!(first.is_some());
    let err = first.unwrap().unwrap_err();
    assert_eq!(err.cycle(), &[a]);
    // The witness is a real closed loop.
    assert!(graph.edges(a).any(|e| e.target() == a));
    // The iterator ends after reporting the cycle.
    assert!(iter.next().is_none());
}

/// A directed cycle a -> b -> c -> a (with a tail d -> a) yields a witness that
/// is a genuine closed loop, not an empty tail batch.
#[test]
fn cycle_witness_is_a_real_loop() {
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let d = graph.add_node(()); // tail feeding the cycle
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.extend_with_edges([(d, a), (a, b), (b, c), (c, a)]);

    let mut iter = topological_layers(&graph);
    // The tail d is emitted first (no predecessors).
    let l0 = iter.next().unwrap().unwrap();
    assert_eq!(l0, &[d]);

    // Then the cycle is reported as a witness.
    let err = iter.next().unwrap().unwrap_err();
    let witness = err.cycle();
    assert_eq!(witness.len(), 3, "witness must be a real 3-node loop");

    // All nodes distinct.
    let mut sorted = witness.to_vec();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 3);

    // Every consecutive pair, and last -> first, is an edge.
    for w in witness.windows(2) {
        assert!(
            graph.edges(w[0]).any(|e| e.target() == w[1]),
            "expected edge {:?} -> {:?}",
            w[0],
            w[1]
        );
    }
    let last = *witness.last().unwrap();
    let first = *witness.first().unwrap();
    assert!(
        graph.edges(last).any(|e| e.target() == first),
        "closing edge {:?} -> {:?}",
        last,
        first
    );

    // No empty tail batch: the iterator ends here.
    assert!(iter.next().is_none());
}

/// With isolated nodes and deleted slots (StableGraph holes), every valid node
/// appears exactly once across all layers.
#[test]
fn stable_graph_holes_and_isolated_nodes() {
    let mut graph: StableGraph<(), (), Directed> = StableGraph::new();
    let a = graph.add_node(()); // index 0
    let b = graph.add_node(()); // index 1 (will be removed -> hole)
    let c = graph.add_node(()); // index 2
    let d = graph.add_node(()); // index 3 (will be removed -> hole)
    let e = graph.add_node(()); // index 4 (isolated)
    let f = graph.add_node(()); // index 5 (isolated)

    graph.remove_node(b);
    graph.remove_node(d);

    // a -> c; e and f are isolated.
    graph.add_edge(a, c, ());

    let valid = [a, c, e, f];
    let layers: Vec<_> = topological_layers(&graph).collect();

    // Count occurrences of every valid node across all layers.
    let mut seen = std::collections::HashMap::new();
    for layer in &layers {
        let nodes = layer.as_ref().unwrap();
        for &n in nodes {
            *seen.entry(n).or_insert(0usize) += 1;
        }
    }

    // Each valid node appears exactly once.
    for &n in &valid {
        assert_eq!(
            seen.get(&n).copied().unwrap_or(0),
            1,
            "node {n:?} should appear exactly once"
        );
    }
    // No extra (hole) nodes.
    assert_eq!(seen.len(), valid.len());

    // Layering: isolated nodes + a in the first layer (index order), c in the
    // second.
    assert_eq!(layers.len(), 2);
    let l0 = layers[0].as_ref().unwrap();
    let l1 = layers[1].as_ref().unwrap();
    assert_eq!(l0, &[a, e, f]);
    assert_eq!(l1, &[c]);
}

/// The input graph is unchanged by iteration.
#[test]
fn input_graph_unchanged() {
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.extend_with_edges([(a, b), (b, c), (a, c)]);

    let node_count_before = graph.node_count();
    let edge_count_before = graph.edge_count();

    let _layers: Vec<_> = topological_layers(&graph).collect();

    assert_eq!(graph.node_count(), node_count_before);
    assert_eq!(graph.edge_count(), edge_count_before);
    // Edges still present.
    assert!(graph.edges(a).any(|e| e.target() == b));
    assert!(graph.edges(b).any(|e| e.target() == c));
    assert!(graph.edges(a).any(|e| e.target() == c));
}

/// Early stop: dropping the iterator after the first layer does no background
/// work and leaves the graph usable.
#[test]
fn early_drop_is_lazy_and_graph_reusable() {
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.extend_with_edges([(a, b), (b, c), (c, d)]);

    let mut iter = topological_layers(&graph);
    let first = iter.next().unwrap().unwrap();
    assert_eq!(first, vec![a]);
    // Drop early: no background work, no panic.
    drop(iter);

    // The graph is untouched and can be iterated again from scratch.
    let again: Vec<_> = topological_layers(&graph).collect();
    assert_eq!(again.len(), 4);
    assert_eq!(again[0].as_ref().unwrap(), &[a]);
    assert_eq!(again[1].as_ref().unwrap(), &[b]);
    assert_eq!(again[2].as_ref().unwrap(), &[c]);
    assert_eq!(again[3].as_ref().unwrap(), &[d]);
}

/// The cycle error accessors work.
#[test]
fn cycle_error_accessors() {
    let err = TopologicalCycle(vec![1u32, 2, 3]);
    assert_eq!(err.cycle(), &[1, 2, 3]);
    assert_eq!(err.clone().into_cycle(), vec![1, 2, 3]);
}

/// Works with [`GraphMap`] too, not just [`Graph`].
#[test]
fn graphmap_layers() {
    let mut graph: GraphMap<u32, (), Directed> = GraphMap::new();
    let a = graph.add_node(1);
    let b = graph.add_node(2);
    let c = graph.add_node(3);
    let d = graph.add_node(4);
    graph.add_edge(a, c, ());
    graph.add_edge(b, c, ());
    graph.add_edge(c, d, ());

    let layers: Vec<_> = topological_layers(&graph).collect();
    assert_eq!(layers.len(), 3);
    assert_eq!(layers[0].as_ref().unwrap(), &[a, b]);
    assert_eq!(layers[1].as_ref().unwrap(), &[c]);
    assert_eq!(layers[2].as_ref().unwrap(), &[d]);
}

/// A longer cycle (a -> b -> c -> d -> a) yields a genuine 4-node witness.
#[test]
fn longer_cycle_witness() {
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.extend_with_edges([(a, b), (b, c), (c, d), (d, a)]);

    let mut iter = topological_layers(&graph);
    let err = iter.next().unwrap().unwrap_err();
    let witness = err.cycle();
    assert_eq!(witness.len(), 4);

    // All distinct.
    let mut sorted = witness.to_vec();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 4);

    // Every consecutive pair, and last -> first, is an edge.
    for w in witness.windows(2) {
        assert!(graph.edges(w[0]).any(|e| e.target() == w[1]));
    }
    assert!(
        graph
            .edges(*witness.last().unwrap())
            .any(|e| e.target() == *witness.first().unwrap())
    );

    assert!(iter.next().is_none());
}
