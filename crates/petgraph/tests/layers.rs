use petgraph::{Graph, algo::layers};

/// a, b both point to c, which points to d => [a, b] / [c] / [d].
#[test]
fn dag_layers_in_index_order() {
    let mut g = Graph::<(), ()>::new();
    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());
    let d = g.add_node(());
    g.extend_with_edges([(a, c), (b, c), (c, d)]);

    let batches: Vec<Vec<_>> = layers(&g).collect();
    assert_eq!(batches, vec![vec![a, b], vec![c], vec![d]]);
}

/// Two parallel edges a -> c must not over-decrement the in-degree of c:
/// the layering is identical to the single-edge graph.
#[test]
fn parallel_edges_do_not_change_layers() {
    let mut single = Graph::<(), ()>::new();
    let mut parallel = Graph::<(), ()>::new();
    for g in [&mut single, &mut parallel] {
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        g.extend_with_edges([(a, c), (b, c), (c, d)]);
    }
    // add a second, parallel a -> c edge
    let a = 0.into();
    let c = 2.into();
    parallel.add_edge(a, c, ());

    let expected: Vec<Vec<_>> = layers(&single).collect();
    let actual: Vec<Vec<_>> = layers(&parallel).collect();
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), 3);
}

/// Cycles: the unlayerable nodes are returned as one real, non-empty final
/// batch (the cycle closure), not as an empty tail batch.
#[test]
fn cycle_tail_is_real_closure() {
    let mut g = Graph::<(), ()>::new();
    let s = g.add_node(()); // 0: feeds the cycle
    let x = g.add_node(()); // 1 \
    let y = g.add_node(()); // 2  | cycle x -> y -> z -> x
    let z = g.add_node(()); // 3 /
    let t = g.add_node(()); // 4: downstream of the cycle
    g.extend_with_edges([(s, x), (x, y), (y, z), (z, x), (z, t)]);

    let batches: Vec<Vec<_>> = layers(&g).collect();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0], vec![s]);
    // the closure contains the cycle *and* everything trapped behind it
    assert_eq!(batches[1], vec![x, y, z, t]);
    assert!(!batches[1].is_empty());

    // every node yielded exactly once
    let mut flat: Vec<_> = batches.iter().flatten().copied().collect();
    flat.sort();
    flat.dedup();
    assert_eq!(flat.len(), g.node_count());
}

/// A self loop is a cycle of length one and lands in the closure batch.
#[test]
fn self_loop_goes_to_cycle_closure() {
    let mut g = Graph::<(), ()>::new();
    let a = g.add_node(());
    let b = g.add_node(());
    g.add_edge(a, b, ());
    g.add_edge(b, b, ());

    let batches: Vec<Vec<_>> = layers(&g).collect();
    assert_eq!(batches, vec![vec![a], vec![b]]);
}

/// Early stop: taking only the first layer and dropping the iterator must
/// not touch the input graph, and a fresh iterator still sees everything.
#[test]
fn early_drop_leaves_graph_untouched() {
    let mut g = Graph::<(), ()>::new();
    let nodes: Vec<_> = (0..8).map(|_| g.add_node(())).collect();
    for w in nodes.windows(2) {
        g.add_edge(w[0], w[1], ());
    }
    let (node_count, edge_count) = (g.node_count(), g.edge_count());

    {
        let mut it = layers(&g);
        assert_eq!(it.next(), Some(vec![nodes[0]]));
    } // iterator dropped after a single layer

    assert_eq!(g.node_count(), node_count);
    assert_eq!(g.edge_count(), edge_count);

    // a fresh iterator reproduces the full chain of layers
    let batches: Vec<Vec<_>> = layers(&g).collect();
    assert_eq!(batches.len(), nodes.len());
    for (layer, &n) in batches.iter().zip(nodes.iter()) {
        assert_eq!(layer, &vec![n]);
    }
}

/// StableGraph with vacant slots and isolated nodes: holes are skipped and
/// every live node appears in exactly one batch.
#[cfg(feature = "stable_graph")]
#[test]
fn stable_graph_holes_and_isolated_nodes() {
    use petgraph::stable_graph::StableGraph;

    let mut g = StableGraph::<(), ()>::new();
    let a = g.add_node(()); // 0
    let hole = g.add_node(()); // 1: removed below
    let b = g.add_node(()); // 2
    let iso = g.add_node(()); // 3: isolated
    let c = g.add_node(()); // 4
    g.remove_node(hole);
    g.extend_with_edges([(a, c), (b, c)]);

    let batches: Vec<Vec<_>> = layers(&g).collect();
    assert_eq!(batches, vec![vec![a, b, iso], vec![c]]);

    // count it out: each valid node appears exactly once
    let mut flat: Vec<_> = batches.iter().flatten().copied().collect();
    assert_eq!(flat.len(), g.node_count());
    flat.sort();
    flat.dedup();
    assert_eq!(flat.len(), g.node_count());
    assert!(!flat.contains(&hole));
}
