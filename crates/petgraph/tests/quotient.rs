extern crate alloc;

use alloc::collections::BTreeMap;

use petgraph::algo::quotient::{Quotient, QuotientError, SelfLoopPolicy, quotient_graph};
use petgraph::prelude::*;
use petgraph::stable_graph::{EdgeReference, StableGraph};
use petgraph::visit::EdgeRef;
use petgraph::{Directed, Undirected};

/// StableGraph with parallel edges, self-loops and a hole left by a removed
/// node. Three live nodes a, b, c are split into two groups x and y.
fn directed_fixture() -> (
    StableGraph<(), i32, Directed>,
    NodeIndex,
    NodeIndex,
    NodeIndex,
    NodeIndex,
) {
    let mut g = StableGraph::<(), i32, Directed>::new();
    let a = g.add_node(()); // 0
    let hole = g.add_node(()); // 1, removed below
    let b = g.add_node(()); // 2
    let c = g.add_node(()); // 3
    g.add_edge(a, a, 10); // e0: self-loop
    g.add_edge(a, b, 1); // e1: intra-group
    g.add_edge(b, c, 2); // e2
    g.add_edge(b, c, 3); // e3: parallel to e2
    g.add_edge(c, b, 5); // e4: reverse direction
    g.remove_node(hole);
    (g, a, b, c, hole)
}

fn partition(a: NodeIndex, b: NodeIndex, c: NodeIndex) -> BTreeMap<NodeIndex, char> {
    [(a, 'x'), (b, 'x'), (c, 'y')].into_iter().collect()
}

fn sum_weights(edges: &[EdgeReference<'_, i32>]) -> Result<i32, String> {
    Ok(edges.iter().map(|e| *e.weight()).sum())
}

#[test]
fn quotient_directed_stable_graph() {
    let (g, a, b, c, hole) = directed_fixture();
    let q: Quotient<NodeIndex, EdgeIndex, char, i32, Directed> =
        quotient_graph(&g, &partition(a, b, c), SelfLoopPolicy::Keep, sum_weights).unwrap();

    // Quotient nodes are created in label order: x = 0, y = 1.
    let x = NodeIndex::new(0);
    let y = NodeIndex::new(1);
    assert_eq!(q.graph.node_count(), 2);
    assert_eq!(q.graph[x], 'x');
    assert_eq!(q.graph[y], 'y');

    // Bidirectional node mapping; the hole is mapped to None.
    assert_eq!(
        q.node_map,
        vec![Some(x), None, Some(x), Some(y)],
        "hole left by remove_node must stay unmapped"
    );
    assert_eq!(q.groups, vec![vec![a, b], vec![c]]);
    assert!(hole.index() < q.node_map.len());

    // Edge groups in stable endpoint order: (x,x), (x,y), (y,x).
    assert_eq!(q.graph.edge_count(), 3);
    let weights: Vec<i32> = q.graph.edge_weights().copied().collect();
    assert_eq!(weights, vec![11, 5, 5], "weights are per-group sums");
    assert_eq!(
        q.edge_sources,
        vec![
            vec![EdgeIndex::new(0), EdgeIndex::new(1)],
            vec![EdgeIndex::new(2), EdgeIndex::new(3)],
            vec![EdgeIndex::new(4)],
        ],
        "parallel edges must reach the aggregator instead of disappearing"
    );
    let edges: Vec<(NodeIndex, NodeIndex)> = q
        .graph
        .edge_indices()
        .map(|e| q.graph.edge_endpoints(e).unwrap())
        .collect();
    assert_eq!(edges, vec![(x, x), (x, y), (y, x)]);
}

#[test]
fn quotient_undirected_merges_directions() {
    let mut g = StableGraph::<(), i32, Undirected>::default();
    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());
    let hole = g.add_node(());
    g.add_edge(a, a, 10); // e0
    g.add_edge(a, b, 1); // e1
    g.add_edge(b, c, 2); // e2
    g.add_edge(b, c, 3); // e3
    g.add_edge(c, b, 5); // e4: same undirected group as e2/e3
    g.remove_node(hole);

    let q: Quotient<NodeIndex, EdgeIndex, char, i32, Undirected> =
        quotient_graph(&g, &partition(a, b, c), SelfLoopPolicy::Keep, sum_weights).unwrap();

    assert_eq!(q.graph.node_count(), 2);
    assert_eq!(q.graph.edge_count(), 2);
    let weights: Vec<i32> = q.graph.edge_weights().copied().collect();
    assert_eq!(
        weights,
        vec![11, 10],
        "(u,v) and (v,u) merge in undirected graphs"
    );
    assert_eq!(
        q.edge_sources,
        vec![
            vec![EdgeIndex::new(0), EdgeIndex::new(1)],
            vec![EdgeIndex::new(2), EdgeIndex::new(3), EdgeIndex::new(4)],
        ]
    );

    // Directed keeps (x,y) and (y,x) apart; undirected merges them.
    let (dg, da, db, dc, _) = directed_fixture();
    let dq: Quotient<NodeIndex, EdgeIndex, char, i32, Directed> = quotient_graph(
        &dg,
        &partition(da, db, dc),
        SelfLoopPolicy::Keep,
        sum_weights,
    )
    .unwrap();
    assert_eq!(dq.graph.edge_count(), 3);
    assert_eq!(q.graph.edge_count(), 2);
}

#[test]
fn quotient_self_loop_policy_remove() {
    let (g, a, b, c, _) = directed_fixture();
    let q: Quotient<NodeIndex, EdgeIndex, char, i32, Directed> =
        quotient_graph(&g, &partition(a, b, c), SelfLoopPolicy::Remove, sum_weights).unwrap();
    assert_eq!(q.graph.edge_count(), 2);
    assert_eq!(
        q.edge_sources,
        vec![
            vec![EdgeIndex::new(2), EdgeIndex::new(3)],
            vec![EdgeIndex::new(4)]
        ],
        "intra-group edges are dropped before aggregation"
    );
    let weights: Vec<i32> = q.graph.edge_weights().copied().collect();
    assert_eq!(weights, vec![5, 5]);
}

#[test]
fn quotient_missing_partition_is_positional_error() {
    let (g, a, b, _, _) = directed_fixture();
    let mut p = BTreeMap::new();
    p.insert(a, 'x');
    p.insert(b, 'x');
    let err: QuotientError<NodeIndex, String> =
        quotient_graph(&g, &p, SelfLoopPolicy::Keep, sum_weights).unwrap_err();
    assert_eq!(
        err,
        QuotientError::MissingPartition {
            node: NodeIndex::new(3)
        }
    );
}

#[test]
fn quotient_invalid_node_is_positional_error() {
    let (g, a, b, c, hole) = directed_fixture();

    // The hole left by remove_node must not be silently grouped.
    let mut p = partition(a, b, c);
    p.insert(hole, 'y');
    let err: QuotientError<NodeIndex, String> =
        quotient_graph(&g, &p, SelfLoopPolicy::Keep, sum_weights).unwrap_err();
    assert_eq!(err, QuotientError::InvalidNode { node: hole });

    // An out-of-bounds index is rejected the same way.
    let mut p = partition(a, b, c);
    p.insert(NodeIndex::new(99), 'z');
    let err: QuotientError<NodeIndex, String> =
        quotient_graph(&g, &p, SelfLoopPolicy::Keep, sum_weights).unwrap_err();
    assert_eq!(
        err,
        QuotientError::InvalidNode {
            node: NodeIndex::new(99)
        }
    );
}

#[test]
fn quotient_aggregation_failure_stops_in_order() {
    let (g, a, b, c, _) = directed_fixture();
    let nodes_before = g.node_count();
    let edges_before = g.edge_count();

    let mut calls: Vec<Vec<EdgeIndex>> = Vec::new();
    let err: QuotientError<NodeIndex, &str> = quotient_graph(
        &g,
        &partition(a, b, c),
        SelfLoopPolicy::Keep,
        |edges: &[EdgeReference<'_, i32>]| {
            calls.push(edges.iter().map(|e| e.id()).collect());
            if calls.len() == 2 {
                Err("boom")
            } else {
                Ok(edges.iter().map(|e| *e.weight()).sum::<i32>())
            }
        },
    )
    .unwrap_err();

    // The second edge group (x -> y) fails; the error carries its position.
    assert_eq!(
        err,
        QuotientError::Aggregation {
            source: NodeIndex::new(0),
            target: NodeIndex::new(1),
            error: "boom",
        }
    );
    // Groups are visited in stable endpoint order and the third group
    // (y -> x) is never aggregated after the failure.
    assert_eq!(
        calls,
        vec![
            vec![EdgeIndex::new(0), EdgeIndex::new(1)],
            vec![EdgeIndex::new(2), EdgeIndex::new(3)],
        ]
    );
    // The input graph is untouched.
    assert_eq!(g.node_count(), nodes_before);
    assert_eq!(g.edge_count(), edges_before);
}

#[test]
fn quotient_is_deterministic_across_runs() {
    let (g, a, b, c, _) = directed_fixture();
    let first: Quotient<NodeIndex, EdgeIndex, char, i32, Directed> =
        quotient_graph(&g, &partition(a, b, c), SelfLoopPolicy::Keep, sum_weights).unwrap();
    let signature = |q: &Quotient<NodeIndex, EdgeIndex, char, i32, Directed>| {
        (
            q.groups.clone(),
            q.edge_sources.clone(),
            q.node_map.clone(),
            q.graph.edge_weights().copied().collect::<Vec<_>>(),
        )
    };
    let expected = signature(&first);
    for _ in 0..20 {
        let q: Quotient<NodeIndex, EdgeIndex, char, i32, Directed> =
            quotient_graph(&g, &partition(a, b, c), SelfLoopPolicy::Keep, sum_weights).unwrap();
        assert_eq!(
            signature(&q),
            expected,
            "result order must not depend on hash iteration"
        );
    }
}
