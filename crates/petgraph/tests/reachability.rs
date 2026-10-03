//! Tests for [`petgraph::reachability`].

#![cfg(feature = "stable_graph")]

use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    reachability::{reachability_snapshot, ReachabilityError},
    stable_graph::StableGraph,
    Graph,
};

/// a -> secret -> b  and  a -> public -> b.
///
/// When `secret` is forbidden, only the public route is recorded.
#[test]
fn forbidden_node_routes_around() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let secret = g.add_node("secret");
    let public = g.add_node("public");
    let b = g.add_node("b");
    g.extend_with_edges(&[(a, secret), (secret, b), (a, public), (public, b)]);

    let snap = reachability_snapshot(&g, a, [a, public, b]).unwrap();

    assert!(!snap.is_allowed(secret));
    assert!(snap.is_reachable(a));
    assert!(snap.is_reachable(public));
    assert!(snap.is_reachable(b));
    assert!(!snap.is_reachable(secret));

    let path = snap.path_to(b).unwrap();
    assert_eq!(path.nodes(), &[a, public, b]);
    assert_eq!(path.len(), 2);
}

/// After the snapshot is built, deleting the original `public` edge from the
/// graph must not change the recorded path.
#[test]
fn snapshot_survives_original_edge_deletion() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let public = g.add_node("public");
    let b = g.add_node("b");
    let public_edge = g.add_edge(a, public, ());
    g.add_edge(public, b, ());

    let snap = reachability_snapshot(&g, a, [a, public, b]).unwrap();
    let path_before = snap.path_to(b).unwrap();
    assert_eq!(path_before.nodes(), &[a, public, b]);

    // Mutate the original graph after the snapshot was built.
    let removed = g.remove_edge(public_edge);
    assert!(removed.is_some());
    assert!(!g.contains_edge(a, public));

    // The snapshot still reports the original path.
    let path_after = snap.path_to(b).unwrap();
    assert_eq!(path_after.nodes(), &[a, public, b]);
    assert_eq!(path_after.edges(), path_before.edges());
}

/// Parallel edges must keep their concrete `EdgeId`; the snapshot must not
/// collapse them to just the endpoints.
#[test]
fn parallel_edges_keep_concrete_edge_id() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let e1 = g.add_edge(a, b, ());
    let e2 = g.add_edge(a, b, ());

    let snap = reachability_snapshot(&g, a, [a, b]).unwrap();
    let path = snap.path_to(b).unwrap();

    assert_eq!(path.nodes(), &[a, b]);
    assert_eq!(path.edges().len(), 1);
    // The first-inserted edge wins the tie.
    assert_eq!(path.edges()[0], e1);
    assert_ne!(path.edges()[0], e2);

    // The reachable edges iterator yields exactly the tree edge.
    let edges: Vec<EdgeIndex> = snap.reachable_edges().collect();
    assert_eq!(edges, vec![e1]);
}

/// When there is no allowed path to the target, `path_to` returns `None`.
#[test]
fn no_path_returns_none() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    // No edge between a and c.
    g.add_edge(a, b, ());

    // c is allowed but disconnected from a.
    let snap = reachability_snapshot(&g, a, [a, b, c]).unwrap();
    assert!(snap.is_reachable(b));
    assert!(!snap.is_reachable(c));
    assert!(snap.path_to(c).is_none());

    // b is reachable, but if we forbid b the path disappears.
    let snap2 = reachability_snapshot(&g, a, [a]).unwrap();
    assert!(!snap2.is_reachable(b));
    assert!(snap2.path_to(b).is_none());
}

/// A deleted node slot in a `StableGraph` must be rejected as an invalid
/// allowed node.
#[test]
fn stable_graph_deleted_slot_is_invalid() {
    let mut g = StableGraph::<&str, ()>::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let deleted = g.add_node("deleted");
    g.add_edge(a, b, ());
    g.remove_node(deleted);

    // The deleted slot is not a live node.
    assert_eq!(
        reachability_snapshot(&g, a, [a, b, deleted]).unwrap_err(),
        ReachabilityError::InvalidAllowedNode(deleted)
    );

    // A start that points at a deleted slot is rejected explicitly.
    assert_eq!(
        reachability_snapshot(&g, deleted, [a, b]).unwrap_err(),
        ReachabilityError::InvalidStart(deleted)
    );
}

/// A start node that is live but not in the allowed set is explicitly
/// rejected.
#[test]
fn start_not_allowed_is_rejected() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_edge(a, b, ());

    assert_eq!(
        reachability_snapshot(&g, a, [b]).unwrap_err(),
        ReachabilityError::StartNotAllowed(a)
    );
}

/// An out-of-bounds node index is rejected.
#[test]
fn out_of_bounds_index_is_invalid() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_edge(a, b, ());

    let oob = NodeIndex::new(99);
    assert_eq!(
        reachability_snapshot(&g, a, [a, b, oob]).unwrap_err(),
        ReachabilityError::InvalidAllowedNode(oob)
    );
    assert_eq!(
        reachability_snapshot(&g, oob, [a, b]).unwrap_err(),
        ReachabilityError::InvalidStart(oob)
    );
}

/// Later node insertions into the original graph do not appear in the snapshot.
#[test]
fn snapshot_does_not_see_later_insertions() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_edge(a, b, ());

    let snap = reachability_snapshot(&g, a, [a, b]).unwrap();
    assert_eq!(snap.reachable_nodes().count(), 2);

    // Insert a new node and edge after the snapshot was built.
    let c = g.add_node("c");
    g.add_edge(b, c, ());

    // The snapshot still only knows about a and b.
    assert_eq!(snap.reachable_nodes().count(), 2);
    assert!(!snap.is_reachable(c));
    assert!(snap.path_to(c).is_none());
}

/// The start node path is a single-node path with no edges.
#[test]
fn path_to_start_is_trivial() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_edge(a, b, ());

    let snap = reachability_snapshot(&g, a, [a, b]).unwrap();
    let path = snap.path_to(a).unwrap();
    assert_eq!(path.nodes(), &[a]);
    assert!(path.edges().is_empty());
    assert!(path.is_empty());
}

/// A longer path is reconstructed in order with the right edges.
#[test]
fn longer_path_reconstruction() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    let d = g.add_node("d");
    let e1 = g.add_edge(a, b, ());
    let e2 = g.add_edge(b, c, ());
    let e3 = g.add_edge(c, d, ());

    let snap = reachability_snapshot(&g, a, [a, b, c, d]).unwrap();
    let path = snap.path_to(d).unwrap();
    assert_eq!(path.nodes(), &[a, b, c, d]);
    assert_eq!(path.edges(), &[e1, e2, e3]);
}

/// The allowed and reachable node iterators yield the right sets.
#[test]
fn node_iterators_yield_correct_sets() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    let d = g.add_node("d");
    g.add_edge(a, b, ());
    g.add_edge(b, c, ());
    // d is allowed but disconnected.

    let snap = reachability_snapshot(&g, a, [a, b, c, d]).unwrap();

    let allowed: Vec<NodeIndex> = snap.allowed_nodes().collect();
    assert_eq!(allowed, vec![a, b, c, d]);

    let reachable: Vec<NodeIndex> = snap.reachable_nodes().collect();
    assert_eq!(reachable, vec![a, b, c]);
}

/// Determinism: building the same snapshot twice yields the same result.
#[test]
fn snapshot_is_deterministic() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    g.add_edge(a, b, ());
    g.add_edge(a, c, ());
    g.add_edge(b, c, ());

    let s1 = reachability_snapshot(&g, a, [a, b, c]).unwrap();
    let s2 = reachability_snapshot(&g, a, [a, b, c]).unwrap();

    let p1 = s1.path_to(c).unwrap();
    let p2 = s2.path_to(c).unwrap();
    assert_eq!(p1, p2);
    // The direct edge a->c (inserted second) beats the longer route.
    assert_eq!(p1.nodes(), &[a, c]);
}

/// Works with `StableGraph` as well.
#[test]
fn works_with_stable_graph() {
    let mut g = StableGraph::<&str, ()>::new();
    let a = g.add_node("a");
    let secret = g.add_node("secret");
    let public = g.add_node("public");
    let b = g.add_node("b");
    g.add_edge(a, secret, ());
    g.add_edge(secret, b, ());
    g.add_edge(a, public, ());
    g.add_edge(public, b, ());

    let snap = reachability_snapshot(&g, a, [a, public, b]).unwrap();
    assert!(!snap.is_reachable(secret));
    let path = snap.path_to(b).unwrap();
    assert_eq!(path.nodes(), &[a, public, b]);
}
