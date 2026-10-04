use petgraph::algo::allowed_reachability::{AllowedReachability, AllowedReachabilityError};
use petgraph::graph::{Graph, UnGraph};
use petgraph::stable_graph::StableGraph;

/// a -> secret -> b and a -> public -> b; banning `secret` must leave only
/// the path through `public`.
#[test]
fn banned_node_only_public_path() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let secret = g.add_node("secret");
    let public = g.add_node("public");
    let b = g.add_node("b");

    let e_as = g.add_edge(a, secret, ());
    let e_sb = g.add_edge(secret, b, ());
    let e_ap = g.add_edge(a, public, ());
    let e_pb = g.add_edge(public, b, ());
    let _ = (e_as, e_sb);

    let snap = AllowedReachability::new(&g, a, [a, public, b]).unwrap();

    assert!(snap.is_reachable(a));
    assert!(!snap.is_reachable(secret));
    assert!(snap.is_reachable(public));
    assert!(snap.is_reachable(b));
    assert_eq!(snap.reachable_len(), 3);

    // Only the public path is reported, with its concrete edge ids.
    assert_eq!(snap.path_to(b), Some(vec![(e_ap, public), (e_pb, b)]));
    assert_eq!(snap.edge_path_to(b), Some(vec![e_ap, e_pb]));
    assert_eq!(snap.path_to(secret), None);
    assert_eq!(snap.path_to(a), Some(vec![]));
}

/// A snapshot built before the original graph is modified still answers
/// with the original paths and edge ids.
#[test]
fn snapshot_survives_graph_mutation() {
    let mut g = StableGraph::<&str, ()>::new();
    let a = g.add_node("a");
    let secret = g.add_node("secret");
    let public = g.add_node("public");
    let b = g.add_node("b");

    g.add_edge(a, secret, ());
    g.add_edge(secret, b, ());
    let e_ap = g.add_edge(a, public, ());
    let e_pb = g.add_edge(public, b, ());

    let snap = AllowedReachability::new(&g, a, [a, public, b]).unwrap();

    // Remove the public edges from the original graph after the snapshot
    // was built: the snapshot must still report the original path.
    g.remove_edge(e_ap);
    g.remove_edge(e_pb);
    assert_eq!(snap.path_to(b), Some(vec![(e_ap, public), (e_pb, b)]));
    assert!(snap.is_reachable(b));

    // Newly inserted nodes/edges are likewise invisible to the snapshot.
    let c = g.add_node("c");
    g.add_edge(a, c, ());
    assert!(!snap.is_reachable(c));
    assert_eq!(snap.path_to(c), None);
    assert_eq!(snap.reachable_nodes(), &[a, public, b]);
}

/// Parallel edges between the same endpoints stay distinct: the recorded
/// path carries the exact `EdgeId` of the traversed edge.
#[test]
fn multi_edges_keep_concrete_edge_ids() {
    let mut g = Graph::<(), ()>::new();
    let a = g.add_node(());
    let b = g.add_node(());

    let e1 = g.add_edge(a, b, ());
    let e2 = g.add_edge(a, b, ());
    assert_ne!(e1, e2);

    let snap = AllowedReachability::new(&g, a, [a, b]).unwrap();

    // The first inserted edge wins the tie, and it is reported by its id,
    // not merely by its endpoints.
    assert_eq!(snap.edge_path_to(b), Some(vec![e1]));

    // Even after the original graph loses that edge, the snapshot keeps
    // referring to the original edge identity.
    g.remove_edge(e1);
    assert_eq!(snap.edge_path_to(b), Some(vec![e1]));
    let _ = e2;
}

/// Nodes with no allowed path between them report no path.
#[test]
fn no_path_reports_none() {
    let mut g = Graph::<(), ()>::new();
    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());
    g.add_edge(a, b, ());
    // `c` is disconnected; `b` is not allowed either.

    let snap = AllowedReachability::new(&g, a, [a, c]).unwrap();
    assert!(!snap.is_reachable(b));
    assert!(!snap.is_reachable(c));
    assert_eq!(snap.path_to(b), None);
    assert_eq!(snap.path_to(c), None);
    assert_eq!(snap.reachable_nodes(), &[a]);
}

/// Vacant `StableGraph` slots (removed nodes) in the allowed set or as the
/// start node are reported as errors, and traversal over graphs with holes
/// still works.
#[test]
fn stable_graph_vacant_slots() {
    let mut g = StableGraph::<(), ()>::new();
    let a = g.add_node(());
    let gone = g.add_node(());
    let b = g.add_node(());
    let e_ab = g.add_edge(a, b, ());
    g.remove_node(gone);

    // A removed node index in the allowed set is an error.
    assert_eq!(
        AllowedReachability::new(&g, a, [a, gone, b]).unwrap_err(),
        AllowedReachabilityError::UnknownNode(gone)
    );
    // A removed node index as the start node is an error.
    assert_eq!(
        AllowedReachability::new(&g, gone, [a, b]).unwrap_err(),
        AllowedReachabilityError::UnknownNode(gone)
    );

    // The graph still has a vacant slot; traversal is unaffected.
    let snap = AllowedReachability::new(&g, a, [a, b]).unwrap();
    assert_eq!(snap.path_to(b), Some(vec![(e_ab, b)]));
}

/// A start node outside the allowed set is explicitly rejected.
#[test]
fn start_not_allowed_is_rejected() {
    let mut g = Graph::<(), ()>::new();
    let a = g.add_node(());
    let b = g.add_node(());
    g.add_edge(a, b, ());

    assert_eq!(
        AllowedReachability::new(&g, a, [b]).unwrap_err(),
        AllowedReachabilityError::StartNotAllowed(a)
    );
}

/// Ties between equally short paths are resolved by node/edge insertion
/// order.
#[test]
fn insertion_order_breaks_ties() {
    let mut g = Graph::<(), ()>::new();
    let s = g.add_node(());
    let p = g.add_node(());
    let q = g.add_node(());
    let t = g.add_node(());

    let e_sp = g.add_edge(s, p, ());
    let _e_sq = g.add_edge(s, q, ());
    let e_pt = g.add_edge(p, t, ());
    let _e_qt = g.add_edge(q, t, ());

    let snap = AllowedReachability::new(&g, s, [s, p, q, t]).unwrap();
    // `p` was inserted before `q`, so the path to `t` goes through `p`.
    assert_eq!(snap.path_to(t), Some(vec![(e_sp, p), (e_pt, t)]));
    assert_eq!(snap.reachable_nodes(), &[s, p, q, t]);
}

/// Undirected graphs are traversed in both directions along allowed edges.
#[test]
fn undirected_graph_traversal() {
    let mut g = UnGraph::<(), ()>::new_undirected();
    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());
    let e_ab = g.add_edge(a, b, ());
    let e_bc = g.add_edge(b, c, ());

    let snap = AllowedReachability::new(&g, c, [c, b, a]).unwrap();
    assert_eq!(snap.path_to(a), Some(vec![(e_bc, b), (e_ab, a)]));
}

/// Paths that would have to cross a forbidden node are not allowed paths,
/// even when the forbidden node itself is reachable in the original graph.
#[test]
fn forbidden_intermediate_node_blocks_path() {
    let mut g = Graph::<(), ()>::new();
    let a = g.add_node(());
    let mid = g.add_node(());
    let b = g.add_node(());
    g.add_edge(a, mid, ());
    g.add_edge(mid, b, ());

    let snap = AllowedReachability::new(&g, a, [a, b]).unwrap();
    assert!(!snap.is_reachable(mid));
    assert!(!snap.is_reachable(b));
    assert_eq!(snap.path_to(b), None);
}
