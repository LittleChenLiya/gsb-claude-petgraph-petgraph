//! Reachability snapshot demo.
//!
//! Builds a small graph with a secret route and a public route from `a` to
//! `b`, then shows how an allowed-node set restricts the recorded path. The
//! snapshot is self-contained: deleting the original edge after the build
//! does not change the recorded path.

use petgraph::reachability::reachability_snapshot;
use petgraph::Graph;

fn main() {
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let secret = g.add_node("secret");
    let public = g.add_node("public");
    let b = g.add_node("b");
    g.extend_with_edges(&[(a, secret), (secret, b), (a, public), (public, b)]);

    // Forbid `secret`; only the public route is recorded.
    let snap = reachability_snapshot(&g, a, [a, public, b]).unwrap();
    let path = snap.path_to(b).unwrap();
    println!("allowed path to b: {:?}", path.nodes());
    assert_eq!(path.nodes(), &[a, public, b]);

    // The snapshot survives later edits to the original graph.
    let public_edge = g.find_edge(a, public).unwrap();
    g.remove_edge(public_edge);
    let path = snap.path_to(b).unwrap();
    println!("path after deleting public edge: {:?}", path.nodes());
    assert_eq!(path.nodes(), &[a, public, b]);
}
