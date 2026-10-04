//! Short demo of `AllowedReachability`: a reusable reachability snapshot
//! over the allowed subset of a graph.
//!
//! Run with: `cargo run -p petgraph --example allowed_reachability`

use petgraph::algo::allowed_reachability::AllowedReachability;
use petgraph::graph::Graph;

fn main() {
    // a -> secret -> b and a -> public -> b
    let mut g = Graph::<&str, ()>::new();
    let a = g.add_node("a");
    let secret = g.add_node("secret");
    let public = g.add_node("public");
    let b = g.add_node("b");

    g.add_edge(a, secret, ());
    g.add_edge(secret, b, ());
    g.add_edge(a, public, ());
    g.add_edge(public, b, ());

    // Ban `secret`: build the snapshot once, query it many times.
    let snap = AllowedReachability::new(&g, a, [a, public, b]).unwrap();

    println!("reachable: {:?}", snap.reachable_nodes());
    println!("secret reachable? {}", snap.is_reachable(secret));
    println!("path a -> b: {:?}", snap.path_to(b));

    // The snapshot is owned: mutating the original graph afterwards does
    // not change the answers.
    let public_edge = g.find_edge(a, public).unwrap();
    g.remove_edge(public_edge);
    println!("after removing a -> public from the graph:");
    println!("path a -> b (snapshot): {:?}", snap.path_to(b));
}
