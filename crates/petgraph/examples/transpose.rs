//! Transpose a directed graph: reverse all edges.
//!
//! This example demonstrates the [`transpose`](petgraph::transpose::transpose)
//! function. It builds a small directed graph with parallel edges and a
//! self-loop, transposes it, and prints the edge mappings.

use petgraph::graph::Graph;
use petgraph::transpose::transpose;

fn main() {
    let mut g: Graph<&str, &str> = Graph::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let e1 = g.add_edge(a, b, "e1");
    let e2 = g.add_edge(a, b, "e2"); // parallel edge
    let e3 = g.add_edge(b, b, "e3"); // self-loop

    let result = transpose(&g).expect("transpose should not fail");
    let t = &result.graph;

    println!("Original graph: {} nodes, {} edges", g.node_count(), g.edge_count());
    println!("Transposed graph: {} nodes, {} edges", t.node_count(), t.edge_count());

    let new_a = result.get_node(a).unwrap();
    let new_b = result.get_node(b).unwrap();

    // The parallel edges are preserved: now both go from b to a.
    let b_to_a: Vec<_> = t.edges_connecting(new_b, new_a).map(|e| *e.weight()).collect();
    println!("Edges from b to a after transpose: {b_to_a:?}");

    // The self-loop is preserved.
    let self_loops: Vec<_> = t.edges_connecting(new_b, new_b).map(|e| *e.weight()).collect();
    println!("Self-loops after transpose: {self_loops:?}");

    // Print the edge mappings one by one.
    println!("\nEdge mappings (original -> transposed):");
    for old in [e1, e2, e3] {
        let new = result.get_edge(old).unwrap();
        let (s, tgt) = t.edge_endpoints(new).unwrap();
        println!("  edge {old:?} -> {new:?}: {s:?} -> {tgt:?}");
    }
}
