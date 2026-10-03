//! Print the topological layers of a directed graph, one batch at a time.
//!
//! A layer is a set of nodes whose predecessors have all already been emitted
//! in earlier layers; nodes within a layer are listed in original node-index
//! order. If the graph contains a cycle, the iterator stops and reports a
//! witness of the actual closed loop instead of an empty tail batch.
//!
//! Run with:
//! ```text
//! cargo run -p petgraph --example topological_layers
//! ```

use petgraph::{Graph, algo::topological_layers, prelude::*};

fn main() {
    // a and b both point to c, c points to d.
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.extend_with_edges([(a, c), (b, c), (c, d)]);

    println!("DAG layers (a,b -> c -> d):");
    for (i, layer) in topological_layers(&graph).enumerate() {
        match layer {
            Ok(nodes) => println!("  layer {i}: {nodes:?}"),
            Err(cycle) => {
                println!("  layer {i}: cycle detected! witness = {:?}", cycle.cycle());
                break;
            }
        }
    }

    // A graph with a directed cycle: a -> b -> c -> a.
    let mut cyclic: Graph<(), (), Directed> = Graph::new();
    let a = cyclic.add_node(());
    let b = cyclic.add_node(());
    let c = cyclic.add_node(());
    cyclic.extend_with_edges([(a, b), (b, c), (c, a)]);

    println!("\nCyclic graph layers (a -> b -> c -> a):");
    for (i, layer) in topological_layers(&cyclic).enumerate() {
        match layer {
            Ok(nodes) => println!("  layer {i}: {nodes:?}"),
            Err(cycle) => {
                println!("  layer {i}: cycle detected! witness = {:?}", cycle.cycle());
                break;
            }
        }
    }
}
