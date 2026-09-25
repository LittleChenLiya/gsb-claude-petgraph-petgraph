//! Batch-by-batch simple cycle enumeration demo.
//!
//! Enumerates the simple directed cycles of a triangle with a parallel edge
//! and a self loop, printing every batch together with its paused/done
//! status. Finishes in far less than 90 seconds.
//!
//! Run with: `cargo run --example simple_cycles`

use petgraph::{
    algo::{CycleStatus, SimpleCycleEnumerator},
    prelude::*,
};

fn main() {
    // Triangle a -> b -> c -> a, a parallel edge b -> c, and a self loop a -> a.
    let mut graph = DiGraph::<&str, ()>::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let c = graph.add_node("c");
    graph.extend_with_edges([(a, b), (b, c), (b, c), (c, a), (a, a)]);

    let mut enumerator = SimpleCycleEnumerator::new(&graph);
    let mut batch_no = 0;
    loop {
        // At most 1 cycle and at most 3 edge visits per batch.
        let batch = enumerator.next_batch(1, 3);
        batch_no += 1;
        let status = match batch.status {
            CycleStatus::Paused => "paused",
            CycleStatus::Done => "done",
        };
        if batch.cycles.is_empty() {
            println!(
                "batch {batch_no}: {status} (no new cycle, {} edge visits)",
                batch.edge_visits
            );
        }
        for cycle in &batch.cycles {
            let edges: Vec<usize> = cycle.iter().map(|e| e.index()).collect();
            println!(
                "batch {batch_no}: {status} cycle {edges:?} ({} edge visits)",
                batch.edge_visits
            );
        }
        if batch.status == CycleStatus::Done {
            break;
        }
    }
    println!("finished after {batch_no} batches");
}
