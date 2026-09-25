//! Example: pausable enumeration of all simple directed cycles.
//!
//! Enumerates the simple cycles of a graph batch by batch, printing each
//! cycle together with the "paused"/"done" status of every batch. The
//! enumeration is driven by a result limit and an edge-visit budget per
//! batch; a paused enumeration is resumed from its cursor without
//! rescanning any edge.
//!
//! Run with: "cargo run -p petgraph --example pausable_simple_cycles"

use petgraph::{
    algo::{CycleBatchStatus, SimpleCycleEnumerator},
    prelude::*,
};

fn main() {
    // Triangle a -> b -> c -> a, a parallel a -> b edge, a self-loop on a,
    // plus a second cycle d -> e -> d sharing no nodes with the triangle.
    let mut graph = DiGraph::<&str, ()>::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let c = graph.add_node("c");
    let d = graph.add_node("d");
    let e = graph.add_node("e");
    graph.extend_with_edges([
        (a, b),
        (b, c),
        (c, a),
        (a, b), // parallel edge: forms its own distinct edge cycle
        (a, a), // self-loop
        (d, e),
        (e, d),
    ]);

    let mut enumerator = SimpleCycleEnumerator::new(&graph);
    let mut cursor = enumerator.cursor();

    const MAX_RESULTS_PER_BATCH: usize = 2;
    const EDGE_VISIT_BUDGET: usize = 3;

    let mut total_cycles = 0;
    let mut total_edge_visits = 0;
    for batch_no in 1.. {
        let batch = enumerator
            .next_batch(
                &graph,
                &cursor,
                MAX_RESULTS_PER_BATCH,
                EDGE_VISIT_BUDGET,
            )
            .expect("the graph is unchanged while enumerating");

        let status = match batch.status {
            CycleBatchStatus::Paused => "paused",
            CycleBatchStatus::Done => "done",
        };
        println!(
            "batch {batch_no}: {} cycle(s), {} edge visit(s) -> {status}",
            batch.cycles.len(),
            batch.edge_visits,
        );
        for cycle in &batch.cycles {
            let nodes: Vec<_> = cycle
                .iter()
                .map(|&edge| graph.edge_endpoints(edge).unwrap().0)
                .map(|node| graph[node])
                .collect();
            println!("  cycle {cycle:?} through nodes {nodes:?}");
        }

        total_cycles += batch.cycles.len();
        total_edge_visits += batch.edge_visits;
        cursor = batch.cursor;
        if batch.status == CycleBatchStatus::Done {
            break;
        }
    }

    println!(
        "enumeration finished: {total_cycles} cycle(s), {total_edge_visits} total edge visit(s)"
    );
}
