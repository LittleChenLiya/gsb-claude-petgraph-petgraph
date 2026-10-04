//! Demonstration of the read-only failure impact analyzer.
//!
//! Run with:
//! ```text
//! cargo run -p petgraph --example failure_impact_demo
//! ```
//!
//! The graph below has 6 nodes. `u` is an isolated node, so it is reported as
//! baseline-unreachable rather than as a new failure of any removal. Node `x` shows
//! that a self-loop (a cycle) never creates new reachability.
//!
//! ```text
//!       a -> t
//!     /       ^
//!   s          |
//!     \       /
//!       b ---
//!
//!   s -> x  (x -> x is a self-loop)
//!
//!   u        (isolated)
//! ```

use petgraph::{
    algo::failure_impact,
    graph::{DiGraph, NodeIndex},
};

const LABELS: [char; 6] = ['s', 'a', 'b', 't', 'x', 'u'];

/// Label a node index, falling back to a placeholder for indexes outside the graph.
fn label_safe(index: NodeIndex) -> String {
    LABELS
        .get(index.index())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("node#{}", index.index()))
}

fn main() {
    // Nodes: 0 = s, 1 = a, 2 = b, 3 = t, 4 = x, 5 = u.
    let graph = DiGraph::<(), ()>::from_edges([
        (0, 1), // s -> a
        (1, 3), // a -> t
        (0, 2), // s -> b
        (2, 3), // b -> t
        (0, 4), // s -> x
        (4, 4), // x -> x (self-loop / cycle)
    ]);

    let start = NodeIndex::new(0);
    let targets = [NodeIndex::new(3), NodeIndex::new(5)]; // t and u
    let candidates = [
        NodeIndex::new(1), // a
        NodeIndex::new(2), // b
        NodeIndex::new(0), // s (the start)
        NodeIndex::new(3), // t (a target itself)
        NodeIndex::new(9), // not present in the graph
    ];

    let report = failure_impact(&graph, start, targets, candidates);

    println!("Failure impact analysis from node `{}`", label_safe(start));
    println!("===========================================");
    for impact in &report.node_impacts {
        let node = label_safe(impact.removed_node);
        if impact.affected_targets.is_empty() {
            println!("remove {node}: no target loses reachability");
        } else {
            for lost in &impact.affected_targets {
                let path = lost
                    .original_path
                    .iter()
                    .map(|node| label_safe(*node))
                    .collect::<Vec<_>>()
                    .join(" -> ");
                println!(
                    "remove {node}: target `{}` becomes unreachable (reachable before: {path})",
                    label_safe(lost.target),
                );
            }
        }
    }

    if !report.baseline_unreachable.is_empty() {
        let unreachable = report
            .baseline_unreachable
            .iter()
            .map(|node| label_safe(*node).to_string())
            .collect::<Vec<_>>()
            .join(", ");
        println!("baseline unreachable (no removal): {unreachable}");
    }
}
