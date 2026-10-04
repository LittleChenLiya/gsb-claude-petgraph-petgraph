//! Demo: read-only failure impact analysis on a 6-node network.
//!
//! This example shows how to use [`failure_impact`] to analyze which targets become
//! unreachable when each candidate node is (virtually) removed from the graph.
//!
//! The graph has 6 nodes:
//! - `s`: the start node (server)
//! - `a`, `b`: intermediate nodes (routers)
//! - `t`: the key target
//! - `c`, `d`: isolated nodes (unreachable from `s`)
//!
//! Graph structure:
//! ```text
//! s -> a -> t
//! s -> b -> t
//!
//! c    d   (isolated, unreachable from s)
//! ```
//!
//! Running this demo:
//! ```bash
//! cargo run --example failure_impact_demo
//! ```

use petgraph::{Graph, algo::failure_impact};

fn main() {
    // Build the graph.
    let mut g = Graph::<&str, ()>::new();
    let s = g.add_node("s");
    let a = g.add_node("a");
    let b = g.add_node("b");
    let t = g.add_node("t");
    let c = g.add_node("c");
    let d = g.add_node("d");

    g.add_edge(s, a, ());
    g.add_edge(a, t, ());
    g.add_edge(s, b, ());
    g.add_edge(b, t, ());
    // c and d are isolated (no edges to the rest of the graph).

    let targets = &[t, c, d];
    let candidates = &[a, b, s, t];

    println!("Graph:");
    println!("  s -> a -> t");
    println!("  s -> b -> t");
    println!("  c    d   (isolated)");
    println!();

    let impacts = failure_impact(&g, s, targets, candidates);

    for impact in &impacts {
        println!("Removing node {:?}:", impact.candidate);

        if impact.lost_targets.is_empty() {
            println!("  No targets lost reachability.");
        } else {
            for lost in &impact.lost_targets {
                let path_str = lost
                    .original_path
                    .iter()
                    .map(|n| format!("{:?}", n))
                    .collect::<Vec<_>>()
                    .join(" -> ");
                println!(
                    "  Lost target {:?} (original path: {})",
                    lost.target, path_str
                );
            }
        }

        if !impact.baseline_unreachable.is_empty() {
            let unreachable: Vec<_> = impact
                .baseline_unreachable
                .iter()
                .map(|n| format!("{:?}", n))
                .collect();
            println!("  Baseline unreachable: {}", unreachable.join(", "));
        }

        println!();
    }

    // Summary of expected results:
    // - Removing a: t still reachable via b; c, d baseline unreachable.
    // - Removing b: t still reachable via a; c, d baseline unreachable.
    // - Removing s: t lost; c, d baseline unreachable.
    // - Removing t: t lost (itself); c, d baseline unreachable.
    println!("Summary:");
    println!("  Removing a or b does NOT disconnect t (redundant paths).");
    println!("  Removing s disconnects t.");
    println!("  Removing t makes t unreachable (itself).");
    println!("  c and d are always baseline unreachable (not counted as new failures).");
}
