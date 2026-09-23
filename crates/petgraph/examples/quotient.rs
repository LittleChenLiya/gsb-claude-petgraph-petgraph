//! Build the quotient graph of a StableGraph (with parallel edges,
//! self-loops and a hole left by a removed node) from an explicit partition
//! and print the full bidirectional mapping.
//!
//! Run with: cargo run -p petgraph --example quotient

extern crate alloc;

use alloc::collections::BTreeMap;

use petgraph::Directed;
use petgraph::algo::quotient::{Quotient, SelfLoopPolicy, quotient_graph};
use petgraph::prelude::*;
use petgraph::stable_graph::StableGraph;

fn main() {
    let mut g = StableGraph::<(), i32, Directed>::new();
    let a = g.add_node(()); // 0
    let hole = g.add_node(()); // 1, removed below
    let b = g.add_node(()); // 2
    let c = g.add_node(()); // 3
    g.add_edge(a, a, 10); // e0: self-loop
    g.add_edge(a, b, 1); // e1: intra-group
    g.add_edge(b, c, 2); // e2
    g.add_edge(b, c, 3); // e3: parallel to e2
    g.add_edge(c, b, 5); // e4: reverse direction
    g.remove_node(hole);

    let partition: BTreeMap<NodeIndex, char> = [(a, 'x'), (b, 'x'), (c, 'y')].into_iter().collect();

    let q: Quotient<NodeIndex, EdgeIndex, char, i32, Directed> =
        quotient_graph(&g, &partition, SelfLoopPolicy::Keep, |edges| {
            Ok::<_, String>(edges.iter().map(|e| *e.weight()).sum())
        })
        .expect("partition covers every live node");

    println!("quotient nodes (label order):");
    for (qi, members) in q.groups.iter().enumerate() {
        let label = q.graph[NodeIndex::new(qi)];
        println!("  {label} -> quotient node {qi}, members {members:?}");
    }

    println!("node_map (input index -> quotient node, None = hole):");
    for (i, mapped) in q.node_map.iter().enumerate() {
        println!("  {i} -> {mapped:?}");
    }

    println!("quotient edges (weight, source EdgeIndex list):");
    for (ei, sources) in q.edge_sources.iter().enumerate() {
        let edge = EdgeIndex::new(ei);
        let (s, t) = q.graph.edge_endpoints(edge).unwrap();
        println!(
            "  {} -> {}: weight {}, from {:?}",
            s.index(),
            t.index(),
            q.graph[edge],
            sources
        );
    }
}
