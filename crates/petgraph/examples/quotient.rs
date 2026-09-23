//! Build a quotient graph from an explicit partition of a `StableGraph` with a deleted node,
//! parallel edges and self-loops, and print the complete node/edge correspondence.
//!
//! Run with: `cargo run -p petgraph --example quotient`

use petgraph::{
    Directed, Undirected,
    algo::quotient::{SelfLoopPolicy, quotient_graph},
    graph::DefaultIx,
    stable_graph::StableGraph,
    visit::EdgeRef,
};

fn build<Ty: petgraph::EdgeType>() -> StableGraph<&'static str, i64, Ty> {
    let mut g = StableGraph::default();
    let a = g.add_node("a");
    let b = g.add_node("b"); // removed below, leaving a hole at index 1
    let c = g.add_node("c");
    let d = g.add_node("d");
    g.add_edge(a, c, 1);
    g.add_edge(a, c, 2); // parallel edge
    g.add_edge(c, a, 4); // reverse direction
    g.add_edge(a, a, 8); // self-loop
    g.add_edge(c, d, 16); // intra-group
    g.add_edge(c, d, 32); // intra-group, parallel
    g.remove_node(b);
    g
}

fn partition() -> [(petgraph::graph::NodeIndex, &'static str); 3] {
    [
        (petgraph::graph::NodeIndex::new(0), "z"),
        (petgraph::graph::NodeIndex::new(2), "m"),
        (petgraph::graph::NodeIndex::new(3), "m"),
    ]
}

fn print_quotient<Ty: petgraph::EdgeType>(
    label: &str,
    g: &StableGraph<&'static str, i64, Ty>,
    policy: SelfLoopPolicy,
) {
    let q = quotient_graph::<_, _, i64, _, (), DefaultIx>(g, partition(), policy, |edges| {
        Ok(edges.iter().map(|&e| g[e]).sum::<i64>())
    })
    .unwrap();

    println!("== {label} ({policy:?}) ==");
    println!("quotient nodes (label order):");
    for n in q.graph.node_indices() {
        println!("  {:?} <- label {:?}", n, q.graph[n]);
    }
    println!(
        "original -> quotient (node_map, holes are None): {:?}",
        q.node_map
    );
    println!("quotient -> originals (groups): {:?}", q.groups);
    println!("quotient edges:");
    for (i, e) in q.graph.edge_references().enumerate() {
        println!(
            "  edge {:?}: {:?} -> {:?}, weight {}, sources {:?}",
            i,
            e.source(),
            e.target(),
            e.weight(),
            q.edge_sources[i],
        );
    }
    println!();
}

fn main() {
    let directed: StableGraph<&'static str, i64, Directed> = build();
    let undirected: StableGraph<&'static str, i64, Undirected> = build();

    print_quotient("directed", &directed, SelfLoopPolicy::Keep);
    print_quotient("directed", &directed, SelfLoopPolicy::Drop);
    print_quotient("undirected", &undirected, SelfLoopPolicy::Keep);
    print_quotient("undirected", &undirected, SelfLoopPolicy::Drop);
}
