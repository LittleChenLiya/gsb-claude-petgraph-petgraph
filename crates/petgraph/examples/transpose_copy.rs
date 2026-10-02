//! Transpose-copy a directed graph: reverse every edge into a new graph
//! while keeping full traceability through node/edge index maps.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example transpose_copy
//! ```

use petgraph::graph::DiGraph;
use petgraph::stable_graph::StableDiGraph;
use petgraph::transpose::{transpose_copy, transpose_stable_copy};

fn main() {
    println!("=== Graph: parallel edges + self-loop ===");

    let mut g = DiGraph::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_edge(a, b, "ab-1"); // parallel edge 1
    g.add_edge(a, b, "ab-2"); // parallel edge 2
    g.add_edge(b, b, "bb-loop"); // self-loop on b

    let t = transpose_copy(&g).expect("directed graph transposes fine");

    println!("node map (old -> new):");
    for n in g.node_indices() {
        println!("  {:?} ({}) -> {:?}", n, g[n], t.node_map[n.index()]);
    }
    println!("edge map (old -> new, direction reversed):");
    for e in g.edge_indices() {
        let (os, ot) = g.edge_endpoints(e).unwrap();
        let ne = t.edge_map[e.index()];
        let (ns, nt) = t.graph.edge_endpoints(ne).unwrap();
        println!(
            "  {:?}: {:?} -> {:?} ({})  =>  {:?}: {:?} -> {:?} ({})",
            e, os, ot, g[e], ne, ns, nt, t.graph[ne]
        );
    }
    assert_eq!(t.graph.edge_count(), 3); // nothing merged
    assert_eq!(
        t.graph
            .edges_connecting(t.node_map[b.index()], t.node_map[a.index()])
            .count(),
        2
    );
    assert_eq!(
        t.graph
            .edges_connecting(t.node_map[b.index()], t.node_map[b.index()])
            .count(),
        1
    );

    println!();
    println!("=== StableGraph: transpose after removing a node ===");

    let mut sg = StableDiGraph::new();
    let x = sg.add_node("x");
    let y = sg.add_node("y"); // will be removed
    let z = sg.add_node("z");
    sg.add_edge(x, y, "xy");
    sg.add_edge(y, z, "yz");
    sg.add_edge(x, z, "xz");
    sg.remove_node(y);

    let st = transpose_stable_copy(&sg).expect("directed graph transposes fine");

    println!("node map (old slot -> new node, removed slots are None):");
    for (slot, mapped) in st.node_map.iter().enumerate() {
        println!("  slot {slot} -> {mapped:?}");
    }
    assert_eq!(st.node_map[y.index()], None); // removed node is not in the map
    println!("edge map (old slot -> new edge):");
    for (slot, mapped) in st.edge_map.iter().enumerate() {
        println!("  slot {slot} -> {mapped:?}");
    }
    assert_eq!(st.graph.node_count(), 2);
    assert_eq!(st.graph.edge_count(), 1); // only x -> z survived, now z -> x
    let nx = st.node_map[x.index()].unwrap();
    let nz = st.node_map[z.index()].unwrap();
    assert_eq!(st.graph.edges_connecting(nz, nx).count(), 1);

    println!();
    println!("OK: transposed copies built, original graphs untouched.");
}
