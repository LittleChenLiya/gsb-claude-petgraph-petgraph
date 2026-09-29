//! Example: shortest path constrained by a DFA.
//!
//! Run with:
//!
//! `text
//! cargo run -p petgraph --example dfa_shortest_path
//! `
//!
//! It builds a small graph whose edges carry a label and a non-negative
//! integer cost, searches for the cheapest path whose label sequence is
//! accepted by a "first 'a', then 'b'" automaton, prints the resulting state
//! path, and then demonstrates how invalid inputs and the absence of a valid
//! path are reported.

use petgraph::{
    Graph,
    algo::dfa_shortest_path::{Dfa, dfa_shortest_path},
    prelude::*,
};

fn main() {
    // A DFA accepting exactly the sequences "a...a b...b" with at least one
    // 'a' and one 'b': first some a's, then some b's.
    let dfa = Dfa::new(
        3,
        0,
        [2],
        [(0, 'a', 1), (1, 'a', 1), (1, 'b', 2), (2, 'b', 2)],
    )
    .expect("the automaton is well-formed");

    let mut graph: Graph<(), (char, u64)> = Graph::new();
    let s = graph.add_node(());
    let t = graph.add_node(());
    let u = graph.add_node(());
    let v = graph.add_node(());
    // A cheap but illegal direct edge ('b' cannot start the sequence).
    graph.add_edge(s, t, ('b', 1));
    // Legal detours.
    graph.add_edge(s, u, ('a', 2));
    graph.add_edge(u, t, ('b', 2));
    graph.add_edge(s, v, ('a', 1));
    graph.add_edge(v, v, ('a', 0)); // zero-cost self-loop
    graph.add_edge(v, u, ('b', 3));

    println!(
        "DFA: {} states, start q{}, accepting q2",
        dfa.num_states(),
        dfa.start_state()
    );
    println!("Looking for the cheapest accepted path from {:?}...", s);

    match dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight()) {
        Ok(Some(path)) => {
            println!("Found a path with total cost {}:", path.cost);
            let mut node = s;
            let mut state = dfa.start_state();
            println!("  start at {:?}, DFA state q{}", node, state);
            let mut accumulated = 0;
            for &edge_id in &path.edges {
                let (source, target) = graph.edge_endpoints(edge_id).unwrap();
                let &(label, cost) = graph.edge_weight(edge_id).unwrap();
                state = dfa.transition(state, &label).unwrap();
                accumulated += cost;
                println!(
                    "  take {:?} ({:?} -> {:?}) labeled {:?} cost {} -> DFA state q{}",
                    edge_id, source, target, label, cost, state
                );
                node = target;
            }
            println!(
                "  arrived at {:?}; DFA state q{} accepting: {}; re-accumulated cost: {}",
                node,
                state,
                dfa.is_accepting(state),
                accumulated
            );
        }
        Ok(None) => println!("No valid path: no label sequence from the start node is accepted."),
        Err(error) => println!("Search failed: {error}"),
    }

    // No-solution case: only 'b' edges leave `w`, so the DFA can never reach
    // its accepting state.
    let w = graph.add_node(());
    graph.add_edge(w, t, ('b', 1));
    match dfa_shortest_path(&graph, w, &dfa, |edge| *edge.weight()) {
        Ok(Some(path)) => println!("Unexpected path from {:?}: {:?}", w, path.edges),
        Ok(None) => println!(
            "From {:?}: no solution - no path whose labels are accepted by the DFA exists.",
            w
        ),
        Err(error) => println!("From {:?}: search failed: {error}", w),
    }

    // Invalid input case: a start node that does not exist in the graph.
    let missing = NodeIndex::new(999);
    match dfa_shortest_path(&graph, missing, &dfa, |edge| *edge.weight()) {
        Ok(_) => println!("Unexpected success from a missing start node."),
        Err(error) => println!("From {:?}: search rejected - {error}", missing),
    }
}
