//! DFA-constrained shortest path example.
//!
//! Builds a small labelled graph and an automaton accepting the label
//! sequence "first a, then b", finds the cheapest accepted path, prints the
//! (node, automaton state) path with its cost, and then demonstrates how
//! unsolvable and invalid inputs are reported. Runs in well under 90 seconds
//! (it is effectively instantaneous).

use petgraph::Graph;
use petgraph::algo::dfa_shortest_path::{Dfa, DfaPathError, dfa_shortest_path};
use petgraph::graph::{EdgeIndex, NodeIndex};

fn a_then_b_dfa() -> Dfa<u32, char> {
    let mut dfa = Dfa::new(0);
    dfa.add_transition(0, 'x', 0); // padding loops in the start state
    dfa.add_transition(0, 'a', 1);
    dfa.add_transition(1, 'a', 1);
    dfa.add_transition(1, 'b', 2);
    dfa.add_accepting(2);
    dfa
}

fn cost_of(label: char) -> u64 {
    match label {
        'x' => 1, // the tempting cheap shortcut
        _ => 5,
    }
}

/// Replay the edge id sequence to print the (node, automaton state) path.
fn print_state_path(g: &Graph<(), char>, dfa: &Dfa<u32, char>, start: NodeIndex, edges: &[EdgeIndex]) {
    let mut node = start;
    let mut state = *dfa.start();
    let mut total = 0;
    println!("  (node {:?}, automaton state {}) [start]", node.index(), state);
    for &eid in edges {
        let label = *g.edge_weight(eid).unwrap();
        let (_src, dst) = g.edge_endpoints(eid).unwrap();
        state = *dfa.transition(&state, &label).expect("accepted label");
        total += cost_of(label);
        node = dst;
        println!(
            "  (node {:?}, automaton state {}) via edge {:?} labelled '{}' (cost {}, total {})",
            node.index(),
            state,
            eid.index(),
            label,
            cost_of(label),
            total
        );
    }
    println!("  accepting state reached: {}", dfa.is_accepting(&state));
}

fn main() {
    let dfa = a_then_b_dfa();

    // --- Scenario 1: a solvable graph -------------------------------------
    let mut g: Graph<(), char> = Graph::new();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, t, 'x'); // cheap (cost 1) but rejected by the automaton
    g.add_edge(s, m, 'a');
    g.add_edge(m, t, 'b');

    println!("Scenario 1: cheapest accepted path (labels must be 'a' then 'b')");
    match dfa_shortest_path(&g, s, &dfa, |e| cost_of(*e.weight())) {
        Ok(Some(path)) => {
            println!("found path with total cost {}:", path.cost);
            print_state_path(&g, &dfa, s, &path.edges);
        }
        Ok(None) => println!("no path accepted by the automaton exists"),
        Err(e) => println!("search failed: {e}"),
    }

    // --- Scenario 2: no accepted path --------------------------------------
    let mut g2: Graph<(), char> = Graph::new();
    let s2 = g2.add_node(());
    let t2 = g2.add_node(());
    g2.add_edge(s2, t2, 'a'); // 'b' is missing: the automaton can never accept

    println!();
    println!("Scenario 2: graph without any 'b' edge");
    match dfa_shortest_path(&g2, s2, &dfa, |e| cost_of(*e.weight())) {
        Ok(Some(path)) => println!("found path with total cost {}", path.cost),
        Ok(None) => println!(
            "no solution: no path from the start node produces a label sequence \
             accepted by the automaton (the accepting state is unreachable)"
        ),
        Err(e) => println!("search failed: {e}"),
    }

    // --- Scenario 3: invalid start node ------------------------------------
    println!();
    println!("Scenario 3: start node that does not exist");
    let missing = NodeIndex::new(999);
    match dfa_shortest_path(&g, missing, &dfa, |e| cost_of(*e.weight())) {
        Err(DfaPathError::InvalidStart { node, state }) => println!(
            "no solution: start node {:?} is not part of the graph \
             (automaton start state {})",
            node.index(),
            state
        ),
        other => println!("unexpected result: {other:?}"),
    }
}

