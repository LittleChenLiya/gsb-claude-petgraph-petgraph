use std::collections::HashSet;

use petgraph::{
    algo::{CycleStatus, SimpleCycleEnumerator},
    graph::{DiGraph, EdgeIndex, NodeIndex},
    stable_graph::StableDiGraph,
};

/// Triangle a -> b -> c -> a with a parallel edge b -> c and a self loop a -> a.
///
/// Returns the graph, its nodes and the edge indices in insertion order:
/// `[a->b, b->c (first), b->c (parallel), c->a, a->a]`.
fn triangle_parallel_self_loop() -> (DiGraph<(), ()>, [NodeIndex; 3], [EdgeIndex; 5]) {
    let mut graph = DiGraph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let ab = graph.add_edge(a, b, ());
    let bc1 = graph.add_edge(b, c, ());
    let bc2 = graph.add_edge(b, c, ()); // parallel edge
    let ca = graph.add_edge(c, a, ());
    let aa = graph.add_edge(a, a, ()); // self loop
    (graph, [a, b, c], [ab, bc1, bc2, ca, aa])
}

fn expected_cycles(edges: [EdgeIndex; 5]) -> HashSet<Vec<EdgeIndex>> {
    let [ab, bc1, bc2, ca, aa] = edges;
    [
        vec![aa],          // self loop kept as its own cycle
        vec![ab, bc1, ca], // triangle via the first b->c edge
        vec![ab, bc2, ca], // triangle via the parallel b->c edge
    ]
    .into_iter()
    .collect()
}

fn enumerate_one_shot(graph: &DiGraph<(), ()>) -> (Vec<Vec<EdgeIndex>>, usize) {
    let mut enumerator = SimpleCycleEnumerator::new(graph);
    let batch = enumerator.next_batch(usize::MAX, usize::MAX);
    assert_eq!(batch.status, CycleStatus::Done);
    assert!(batch.cursor.is_done());
    (batch.cycles, batch.edge_visits)
}

#[test]
fn batched_capacity_one_matches_one_shot() {
    let (graph, _, edges) = triangle_parallel_self_loop();
    let (one_shot, one_shot_visits) = enumerate_one_shot(&graph);
    assert_eq!(
        one_shot.iter().cloned().collect::<HashSet<_>>(),
        expected_cycles(edges),
    );

    // Drain the enumerator one cycle at a time.
    let mut enumerator = SimpleCycleEnumerator::new(&graph);
    let mut batched = Vec::new();
    let mut batched_visits = 0;
    loop {
        let batch = enumerator.next_batch(1, usize::MAX);
        assert!(batch.cycles.len() <= 1);
        batched_visits += batch.edge_visits;
        batched.extend(batch.cycles);
        if batch.status == CycleStatus::Done {
            break;
        }
        assert_eq!(batch.status, CycleStatus::Paused);
    }

    // Same cycles, in the same deterministic order, with no repeated work.
    assert_eq!(batched, one_shot);
    assert_eq!(batched_visits, one_shot_visits);
    assert_eq!(enumerator.total_edge_visits(), one_shot_visits);
}

#[test]
fn resume_from_pause_continues_at_position() {
    let (graph, _, _) = triangle_parallel_self_loop();
    let (one_shot, one_shot_visits) = enumerate_one_shot(&graph);

    // Pause after the first cycle, then resume from the cursor.
    let mut enumerator = SimpleCycleEnumerator::new(&graph);
    let first = enumerator.next_batch(1, usize::MAX);
    assert_eq!(first.status, CycleStatus::Paused);
    assert_eq!(first.cycles.len(), 1);

    let mut resumed = SimpleCycleEnumerator::resume(first.cursor, &graph).unwrap();
    let rest = resumed.next_batch(usize::MAX, usize::MAX);
    assert_eq!(rest.status, CycleStatus::Done);

    let mut combined = first.cycles.clone();
    combined.extend(rest.cycles.iter().cloned());
    assert_eq!(combined, one_shot);

    // Resuming must not rescan the already enumerated prefix: the total number
    // of edge visits is exactly the one-shot number of edge visits.
    assert_eq!(first.edge_visits + rest.edge_visits, one_shot_visits);
    assert_eq!(resumed.total_edge_visits(), rest.edge_visits);
}

#[test]
fn resume_from_budget_pause_continues_at_position() {
    let (graph, _, _) = triangle_parallel_self_loop();
    let (one_shot, one_shot_visits) = enumerate_one_shot(&graph);

    // Pause in the middle of the depth-first search via the edge budget.
    let mut enumerator = SimpleCycleEnumerator::new(&graph);
    let first = enumerator.next_batch(usize::MAX, 2);
    assert_eq!(first.status, CycleStatus::Paused);
    assert_eq!(first.edge_visits, 2);

    let mut resumed = SimpleCycleEnumerator::resume(first.cursor, &graph).unwrap();
    let mut rest_cycles = Vec::new();
    let mut rest_visits = 0;
    loop {
        let batch = resumed.next_batch(usize::MAX, 2);
        rest_visits += batch.edge_visits;
        rest_cycles.extend(batch.cycles);
        if batch.status == CycleStatus::Done {
            break;
        }
    }

    let mut combined = first.cycles.clone();
    combined.extend(rest_cycles);
    assert_eq!(combined, one_shot);
    assert_eq!(first.edge_visits + rest_visits, one_shot_visits);
}

#[test]
fn acyclic_graph_is_done_immediately() {
    let graph = DiGraph::<(), ()>::from_edges([(0, 1), (1, 2), (2, 3), (0, 3)]);
    let mut enumerator = SimpleCycleEnumerator::new(&graph);
    let batch = enumerator.next_batch(usize::MAX, usize::MAX);
    assert_eq!(batch.status, CycleStatus::Done);
    assert!(batch.cycles.is_empty());
    assert!(enumerator.is_done());

    // Calling again still reports Done, not Paused.
    let batch = enumerator.next_batch(usize::MAX, usize::MAX);
    assert_eq!(batch.status, CycleStatus::Done);
    assert!(batch.cycles.is_empty());
}

#[test]
fn stable_graph_with_empty_slots() {
    let mut graph = StableDiGraph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.remove_node(b); // leave an empty slot in the middle
    let ac = graph.add_edge(a, c, ());
    let cd = graph.add_edge(c, d, ());
    let da = graph.add_edge(d, a, ());

    let mut enumerator = SimpleCycleEnumerator::new(&graph);
    let batch = enumerator.next_batch(usize::MAX, usize::MAX);
    assert_eq!(batch.status, CycleStatus::Done);
    assert_eq!(batch.cycles, vec![vec![ac, cd, da]]);
}

#[test]
fn tiny_budget_pauses_instead_of_finishing() {
    let (graph, _, _) = triangle_parallel_self_loop();
    let (one_shot, one_shot_visits) = enumerate_one_shot(&graph);

    // A zero edge budget pauses immediately without any progress.
    let mut enumerator = SimpleCycleEnumerator::new(&graph);
    let batch = enumerator.next_batch(usize::MAX, 0);
    assert_eq!(batch.status, CycleStatus::Paused);
    assert!(batch.cycles.is_empty());
    assert_eq!(batch.edge_visits, 0);

    // A zero result capacity pauses immediately as well.
    let batch = enumerator.next_batch(0, usize::MAX);
    assert_eq!(batch.status, CycleStatus::Paused);
    assert!(batch.cycles.is_empty());

    // One edge visit per batch still reaches the exact same cycles.
    let mut batched = Vec::new();
    let mut batched_visits = 0;
    loop {
        let batch = enumerator.next_batch(usize::MAX, 1);
        assert!(batch.edge_visits <= 1);
        batched_visits += batch.edge_visits;
        batched.extend(batch.cycles);
        if batch.status == CycleStatus::Done {
            break;
        }
    }
    assert_eq!(batched, one_shot);
    assert_eq!(batched_visits, one_shot_visits);
}

#[test]
fn resume_after_graph_change_is_rejected() {
    let (graph, nodes, _) = triangle_parallel_self_loop();
    let [a, b, _] = nodes;

    let mut enumerator = SimpleCycleEnumerator::new(&graph);
    let first = enumerator.next_batch(1, usize::MAX);
    assert_eq!(first.status, CycleStatus::Paused);
    let cursor = first.cursor;

    // Resuming against the unchanged graph works.
    assert!(SimpleCycleEnumerator::resume(cursor.clone(), &graph).is_ok());

    // Structural changes are rejected.
    let mut added_edge = graph.clone();
    added_edge.add_edge(b, a, ());
    assert!(SimpleCycleEnumerator::resume(cursor.clone(), &added_edge).is_err());

    let mut removed_edge = graph.clone();
    removed_edge.remove_edge(removed_edge.find_edge(a, b).unwrap());
    assert!(SimpleCycleEnumerator::resume(cursor.clone(), &removed_edge).is_err());

    let mut added_node = graph.clone();
    added_node.add_node(());
    assert!(SimpleCycleEnumerator::resume(cursor.clone(), &added_node).is_err());

    // Mutating the graph after the cursor was created invalidates it.
    let mut mutated = graph.clone();
    mutated.add_edge(a, a, ());
    assert!(SimpleCycleEnumerator::resume(cursor, &mutated).is_err());
}
