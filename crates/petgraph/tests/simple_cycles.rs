use petgraph::{
    algo::{CycleBatchStatus, CycleEnumeratorError, SimpleCycleEnumerator},
    graph::{DiGraph, EdgeIndex},
    stable_graph::StableDiGraph,
    visit::{GraphRef, IntoEdgeReferences, NodeIndexable},
};

/// Collect all cycles in a single unpaused enumeration, together with the
/// total number of edge visits performed.
fn collect_all<G>(g: G) -> (Vec<Vec<G::EdgeId>>, u64)
where
    G: GraphRef + IntoEdgeReferences + NodeIndexable,
{
    let mut enumerator = SimpleCycleEnumerator::new(g);
    let mut cursor = enumerator.cursor();
    let mut cycles = Vec::new();
    loop {
        let batch = enumerator.next_batch(g, &cursor, usize::MAX, usize::MAX).unwrap();
        assert_eq!(batch.status, CycleBatchStatus::Done);
        cycles.extend(batch.cycles);
        cursor = batch.cursor;
        if batch.status == CycleBatchStatus::Done {
            break;
        }
    }
    (cycles, enumerator.total_edge_visits())
}

/// Collect all cycles batch by batch, together with the per-batch statuses
/// and the summed edge visits.
fn collect_batched<G>(
    g: G,
    max_results: usize,
    edge_visit_budget: usize,
) -> (Vec<Vec<G::EdgeId>>, Vec<CycleBatchStatus>, u64)
where
    G: GraphRef + IntoEdgeReferences + NodeIndexable,
{
    let mut enumerator = SimpleCycleEnumerator::new(g);
    let mut cursor = enumerator.cursor();
    let mut cycles = Vec::new();
    let mut statuses = Vec::new();
    let mut visits = 0;
    loop {
        let batch = enumerator
            .next_batch(g, &cursor, max_results, edge_visit_budget)
            .unwrap();
        cycles.extend(batch.cycles);
        statuses.push(batch.status);
        visits += batch.edge_visits;
        cursor = batch.cursor;
        if batch.status == CycleBatchStatus::Done {
            break;
        }
    }
    (cycles, statuses, visits)
}

fn sorted(mut cycles: Vec<Vec<EdgeIndex>>) -> Vec<Vec<EdgeIndex>> {
    cycles.sort();
    cycles
}

/// Triangle a -> b -> c -> a with a parallel a -> b edge and a self-loop on a.
fn triangle_parallel_self_loop() -> (DiGraph<(), ()>, [EdgeIndex; 5]) {
    let mut g = DiGraph::new();
    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());
    let e0 = g.add_edge(a, b, ());
    let e1 = g.add_edge(b, c, ());
    let e2 = g.add_edge(c, a, ());
    let e3 = g.add_edge(a, b, ()); // parallel edge
    let e4 = g.add_edge(a, a, ()); // self-loop
    (g, [e0, e1, e2, e3, e4])
}

#[test]
fn batched_capacity_one_matches_one_shot() {
    let (g, [e0, e1, e2, e3, e4]) = triangle_parallel_self_loop();

    let expected = sorted(vec![vec![e0, e1, e2], vec![e3, e1, e2], vec![e4]]);

    let (one_shot, one_shot_visits) = collect_all(&g);
    assert_eq!(sorted(one_shot), expected);

    let (batched, statuses, batched_visits) = collect_batched(&g, 1, usize::MAX);
    // Same edge-cycle set, each cycle emitted exactly once.
    assert_eq!(sorted(batched), expected);
    // Three cycles with capacity 1: three paused batches, then done.
    assert_eq!(
        statuses,
        vec![
            CycleBatchStatus::Paused,
            CycleBatchStatus::Paused,
            CycleBatchStatus::Paused,
            CycleBatchStatus::Done,
        ]
    );
    // Resuming never rescans the already enumerated prefix.
    assert_eq!(batched_visits, one_shot_visits);
}

#[test]
fn cycles_are_rotation_normalized_and_unique() {
    let (g, _) = triangle_parallel_self_loop();
    let (cycles, _) = collect_all(&g);

    assert_eq!(cycles.len(), 3);
    for cycle in &cycles {
        // The first edge leaves the minimum-index node of the cycle.
        let nodes: Vec<usize> = cycle
            .iter()
            .map(|&e| g.edge_endpoints(e).unwrap().0.index())
            .collect();
        let min = *nodes.iter().min().unwrap();
        assert_eq!(nodes[0], min);
    }
    // No duplicates.
    let mut deduped = cycles.clone();
    deduped.sort();
    deduped.dedup();
    assert_eq!(deduped.len(), cycles.len());
}

#[test]
fn resume_continues_at_paused_position() {
    let (g, _) = triangle_parallel_self_loop();
    let (one_shot, _) = collect_all(&g);

    // Pause after every single cycle and resume: the cycle sequence must be
    // identical to the one-shot sequence, not a restart from the first cycle.
    let (batched, _, _) = collect_batched(&g, 1, usize::MAX);
    assert_eq!(batched, one_shot);
}

#[test]
fn acyclic_graph_is_done_immediately() {
    let mut g = DiGraph::<(), ()>::new();
    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());
    g.extend_with_edges([(a, b), (b, c), (a, c)]);

    let mut enumerator = SimpleCycleEnumerator::new(&g);
    let cursor = enumerator.cursor();
    let batch = enumerator.next_batch(&g, &cursor, 10, 10).unwrap();
    assert_eq!(batch.status, CycleBatchStatus::Done);
    assert!(batch.cycles.is_empty());
    assert!(enumerator.is_done());
}

#[test]
fn stable_graph_with_empty_slots() {
    let mut g = StableDiGraph::<(), ()>::new();
    let n0 = g.add_node(());
    let n1 = g.add_node(());
    let n2 = g.add_node(());
    let n3 = g.add_node(());
    let n4 = g.add_node(());
    // Leave an empty slot in the node storage.
    g.remove_node(n1);

    let e0 = g.add_edge(n0, n2, ());
    let e1 = g.add_edge(n2, n3, ());
    let e2 = g.add_edge(n3, n0, ());
    let e3 = g.add_edge(n4, n4, ()); // self-loop after the empty slot

    let (cycles, _) = collect_all(&g);
    let mut cycles = cycles;
    cycles.sort();
    assert_eq!(cycles, vec![vec![e0, e1, e2], vec![e3]]);

    // Batched enumeration over the empty slots works as well.
    let (batched, _, _) = collect_batched(&g, 1, 2);
    let mut batched = batched;
    batched.sort();
    assert_eq!(batched, vec![vec![e0, e1, e2], vec![e3]]);
}

#[test]
fn tiny_budget_pauses_instead_of_finishing() {
    let (g, _) = triangle_parallel_self_loop();

    // A zero edge-visit budget pauses immediately without emitting anything.
    let mut enumerator = SimpleCycleEnumerator::new(&g);
    let cursor = enumerator.cursor();
    let batch = enumerator.next_batch(&g, &cursor, usize::MAX, 0).unwrap();
    assert_eq!(batch.status, CycleBatchStatus::Paused);
    assert!(batch.cycles.is_empty());
    assert_eq!(batch.edge_visits, 0);
    assert!(!enumerator.is_done());

    // One edge visit per batch still reaches every cycle eventually.
    let (one_shot, one_shot_visits) = collect_all(&g);
    let (batched, statuses, visits) = collect_batched(&g, usize::MAX, 1);
    assert_eq!(batched, one_shot);
    assert_eq!(visits, one_shot_visits);
    assert!(statuses.len() > 1);
    assert!(statuses[..statuses.len() - 1]
        .iter()
        .all(|&s| s == CycleBatchStatus::Paused));
    assert_eq!(statuses[statuses.len() - 1], CycleBatchStatus::Done);
}

#[test]
fn edge_visits_do_not_rescan_prefix() {
    let (g, _) = triangle_parallel_self_loop();
    let (_, one_shot_visits) = collect_all(&g);

    for &(max_results, budget) in &[(1, usize::MAX), (usize::MAX, 1), (1, 1), (2, 3)] {
        let (_, _, visits) = collect_batched(&g, max_results, budget);
        assert_eq!(
            visits, one_shot_visits,
            "batched enumeration with max_results={max_results} and              budget={budget} rescanned edges"
        );
    }
}

#[test]
fn graph_change_is_rejected() {
    let (mut g, _) = triangle_parallel_self_loop();

    let mut enumerator = SimpleCycleEnumerator::new(&g);
    let cursor = enumerator.cursor();
    let batch = enumerator.next_batch(&g, &cursor, 1, usize::MAX).unwrap();
    assert_eq!(batch.status, CycleBatchStatus::Paused);

    // Structurally modifying the graph invalidates the enumerator's version.
    let x = g.add_node(());
    g.add_edge(x, x, ());
    let err = enumerator
        .next_batch(&g, &batch.cursor, 1, usize::MAX)
        .unwrap_err();
    assert_eq!(err, CycleEnumeratorError::GraphChanged);
}

#[test]
fn stale_cursor_is_rejected() {
    let (g, _) = triangle_parallel_self_loop();

    let mut enumerator = SimpleCycleEnumerator::new(&g);
    let cursor = enumerator.cursor();
    let batch = enumerator.next_batch(&g, &cursor, 1, usize::MAX).unwrap();

    // Reusing the previous cursor instead of the fresh one is rejected.
    let err = enumerator
        .next_batch(&g, &cursor, 1, usize::MAX)
        .unwrap_err();
    assert_eq!(err, CycleEnumeratorError::StaleCursor);

    // The fresh cursor resumes normally.
    let batch = enumerator
        .next_batch(&g, &batch.cursor, usize::MAX, usize::MAX)
        .unwrap();
    assert_eq!(batch.status, CycleBatchStatus::Done);
}
