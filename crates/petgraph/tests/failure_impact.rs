use petgraph::{
    algo::{FailureImpactReport, LostTarget, NodeFailureImpact, failure_impact},
    graph::{DiGraph, NodeIndex, UnGraph},
};

fn ix(n: u32) -> NodeIndex {
    NodeIndex::new(n as usize)
}

/// s -> a -> t and s -> b -> t.
fn diamond() -> DiGraph<(), ()> {
    DiGraph::from_edges([(0, 1), (1, 3), (0, 2), (2, 3)])
}

#[test]
fn acceptance_removing_a_does_not_affect_t_but_removing_s_does() {
    let g = diamond();

    let report = failure_impact(&g, ix(0), [ix(3)], [ix(1), ix(0)]);

    // Candidates stay in caller order.
    assert_eq!(report.node_impacts.len(), 2);
    assert_eq!(report.node_impacts[0].removed_node, ix(1));
    assert_eq!(report.node_impacts[1].removed_node, ix(0));

    // s -> b -> t survives the removal of `a`.
    assert!(report.node_impacts[0].affected_targets.is_empty());

    // Removing the start `s` loses `t`, and a reachable baseline path is reported.
    assert_eq!(report.node_impacts[1].affected_targets.len(), 1);
    let lost = &report.node_impacts[1].affected_targets[0];
    assert_eq!(lost.target, ix(3));
    assert_eq!(lost.original_path.first(), Some(&ix(0)));
    assert_eq!(lost.original_path.last(), Some(&ix(3)));
    assert_eq!(lost.original_path.len(), 3);
    // The reported path exists in the original graph.
    for window in lost.original_path.windows(2) {
        assert!(g.contains_edge(window[0], window[1]));
    }

    assert!(report.baseline_unreachable.is_empty());
}

#[test]
fn baseline_unreachable_target_is_listed_separately_and_never_affected() {
    let g = diamond();

    let report = failure_impact(&g, ix(0), [ix(3), ix(4)], [ix(1), ix(0)]);

    assert_eq!(report.baseline_unreachable, vec![ix(4)]);
    for impact in &report.node_impacts {
        assert!(
            impact
                .affected_targets
                .iter()
                .all(|lost| lost.target != ix(4)),
            "originally unreachable target must not be counted as a new failure",
        );
    }
}

#[test]
fn removing_target_itself_counts_as_affected() {
    let g = diamond();

    let report = failure_impact(&g, ix(0), [ix(3)], [ix(3)]);

    assert_eq!(report.node_impacts.len(), 1);
    let impact = &report.node_impacts[0];
    assert_eq!(impact.removed_node, ix(3));
    assert_eq!(impact.affected_targets.len(), 1);
    let lost = &impact.affected_targets[0];
    assert_eq!(lost.target, ix(3));
    assert_eq!(lost.original_path.first(), Some(&ix(0)));
    assert_eq!(lost.original_path.last(), Some(&ix(3)));
    for window in lost.original_path.windows(2) {
        assert!(g.contains_edge(window[0], window[1]));
    }
    assert!(report.baseline_unreachable.is_empty());
}

#[test]
fn cycles_are_supported_and_reported_paths_are_simple() {
    // s <-> a -> t, with an extra back edge t -> a forming a cycle.
    let g = DiGraph::<(), ()>::from_edges([(0, 1), (1, 0), (1, 2), (2, 1)]);

    let report = failure_impact(&g, ix(0), [ix(2)], [ix(1), ix(2)]);

    assert!(report.baseline_unreachable.is_empty());
    for impact in &report.node_impacts {
        assert_eq!(impact.affected_targets.len(), 1);
        let path = &impact.affected_targets[0].original_path;
        assert_eq!(path.first(), Some(&ix(0)));
        assert_eq!(path.last(), Some(&ix(2)));
        // A simple path despite the cycles.
        let mut nodes = path.clone();
        nodes.sort_unstable();
        let len_before = nodes.len();
        nodes.dedup();
        assert_eq!(nodes.len(), len_before);
    }
}

#[test]
fn parallel_edges_are_supported() {
    // Two parallel edges s -> t; removing neither endpoint disconnects the target,
    // but removing s or t does.
    let g = DiGraph::<(), ()>::from_edges([(0, 1), (0, 1)]);

    let report = failure_impact(&g, ix(0), [ix(1)], [ix(0), ix(1)]);

    assert_eq!(
        report.node_impacts[0].affected_targets[0].original_path,
        vec![ix(0), ix(1)],
    );
    assert_eq!(
        report.node_impacts[1].affected_targets[0].original_path,
        vec![ix(0), ix(1)],
    );
}

#[test]
fn duplicate_candidates_and_targets_are_deduplicated_in_caller_order() {
    let g = diamond();

    let report = failure_impact(
        &g,
        ix(0),
        [ix(3), ix(3), ix(2)],
        [ix(0), ix(0), ix(1), ix(1)],
    );

    let removed: Vec<_> = report
        .node_impacts
        .iter()
        .map(|impact| impact.removed_node)
        .collect();
    assert_eq!(removed, vec![ix(0), ix(1)]);

    // Targets within an impact keep first-occurrence order too.
    let lost: Vec<_> = report
        .node_impacts
        .iter()
        .find(|impact| impact.removed_node == ix(0))
        .unwrap()
        .affected_targets
        .iter()
        .map(|lost| lost.target)
        .collect();
    assert_eq!(lost, vec![ix(3), ix(2)]);
}

#[test]
fn candidate_not_in_graph_affects_nothing() {
    let g = diamond();
    let ghost = ix(99);

    let report = failure_impact(&g, ix(0), [ix(3)], [ghost, ix(1)]);

    assert_eq!(report.node_impacts.len(), 2);
    assert_eq!(report.node_impacts[0].removed_node, ghost);
    assert!(report.node_impacts[0].affected_targets.is_empty());
    assert_eq!(report.node_impacts[1].removed_node, ix(1));
    assert!(report.node_impacts[1].affected_targets.is_empty());
}

#[test]
fn graph_is_not_mutated_by_the_analysis() {
    let g = diamond();
    let edges_before: Vec<_> = g
        .raw_edges()
        .iter()
        .map(|edge| (edge.source(), edge.target()))
        .collect();
    let nodes_before = g.node_count();

    let _report: FailureImpactReport<NodeIndex> =
        failure_impact(&g, ix(0), [ix(3), ix(4)], [ix(0), ix(1), ix(99)]);

    assert_eq!(g.node_count(), nodes_before);
    let edges_after: Vec<_> = g
        .raw_edges()
        .iter()
        .map(|edge| (edge.source(), edge.target()))
        .collect();
    assert_eq!(edges_after, edges_before);
}

#[test]
fn unknown_start_makes_every_target_baseline_unreachable() {
    let g = diamond();

    let report = failure_impact(&g, ix(99), [ix(3), ix(4)], [ix(1)]);

    assert_eq!(report.baseline_unreachable, vec![ix(3), ix(4)]);
    assert!(report.node_impacts[0].affected_targets.is_empty());
}

#[test]
fn works_on_undirected_graphs() {
    // s - a - t and s - b - t as an undirected diamond.
    let g = UnGraph::<(), ()>::from_edges([(0, 1), (1, 3), (0, 2), (2, 3)]);

    let report = failure_impact(&g, ix(0), [ix(3)], [ix(1), ix(0), ix(3)]);

    assert!(report.node_impacts[0].affected_targets.is_empty());
    assert_eq!(report.node_impacts[1].affected_targets[0].target, ix(3),);
    assert_eq!(report.node_impacts[2].affected_targets[0].target, ix(3),);
    assert!(report.baseline_unreachable.is_empty());
}

#[test]
fn all_candidates_together_match_independent_reports() {
    // A linear chain s -> a -> t: only removing s or a loses t.
    let g = DiGraph::<(), ()>::from_edges([(0, 1), (1, 2)]);

    let report = failure_impact(&g, ix(0), [ix(2)], [ix(0), ix(1), ix(2)]);

    let expected = vec![
        NodeFailureImpact {
            removed_node: ix(0),
            affected_targets: vec![LostTarget {
                target: ix(2),
                original_path: vec![ix(0), ix(1), ix(2)],
            }],
        },
        NodeFailureImpact {
            removed_node: ix(1),
            affected_targets: vec![LostTarget {
                target: ix(2),
                original_path: vec![ix(0), ix(1), ix(2)],
            }],
        },
        NodeFailureImpact {
            removed_node: ix(2),
            affected_targets: vec![LostTarget {
                target: ix(2),
                original_path: vec![ix(0), ix(1), ix(2)],
            }],
        },
    ];
    assert_eq!(report.node_impacts, expected);
}

#[test]
fn works_on_graphmap_with_integer_node_ids() {
    use petgraph::graphmap::DiGraphMap;

    let g = DiGraphMap::<u32, ()>::from_edges([(0, 1), (1, 3), (0, 2), (2, 3)]);

    // Candidate 7 is not a node of the graph, so it affects nothing.
    let report = failure_impact(&g, 0u32, [3u32, 4u32], [1u32, 0u32, 7u32]);

    assert!(report.node_impacts[0].affected_targets.is_empty());
    assert_eq!(report.node_impacts[1].affected_targets[0].target, 3u32);
    assert!(report.node_impacts[2].affected_targets.is_empty());
    assert_eq!(report.baseline_unreachable, vec![4u32]);
}
