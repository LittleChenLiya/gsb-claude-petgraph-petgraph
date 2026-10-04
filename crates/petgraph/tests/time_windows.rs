//! Cargo-directed tests for the time-window earliest-arrival path algorithm.

use petgraph::{
    algo::time_windows::{earliest_arrival_path, EdgeSchedule, TimeWindow},
    graph::Graph,
    stable_graph::StableGraph,
};

type Schedule = EdgeSchedule<u32, Vec<TimeWindow<u32>>>;

fn window(start: u32, end: u32) -> TimeWindow<u32> {
    TimeWindow::new(start, end)
}

/// Acceptance scenario: the direct edge has window [0, 5) and duration 3,
/// so departing at 4 cannot use it. A two-edge route that can depart at 4
/// and arrives at 8 must be selected, with wait details reported.
#[test]
fn direct_edge_rejected_at_four_two_edge_route_selected() {
    let mut graph: Graph<(), Schedule> = Graph::new();
    let start = graph.add_node(());
    let goal = graph.add_node(());
    let via = graph.add_node(());

    // Direct edge: window [0, 5), duration 3.
    let direct = graph.add_edge(start, goal, EdgeSchedule::new(3, vec![window(0, 5)]));
    // Two-edge route via `via`: both legs feasible at time 4.
    let leg1 = graph.add_edge(start, via, EdgeSchedule::new(2, vec![window(4, 7)]));
    let leg2 = graph.add_edge(via, goal, EdgeSchedule::new(2, vec![window(6, 9)]));

    let path = earliest_arrival_path(&graph, start, goal, 4, |edge| edge.weight().clone())
        .expect("a route exists");

    assert_eq!(path.departure_time, 4);
    assert_eq!(path.arrival_time, 8);
    assert_eq!(path.total_duration(), 4);
    assert_eq!(path.total_wait(), 0);
    assert_eq!(path.steps.len(), 2);

    // First leg: depart 4, arrive 6, no wait.
    assert_eq!(path.steps[0].from, start);
    assert_eq!(path.steps[0].to, via);
    assert_eq!(path.steps[0].edge, leg1);
    assert_eq!(path.steps[0].departure, 4);
    assert_eq!(path.steps[0].wait, 0);
    assert_eq!(path.steps[0].arrival, 6);

    // Second leg: depart 6, arrive 8, no wait.
    assert_eq!(path.steps[1].from, via);
    assert_eq!(path.steps[1].to, goal);
    assert_eq!(path.steps[1].edge, leg2);
    assert_eq!(path.steps[1].departure, 6);
    assert_eq!(path.steps[1].wait, 0);
    assert_eq!(path.steps[1].arrival, 8);

    // The direct edge is not part of the selected route.
    assert!(path.steps.iter().all(|step| step.edge != direct));
}

/// Waiting at a node until the next window is reported in the wait details.
#[test]
fn reports_wait_at_nodes() {
    let mut graph: Graph<(), Schedule> = Graph::new();
    let start = graph.add_node(());
    let goal = graph.add_node(());
    let via = graph.add_node(());

    // Direct edge: window [0, 5), duration 3 — infeasible at t=4.
    graph.add_edge(start, goal, EdgeSchedule::new(3, vec![window(0, 5)]));
    // Route via `via`: wait at start until 5, arrive 7, depart 7, arrive 9.
    graph.add_edge(start, via, EdgeSchedule::new(2, vec![window(5, 8)]));
    graph.add_edge(via, goal, EdgeSchedule::new(2, vec![window(7, 10)]));

    let path = earliest_arrival_path(&graph, start, goal, 4, |edge| edge.weight().clone())
        .expect("a route exists");

    assert_eq!(path.arrival_time, 9);
    assert_eq!(path.steps.len(), 2);
    assert_eq!(path.steps[0].wait, 1);
    assert_eq!(path.steps[0].departure, 5);
    assert_eq!(path.steps[0].arrival, 7);
    assert_eq!(path.steps[1].wait, 0);
    assert_eq!(path.steps[1].departure, 7);
    assert_eq!(path.steps[1].arrival, 9);
    assert_eq!(path.total_wait(), 1);
}

/// Zero-duration edges are handled: arrival equals departure, and the
/// completion rule is still checked against the window end.
#[test]
fn zero_duration_edges() {
    let mut graph: Graph<(), Schedule> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());

    // Zero-duration edge: depart at 3, arrive at 3.
    let e1 = graph.add_edge(a, b, EdgeSchedule::new(0, vec![window(0, 10)]));
    // Zero-duration edge that must wait for its window to open.
    let e2 = graph.add_edge(b, c, EdgeSchedule::new(0, vec![window(6, 10)]));

    let path = earliest_arrival_path(&graph, a, c, 3, |edge| edge.weight().clone())
        .expect("a route exists");

    assert_eq!(path.arrival_time, 6);
    assert_eq!(path.steps.len(), 2);
    assert_eq!(path.steps[0].edge, e1);
    assert_eq!(path.steps[0].departure, 3);
    assert_eq!(path.steps[0].wait, 0);
    assert_eq!(path.steps[0].arrival, 3);
    assert_eq!(path.steps[1].edge, e2);
    assert_eq!(path.steps[1].departure, 6);
    assert_eq!(path.steps[1].wait, 3);
    assert_eq!(path.steps[1].arrival, 6);

    // A zero-duration edge cannot be departed after its window closed.
    assert!(earliest_arrival_path(&graph, a, c, 11, |edge| edge.weight().clone()).is_none());
}

/// When no window allows a traversal to complete, no route exists and the
/// edge is not treated as zero cost.
#[test]
fn no_route_when_window_is_missed() {
    let mut graph: Graph<(), Schedule> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    graph.add_edge(a, b, EdgeSchedule::new(3, vec![window(0, 5)]));

    // Departing at 4: 4 + 3 > 5, the direct edge cannot be used.
    assert!(earliest_arrival_path(&graph, a, b, 4, |edge| edge.weight().clone()).is_none());
    // Departing at 2: 2 + 3 <= 5, the edge is feasible.
    let path = earliest_arrival_path(&graph, a, b, 2, |edge| edge.weight().clone())
        .expect("the edge is feasible at t=2");
    assert_eq!(path.arrival_time, 5);
}

/// A disconnected goal is unreachable.
#[test]
fn disconnected_goal_returns_none() {
    let mut graph: Graph<(), Schedule> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.add_edge(a, b, EdgeSchedule::new(1, vec![window(0, 10)]));

    assert!(earliest_arrival_path(&graph, a, c, 0, |edge| edge.weight().clone()).is_none());
}

/// Parallel edges keep their real edge identifiers.
#[test]
fn parallel_edges_keep_real_edge_id() {
    let mut graph: Graph<(), Schedule> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    // Two parallel edges: the first window is missed at t=4, the second
    // edge can be used.
    let e1 = graph.add_edge(a, b, EdgeSchedule::new(3, vec![window(0, 5)]));
    let e2 = graph.add_edge(a, b, EdgeSchedule::new(2, vec![window(4, 9)]));

    let path = earliest_arrival_path(&graph, a, b, 4, |edge| edge.weight().clone())
        .expect("a route exists");
    assert_eq!(path.arrival_time, 6);
    assert_eq!(path.steps.len(), 1);
    assert_eq!(path.steps[0].edge, e2);
    assert_ne!(path.steps[0].edge, e1);
}

/// The algorithm works on a StableGraph with holes in the node and edge
/// index spaces, and keeps the real edge identifiers.
#[test]
fn stable_graph_with_hole_indices() {
    let mut graph: StableGraph<(), Schedule> = StableGraph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());

    // Create a hole in the node index space.
    graph.remove_node(b);
    assert_eq!(graph.node_count(), 3);

    // Create edges; the direct edge gets index 0, the route edges get
    // indices 1 and 2.
    let direct = graph.add_edge(a, d, EdgeSchedule::new(3, vec![window(0, 5)]));
    let leg1 = graph.add_edge(a, c, EdgeSchedule::new(2, vec![window(4, 7)]));
    let leg2 = graph.add_edge(c, d, EdgeSchedule::new(2, vec![window(6, 9)]));

    // Create a hole in the edge index space.
    graph.remove_edge(direct);
    assert_eq!(graph.edge_count(), 2);

    // The graph is not modified by the search.
    let edges_before = graph.edge_count();
    let nodes_before = graph.node_count();

    let path = earliest_arrival_path(&graph, a, d, 4, |edge| edge.weight().clone())
        .expect("a route exists");

    assert_eq!(graph.edge_count(), edges_before);
    assert_eq!(graph.node_count(), nodes_before);

    assert_eq!(path.arrival_time, 8);
    assert_eq!(path.steps.len(), 2);
    assert_eq!(path.steps[0].from, a);
    assert_eq!(path.steps[0].to, c);
    assert_eq!(path.steps[0].edge, leg1);
    assert_eq!(path.steps[0].departure, 4);
    assert_eq!(path.steps[0].wait, 0);
    assert_eq!(path.steps[0].arrival, 6);
    assert_eq!(path.steps[1].from, c);
    assert_eq!(path.steps[1].to, d);
    assert_eq!(path.steps[1].edge, leg2);
    assert_eq!(path.steps[1].departure, 6);
    assert_eq!(path.steps[1].wait, 0);
    assert_eq!(path.steps[1].arrival, 8);
    // The removed edge must not appear in the route.
    assert!(path.steps.iter().all(|step| step.edge != direct));
}

/// The start node is trivially reachable from itself.
#[test]
fn start_equals_goal() {
    let mut graph: Graph<(), Schedule> = Graph::new();
    let a = graph.add_node(());

    let path = earliest_arrival_path(&graph, a, a, 7, |edge| edge.weight().clone())
        .expect("trivial route exists");
    assert_eq!(path.departure_time, 7);
    assert_eq!(path.arrival_time, 7);
    assert!(path.steps.is_empty());
}
