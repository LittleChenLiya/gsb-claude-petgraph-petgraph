//! Targeted tests for the explicit time-window earliest-arrival module.

use petgraph::{
    Graph,
    algo::time_windows::{
        InvalidSchedule, ScheduleError, TimeWindow, TimeWindowError, TimedEdge,
        earliest_arrival_path, validate_graph_schedules, validate_windows,
    },
    prelude::*,
    stable_graph::StableGraph,
};

fn w(start: u64, end: u64) -> TimeWindow {
    TimeWindow::new(start, end).unwrap()
}

fn edge(duration: u64, windows: Vec<TimeWindow>) -> TimedEdge {
    TimedEdge::new(duration, windows).unwrap()
}

/// Acceptance case:
/// direct edge window `[0, 5)` with duration 3 cannot be used departing at 4;
/// the two-edge alternative usable at second 4 (arrival at second 8) is chosen,
/// and the waiting breakdown is reported.
#[test]
fn direct_edge_window_too_tight_two_edge_route_with_wait_is_chosen() {
    let mut g = Graph::<(), TimedEdge, Directed>::new();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());

    let direct = g.add_edge(s, t, edge(3, vec![w(0, 5)]));
    let first = g.add_edge(s, m, edge(0, vec![w(0, 10)]));
    let second = g.add_edge(m, t, edge(3, vec![w(5, 9)]));

    let path = earliest_arrival_path(&g, s, t, 4).expect("two-edge route exists");

    assert_eq!(path.arrival_time(), 8);
    assert_eq!(path.total_wait(), 1);
    assert_eq!(path.len(), 2);
    assert_eq!(path.nodes(), vec![&s, &m, &t]);

    let steps = path.steps();
    assert_eq!(steps[0].from(), &s);
    assert_eq!(steps[0].to(), &m);
    assert_eq!(steps[0].edge_id(), &first);
    assert_eq!(steps[0].arrival_at_from(), 4);
    assert_eq!(steps[0].wait_time(), 0);
    assert_eq!(steps[0].departure_time(), 4);
    assert_eq!(steps[0].arrival_at_to(), 4);

    assert_eq!(steps[1].from(), &m);
    assert_eq!(steps[1].to(), &t);
    assert_eq!(steps[1].edge_id(), &second);
    assert_eq!(steps[1].arrival_at_from(), 4);
    assert_eq!(steps[1].wait_time(), 1);
    assert_eq!(steps[1].departure_time(), 5);
    assert_eq!(steps[1].arrival_at_to(), 8);

    // The direct edge must not appear in the answer.
    assert!(path.steps().iter().all(|step| step.edge_id() != &direct));

    // Leaving early enough, the direct edge is usable (finish exactly at 5).
    let early = earliest_arrival_path(&g, s, t, 2).unwrap();
    assert_eq!(early.arrival_time(), 5);
    assert_eq!(early.steps().len(), 1);
    assert_eq!(early.steps()[0].edge_id(), &direct);
}

/// Unified end-of-window rule: positive-duration traversal may *complete*
/// exactly at the (exclusive) end, while departure at the end is never allowed;
/// a zero-duration edge at `end` is impassable but works at `end - 1`.
#[test]
fn window_end_completion_rule_is_unified() {
    let positive = edge(3, vec![w(0, 5)]);
    assert_eq!(positive.earliest_departure(0), Some(0));
    assert_eq!(positive.earliest_departure(2), Some(2)); // arrives exactly at 5
    assert_eq!(positive.earliest_departure(3), None);
    assert_eq!(positive.earliest_departure(4), None);

    let zero = edge(0, vec![w(0, 5)]);
    assert_eq!(zero.earliest_departure(0), Some(0));
    assert_eq!(zero.earliest_departure(4), Some(4));
    assert_eq!(zero.earliest_departure(5), None); // half-open: end excluded
}

/// Waiting at a node until a later window opens.
#[test]
fn waits_at_node_for_next_window() {
    let mut g = Graph::<(), TimedEdge, Directed>::new();
    let s = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, t, edge(1, vec![w(0, 2), w(5, 8)]));

    // First window too tight for ready=2; the second window requires waiting.
    let path = earliest_arrival_path(&g, s, t, 2).unwrap();
    assert_eq!(path.total_wait(), 3);
    assert_eq!(path.steps()[0].departure_time(), 5);
    assert_eq!(path.arrival_time(), 6);
}

/// No usable route yields `None`; an impassable edge is never taken as a
/// zero-cost transition.
#[test]
fn no_route_returns_none_and_impassable_is_not_zero_cost() {
    let mut g = Graph::<(), TimedEdge, Directed>::new();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    // Window [0, 3) cannot contain a traversal of duration 5.
    g.add_edge(s, t, edge(5, vec![w(0, 3)]));
    g.add_edge(s, m, edge(1, vec![w(0, 10)]));
    // m -> t window is already closed at the only possible arrival times.
    g.add_edge(m, t, edge(1, vec![w(0, 1)]));

    assert!(earliest_arrival_path(&g, s, t, 4).is_none());

    // Minimal zero-cost regression: one impassable edge only.
    let mut solo = Graph::<(), TimedEdge, Directed>::new();
    let a = solo.add_node(());
    let b = solo.add_node(());
    solo.add_edge(a, b, edge(5, vec![w(0, 3)]));
    assert!(earliest_arrival_path(&solo, a, b, 0).is_none());
}

/// Zero-duration edges are traversable strictly inside a window, and clock
/// boundary arithmetic (duration beyond a window, or a departure that would
/// complete past `u64::MAX`) leaves the edge impassable.
#[test]
fn zero_duration_and_overflow_edges() {
    let zero = TimedEdge::new(0, vec![w(5, 9)]).unwrap();
    assert_eq!(zero.earliest_arrival(0), Some(5)); // wait then instant hop
    assert_eq!(zero.earliest_arrival(8), Some(8));
    assert_eq!(zero.earliest_arrival(9), None);

    // duration == u64::MAX fits in [0, u64::MAX) only when departing exactly
    // at 0; arriving at the exclusive end u64::MAX is allowed.
    let maximal = TimedEdge::new(u64::MAX, vec![w(0, u64::MAX)]).unwrap();
    assert_eq!(maximal.earliest_departure(0), Some(0));
    assert_eq!(maximal.earliest_arrival(0), Some(u64::MAX));
    assert_eq!(maximal.earliest_departure(1), None);

    // A duration beyond the window end cannot fit (end - duration underflows):
    // the edge is impassable, never a zero-cost transition.
    let beyond = TimedEdge::new(u64::MAX, vec![w(0, 5)]).unwrap();
    assert_eq!(beyond.earliest_arrival(0), None);

    // Boundary where the completion would leave the u64 clock: ready=2 would
    // make departure+duration wrap, so the latest-departure bound rejects it.
    let wrapping = TimedEdge::new(u64::MAX - 1, vec![w(0, u64::MAX)]).unwrap();
    assert_eq!(wrapping.earliest_departure(1), Some(1));
    assert_eq!(wrapping.earliest_departure(2), None);

    let mut g = Graph::<(), TimedEdge, Directed>::new();
    let a = g.add_node(());
    let b = g.add_node(());
    g.add_edge(a, b, beyond);
    assert!(earliest_arrival_path(&g, a, b, 0).is_none());
}

/// Parallel edges keep their real [`EdgeId`]; the faster-scheduled parallel
/// edge wins even though it incurs waiting.
#[test]
fn parallel_edges_report_real_edge_id() {
    let mut g = Graph::<(), TimedEdge, Directed>::new();
    let s = g.add_node(());
    let t = g.add_node(());
    let slow = g.add_edge(s, t, edge(5, vec![w(0, 10)])); // depart 4 -> arrive 9
    let quick = g.add_edge(s, t, edge(1, vec![w(6, 10)])); // wait -> arrive 7

    let path = earliest_arrival_path(&g, s, t, 4).unwrap();
    assert_eq!(path.arrival_time(), 7);
    assert_eq!(path.total_wait(), 2);
    assert_eq!(path.steps().len(), 1);
    assert_eq!(path.steps()[0].edge_id(), &quick);
    assert_ne!(quick, slow);
}

/// `StableGraph` node/edge holes (freed indices) must not disturb the search,
/// which is indexed by identifier maps rather than dense arrays.
#[test]
fn stable_graph_hole_indices() {
    let mut g = StableGraph::<(), TimedEdge, Directed>::new();
    let a = g.add_node(());
    let ghost1 = g.add_node(());
    let b = g.add_node(());
    let ghost2 = g.add_node(());
    let c = g.add_node(());

    g.add_edge(a, b, edge(2, vec![w(0, 10)]));
    g.add_edge(b, c, edge(1, vec![w(4, 10)]));
    // Edges touching removed nodes; they must disappear with the nodes.
    g.add_edge(ghost1, ghost2, edge(1, vec![w(0, 10)]));
    g.add_edge(a, ghost1, edge(1, vec![w(0, 10)]));

    g.remove_node(ghost1);
    g.remove_node(ghost2);

    // The surviving indices are non-contiguous (b is index 2, c is index 4).
    assert_eq!(b.index(), 2);
    assert_eq!(c.index(), 4);

    let path = earliest_arrival_path(&g, a, c, 1).unwrap();
    assert_eq!(path.arrival_time(), 5);
    assert_eq!(path.total_wait(), 1);
    assert_eq!(path.nodes(), vec![&a, &b, &c]);
    let step = &path.steps()[1];
    assert_eq!(step.arrival_at_from(), 3);
    assert_eq!(step.departure_time(), 4);
    assert_eq!(step.arrival_at_to(), 5);

    validate_graph_schedules(&g).unwrap();
}

/// Start equal to goal yields an empty path at the start time.
#[test]
fn start_is_goal() {
    let mut g = Graph::<(), TimedEdge, Directed>::new();
    let s = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, t, edge(1, vec![w(0, 5)]));

    let path = earliest_arrival_path(&g, s, s, 7).unwrap();
    assert!(path.is_empty());
    assert_eq!(path.arrival_time(), 7);
    assert_eq!(path.total_wait(), 0);
    assert!(path.nodes().is_empty());
}

/// Constructor validation covers empty windows, ordering and overlap; touching
/// windows remain legal.
#[test]
fn window_schedule_validation() {
    assert_eq!(TimeWindow::new(3, 3), Err(TimeWindowError::Empty));
    assert_eq!(TimeWindow::new(4, 3), Err(TimeWindowError::Empty));
    assert!(matches!(
        validate_windows(&[w(5, 9), w(0, 3)]),
        Err(ScheduleError::UnsortedWindows)
    ));
    assert!(matches!(
        validate_windows(&[w(0, 10), w(5, 8)]),
        Err(ScheduleError::OverlappingWindows)
    ));
    validate_windows(&[w(0, 5), w(5, 9)]).unwrap();

    let mut g = Graph::<(), TimedEdge, Directed>::new();
    let a = g.add_node(());
    let b = g.add_node(());
    let e = g.add_edge(a, b, edge(1, vec![w(0, 5), w(5, 9)]));
    validate_graph_schedules(&g).unwrap();
    assert_eq!(e.index(), 0);

    // `InvalidSchedule` carries the real edge id and the underlying error.
    let invalid: InvalidSchedule<EdgeIndex> = InvalidSchedule {
        edge_id: e,
        error: ScheduleError::OverlappingWindows,
    };
    assert_eq!(invalid.edge_id, e);
}

/// Windows too short for the duration are skipped in favour of later, wider
/// windows; errors carry descriptive messages.
#[test]
fn skips_short_windows_and_error_descriptions() {
    let timed = edge(4, vec![w(0, 3), w(5, 10)]);
    assert_eq!(timed.earliest_departure(0), Some(5));
    assert_eq!(timed.earliest_arrival(0), Some(9));
    assert_eq!(timed.earliest_departure(7), None); // only 3 units remain

    assert_eq!(
        ScheduleError::EmptyWindow.description(),
        "time window is empty or reversed (start >= end)"
    );
    assert_eq!(
        ScheduleError::UnsortedWindows.description(),
        "time windows are not sorted by ascending start"
    );
    assert_eq!(
        ScheduleError::OverlappingWindows.description(),
        "time windows overlap"
    );

    // A duration longer than every window means the edge is never usable.
    let never = edge(10, vec![w(0, 5), w(6, 9)]);
    assert_eq!(never.earliest_departure(0), None);
    assert!(TimedEdge::impassable(2).windows().is_empty());
}

/// FIFO property of the edge departure function: an earlier arrival at the
/// source can never cause a later arrival at the target.
#[test]
fn earliest_arrival_is_fifo_monotone() {
    let schedules = vec![
        edge(0, vec![w(0, 5)]),
        edge(3, vec![w(2, 6), w(10, 20)]),
        edge(7, vec![w(0, 4), w(4, 8)]),
        edge(1, vec![]),
    ];
    for timed in schedules {
        let mut prev = timed.earliest_arrival(0);
        for ready in 1..40u64 {
            let now = timed.earliest_arrival(ready);
            // FIFO: f(ready) must be non-decreasing, with unreachable treated
            // as +infinity (so `Some -> None` after the last window is fine).
            match (prev, now) {
                (Some(a), Some(b)) => assert!(a <= b, "FIFO violated at {ready}: {a} > {b}"),
                (None, Some(_)) => {
                    panic!("FIFO violated: unreachable earlier, reachable at {ready}")
                }
                _ => {}
            }
            prev = now;
        }
    }
}

/// A multi-hop route reports arrival, wait and departure for every step, and
/// choosing a later start never yields an earlier arrival (FIFO end to end).
#[test]
fn multi_step_timing_breakdown_and_late_start() {
    let mut g = Graph::<(), TimedEdge, Directed>::new();
    let s = g.add_node(());
    let a = g.add_node(());
    let b = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, a, edge(2, vec![w(0, 10)]));
    g.add_edge(a, b, edge(1, vec![w(6, 12)]));
    g.add_edge(b, t, edge(1, vec![w(0, 100)]));

    let path = earliest_arrival_path(&g, s, t, 0).unwrap();
    assert_eq!(path.nodes(), vec![&s, &a, &b, &t]);
    assert_eq!(path.arrival_time(), 8);
    assert_eq!(path.total_wait(), 4);

    let expected = [
        // (arrival_at_from, departure, arrival_at_to)
        (0u64, 0u64, 2u64),
        (2, 6, 7),
        (7, 7, 8),
    ];
    for (step, (ready, dep, arr)) in path.steps().iter().zip(expected) {
        assert_eq!(step.arrival_at_from(), ready);
        assert_eq!(step.departure_time(), dep);
        assert_eq!(step.arrival_at_to(), arr);
        assert_eq!(step.wait_time(), dep - ready);
    }

    // Starting later cannot improve the earliest arrival (FIFO).
    let later = earliest_arrival_path(&g, s, t, 1).unwrap();
    assert!(later.arrival_time() >= path.arrival_time());
}

/// The search is generic over graph types; undirected graphs work as well.
#[test]
fn undirected_graph_is_supported() {
    let mut g = Graph::<(), TimedEdge, Undirected>::new_undirected();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, m, edge(1, vec![w(0, 10)]));
    g.add_edge(m, t, edge(2, vec![w(0, 4), w(8, 12)]));

    let path = earliest_arrival_path(&g, s, t, 0).unwrap();
    assert_eq!(path.steps()[0].arrival_at_to(), 1);
    assert_eq!(path.steps()[1].wait_time(), 0);
    assert_eq!(path.arrival_time(), 3);

    // Arriving at m later than 2 forces waiting for the [8, 12) window.
    let later = earliest_arrival_path(&g, s, t, 5).unwrap();
    assert_eq!(later.steps()[1].departure_time(), 8);
    assert_eq!(later.arrival_time(), 10);
}
