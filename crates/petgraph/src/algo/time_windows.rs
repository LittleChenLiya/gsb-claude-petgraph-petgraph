/*!
Earliest-arrival paths in a time-windowed network.

This module solves a time-dependent shortest path problem: every edge has a
non-negative *duration* (time needed to traverse it) and a set of half-open
*departure windows* `[start, end)` during which the edge may be entered.
Waiting at a node until a later window opens is always allowed. Given a
departure time at the start node, the algorithm finds a route to the target
node that arrives as early as possible.

# Feasibility of a traversal

An edge of duration `d` with a window `[s, e)` may be departed at time `t`
if and only if

* `s <= t` (the window has opened), and
* `t + d <= e` (the whole traversal completes no later than the window end).

The earliest feasible departure is `max(t, s)` if it satisfies the
completion rule, otherwise the next window is considered. Windows are
half-open: the completion rule `t + d <= e` is checked uniformly against
the exclusive end of every window.

# FIFO condition

The label-setting search used here is correct under the first-in-first-out
(FIFO) condition: arriving later at a node can never result in arriving
earlier at any other node. Sufficient conditions, all enforced by
[`validate_schedule`]:

1. durations are non-negative;
2. windows are sorted by start and non-overlapping
   (`next.start >= previous.end`);
3. vehicles may wait at a node for an arbitrary amount of time.

Under these conditions the earliest-arrival time at a node is a
non-decreasing function of the arrival time at its predecessor, so the
node with the smallest arrival time can be settled permanently when it is
popped from the priority queue.

# Example

```rust
use petgraph::{
    algo::time_windows::{earliest_arrival_path, EdgeSchedule, TimeWindow},
    graph::Graph,
};

let mut graph: Graph<(), EdgeSchedule<u32, Vec<TimeWindow<u32>>>> = Graph::new();
let a = graph.add_node(());
let b = graph.add_node(());
let c = graph.add_node(());
// Direct edge: window [0, 5), duration 3. Departing at 4 cannot complete
// the traversal in time.
let direct = graph.add_edge(a, b, EdgeSchedule::new(3, vec![TimeWindow::new(0, 5)]));
// Two-edge route via `c`: both legs are feasible at time 4.
let via_c_1 = graph.add_edge(a, c, EdgeSchedule::new(2, vec![TimeWindow::new(4, 7)]));
let via_c_2 = graph.add_edge(c, b, EdgeSchedule::new(2, vec![TimeWindow::new(6, 9)]));

let path = earliest_arrival_path(&graph, a, b, 4, |edge| edge.weight().clone())
    .expect("a route exists");

assert_eq!(path.arrival_time, 8);
assert_eq!(path.steps.len(), 2);
assert_eq!(path.steps[0].from, a);
assert_eq!(path.steps[0].to, c);
assert_eq!(path.steps[0].departure, 4);
assert_eq!(path.steps[0].wait, 0);
assert_eq!(path.steps[0].arrival, 6);
assert_eq!(path.steps[1].from, c);
assert_eq!(path.steps[1].to, b);
assert_eq!(path.steps[1].departure, 6);
assert_eq!(path.steps[1].wait, 0);
assert_eq!(path.steps[1].arrival, 8);
// The direct edge is not part of the selected route.
assert!(path.steps.iter().all(|step| step.edge != direct));
```
*/

use alloc::vec::Vec;
use core::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, Sub},
};

use hashbrown::hash_map::{Entry, HashMap};

use crate::{
    scored::MinScored,
    visit::{EdgeRef, IntoEdgeReferences, IntoEdges, VisitMap, Visitable},
};

/// A half-open departure window `[start, end)`.
///
/// The window contains all departure times `t` with `start <= t < end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TimeWindow<T> {
    /// Inclusive start of the window.
    pub start: T,
    /// Exclusive end of the window.
    pub end: T,
}

impl<T> TimeWindow<T> {
    /// Create a new window `[start, end)`.
    pub fn new(start: T, end: T) -> Self {
        TimeWindow { start, end }
    }
}

impl<T> From<(T, T)> for TimeWindow<T> {
    fn from((start, end): (T, T)) -> Self {
        TimeWindow { start, end }
    }
}

/// The time-dependent schedule of an edge: a non-negative traversal
/// [`duration`](EdgeSchedule::duration) and the set of half-open departure
/// [`windows`](EdgeSchedule::windows) during which the edge may be entered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeSchedule<T, W> {
    /// Time needed to traverse the edge. Must be non-negative.
    pub duration: T,
    /// Departure windows of the edge. Must be sorted and non-overlapping.
    pub windows: W,
}

impl<T, W> EdgeSchedule<T, W> {
    /// Create a new edge schedule.
    pub fn new(duration: T, windows: W) -> Self {
        EdgeSchedule { duration, windows }
    }
}

/// An invalid edge schedule, as reported by [`validate_schedule`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError<T> {
    /// The traversal duration is negative.
    NegativeDuration(T),
    /// A window does not satisfy `start < end`.
    InvalidWindow {
        /// Window start.
        start: T,
        /// Window end.
        end: T,
    },
    /// The windows are not sorted by start time.
    UnsortedWindows,
    /// Two windows overlap (`next.start < previous.end`).
    OverlappingWindows {
        /// End of the earlier window.
        previous_end: T,
        /// Start of the later window.
        next_start: T,
    },
}

/// Validate an edge schedule.
///
/// Checks that
///
/// * the duration is non-negative;
/// * every window is non-empty (`start < end`);
/// * windows are sorted by start and non-overlapping
///   (`next.start >= previous.end`).
///
/// On success the windows are returned as a sorted [`Vec`].
///
/// # Examples
///
/// ```rust
/// use petgraph::algo::time_windows::{validate_schedule, ScheduleError, TimeWindow};
///
/// assert!(validate_schedule(3, vec![TimeWindow::new(0, 5), TimeWindow::new(6, 9)]).is_ok());
/// assert_eq!(
///     validate_schedule(-1, vec![TimeWindow::new(0, 5)]),
///     Err(ScheduleError::NegativeDuration(-1)),
/// );
/// assert_eq!(
///     validate_schedule(1, vec![TimeWindow::new(5, 5)]),
///     Err(ScheduleError::InvalidWindow { start: 5, end: 5 }),
/// );
/// assert_eq!(
///     validate_schedule(1, vec![TimeWindow::new(6, 9), TimeWindow::new(0, 5)]),
///     Err(ScheduleError::UnsortedWindows),
/// );
/// assert_eq!(
///     validate_schedule(1, vec![TimeWindow::new(0, 6), TimeWindow::new(5, 9)]),
///     Err(ScheduleError::OverlappingWindows { previous_end: 6, next_start: 5 }),
/// );
/// ```
pub fn validate_schedule<T, W>(
    duration: T,
    windows: W,
) -> Result<Vec<TimeWindow<T>>, ScheduleError<T>>
where
    T: Copy + Ord + Default,
    W: IntoIterator<Item = TimeWindow<T>>,
{
    if duration < T::default() {
        return Err(ScheduleError::NegativeDuration(duration));
    }
    let mut result: Vec<TimeWindow<T>> = Vec::new();
    for window in windows {
        if window.start >= window.end {
            return Err(ScheduleError::InvalidWindow {
                start: window.start,
                end: window.end,
            });
        }
        if let Some(previous) = result.last() {
            if window.start < previous.start {
                return Err(ScheduleError::UnsortedWindows);
            }
            if window.start < previous.end {
                return Err(ScheduleError::OverlappingWindows {
                    previous_end: previous.end,
                    next_start: window.start,
                });
            }
        }
        result.push(window);
    }
    Ok(result)
}

/// Earliest feasible departure time for an edge.
///
/// Given the arrival time `t` at the tail of the edge, returns the earliest
/// departure time `departure >= t` and the corresponding arrival time
/// `departure + duration`, considering the half-open departure windows.
/// Returns `None` if no window allows the traversal to complete.
///
/// Because windows are sorted, the first feasible window yields the earliest
/// feasible departure.
fn earliest_departure<T>(
    arrival: T,
    duration: T,
    windows: &[TimeWindow<T>],
) -> Option<(T, T)>
where
    T: Copy + Ord + Add<Output = T>,
{
    for window in windows {
        let departure = arrival.max(window.start);
        // Completion rule: the whole traversal must finish no later than
        // the (exclusive) window end.
        if departure + duration <= window.end {
            return Some((departure, departure + duration));
        }
    }
    None
}

/// One step of a time-window route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeWindowStep<N, E, T> {
    /// Source node of the edge.
    pub from: N,
    /// Target node of the edge.
    pub to: N,
    /// Identifier of the traversed edge.
    ///
    /// For parallel edges this is the real edge identifier, so edges
    /// between the same pair of nodes remain distinguishable.
    pub edge: E,
    /// Departure time from [`from`](Self::from).
    pub departure: T,
    /// Time spent waiting at [`from`](Self::from) before departure.
    pub wait: T,
    /// Arrival time at [`to`](Self::to).
    pub arrival: T,
}

/// An earliest-arrival route found by [`earliest_arrival_path`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeWindowPath<N, E, T> {
    /// Start node of the route.
    pub start: N,
    /// Goal node of the route.
    pub goal: N,
    /// Departure time from the start node.
    pub departure_time: T,
    /// Arrival time at the goal node.
    pub arrival_time: T,
    /// Steps of the route, in order from start to goal.
    pub steps: Vec<TimeWindowStep<N, E, T>>,
}

impl<N, E, T> TimeWindowPath<N, E, T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Default,
{
    /// Total time spent waiting at intermediate nodes.
    pub fn total_wait(&self) -> T {
        self.steps
            .iter()
            .fold(T::default(), |total, step| total + step.wait)
    }

    /// Total travel time from departure to arrival.
    pub fn total_duration(&self) -> T {
        self.arrival_time - self.departure_time
    }
}

/// Find the earliest-arrival route from `start` to `goal` when departing at
/// `departure_time`.
///
/// `edge_schedule` is called once for every edge of the graph and must
/// return the edge's [`EdgeSchedule`] (traversal duration and departure
/// windows). It should be a pure function of the edge reference.
///
/// The algorithm first validates every edge schedule (non-negative duration,
/// sorted non-overlapping windows) and then runs a label-setting search
/// that always expands the node with the earliest arrival time. Waiting at
/// a node until a later window opens is allowed, and edges for which no
/// feasible window exists at the current time are skipped rather than
/// treated as zero cost. The original graph is never modified.
///
/// # Returns
///
/// * `Some(path)` with the earliest-arrival route, including per-step
///   departure, wait and arrival times.
/// * `None` if `goal` is unreachable from `start` at `departure_time`.
///
/// # Panics
///
/// Panics if an edge schedule is invalid (see [`validate_schedule`]).
///
/// # FIFO condition
///
/// The algorithm requires the FIFO condition described in the
/// [module documentation](self): arriving later at a node must never make it
/// possible to arrive earlier at the goal. The conditions enforced by
/// validation (non-negative durations, sorted non-overlapping windows,
/// waiting allowed) guarantee this.
pub fn earliest_arrival_path<G, F, W, T>(
    graph: G,
    start: G::NodeId,
    goal: G::NodeId,
    departure_time: T,
    mut edge_schedule: F,
) -> Option<TimeWindowPath<G::NodeId, G::EdgeId, T>>
where
    G: IntoEdges + IntoEdgeReferences + Visitable,
    G::NodeId: Eq + Hash + Debug,
    G::EdgeId: Eq + Hash + Debug,
    F: FnMut(G::EdgeRef) -> EdgeSchedule<T, W>,
    W: IntoIterator<Item = TimeWindow<T>>,
    T: Copy + Ord + Add<Output = T> + Sub<Output = T> + Default + Debug,
{
    // Phase 1: validate every edge schedule before searching.
    let mut schedules: HashMap<G::EdgeId, (T, Vec<TimeWindow<T>>)> = HashMap::new();
    for edge in graph.edge_references() {
        let schedule = edge_schedule(edge);
        let windows = match validate_schedule(schedule.duration, schedule.windows) {
            Ok(windows) => windows,
            Err(error) => panic!(
                "invalid time-window schedule for edge {:?}: {:?}",
                edge.id(),
                error,
            ),
        };
        schedules.insert(edge.id(), (schedule.duration, windows));
    }

    // Phase 2: label-setting search (earliest-arrival labels).
    let mut node_arrival: HashMap<G::NodeId, T> = HashMap::new();
    let mut predecessor: HashMap<G::NodeId, (G::NodeId, G::EdgeId, T)> = HashMap::new();
    let mut visit_next = alloc::collections::BinaryHeap::new();
    let mut visited = graph.visit_map();

    node_arrival.insert(start, departure_time);
    visit_next.push(MinScored(departure_time, start));

    let mut goal_reached = false;
    while let Some(MinScored(current_arrival, current)) = visit_next.pop() {
        if current == goal {
            goal_reached = true;
            break;
        }
        // Skip stale heap entries and nodes already settled.
        if visited.is_visited(&current)
            || node_arrival
                .get(&current)
                .is_none_or(|&best| best < current_arrival)
        {
            continue;
        }
        visited.visit(current);
        for edge in graph.edges(current) {
            let (duration, windows) = &schedules[&edge.id()];
            let Some((departure, next_arrival)) =
                earliest_departure(current_arrival, *duration, windows)
            else {
                // No feasible window at this time: the edge cannot be
                // traversed, it is not a zero-cost edge.
                continue;
            };
            let next = edge.target();
            match node_arrival.entry(next) {
                Entry::Occupied(mut entry) => {
                    if next_arrival < *entry.get() {
                        *entry.get_mut() = next_arrival;
                        predecessor.insert(next, (current, edge.id(), departure));
                        visit_next.push(MinScored(next_arrival, next));
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(next_arrival);
                    predecessor.insert(next, (current, edge.id(), departure));
                    visit_next.push(MinScored(next_arrival, next));
                }
            }
        }
    }

    if !goal_reached {
        return None;
    }

    // Phase 3: reconstruct the route and its per-step timing.
    let mut steps = Vec::new();
    let mut node = goal;
    while node != start {
        let (previous, edge, departure) = predecessor[&node];
        let arrival = node_arrival[&node];
        steps.push(TimeWindowStep {
            from: previous,
            to: node,
            edge,
            departure,
            wait: departure - node_arrival[&previous],
            arrival,
        });
        node = previous;
    }
    steps.reverse();

    Some(TimeWindowPath {
        start,
        goal,
        departure_time,
        arrival_time: node_arrival[&goal],
        steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use alloc::vec;

    type Schedule = EdgeSchedule<u32, Vec<TimeWindow<u32>>>;

    fn window(start: u32, end: u32) -> TimeWindow<u32> {
        TimeWindow::new(start, end)
    }

    #[test]
    fn validate_accepts_sorted_non_overlapping_windows() {
        assert!(validate_schedule(0, vec![window(0, 5), window(5, 10)]).is_ok());
        assert!(validate_schedule(3, vec![window(0, 5)]).is_ok());
        assert!(validate_schedule(3, Vec::<TimeWindow<u32>>::new()).is_ok());
    }

    #[test]
    fn validate_rejects_negative_duration() {
        assert_eq!(
            validate_schedule(-1i32, vec![TimeWindow::new(0i32, 5i32)]),
            Err(ScheduleError::NegativeDuration(-1)),
        );
    }

    #[test]
    fn validate_rejects_invalid_window() {
        assert_eq!(
            validate_schedule(1, vec![window(5, 5)]),
            Err(ScheduleError::InvalidWindow { start: 5, end: 5 }),
        );
        assert_eq!(
            validate_schedule(1, vec![window(6, 5)]),
            Err(ScheduleError::InvalidWindow { start: 6, end: 5 }),
        );
    }

    #[test]
    fn validate_rejects_unsorted_windows() {
        assert_eq!(
            validate_schedule(1, vec![window(6, 9), window(0, 5)]),
            Err(ScheduleError::UnsortedWindows),
        );
    }

    #[test]
    fn validate_rejects_overlapping_windows() {
        assert_eq!(
            validate_schedule(1, vec![window(0, 6), window(5, 9)]),
            Err(ScheduleError::OverlappingWindows {
                previous_end: 6,
                next_start: 5,
            }),
        );
    }

    #[test]
    fn earliest_departure_waits_for_next_window() {
        // Window [0, 5) with duration 3: at t=4 the traversal cannot
        // complete (4 + 3 > 5), so the next window must be used.
        let windows = vec![window(0, 5), window(8, 12)];
        assert_eq!(earliest_departure(0, 3, &windows), Some((0, 3)));
        assert_eq!(earliest_departure(2, 3, &windows), Some((2, 5)));
        assert_eq!(earliest_departure(3, 3, &windows), Some((8, 11)));
        assert_eq!(earliest_departure(4, 3, &windows), Some((8, 11)));
        assert_eq!(earliest_departure(9, 3, &windows), Some((9, 12)));
        assert_eq!(earliest_departure(10, 3, &windows), None);
    }

    #[test]
    fn earliest_departure_zero_duration() {
        let windows = vec![window(0, 5)];
        assert_eq!(earliest_departure(0, 0, &windows), Some((0, 0)));
        assert_eq!(earliest_departure(3, 0, &windows), Some((3, 3)));
        assert_eq!(earliest_departure(5, 0, &windows), Some((5, 5)));
        assert_eq!(earliest_departure(6, 0, &windows), None);
    }

    #[test]
    fn picks_earliest_arrival_and_reports_wait() {
        let mut graph: Graph<(), Schedule> = Graph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        // Direct edge: window [0, 5), duration 3.
        let direct = graph.add_edge(a, b, EdgeSchedule::new(3, vec![window(0, 5)]));
        // Two-edge route via `c`.
        let via_c_1 = graph.add_edge(a, c, EdgeSchedule::new(2, vec![window(4, 7)]));
        let via_c_2 = graph.add_edge(c, b, EdgeSchedule::new(2, vec![window(6, 9)]));

        // Departing at 4: the direct edge cannot be used, the route via c
        // arrives at 8.
        let path = earliest_arrival_path(&graph, a, b, 4, |edge| edge.weight().clone())
            .expect("route exists");
        assert_eq!(path.arrival_time, 8);
        assert_eq!(path.departure_time, 4);
        assert_eq!(path.total_duration(), 4);
        assert_eq!(path.total_wait(), 0);
        assert_eq!(path.steps.len(), 2);
        assert_eq!(path.steps[0].edge, via_c_1);
        assert_eq!(path.steps[0].from, a);
        assert_eq!(path.steps[0].to, c);
        assert_eq!(path.steps[0].departure, 4);
        assert_eq!(path.steps[0].wait, 0);
        assert_eq!(path.steps[0].arrival, 6);
        assert_eq!(path.steps[1].edge, via_c_2);
        assert_eq!(path.steps[1].from, c);
        assert_eq!(path.steps[1].to, b);
        assert_eq!(path.steps[1].departure, 6);
        assert_eq!(path.steps[1].wait, 0);
        assert_eq!(path.steps[1].arrival, 8);
        assert!(path.steps.iter().all(|step| step.edge != direct));

        // Departing at 2: the direct edge is feasible and arrives at 5.
        let path = earliest_arrival_path(&graph, a, b, 2, |edge| edge.weight().clone())
            .expect("route exists");
        assert_eq!(path.arrival_time, 5);
        assert_eq!(path.steps.len(), 1);
        assert_eq!(path.steps[0].edge, direct);
        assert_eq!(path.steps[0].departure, 2);
        assert_eq!(path.steps[0].arrival, 5);
    }

    #[test]
    fn waits_for_later_window_when_beneficial() {
        let mut graph: Graph<(), Schedule> = Graph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        // Edge with two windows: the first one is missed at t=6, so the
        // second window must be waited for.
        graph.add_edge(a, b, EdgeSchedule::new(3, vec![window(0, 5), window(8, 12)]));

        let path = earliest_arrival_path(&graph, a, b, 6, |edge| edge.weight().clone())
            .expect("route exists");
        assert_eq!(path.arrival_time, 11);
        assert_eq!(path.steps[0].wait, 2);
        assert_eq!(path.steps[0].departure, 8);
        assert_eq!(path.steps[0].arrival, 11);
        assert_eq!(path.total_wait(), 2);
    }

    #[test]
    fn no_route_when_no_window_is_feasible() {
        let mut graph: Graph<(), Schedule> = Graph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(a, b, EdgeSchedule::new(3, vec![window(0, 5)]));

        assert!(earliest_arrival_path(&graph, a, b, 4, |edge| edge.weight().clone()).is_none());
        assert!(earliest_arrival_path(&graph, a, b, 6, |edge| edge.weight().clone()).is_none());
    }

    #[test]
    fn unreachable_goal_returns_none() {
        let mut graph: Graph<(), Schedule> = Graph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.add_edge(a, b, EdgeSchedule::new(1, vec![window(0, 10)]));

        assert!(earliest_arrival_path(&graph, a, c, 0, |edge| edge.weight().clone()).is_none());
    }

    #[test]
    fn start_equals_goal() {
        let mut graph: Graph<(), Schedule> = Graph::new();
        let a = graph.add_node(());

        let path = earliest_arrival_path(&graph, a, a, 7, |edge| edge.weight().clone())
            .expect("trivial route exists");
        assert_eq!(path.arrival_time, 7);
        assert_eq!(path.departure_time, 7);
        assert!(path.steps.is_empty());
    }

    #[test]
    fn keeps_real_edge_id_for_parallel_edges() {
        let mut graph: Graph<(), Schedule> = Graph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        // Two parallel edges: the first window is missed at t=4, the second
        // edge can be used.
        let e1 = graph.add_edge(a, b, EdgeSchedule::new(3, vec![window(0, 5)]));
        let e2 = graph.add_edge(a, b, EdgeSchedule::new(2, vec![window(4, 9)]));

        let path = earliest_arrival_path(&graph, a, b, 4, |edge| edge.weight().clone())
            .expect("route exists");
        assert_eq!(path.arrival_time, 6);
        assert_eq!(path.steps.len(), 1);
        assert_eq!(path.steps[0].edge, e2);
        assert_ne!(path.steps[0].edge, e1);
    }

    #[test]
    #[should_panic(expected = "invalid time-window schedule")]
    fn panics_on_invalid_schedule() {
        type ScheduleI32 = EdgeSchedule<i32, Vec<TimeWindow<i32>>>;
        let mut graph: Graph<(), ScheduleI32> = Graph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(
            a,
            b,
            EdgeSchedule::new(-1, vec![TimeWindow::new(0, 5)]),
        );

        earliest_arrival_path(&graph, a, b, 0, |edge| edge.weight().clone());
    }
}
