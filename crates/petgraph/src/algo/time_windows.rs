//! Earliest-arrival routing over edges with explicit, discrete time windows.
//!
//! Each edge weight is a [`TimedEdge`]: it carries a non-negative, integer
//! *travel duration* and a sorted list of half-open departure windows
//! `[start, end)`. Traversal may start at any integer time `t` such that the
//! *whole* traversal fits inside a single window:
//!
//! ```text
//! start <= t  and  t + duration <= end
//! ```
//!
//! The boundary rule is therefore uniform:
//!
//! * an edge with positive duration may finish exactly at `end` (the instant
//!   `end` is excluded only as a *departure* instant);
//! * a zero-duration edge may be taken at any `t` with `start <= t < end`.
//!
//! Waiting is allowed at every node until the next window of an outgoing edge
//! opens. Windows are given in explicit integer time units (a `u64` clock) and
//! are never periodic; edges that cannot be entered from the current arrival
//! instant (including those with no usable window at all) are *impassable* and
//! never relaxed as if they had zero cost.
//!
//! # Correctness: the FIFO condition
//!
//! [`earliest_arrival_path`] is a label-setting (Dijkstra-style) search and is
//! only correct under the **FIFO (first-in-first-out) property**: arriving at a
//! node no later can never make the earliest possible arrival at any successor
//! strictly later. It holds here because
//!
//! 1. waiting is always allowed, so each edge's earliest-departure function is
//!    non-decreasing in the arrival time at its source, and
//! 2. travel durations are fixed and independent of the departure time.
//!
//! It must not be applied to schedules that violate FIFO (for example
//! last-boarding rules that discard an early arrival, or departure-dependent
//! travel times where a later departure overtakes an earlier one); a
//! label-correcting algorithm would be required in that case.
//!
//! # Example
//! ```
//! use petgraph::{Graph, prelude::*};
//! use petgraph::algo::time_windows::{
//!     TimeWindow, TimedEdge, earliest_arrival_path,
//! };
//!
//! // A direct edge that only admits departures in [0, 5), duration 3, plus a
//! // two-edge alternative.
//! let mut g = Graph::new();
//! let s = g.add_node("start");
//! let m = g.add_node("mid");
//! let t = g.add_node("goal");
//! g.add_edge(s, t, TimedEdge::new(3, vec![TimeWindow::new(0, 5).unwrap()]).unwrap());
//! g.add_edge(s, m, TimedEdge::new(0, vec![TimeWindow::new(0, 10).unwrap()]).unwrap());
//! g.add_edge(m, t, TimedEdge::new(3, vec![TimeWindow::new(5, 9).unwrap()]).unwrap());
//!
//! let path = earliest_arrival_path(&g, s, t, 4).unwrap();
//! assert_eq!(path.arrival_time(), 8);
//! assert_eq!(path.total_wait(), 1);
//! ```

use alloc::{collections::BinaryHeap, vec::Vec};
use core::hash::Hash;

use hashbrown::HashMap;

use crate::{
    scored::MinScored,
    visit::{EdgeRef, GraphBase, IntoEdgeReferences, IntoEdges, Visitable},
};

/// A half-open interval `[start, end)` of integer departure times.
///
/// Every integer `t` with `start <= t < end` belongs to the window. A window is
/// considered *empty* when `start == end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimeWindow {
    start: u64,
    end: u64,
}

impl TimeWindow {
    /// Create the half-open window `[start, end)`.
    ///
    /// # Errors
    /// Returns [`TimeWindowError::Empty`] when `start >= end`, as an empty or
    /// reversed window can never admit a departure.
    pub fn new(start: u64, end: u64) -> Result<Self, TimeWindowError> {
        if start >= end {
            return Err(TimeWindowError::Empty);
        }
        Ok(TimeWindow { start, end })
    }

    /// Inclusive lower bound of the window.
    #[must_use]
    pub fn start(&self) -> u64 {
        self.start
    }

    /// Exclusive upper bound of the window.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.end
    }

    /// Return `true` if the integer instant `t` belongs to the window, i.e.
    /// `start <= t < end`.
    #[must_use]
    pub fn contains(&self, t: u64) -> bool {
        self.start <= t && t < self.end
    }
}

/// Failure when constructing a single [`TimeWindow`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeWindowError {
    /// `start >= end`: the window is empty or reversed.
    Empty,
}

/// Edge schedule data: a non-negative integer travel duration and the sorted,
/// non-overlapping half-open departure windows of an edge.
///
/// Build values with [`TimedEdge::new`]; the constructor validates the window
/// ordering so that a well-formed `TimedEdge` keeps the invariant required by
/// [`earliest_arrival_path`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimedEdge {
    duration: u64,
    windows: Vec<TimeWindow>,
}

impl TimedEdge {
    /// Create a timed edge with `duration` and the given departure `windows`.
    ///
    /// `duration` is a `u64`, hence always non-negative. The windows are
    /// validated before use: each one must be non-empty, they must be sorted by
    /// ascending start time and must not overlap (touching windows such as
    /// `[0, 5)` and `[5, 9)` are allowed).
    ///
    /// An empty window list is valid: the resulting edge is simply never
    /// traversable.
    ///
    /// # Errors
    /// Returns [`ScheduleError`] if any window is empty/reversed or if the
    /// windows are unsorted or overlapping.
    pub fn new(duration: u64, windows: Vec<TimeWindow>) -> Result<Self, ScheduleError> {
        validate_windows(&windows)?;
        Ok(TimedEdge { duration, windows })
    }

    /// Create an edge with travel `duration` and no departure windows; such an
    /// edge is never traversable.
    #[must_use]
    pub fn impassable(duration: u64) -> Self {
        TimedEdge {
            duration,
            windows: Vec::new(),
        }
    }

    /// Non-negative travel duration in integer time units.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.duration
    }

    /// Departure windows, sorted by ascending start time.
    #[must_use]
    pub fn windows(&self) -> &[TimeWindow] {
        &self.windows
    }

    /// Earliest departure instant `>= ready` for which the whole traversal
    /// fits in one window. Returns `None` when the edge cannot be entered.
    ///
    /// A zero-duration edge can depart at `end - 1` at the latest; a
    /// positive-duration edge may finish exactly at `end`.
    #[must_use]
    pub fn earliest_departure(&self, ready: u64) -> Option<u64> {
        for window in &self.windows {
            // The complete traversal (departure + duration) must not pass the
            // exclusive end of the window. Skip (rather than abort on) windows
            // too short for the duration, a later window might be wider.
            let Some(latest) = window.end.checked_sub(self.duration) else {
                continue;
            };
            if ready <= latest {
                let departure = core::cmp::max(ready, window.start);
                // For a positive duration `latest < end` already, but for a
                // zero-duration edge `latest == end`, so the half-open
                // departure bound has to be enforced explicitly.
                if departure <= latest && departure < window.end {
                    return Some(departure);
                }
            }
        }
        None
    }

    /// Earliest arrival instant at the target when reaching the source at
    /// `ready`; `None` if the edge is impassable from that instant.
    #[must_use]
    pub fn earliest_arrival(&self, ready: u64) -> Option<u64> {
        let departure = self.earliest_departure(ready)?;
        departure.checked_add(self.duration)
    }
}

impl Default for TimedEdge {
    /// A zero-duration edge with no windows (always impassable).
    fn default() -> Self {
        TimedEdge::impassable(0)
    }
}

/// Reflexive conversion so graphs whose edge weight *is* a [`TimedEdge`]
/// satisfy the `EdgeWeight: AsRef<TimedEdge>` bound, alongside custom wrappers
/// that provide their own implementation.
impl AsRef<TimedEdge> for TimedEdge {
    fn as_ref(&self) -> &Self {
        self
    }
}

/// Failure when validating the window ordering of a [`TimedEdge`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// A window is empty or reversed (`start >= end`).
    EmptyWindow,
    /// The windows are not ordered by ascending start time.
    UnsortedWindows,
    /// Two windows overlap (`next.start < previous.end`); touching is allowed.
    OverlappingWindows,
}

impl ScheduleError {
    /// Static English description of the error.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            ScheduleError::EmptyWindow => "time window is empty or reversed (start >= end)",
            ScheduleError::UnsortedWindows => "time windows are not sorted by ascending start",
            ScheduleError::OverlappingWindows => "time windows overlap",
        }
    }
}

/// Validate a slice of departure windows: each one must be non-empty, windows
/// must be sorted by ascending start time and must not overlap.
///
/// Ordering and durations are checked up front, before any search runs.
///
/// # Errors
/// Returns [`ScheduleError`] describing the first violation.
pub fn validate_windows(windows: &[TimeWindow]) -> Result<(), ScheduleError> {
    for window in windows {
        if window.start >= window.end {
            return Err(ScheduleError::EmptyWindow);
        }
    }
    for pair in windows.windows(2) {
        if pair[0].start >= pair[1].start {
            return Err(ScheduleError::UnsortedWindows);
        }
        if pair[1].start < pair[0].end {
            return Err(ScheduleError::OverlappingWindows);
        }
    }
    Ok(())
}

/// One traversed edge of an earliest-arrival path.
///
/// Reports the arrival at the source node, how long the route waits there, the
/// actual departure instant (the first window instant that fits), the edge that
/// was taken, and the resulting arrival at the target node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathStep<N, E> {
    from: N,
    to: N,
    edge_id: E,
    arrival_at_from: u64,
    departure: u64,
    arrival_at_to: u64,
}

impl<N, E> PathStep<N, E> {
    /// Node the step leaves from.
    #[must_use]
    pub fn from(&self) -> &N {
        &self.from
    }

    /// Node the step arrives at.
    #[must_use]
    pub fn to(&self) -> &N {
        &self.to
    }

    /// Identifier of the traversed edge (kept verbatim, so parallel edges keep
    /// their real `EdgeId`).
    #[must_use]
    pub fn edge_id(&self) -> &E {
        &self.edge_id
    }

    /// Arrival instant at [`from`](Self::from), equal to the start time for the
    /// first step.
    #[must_use]
    pub fn arrival_at_from(&self) -> u64 {
        self.arrival_at_from
    }

    /// Time spent waiting at [`from`](Self::from) before departing.
    #[must_use]
    pub fn wait_time(&self) -> u64 {
        self.departure - self.arrival_at_from
    }

    /// Actual departure instant.
    #[must_use]
    pub fn departure_time(&self) -> u64 {
        self.departure
    }

    /// Arrival instant at [`to`](Self::to), i.e. `departure + duration`.
    #[must_use]
    pub fn arrival_at_to(&self) -> u64 {
        self.arrival_at_to
    }
}

/// An earliest-arrival route found by [`earliest_arrival_path`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeWindowPath<N, E> {
    steps: Vec<PathStep<N, E>>,
    arrival: u64,
    wait: u64,
}

impl<N, E> TimeWindowPath<N, E> {
    /// Traversed edges in travel order.
    #[must_use]
    pub fn steps(&self) -> &[PathStep<N, E>] {
        &self.steps
    }

    /// Nodes visited, in travel order. Empty for a zero-edge (start == goal)
    /// route.
    #[must_use]
    pub fn nodes(&self) -> Vec<&N> {
        let mut nodes = Vec::with_capacity(self.steps.len() + 1);
        if let Some(first) = self.steps.first() {
            nodes.push(&first.from);
            nodes.extend(self.steps.iter().map(|step| &step.to));
        }
        nodes
    }

    /// Arrival instant at the goal (the start time for an empty route).
    #[must_use]
    pub fn arrival_time(&self) -> u64 {
        self.arrival
    }

    /// Sum of waiting times over every step.
    #[must_use]
    pub fn total_wait(&self) -> u64 {
        self.wait
    }

    /// Number of traversed edges.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Return `true` if the route contains no edges (start and goal are the
    /// same node).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Description of a malformed edge weight found while checking a whole graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidSchedule<E> {
    /// Identifier of the offending edge.
    pub edge_id: E,
    /// What was wrong with its schedule.
    pub error: ScheduleError,
}

/// Validate that every edge weight of `graph` is a well-formed [`TimedEdge`].
///
/// Construction via [`TimedEdge::new`] already validates individual edges, but
/// this allows a graph built with unchecked weights to be audited explicitly
/// before searching. The graph is only read, never modified.
///
/// # Errors
/// Returns the first [`InvalidSchedule`] encountered.
pub fn validate_graph_schedules<G>(
    graph: G,
) -> Result<(), InvalidSchedule<<G as GraphBase>::EdgeId>>
where
    G: IntoEdgeReferences,
    G::EdgeWeight: AsRef<TimedEdge>,
    <G as GraphBase>::EdgeId: Clone,
{
    for edge in graph.edge_references() {
        let timed: &TimedEdge = edge.weight().as_ref();
        validate_windows(&timed.windows).map_err(|error| InvalidSchedule {
            edge_id: edge.id().clone(),
            error,
        })?;
    }
    Ok(())
}

/// Find the route from `start` to `goal` with the earliest arrival when leaving
/// `start` no earlier than `start_time`.
///
/// Waiting at nodes is free and unlimited; the whole traversal of every edge
/// must fit inside one of its half-open windows (see the [module docs](self)).
/// Impassable edges — those with no window that can be entered from the current
/// arrival instant, or whose departure/arrival arithmetic overflows — are
/// skipped rather than treated as zero-cost transitions.
///
/// The graph is never mutated; only edge weights are read.
///
/// Requires the [FIFO condition](self#correctness-the-fifo-condition).
///
/// Returns `None` when `goal` is unreachable from `start` under the schedules.
/// If `start == goal`, an empty path arriving at `start_time` is returned.
///
/// # Complexity
/// **O((|E| · k + |V|) log |V|)** where `k` is the maximum number of windows
/// per edge; auxiliary space **O(|V|)**.
#[must_use]
pub fn earliest_arrival_path<G>(
    graph: G,
    start: G::NodeId,
    goal: G::NodeId,
    start_time: u64,
) -> Option<TimeWindowPath<G::NodeId, G::EdgeId>>
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash + Copy,
    G::EdgeId: Eq + Hash + Copy,
    G::EdgeWeight: AsRef<TimedEdge>,
{
    // Best (earliest) arrival settled so far at each node.
    let mut best: HashMap<G::NodeId, u64> = HashMap::new();
    // Search tree used to reconstruct arrival/wait/departure per step.
    let mut predecessors: HashMap<G::NodeId, Pred<G::NodeId, G::EdgeId>> = HashMap::new();
    let mut heap = BinaryHeap::new();

    best.insert(start, start_time);
    heap.push(MinScored(start_time, start));

    while let Some(MinScored(arrival, node)) = heap.pop() {
        // Stale heap entry: this node was settled earlier with an earlier
        // arrival, which is always at least as good under the FIFO property.
        if best.get(&node).copied() != Some(arrival) {
            continue;
        }
        if node == goal {
            return Some(reconstruct(goal, start_time, arrival, &predecessors));
        }

        for edge in graph.edges(node) {
            let timed: &TimedEdge = edge.weight().as_ref();
            let Some(departure) = timed.earliest_departure(arrival) else {
                // No window fits: the edge is impassable from this arrival,
                // never a zero-cost transition.
                continue;
            };
            let Some(next_arrival) = departure.checked_add(timed.duration()) else {
                // Arrival instant does not fit the u64 clock: impassable.
                continue;
            };

            let target = edge.target();
            if best
                .get(&target)
                .is_none_or(|current| next_arrival < *current)
            {
                best.insert(target, next_arrival);
                predecessors.insert(
                    target,
                    Pred {
                        from: node,
                        edge_id: edge.id(),
                        arrival_at_from: arrival,
                        departure,
                        arrival_at_to: next_arrival,
                    },
                );
                heap.push(MinScored(next_arrival, target));
            }
        }
    }

    None
}

struct Pred<N, E> {
    from: N,
    edge_id: E,
    arrival_at_from: u64,
    departure: u64,
    arrival_at_to: u64,
}

fn reconstruct<N, E>(
    goal: N,
    start_time: u64,
    goal_arrival: u64,
    predecessors: &HashMap<N, Pred<N, E>>,
) -> TimeWindowPath<N, E>
where
    N: Eq + Hash + Copy,
    E: Copy,
{
    let mut steps: Vec<PathStep<N, E>> = Vec::new();
    let mut current = goal;
    while let Some(pred) = predecessors.get(&current) {
        steps.push(PathStep {
            from: pred.from,
            to: current,
            edge_id: pred.edge_id,
            arrival_at_from: pred.arrival_at_from,
            departure: pred.departure,
            arrival_at_to: pred.arrival_at_to,
        });
        current = pred.from;
    }
    steps.reverse();
    let wait = steps.iter().map(PathStep::wait_time).sum::<u64>();
    debug_assert_eq!(
        steps
            .first()
            .map_or(start_time, |step| step.arrival_at_from),
        start_time
    );
    TimeWindowPath {
        steps,
        arrival: goal_arrival,
        wait,
    }
}
