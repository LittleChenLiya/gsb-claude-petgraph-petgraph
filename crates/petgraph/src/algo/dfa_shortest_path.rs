//! Shortest path constrained by a deterministic finite automaton (DFA).
//!
//! This module implements a shortest-path search where every edge carries a
//! label and a non-negative integer cost, and a path is only *valid* if the
//! sequence of labels along its edges is accepted by a given DFA. The search
//! runs Dijkstra's algorithm over the product of the graph nodes and the
//! automaton states, so each search state is a `(node, dfa state)` pair.

use alloc::{collections::BinaryHeap, vec::Vec};
use core::{fmt, hash::Hash};

use hashbrown::{
    HashMap, HashSet,
    hash_map::Entry::{Occupied, Vacant},
};

use crate::{
    algo::Measure,
    scored::MinScored,
    visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers},
};

/// A deterministic finite automaton over edge labels of type `L`.
///
/// The automaton is an immutable input to `dfa_shortest_path`: it is
/// borrowed shared for the whole search and cannot be modified by the graph
/// traversal callbacks while the search is running.
///
/// States are identified by indices in `0..num_states`. Missing transitions
/// are *dead ends*: a path whose next label has no transition cannot be
/// extended and is discarded.
#[derive(Clone, Debug)]
pub struct Dfa<L> {
    num_states: usize,
    start: usize,
    accepting: Vec<bool>,
    transitions: HashMap<usize, HashMap<L, usize>>,
}

impl<L> PartialEq for Dfa<L>
where
    L: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.num_states == other.num_states
            && self.start == other.start
            && self.accepting == other.accepting
            && self.transitions == other.transitions
    }
}

impl<L> Eq for Dfa<L> where L: Eq + Hash {}

/// The error type for `Dfa::new`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfaError {
    /// The automaton must have at least one state.
    NoStates,

    /// The start state is not a valid state index.
    StartStateOutOfBounds {
        /// The offending state index.
        state: usize,
        /// The number of states of the automaton.
        num_states: usize,
    },

    /// An accepting state is not a valid state index.
    AcceptingStateOutOfBounds {
        /// The offending state index.
        state: usize,
        /// The number of states of the automaton.
        num_states: usize,
    },

    /// A transition source is not a valid state index.
    TransitionSourceOutOfBounds {
        /// The offending state index.
        state: usize,
        /// The number of states of the automaton.
        num_states: usize,
    },

    /// A transition target is not a valid state index.
    TransitionTargetOutOfBounds {
        /// The offending state index.
        state: usize,
        /// The number of states of the automaton.
        num_states: usize,
    },
}

impl core::error::Error for DfaError {}

impl fmt::Display for DfaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DfaError::NoStates => write!(f, "the DFA must have at least one state"),
            DfaError::StartStateOutOfBounds { state, num_states } => write!(
                f,
                "the DFA start state {state} is out of bounds for {num_states} states"
            ),
            DfaError::AcceptingStateOutOfBounds { state, num_states } => write!(
                f,
                "the DFA accepting state {state} is out of bounds for {num_states} states"
            ),
            DfaError::TransitionSourceOutOfBounds { state, num_states } => write!(
                f,
                "a DFA transition source {state} is out of bounds for {num_states} states"
            ),
            DfaError::TransitionTargetOutOfBounds { state, num_states } => write!(
                f,
                "a DFA transition target {state} is out of bounds for {num_states} states"
            ),
        }
    }
}

impl<L> Dfa<L>
where
    L: Eq + Hash,
{
    /// Create a new DFA with `num_states` states, the given `start` state,
    /// the given set of `accepting` states, and `transitions` given as
    /// `(source, label, target)` triples.
    ///
    /// Returns a `DfaError` if the automaton has no states or if any state
    /// index (including transition targets) is not in `0..num_states`.
    ///
    /// If several transitions are given for the same `(source, label)` pair,
    /// the last one wins.
    pub fn new(
        num_states: usize,
        start: usize,
        accepting: impl IntoIterator<Item = usize>,
        transitions: impl IntoIterator<Item = (usize, L, usize)>,
    ) -> Result<Self, DfaError> {
        if num_states == 0 {
            return Err(DfaError::NoStates);
        }
        if start >= num_states {
            return Err(DfaError::StartStateOutOfBounds {
                state: start,
                num_states,
            });
        }
        let mut accepting_flags = alloc::vec![false; num_states];
        for state in accepting {
            if state >= num_states {
                return Err(DfaError::AcceptingStateOutOfBounds { state, num_states });
            }
            accepting_flags[state] = true;
        }
        let mut transition_map: HashMap<usize, HashMap<L, usize>> = HashMap::new();
        for (source, label, target) in transitions {
            if source >= num_states {
                return Err(DfaError::TransitionSourceOutOfBounds {
                    state: source,
                    num_states,
                });
            }
            if target >= num_states {
                return Err(DfaError::TransitionTargetOutOfBounds {
                    state: target,
                    num_states,
                });
            }
            transition_map
                .entry(source)
                .or_default()
                .insert(label, target);
        }
        Ok(Dfa {
            num_states,
            start,
            accepting: accepting_flags,
            transitions: transition_map,
        })
    }

    /// Return the number of states of the automaton.
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Return the start state of the automaton.
    pub fn start_state(&self) -> usize {
        self.start
    }

    /// Return `true` if `state` is an accepting state.
    ///
    /// Out-of-bounds state indices are never accepting.
    pub fn is_accepting(&self, state: usize) -> bool {
        self.accepting.get(state).copied().unwrap_or(false)
    }

    /// Return the target state of the transition from `state` on `label`,
    /// or `None` if the automaton has no such transition.
    pub fn transition(&self, state: usize, label: &L) -> Option<usize> {
        self.transitions.get(&state)?.get(label).copied()
    }
}

/// A measure (edge cost) type that can detect addition overflow.
///
/// This trait is implemented for all primitive integer types. It is used by
/// `dfa_shortest_path` to report `DfaShortestPathError::CostOverflow`
/// instead of silently wrapping around when path costs accumulate past the
/// representable range.
pub trait CheckedAddMeasure: Measure + Copy {
    /// Return `self + rhs`, or `None` if the result is not representable.
    fn checked_add(self, rhs: Self) -> Option<Self>;
}

macro_rules! impl_checked_add_measure {
    ($($t:ty),* $(,)?) => {
        $(
            impl CheckedAddMeasure for $t {
                #[inline]
                fn checked_add(self, rhs: Self) -> Option<Self> {
                    <$t>::checked_add(self, rhs)
                }
            }
        )*
    };
}

impl_checked_add_measure!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

/// A shortest DFA-constrained path, as returned by `dfa_shortest_path`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DfaPath<E, K> {
    /// The sequence of edge identifiers of the path, in traversal order.
    ///
    /// Parallel edges keep their identity: each entry is the exact edge that
    /// was traversed. The sequence is empty if the start state of the DFA is
    /// already accepting.
    pub edges: Vec<E>,

    /// The total cost of the path, i.e. the sum of its edge costs.
    pub cost: K,
}

/// The error type for `dfa_shortest_path`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfaShortestPathError<N> {
    /// The start node does not exist in the graph.
    InvalidStartNode(N),

    /// An edge carried a negative cost; only non-negative costs are supported.
    NegativeCost,

    /// Accumulating the path cost overflowed the cost type.
    CostOverflow,
}

impl<N> core::error::Error for DfaShortestPathError<N> where N: fmt::Debug {}

impl<N> fmt::Display for DfaShortestPathError<N>
where
    N: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DfaShortestPathError::InvalidStartNode(node) => {
                write!(f, "the start node {node:?} does not exist in the graph")
            }
            DfaShortestPathError::NegativeCost => {
                write!(f, "edge costs must be non-negative")
            }
            DfaShortestPathError::CostOverflow => {
                write!(f, "the accumulated path cost overflowed")
            }
        }
    }
}

/// Compute the shortest path from `start` whose edge label sequence is
/// accepted by `dfa`.
///
/// Every edge carries a label and a non-negative integer cost, extracted with
/// the `edge_data` closure. A path is *valid* if feeding the labels of its
/// edges, in order, into `dfa` ends in an accepting state. The search state
/// is the pair `(node, dfa state)`, so zero-cost cycles in the graph still
/// terminate: each pair is settled at most once.
///
/// The DFA is an immutable input: it is only borrowed shared, so it cannot be
/// modified by the `edge_data` callback (or any other traversal callback)
/// while the search is running.
///
/// # Arguments
/// * `graph`: the graph to search. Works with any graph implementing `IntoEdges` +
///   `IntoNodeIdentifiers`, including `Graph` and `StableGraph`.
/// * `start`: the node where the path starts. Must exist in the graph.
/// * `dfa`: the automaton constraining the valid label sequences.
/// * `edge_data`: closure mapping an edge to its `(label, cost)` pair.
///
/// # Returns
/// * `Ok(Some(path))`: the cheapest valid path as a sequence of edge identifiers together with its
///   total cost. If the DFA start state is accepting, the empty path (no edges, cost zero) is a
///   valid answer.
/// * `Ok(None)`: no valid path exists.
/// * `Err(_)`: the start node is invalid, an edge cost is negative, or the accumulated cost
///   overflowed.
///
/// # Complexity
/// * Time complexity: **O((|V|·|S| + |E|·|S|) log(|V|·|S|))**, where **|V|** and **|E|** are the
///   numbers of nodes and edges of the graph and **|S|** is the number of DFA states.
/// * Auxiliary space: **O(|V|·|S|)**.
///
/// # Example
/// `rust
/// use petgraph::algo::dfa_shortest_path::{Dfa, dfa_shortest_path};
/// use petgraph::prelude::*;
///
/// // Accept exactly the label sequences that start with 'a' and end with 'b'.
/// let dfa = Dfa::new(
///     3,
///     0,
///     [2],
///     [
///         (0, 'a', 1),
///         (1, 'a', 1),
///         (1, 'b', 2),
///         (2, 'a', 1),
///         (2, 'b', 2),
///     ],
/// )
/// .unwrap();
///
/// let mut graph: Graph<(), (char, u32)> = Graph::new();
/// let s = graph.add_node(());
/// let t = graph.add_node(());
/// let u = graph.add_node(());
/// // A cheap but invalid direct edge, and a legal two-edge detour.
/// graph.add_edge(s, t, ('b', 1));
/// graph.add_edge(s, u, ('a', 2));
/// graph.add_edge(u, t, ('b', 2));
///
/// let path = dfa_shortest_path(&graph, s, &dfa, |edge| *edge.weight())
///     .unwrap()
///     .unwrap();
/// assert_eq!(path.edges.len(), 2);
/// assert_eq!(path.cost, 4);
/// `
///
/// `Graph`: crate::Graph
/// `StableGraph`: crate::stable_graph::StableGraph
pub fn dfa_shortest_path<G, L, K, F>(
    graph: G,
    start: G::NodeId,
    dfa: &Dfa<L>,
    mut edge_data: F,
) -> Result<Option<DfaPath<G::EdgeId, K>>, DfaShortestPathError<G::NodeId>>
where
    G: IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Copy,
    G::EdgeId: Copy,
    L: Eq + Hash,
    K: CheckedAddMeasure,
    F: FnMut(G::EdgeRef) -> (L, K),
{
    if !graph.node_identifiers().any(|node| node == start) {
        return Err(DfaShortestPathError::InvalidStartNode(start));
    }

    let zero_score = K::default();
    let start_state = (start, dfa.start_state());
    let mut scores = HashMap::new();
    let mut visited = HashSet::new();
    let mut predecessors: HashMap<(G::NodeId, usize), (G::NodeId, usize, G::EdgeId)> =
        HashMap::new();
    let mut visit_next = BinaryHeap::new();
    scores.insert(start_state, zero_score);
    visit_next.push(MinScored(zero_score, start_state));

    while let Some(MinScored(node_score, (node, dfa_state))) = visit_next.pop() {
        if !visited.insert((node, dfa_state)) {
            continue;
        }
        if dfa.is_accepting(dfa_state) {
            let mut edges = Vec::new();
            let mut current = (node, dfa_state);
            while let Some(&(prev_node, prev_state, edge_id)) = predecessors.get(&current) {
                edges.push(edge_id);
                current = (prev_node, prev_state);
            }
            edges.reverse();
            return Ok(Some(DfaPath {
                edges,
                cost: node_score,
            }));
        }
        for edge in graph.edges(node) {
            let (label, cost) = edge_data(edge);
            if cost < zero_score {
                return Err(DfaShortestPathError::NegativeCost);
            }
            let Some(next_dfa_state) = dfa.transition(dfa_state, &label) else {
                continue;
            };
            let next = (edge.target(), next_dfa_state);
            if visited.contains(&next) {
                continue;
            }
            let Some(next_score) = node_score.checked_add(cost) else {
                return Err(DfaShortestPathError::CostOverflow);
            };
            match scores.entry(next) {
                Occupied(mut entry) => {
                    if next_score < *entry.get() {
                        *entry.get_mut() = next_score;
                        visit_next.push(MinScored(next_score, next));
                        predecessors.insert(next, (node, dfa_state, edge.id()));
                    }
                }
                Vacant(entry) => {
                    entry.insert(next_score);
                    visit_next.push(MinScored(next_score, next));
                    predecessors.insert(next, (node, dfa_state, edge.id()));
                }
            }
        }
    }
    Ok(None)
}
