/*!
Shortest path constrained by a deterministic finite automaton (DFA).
*/

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::fmt;
use core::hash::Hash;

use hashbrown::{HashMap, HashSet};

use crate::{
    algo::Measure,
    scored::MinScored,
    visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers},
};

/// A deterministic finite automaton used to constrain valid paths.
///
/// The automaton is defined by a start state, a set of labelled transitions
/// and a set of accepting states. A path through a graph is *accepted* when
/// the sequence of edge labels along the path drives the automaton from its
/// start state to one of its accepting states.
///
/// The automaton is treated as immutable input during the search: it is
/// borrowed by `dfa_shortest_path` and cannot be modified by the graph
/// traversal or the cost callback.
///
/// The automaton is expected to be deterministic: for a given state and
/// label there should be at most one transition. If several transitions
/// share the same (state, label) pair, one of them is picked arbitrarily.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dfa<S, L> {
    start: S,
    transitions: Vec<(S, L, S)>,
    accepting: Vec<S>,
}

impl<S, L> Dfa<S, L> {
    /// Create a new automaton with the given start state and no transitions.
    pub fn new(start: S) -> Self {
        Dfa {
            start,
            transitions: Vec::new(),
            accepting: Vec::new(),
        }
    }

    /// Add a transition from `from` to `to` taken when reading `label`.
    pub fn add_transition(&mut self, from: S, label: L, to: S) -> &mut Self {
        self.transitions.push((from, label, to));
        self
    }

    /// Mark `state` as an accepting state.
    pub fn add_accepting(&mut self, state: S) -> &mut Self {
        self.accepting.push(state);
        self
    }

    /// The start state of the automaton.
    pub fn start(&self) -> &S {
        &self.start
    }

    /// All transitions of the automaton, as (from, label, to) triples.
    pub fn transitions(&self) -> &[(S, L, S)] {
        &self.transitions
    }

    /// Return `true` if `state` is an accepting state.
    pub fn is_accepting(&self, state: &S) -> bool
    where
        S: PartialEq,
    {
        self.accepting.contains(state)
    }

    /// Return the state reached from `from` after reading `label`, if such a
    /// transition exists.
    pub fn transition(&self, from: &S, label: &L) -> Option<&S>
    where
        S: PartialEq,
        L: PartialEq,
    {
        self.transitions
            .iter()
            .find(|(f, l, _)| f == from && l == label)
            .map(|(_, _, to)| to)
    }
}

/// A path cost measure that can detect overflow.
///
/// This is implemented for all primitive integer types. The DFA-constrained
/// shortest path search uses it to report `DfaPathError::CostOverflow`
/// instead of silently wrapping on overflow.
pub trait CheckedMeasure: Measure + Copy {
    /// Additive identity, same as `Default::default`.
    fn zero() -> Self {
        Self::default()
    }

    /// Return `self + rhs`, or `None` if the addition overflows.
    fn checked_add(self, rhs: Self) -> Option<Self>;
}

macro_rules! impl_checked_measure {
    ($($t:ty),*) => {
        $(
            impl CheckedMeasure for $t {
                #[inline]
                fn checked_add(self, rhs: Self) -> Option<Self> {
                    <$t>::checked_add(self, rhs)
                }
            }
        )*
    };
}

impl_checked_measure! { u8, u16, u32, u64, u128, usize }
impl_checked_measure! { i8, i16, i32, i64, i128, isize }

/// Errors returned by `dfa_shortest_path`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DfaPathError<N, S> {
    /// The start node does not exist in the graph. Carries the offending
    /// node and the automaton start state the search would have used.
    InvalidStart {
        /// The requested start node, which is not part of the graph.
        node: N,
        /// The automaton start state paired with the invalid node.
        state: S,
    },
    /// An edge points to a node that does not exist in the graph.
    MissingTransitionTarget {
        /// The edge target that is not part of the graph.
        node: N,
    },
    /// Accumulating the cost of a candidate path overflowed. Carries the
    /// product state (node, automaton state) from which the overflowing
    /// edge was relaxed.
    CostOverflow {
        /// The graph node of the product state being expanded.
        node: N,
        /// The automaton state of the product state being expanded.
        state: S,
    },
}

impl<N: fmt::Debug, S: fmt::Debug> fmt::Display for DfaPathError<N, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DfaPathError::InvalidStart { node, state } => write!(
                f,
                "start node {node:?} does not exist in the graph (automaton start state: {state:?})"
            ),
            DfaPathError::MissingTransitionTarget { node } => {
                write!(f, "edge target node {node:?} does not exist in the graph")
            }
            DfaPathError::CostOverflow { node, state } => write!(
                f,
                "path cost overflowed while expanding node {node:?} in automaton state {state:?}"
            ),
        }
    }
}

impl<N, S> core::error::Error for DfaPathError<N, S>
where
    N: fmt::Debug,
    S: fmt::Debug,
{
}

/// The result of a successful `dfa_shortest_path` search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DfaPath<E, K> {
    /// The edge identifiers of the path, in path order. Empty when the
    /// empty path is accepted (i.e. the automaton start state is
    /// accepting).
    pub edges: Vec<E>,
    /// The total cost of the path.
    pub cost: K,
}

/// Dijkstra's shortest path algorithm constrained by a DFA.
///
/// Computes the shortest path from `start` whose edge label sequence is
/// accepted by `dfa`. Every edge of the graph carries a label (its weight)
/// and a non-negative integer cost (returned by `edge_cost`). The search
/// space is the product of graph nodes and automaton states: a search state
/// (node, state) means the search sits at `node` and the labels consumed
/// so far drove the automaton to `state`. A path is valid only if it ends
/// in an accepting automaton state.
///
/// Because the search is performed over the product space, zero-cost cycles
/// (including zero-cost self loops) terminate: each (node, state) pair is
/// settled at most once. Parallel edges are handled independently and the
/// returned path preserves their identity through their edge ids. If the
/// automaton start state is accepting, the empty path is a valid result.
///
/// The automaton is an immutable input: it is only borrowed for the
/// duration of the call and neither the graph traversal nor `edge_cost`
/// can modify it.
///
/// # Arguments
/// * `graph`: the input graph. Edge weights are the labels consumed by the
///   automaton.
/// * `start`: the node where the path starts.
/// * `dfa`: the automaton constraining valid label sequences.
/// * `edge_cost`: closure returning the non-negative cost of an edge.
///
/// # Returns
/// * `Ok(Some(path))` with the edge ids and total cost of the cheapest
///   accepted path.
/// * `Ok(None)` if no accepted path exists.
/// * `Err(DfaPathError::InvalidStart)` if `start` is not a node of the
///   graph.
/// * `Err(DfaPathError::MissingTransitionTarget)` if an edge points to a
///   node that does not exist in the graph.
/// * `Err(DfaPathError::CostOverflow)` if a candidate path cost overflows
///   the cost type.
///
/// # Complexity
/// Let |S| be the number of automaton states, |V| the number of
/// nodes and |E| the number of edges. The search runs Dijkstra over the
/// product graph with |V|·|S| states and at most |E|·|S| transitions.
/// * Time complexity: **O(|S|·(|E| + |V|·log(|V|·|S|)))**.
/// * Auxiliary space: **O(|V|·|S| + |T|)**, where |T| is the number of
///   automaton transitions.
///
/// # Example
/// ```rust
/// use petgraph::Graph;
/// use petgraph::algo::dfa_shortest_path::{dfa_shortest_path, Dfa};
///
/// // Automaton accepting label sequences "a" followed by "b".
/// let mut dfa = Dfa::new(0);
/// dfa.add_transition(0, 'a', 1);
/// dfa.add_transition(1, 'b', 2);
/// dfa.add_accepting(2);
///
/// let mut graph: Graph<(), char> = Graph::new();
/// let s = graph.add_node(());
/// let m = graph.add_node(());
/// let t = graph.add_node(());
/// // A cheap direct edge that the automaton rejects ...
/// graph.add_edge(s, t, 'x');
/// // ... and the legal two-edge path.
/// let e1 = graph.add_edge(s, m, 'a');
/// let e2 = graph.add_edge(m, t, 'b');
///
/// let costs = |e: petgraph::graph::EdgeReference<'_, char>| match *e.weight() {
///     'x' => 1u32,
///     _ => 5,
/// };
/// let path = dfa_shortest_path(&graph, s, &dfa, costs).unwrap().unwrap();
/// assert_eq!(path.edges, vec![e1, e2]);
/// assert_eq!(path.cost, 10);
/// ```
pub fn dfa_shortest_path<G, L, S, F, K>(
    graph: G,
    start: G::NodeId,
    dfa: &Dfa<S, L>,
    mut edge_cost: F,
) -> Result<Option<DfaPath<G::EdgeId, K>>, DfaPathError<G::NodeId, S>>
where
    G: IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    G::EdgeId: Eq + Hash,
    G::EdgeRef: EdgeRef<Weight = L>,
    L: Eq + Hash,
    S: Eq + Hash + Copy,
    F: FnMut(G::EdgeRef) -> K,
    K: CheckedMeasure,
{
    let nodes: HashSet<G::NodeId> = graph.node_identifiers().collect();
    if !nodes.contains(&start) {
        return Err(DfaPathError::InvalidStart {
            node: start,
            state: dfa.start,
        });
    }

    let mut transitions: HashMap<(S, &L), S> = HashMap::with_capacity(dfa.transitions.len());
    for (from, label, to) in &dfa.transitions {
        transitions.insert((*from, label), *to);
    }
    let accepting: HashSet<S> = dfa.accepting.iter().copied().collect();

    let start_state = (start, dfa.start);
    let mut dist: HashMap<(G::NodeId, S), K> = HashMap::new();
    let mut prev: HashMap<(G::NodeId, S), (G::NodeId, S, G::EdgeId)> = HashMap::new();
    let mut settled: HashSet<(G::NodeId, S)> = HashSet::new();
    let mut visit_next = BinaryHeap::new();
    dist.insert(start_state, K::zero());
    visit_next.push(MinScored(K::zero(), start_state));

    while let Some(MinScored(score, (node, state))) = visit_next.pop() {
        if !settled.insert((node, state)) {
            continue;
        }
        if accepting.contains(&state) {
            let mut edges = Vec::new();
            let mut current = (node, state);
            while current != start_state {
                let &(prev_node, prev_state, edge_id) = prev
                    .get(&current)
                    .expect("every settled non-start state has a predecessor");
                edges.push(edge_id);
                current = (prev_node, prev_state);
            }
            edges.reverse();
            return Ok(Some(DfaPath {
                edges,
                cost: score,
            }));
        }
        for edge in graph.edges(node) {
            let target = edge.target();
            if !nodes.contains(&target) {
                return Err(DfaPathError::MissingTransitionTarget { node: target });
            }
            let Some(&next_state) = transitions.get(&(state, edge.weight())) else {
                continue;
            };
            if settled.contains(&(target, next_state)) {
                continue;
            }
            let Some(next_score) = score.checked_add(edge_cost(edge)) else {
                return Err(DfaPathError::CostOverflow { node, state });
            };
            let improves = match dist.get(&(target, next_state)) {
                Some(&old) => next_score < old,
                None => true,
            };
            if improves {
                dist.insert((target, next_state), next_score);
                prev.insert((target, next_state), (node, state, edge.id()));
                visit_next.push(MinScored(next_score, (target, next_state)));
            }
        }
    }
    Ok(None)
}
