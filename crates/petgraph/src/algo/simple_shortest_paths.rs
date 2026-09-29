//! Lazily enumerate the cheapest simple paths between two nodes.
//!
//! See [`simple_shortest_paths`] for details.

use alloc::{collections::BinaryHeap, vec, vec::Vec};
use core::{cmp::Ordering, hash::Hash};

use hashbrown::HashSet;

use crate::{
    algo::BoundedMeasure,
    visit::{EdgeRef, IntoEdges},
};

/// A simple path from a start node to a goal node together with its total cost.
///
/// A simple path never visits a node more than once, so paths that would
/// traverse a cycle are not reported. Different paths may still share edges
/// and nodes with each other.
///
/// This is the item type of the iterator returned by
/// [`simple_shortest_paths`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimpleShortestPath<N, E, K> {
    /// Sum of the costs of all edges along the path.
    pub cost: K,
    /// Nodes visited along the path, including the start and goal nodes.
    ///
    /// For a path from a node to itself this contains only the single node.
    pub nodes: Vec<N>,
    /// Edges traversed along the path, in traversal order.
    ///
    /// Parallel edges are distinguished by their edge identifiers.
    pub edges: Vec<E>,
}

/// Lazily produce simple paths from a start node to a goal node in non-decreasing
/// cost order.
///
/// This iterator is created by [`simple_shortest_paths`].
pub struct SimpleShortestPaths<G, F, K>
where
    G: IntoEdges,
{
    graph: G,
    edge_cost: F,
    goal: G::NodeId,
    heap: BinaryHeap<HeapEntry<G::NodeId, G::EdgeId, K>>,
}

impl<G, F, K> Iterator for SimpleShortestPaths<G, F, K>
where
    G: IntoEdges,
    G::NodeId: Eq + Hash,
    G::EdgeId: Ord,
    F: FnMut(G::EdgeRef) -> K,
    K: BoundedMeasure + Copy,
{
    type Item = SimpleShortestPath<G::NodeId, G::EdgeId, K>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let entry = self.heap.pop()?;
            let PathState {
                cost,
                nodes,
                edges,
                visited,
            } = entry.0;

            let node = *nodes.last().expect("a path always contains nodes");
            if node == self.goal {
                return Some(SimpleShortestPath { cost, nodes, edges });
            }

            for edge in self.graph.edges(node) {
                let target = edge.target();
                if visited.contains(&target) {
                    continue;
                }

                let edge_cost = (self.edge_cost)(edge);
                debug_assert!(
                    edge_cost >= K::default(),
                    "simple_shortest_paths requires non-negative edge costs"
                );

                // Paths whose cost cannot be represented can never be compared
                // with the cheaper paths, so they are dropped instead of wrapping.
                let (next_cost, overflowed) = cost.overflowing_add(edge_cost);
                if overflowed {
                    continue;
                }

                let mut next_visited = visited.clone();
                next_visited.insert(target);
                let mut next_nodes = nodes.clone();
                next_nodes.push(target);
                let mut next_edges = edges.clone();
                next_edges.push(edge.id());
                self.heap.push(HeapEntry(PathState {
                    cost: next_cost,
                    nodes: next_nodes,
                    edges: next_edges,
                    visited: next_visited,
                }));
            }
        }
    }
}

/// Enumerate simple paths from `start` to `goal` in a directed graph, ordered
/// from cheapest to most expensive.
///
/// Paths are ordered first by their total cost (the sum of the edge costs)
/// and, for paths with equal total cost, by the lexicographic order of their
/// edge identifier sequences. Parallel edges are distinguished by their edge
/// identifiers, so each parallel edge may appear in a separate path.
///
/// Every reported path is simple: no node is visited more than once within a
/// single path. Different reported paths may share edges or nodes.
///
/// The returned iterator is lazy: paths are expanded with a best-first search
/// only as results are pulled, so taking the first path does not enumerate all
/// simple paths. The number of simple paths can still grow exponentially, so
/// use [`Iterator::take`] when only the cheapest few are needed.
///
/// If `start == goal`, the first (and only) result is the empty path with a
/// cost of zero and no edges.
///
/// Edge costs must be non-negative integers. Extensions whose total cost would
/// overflow the cost type are skipped instead of wrapping, which terminates the
/// iteration once no representable path remains on the frontier.
///
/// # Example
/// ```
/// use petgraph::{algo::simple_shortest_paths, prelude::*};
///
/// let mut graph = DiGraph::<(), u32>::new();
/// let a = graph.add_node(());
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// graph.extend_with_edges([(a, b, 1), (a, c, 2), (b, c, 1)]);
///
/// let paths: Vec<_> = simple_shortest_paths(&graph, a, c, |edge| *edge.weight())
///     .map(|path| (path.cost, path.edges))
///     .collect();
/// assert_eq!(paths[0].0, 2); // a -> c or a -> b -> c
/// assert_eq!(paths[1].0, 2);
/// ```
pub fn simple_shortest_paths<G, F, K>(
    graph: G,
    start: G::NodeId,
    goal: G::NodeId,
    edge_cost: F,
) -> SimpleShortestPaths<G, F, K>
where
    G: IntoEdges,
    G::NodeId: Eq + Hash,
    G::EdgeId: Ord,
    F: FnMut(G::EdgeRef) -> K,
    K: BoundedMeasure + Copy,
{
    let mut visited = HashSet::new();
    visited.insert(start);
    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry(PathState {
        cost: K::default(),
        nodes: vec![start],
        edges: Vec::new(),
        visited,
    }));

    SimpleShortestPaths {
        graph,
        edge_cost,
        goal,
        heap,
    }
}

#[derive(Clone, Debug)]
struct PathState<N, E, K> {
    cost: K,
    nodes: Vec<N>,
    edges: Vec<E>,
    visited: HashSet<N>,
}

/// Orders frontier paths for use with a [`BinaryHeap`] as a min-heap keyed by
/// `(cost, edge identifier sequence)`.
#[derive(Clone, Debug)]
struct HeapEntry<N, E, K>(PathState<N, E, K>);

impl<N, E: PartialEq, K: PartialEq> PartialEq for HeapEntry<N, E, K> {
    fn eq(&self, other: &Self) -> bool {
        self.0.cost == other.0.cost && self.0.edges == other.0.edges
    }
}

impl<N, E: Eq, K: PartialEq> Eq for HeapEntry<N, E, K> {}

impl<N, E, K> PartialOrd for HeapEntry<N, E, K>
where
    E: Ord,
    K: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<N, E, K> Ord for HeapEntry<N, E, K>
where
    E: Ord,
    K: PartialOrd,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match other.0.cost.partial_cmp(&self.0.cost) {
            Some(Ordering::Equal) => other.0.edges.cmp(&self.0.edges),
            ordering => ordering.unwrap_or(Ordering::Equal),
        }
    }
}
