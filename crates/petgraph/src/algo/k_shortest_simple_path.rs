use alloc::{collections::BinaryHeap, vec::Vec};
use core::{cmp::Ordering, hash::Hash};

use hashbrown::HashSet;

use crate::{
    Direction::Outgoing,
    algo::Measure,
    visit::{EdgeRef, IntoEdgesDirected},
};

/// A [`Measure`] for non-negative integer edge costs whose sums can be
/// checked for overflow.
///
/// This trait is implemented for all unsigned integer types. It is used by
/// [`k_shortest_simple_paths`] to silently discard paths whose total cost
/// would overflow the cost type instead of panicking or wrapping around.
pub trait CheckedMeasure: Measure + Copy {
    /// Adds `rhs` to `self`, returning `None` if the result cannot be
    /// represented by the type.
    fn checked_add(self, rhs: Self) -> Option<Self>;
}

macro_rules! impl_checked_measure {
    ($($t:ty),* $(,)?) => {
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

impl_checked_measure!(u8, u16, u32, u64, u128, usize);

/// A partially explored simple path, used as a heap entry by
/// [`KShortestSimplePaths`].
struct PathCandidate<N, E, K> {
    cost: K,
    edges: Vec<E>,
    node: N,
    visited: HashSet<N>,
}

impl<N, E: Ord, K: PartialOrd> PartialEq for PathCandidate<N, E, K> {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.edges == other.edges
    }
}

impl<N, E: Ord, K: PartialOrd> Eq for PathCandidate<N, E, K> {}

impl<N, E: Ord, K: PartialOrd> PartialOrd for PathCandidate<N, E, K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<N, E: Ord, K: PartialOrd> Ord for PathCandidate<N, E, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed, so that the `BinaryHeap` max-heap yields the cheapest
        // candidate first. Ties in cost are broken by the lexicographic
        // order of the edge id sequence.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.edges.cmp(&self.edges))
    }
}

/// Lazy iterator over the cheapest simple paths between two nodes.
///
/// Created by [`k_shortest_simple_paths`]; see its documentation for details.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct KShortestSimplePaths<G, F, K>
where
    G: IntoEdgesDirected,
    F: FnMut(G::EdgeRef) -> K,
{
    graph: G,
    to: G::NodeId,
    edge_cost: F,
    heap: BinaryHeap<PathCandidate<G::NodeId, G::EdgeId, K>>,
}

impl<G, F, K> Iterator for KShortestSimplePaths<G, F, K>
where
    G: IntoEdgesDirected,
    G::NodeId: Eq + Hash,
    G::EdgeId: Ord,
    F: FnMut(G::EdgeRef) -> K,
    K: CheckedMeasure,
{
    /// The total cost of a path and the sequence of edge ids it traverses.
    type Item = (K, Vec<G::EdgeId>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(candidate) = self.heap.pop() {
            if candidate.node == self.to {
                return Some((candidate.cost, candidate.edges));
            }
            for edge in self.graph.edges_directed(candidate.node, Outgoing) {
                let target = edge.target();
                if candidate.visited.contains(&target) {
                    // Extending with this edge would repeat a node.
                    continue;
                }
                // Skip extensions whose total cost would overflow.
                let Some(cost) = candidate.cost.checked_add((self.edge_cost)(edge)) else {
                    continue;
                };
                let mut edges = candidate.edges.clone();
                edges.push(edge.id());
                let mut visited = candidate.visited.clone();
                visited.insert(target);
                self.heap.push(PathCandidate {
                    cost,
                    edges,
                    node: target,
                    visited,
                });
            }
        }
        None
    }
}

/// \[Generic\] Enumerate the cheapest simple paths from `from` to `to`, in
/// order.
///
/// A simple path is a path without repeating nodes. Different paths may share
/// edges and nodes freely, and parallel edges are treated as distinct: two
/// paths that visit the same nodes through different parallel edges are both
/// produced.
///
/// Paths are yielded in non-decreasing order of their total cost; ties are
/// broken by the lexicographic order of the sequence of edge ids the path
/// traverses. Each item is the total cost of the path paired with that
/// sequence of edge ids.
///
/// The iterator is lazy: it performs a best-first search over the partial
/// paths of the graph, so pulling the first path does **not** enumerate all
/// simple paths. Use [`Iterator::take`] to keep only the first **k** paths.
///
/// If `from` == `to`, the only simple path is the empty path, so a single
/// item with cost `K::default()` and no edges is produced.
///
/// Edge costs must be non-negative. Paths whose total cost would overflow the
/// cost type `K` are silently skipped (see [`CheckedMeasure`]).
///
/// # Arguments
/// * `graph`: an input graph. For undirected graphs every edge can be
///   traversed in both directions.
/// * `from`: the *start* node of the desired paths.
/// * `to`: the *goal* node of the desired paths.
/// * `edge_cost`: closure that returns the non-negative cost of a
///   particular edge.
///
/// # Returns
/// An iterator of `(cost, edge ids)` pairs, one per simple path from
/// `from` to `to`, ordered by cost and then by edge id sequence.
///
/// # Complexity
/// * Time complexity: producing the **k**-th path requires **O(k|V|)** heap
///   operations on graphs where every explored partial path can be extended
///   to `to`, each of cost **O(log(k|V||E|))**, plus one scan of the
///   outgoing edges of every popped partial path.
/// * Auxiliary space: proportional to the number of partial paths kept in
///   the heap, each of size **O(|V|)**.
///
/// where **|V|** is the number of nodes and **|E|** is the number of edges.
///
/// # Example
/// ```rust
/// use petgraph::{algo::k_shortest_simple_paths, graph::EdgeIndex, prelude::*};
///
/// let mut graph = DiGraph::<(), u32>::new();
/// let a = graph.add_node(());
/// let b = graph.add_node(());
/// let c = graph.add_node(());
///
/// // Two parallel edges from `a` to `b` with the same cost.
/// graph.extend_with_edges([(a, b, 1), (a, b, 1), (b, c, 1), (a, c, 5)]);
///
/// let paths: Vec<_> = k_shortest_simple_paths(&graph, a, c, |edge| *edge.weight()).collect();
///
/// // The two paths through the parallel edges tie in cost and are ordered
/// // by their edge id sequence.
/// assert_eq!(
///     paths,
///     vec![
///         (2, vec![EdgeIndex::new(0), EdgeIndex::new(2)]),
///         (2, vec![EdgeIndex::new(1), EdgeIndex::new(2)]),
///         (5, vec![EdgeIndex::new(3)]),
///     ]
/// );
/// ```
pub fn k_shortest_simple_paths<G, F, K>(
    graph: G,
    from: G::NodeId,
    to: G::NodeId,
    edge_cost: F,
) -> KShortestSimplePaths<G, F, K>
where
    G: IntoEdgesDirected,
    G::NodeId: Eq + Hash,
    G::EdgeId: Ord,
    F: FnMut(G::EdgeRef) -> K,
    K: CheckedMeasure,
{
    let mut visited = HashSet::new();
    visited.insert(from);
    let heap = BinaryHeap::from([PathCandidate {
        cost: K::default(),
        edges: Vec::new(),
        node: from,
        visited,
    }]);
    KShortestSimplePaths {
        graph,
        to,
        edge_cost,
        heap,
    }
}

