//! Transpose (reverse) directed graphs.
//!
//! This module provides a [`transpose`] function that creates a new graph with
//! every edge reversed, together with mappings from the original node and edge
//! indices to the new indices. Self-loops and parallel edges are preserved
//! (never merged).
//!
//! Both [`Graph`] and [`StableGraph`] are supported. For `StableGraph`, deleted
//! slots are **not** treated as valid nodes or edges: they are skipped when
//! building the transposed graph and appear as `None` in the mappings.
//!
//! # Fallible cloning
//!
//! Node and edge weights are copied through the [`TryClone`] trait. A blanket
//! implementation is provided for every [`Clone`] type (with [`Infallible`] as
//! the error type), so the common case is infallible. If any weight fails to
//! clone, the whole operation returns an error and no partial graph is
//! produced.
//!
//! The original graph is never mutated: neither its indices nor its weights
//! are changed.
//!
//! # Examples
//!
//! ```
//! use petgraph::graph::Graph;
//! use petgraph::transpose::transpose;
//!
//! let mut g: Graph<&str, &str> = Graph::new();
//! let a = g.add_node("a");
//! let b = g.add_node("b");
//! let e1 = g.add_edge(a, b, "e1");
//! let e2 = g.add_edge(a, b, "e2"); // parallel edge
//! let e3 = g.add_edge(b, b, "e3"); // self-loop
//!
//! let result = transpose(&g).unwrap();
//! let t = &result.graph;
//!
//! // The transposed graph has the same node and edge counts.
//! assert_eq!(t.node_count(), 2);
//! assert_eq!(t.edge_count(), 3);
//!
//! // The parallel edges are preserved: now both go from b to a.
//! let new_a = result.get_node(a).unwrap();
//! let new_b = result.get_node(b).unwrap();
//! assert!(t.find_edge(new_b, new_a).is_some());
//!
//! // The self-loop is preserved (now a self-loop on new_b).
//! assert!(t.find_edge(new_b, new_b).is_some());
//!
//! // The edge mapping traces each original edge to its new index.
//! for old in g.edge_indices() {
//!     let new = result.get_edge(old).unwrap();
//!     let (s, tgt) = t.edge_endpoints(new).unwrap();
//!     std::println!("edge {:?} -> {:?}: {:?} -> {:?}", old, new, s, tgt);
//! }
//! ```
//!
//! # Undirected graphs are rejected
//!
//! Transpose is only defined for directed graphs. Calling it on an undirected
//! graph is a compile error:
//!
//! ```compile_fail
//! use petgraph::graph::UnGraph;
//! use petgraph::transpose::transpose;
//!
//! let g: UnGraph<&str, &str> = UnGraph::new_undirected();
//! // This fails to compile: transpose is only implemented for Directed graphs.
//! let _ = transpose(&g);
//! ```

use alloc::vec;
use alloc::vec::Vec;
use core::{convert::Infallible, fmt};

use crate::{
    Directed,
    graph::{EdgeIndex, Graph, IndexType, NodeIndex},
    visit::EdgeRef,
};

#[cfg(feature = "stable_graph")]
use crate::visit::{EdgeIndexable, IntoEdgeReferences, NodeIndexable};

#[cfg(feature = "stable_graph")]
use crate::stable_graph::StableGraph;

/// A trait for types that can be cloned fallibly.
///
/// This is used by the transpose functions to copy node and edge weights.
/// A blanket implementation is provided for all [`Clone`] types, with
/// [`Infallible`] as the error type.
///
/// # Examples
///
/// ```
/// use petgraph::transpose::TryClone;
///
/// #[derive(Clone)]
/// struct Weight(u32);
///
/// // `Weight` implements `TryClone` with `Infallible` error via the blanket impl.
/// let w = Weight(42);
/// let _ = w.try_clone().unwrap();
/// ```
pub trait TryClone: Sized {
    /// The error type returned when cloning fails.
    type Error;

    /// Try to clone this value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value cannot be cloned.
    fn try_clone(&self) -> Result<Self, Self::Error>;
}

impl<T: Clone> TryClone for T {
    type Error = Infallible;

    #[inline]
    fn try_clone(&self) -> Result<Self, Infallible> {
        Ok(Clone::clone(self))
    }
}

/// An error that can occur when transposing a graph.
///
/// # Examples
///
/// ```
/// use petgraph::transpose::TransposeError;
///
/// let err: TransposeError<&str, &str> = TransposeError::NodeClone("boom");
/// assert_eq!(err.to_string(), "failed to clone node weight: boom");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeError<NE, EE> {
    /// Cloning a node weight failed.
    NodeClone(NE),
    /// Cloning an edge weight failed.
    EdgeClone(EE),
}

impl<NE: fmt::Display, EE: fmt::Display> fmt::Display for TransposeError<NE, EE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransposeError::NodeClone(e) => write!(f, "failed to clone node weight: {e}"),
            TransposeError::EdgeClone(e) => write!(f, "failed to clone edge weight: {e}"),
        }
    }
}

impl<NE, EE> core::error::Error for TransposeError<NE, EE>
where
    NE: fmt::Debug + fmt::Display,
    EE: fmt::Debug + fmt::Display,
{
}

/// The result of transposing a graph.
///
/// Contains the transposed graph and mappings from the original node and edge
/// indices to the new indices.
///
/// For [`Graph`], every entry in the mappings is `Some`. For [`StableGraph`],
/// deleted slots are `None`.
#[derive(Debug, Clone)]
pub struct TransposedGraph<G, Ix = crate::graph::DefaultIx> {
    /// The transposed graph.
    pub graph: G,
    /// Mapping from original node indices to new node indices.
    ///
    /// Indexed by the original node index. `None` means the original node was
    /// deleted (only possible for `StableGraph`).
    pub node_map: Vec<Option<NodeIndex<Ix>>>,
    /// Mapping from original edge indices to new edge indices.
    ///
    /// Indexed by the original edge index. `None` means the original edge was
    /// deleted (only possible for `StableGraph`).
    pub edge_map: Vec<Option<EdgeIndex<Ix>>>,
}

impl<G, Ix: IndexType> TransposedGraph<G, Ix> {
    /// Look up the new node index for an original node index.
    ///
    /// Returns `None` if the original node was deleted (only possible for
    /// `StableGraph`) or if the index is out of bounds.
    pub fn get_node(&self, old: NodeIndex<Ix>) -> Option<NodeIndex<Ix>> {
        self.node_map.get(old.index()).copied().flatten()
    }

    /// Look up the new edge index for an original edge index.
    ///
    /// Returns `None` if the original edge was deleted (only possible for
    /// `StableGraph`) or if the index is out of bounds.
    pub fn get_edge(&self, old: EdgeIndex<Ix>) -> Option<EdgeIndex<Ix>> {
        self.edge_map.get(old.index()).copied().flatten()
    }
}

/// Transpose a directed graph: reverse all edges.
///
/// Returns a [`TransposedGraph`] containing the new graph and mappings from
/// the original node and edge indices to the new indices. Self-loops and
/// parallel edges are preserved.
///
/// The original graph is not mutated.
///
/// # Errors
///
/// Returns [`TransposeError::NodeClone`] if a node weight fails to clone, or
/// [`TransposeError::EdgeClone`] if an edge weight fails to clone. In either
/// case, no partial graph is returned.
///
/// # Examples
///
/// ```
/// use petgraph::graph::Graph;
/// use petgraph::transpose::transpose;
///
/// let mut g: Graph<&str, &str> = Graph::new();
/// let a = g.add_node("a");
/// let b = g.add_node("b");
/// g.add_edge(a, b, "e1");
/// g.add_edge(a, b, "e2");
/// g.add_edge(b, b, "e3");
///
/// let result = transpose(&g).unwrap();
/// assert_eq!(result.graph.node_count(), 2);
/// assert_eq!(result.graph.edge_count(), 3);
/// ```
pub fn transpose<N, E, Ix>(
    graph: &Graph<N, E, Directed, Ix>,
) -> Result<TransposedGraph<Graph<N, E, Directed, Ix>, Ix>, TransposeError<N::Error, E::Error>>
where
    N: TryClone,
    E: TryClone,
    Ix: IndexType,
{
    transpose_graph(graph)
}

/// Transpose a directed [`Graph`]: reverse all edges.
///
/// This is the same as [`transpose`], but named explicitly for clarity.
///
/// # Errors
///
/// Returns [`TransposeError`] if any weight fails to clone.
pub fn transpose_graph<N, E, Ix>(
    graph: &Graph<N, E, Directed, Ix>,
) -> Result<TransposedGraph<Graph<N, E, Directed, Ix>, Ix>, TransposeError<N::Error, E::Error>>
where
    N: TryClone,
    E: TryClone,
    Ix: IndexType,
{
    let mut new_graph = Graph::with_capacity(graph.node_count(), graph.edge_count());
    let mut node_map = vec![None; graph.node_count()];
    let mut edge_map = vec![None; graph.edge_count()];

    for old_idx in graph.node_indices() {
        let weight = graph.node_weight(old_idx).expect("node weight exists");
        let cloned = weight.try_clone().map_err(TransposeError::NodeClone)?;
        let new_idx = new_graph.add_node(cloned);
        node_map[old_idx.index()] = Some(new_idx);
    }

    for edge in graph.edge_references() {
        let old_idx = edge.id();
        let (source, target) = (edge.source(), edge.target());
        let weight = edge.weight();
        let cloned = weight.try_clone().map_err(TransposeError::EdgeClone)?;
        let new_source = node_map[source.index()].expect("source node mapped");
        let new_target = node_map[target.index()].expect("target node mapped");
        let new_idx = new_graph.add_edge(new_target, new_source, cloned);
        edge_map[old_idx.index()] = Some(new_idx);
    }

    Ok(TransposedGraph {
        graph: new_graph,
        node_map,
        edge_map,
    })
}

/// Transpose a directed [`StableGraph`]: reverse all edges.
///
/// Deleted slots are skipped: they are not added to the new graph and appear
/// as `None` in the mappings.
///
/// # Errors
///
/// Returns [`TransposeError`] if any weight fails to clone.
///
/// # Examples
///
/// ```
/// use petgraph::stable_graph::StableGraph;
/// use petgraph::transpose::transpose_stable_graph;
///
/// let mut g: StableGraph<&str, &str> = StableGraph::new();
/// let a = g.add_node("a");
/// let b = g.add_node("b");
/// g.add_edge(a, b, "e1");
/// let deleted = g.add_node("deleted");
/// g.remove_node(deleted);
///
/// let result = transpose_stable_graph(&g).unwrap();
/// assert_eq!(result.graph.node_count(), 2);
/// // The deleted node does not appear in the mapping.
/// assert!(result.get_node(deleted).is_none());
/// ```
#[cfg(feature = "stable_graph")]
pub fn transpose_stable_graph<N, E, Ix>(
    graph: &StableGraph<N, E, Directed, Ix>,
) -> Result<
    TransposedGraph<StableGraph<N, E, Directed, Ix>, Ix>,
    TransposeError<N::Error, E::Error>,
>
where
    N: TryClone,
    E: TryClone,
    Ix: IndexType,
{
    let mut new_graph = StableGraph::with_capacity(graph.node_count(), graph.edge_count());
    let mut node_map = vec![None; graph.node_bound()];
    let mut edge_map = vec![None; graph.edge_bound()];

    for old_idx in graph.node_indices() {
        let weight = graph.node_weight(old_idx).expect("node weight exists");
        let cloned = weight.try_clone().map_err(TransposeError::NodeClone)?;
        let new_idx = new_graph.add_node(cloned);
        node_map[old_idx.index()] = Some(new_idx);
    }

    for edge in graph.edge_references() {
        let old_idx = edge.id();
        let (source, target) = (edge.source(), edge.target());
        let weight = edge.weight();
        let cloned = weight.try_clone().map_err(TransposeError::EdgeClone)?;
        let new_source = node_map[source.index()].expect("source node mapped");
        let new_target = node_map[target.index()].expect("target node mapped");
        let new_idx = new_graph.add_edge(new_target, new_source, cloned);
        edge_map[old_idx.index()] = Some(new_idx);
    }

    Ok(TransposedGraph {
        graph: new_graph,
        node_map,
        edge_map,
    })
}

/// Inherent transpose method for directed [`Graph`].
impl<N, E, Ix> Graph<N, E, Directed, Ix>
where
    N: TryClone,
    E: TryClone,
    Ix: IndexType,
{
    /// Transpose the graph: reverse all edges.
    ///
    /// Returns a [`TransposedGraph`] containing the new graph and mappings
    /// from the original node and edge indices to the new indices.
    ///
    /// The original graph is not mutated.
    ///
    /// # Errors
    ///
    /// Returns [`TransposeError`] if any weight fails to clone.
    ///
    /// # Examples
    ///
    /// ```
    /// use petgraph::graph::Graph;
    ///
    /// let mut g: Graph<&str, &str> = Graph::new();
    /// let a = g.add_node("a");
    /// let b = g.add_node("b");
    /// g.add_edge(a, b, "e1");
    ///
    /// let result = g.transpose().unwrap();
    /// assert_eq!(result.graph.node_count(), 2);
    /// assert_eq!(result.graph.edge_count(), 1);
    /// ```
    pub fn transpose(
        &self,
    ) -> Result<TransposedGraph<Graph<N, E, Directed, Ix>, Ix>, TransposeError<N::Error, E::Error>>
    {
        transpose_graph(self)
    }
}

/// Inherent transpose method for directed [`StableGraph`].
#[cfg(feature = "stable_graph")]
impl<N, E, Ix> StableGraph<N, E, Directed, Ix>
where
    N: TryClone,
    E: TryClone,
    Ix: IndexType,
{
    /// Transpose the graph: reverse all edges.
    ///
    /// Deleted slots are skipped and appear as `None` in the mappings.
    ///
    /// The original graph is not mutated.
    ///
    /// # Errors
    ///
    /// Returns [`TransposeError`] if any weight fails to clone.
    ///
    /// # Examples
    ///
    /// ```
    /// use petgraph::stable_graph::StableGraph;
    ///
    /// let mut g: StableGraph<&str, &str> = StableGraph::new();
    /// let a = g.add_node("a");
    /// let b = g.add_node("b");
    /// g.add_edge(a, b, "e1");
    ///
    /// let result = g.transpose().unwrap();
    /// assert_eq!(result.graph.node_count(), 2);
    /// assert_eq!(result.graph.edge_count(), 1);
    /// ```
    pub fn transpose(
        &self,
    ) -> Result<
        TransposedGraph<StableGraph<N, E, Directed, Ix>, Ix>,
        TransposeError<N::Error, E::Error>,
    > {
        transpose_stable_graph(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn transpose_preserves_parallel_edges_and_self_loop() {
        let mut g: Graph<&str, &str> = Graph::new();
        let a = g.add_node("a");
        let b = g.add_node("b");
        let e1 = g.add_edge(a, b, "e1");
        let e2 = g.add_edge(a, b, "e2");
        let e3 = g.add_edge(b, b, "e3");

        let result = transpose(&g).unwrap();
        let t = &result.graph;

        assert_eq!(t.node_count(), 2);
        assert_eq!(t.edge_count(), 3);

        let new_a = result.get_node(a).unwrap();
        let new_b = result.get_node(b).unwrap();

        // Parallel edges preserved: both go from b to a.
        assert!(t.find_edge(new_b, new_a).is_some());
        let b_to_a: Vec<_> = t
            .edges_connecting(new_b, new_a)
            .map(|e| *e.weight())
            .collect();
        assert_eq!(b_to_a.len(), 2);
        assert!(b_to_a.contains(&"e1"));
        assert!(b_to_a.contains(&"e2"));

        // Self-loop preserved.
        assert!(t.find_edge(new_b, new_b).is_some());
        let self_loops: Vec<_> = t
            .edges_connecting(new_b, new_b)
            .map(|e| *e.weight())
            .collect();
        assert_eq!(self_loops.len(), 1);
        assert_eq!(self_loops[0], "e3");

        // Edge mapping traces each original edge to its new index.
        for old in [e1, e2, e3] {
            let new = result.get_edge(old).unwrap();
            let (s, tgt) = t.edge_endpoints(new).unwrap();
            std::println!("edge {:?} -> {:?}: {:?} -> {:?}", old, new, s, tgt);
        }

        // Original graph is unchanged.
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 3);
        assert_eq!(g.node_weight(a), Some(&"a"));
        assert_eq!(g.node_weight(b), Some(&"b"));
    }

    #[test]
    fn transpose_empty_graph() {
        let g: Graph<(), ()> = Graph::new();
        let result = transpose(&g).unwrap();
        assert_eq!(result.graph.node_count(), 0);
        assert_eq!(result.graph.edge_count(), 0);
        assert!(result.node_map.is_empty());
        assert!(result.edge_map.is_empty());
    }

    #[test]
    fn transpose_original_unchanged() {
        let mut g: Graph<i32, i32> = Graph::new();
        let a = g.add_node(1);
        let b = g.add_node(2);
        g.add_edge(a, b, 10);
        g.add_edge(b, a, 20);

        let _ = transpose(&g).unwrap();

        assert_eq!(g.node_weight(a), Some(&1));
        assert_eq!(g.node_weight(b), Some(&2));
        assert_eq!(g.edge_weight(g.find_edge(a, b).unwrap()), Some(&10));
        assert_eq!(g.edge_weight(g.find_edge(b, a).unwrap()), Some(&20));
    }

    #[test]
    fn transpose_method_works() {
        let mut g: Graph<&str, &str> = Graph::new();
        let a = g.add_node("a");
        let b = g.add_node("b");
        g.add_edge(a, b, "e1");

        let result = g.transpose().unwrap();
        assert_eq!(result.graph.node_count(), 2);
        assert_eq!(result.graph.edge_count(), 1);
        let new_a = result.get_node(a).unwrap();
        let new_b = result.get_node(b).unwrap();
        assert!(result.graph.find_edge(new_b, new_a).is_some());
    }

    /// A weight that always fails to clone.
    #[derive(Debug)]
    struct FailClone;

    impl TryClone for FailClone {
        type Error = &'static str;

        fn try_clone(&self) -> Result<Self, Self::Error> {
            Err("clone failed")
        }
    }

    #[test]
    fn transpose_node_clone_failure_returns_error() {
        let mut g: Graph<FailClone, ()> = Graph::new();
        g.add_node(FailClone);
        g.add_node(FailClone);

        let result = transpose(&g);
        assert!(result.is_err());
        match result {
            Err(TransposeError::NodeClone("clone failed")) => {}
            _ => panic!("expected NodeClone error"),
        }
    }

    #[test]
    fn transpose_edge_clone_failure_returns_error() {
        let mut g: Graph<(), FailClone> = Graph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        g.add_edge(a, b, FailClone);

        let result = transpose(&g);
        assert!(result.is_err());
        match result {
            Err(TransposeError::EdgeClone("clone failed")) => {}
            _ => panic!("expected EdgeClone error"),
        }
    }

    /// A weight that can be configured to fail cloning.
    #[derive(Debug)]
    struct MaybeClone {
        fail: bool,
    }

    impl TryClone for MaybeClone {
        type Error = &'static str;

        fn try_clone(&self) -> Result<Self, Self::Error> {
            if self.fail {
                Err("clone failed")
            } else {
                Ok(MaybeClone { fail: false })
            }
        }
    }

    #[test]
    fn transpose_clone_failure_returns_no_partial_graph() {
        // The first node clones successfully, but the second fails.
        // No partial graph should be returned.
        let mut g: Graph<MaybeClone, ()> = Graph::new();
        g.add_node(MaybeClone { fail: false });
        g.add_node(MaybeClone { fail: true });

        let result = transpose(&g);
        assert!(result.is_err());
        match result {
            Err(TransposeError::NodeClone("clone failed")) => {}
            _ => panic!("expected NodeClone error"),
        }
    }

    #[cfg(feature = "stable_graph")]
    #[test]
    fn transpose_stable_graph_preserves_edges() {
        use crate::stable_graph::StableGraph;

        let mut g: StableGraph<&str, &str> = StableGraph::new();
        let a = g.add_node("a");
        let b = g.add_node("b");
        let e1 = g.add_edge(a, b, "e1");
        let e2 = g.add_edge(a, b, "e2");
        let e3 = g.add_edge(b, b, "e3");

        let result = transpose_stable_graph(&g).unwrap();
        let t = &result.graph;

        assert_eq!(t.node_count(), 2);
        assert_eq!(t.edge_count(), 3);

        let new_a = result.get_node(a).unwrap();
        let new_b = result.get_node(b).unwrap();

        // Parallel edges preserved: both go from b to a.
        let b_to_a: Vec<_> = t.edges_connecting(new_b, new_a).map(|e| *e.weight()).collect();
        assert_eq!(b_to_a.len(), 2);

        // Self-loop preserved.
        let self_loops: Vec<_> = t.edges_connecting(new_b, new_b).map(|e| *e.weight()).collect();
        assert_eq!(self_loops.len(), 1);

        // Edge mapping traces each original edge to its new index.
        for old in [e1, e2, e3] {
            let new = result.get_edge(old).unwrap();
            let (s, tgt) = t.edge_endpoints(new).unwrap();
            std::println!("edge {:?} -> {:?}: {:?} -> {:?}", old, new, s, tgt);
        }
    }

    #[cfg(feature = "stable_graph")]
    #[test]
    fn transpose_stable_graph_deleted_node_excluded() {
        use crate::stable_graph::StableGraph;

        let mut g: StableGraph<&str, &str> = StableGraph::new();
        let a = g.add_node("a");
        let b = g.add_node("b");
        g.add_edge(a, b, "e1");
        let deleted = g.add_node("deleted");
        g.remove_node(deleted);

        let result = transpose_stable_graph(&g).unwrap();

        // The deleted node must not appear in the mapping.
        assert!(result.get_node(deleted).is_none());

        // The valid nodes are mapped.
        assert!(result.get_node(a).is_some());
        assert!(result.get_node(b).is_some());

        // The transposed graph has only 2 nodes.
        assert_eq!(result.graph.node_count(), 2);
        assert_eq!(result.graph.edge_count(), 1);
    }

    #[cfg(feature = "stable_graph")]
    #[test]
    fn transpose_stable_graph_deleted_edge_excluded() {
        use crate::stable_graph::StableGraph;

        let mut g: StableGraph<&str, &str> = StableGraph::new();
        let a = g.add_node("a");
        let b = g.add_node("b");
        let e1 = g.add_edge(a, b, "e1");
        let e2 = g.add_edge(a, b, "e2");
        g.remove_edge(e1);

        let result = transpose_stable_graph(&g).unwrap();

        // The deleted edge must not appear in the mapping.
        assert!(result.get_edge(e1).is_none());
        // The valid edge is mapped.
        assert!(result.get_edge(e2).is_some());

        // The transposed graph has only 1 edge.
        assert_eq!(result.graph.edge_count(), 1);
    }
}
