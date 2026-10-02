//! Transpose a directed graph into a new graph with every edge reversed.
//!
//! The functions in this module build a **new** graph in which every edge
//! `a -> b` of the source graph appears as `b -> a`, while node and edge
//! weights are copied through an explicit, user-controlled clone step.
//! Parallel edges and self-loops are preserved (never merged), and the
//! source graph is left completely untouched.
//!
//! Besides the transposed graph, the functions return the mapping from the
//! original [`NodeIndex`] / [`EdgeIndex`] to the indices of the copied
//! nodes and edges, so every copied item stays traceable back to its
//! origin.
//!
//! * [`transpose_copy`] / [`try_transpose_copy`] work on [`Graph`].
//! * [`transpose_stable_copy`] / [`try_transpose_stable_copy`] work on
//!   [`StableGraph`](crate::stable_graph::StableGraph); vacant slots left
//!   by removed nodes or edges are **not** treated as live elements and
//!   map to `None` in the returned index maps.
//!
//! Only directed graphs are supported; passing an undirected graph is
//! rejected with [`TransposeError::Undirected`].
//!
//! The `try_*` variants take fallible clone closures. If cloning any node
//! or edge weight fails, the whole operation fails and no partially built
//! graph is returned.
//!
//! # Example
//!
//! ```
//! use petgraph::graph::DiGraph;
//! use petgraph::transpose::transpose_copy;
//!
//! let mut g = DiGraph::new();
//! let a = g.add_node("a");
//! let b = g.add_node("b");
//! let e0 = g.add_edge(a, b, 1); // parallel edge 1
//! let e1 = g.add_edge(a, b, 2); // parallel edge 2
//! let e2 = g.add_edge(b, b, 3); // self-loop
//!
//! let t = transpose_copy(&g).unwrap();
//!
//! // Both parallel edges are reversed, the self-loop stays a self-loop.
//! let (s0, t0) = t.graph.edge_endpoints(t.edge_map[e0.index()]).unwrap();
//! assert_eq!((s0, t0), (t.node_map[b.index()], t.node_map[a.index()]));
//! let (s1, t1) = t.graph.edge_endpoints(t.edge_map[e1.index()]).unwrap();
//! assert_eq!((s1, t1), (t.node_map[b.index()], t.node_map[a.index()]));
//! let (s2, t2) = t.graph.edge_endpoints(t.edge_map[e2.index()]).unwrap();
//! assert_eq!((s2, t2), (t.node_map[b.index()], t.node_map[b.index()]));
//!
//! // The original graph is unchanged.
//! assert_eq!(g.edge_endpoints(e0), Some((a, b)));
//! ```

#[cfg(feature = "stable_graph")]
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;
use core::fmt;

use crate::graph::{DefaultIx, EdgeIndex, Graph, IndexType, NodeIndex};
use crate::{Directed, EdgeType};

#[cfg(feature = "stable_graph")]
use crate::stable_graph::StableGraph;
#[cfg(feature = "stable_graph")]
use crate::visit::{EdgeIndexable, NodeIndexable};

/// Error returned by the transpose-copy functions in this module.
///
/// The type parameter `E` is the error type produced by the user supplied
/// clone closures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransposeError<E> {
    /// The source graph is undirected; only directed graphs can be
    /// transposed into a reversed copy.
    Undirected,
    /// Cloning the weight of the node with the given (source graph) index
    /// failed. No partial graph is returned.
    NodeClone {
        /// Index of the node whose weight failed to clone.
        index: usize,
        /// The error produced by the clone closure.
        error: E,
    },
    /// Cloning the weight of the edge with the given (source graph) index
    /// failed. No partial graph is returned.
    EdgeClone {
        /// Index of the edge whose weight failed to clone.
        index: usize,
        /// The error produced by the clone closure.
        error: E,
    },
}

impl<E: fmt::Display> fmt::Display for TransposeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransposeError::Undirected => {
                write!(f, "cannot transpose-copy an undirected graph")
            }
            TransposeError::NodeClone { index, error } => {
                write!(f, "failed to clone weight of node {index}: {error}")
            }
            TransposeError::EdgeClone { index, error } => {
                write!(f, "failed to clone weight of edge {index}: {error}")
            }
        }
    }
}

impl<E> core::error::Error for TransposeError<E>
where
    E: core::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            TransposeError::Undirected => None,
            TransposeError::NodeClone { error, .. } | TransposeError::EdgeClone { error, .. } => {
                Some(error)
            }
        }
    }
}

/// Result of transposing a [`Graph`]: the reversed copy together with the
/// index mappings from the original graph.
///
/// * `node_map[i]` is the node in `graph` that corresponds to the original
///   node `NodeIndex::new(i)`.
/// * `edge_map[i]` is the edge in `graph` that corresponds to the original
///   edge `EdgeIndex::new(i)` (with its direction reversed).
#[derive(Debug, Clone)]
pub struct TransposedGraph<N, E, Ix: IndexType = DefaultIx> {
    /// The transposed graph: every edge of the original graph reversed.
    pub graph: Graph<N, E, Directed, Ix>,
    /// Maps each original node index to the corresponding new node index.
    pub node_map: Vec<NodeIndex<Ix>>,
    /// Maps each original edge index to the corresponding new (reversed)
    /// edge index.
    pub edge_map: Vec<EdgeIndex<Ix>>,
}

/// Result of transposing a [`StableGraph`](crate::stable_graph::StableGraph).
///
/// The maps are indexed by the original node/edge index and have one entry
/// per slot of the source graph (up to its node/edge bound). Slots that
/// were vacant in the source graph — nodes or edges that had been removed —
/// map to `None`; they are never treated as live elements and never appear
/// in the transposed graph.
#[cfg(feature = "stable_graph")]
#[derive(Debug, Clone)]
pub struct TransposedStableGraph<N, E, Ix: IndexType = DefaultIx> {
    /// The transposed graph: every edge of the original graph reversed.
    pub graph: StableGraph<N, E, Directed, Ix>,
    /// Maps each original node index to the corresponding new node index,
    /// or `None` for vacant (removed) slots.
    pub node_map: Vec<Option<NodeIndex<Ix>>>,
    /// Maps each original edge index to the corresponding new (reversed)
    /// edge index, or `None` for vacant (removed) slots.
    pub edge_map: Vec<Option<EdgeIndex<Ix>>>,
}

/// Transpose the directed graph `g` into a new graph with all edges
/// reversed, cloning node and edge weights with the given closures.
///
/// Every edge `a -> b` becomes `b -> a` in the new graph. Parallel edges
/// and self-loops are preserved one-to-one; nothing is merged. The clone
/// closures are called exactly once per live node/edge weight, in index
/// order. If either closure fails, the whole operation fails with
/// [`TransposeError::NodeClone`] or [`TransposeError::EdgeClone`] and no
/// partially built graph is returned. The source graph is not modified.
///
/// Returns [`TransposeError::Undirected`] if `g` is undirected.
pub fn try_transpose_copy<N, E, Ty, Ix, N2, E2, NF, EF, CE>(
    g: &Graph<N, E, Ty, Ix>,
    mut node_clone: NF,
    mut edge_clone: EF,
) -> Result<TransposedGraph<N2, E2, Ix>, TransposeError<CE>>
where
    Ty: EdgeType,
    Ix: IndexType,
    NF: FnMut(&N) -> Result<N2, CE>,
    EF: FnMut(&E) -> Result<E2, CE>,
{
    if !Ty::is_directed() {
        return Err(TransposeError::Undirected);
    }

    let mut graph = Graph::with_capacity(g.node_count(), g.edge_count());

    let mut node_map = Vec::with_capacity(g.node_count());
    for i in g.node_indices() {
        let weight = node_clone(&g[i]).map_err(|error| TransposeError::NodeClone {
            index: i.index(),
            error,
        })?;
        node_map.push(graph.add_node(weight));
    }

    let mut edge_map = Vec::with_capacity(g.edge_count());
    for e in g.edge_indices() {
        let (source, target) = g
            .edge_endpoints(e)
            .expect("edge index produced by edge_indices must be valid");
        let weight = edge_clone(&g[e]).map_err(|error| TransposeError::EdgeClone {
            index: e.index(),
            error,
        })?;
        // Reverse the edge; self-loops and parallel edges are kept as-is.
        let new_edge = graph.add_edge(node_map[target.index()], node_map[source.index()], weight);
        edge_map.push(new_edge);
    }

    Ok(TransposedGraph {
        graph,
        node_map,
        edge_map,
    })
}

/// Transpose the directed graph `g` into a new graph with all edges
/// reversed, cloning the weights with [`Clone`].
///
/// This is the infallible-clone shorthand for [`try_transpose_copy`]; the
/// only possible error is [`TransposeError::Undirected`]. The source graph
/// is not modified.
pub fn transpose_copy<N, E, Ty, Ix>(
    g: &Graph<N, E, Ty, Ix>,
) -> Result<TransposedGraph<N, E, Ix>, TransposeError<Infallible>>
where
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
{
    try_transpose_copy(
        g,
        |n| Ok::<_, Infallible>(n.clone()),
        |e| Ok::<_, Infallible>(e.clone()),
    )
}

/// Transpose the directed [`StableGraph`](crate::stable_graph::StableGraph)
/// `g` into a new graph with all edges reversed, cloning node and edge
/// weights with the given closures.
///
/// Vacant slots (nodes or edges removed from `g`) are skipped: they are
/// not cloned, not added to the new graph, and their entries in the
/// returned maps are `None`. Edges that were removed together with an
/// endpoint no longer exist in `g` and therefore map to `None` as well.
/// Parallel edges and self-loops are preserved one-to-one. If either clone
/// closure fails, the whole operation fails and no partially built graph
/// is returned. The source graph is not modified.
///
/// Returns [`TransposeError::Undirected`] if `g` is undirected.
#[cfg(feature = "stable_graph")]
pub fn try_transpose_stable_copy<N, E, Ty, Ix, N2, E2, NF, EF, CE>(
    g: &StableGraph<N, E, Ty, Ix>,
    mut node_clone: NF,
    mut edge_clone: EF,
) -> Result<TransposedStableGraph<N2, E2, Ix>, TransposeError<CE>>
where
    Ty: EdgeType,
    Ix: IndexType,
    NF: FnMut(&N) -> Result<N2, CE>,
    EF: FnMut(&E) -> Result<E2, CE>,
{
    if !Ty::is_directed() {
        return Err(TransposeError::Undirected);
    }

    let node_bound = g.node_bound();
    let edge_bound = g.edge_bound();
    let mut graph = StableGraph::with_capacity(node_bound, edge_bound);

    // One map entry per slot; vacant slots stay `None`.
    let mut node_map = vec![None; node_bound];
    for i in g.node_indices() {
        let weight = g
            .node_weight(i)
            .expect("node index produced by node_indices must be occupied");
        let weight = node_clone(weight).map_err(|error| TransposeError::NodeClone {
            index: i.index(),
            error,
        })?;
        node_map[i.index()] = Some(graph.add_node(weight));
    }

    let mut edge_map = vec![None; edge_bound];
    for e in g.edge_indices() {
        let (source, target) = g
            .edge_endpoints(e)
            .expect("edge index produced by edge_indices must be valid");
        let weight = g
            .edge_weight(e)
            .expect("edge index produced by edge_indices must be occupied");
        let weight = edge_clone(weight).map_err(|error| TransposeError::EdgeClone {
            index: e.index(),
            error,
        })?;
        // Reverse the edge; self-loops and parallel edges are kept as-is.
        let new_source =
            node_map[target.index()].expect("endpoint of a live edge must be a live node");
        let new_target =
            node_map[source.index()].expect("endpoint of a live edge must be a live node");
        edge_map[e.index()] = Some(graph.add_edge(new_source, new_target, weight));
    }

    Ok(TransposedStableGraph {
        graph,
        node_map,
        edge_map,
    })
}

/// Transpose the directed [`StableGraph`](crate::stable_graph::StableGraph)
/// `g` into a new graph with all edges reversed, cloning the weights with
/// [`Clone`].
///
/// This is the infallible-clone shorthand for
/// [`try_transpose_stable_copy`]; the only possible error is
/// [`TransposeError::Undirected`]. Vacant slots map to `None`. The source
/// graph is not modified.
#[cfg(feature = "stable_graph")]
pub fn transpose_stable_copy<N, E, Ty, Ix>(
    g: &StableGraph<N, E, Ty, Ix>,
) -> Result<TransposedStableGraph<N, E, Ix>, TransposeError<Infallible>>
where
    N: Clone,
    E: Clone,
    Ty: EdgeType,
    Ix: IndexType,
{
    try_transpose_stable_copy(
        g,
        |n| Ok::<_, Infallible>(n.clone()),
        |e| Ok::<_, Infallible>(e.clone()),
    )
}
