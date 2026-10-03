//! Reachability snapshots with an allowed-node set.
//!
//! This module provides [`ReachabilitySnapshot`], a self-contained record of
//! the nodes reachable from a start node when traversal is restricted to an
//! *allowed* set of nodes. Only edges whose two endpoints are both in the
//! allowed set are traversed, so a path that would pass through a forbidden
//! node is never recorded.
//!
//! The snapshot owns all of its data - the allowed set, the reachable node
//! set, and the concrete [`EdgeId`]s used to reach each node - so it does not
//! borrow the graph it was built from. Queries therefore reflect the state of
//! the graph at build time: later node or edge insertions and removals do not
//! affect the snapshot.
//!
//! # Determinism
//!
//! Ties between alternative paths are resolved by node and edge insertion
//! order. Edges are considered in the order they were added to the graph
//! (ascending edge index), and a breadth-first search records the first edge
//! that reaches each node. Parallel edges are distinguished by their concrete
//! [`EdgeId`].
//!
//! # Examples
//!
//! ```
//! use petgraph::Graph;
//! use petgraph::reachability::reachability_snapshot;
//!
//! let mut g = Graph::<&str, ()>::new();
//! let a = g.add_node("a");
//! let secret = g.add_node("secret");
//! let public = g.add_node("public");
//! let b = g.add_node("b");
//! g.extend_with_edges(&[(a, secret), (secret, b), (a, public), (public, b)]);
//!
//! // Forbid secret; only the public route is recorded.
//! let snap = reachability_snapshot(&g, a, [a, public, b]).unwrap();
//! assert!(snap.is_reachable(b));
//! let path = snap.path_to(b).unwrap();
//! assert_eq!(path.nodes(), &[a, public, b]);
//! ```
//!
//! [`EdgeId`]: visit::GraphBase::EdgeId

use alloc::{collections::VecDeque, vec, vec::Vec};

use fixedbitset::FixedBitSet;

use crate::{
    graph::{DefaultIx, EdgeIndex, IndexType, NodeIndex},
    visit::{EdgeRef, GraphBase, GraphProp, IntoEdges, NodeIndexable},
    EdgeType,
};

/// A graph that can determine whether a node identifier refers to a live node.
///
/// "Live" means the identifier is in bounds and, for graphs with removable
/// nodes (such as [`StableGraph`]), does not point at a vacated slot.
///
/// [`StableGraph`]: crate::stable_graph::StableGraph
pub trait NodeLiveness: GraphBase {
    /// Returns `true` if `id` is a live node in this graph.
    fn contains_node(&self, id: Self::NodeId) -> bool;
}

impl<N, E, Ty, Ix> NodeLiveness for crate::Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn contains_node(&self, id: NodeIndex<Ix>) -> bool {
        self.node_weight(id).is_some()
    }
}

#[cfg(feature = "stable_graph")]
impl<N, E, Ty, Ix> NodeLiveness for crate::stable_graph::StableGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn contains_node(&self, id: NodeIndex<Ix>) -> bool {
        self.node_weight(id).is_some()
    }
}

impl<G: NodeLiveness> NodeLiveness for &G {
    fn contains_node(&self, id: Self::NodeId) -> bool {
        (**self).contains_node(id)
    }
}

/// An error returned when building a [`ReachabilitySnapshot`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReachabilityError<Ix = DefaultIx> {
    /// The start node identifier does not refer to a live node.
    InvalidStart(NodeIndex<Ix>),
    /// The start node is not contained in the allowed set.
    StartNotAllowed(NodeIndex<Ix>),
    /// A node identifier in the allowed set does not refer to a live node.
    InvalidAllowedNode(NodeIndex<Ix>),
}

impl<Ix: IndexType> core::fmt::Display for ReachabilityError<Ix> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReachabilityError::InvalidStart(n) => {
                write!(f, "start node {n:?} is not a live node")
            }
            ReachabilityError::StartNotAllowed(n) => {
                write!(f, "start node {n:?} is not in the allowed set")
            }
            ReachabilityError::InvalidAllowedNode(n) => {
                write!(f, "allowed node {n:?} is not a live node")
            }
        }
    }
}

impl<Ix: IndexType> core::error::Error for ReachabilityError<Ix> {}

/// A self-contained snapshot of reachability from a single start node.
///
/// The snapshot records the allowed set, the reachable node set, and the
/// concrete edge used to first reach each node. It does not borrow the graph,
/// so later mutations of the graph do not affect queries.
///
/// See the [module-level documentation](self) for details.
#[derive(Clone, Debug)]
pub struct ReachabilitySnapshot<Ix = DefaultIx> {
    start: NodeIndex<Ix>,
    allowed: FixedBitSet,
    reachable: FixedBitSet,
    parent: Vec<Option<(NodeIndex<Ix>, EdgeIndex<Ix>)>>,
}

/// Build a [`ReachabilitySnapshot`] of nodes reachable from `start`,
/// traversing only edges whose endpoints are both in the `allowed` set.
///
/// The build validates every identifier it is given: the start must be a live
/// node and must be contained in the allowed set, and every allowed node must
/// be live. A deleted node slot (for graphs with removable nodes) is treated
/// as invalid.
///
/// # Errors
///
/// Returns an error if:
/// - `start` is not a live node ([`ReachabilityError::InvalidStart`]),
/// - `start` is not in the allowed set ([`ReachabilityError::StartNotAllowed`]),
/// - any identifier in `allowed` is not a live node
///   ([`ReachabilityError::InvalidAllowedNode`]).
///
/// # Determinism
///
/// Ties between alternative paths are resolved by insertion order: edges are
/// considered in ascending edge-index order, and a breadth-first search
/// records the first edge that reaches each node.
pub fn reachability_snapshot<G, Ix>(
    graph: G,
    start: NodeIndex<Ix>,
    allowed: impl IntoIterator<Item = NodeIndex<Ix>>,
) -> Result<ReachabilitySnapshot<Ix>, ReachabilityError<Ix>>
where
    G: IntoEdges<NodeId = NodeIndex<Ix>, EdgeId = EdgeIndex<Ix>>
        + NodeIndexable<NodeId = NodeIndex<Ix>>
        + GraphProp
        + NodeLiveness<NodeId = NodeIndex<Ix>>,
    Ix: IndexType,
{
    let bound = graph.node_bound();

    if !graph.contains_node(start) {
        return Err(ReachabilityError::InvalidStart(start));
    }

    let mut allowed_set = FixedBitSet::with_capacity(bound);
    for node in allowed {
        if !graph.contains_node(node) {
            return Err(ReachabilityError::InvalidAllowedNode(node));
        }
        allowed_set.put(node.index());
    }

    if !allowed_set.contains(start.index()) {
        return Err(ReachabilityError::StartNotAllowed(start));
    }

    // Build adjacency in edge-insertion order (ascending edge index).
    let mut adjacency: Vec<Vec<(NodeIndex<Ix>, EdgeIndex<Ix>)>> = vec![Vec::new(); bound];
    for edge in graph.edge_references() {
        let (s, t, eid) = (edge.source(), edge.target(), edge.id());
        adjacency[s.index()].push((t, eid));
        if !graph.is_directed() {
            adjacency[t.index()].push((s, eid));
        }
    }

    let mut reachable = FixedBitSet::with_capacity(bound);
    let mut parent: Vec<Option<(NodeIndex<Ix>, EdgeIndex<Ix>)>> = vec![None; bound];
    let mut queue = VecDeque::new();

    reachable.put(start.index());
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        for &(v, eid) in &adjacency[u.index()] {
            if !allowed_set.contains(v.index()) {
                continue;
            }
            // `put` returns the previous value: `false` means the bit was
            // newly set (first visit).
            if !reachable.put(v.index()) {
                parent[v.index()] = Some((u, eid));
                queue.push_back(v);
            }
        }
    }

    Ok(ReachabilitySnapshot {
        start,
        allowed: allowed_set,
        reachable,
        parent,
    })
}

impl<Ix: IndexType> ReachabilitySnapshot<Ix> {
    /// The start node of the snapshot.
    pub fn start(&self) -> NodeIndex<Ix> {
        self.start
    }

    /// Returns `true` if `node` is in the allowed set.
    pub fn is_allowed(&self, node: NodeIndex<Ix>) -> bool {
        self.allowed.contains(node.index())
    }

    /// Returns `true` if `node` is reachable from the start.
    ///
    /// A node is reachable if there is a path from the start to it using only
    /// allowed nodes and edges whose endpoints are both allowed.
    pub fn is_reachable(&self, node: NodeIndex<Ix>) -> bool {
        self.reachable.contains(node.index())
    }

    /// Iterate over the allowed node identifiers.
    pub fn allowed_nodes(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.allowed.ones().map(NodeIndex::new)
    }

    /// Iterate over the reachable node identifiers.
    ///
    /// The start node is included.
    pub fn reachable_nodes(&self) -> impl Iterator<Item = NodeIndex<Ix>> + '_ {
        self.reachable.ones().map(NodeIndex::new)
    }

    /// Iterate over the edges used to reach each reachable node.
    ///
    /// These are the edges of the reachability tree: for every reachable node
    /// except the start, the single edge that first connected it to the start
    /// during the build. Concrete [`EdgeId`]s are preserved, so parallel edges
    /// are distinguished.
    ///
    /// [`EdgeId`]: visit::GraphBase::EdgeId
    pub fn reachable_edges(&self) -> impl Iterator<Item = EdgeIndex<Ix>> + '_ {
        self.parent
            .iter()
            .filter_map(|p| p.as_ref().map(|(_, e)| *e))
    }

    /// Reconstruct the path from the start to `target`, if `target` is
    /// reachable.
    ///
    /// The path is expressed using the concrete node and edge identifiers
    /// recorded at build time. Returns `None` if `target` is not reachable
    /// (including when it is not a live or allowed node).
    pub fn path_to(&self, target: NodeIndex<Ix>) -> Option<ReachabilityPath<Ix>> {
        if target == self.start {
            return Some(ReachabilityPath {
                start: self.start,
                end: target,
                nodes: vec![self.start],
                edges: Vec::new(),
            });
        }
        if !self.reachable.contains(target.index()) {
            return None;
        }
        let mut nodes = vec![target];
        let mut edges = Vec::new();
        let mut cur = target;
        while cur != self.start {
            let (p, e) = self.parent[cur.index()]?;
            nodes.push(p);
            edges.push(e);
            cur = p;
        }
        nodes.reverse();
        edges.reverse();
        Some(ReachabilityPath {
            start: self.start,
            end: target,
            nodes,
            edges,
        })
    }
}

/// A path from the start node of a [`ReachabilitySnapshot`] to a reachable
/// target, expressed with concrete node and edge identifiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReachabilityPath<Ix = DefaultIx> {
    start: NodeIndex<Ix>,
    end: NodeIndex<Ix>,
    nodes: Vec<NodeIndex<Ix>>,
    edges: Vec<EdgeIndex<Ix>>,
}

impl<Ix: IndexType> ReachabilityPath<Ix> {
    /// The start node of the path.
    pub fn start(&self) -> NodeIndex<Ix> {
        self.start
    }

    /// The end node of the path.
    pub fn end(&self) -> NodeIndex<Ix> {
        self.end
    }

    /// The nodes on the path, in order from start to end (both endpoints
    /// included).
    pub fn nodes(&self) -> &[NodeIndex<Ix>] {
        &self.nodes
    }

    /// The edges on the path, in order from start to end.
    pub fn edges(&self) -> &[EdgeIndex<Ix>] {
        &self.edges
    }

    /// The number of edges in the path.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if the path has no edges (start == end).
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}
