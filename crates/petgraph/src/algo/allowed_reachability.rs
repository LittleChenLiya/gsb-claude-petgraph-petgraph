//! Reachability snapshots restricted to an allowed set of nodes.
//!
//! [`AllowedReachability`] captures, at construction time, every node reachable
//! from a chosen start node while only traversing edges whose **both**
//! endpoints belong to a caller-provided allowed set. The result is an
//! owned snapshot: it stores only node and edge ids, keeps no reference to
//! the input graph, and is therefore immune to any later mutation of that
//! graph. Queries against the snapshot always reflect the graph as it was
//! when the snapshot was built.
//!
//! Paths that would have to pass through a forbidden (not allowed) node are
//! never reported, and parallel edges are kept distinct: every step of a
//! reported path carries the concrete [`EdgeId`](crate::visit::GraphBase::EdgeId)
//! that was used, not just its endpoints.

use alloc::{collections::VecDeque, vec::Vec};
use core::{fmt, hash::Hash};

use hashbrown::{HashMap, HashSet};

use crate::{
    EdgeType,
    visit::{EdgeRef, GraphProp, IntoEdgeReferences, IntoNodeIdentifiers},
};

/// Error returned when an [`AllowedReachability`] snapshot cannot be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllowedReachabilityError<N> {
    /// A node index that does not exist in the graph (invalid or previously
    /// removed, e.g. a vacant `StableGraph` slot) was found in the allowed
    /// set or used as the start node.
    UnknownNode(N),
    /// The start node exists in the graph but is not part of the allowed
    /// set, so no traversal may begin from it.
    StartNotAllowed(N),
}

impl<N: fmt::Debug> fmt::Display for AllowedReachabilityError<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNode(n) => write!(
                f,
                "node {n:?} does not exist in the graph (invalid or removed index)"
            ),
            Self::StartNotAllowed(n) => {
                write!(f, "start node {n:?} is not in the allowed set")
            }
        }
    }
}

impl<N: fmt::Debug> core::error::Error for AllowedReachabilityError<N> {}

/// An owned reachability snapshot over the allowed subset of a graph.
///
/// Built once from a graph, a start node and an allowed node set; afterwards
/// it can be queried any number of times without touching the graph again.
///
/// # Semantics
///
/// * Only edges whose **both** endpoints are allowed are traversed, so no
///   reported path ever passes through a forbidden node.
/// * The snapshot stores node and edge ids by value and holds no reference
///   to the input graph: later insertions or removals in the graph do not
///   affect query results.
/// * Ties between equally short paths are resolved by node/edge insertion
///   order: the snapshot is a breadth-first traversal in which the outgoing
///   edges of every node are examined in the order the graph's edge
///   references report them (insertion order for `Graph` and `StableGraph`).
/// * Parallel edges are preserved: each path step records the exact edge id
///   that was traversed.
///
/// The snapshot is generic over the node id type `N` and the edge id type
/// `E` of the graph it was built from (e.g. [`NodeIndex`](crate::graph::NodeIndex)
/// and [`EdgeIndex`](crate::graph::EdgeIndex) for `Graph`/`StableGraph`); it
/// carries no lifetime and no reference into the graph.
///
/// # Examples
///
/// ```
/// use petgraph::algo::allowed_reachability::AllowedReachability;
/// use petgraph::graph::Graph;
///
/// let mut g = Graph::<&str, ()>::new();
/// let a = g.add_node("a");
/// let secret = g.add_node("secret");
/// let public = g.add_node("public");
/// let b = g.add_node("b");
///
/// g.add_edge(a, secret, ());
/// g.add_edge(secret, b, ());
/// let e_ap = g.add_edge(a, public, ());
/// let e_pb = g.add_edge(public, b, ());
///
/// // Ban `secret`: only the path through `public` may be reported.
/// let snap = AllowedReachability::new(&g, a, [a, public, b]).unwrap();
/// assert!(!snap.is_reachable(secret));
/// assert_eq!(snap.path_to(b), Some(vec![(e_ap, public), (e_pb, b)]));
/// ```
#[derive(Clone, Debug)]
pub struct AllowedReachability<N, E> {
    start: N,
    /// For every reachable node except `start`: its BFS parent and the exact
    /// edge id through which it was first discovered.
    parent: HashMap<N, (N, E)>,
    /// All reachable nodes (including `start`) in discovery order.
    order: Vec<N>,
}

impl<N, E> AllowedReachability<N, E>
where
    N: Copy + Eq + Hash,
    E: Copy,
{
    /// Build a reachability snapshot from `start`, traversing only edges
    /// whose endpoints are both present in `allowed`.
    ///
    /// The graph is only read during this call; the returned snapshot owns
    /// copies of everything it needs and can outlive or ignore any later
    /// mutation of `graph`.
    ///
    /// # Errors
    ///
    /// * [`AllowedReachabilityError::UnknownNode`] if `start` or any entry of
    ///   `allowed` is not an existing node of `graph` (this includes removed
    ///   nodes / vacant slots of a `StableGraph`).
    /// * [`AllowedReachabilityError::StartNotAllowed`] if `start` exists but
    ///   is not contained in `allowed`.
    ///
    /// # Complexity
    ///
    /// **O(|V| + |E|)** time, **O(|V| + |E|)** auxiliary space.
    pub fn new<G, I>(graph: G, start: N, allowed: I) -> Result<Self, AllowedReachabilityError<N>>
    where
        G: IntoEdgeReferences + IntoNodeIdentifiers + GraphProp<NodeId = N, EdgeId = E>,
        I: IntoIterator<Item = N>,
    {
        let existing: HashSet<N> = graph.node_identifiers().collect();
        let allowed: HashSet<N> = allowed.into_iter().collect();

        for &node in &allowed {
            if !existing.contains(&node) {
                return Err(AllowedReachabilityError::UnknownNode(node));
            }
        }
        if !existing.contains(&start) {
            return Err(AllowedReachabilityError::UnknownNode(start));
        }
        if !allowed.contains(&start) {
            return Err(AllowedReachabilityError::StartNotAllowed(start));
        }

        // Adjacency lists in edge-reference (insertion) order, so that ties
        // between equally short paths are resolved deterministically by the
        // order in which nodes and edges were inserted into the graph.
        // Every edge reference contributes its own entry, keeping parallel
        // edges (and their distinct edge ids) intact.
        let mut adjacency: HashMap<N, Vec<(N, E)>> = HashMap::new();
        for edge in graph.edge_references() {
            adjacency
                .entry(edge.source())
                .or_default()
                .push((edge.target(), edge.id()));
            if !G::EdgeType::is_directed() {
                adjacency
                    .entry(edge.target())
                    .or_default()
                    .push((edge.source(), edge.id()));
            }
        }

        let mut parent: HashMap<N, (N, E)> = HashMap::new();
        let mut order = Vec::new();
        let mut queue = VecDeque::new();

        order.push(start);
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            if let Some(outgoing) = adjacency.get(&u) {
                for &(v, edge_id) in outgoing {
                    // Never step onto a forbidden node: paths crossing one
                    // are not allowed paths.
                    if !allowed.contains(&v) {
                        continue;
                    }
                    if v == start || parent.contains_key(&v) {
                        continue;
                    }
                    parent.insert(v, (u, edge_id));
                    order.push(v);
                    queue.push_back(v);
                }
            }
        }

        Ok(AllowedReachability {
            start,
            parent,
            order,
        })
    }

    /// The start node this snapshot was built from.
    pub fn start(&self) -> N {
        self.start
    }

    /// Return `true` if `node` is reachable from the start node through
    /// allowed nodes only, as of snapshot construction time.
    pub fn is_reachable(&self, node: N) -> bool {
        node == self.start || self.parent.contains_key(&node)
    }

    /// The number of reachable nodes, including the start node.
    pub fn reachable_len(&self) -> usize {
        self.order.len()
    }

    /// All reachable nodes, including the start node, in discovery order.
    pub fn reachable_nodes(&self) -> &[N] {
        &self.order
    }

    /// Reconstruct a path from the start node to `node`.
    ///
    /// Returns `None` if `node` is not reachable through allowed nodes.
    /// Otherwise returns the path as a list of `(edge id, node)` steps in
    /// order from the start node to `node`, where each step records the
    /// exact edge that was traversed to reach that node. The path to the
    /// start node itself is `Some(vec![])`.
    ///
    /// The returned edge ids are the original ids the edges had in the graph
    /// at construction time; they remain meaningful to the caller even if
    /// the graph has been mutated since.
    pub fn path_to(&self, node: N) -> Option<Vec<(E, N)>> {
        if !self.is_reachable(node) {
            return None;
        }
        let mut steps = Vec::new();
        let mut current = node;
        while current != self.start {
            let &(prev, edge_id) = self
                .parent
                .get(&current)
                .expect("reachable non-start node has a parent");
            steps.push((edge_id, current));
            current = prev;
        }
        steps.reverse();
        Some(steps)
    }

    /// Like [`path_to`](Self::path_to), but returns only the traversed edge
    /// ids in order from the start node to `node`.
    pub fn edge_path_to(&self, node: N) -> Option<Vec<E>> {
        self.path_to(node)
            .map(|steps| steps.into_iter().map(|(edge_id, _)| edge_id).collect())
    }
}
