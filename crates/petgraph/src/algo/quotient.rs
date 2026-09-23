//! [Graph] Explicit-partition quotient graph construction.
//!
//! Unlike [`condensation`](super::condensation), which derives node groups by
//! running a strongly-connected-components algorithm, the functions here let
//! the caller supply an explicit partition of the nodes and build the
//! corresponding quotient graph directly.

use alloc::{collections::BTreeMap, vec, vec::Vec};

use crate::graph::NodeIndex;
use crate::graph::{DefaultIx, IndexType};
use crate::visit::{
    EdgeIndexable, EdgeRef, GraphProp, IntoEdgeReferences, IntoNodeIdentifiers, NodeIndexable,
};
use crate::{EdgeType, Graph};

/// Whether edges whose endpoints fall into the same group are kept as
/// self-loops in the quotient graph or discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SelfLoopPolicy {
    /// Keep intra-group edges as self-loops of the quotient node.
    Keep,
    /// Discard intra-group edges; they are neither aggregated nor recorded.
    Remove,
}

/// Error returned by [`quotient_graph`].
///
/// Every variant carries the position (node id or quotient edge endpoints)
/// at which the problem was detected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuotientError<N, E, Ix: IndexType = DefaultIx> {
    /// A node that exists in the graph has no group assignment in the
    /// partition. `node` is the first such node (in node id order).
    MissingPartition {
        /// The live node without a group assignment.
        node: N,
    },
    /// The partition assigns a group to a node id that does not exist in the
    /// graph (for a [`StableGraph`](crate::stable_graph::StableGraph) this
    /// includes indices left behind by removed nodes). `node` is the first
    /// such id (in node id order).
    InvalidNode {
        /// The partition key that is not a live node of the graph.
        node: N,
    },
    /// The user-supplied aggregator failed while building the quotient edge
    /// `source -> target`. Aggregation stops at this edge group; no further
    /// groups are passed to the aggregator.
    Aggregation {
        /// Source quotient node of the edge group that failed.
        source: NodeIndex<Ix>,
        /// Target quotient node of the edge group that failed.
        target: NodeIndex<Ix>,
        /// The error returned by the aggregator.
        error: E,
    },
}

/// The result of [`quotient_graph`]: the quotient graph together with the
/// bidirectional mapping between input nodes and quotient nodes and the
/// provenance of every quotient edge.
#[derive(Clone, Debug)]
pub struct Quotient<N, E, L, W, Ty: EdgeType, Ix: IndexType = DefaultIx> {
    /// The quotient graph. Nodes are created in ascending group-label order
    /// and carry the label as weight; each edge carries the weight produced
    /// by the aggregator for its edge group.
    pub graph: Graph<L, W, Ty, Ix>,
    /// Maps every position of the input graph's index space
    /// (`0 .. node_bound`) to the quotient node: `Some(quotient_node)` for
    /// live nodes, `None` for holes (e.g. indices of removed nodes in a
    /// [`StableGraph`](crate::stable_graph::StableGraph)).
    pub node_map: Vec<Option<NodeIndex<Ix>>>,
    /// For each quotient node (in label order), the input nodes it groups,
    /// sorted by node id.
    pub groups: Vec<Vec<N>>,
    /// For each quotient edge (in creation order), the input edge ids that
    /// were aggregated into it, in ascending edge index order.
    pub edge_sources: Vec<Vec<E>>,
}

/// [Graph] Build the quotient graph of `g` for an explicit node partition.
///
/// This does **not** run strongly-connected-components analysis; the caller
/// provides the partition directly. The input graph is borrowed and left
/// unmodified.
///
/// # Arguments
/// * `g`: the input graph. Works with any graph type implementing the
///   required visitor traits, e.g. [`Graph`],
///   [`StableGraph`](crate::stable_graph::StableGraph) and
///   [`GraphMap`](crate::graphmap::GraphMap) (pass them by reference).
/// * `partition`: maps every **live** node of `g` to its group label. Labels
///   must be orderable; quotient nodes are created in ascending label order.
/// * `self_loops`: whether edges inside one group are kept as self-loops or
///   discarded.
/// * `aggregate`: fallible aggregator called once per quotient edge group
///   with the input edge references of that group, in ascending edge index
///   order. Its `Ok` value becomes the quotient edge weight.
///
/// # Returns
/// A [`Quotient`] holding the quotient graph and the mappings:
/// original node -> quotient node ([`Quotient::node_map`], with `None` for
/// holes), quotient node -> original node set ([`Quotient::groups`]), and
/// quotient edge -> original edge ids ([`Quotient::edge_sources`]).
///
/// Edges are grouped by quotient endpoints. In a directed graph `(u, v)` and
/// `(v, u)` form separate groups; in an undirected graph they are merged.
/// Parallel input edges never disappear before aggregation: every input edge
/// is passed to the aggregator of its group. The aggregator is invoked in a
/// stable order (ascending quotient endpoint pairs); on the first failure
/// the call returns and no later edge group is aggregated.
///
/// # Errors
/// * [`QuotientError::MissingPartition`] - a live node has no group.
/// * [`QuotientError::InvalidNode`] - the partition mentions a node id that
///   is not a live node of `g` (including holes of a `StableGraph`).
/// * [`QuotientError::Aggregation`] - the aggregator returned an error.
///
/// # Panics
/// Panics if the number of groups or quotient edges exceeds the index
/// capacity of `Ix` (same rule as [`Graph::add_node`]). Choose an `Ix` wide
/// enough for the quotient graph; since the number of groups never exceeds
/// the input node count, reusing the input's index type is always safe.
///
/// # Complexity
/// * Time complexity: **O((|V| + |E|) log (|V| + |E|))** (ordered maps and
///   one edge sort; no SCC computation).
/// * Auxiliary space: **O(|V| + |E|)**.
///
/// where **|V|** is the number of nodes and **|E|** is the number of edges.
///
/// # Example
/// ```
/// use petgraph::algo::quotient::{quotient_graph, SelfLoopPolicy};
/// use petgraph::graph::DefaultIx;
/// use petgraph::prelude::*;
/// use petgraph::Graph;
/// use std::collections::BTreeMap;
///
/// let mut g: Graph<(), i32> = Graph::new();
/// let a = g.add_node(());
/// let b = g.add_node(());
/// let c = g.add_node(());
/// g.add_edge(a, b, 1);
/// g.add_edge(a, b, 2); // parallel edge, kept for the aggregator
/// g.add_edge(b, c, 4);
///
/// let partition: BTreeMap<_, char> = [(a, 'x'), (b, 'x'), (c, 'y')].into_iter().collect();
/// let q = quotient_graph::<_, _, i32, _, (), DefaultIx>(
///     &g,
///     &partition,
///     SelfLoopPolicy::Keep,
///     |edges| Ok(edges.iter().map(|e| *e.weight()).sum()),
/// )
/// .unwrap();
///
/// assert_eq!(q.graph.node_count(), 2);
/// assert_eq!(q.graph.edge_count(), 2);
/// assert_eq!(q.edge_sources[1].len(), 1);
/// assert_eq!(q.node_map[a.index()], q.node_map[b.index()]);
/// ```
pub fn quotient_graph<G, L, W, F, E, Ix>(
    g: G,
    partition: &BTreeMap<G::NodeId, L>,
    self_loops: SelfLoopPolicy,
    mut aggregate: F,
) -> Result<Quotient<G::NodeId, G::EdgeId, L, W, G::EdgeType, Ix>, QuotientError<G::NodeId, E, Ix>>
where
    G: IntoNodeIdentifiers + IntoEdgeReferences + NodeIndexable + EdgeIndexable + GraphProp,
    G::NodeId: Ord,
    G::EdgeType: EdgeType,
    L: Ord + Clone,
    F: FnMut(&[G::EdgeRef]) -> Result<W, E>,
    Ix: IndexType,
{
    // Resolve the group label of every live node; iteration order of
    // `node_identifiers` does not matter because the result is stored in an
    // ordered map keyed by node id.
    let mut assignments: BTreeMap<G::NodeId, L> = BTreeMap::new();
    for node in g.node_identifiers() {
        match partition.get(&node) {
            Some(label) => {
                assignments.insert(node, label.clone());
            }
            None => return Err(QuotientError::MissingPartition { node }),
        }
    }
    // Reject partition entries that do not name a live node (this is how
    // `StableGraph` holes are caught instead of being silently grouped).
    for &node in partition.keys() {
        if !assignments.contains_key(&node) {
            return Err(QuotientError::InvalidNode { node });
        }
    }

    // Group live nodes by label; quotient nodes are created in ascending
    // label order and members stay sorted by node id.
    let mut by_label: BTreeMap<L, Vec<G::NodeId>> = BTreeMap::new();
    for (node, label) in assignments {
        by_label.entry(label).or_default().push(node);
    }
    let mut graph: Graph<L, W, G::EdgeType, Ix> =
        Graph::with_capacity(by_label.len(), g.edge_bound());
    let mut groups: Vec<Vec<G::NodeId>> = Vec::with_capacity(by_label.len());
    let mut node_map: Vec<Option<NodeIndex<Ix>>> = vec![None; g.node_bound()];
    for (label, members) in by_label {
        let q = graph.add_node(label);
        for &node in &members {
            node_map[NodeIndexable::to_index(&g, node)] = Some(q);
        }
        groups.push(members);
    }

    // Group edges by quotient endpoints. Sorting by edge index first makes
    // both the per-group order and the aggregator input order stable,
    // independent of the graph's edge iteration order.
    let mut edge_refs: Vec<G::EdgeRef> = g.edge_references().collect();
    edge_refs.sort_by_key(|e| EdgeIndexable::to_index(&g, e.id()));
    let directed = g.is_directed();
    let mut edge_groups: BTreeMap<(usize, usize), Vec<G::EdgeRef>> = BTreeMap::new();
    for edge in edge_refs {
        let source = node_map[NodeIndexable::to_index(&g, edge.source())]
            .expect("edge endpoints are live nodes")
            .index();
        let target = node_map[NodeIndexable::to_index(&g, edge.target())]
            .expect("edge endpoints are live nodes")
            .index();
        let (source, target) = if !directed && source > target {
            (target, source)
        } else {
            (source, target)
        };
        if source == target && self_loops == SelfLoopPolicy::Remove {
            continue;
        }
        edge_groups.entry((source, target)).or_default().push(edge);
    }

    // Aggregate one quotient edge per group, in ascending endpoint order.
    // Stop at the first aggregator failure.
    let mut edge_sources: Vec<Vec<G::EdgeId>> = Vec::with_capacity(edge_groups.len());
    for (&(source, target), group) in &edge_groups {
        let weight = aggregate(group).map_err(|error| QuotientError::Aggregation {
            source: NodeIndex::new(source),
            target: NodeIndex::new(target),
            error,
        })?;
        graph.add_edge(NodeIndex::new(source), NodeIndex::new(target), weight);
        edge_sources.push(group.iter().map(|edge| edge.id()).collect());
    }

    Ok(Quotient {
        graph,
        node_map,
        groups,
        edge_sources,
    })
}
