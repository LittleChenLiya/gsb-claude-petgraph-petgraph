//! Quotient graph construction from an explicit node partition.
//!
//! Unlike [`condensation`](crate::algo::condensation), which derives the partition by running a
//! strongly-connected-components algorithm, [`quotient_graph`] takes the partition as input: the
//! caller already knows which group every node belongs to and only needs the quotient graph plus
//! the node/edge correspondences.

use alloc::{collections::BTreeMap, vec, vec::Vec};
use core::fmt;

use crate::{
    EdgeType,
    graph::{Graph, IndexType, NodeIndex},
    visit::{EdgeRef, GraphProp, IntoEdgeReferences, IntoNodeIdentifiers, NodeIndexable},
};

/// Whether edges whose endpoints fall into the same group are kept as self-loops in the quotient
/// graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SelfLoopPolicy {
    /// Keep intra-group edges; they form edge groups `(g, g)` and are aggregated like any other
    /// edge group.
    Keep,
    /// Discard intra-group edges before aggregation; the aggregator never sees them.
    ///
    /// This is the default.
    #[default]
    Drop,
}

/// Error returned by [`quotient_graph`]. Every variant carries the position (node or edge-group
/// index) at which the problem was detected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuotientError<N, E> {
    /// A node that exists in the graph has no group assignment. `node` is the offending node.
    MissingPartition {
        /// The existing node that was not assigned to any group.
        node: N,
    },
    /// The partition assigns a group to a node index that does not exist in the graph, e.g. a
    /// vacant slot ("hole") of a [`StableGraph`](crate::stable_graph::StableGraph) or an
    /// out-of-bounds index. `node` is the offending identifier.
    InvalidNode {
        /// The node identifier that does not refer to an existing node.
        node: N,
    },
    /// A node was assigned to a group more than once. `node` is the offending node.
    DuplicateAssignment {
        /// The node that received more than one group assignment.
        node: N,
    },
    /// The user-supplied edge aggregator failed. `position` is the 0-based position of the edge
    /// group in the deterministic processing order (sorted by quotient endpoints); no further
    /// edge groups were submitted to the aggregator after the failure.
    Aggregation {
        /// The 0-based position of the failing edge group in processing order.
        position: usize,
        /// The error returned by the aggregator.
        error: E,
    },
}

impl<N: fmt::Debug, E: fmt::Debug> fmt::Display for QuotientError<N, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuotientError::MissingPartition { node } => {
                write!(
                    f,
                    "node {node:?} exists in the graph but has no group assignment"
                )
            }
            QuotientError::InvalidNode { node } => {
                write!(
                    f,
                    "partition assigns a group to {node:?}, which is not a node of the graph"
                )
            }
            QuotientError::DuplicateAssignment { node } => {
                write!(f, "node {node:?} was assigned to a group more than once")
            }
            QuotientError::Aggregation { position, error } => {
                write!(
                    f,
                    "edge aggregator failed for edge group {position}: {error:?}"
                )
            }
        }
    }
}

impl<N: fmt::Debug, E: fmt::Debug> core::error::Error for QuotientError<N, E> {}

/// The result of [`quotient_graph`]: the quotient graph together with the complete
/// correspondence between the input graph and the quotient.
#[derive(Clone, Debug)]
pub struct Quotient<G, L, W, Ix>
where
    G: GraphProp,
    Ix: IndexType,
{
    /// The quotient graph. Nodes are created in ascending label order and carry the group label
    /// as weight; edge weights are produced by the aggregator.
    pub graph: Graph<L, W, G::EdgeType, Ix>,
    /// Forward map, indexed by the raw node index of the input graph (length
    /// `graph.node_bound()`). `Some(quotient_node)` for every existing node, `None` for vacant
    /// slots, so holes of a [`StableGraph`](crate::stable_graph::StableGraph) are distinguishable
    /// from live nodes.
    pub node_map: Vec<Option<NodeIndex<Ix>>>,
    /// Reverse map: `groups[i]` lists the original nodes of quotient node `i`, in the input
    /// graph's node-identifier order.
    pub groups: Vec<Vec<G::NodeId>>,
    /// `edge_sources[i]` lists every original edge that was aggregated into quotient edge `i`
    /// (parallel to `graph.edge_indices()`), sorted by edge id. Parallel input edges are all
    /// preserved here.
    pub edge_sources: Vec<Vec<G::EdgeId>>,
}

/// Build the quotient graph of `graph` for an explicit node partition, without modifying the
/// input graph.
///
/// This is *not* a renaming of [`condensation`](crate::algo::condensation): no
/// strongly-connected-components computation is involved. The partition is supplied by the
/// caller, the input graph is only borrowed, and edge weights of the quotient are produced by a
/// user-supplied, fallible aggregator.
///
/// # Applicable graph types
///
/// Any graph type implementing [`IntoNodeIdentifiers`] + [`IntoEdgeReferences`] +
/// [`NodeIndexable`] + [`GraphProp`] with a `Copy` node id and an `Ord` edge id works; this
/// includes [`Graph`] and [`StableGraph`](crate::stable_graph::StableGraph), both directed and
/// undirected. For `StableGraph`, vacant slots left by removed nodes are *not* required to appear
/// in the partition, and [`Quotient::node_map`] maps them to `None`.
///
/// # Partition requirements
///
/// `partition` yields `(node, label)` pairs. Every node that exists in the graph must be
/// assigned to exactly one group; labels must be `Ord`. Missing nodes, assignments to
/// non-existent nodes (including `StableGraph` holes), and duplicate assignments are reported as
/// [`QuotientError::MissingPartition`], [`QuotientError::InvalidNode`] and
/// [`QuotientError::DuplicateAssignment`] respectively, each carrying the offending node.
///
/// # Quotient construction
///
/// * Quotient nodes are created in ascending label order, with the label as node weight.
/// * Input edges are grouped by the quotient endpoints of their endpoints. Directed graphs
///   distinguish `(u, v)` from `(v, u)`; undirected graphs merge the two directions. Parallel input
///   edges are never deduplicated before aggregation — every edge of a group is passed to the
///   aggregator and recorded in [`Quotient::edge_sources`].
/// * Intra-group edges are kept or dropped according to `self_loops`.
/// * Edge groups are submitted to `aggregate` in a stable order: sorted by quotient endpoints, with
///   each group's edge ids sorted by index. If the aggregator fails, the error is returned as
///   [`QuotientError::Aggregation`] with the group's position, and no further edge groups are
///   submitted.
///
/// All ordering is derived from sorted maps and index order, never from hash-map iteration, so
/// repeated runs on the same input produce identical output.
///
/// # Index width
///
/// The quotient graph uses index type `Ix`; the number of groups and of quotient edges must fit
/// `Ix`, exactly as for a manually built [`Graph<N, E, Ty, Ix>`](Graph). The default
/// [`DefaultIx`](crate::graph::DefaultIx) (`u32`) suffices unless the quotient has more than
/// `u32::MAX` nodes or edges.
///
/// # Complexity
///
/// `O(V + E log E)` time and `O(V + E)` additional space, where `V` and `E` are the input
/// graph's node and edge counts; the logarithmic factor comes from grouping edges and labels in
/// sorted maps.
///
/// # Example
///
/// ```
/// use petgraph::{
///     algo::quotient::{SelfLoopPolicy, quotient_graph},
///     graph::DefaultIx,
///     prelude::*,
/// };
///
/// let mut g = DiGraph::<&str, i64>::new();
/// let a = g.add_node("a");
/// let b = g.add_node("b");
/// let c = g.add_node("c");
/// g.add_edge(a, c, 1);
/// g.add_edge(a, c, 2); // parallel edge, preserved
/// g.add_edge(b, c, 4);
///
/// let partition = [(a, "x"), (b, "x"), (c, "y")];
/// let q = quotient_graph::<_, _, i64, _, _, DefaultIx>(
///     &g,
///     partition,
///     SelfLoopPolicy::Drop,
///     |edges| Ok::<i64, ()>(edges.iter().map(|&e| g[e]).sum::<i64>()),
/// )
/// .unwrap();
///
/// assert_eq!(q.graph.node_count(), 2);
/// assert_eq!(q.graph.edge_count(), 1);
/// assert_eq!(q.edge_sources[0].len(), 3); // parallel edges aggregated together
/// ```
pub fn quotient_graph<G, L, W, F, Err, Ix>(
    graph: G,
    partition: impl IntoIterator<Item = (G::NodeId, L)>,
    self_loops: SelfLoopPolicy,
    mut aggregate: F,
) -> Result<Quotient<G, L, W, Ix>, QuotientError<G::NodeId, Err>>
where
    G: IntoNodeIdentifiers + IntoEdgeReferences + NodeIndexable + GraphProp,
    G::NodeId: Copy,
    G::EdgeId: Ord,
    L: Ord + Clone,
    Ix: IndexType,
    F: FnMut(&[G::EdgeId]) -> Result<W, Err>,
{
    let bound = graph.node_bound();

    // Mark which raw slots hold live nodes (distinguishes StableGraph holes).
    let mut exists = vec![false; bound];
    for node in graph.node_identifiers() {
        exists[graph.to_index(node)] = true;
    }

    // Apply the partition, validating indices and duplicates.
    let mut slot_label: Vec<Option<L>> = vec![None; bound];
    for (node, label) in partition {
        let idx = graph.to_index(node);
        if idx >= bound || !exists[idx] {
            return Err(QuotientError::InvalidNode { node });
        }
        if slot_label[idx].is_some() {
            return Err(QuotientError::DuplicateAssignment { node });
        }
        slot_label[idx] = Some(label);
    }
    for node in graph.node_identifiers() {
        if slot_label[graph.to_index(node)].is_none() {
            return Err(QuotientError::MissingPartition { node });
        }
    }

    // Assign group ordinals in ascending label order (BTreeMap: deterministic).
    let mut label_to_group: BTreeMap<L, usize> = BTreeMap::new();
    for label in slot_label.iter().flatten() {
        label_to_group.entry(label.clone()).or_insert(0);
    }
    for (ordinal, slot) in label_to_group.values_mut().enumerate() {
        *slot = ordinal;
    }

    // Create quotient nodes in label order.
    let mut quotient: Graph<L, W, G::EdgeType, Ix> = Graph::default();
    let mut group_nodes = Vec::with_capacity(label_to_group.len());
    for label in label_to_group.keys() {
        group_nodes.push(quotient.add_node(label.clone()));
    }

    // Node correspondences.
    let mut node_map = vec![None; bound];
    let mut groups: Vec<Vec<G::NodeId>> = (0..group_nodes.len()).map(|_| Vec::new()).collect();
    for node in graph.node_identifiers() {
        let idx = graph.to_index(node);
        let group = label_to_group[slot_label[idx].as_ref().unwrap()];
        node_map[idx] = Some(group_nodes[group]);
        groups[group].push(node);
    }

    // Group edges by quotient endpoints. Directed graphs keep (u, v) and (v, u) apart;
    // undirected graphs normalize to (min, max). Parallel edges all land in the same group.
    let directed = G::EdgeType::is_directed();
    let mut edge_groups: BTreeMap<(usize, usize), Vec<G::EdgeId>> = BTreeMap::new();
    for edge in graph.edge_references() {
        let source = label_to_group[slot_label[graph.to_index(edge.source())].as_ref().unwrap()];
        let target = label_to_group[slot_label[graph.to_index(edge.target())].as_ref().unwrap()];
        if source == target && self_loops == SelfLoopPolicy::Drop {
            continue;
        }
        let key = if directed || source <= target {
            (source, target)
        } else {
            (target, source)
        };
        edge_groups.entry(key).or_default().push(edge.id());
    }

    // Aggregate edge groups in sorted-endpoint order; stop at the first failure.
    let mut edge_sources = Vec::with_capacity(edge_groups.len());
    for (position, ((source, target), mut ids)) in edge_groups.into_iter().enumerate() {
        ids.sort_unstable();
        let weight =
            aggregate(&ids).map_err(|error| QuotientError::Aggregation { position, error })?;
        quotient.add_edge(group_nodes[source], group_nodes[target], weight);
        edge_sources.push(ids);
    }

    Ok(Quotient {
        graph: quotient,
        node_map,
        groups,
        edge_sources,
    })
}

#[cfg(all(test, feature = "stable_graph"))]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::{QuotientError, SelfLoopPolicy, quotient_graph};
    use crate::{
        Directed, Undirected,
        graph::{DefaultIx, EdgeIndex, NodeIndex},
        stable_graph::StableGraph,
        visit::{EdgeRef, IntoEdgeReferences},
    };

    type DiStable = StableGraph<&'static str, i64, Directed>;
    type UnStable = StableGraph<&'static str, i64, Undirected>;

    /// Nodes: a(0), b(1), c(2), d(3); b is removed, leaving a hole at index 1.
    /// Edges (directed): a->c(1), a->c(2) [parallel], c->a(4) [reverse], a->a(8) [self-loop],
    /// c->d(16), c->d(32) [parallel, intra-group].
    fn sample<Ty: crate::EdgeType>() -> StableGraph<&'static str, i64, Ty> {
        let mut g = StableGraph::default();
        let a = g.add_node("a");
        let b = g.add_node("b");
        let c = g.add_node("c");
        let d = g.add_node("d");
        g.add_edge(a, c, 1);
        g.add_edge(a, c, 2);
        g.add_edge(c, a, 4);
        g.add_edge(a, a, 8);
        g.add_edge(c, d, 16);
        g.add_edge(c, d, 32);
        g.remove_node(b);
        g
    }

    fn partition() -> [(NodeIndex, &'static str); 3] {
        // Deliberately not in label order: "z" sorts after "m", so group 0 is {c, d}.
        [
            (NodeIndex::new(0), "z"),
            (NodeIndex::new(2), "m"),
            (NodeIndex::new(3), "m"),
        ]
    }

    fn sum_aggregator(g: &DiStable) -> impl FnMut(&[EdgeIndex]) -> Result<i64, ()> + '_ {
        move |edges| Ok(edges.iter().map(|&e| g[e]).sum::<i64>())
    }

    #[test]
    fn directed_drop_self_loops() {
        let g: DiStable = sample();
        let q = quotient_graph::<_, _, i64, _, (), DefaultIx>(
            &g,
            partition(),
            SelfLoopPolicy::Drop,
            sum_aggregator(&g),
        )
        .unwrap();

        // Quotient nodes created in label order: "m" then "z".
        assert_eq!(q.graph.node_count(), 2);
        assert_eq!(q.graph[NodeIndex::new(0)], "m");
        assert_eq!(q.graph[NodeIndex::new(1)], "z");

        // Edge groups: (m, z) from c->a, then (z, m) from the two parallel a->c edges.
        assert_eq!(q.graph.edge_count(), 2);
        let mut edges = q.graph.edge_references();
        let e0 = edges.next().unwrap();
        let e1 = edges.next().unwrap();
        assert_eq!(
            (e0.source(), e0.target(), *e0.weight()),
            (NodeIndex::new(0), NodeIndex::new(1), 4)
        );
        assert_eq!(
            (e1.source(), e1.target(), *e1.weight()),
            (NodeIndex::new(1), NodeIndex::new(0), 3)
        );
        assert!(edges.next().is_none());

        // Every original edge id recorded per quotient edge, sorted by index.
        assert_eq!(
            q.edge_sources,
            vec![
                vec![EdgeIndex::new(2)],
                vec![EdgeIndex::new(0), EdgeIndex::new(1)]
            ]
        );

        // Forward map: the hole at index 1 maps to None, live nodes to their group.
        assert_eq!(
            q.node_map,
            vec![
                Some(NodeIndex::new(1)),
                None,
                Some(NodeIndex::new(0)),
                Some(NodeIndex::new(0))
            ]
        );
        // Reverse map.
        assert_eq!(
            q.groups,
            vec![
                vec![NodeIndex::new(2), NodeIndex::new(3)],
                vec![NodeIndex::new(0)]
            ]
        );
    }

    #[test]
    fn directed_keep_self_loops() {
        let g: DiStable = sample();
        let q = quotient_graph::<_, _, i64, _, (), DefaultIx>(
            &g,
            partition(),
            SelfLoopPolicy::Keep,
            sum_aggregator(&g),
        )
        .unwrap();

        // Groups in order: (m,m) = {c->d x2}, (m,z) = {c->a}, (z,m) = {a->c x2}, (z,z) = {a->a}.
        let weights: Vec<i64> = q.graph.edge_weights().copied().collect();
        assert_eq!(weights, vec![48, 4, 3, 8]);
        assert_eq!(
            q.edge_sources,
            vec![
                vec![EdgeIndex::new(4), EdgeIndex::new(5)],
                vec![EdgeIndex::new(2)],
                vec![EdgeIndex::new(0), EdgeIndex::new(1)],
                vec![EdgeIndex::new(3)],
            ]
        );
        // Parallel intra-group edges did not vanish before aggregation.
        assert_eq!(q.edge_sources[0].len(), 2);
    }

    #[test]
    fn undirected_merges_directions() {
        let g: UnStable = sample();
        let q = quotient_graph::<_, _, i64, _, (), DefaultIx>(
            &g,
            partition(),
            SelfLoopPolicy::Drop,
            |edges| Ok(edges.iter().map(|&e| g[e]).sum::<i64>()),
        )
        .unwrap();

        // (m, z) merges a->c, a->c and c->a: 1 + 2 + 4 = 7; directed kept them apart.
        assert_eq!(q.graph.edge_count(), 1);
        let e = q.graph.edge_references().next().unwrap();
        assert_eq!(*e.weight(), 7);
        assert_eq!(
            q.edge_sources,
            vec![vec![
                EdgeIndex::new(0),
                EdgeIndex::new(1),
                EdgeIndex::new(2)
            ]]
        );
    }

    #[test]
    fn aggregation_failure_stops_and_preserves_input() {
        let g: DiStable = sample();
        let before: Vec<(usize, usize, i64)> = g
            .edge_references()
            .map(|e| (e.source().index(), e.target().index(), *e.weight()))
            .collect();

        let mut calls: Vec<Vec<EdgeIndex>> = Vec::new();
        let result = quotient_graph::<_, _, i64, _, &'static str, DefaultIx>(
            &g,
            partition(),
            SelfLoopPolicy::Keep,
            |edges| {
                calls.push(edges.to_vec());
                if calls.len() == 2 {
                    Err("boom")
                } else {
                    Ok(edges.len() as i64)
                }
            },
        );

        // Failed at the second edge group (position 1); groups 2 and 3 never submitted.
        assert_eq!(
            result.unwrap_err(),
            QuotientError::Aggregation {
                position: 1,
                error: "boom"
            }
        );
        assert_eq!(
            calls,
            vec![
                vec![EdgeIndex::new(4), EdgeIndex::new(5)],
                vec![EdgeIndex::new(2)]
            ]
        );

        // Input graph untouched.
        let after: Vec<(usize, usize, i64)> = g
            .edge_references()
            .map(|e| (e.source().index(), e.target().index(), *e.weight()))
            .collect();
        assert_eq!(before, after);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 6);
    }

    #[test]
    fn partition_errors_carry_position() {
        let g: DiStable = sample();
        let agg = |_: &[EdgeIndex]| Ok::<i64, ()>(0);

        // Missing: node d has no group.
        let missing = [(NodeIndex::new(0), "z"), (NodeIndex::new(2), "m")];
        assert_eq!(
            quotient_graph::<_, _, i64, _, (), DefaultIx>(&g, missing, SelfLoopPolicy::Drop, agg)
                .unwrap_err(),
            QuotientError::MissingPartition {
                node: NodeIndex::new(3)
            }
        );

        // Invalid: index 1 is a hole, index 99 is out of bounds.
        for bad in [NodeIndex::new(1), NodeIndex::new(99)] {
            let mut p = partition().to_vec();
            p.push((bad, "m"));
            assert_eq!(
                quotient_graph::<_, _, i64, _, (), DefaultIx>(&g, p, SelfLoopPolicy::Drop, agg)
                    .unwrap_err(),
                QuotientError::InvalidNode { node: bad }
            );
        }

        // Duplicate: node a assigned twice.
        let mut p = partition().to_vec();
        p.push((NodeIndex::new(0), "m"));
        assert_eq!(
            quotient_graph::<_, _, i64, _, (), DefaultIx>(&g, p, SelfLoopPolicy::Drop, agg)
                .unwrap_err(),
            QuotientError::DuplicateAssignment {
                node: NodeIndex::new(0)
            }
        );
    }

    #[test]
    fn repeated_runs_are_identical() {
        let g: DiStable = sample();
        let run = || {
            let q = quotient_graph::<_, _, i64, _, (), DefaultIx>(
                &g,
                partition(),
                SelfLoopPolicy::Keep,
                sum_aggregator(&g),
            )
            .unwrap();
            (
                q.graph.node_weights().cloned().collect::<Vec<_>>(),
                q.graph.edge_weights().copied().collect::<Vec<_>>(),
                q.node_map,
                q.groups,
                q.edge_sources,
            )
        };
        let first = run();
        for _ in 0..10 {
            assert_eq!(first, run());
        }
    }
}
