//! Bi-criteria (cost, resource) Pareto-optimal shortest paths.

use alloc::{collections::BinaryHeap, vec, vec::Vec};
use core::{cmp::Ordering, hash::Hash};

use hashbrown::HashMap;

use super::BoundedMeasure;
use crate::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers};

/// A single Pareto-optimal path from the source to the target.
///
/// The path is stored as the sequence of edge identities, so parallel edges
/// between the same pair of nodes are still distinguished.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParetoPath<EdgeId, K> {
    /// Total cost of the path.
    pub cost: K,
    /// Total resource consumption of the path.
    pub resource: K,
    /// Edge identities of the path, in traversal order.
    pub edges: Vec<EdgeId>,
}

/// The result of [`pareto_shortest_paths`]: all non-dominated `(cost, resource)`
/// labels at the target, each with one representative path.
#[derive(Clone, Debug, PartialEq)]
pub struct ParetoFrontier<NodeId, EdgeId, K> {
    /// The source node the search started from.
    pub source: NodeId,
    /// The target node the frontier belongs to.
    pub target: NodeId,
    /// Non-dominated labels at the target, sorted by ascending cost
    /// (and hence descending resource). For each label the lexicographically
    /// smallest simple path (by edge identity sequence) is kept.
    pub paths: Vec<ParetoPath<EdgeId, K>>,
    /// Statistics collected during the search.
    pub stats: ParetoStats,
}

/// Statistics about a [`pareto_shortest_paths`] run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParetoStats {
    /// Number of labels accepted into a node frontier.
    pub labels_created: usize,
    /// Number of candidate labels discarded (over budget, dominated, or
    /// an equal label with a lexicographically larger path).
    pub labels_pruned: usize,
    /// Number of frontier labels expanded through the priority queue.
    pub labels_expanded: usize,
    /// The label limit the search was run with.
    pub max_labels: usize,
}

/// Error returned by [`pareto_shortest_paths`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParetoError<NodeId> {
    /// The source node is not present in the graph.
    SourceNotFound(NodeId),
    /// The number of created labels exceeded `max_labels`. The returned
    /// frontier would be incomplete, so an error with statistics is
    /// reported instead of a truncated result.
    LabelLimitExceeded(ParetoStats),
    /// Integer overflow while accumulating cost or resource along a path.
    Overflow,
}

struct Label<NodeId, EdgeId, K> {
    cost: K,
    resource: K,
    node: NodeId,
    edges: Vec<EdgeId>,
    nodes: Vec<NodeId>,
}

impl<NodeId: Clone, EdgeId: Clone, K: Clone> Clone for Label<NodeId, EdgeId, K> {
    fn clone(&self) -> Self {
        Label {
            cost: self.cost.clone(),
            resource: self.resource.clone(),
            node: self.node.clone(),
            edges: self.edges.clone(),
            nodes: self.nodes.clone(),
        }
    }
}

struct QueueLabel<NodeId, EdgeId, K>(Label<NodeId, EdgeId, K>);

impl<NodeId, EdgeId: Ord, K: PartialOrd> PartialEq for QueueLabel<NodeId, EdgeId, K> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<NodeId, EdgeId: Ord, K: PartialOrd> Eq for QueueLabel<NodeId, EdgeId, K> {}

impl<NodeId, EdgeId: Ord, K: PartialOrd> PartialOrd for QueueLabel<NodeId, EdgeId, K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<NodeId, EdgeId: Ord, K: PartialOrd> Ord for QueueLabel<NodeId, EdgeId, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed so that the binary heap pops the smallest label first.
        other
            .0
            .cost
            .partial_cmp(&self.0.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                other
                    .0
                    .resource
                    .partial_cmp(&self.0.resource)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| other.0.edges.cmp(&self.0.edges))
    }
}

fn k_le<K: PartialOrd>(a: &K, b: &K) -> bool {
    matches!(a.partial_cmp(b), Some(Ordering::Less | Ordering::Equal))
}

fn k_eq<K: PartialOrd>(a: &K, b: &K) -> bool {
    matches!(a.partial_cmp(b), Some(Ordering::Equal))
}
/// Compute all Pareto-optimal `(cost, resource)` paths from `source` to
/// `target` under a resource budget.
///
/// Each edge carries a non-negative integer `cost` and `resource` consumption,
/// returned by the `edge_metrics` closure. A label `A` dominates a label `B`
/// if `A` is no worse in both metrics and strictly better in at least one. The
/// search maintains a non-dominated frontier per node and returns the frontier
/// at `target` together with one simple path per label. Among simple paths
/// with identical `(cost, resource)` labels, the lexicographically smallest
/// edge identity sequence is kept, so zero-weight cycles cannot produce
/// infinitely many results.
///
/// Paths are stored as sequences of edge identities, which keeps parallel
/// edges distinguishable. Directed graphs, undirected graphs, and graphs with
/// holes in their node/edge indices (e.g. `StableGraph` after removals) are
/// all supported; the input graph is only borrowed, never modified.
///
/// # Arguments
/// * `graph`: weighted graph.
/// * `source`: the source node. Returns [`ParetoError::SourceNotFound`] if it is not present in the
///   graph.
/// * `target`: the target node. If `source == target`, the frontier is the single label `(0, 0)`
///   with an empty path.
/// * `resource_budget`: maximum allowed total resource; labels exactly on the budget are kept,
///   labels above it are discarded.
/// * `max_labels`: hard limit on the number of labels created. If the limit is exceeded,
///   [`ParetoError::LabelLimitExceeded`] is returned with statistics instead of a silently
///   truncated frontier.
/// * `edge_metrics`: closure that maps an edge to its `(cost, resource)` pair. Both must be
///   non-negative.
///
/// # Returns
/// * `Ok`: [`ParetoFrontier`] with all non-dominated labels at `target`.
/// * `Err`: [`ParetoError`] if the source is missing, the label limit is exceeded, or integer
///   addition overflows.
///
/// # Complexity
/// * Time complexity: **O(L * |E| * log(L * |V|))**.
/// * Auxiliary space: **O(L * |V|^2)** for the stored paths.
///
/// where **|V|** is the number of nodes, **|E|** is the number of edges, and
/// **L** is the maximum frontier size per node. **L** is bounded by
/// `max_labels` and, for integer weights, by `resource_budget + 1`, since a
/// frontier keeps at most one label per resource value.
///
/// # Example
/// ```rust
/// use petgraph::{Graph, algo::pareto_shortest_paths, prelude::*};
///
/// let mut g: Graph<(), (u64, u64)> = Graph::new();
/// let s = g.add_node(());
/// let a = g.add_node(());
/// let b = g.add_node(());
/// let c = g.add_node(());
/// let t = g.add_node(());
/// g.extend_with_edges([
///     (s, a, (1, 4)),
///     (a, t, (1, 4)), // s -> a -> t: (cost 2, resource 8)
///     (s, b, (3, 2)),
///     (b, t, (2, 1)), // s -> b -> t: (cost 5, resource 3)
///     (s, c, (3, 5)),
///     (c, t, (3, 4)), // s -> c -> t: (cost 6, resource 9), dominated
/// ]);
///
/// let frontier = pareto_shortest_paths(&g, s, t, 10, 1024, |e| *e.weight()).unwrap();
/// for path in &frontier.paths {
///     println!(
///         "cost {} resource {} edges {:?}",
///         path.cost, path.resource, path.edges
///     );
/// }
/// assert_eq!(frontier.paths.len(), 2);
/// assert_eq!((frontier.paths[0].cost, frontier.paths[0].resource), (2, 8));
/// assert_eq!((frontier.paths[1].cost, frontier.paths[1].resource), (5, 3));
/// ```
pub fn pareto_shortest_paths<G, F, K>(
    graph: G,
    source: G::NodeId,
    target: G::NodeId,
    resource_budget: K,
    max_labels: usize,
    mut edge_metrics: F,
) -> Result<ParetoFrontier<G::NodeId, G::EdgeId, K>, ParetoError<G::NodeId>>
where
    G: IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    G::EdgeId: Ord,
    F: FnMut(G::EdgeRef) -> (K, K),
    K: BoundedMeasure,
{
    if !graph.node_identifiers().any(|node| node == source) {
        return Err(ParetoError::SourceNotFound(source));
    }

    let mut stats = ParetoStats {
        max_labels,
        ..ParetoStats::default()
    };

    let mut frontiers: HashMap<G::NodeId, Vec<Label<G::NodeId, G::EdgeId, K>>> = HashMap::new();
    let mut heap = BinaryHeap::new();

    let start = Label {
        cost: K::default(),
        resource: K::default(),
        node: source,
        edges: Vec::new(),
        nodes: vec![source],
    };
    stats.labels_created = 1;
    if stats.labels_created > max_labels {
        return Err(ParetoError::LabelLimitExceeded(stats));
    }
    frontiers.entry(source).or_default().push(start.clone());
    heap.push(QueueLabel(start));

    while let Some(QueueLabel(label)) = heap.pop() {
        // Skip labels that were removed from or replaced in the frontier
        // since they were queued.
        let is_current = frontiers
            .get(&label.node)
            .map(|frontier| {
                frontier.iter().any(|entry| {
                    k_eq(&entry.cost, &label.cost)
                        && k_eq(&entry.resource, &label.resource)
                        && entry.edges == label.edges
                })
            })
            .unwrap_or(false);
        if !is_current {
            continue;
        }
        stats.labels_expanded += 1;

        // Simple paths never pass through the target midway, so labels that
        // have arrived at the target are final and need no expansion.
        if label.node == target {
            continue;
        }

        for edge in graph.edges(label.node) {
            let (edge_cost, edge_resource) = edge_metrics(edge);
            let (cost, overflowed) = label.cost.clone().overflowing_add(edge_cost);
            if overflowed {
                return Err(ParetoError::Overflow);
            }
            let (resource, overflowed) = label.resource.clone().overflowing_add(edge_resource);
            if overflowed {
                return Err(ParetoError::Overflow);
            }
            if !k_le(&resource, &resource_budget) {
                stats.labels_pruned += 1;
                continue;
            }
            let next = edge.target();
            // Only simple paths are considered; this also guarantees
            // termination in the presence of zero-weight cycles.
            if label.nodes.contains(&next) {
                continue;
            }
            let mut edges = label.edges.clone();
            edges.push(edge.id());
            let mut nodes = label.nodes.clone();
            nodes.push(next);
            let candidate = Label {
                cost,
                resource,
                node: next,
                edges,
                nodes,
            };

            let frontier = frontiers.entry(next).or_default();
            let mut dominated = false;
            let mut equal_index = None;
            for (index, entry) in frontier.iter().enumerate() {
                if k_le(&entry.cost, &candidate.cost) && k_le(&entry.resource, &candidate.resource)
                {
                    if k_eq(&entry.cost, &candidate.cost)
                        && k_eq(&entry.resource, &candidate.resource)
                    {
                        equal_index = Some(index);
                    } else {
                        dominated = true;
                        break;
                    }
                }
            }
            if dominated {
                stats.labels_pruned += 1;
                continue;
            }
            if let Some(index) = equal_index {
                if candidate.edges < frontier[index].edges {
                    // Same label, lexicographically smaller path: replace and
                    // re-expand so the improvement propagates.
                    frontier.remove(index);
                } else {
                    stats.labels_pruned += 1;
                    continue;
                }
            }
            // Drop existing labels that the candidate dominates.
            frontier.retain(|entry| {
                !(k_le(&candidate.cost, &entry.cost) && k_le(&candidate.resource, &entry.resource))
            });

            stats.labels_created += 1;
            if stats.labels_created > max_labels {
                return Err(ParetoError::LabelLimitExceeded(stats));
            }
            frontier.push(candidate.clone());
            heap.push(QueueLabel(candidate));
        }
    }

    let mut paths: Vec<ParetoPath<G::EdgeId, K>> = frontiers
        .get(&target)
        .map(|frontier| {
            frontier
                .iter()
                .map(|label| ParetoPath {
                    cost: label.cost.clone(),
                    resource: label.resource.clone(),
                    edges: label.edges.clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    paths.sort_by(|a, b| {
        a.cost
            .partial_cmp(&b.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                a.resource
                    .partial_cmp(&b.resource)
                    .unwrap_or(Ordering::Equal)
            })
    });

    Ok(ParetoFrontier {
        source,
        target,
        paths,
        stats,
    })
}
