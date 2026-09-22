use alloc::{collections::BinaryHeap, vec, vec::Vec};
use core::cmp::Ordering;

use crate::{
    algo::ParetoWeight,
    scored::MinScored,
    visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeIndexable},
};

/// A non-dominated (cost, resource) label together with one simple path.
///
/// The path records the edge identities of the edges that make it up, in
/// traversal order. Edge identities (rather than just a node sequence) are
/// returned so that parallel edges can be told apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParetoLabel<N, E, W> {
    /// The total edge cost of `edges`.
    pub cost: W,
    /// The total resource consumption of `edges`.
    pub resource: W,
    /// Nodes visited by the path, starting with the source and ending at the
    /// node the label belongs to. Contains one more element than `edges`.
    pub nodes: Vec<N>,
    /// Identifiers of the edges of the path, in traversal order. Empty for a
    /// path from the source to itself.
    pub edges: Vec<E>,
}

/// Statistics collected while computing non-dominated labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParetoSearchStats {
    /// Number of labels (including later dominated ones) pushed onto the
    /// internal priority queue.
    pub labels_generated: usize,
    /// Number of labels popped from the priority queue while still
    /// non-dominated, and therefore expanded.
    pub labels_settled: usize,
    /// Largest number of simultaneously non-dominated labels ever held at a
    /// single node.
    pub max_frontier_size: usize,
}

/// The complete set of non-dominated source-to-target labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParetoPaths<N, E, W> {
    /// Non-dominated labels sorted by ascending cost (ties by ascending
    /// resource), each paired with one simple edge path.
    pub labels: Vec<ParetoLabel<N, E, W>>,
    /// Search statistics.
    pub stats: ParetoSearchStats,
}

/// Why a complete non-dominated frontier could not be produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParetoPathError {
    /// The source or target node does not belong to the graph.
    NodeNotFound,
    /// An edge closure returned a negative cost or resource consumption.
    NegativeWeight,
    /// Adding edge costs overflowed the weight type. Resource sums that
    /// overflow are silently discarded instead, as they exceed every budget.
    WeightOverflow,
    /// A node's non-dominated frontier would have to hold more than
    /// `max_labels` labels. This is a hard bound: no partial result is
    /// returned.
    LabelLimitReached {
        /// Compact index of the node at which the bound was exceeded.
        node_index: usize,
        /// The configured bound.
        max_labels: usize,
        /// Statistics gathered up to the point of failure.
        stats: ParetoSearchStats,
    },
}

#[derive(Clone)]
struct Label<N, E, W> {
    cost: W,
    resource: W,
    last_node: N,
    nodes: Vec<N>,
    edges: Vec<E>,
}

#[derive(Copy, Clone)]
struct HeapKey<W>(W, W);

impl<W: PartialOrd> PartialEq for HeapKey<W> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<W: PartialOrd> Eq for HeapKey<W> {}

impl<W: PartialOrd> PartialOrd for HeapKey<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<W: PartialOrd> Ord for HeapKey<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by resource first, then cost; ties pop in arbitrary order and
        // do not affect the resulting frontier.
        match self.0.partial_cmp(&other.0) {
            Some(Ordering::Equal) | None => {}
            Some(order) => return order,
        }
        match self.1.partial_cmp(&other.1) {
            Some(order) => order,
            None => Ordering::Equal,
        }
    }
}

#[derive(Copy, Clone)]
struct HeapEntry<W> {
    resource: W,
    cost: W,
    node_index: usize,
    label_index: usize,
}

#[derive(Clone)]
struct FrontierSlot<W> {
    cost: W,
    resource: W,
    label_index: usize,
}

/// Find all non-dominated (cost, resource) labels of simple paths from
/// `source` to `target` whose total resource consumption is at most
/// `resource_budget`.
///
/// Every edge carries a non-negative integer cost and resource consumption. A
/// label `(c1, r1)` *dominates* a label `(c2, r2)` when `c1 <= c2` and
/// `r1 <= r2` with at least one strict inequality. For identical labels the
/// path whose edge identifier sequence is lexicographically smallest is kept.
/// Paths that would repeat a node are never extended: with non-negative
/// weights the removed cycle cannot improve the label, and this also makes the
/// search terminate in the presence of zero-weight cycles instead of producing
/// infinitely many cyclic results.
///
/// Every node keeps at most `max_labels` simultaneously non-dominated labels.
/// The bound is a hard resource limit rather than a truncation: if a frontier
/// would grow beyond it, [`ParetoPathError::LabelLimitReached`] is returned
/// together with statistics, never a partial frontier presented as complete.
///
/// # Arguments
/// * `graph`: an input directed graph. It is only borrowed and never modified.
/// * `source`: the start node.
/// * `target`: the goal node; it may equal `source`.
/// * `resource_budget`: inclusive upper bound on total resource consumption.
/// * `max_labels`: maximum number of non-dominated labels kept at any node.
/// * `edge_cost`: closure returning the non-negative integer cost of an edge.
/// * `edge_resource`: closure returning the non-negative integer resource of
///   an edge.
///
/// # Returns
/// * `Ok`: the complete non-dominated frontier at `target`, one simple path
///   per label, plus statistics.
/// * `Err`: why the complete frontier could not be produced.
///
/// # Complexity
/// Let `L` be the largest number of non-dominated labels ever held at a single
/// node (`L <= max_labels`) and `p` the longest retained simple path. Each
/// settled label scans its outgoing edges once, and each frontier insertion
/// compares against at most `L` labels.
/// * Time complexity: **O(L |E| (log(L |E|) + L))** in the worst case.
/// * Auxiliary space: **O(|V| + L |E| · p)** because whole paths are retained
///   to reconstruct edge identities.
///
/// # Example
/// ```rust
/// use petgraph::{algo::pareto_paths, prelude::*};
///
/// let mut g = Graph::<(), (u32, u32), Directed>::new();
/// let s = g.add_node(());
/// let t = g.add_node(());
/// // Three parallel s -> t edges with (cost, resource) (2,8), (5,3), (6,9).
/// g.add_edge(s, t, (2, 8));
/// g.add_edge(s, t, (5, 3));
/// g.add_edge(s, t, (6, 9));
///
/// let res = pareto_paths(&g, s, t, 10, 64,
///     |e| e.weight().0, |e| e.weight().1).unwrap();
/// let frontier: Vec<(u32, u32)> =
///     res.labels.iter().map(|l| (l.cost, l.resource)).collect();
/// assert_eq!(frontier, vec![(2, 8), (5, 3)]);
///
/// let tight = pareto_paths(&g, s, t, 5, 64,
///     |e| e.weight().0, |e| e.weight().1).unwrap();
/// let frontier: Vec<(u32, u32)> =
///     tight.labels.iter().map(|l| (l.cost, l.resource)).collect();
/// assert_eq!(frontier, vec![(5, 3)]);
/// ```
pub fn pareto_paths<G, Fc, Fr, W>(
    graph: G,
    source: G::NodeId,
    target: G::NodeId,
    resource_budget: W,
    max_labels: usize,
    mut edge_cost: Fc,
    mut edge_resource: Fr,
) -> Result<ParetoPaths<G::NodeId, G::EdgeId, W>, ParetoPathError>
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
    G::NodeId: Eq + Copy,
    G::EdgeId: Ord + Clone,
    Fc: FnMut(G::EdgeRef) -> W,
    Fr: FnMut(G::EdgeRef) -> W,
    W: ParetoWeight,
{
    assert!(max_labels >= 1, "max_labels must be at least 1");

    let zero = W::zero_weight();

    let mut source_found = false;
    let mut target_found = false;
    for node in graph.node_identifiers() {
        if node == source {
            source_found = true;
        }
        if node == target {
            target_found = true;
        }
    }
    if !source_found || !target_found {
        return Err(ParetoPathError::NodeNotFound);
    }

    // Even the empty source-to-source path consumes zero resource.
    if resource_budget < zero {
        return Ok(ParetoPaths {
            labels: Vec::new(),
            stats: ParetoSearchStats {
                labels_generated: 0,
                labels_settled: 0,
                max_frontier_size: 0,
            },
        });
    }

    let bound = graph.node_bound();
    let mut paths: Vec<Vec<Label<G::NodeId, G::EdgeId, W>>> = vec![Vec::new(); bound];
    let mut frontiers: Vec<Vec<FrontierSlot<W>>> = vec![Vec::new(); bound];
    let mut heap: BinaryHeap<MinScored<HeapKey<W>, HeapEntry<W>>> = BinaryHeap::new();

    let mut stats = ParetoSearchStats {
        labels_generated: 0,
        labels_settled: 0,
        max_frontier_size: 0,
    };

    let source_index = graph.to_index(source);
    paths[source_index].push(Label {
        cost: zero,
        resource: zero,
        last_node: source,
        nodes: vec![source],
        edges: Vec::new(),
    });
    frontiers[source_index].push(FrontierSlot {
        cost: zero,
        resource: zero,
        label_index: 0,
    });
    stats.max_frontier_size = 1;
    push_heap(
        &mut heap,
        &mut stats,
        HeapEntry {
            resource: zero,
            cost: zero,
            node_index: source_index,
            label_index: 0,
        },
    );

    while let Some(MinScored(_, entry)) = heap.pop() {
        let node_index = entry.node_index;

        // Lazy deletion: skip heap entries whose label has since been dominated
        // or replaced by a lexicographically smaller equal label.
        let still_present = frontiers[node_index].iter().any(|slot| {
            slot.label_index == entry.label_index
                && slot.cost == entry.cost
                && slot.resource == entry.resource
        });
        if !still_present {
            continue;
        }
        stats.labels_settled += 1;

        let (cost, resource, node) = {
            let label = &paths[node_index][entry.label_index];
            (label.cost, label.resource, label.last_node)
        };

        if node == target {
            // Target labels are kept for the answer but never expanded.
            continue;
        }

        for edge in graph.edges(node) {
            let edge_cost_value = edge_cost(edge);
            let edge_resource_value = edge_resource(edge);
            if edge_cost_value < zero || edge_resource_value < zero {
                return Err(ParetoPathError::NegativeWeight);
            }

            let next_cost = match edge_cost_value.checked_add_weight(cost) {
                Some(value) => value,
                None => return Err(ParetoPathError::WeightOverflow),
            };
            let next_resource = match edge_resource_value.checked_add_weight(resource) {
                Some(value) => value,
                // Resource overflow cannot fit any representable budget.
                None => continue,
            };
            if next_resource > resource_budget {
                continue;
            }

            let next_node = edge.target();
            if next_node == source {
                // Returning to the source can never beat the empty path.
                continue;
            }

            let parent = &paths[node_index][entry.label_index];
            if parent.nodes.contains(&next_node) {
                // Repeating a node closes a cycle; the non-negative cycle
                // weight makes this candidate dominated by (or equal to, with
                // a longer edge sequence) the label from the first visit.
                continue;
            }

            let mut next_nodes = parent.nodes.clone();
            next_nodes.push(next_node);
            let mut next_edges = parent.edges.clone();
            next_edges.push(edge.id());

            let next_index = graph.to_index(next_node);

            let mut dominated = false;
            let mut dominated_slots = 0usize;
            for slot in frontiers[next_index].iter() {
                let cost_cmp = slot.cost.cmp(&next_cost);
                let resource_cmp = slot.resource.cmp(&next_resource);
                match (cost_cmp, resource_cmp) {
                    (Ordering::Equal, Ordering::Equal) => {
                        // Identical metric: keep the lexicographically smallest
                        // edge identifier sequence.
                        let existing_edges = &paths[next_index][slot.label_index].edges;
                        if existing_edges.as_slice().cmp(next_edges.as_slice()) <= Ordering::Equal {
                            dominated = true;
                        } else {
                            dominated_slots += 1;
                        }
                    }
                    (Ordering::Less, Ordering::Less | Ordering::Equal)
                    | (Ordering::Equal, Ordering::Less) => {
                        dominated = true;
                    }
                    (Ordering::Greater, Ordering::Greater | Ordering::Equal)
                    | (Ordering::Equal, Ordering::Greater) => {
                        dominated_slots += 1;
                    }
                    (Ordering::Greater, Ordering::Less)
                    | (Ordering::Less, Ordering::Greater) => {
                        // Incomparable labels: both remain on the frontier.
                    }
                }
            }
            if dominated {
                continue;
            }

            // Drop every dominated slot before appending the new label. Using
            // retain (instead of swap_remove) preserves the correspondence
            // between each slot's label_index and entries already waiting in
            // the priority queue, which is required for lazy deletion.
            if dominated_slots > 0 {
                frontiers[next_index].retain(|slot| {
                    slot.cost < next_cost || slot.resource < next_resource
                });
            }

            let new_label_index = paths[next_index].len();
            paths[next_index].push(Label {
                cost: next_cost,
                resource: next_resource,
                last_node: next_node,
                nodes: next_nodes,
                edges: next_edges,
            });
            frontiers[next_index].push(FrontierSlot {
                cost: next_cost,
                resource: next_resource,
                label_index: new_label_index,
            });

            if frontiers[next_index].len() > max_labels {
                return Err(ParetoPathError::LabelLimitReached {
                    node_index: next_index,
                    max_labels,
                    stats: stats.clone(),
                });
            }
            if frontiers[next_index].len() > stats.max_frontier_size {
                stats.max_frontier_size = frontiers[next_index].len();
            }

            push_heap(
                &mut heap,
                &mut stats,
                HeapEntry {
                    resource: next_resource,
                    cost: next_cost,
                    node_index: next_index,
                    label_index: new_label_index,
                },
            );
        }
    }

    let target_index = graph.to_index(target);
    let mut labels: Vec<ParetoLabel<G::NodeId, G::EdgeId, W>> = frontiers[target_index]
        .iter()
        .map(|slot| {
            let label = &paths[target_index][slot.label_index];
            ParetoLabel {
                cost: label.cost,
                resource: label.resource,
                nodes: label.nodes.clone(),
                edges: label.edges.clone(),
            }
        })
        .collect();
    // Ascending cost, then ascending resource, for a deterministic answer.
    labels.sort_by(|a, b| {
        a.cost
            .cmp(&b.cost)
            .then_with(|| a.resource.cmp(&b.resource))
    });

    Ok(ParetoPaths { labels, stats })
}

fn push_heap<W>(
    heap: &mut BinaryHeap<MinScored<HeapKey<W>, HeapEntry<W>>>,
    stats: &mut ParetoSearchStats,
    entry: HeapEntry<W>,
) where
    W: Copy + PartialOrd,
{
    heap.push(MinScored(HeapKey(entry.resource, entry.cost), entry));
    stats.labels_generated += 1;
}
