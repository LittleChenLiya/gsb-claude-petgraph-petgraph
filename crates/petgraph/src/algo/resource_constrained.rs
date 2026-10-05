//! Resource-constrained shortest path: cheapest path within a resource budget.
//!
//! See [`resource_constrained_shortest_path`].

use alloc::{collections::BinaryHeap, vec::Vec};
use core::hash::Hash;

use hashbrown::HashMap;

use crate::{
    algo::Measure,
    scored::MinScored,
    visit::{EdgeRef, IntoEdges},
};

/// A measure (cost or resource) that supports checked addition.
///
/// This is implemented for all primitive integer types. The resource
/// consumption of an edge is expected to be a non-negative integer, and the
/// algorithm uses [`checked_add`](CheckedMeasure::checked_add) so that
/// arithmetic overflow is reported as an error instead of wrapping around.
pub trait CheckedMeasure: Measure + Copy + PartialEq {
    /// Checked addition. Returns `None` on overflow.
    fn checked_add(&self, rhs: &Self) -> Option<Self>;
}

macro_rules! impl_checked_measure {
    ($($t:ty),*) => {
        $(
            impl CheckedMeasure for $t {
                #[inline]
                fn checked_add(&self, rhs: &Self) -> Option<Self> {
                    <$t>::checked_add(*self, *rhs)
                }
            }
        )*
    };
}

impl_checked_measure!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

/// Errors returned by [`resource_constrained_shortest_path`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceConstrainedError {
    /// An edge examined during the search has a negative cost.
    NegativeCost,
    /// An edge examined during the search has a negative resource consumption.
    NegativeResource,
    /// Accumulating the cost of a path overflowed the cost type.
    CostOverflow,
    /// Accumulating the resource consumption of a path overflowed the resource type.
    ResourceOverflow,
}

impl core::fmt::Display for ResourceConstrainedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NegativeCost => f.write_str("edge cost must be non-negative"),
            Self::NegativeResource => f.write_str("edge resource consumption must be non-negative"),
            Self::CostOverflow => f.write_str("path cost overflowed"),
            Self::ResourceOverflow => f.write_str("path resource consumption overflowed"),
        }
    }
}

impl core::error::Error for ResourceConstrainedError {}

/// The result of [`resource_constrained_shortest_path`]: a cheapest feasible
/// path together with its exact cost and resource consumption.
///
/// Parallel edges are preserved: `edges` contains the actual edge ids of the
/// traversed edges, so the returned path is unambiguous even in a multigraph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceConstrainedPath<N, E, K, R> {
    /// Total cost of the path (sum of edge costs).
    pub cost: K,
    /// Total resource consumption of the path (sum of edge resources).
    pub resource: R,
    /// Nodes of the path, in order, starting at `start` and ending at `goal`.
    pub nodes: Vec<N>,
    /// Edges of the path, in order, identified by their edge id.
    pub edges: Vec<E>,
}

/// One label (state) of the label-setting search.
#[derive(Debug, Clone)]
struct Label<N, E, K, R> {
    node: N,
    cost: K,
    resource: R,
    /// Index of the parent label in the label arena (`None` for the start label).
    parent: Option<usize>,
    /// Edge through which this label was reached (`None` for the start label).
    edge: Option<E>,
}

/// Resource-constrained shortest path between two nodes.
///
/// Every edge carries a **cost** and a **non-negative integer resource
/// consumption**. This function finds a path from `start` to `goal` whose
/// total resource consumption does not exceed `budget` and whose total cost
/// is minimal among all such paths.
///
/// The graph is only read, never modified.
///
/// # Algorithm: label-setting with per-node Pareto frontiers
///
/// A *state* (label) is a pair `(cost, resource)` at a node. Because a cheap
/// but resource-hungry prefix can be a dead end while a more expensive but
/// frugal prefix still reaches the goal, **each node keeps multiple
/// non-dominated states** instead of only its cheapest value. A state `a`
/// *dominates* a state `b` at the same node iff
/// `a.cost <= b.cost && a.resource <= b.resource`; dominated states can never
/// lead to a better feasible solution and are discarded, while mutually
/// non-dominated states (the Pareto frontier) are all kept and expanded.
///
/// Labels are expanded in increasing cost order (a min-heap), so the first
/// time a non-dominated label at `goal` is popped, it is optimal.
///
/// # State limit: the budget is a hard cap
///
/// The `budget` is a hard **state limit**: any candidate state whose resource
/// consumption exceeds `budget` is rejected immediately and is never created,
/// queued, or expanded. The search therefore never explores the over-budget
/// part of the state space, and if every route to `goal` exceeds the budget
/// the function reports `Ok(None)` -- there is explicitly no route.
///
/// # Errors
///
/// * [`ResourceConstrainedError::NegativeCost`] /
///   [`ResourceConstrainedError::NegativeResource`]: an edge examined during
///   the search has a negative cost or resource consumption.
/// * [`ResourceConstrainedError::CostOverflow`] /
///   [`ResourceConstrainedError::ResourceOverflow`]: accumulating the cost or
///   the resource consumption of a path overflowed the (integer) measure type.
///
/// # Arguments
/// * `graph`: the graph to search (borrowed, not modified).
/// * `start`: the start node.
/// * `goal`: the goal node.
/// * `budget`: maximum allowed total resource consumption (hard state limit).
/// * `edge_metrics`: closure returning `(cost, resource)` for an edge.
///
/// # Returns
/// * `Ok(Some(path))`: a cheapest path within the budget, with the real
///   (recomputable) total cost, total resource consumption, node sequence and
///   the edge id sequence (parallel edges are preserved).
/// * `Ok(None)`: no path from `start` to `goal` fits within `budget`.
/// * `Err(_)`: negative edge metric or arithmetic overflow.
///
/// # Complexity
/// Each node can hold at most `budget + 1` non-dominated states (one per
/// attainable resource value), so the worst-case time complexity is
/// **O(budget * (|E| + |V| * budget) * log(|V| * budget))** and the auxiliary
/// space is **O(|V| * budget)** -- pseudo-polynomial in the budget. The
/// dominance pruning keeps the frontiers much smaller in practice.
///
/// # Example
/// ```rust
/// use petgraph::{Graph, algo::resource_constrained_shortest_path, prelude::*};
///
/// // Edge weight: (cost, resource).
/// let mut graph: Graph<(), (u64, u64)> = Graph::new();
/// let s = graph.add_node(());
/// let a = graph.add_node(());
/// let b = graph.add_node(());
/// let t = graph.add_node(());
///
/// // Cheap but resource-hungry prefix: s -> a -> t costs 2, consumes 20.
/// graph.add_edge(s, a, (1, 10));
/// graph.add_edge(a, t, (1, 10));
/// // Expensive but frugal prefix: s -> b -> t costs 10, consumes 2.
/// graph.add_edge(s, b, (5, 1));
/// graph.add_edge(b, t, (5, 1));
///
/// // Budget 11: the cheap route (resource 20) is infeasible, so the
/// // expensive frugal route wins. A per-node "cheapest only" search would
/// // wrongly discard it.
/// let path = resource_constrained_shortest_path(&graph, s, t, 11, |e| *e.weight())
///     .unwrap()
///     .unwrap();
/// assert_eq!(path.cost, 10);
/// assert_eq!(path.resource, 2);
/// assert_eq!(path.nodes, vec![s, b, t]);
///
/// // Budget 1: no route fits any more.
/// assert!(
///     resource_constrained_shortest_path(&graph, s, t, 1, |e| *e.weight())
///         .unwrap()
///         .is_none()
/// );
/// ```
pub fn resource_constrained_shortest_path<G, F, K, R>(
    graph: G,
    start: G::NodeId,
    goal: G::NodeId,
    budget: R,
    mut edge_metrics: F,
) -> Result<Option<ResourceConstrainedPath<G::NodeId, G::EdgeId, K, R>>, ResourceConstrainedError>
where
    G: IntoEdges,
    G::NodeId: Eq + Hash + Copy,
    G::EdgeId: Copy,
    F: FnMut(G::EdgeRef) -> (K, R),
    K: CheckedMeasure,
    R: CheckedMeasure,
{
    let zero_k = K::default();
    let zero_r = R::default();

    // Arena of all created labels; `alive[i]` is false once label `i` has
    // been dominated and removed from the frontier of its node.
    let mut labels: Vec<Label<G::NodeId, G::EdgeId, K, R>> = Vec::new();
    let mut alive: Vec<bool> = Vec::new();
    // Pareto frontier per node: indices of its non-dominated labels.
    let mut frontiers: HashMap<G::NodeId, Vec<usize>> = HashMap::new();
    // Min-heap of (cost, label index), expanded in increasing cost order.
    let mut heap: BinaryHeap<MinScored<K, usize>> = BinaryHeap::new();

    labels.push(Label {
        node: start,
        cost: zero_k,
        resource: zero_r,
        parent: None,
        edge: None,
    });
    alive.push(true);
    frontiers.entry(start).or_default().push(0);
    heap.push(MinScored(zero_k, 0));

    while let Some(MinScored(_, label_ix)) = heap.pop() {
        if !alive[label_ix] {
            // Stale heap entry: this label was dominated after being queued.
            continue;
        }
        let (node, cost, resource) = {
            let label = &labels[label_ix];
            (label.node, label.cost, label.resource)
        };

        if node == goal {
            // The first non-dominated label popped at the goal is optimal.
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            let mut cursor = Some(label_ix);
            while let Some(ix) = cursor {
                let label = &labels[ix];
                nodes.push(label.node);
                if let Some(edge) = label.edge {
                    edges.push(edge);
                }
                cursor = label.parent;
            }
            nodes.reverse();
            edges.reverse();
            return Ok(Some(ResourceConstrainedPath {
                cost,
                resource,
                nodes,
                edges,
            }));
        }

        for edge in graph.edges(node) {
            let (edge_cost, edge_resource) = edge_metrics(edge);
            if edge_cost < zero_k {
                return Err(ResourceConstrainedError::NegativeCost);
            }
            if edge_resource < zero_r {
                return Err(ResourceConstrainedError::NegativeResource);
            }
            let next_cost = cost
                .checked_add(&edge_cost)
                .ok_or(ResourceConstrainedError::CostOverflow)?;
            let next_resource = resource
                .checked_add(&edge_resource)
                .ok_or(ResourceConstrainedError::ResourceOverflow)?;

            // Hard state limit: never create a state that exceeds the budget.
            if next_resource > budget {
                continue;
            }

            let target = edge.target();
            let frontier = frontiers.entry(target).or_default();

            // Dominance check: skip the new state if an existing state at
            // `target` is at least as good in both dimensions.
            let dominated = frontier
                .iter()
                .any(|&ix| labels[ix].cost <= next_cost && labels[ix].resource <= next_resource);
            if dominated {
                continue;
            }
            // Remove (and kill) existing states dominated by the new state.
            frontier.retain(|&ix| {
                let keep = !(next_cost <= labels[ix].cost && next_resource <= labels[ix].resource);
                if !keep {
                    alive[ix] = false;
                }
                keep
            });

            let next_ix = labels.len();
            labels.push(Label {
                node: target,
                cost: next_cost,
                resource: next_resource,
                parent: Some(label_ix),
                edge: Some(edge.id()),
            });
            alive.push(true);
            frontier.push(next_ix);
            heap.push(MinScored(next_cost, next_ix));
        }
    }

    Ok(None)
}
