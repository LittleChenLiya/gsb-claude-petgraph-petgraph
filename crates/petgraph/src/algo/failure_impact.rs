//! Read-only single-node failure impact analysis.
//!
//! Given a start node and a list of important target nodes, this module computes,
//! for every *candidate* node, which targets become unreachable when that candidate
//! is removed.
//!
//! The analysis never mutates the input graph: each hypothetical removal is simulated
//! during a graph traversal by simply skipping the candidate node (and therefore every
//! edge incident to it). Removing a candidate node that is not part of the graph has no
//! effect, exactly as if it were temporarily deleted from an unchanged graph and then
//! re-added.
//!
//! Graphs may contain cycles and parallel (multi-)edges. Only a single reachable path
//! (a breadth-first search tree path) is reported per affected target; the search does
//! **not** enumerate all paths. Candidate nodes and target nodes are reported in the
//! stable order in which the caller provided them (de-duplicated by first occurrence).

use alloc::{fmt, vec, vec::Vec};

use crate::visit::{
    GraphRef, IntoNeighbors, IntoNodeIdentifiers, NodeIndexable, VisitMap, Visitable,
};

/// A target that lost reachability after a candidate node was removed.
#[derive(Clone, PartialEq, Eq)]
pub struct LostTarget<N> {
    /// The target node that is no longer reachable.
    pub target: N,

    /// One path from the start node to `target` that existed in the original graph
    /// (before the removal). It includes the start node as its first element and
    /// `target` as its last element. The path is a breadth-first search tree path,
    /// therefore it is simple (contains no repeated node) even when the graph has
    /// cycles or parallel edges.
    pub original_path: Vec<N>,
}

/// The failure impact of removing a single candidate node.
#[derive(Clone, PartialEq, Eq)]
pub struct NodeFailureImpact<N> {
    /// The candidate node whose hypothetical removal was analyzed.
    ///
    /// Candidates that do not identify a node of the graph are kept in the report as
    /// well; their hypothetical removal simply affects no targets.
    pub removed_node: N,

    /// Targets that were reachable from the start node in the original graph but are
    /// unreachable once `removed_node` is taken away, in caller-provided target order.
    ///
    /// Removing a target node itself counts: a target whose own node is the candidate
    /// is always reported here (unless it was already unreachable in the original
    /// graph).
    pub affected_targets: Vec<LostTarget<N>>,
}

/// The result of a failure impact analysis.
#[derive(Clone, PartialEq, Eq)]
pub struct FailureImpactReport<N> {
    /// One entry per unique candidate node, in caller-provided candidate order.
    pub node_impacts: Vec<NodeFailureImpact<N>>,

    /// Targets that were already unreachable from the start node in the original
    /// graph, in caller-provided target order. These are never counted as new failure
    /// impacts of any candidate removal.
    pub baseline_unreachable: Vec<N>,
}

impl<N: fmt::Debug> fmt::Debug for LostTarget<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LostTarget")
            .field("target", &self.target)
            .field("original_path", &self.original_path)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for NodeFailureImpact<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeFailureImpact")
            .field("removed_node", &self.removed_node)
            .field("affected_targets", &self.affected_targets)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for FailureImpactReport<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FailureImpactReport")
            .field("node_impacts", &self.node_impacts)
            .field("baseline_unreachable", &self.baseline_unreachable)
            .finish()
    }
}

/// Analyze which targets become unreachable when each candidate node is removed.
///
/// The graph is treated as read-only. For every candidate node a traversal is run on
/// the graph while pretending the candidate is absent; the original graph is never
/// modified (a candidate that is not part of the graph therefore affects no targets).
///
/// `targets` and `candidate_nodes` may contain duplicates; only the first occurrence of
/// each node is kept, so the output order is exactly the caller-provided order.
///
/// Cycles and parallel edges are supported, and for every affected target exactly one
/// previously reachable path is reported; the search never enumerates all paths.
///
/// If `start` does not identify a node of the graph, every target is considered
/// baseline-unreachable.
///
/// # Complexity
/// With `k` unique candidate nodes, `|V|` nodes and `|E|` edges:
/// * Time: **O((k + 1) · (|V| + |E|))**.
/// * Auxiliary space: **O(|V|)**.
///
/// # Example
/// ```rust
/// use petgraph::algo::failure_impact;
/// use petgraph::graph::DiGraph;
///
/// //     a -> t
/// //   /       ^
/// // s          |
/// //   \       /
/// //     b ---
/// let g = DiGraph::<(), ()>::from_edges([(0, 1), (1, 3), (0, 2), (2, 3)]);
///
/// let report = failure_impact(&g, 0.into(), [3.into()], [1.into(), 0.into()]);
/// // Removing `a` leaves s -> b -> t intact, so `t` is not affected.
/// assert!(report.node_impacts[0].affected_targets.is_empty());
/// // Removing the start node `s` disconnects `t`.
/// let lost = &report.node_impacts[1].affected_targets[0];
/// assert_eq!(lost.target, 3.into());
/// assert_eq!(lost.original_path.first(), Some(&0.into()));
/// assert_eq!(lost.original_path.last(), Some(&3.into()));
/// // The reported path actually exists in the original graph.
/// for edge in lost.original_path.windows(2) {
///     assert!(g.contains_edge(edge[0], edge[1]));
/// }
/// assert!(report.baseline_unreachable.is_empty());
/// ```
pub fn failure_impact<G, I, J>(
    graph: G,
    start: G::NodeId,
    targets: I,
    candidate_nodes: J,
) -> FailureImpactReport<G::NodeId>
where
    G: GraphRef + Visitable + NodeIndexable + IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: PartialEq,
    I: IntoIterator<Item = G::NodeId>,
    J: IntoIterator<Item = G::NodeId>,
{
    let targets = dedup(targets.into_iter());
    let candidates = dedup(candidate_nodes.into_iter());

    // Membership map of the nodes of the graph.
    let mut present = graph.visit_map();
    for node in graph.node_identifiers() {
        present.visit(node);
    }

    // A single breadth-first search of the original graph gives the baseline
    // reachability and one reachable path per target.
    let (baseline_parent, baseline_seen) = bfs_parents(graph, start, None, &present);

    let mut baseline_unreachable = Vec::new();
    for target in &targets {
        if !present.is_visited(target) || !baseline_seen.is_visited(target) {
            baseline_unreachable.push(*target);
        }
    }

    let mut node_impacts = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let mut affected_targets = Vec::new();

        if present.is_visited(&candidate) {
            // Re-run the traversal while pretending `candidate` is removed.
            let (_parent, seen) = bfs_parents(graph, start, Some(candidate), &present);
            for target in &targets {
                if baseline_seen.is_visited(target) && !seen.is_visited(target) {
                    affected_targets.push(LostTarget {
                        target: *target,
                        original_path: reconstruct_path(&baseline_parent, graph, start, *target),
                    });
                }
            }
        }

        node_impacts.push(NodeFailureImpact {
            removed_node: candidate,
            affected_targets,
        });
    }

    FailureImpactReport {
        node_impacts,
        baseline_unreachable,
    }
}

/// Keep the first occurrence of every node, preserving iterator order.
fn dedup<N: PartialEq, I: Iterator<Item = N>>(iter: I) -> Vec<N> {
    let mut unique = Vec::new();
    for node in iter {
        if !unique.contains(&node) {
            unique.push(node);
        }
    }
    unique
}

/// Breadth-first search from `start`, recording the parent of every reached node.
///
/// If `forbidden` is set, that node (and every edge incident to it) is skipped, which
/// simulates its removal without touching the graph.
///
/// `present` records which identifiers actually belong to the graph, so that an
/// out-of-range start identifier is handled as an empty traversal instead of indexing
/// into the visitor map out of bounds.
fn bfs_parents<G>(
    graph: G,
    start: G::NodeId,
    forbidden: Option<G::NodeId>,
    present: &G::Map,
) -> (Vec<Option<G::NodeId>>, G::Map)
where
    G: GraphRef + Visitable + NodeIndexable + IntoNeighbors,
{
    let mut parent = vec![None; graph.node_bound()];
    let mut seen = graph.visit_map();

    if !present.is_visited(&start)
        || forbidden.is_some_and(|forbidden_node| forbidden_node == start)
    {
        return (parent, seen);
    }
    seen.visit(start);

    let mut queue = alloc::collections::VecDeque::new();
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        for neighbor in graph.neighbors(node) {
            if forbidden.is_some_and(|forbidden_node| forbidden_node == neighbor) {
                continue;
            }
            if seen.visit(neighbor) {
                parent[graph.to_index(neighbor)] = Some(node);
                queue.push_back(neighbor);
            }
        }
    }

    (parent, seen)
}

/// Rebuild the path from `start` to `goal` stored in a BFS parent table.
fn reconstruct_path<G>(
    parent: &[Option<G::NodeId>],
    graph: G,
    start: G::NodeId,
    goal: G::NodeId,
) -> Vec<G::NodeId>
where
    G: NodeIndexable,
{
    let mut path = Vec::new();
    let mut current = Some(goal);
    while let Some(node) = current {
        path.push(node);
        if node == start {
            break;
        }
        current = parent[graph.to_index(node)];
    }
    path.reverse();
    path
}
