//! Read-only failure impact analysis for graphs.
//!
//! This module provides a [`failure_impact`] function that analyzes the impact of
//! (virtually) removing each candidate node from a graph. The analysis is **read-only**:
//! the input graph is never modified. Instead of actually deleting nodes and restoring
//! them, the removal is simulated by treating the candidate node as blocked during
//! traversal.

use alloc::{collections::VecDeque, vec, vec::Vec};

use crate::visit::{IntoNeighbors, NodeIndexable, VisitMap, Visitable};

/// The impact of (virtually) removing a single candidate node from the graph.
///
/// See [`failure_impact`] for the full analysis and field semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailureImpact<N> {
    /// The candidate node whose removal was simulated.
    pub candidate: N,
    /// Targets that were reachable from `start` in the original graph but became
    /// unreachable after removing [`candidate`](Self::candidate).
    ///
    /// Each entry pairs the lost target with one path from `start` to that target
    /// in the original (unmodified) graph. Output follows the caller's target order.
    pub lost_targets: Vec<LostTarget<N>>,
    /// Targets that were already unreachable from `start` in the original graph.
    ///
    /// These are **not** counted as new failures: they were unreachable before any
    /// node was removed. Output follows the caller's target order.
    pub baseline_unreachable: Vec<N>,
}

/// A target that lost reachability after removing a candidate node, together with
/// one path from `start` to that target in the original graph.
///
/// The path is guaranteed to exist in the original graph (before the candidate was
/// removed). It is one arbitrary path, not necessarily the shortest, and the analyzer
/// does not enumerate all paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LostTarget<N> {
    /// The target node that became unreachable.
    pub target: N,
    /// One path from `start` to [`target`](Self::target) in the original graph.
    ///
    /// The path is a sequence of nodes starting at `start` and ending at `target`.
    pub original_path: Vec<N>,
}

/// Analyze the impact of (virtually) removing each candidate node from the graph.
///
/// Given a `start` node and a list of key `targets`, this function computes, for each
/// `candidate` node, which targets become unreachable from `start` after virtually
/// removing the candidate. The analysis is **read-only**: the input graph is never
/// modified. The candidate node is never actually deleted; its removal is simulated
/// by treating it as blocked during traversal.
///
/// For each candidate, the function reports:
/// - Which targets lose reachability (were reachable before, unreachable after).
/// - For each lost target, one path from `start` to that target in the original graph.
/// - Which targets were already unreachable in the baseline (and thus are not new failures).
///
/// Targets that were already unreachable from `start` in the original graph are listed
/// separately in [`FailureImpact::baseline_unreachable`] and are **not** counted as new
/// failure impacts.
///
/// Removing a target node itself counts as affected: if the candidate is a target, that
/// target becomes unreachable (unless another path bypasses the candidate).
///
/// Cycles and multi-edges are supported. The analysis uses BFS, so it does not enumerate
/// all paths; it finds one path per reachable target.
///
/// # Arguments
/// * `graph`: the input graph. It is not modified.
/// * `start`: the starting node for reachability analysis.
/// * `targets`: the key target nodes to check. Output order follows this slice.
/// * `candidates`: the candidate nodes whose removal impact is analyzed. Output order
///   follows this slice.
///
/// # Returns
/// A `Vec<FailureImpact>` with one entry per candidate, in the caller's candidate order.
///
/// # Complexity
/// * Time complexity: **O(C * (|V| + |E|) + T * |V|)**, where `C` is the number of
///   candidates and `T` is the number of targets.
/// * Auxiliary space: **O(|V| + T)** per candidate.
///
/// where **|V|** is the number of nodes and **|E|** is the number of edges.
///
/// # Examples
/// ```
/// use petgraph::{Graph, algo::failure_impact};
///
/// // s -> a -> t
/// // s -> b -> t
/// let mut g = Graph::<(), ()>::new();
/// let s = g.add_node(());
/// let a = g.add_node(());
/// let b = g.add_node(());
/// let t = g.add_node(());
/// let u = g.add_node(()); // unreachable from s
/// g.add_edge(s, a, ());
/// g.add_edge(a, t, ());
/// g.add_edge(s, b, ());
/// g.add_edge(b, t, ());
///
/// let impacts = failure_impact(&g, s, &[t, u], &[a, s]);
///
/// // Removing a: t is still reachable via b, so no lost targets.
/// assert_eq!(impacts[0].candidate, a);
/// assert!(impacts[0].lost_targets.is_empty());
/// assert_eq!(impacts[0].baseline_unreachable, vec![u]);
///
/// // Removing s: nothing is reachable, so t is lost.
/// assert_eq!(impacts[1].candidate, s);
/// assert_eq!(impacts[1].lost_targets.len(), 1);
/// assert_eq!(impacts[1].lost_targets[0].target, t);
/// // The path is a valid path from s to t in the original graph.
/// let path = &impacts[1].lost_targets[0].original_path;
/// assert_eq!(path.first(), Some(&s));
/// assert_eq!(path.last(), Some(&t));
/// assert_eq!(impacts[1].baseline_unreachable, vec![u]);
/// ```
pub fn failure_impact<G>(
    graph: G,
    start: G::NodeId,
    targets: &[G::NodeId],
    candidates: &[G::NodeId],
) -> Vec<FailureImpact<G::NodeId>>
where
    G: IntoNeighbors + Visitable + NodeIndexable,
    G::NodeId: Copy + Eq,
{
    // Baseline BFS: compute reachability and parent map in the original graph.
    let (baseline_visited, baseline_parent) = bfs(graph, start, None);

    let mut results = Vec::with_capacity(candidates.len());

    for &candidate in candidates {
        // Simulate removing the candidate by treating it as blocked.
        let (candidate_visited, _) = bfs(graph, start, Some(candidate));

        let mut lost_targets = Vec::new();
        let mut baseline_unreachable = Vec::new();

        for &target in targets {
            let baseline_reachable = baseline_visited.is_visited(&target);
            let candidate_reachable = candidate_visited.is_visited(&target);

            if !baseline_reachable {
                // Was already unreachable before any removal; not a new failure.
                baseline_unreachable.push(target);
            } else if !candidate_reachable {
                // Lost reachability due to the candidate's removal.
                let original_path = reconstruct_path(graph, &baseline_parent, start, target);
                lost_targets.push(LostTarget {
                    target,
                    original_path,
                });
            }
        }

        results.push(FailureImpact {
            candidate,
            lost_targets,
            baseline_unreachable,
        });
    }

    results
}

/// BFS from `start`, optionally treating `blocked` as removed.
///
/// Returns the visit map and a parent map (for path reconstruction).
fn bfs<G>(
    graph: G,
    start: G::NodeId,
    blocked: Option<G::NodeId>,
) -> (G::Map, Vec<Option<G::NodeId>>)
where
    G: IntoNeighbors + Visitable + NodeIndexable,
    G::NodeId: Copy + Eq,
{
    let mut visited = graph.visit_map();
    let mut parent = vec![None; graph.node_bound()];
    let mut queue = VecDeque::new();

    // If the start node is blocked, nothing is reachable.
    if let Some(b) = blocked {
        if b == start {
            return (visited, parent);
        }
    }

    visited.visit(start);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        for neighbor in graph.neighbors(node) {
            // Skip the blocked node as if it were removed.
            if let Some(b) = blocked {
                if neighbor == b {
                    continue;
                }
            }
            if visited.visit(neighbor) {
                parent[graph.to_index(neighbor)] = Some(node);
                queue.push_back(neighbor);
            }
        }
    }

    (visited, parent)
}

/// Reconstruct a path from `start` to `target` using the parent map.
fn reconstruct_path<G>(
    graph: G,
    parent: &[Option<G::NodeId>],
    start: G::NodeId,
    target: G::NodeId,
) -> Vec<G::NodeId>
where
    G: NodeIndexable,
    G::NodeId: Copy + Eq,
{
    let mut path = Vec::new();
    let mut current = target;
    while current != start {
        path.push(current);
        match parent[graph.to_index(current)] {
            Some(p) => current = p,
            None => {
                // Should not happen if target is reachable, but return what we have.
                return path;
            }
        }
    }
    path.push(start);
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_basic_failure_impact() {
        // s -> a -> t
        // s -> b -> t
        let mut g = Graph::<(), ()>::new();
        let s = g.add_node(());
        let a = g.add_node(());
        let b = g.add_node(());
        let t = g.add_node(());
        let u = g.add_node(()); // unreachable from s
        g.add_edge(s, a, ());
        g.add_edge(a, t, ());
        g.add_edge(s, b, ());
        g.add_edge(b, t, ());

        let impacts = failure_impact(&g, s, &[t, u], &[a, s]);

        // Removing a: t is still reachable via b, so no lost targets.
        assert_eq!(impacts[0].candidate, a);
        assert!(impacts[0].lost_targets.is_empty());
        assert_eq!(impacts[0].baseline_unreachable, vec![u]);

        // Removing s: nothing is reachable, so t is lost.
        assert_eq!(impacts[1].candidate, s);
        assert_eq!(impacts[1].lost_targets.len(), 1);
        assert_eq!(impacts[1].lost_targets[0].target, t);
        // The path must be a valid path from s to t in the original graph.
        let path = &impacts[1].lost_targets[0].original_path;
        assert_eq!(path.first(), Some(&s));
        assert_eq!(path.last(), Some(&t));
        // Every consecutive pair in the path must be connected by an edge.
        for window in path.windows(2) {
            assert!(
                g.find_edge(window[0], window[1]).is_some(),
                "expected edge from {:?} to {:?}",
                window[0],
                window[1]
            );
        }
        assert_eq!(impacts[1].baseline_unreachable, vec![u]);
    }

    #[test]
    fn test_remove_target_itself() {
        // s -> a -> t
        let mut g = Graph::<(), ()>::new();
        let s = g.add_node(());
        let a = g.add_node(());
        let t = g.add_node(());
        g.add_edge(s, a, ());
        g.add_edge(a, t, ());

        // Removing t itself should make t unreachable.
        let impacts = failure_impact(&g, s, &[t], &[t]);
        assert_eq!(impacts[0].candidate, t);
        assert_eq!(impacts[0].lost_targets.len(), 1);
        assert_eq!(impacts[0].lost_targets[0].target, t);
        // The path must be a valid path from s to t in the original graph.
        let path = &impacts[0].lost_targets[0].original_path;
        assert_eq!(path.first(), Some(&s));
        assert_eq!(path.last(), Some(&t));
        for window in path.windows(2) {
            assert!(
                g.find_edge(window[0], window[1]).is_some(),
                "expected edge from {:?} to {:?}",
                window[0],
                window[1]
            );
        }
        assert!(impacts[0].baseline_unreachable.is_empty());
    }

    #[test]
    fn test_cycles_and_multi_edges() {
        // s -> a -> t
        // s -> b -> t  (with a cycle and multi-edge)
        let mut g = Graph::<(), ()>::new();
        let s = g.add_node(());
        let a = g.add_node(());
        let b = g.add_node(());
        let t = g.add_node(());
        g.add_edge(s, a, ());
        g.add_edge(a, t, ());
        g.add_edge(s, b, ());
        g.add_edge(b, t, ());
        g.add_edge(a, b, ()); // extra edge
        g.add_edge(a, b, ()); // multi-edge
        g.add_edge(t, s, ()); // cycle

        let impacts = failure_impact(&g, s, &[t], &[a]);
        // Removing a: t is still reachable via b.
        assert!(impacts[0].lost_targets.is_empty());
    }

    #[test]
    fn test_duplicate_candidates() {
        let mut g = Graph::<(), ()>::new();
        let s = g.add_node(());
        let a = g.add_node(());
        let t = g.add_node(());
        g.add_edge(s, a, ());
        g.add_edge(a, t, ());

        // Duplicate candidates should produce duplicate results.
        let impacts = failure_impact(&g, s, &[t], &[a, a]);
        assert_eq!(impacts.len(), 2);
        assert_eq!(impacts[0].candidate, a);
        assert_eq!(impacts[1].candidate, a);
        // Both should report t as lost.
        assert_eq!(impacts[0].lost_targets.len(), 1);
        assert_eq!(impacts[1].lost_targets.len(), 1);
    }

    #[test]
    fn test_graph_unchanged() {
        let mut g = Graph::<(), ()>::new();
        let s = g.add_node(());
        let a = g.add_node(());
        let t = g.add_node(());
        g.add_edge(s, a, ());
        g.add_edge(a, t, ());

        let node_count_before = g.node_count();
        let edge_count_before = g.edge_count();
        let edges_before: Vec<_> = g
            .raw_edges()
            .iter()
            .map(|e| (e.source().index(), e.target().index()))
            .collect();

        let _ = failure_impact(&g, s, &[t], &[a, s]);

        // The graph should be unchanged after analysis.
        assert_eq!(g.node_count(), node_count_before);
        assert_eq!(g.edge_count(), edge_count_before);
        let edges_after: Vec<_> = g
            .raw_edges()
            .iter()
            .map(|e| (e.source().index(), e.target().index()))
            .collect();
        assert_eq!(edges_after, edges_before);
    }

    #[test]
    fn test_caller_order() {
        let mut g = Graph::<(), ()>::new();
        let s = g.add_node(());
        let a = g.add_node(());
        let b = g.add_node(());
        let t1 = g.add_node(());
        let t2 = g.add_node(());
        let u = g.add_node(());
        g.add_edge(s, a, ());
        g.add_edge(a, t1, ());
        g.add_edge(s, b, ());
        g.add_edge(b, t2, ());

        // Targets in a specific order; output should follow it.
        let targets = &[t2, u, t1];
        let candidates = &[b, a];
        let impacts = failure_impact(&g, s, targets, candidates);

        // Candidate b: t2 is lost, t1 is reachable via a, u is baseline unreachable.
        assert_eq!(impacts[0].candidate, b);
        assert_eq!(impacts[0].lost_targets.len(), 1);
        assert_eq!(impacts[0].lost_targets[0].target, t2);
        assert_eq!(impacts[0].baseline_unreachable, vec![u]);

        // Candidate a: t1 is lost, t2 is reachable via b, u is baseline unreachable.
        assert_eq!(impacts[1].candidate, a);
        assert_eq!(impacts[1].lost_targets.len(), 1);
        assert_eq!(impacts[1].lost_targets[0].target, t1);
        assert_eq!(impacts[1].baseline_unreachable, vec![u]);
    }

    #[test]
    fn test_undirected_graph() {
        use crate::graph::UnGraph;
        let mut g = UnGraph::<(), ()>::new_undirected();
        let s = g.add_node(());
        let a = g.add_node(());
        let t = g.add_node(());
        g.add_edge(s, a, ());
        g.add_edge(a, t, ());

        let impacts = failure_impact(&g, s, &[t], &[a]);
        assert_eq!(impacts[0].lost_targets.len(), 1);
        assert_eq!(impacts[0].lost_targets[0].target, t);
    }

    #[test]
    fn test_no_candidates_or_targets() {
        let mut g = Graph::<(), ()>::new();
        let s = g.add_node(());
        let a = g.add_node(());
        g.add_edge(s, a, ());

        // Empty candidates.
        let impacts = failure_impact(&g, s, &[a], &[]);
        assert!(impacts.is_empty());

        // Empty targets.
        let impacts = failure_impact(&g, s, &[], &[a]);
        assert_eq!(impacts.len(), 1);
        assert!(impacts[0].lost_targets.is_empty());
        assert!(impacts[0].baseline_unreachable.is_empty());
    }

    #[test]
    fn test_graphmap() {
        use crate::Directed;
        use crate::graphmap::GraphMap;

        // GraphMap uses node weights as NodeId.
        let mut g = GraphMap::<char, (), Directed>::new();
        g.add_edge('s', 'a', ());
        g.add_edge('a', 't', ());
        g.add_edge('s', 'b', ());
        g.add_edge('b', 't', ());
        g.add_node('u'); // unreachable from s

        let impacts = failure_impact(&g, 's', &['t', 'u'], &['a', 's']);

        // Removing a: t is still reachable via b.
        assert_eq!(impacts[0].candidate, 'a');
        assert!(impacts[0].lost_targets.is_empty());
        assert_eq!(impacts[0].baseline_unreachable, vec!['u']);

        // Removing s: t is lost.
        assert_eq!(impacts[1].candidate, 's');
        assert_eq!(impacts[1].lost_targets.len(), 1);
        assert_eq!(impacts[1].lost_targets[0].target, 't');
        assert_eq!(impacts[1].baseline_unreachable, vec!['u']);
    }
}
