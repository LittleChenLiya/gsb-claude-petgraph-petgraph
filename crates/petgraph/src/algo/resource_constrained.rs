//! Resource-constrained shortest path: a label-setting algorithm with Pareto
//! dominance pruning.
//!
//! Every edge carries two non-negative attributes: a *cost* and a *resource
//! consumption*. The goal is to find a route from `start` to `goal` whose total
//! resource consumption does not exceed a given *budget* while minimizing the
//! total cost.
//!
//! Unlike plain Dijkstra, keeping only the cheapest value per node is **not**
//! correct here: a cheap but resource-hungry prefix must not prune a more
//! expensive but resource-light prefix, because the latter might be the only
//! one that can reach the goal within the budget. Instead, every node keeps a
//! *Pareto frontier* of non-dominated `(cost, resource)` labels.
//!
//! # Dominance
//!
//! A label `(c1, r1)` at a node *dominates* another label `(c2, r2)` at the
//! same node iff `c1 <= c2` **and** `r1 <= r2` (with at least one strict
//! inequality). The dominated label can safely be discarded: any extension of
//! it adds the same non-negative amounts to both cost and resource, so the
//! dominating label can follow the exact same extension and remain no worse on
//! both axes.
//!
//! Because all edge costs are non-negative, labels are popped from the heap in
//! non-decreasing cost order, so the first label popped at the goal is the
//! minimum-cost feasible route.
//!
//! # State limit
//!
//! The number of non-dominated labels at a node is bounded by `budget + 1`
//! (one label per distinct resource value in `0..=budget`), so for small
//! budgets the search is tiny. For large budgets the Pareto frontier can still
//! grow, hence the `max_states` parameter: it caps the total number of labels
//! ever generated. When the cap is exceeded the search aborts with
//! [`ResourceConstrainedError::StateLimitExceeded`] instead of consuming
//! unbounded time/memory. Pass a large value (e.g. `usize::MAX`) to disable the
//! cap.

use alloc::{collections::BinaryHeap, vec, vec::Vec};
use core::hash::Hash;

use crate::{
    algo::BoundedMeasure,
    scored::MinScored,
    visit::{EdgeRef, IntoEdges, NodeCount, NodeIndexable},
};

/// A route found by the resource-constrained shortest path algorithm.
///
/// The path is the minimum-cost route from `start` to `goal` whose total
/// resource consumption is at most the configured budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstrainedPath<N, E, C, R> {
    /// Nodes on the path, in order from `start` to `goal` (both inclusive).
    pub nodes: Vec<N>,
    /// Identifiers of the edges traversed, in order. Parallel edges between
    /// the same endpoints keep their distinct edge identifiers.
    pub edges: Vec<E>,
    /// Total cost of the path (sum of the edge costs).
    pub cost: C,
    /// Total resource consumed by the path (sum of the edge consumptions).
    pub resource: R,
}

/// Errors returned by the resource-constrained shortest path algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceConstrainedError {
    /// An edge has a negative cost. All edge costs must be non-negative.
    NegativeCost,
    /// An edge consumes a negative amount of resource. All consumptions must
    /// be non-negative.
    NegativeResource,
    /// An arithmetic overflow occurred while accumulating the total cost or
    /// resource consumption of a path.
    Overflow,
    /// The search generated more non-dominated states than the configured
    /// `max_states` limit.
    StateLimitExceeded {
        /// Number of states generated when the limit was hit.
        generated: usize,
        /// The configured limit.
        limit: usize,
    },
}

/// A non-dominated path state (label) attached to a node.
#[derive(Debug, Clone)]
struct Label<C, R, N, E> {
    /// Accumulated cost from the start to this node.
    cost: C,
    /// Accumulated resource consumption from the start to this node.
    resource: R,
    /// Predecessor information: the node, the index of the label at that node
    /// and the edge identifier used to reach this label. `None` for the start
    /// label.
    pred: Option<(N, usize, E)>,
}

/// Lowest-cost route from `start` to `goal` subject to a resource budget.
///
/// Every edge provides a *cost* (non-negative) and a *resource consumption*
/// (non-negative). The algorithm finds the route minimizing the total cost
/// among all routes whose total resource consumption does not exceed `budget`.
///
/// Nodes may hold several non-dominated labels, so a cheap but resource-heavy
/// prefix cannot prune a more expensive but resource-light prefix that is the
/// only one able to reach the goal within the budget.
///
/// # Arguments
/// * `graph`: the graph to search (it is not modified).
/// * `start`: the start node.
/// * `goal`: the goal node.
/// * `budget`: the maximum total resource consumption allowed.
/// * `max_states`: upper bound on the number of non-dominated labels generated
///   by the search. The search aborts with
///   [`ResourceConstrainedError::StateLimitExceeded`] if it is exceeded. For
///   small budgets the number of states is naturally bounded by
///   `|V| * (budget + 1)`; pass a large value to effectively disable the cap.
/// * `edge_cost`: closure returning the cost of an edge.
/// * `edge_resource`: closure returning the resource consumption of an edge.
///
/// # Returns
/// * `Ok(Some(path))`: the minimum-cost feasible route.
/// * `Ok(None)`: no route stays within the budget.
/// * `Err(ResourceConstrainedError)`: a negative cost/resource, an arithmetic
///   overflow, or the state limit was hit.
///
/// # Complexity
/// Let `S` be the number of non-dominated labels generated (`S <= max_states`).
/// * Time complexity: **O(S * d * log(S))** where `d` is the average node
///   degree (each label is expanded once and dominance is checked against the
///   labels of each reached node).
/// * Auxiliary space: **O(S)**.
///
/// # Example
/// ```rust
/// use petgraph::{Graph, algo::resource_constrained_shortest_path, prelude::*};
///
/// // start -> v: cheap but heavy (cost 1, resource 10) vs expensive but light
/// // (cost 5, resource 2); v -> goal: cost 1, resource 0. Budget is 5.
/// let mut graph: Graph<(), (u32, u32), Directed> = Graph::new();
/// let s = graph.add_node(());
/// let v = graph.add_node(());
/// let t = graph.add_node(());
/// let heavy = graph.add_edge(s, v, (1u32, 10u32));
/// let light = graph.add_edge(s, v, (5u32, 2u32));
/// let to_goal = graph.add_edge(v, t, (1u32, 0u32));
///
/// let path = resource_constrained_shortest_path(
///     &graph, s, t, 5u32, 1000,
///     |e| e.weight().0, |e| e.weight().1,
/// )
/// .unwrap()
/// .unwrap();
/// assert_eq!(path.cost, 6);
/// assert_eq!(path.resource, 2);
/// assert_eq!(path.edges, vec![light, to_goal]);
///
/// // With a budget of 1, no route is feasible.
/// assert!(resource_constrained_shortest_path(
///     &graph, s, t, 1u32, 1000,
///     |e| e.weight().0, |e| e.weight().1,
/// )
/// .unwrap()
/// .is_none());
/// ```
pub fn resource_constrained_shortest_path<G, F, H, C, R>(
    graph: G,
    start: G::NodeId,
    goal: G::NodeId,
    budget: R,
    max_states: usize,
    mut edge_cost: F,
    mut edge_resource: H,
) -> Result<Option<ConstrainedPath<G::NodeId, G::EdgeId, C, R>>, ResourceConstrainedError>
where
    G: IntoEdges + NodeCount + NodeIndexable,
    G::NodeId: Eq + Hash + Copy,
    G::EdgeId: Copy,
    F: FnMut(G::EdgeRef) -> C,
    H: FnMut(G::EdgeRef) -> R,
    C: BoundedMeasure + Copy,
    R: BoundedMeasure + Copy + Ord,
{
    let zero_cost = C::default();
    let zero_resource = R::default();

    // Per node: the list of non-dominated labels. Dominated labels are set to
    // `None` (tombstones) so that indices referenced by the heap stay stable.
    let mut labels: Vec<Vec<Option<Label<C, R, G::NodeId, G::EdgeId>>>> =
        vec![Vec::new(); graph.node_count()];
    let mut heap: BinaryHeap<MinScored<C, (G::NodeId, usize)>> = BinaryHeap::new();
    // Number of non-dominated labels ever generated (the start label counts).
    let mut generated = 1usize;

    let start_idx = graph.to_index(start);
    labels[start_idx].push(Some(Label {
        cost: zero_cost,
        resource: zero_resource,
        pred: None,
    }));
    heap.push(MinScored(zero_cost, (start, 0)));

    while let Some(MinScored(cost, (node, idx))) = heap.pop() {
        // Skip stale entries: the label may have been dominated after being
        // pushed onto the heap.
        let is_active = labels[graph.to_index(node)]
            .get(idx)
            .and_then(|l| l.as_ref())
            .is_some_and(|l| l.cost == cost);
        if !is_active {
            continue;
        }

        let label = labels[graph.to_index(node)][idx].as_ref().unwrap();
        if label.resource > budget {
            // Only the start label can reach this point (a negative budget
            // makes even the empty start path infeasible).
            continue;
        }

        if node == goal {
            // Reconstruct the path by following the predecessor links.
            let mut nodes = vec![goal];
            let mut edges = Vec::new();
            let mut cur_node = goal;
            let mut cur_idx = idx;
            while let Some(l) = labels[graph.to_index(cur_node)]
                .get(cur_idx)
                .and_then(|l| l.as_ref())
            {
                match l.pred {
                    Some((pred_node, pred_idx, edge)) => {
                        nodes.push(pred_node);
                        edges.push(edge);
                        cur_node = pred_node;
                        cur_idx = pred_idx;
                    }
                    None => break,
                }
            }
            nodes.reverse();
            edges.reverse();

            let goal_label = labels[graph.to_index(goal)][idx].as_ref().unwrap();
            return Ok(Some(ConstrainedPath {
                nodes,
                edges,
                cost: goal_label.cost,
                resource: goal_label.resource,
            }));
        }

        // Copy the label data out: the loop below mutates labels.
        let (label_cost, label_resource) = {
            let l = labels[graph.to_index(node)][idx].as_ref().unwrap();
            (l.cost, l.resource)
        };

        for edge in graph.edges(node) {
            let next = edge.target();
            let edge_cost = edge_cost(edge);
            let edge_resource = edge_resource(edge);

            if edge_cost < zero_cost {
                return Err(ResourceConstrainedError::NegativeCost);
            }
            if edge_resource < zero_resource {
                return Err(ResourceConstrainedError::NegativeResource);
            }

            let (new_cost, cost_overflow) = label_cost.overflowing_add(edge_cost);
            let (new_resource, resource_overflow) =
                label_resource.overflowing_add(edge_resource);
            if cost_overflow || resource_overflow {
                return Err(ResourceConstrainedError::Overflow);
            }
            if new_resource > budget {
                continue;
            }

            let next_idx = graph.to_index(next);

            // Dominance check: is the new label dominated by an existing one?
            let dominated = labels[next_idx].iter().any(|l| {
                l.as_ref()
                    .is_some_and(|l| l.cost <= new_cost && l.resource <= new_resource)
            });
            if dominated {
                continue;
            }

            // Remove existing labels dominated by the new label.
            for slot in labels[next_idx].iter_mut() {
                if let Some(l) = slot {
                    if new_cost <= l.cost && new_resource <= l.resource {
                        *slot = None;
                    }
                }
            }

            let new_idx = labels[next_idx].len();
            labels[next_idx].push(Some(Label {
                cost: new_cost,
                resource: new_resource,
                pred: Some((node, idx, edge.id())),
            }));
            heap.push(MinScored(new_cost, (next, new_idx)));

            generated += 1;
            if generated > max_states {
                return Err(ResourceConstrainedError::StateLimitExceeded {
                    generated,
                    limit: max_states,
                });
            }
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use crate::Graph;

    /// The key scenario: a cheap but resource-heavy prefix must not prune an
    /// expensive but resource-light prefix, because only the latter reaches
    /// the goal within the budget.
    #[test]
    fn cheap_heavy_prefix_does_not_prune_expensive_light_one() {
        let mut graph: Graph<(), (u32, u32), Directed> = Graph::new();
        let s = graph.add_node(());
        let v = graph.add_node(());
        let t = graph.add_node(());
        // s -> v: (cost 1, resource 10) vs (cost 5, resource 2)
        graph.add_edge(s, v, (1, 10));
        graph.add_edge(s, v, (5, 2));
        graph.add_edge(v, t, (1, 0));

        // Budget 5: the heavy prefix (resource 10) is infeasible, the light
        // one (resource 2) is. Cheapest feasible cost is 6.
        let path = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            5,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(path.cost, 6);
        assert_eq!(path.resource, 2);
        assert_eq!(path.nodes, vec![s, v, t]);

        // If we had kept only the cheapest label per node, the (1, 10) prefix
        // would have been the only one and no route would exist. Verify the
        // budget reduction makes the route disappear explicitly.
        assert!(resource_constrained_shortest_path(
            &graph,
            s,
            t,
            1,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap()
        .is_none());
    }

    /// Parallel edges keep their distinct edge identifiers in the result.
    #[test]
    fn parallel_edges_keep_edge_id() {
        let mut graph: Graph<(), (u32, u32), Directed> = Graph::new();
        let s = graph.add_node(());
        let t = graph.add_node(());
        let e1 = graph.add_edge(s, t, (10, 0));
        let e2 = graph.add_edge(s, t, (3, 1));
        let e3 = graph.add_edge(s, t, (5, 5));

        // e2 (cost 3, resource 1) is the cheapest feasible edge.
        let path = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            5,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(path.cost, 3);
        assert_eq!(path.resource, 1);
        assert_eq!(path.edges, vec![e2]);
        assert_ne!(e1, e2);
        assert_ne!(e2, e3);
    }

    /// Negative edge costs are rejected.
    #[test]
    fn negative_cost_rejected() {
        let mut graph: Graph<(), (i32, i32), Directed> = Graph::new();
        let s = graph.add_node(());
        let t = graph.add_node(());
        graph.add_edge(s, t, (-1, 0));
        let err = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            10,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap_err();
        assert_eq!(err, ResourceConstrainedError::NegativeCost);
    }

    /// Negative edge resource consumptions are rejected.
    #[test]
    fn negative_resource_rejected() {
        let mut graph: Graph<(), (i32, i32), Directed> = Graph::new();
        let s = graph.add_node(());
        let t = graph.add_node(());
        graph.add_edge(s, t, (1, -1));
        let err = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            10,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap_err();
        assert_eq!(err, ResourceConstrainedError::NegativeResource);
    }

    /// Arithmetic overflow in cost or resource is rejected.
    #[test]
    fn overflow_rejected() {
        let mut graph: Graph<(), (u8, u8), Directed> = Graph::new();
        let s = graph.add_node(());
        let v = graph.add_node(());
        let t = graph.add_node(());
        graph.add_edge(s, v, (200, 0));
        graph.add_edge(v, t, (100, 0));
        let err = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            255,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap_err();
        assert_eq!(err, ResourceConstrainedError::Overflow);

        let mut graph: Graph<(), (u8, u8), Directed> = Graph::new();
        let s = graph.add_node(());
        let v = graph.add_node(());
        let t = graph.add_node(());
        graph.add_edge(s, v, (0, 200));
        graph.add_edge(v, t, (0, 100));
        let err = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            255,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap_err();
        assert_eq!(err, ResourceConstrainedError::Overflow);
    }

    /// The state limit explicitly aborts the search.
    #[test]
    fn state_limit_aborts() {
        let mut graph: Graph<(), (u32, u32), Directed> = Graph::new();
        let s = graph.add_node(());
        let v = graph.add_node(());
        let t = graph.add_node(());
        graph.add_edge(s, v, (1, 1));
        graph.add_edge(v, t, (1, 1));
        // max_states = 1: the start label is the only one allowed.
        let err = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            10,
            1,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ResourceConstrainedError::StateLimitExceeded {
                generated: 2,
                limit: 1
            }
        );
    }

    /// Start equals goal: the empty path is returned with zero cost and
    /// resource.
    #[test]
    fn start_equals_goal() {
        let mut graph: Graph<(), (u32, u32), Directed> = Graph::new();
        let s = graph.add_node(());
        let path = resource_constrained_shortest_path(
            &graph,
            s,
            s,
            5,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(path.cost, 0);
        assert_eq!(path.resource, 0);
        assert_eq!(path.nodes, vec![s]);
        assert!(path.edges.is_empty());
    }

    /// Unreachable goal yields no route.
    #[test]
    fn unreachable_goal() {
        let mut graph: Graph<(), (u32, u32), Directed> = Graph::new();
        let s = graph.add_node(());
        let t = graph.add_node(());
        graph.add_edge(s, s, (1, 1));
        assert!(resource_constrained_shortest_path(
            &graph,
            s,
            t,
            10,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap()
        .is_none());
    }

    /// A zero-cost cycle does not generate infinitely many states: the
    /// dominated label is pruned.
    #[test]
    fn zero_cost_cycle_is_pruned() {
        let mut graph: Graph<(), (u32, u32), Directed> = Graph::new();
        let s = graph.add_node(());
        let v = graph.add_node(());
        let t = graph.add_node(());
        graph.add_edge(s, v, (1, 1));
        graph.add_edge(v, s, (0, 0)); // zero-cost cycle back
        graph.add_edge(v, t, (1, 1));
        let path = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            10,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(path.cost, 2);
        assert_eq!(path.resource, 2);
    }

    /// Exhaustive check on a small graph: enumerate every simple path and
    /// compare the minimum feasible cost against the result for several
    /// budgets. This validates the dominance relation: a dominated prefix must
    /// never hide the optimal route.
    #[test]
    fn exhaustive_small_graph() {
        let mut graph: Graph<(), (u32, u32), Directed> = Graph::new();
        let nodes: Vec<_> = (0..5).map(|_| graph.add_node(())).collect();
        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    continue;
                }
                let c = ((i * 7 + j * 13) % 4) as u32;
                let r = ((i * 5 + j * 11) % 4) as u32;
                graph.add_edge(nodes[i], nodes[j], (c, r));
                if (i + j) % 2 == 0 {
                    // A parallel edge with different attributes.
                    let c = ((i * 3 + j * 17 + 1) % 4) as u32;
                    let r = ((i * 19 + j * 7 + 1) % 4) as u32;
                    graph.add_edge(nodes[i], nodes[j], (c, r));
                }
            }
        }

        let s = nodes[0];
        let t = nodes[4];

        // Brute-force enumeration of every simple path.
        let mut all_paths: Vec<(u32, u32)> = Vec::new();
        let mut stack: Vec<(NodeIndex, u32, u32, Vec<NodeIndex>)> =
            vec![(s, 0, 0, vec![s])];
        while let Some((node, cost, res, visited)) = stack.pop() {
            if node == t {
                all_paths.push((cost, res));
                continue;
            }
            for edge in graph.edges(node) {
                let next = edge.target();
                if visited.contains(&next) {
                    continue;
                }
                let mut v = visited.clone();
                v.push(next);
                stack.push((next, cost + edge.weight().0, res + edge.weight().1, v));
            }
        }
        assert!(!all_paths.is_empty());

        for budget in 0..=12u32 {
            let expected = all_paths
                .iter()
                .filter(|&&(_, r)| r <= budget)
                .map(|&(c, _)| c)
                .min();
            let got = resource_constrained_shortest_path(
                &graph,
                s,
                t,
                budget,
                100_000,
                |e| e.weight().0,
                |e| e.weight().1,
            )
            .unwrap();
            match (expected, got) {
                (Some(exp), Some(path)) => {
                    assert_eq!(path.cost, exp, "budget {budget}");
                    assert!(path.resource <= budget, "budget {budget}");
                    // The returned accumulated values must be real.
                    let (real_cost, real_res) =
                        path.edges.iter().fold((0u32, 0u32), |(c, r), &e| {
                            let w = graph.edge_weight(e).unwrap();
                            (c + w.0, r + w.1)
                        });
                    assert_eq!(real_cost, path.cost);
                    assert_eq!(real_res, path.resource);
                    // Nodes must form a valid path.
                    assert_eq!(path.nodes.len(), path.edges.len() + 1);
                    assert_eq!(path.nodes[0], s);
                    assert_eq!(*path.nodes.last().unwrap(), t);
                }
                (None, None) => {}
                (exp, got) => panic!("budget {budget}: expected {exp:?}, got {got:?}"),
            }
        }
    }

    /// Both non-dominated labels at an intermediate node survive (neither
    /// dominates the other), and each leads to a different feasible route.
    /// This directly verifies the dominance relation: a label is pruned only
    /// when another label is no worse on *both* axes.
    #[test]
    fn pareto_frontier_at_intermediate_node() {
        let mut graph: Graph<(), (u32, u32), Directed> = Graph::new();
        let s = graph.add_node(());
        let v = graph.add_node(());
        let t = graph.add_node(());
        // s -> v: (cost 1, resource 5) and (cost 3, resource 2).
        // Neither dominates the other: 1 < 3 but 5 > 2.
        graph.add_edge(s, v, (1, 5));
        graph.add_edge(s, v, (3, 2));
        // v -> t: (cost 1, resource 0) and (cost 0, resource 4).
        graph.add_edge(v, t, (1, 0));
        graph.add_edge(v, t, (0, 4));

        // Budget 5: feasible routes are (2,5) [via (1,5)+(1,0)] and
        // (4,2) [via (3,2)+(1,0)]. The cheapest is cost 2.
        let path = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            5,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(path.cost, 2);
        assert_eq!(path.resource, 5);

        // Budget 2: only (4,2) is feasible. If the (1,5) label had pruned the
        // (3,2) label at v, no route would exist.
        let path = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            2,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(path.cost, 4);
        assert_eq!(path.resource, 2);

        // Budget 9: the cheapest overall is (1,9) via (1,5)+(0,4).
        let path = resource_constrained_shortest_path(
            &graph,
            s,
            t,
            9,
            1000,
            |e| e.weight().0,
            |e| e.weight().1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(path.cost, 1);
        assert_eq!(path.resource, 9);
    }
}
