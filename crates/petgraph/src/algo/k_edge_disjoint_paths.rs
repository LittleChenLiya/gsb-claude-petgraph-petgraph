use alloc::{collections::BinaryHeap, vec, vec::Vec};

use crate::{
    scored::MinScored,
    visit::{EdgeRef, IntoEdgesDirected, NodeIndexable},
};

/// A residual arc in the flow network built by [`k_edge_disjoint_paths`].
#[derive(Debug, Clone, Copy)]
struct Arc {
    to: usize,
    /// Remaining residual capacity, either `0` or `1`.
    capacity: u32,
    /// Cost of the arc; reverse arcs carry the negated cost of their edge.
    cost: i128,
    /// Index of the reverse arc.
    reverse: usize,
    /// For forward arcs, the index of the original graph edge.
    edge: Option<usize>,
}

/// Minimum-cost `k` edge-disjoint paths.
///
/// Compute `k` paths from `start` to `goal` in a directed graph such that no
/// two paths share an edge, and the total cost of all used edges is minimal.
/// Every edge of the input graph keeps its own identity, so parallel edges
/// are accounted for separately through their distinct `EdgeId`s. Each
/// returned path is simple: it never visits the same node twice. The input
/// graph is not modified.
///
/// Edge costs must be non-negative integers. The problem is solved as a
/// minimum-cost flow of value `k` with unit capacities, using successive
/// shortest augmenting paths with Johnson potentials.
///
/// # Arguments
/// * `graph`: an input directed graph.
/// * `start`: the *start* node, source of all paths.
/// * `goal`: the *goal* node, target of all paths.
/// * `k`: the number of edge-disjoint paths to find.
/// * `edge_cost`: closure that should return the non-negative cost of a
///   particular edge.
///
/// # Returns
/// * `Some((total_cost, paths))` where `paths` contains exactly `k` paths,
///   each given as the sequence of `EdgeId`s it uses, and `total_cost` is the
///   minimal sum of the costs of all used edges.
/// * `None` if fewer than `k` edge-disjoint paths from `start` to `goal`
///   exist, or if the total cost of the optimal solution overflows `u64`.
///
/// # Complexity
/// * Time complexity: **O(k(|E| + |V|) log |V|)**.
/// * Auxiliary space: **O(|V| + |E|)**.
///
/// where **|V|** is the number of nodes and **|E|** is the number of edges.
///
/// # Example
/// ```rust
/// use petgraph::{Graph, algo::k_edge_disjoint_paths, prelude::*};
///
/// let mut graph: Graph<(), u64> = Graph::new();
/// let s = graph.add_node(());
/// let a = graph.add_node(());
/// let b = graph.add_node(());
/// let t = graph.add_node(());
///
/// // The shortest s-t path (s -> a -> b -> t, cost 0) blocks every other
/// // s-t path, so the optimum uses the two cost-5 detours instead.
/// graph.extend_with_edges([
///     (s, a, 0),
///     (a, b, 0),
///     (b, t, 0),
///     (s, b, 5),
///     (a, t, 5),
/// ]);
///
/// let (total_cost, paths) = k_edge_disjoint_paths(&graph, s, t, 2, |e| *e.weight()).unwrap();
/// assert_eq!(total_cost, 10);
/// assert_eq!(paths.len(), 2);
/// ```
pub fn k_edge_disjoint_paths<G, F>(
    graph: G,
    start: G::NodeId,
    goal: G::NodeId,
    k: usize,
    mut edge_cost: F,
) -> Option<(u64, Vec<Vec<G::EdgeId>>)>
where
    G: IntoEdgesDirected + NodeIndexable + Copy,
    F: FnMut(G::EdgeRef) -> u64,
{
    let node_bound = graph.node_bound();
    let start = graph.to_index(start);
    let goal = graph.to_index(goal);

    if k == 0 {
        return Some((0, Vec::new()));
    }
    if start == goal {
        return Some((0, vec![Vec::new(); k]));
    }

    // Build the residual network: one unit-capacity arc per original edge,
    // plus a reverse arc with negated cost for flow cancellation.
    let mut arcs: Vec<Arc> = Vec::new();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); node_bound];
    let mut edge_ids: Vec<G::EdgeId> = Vec::new();
    let mut edge_costs: Vec<u64> = Vec::new();
    let mut edge_endpoints: Vec<(usize, usize)> = Vec::new();

    for node in 0..node_bound {
        for edge in graph.edges(graph.from_index(node)) {
            let source = graph.to_index(edge.source());
            let target = graph.to_index(edge.target());
            let cost = edge_cost(edge);
            let forward = arcs.len();
            let backward = forward + 1;
            arcs.push(Arc {
                to: target,
                capacity: 1,
                cost: i128::from(cost),
                reverse: backward,
                edge: Some(edge_ids.len()),
            });
            arcs.push(Arc {
                to: source,
                capacity: 0,
                cost: -(i128::from(cost)),
                reverse: forward,
                edge: None,
            });
            adjacency[source].push(forward);
            adjacency[target].push(backward);
            edge_ids.push(edge.id());
            edge_costs.push(cost);
            edge_endpoints.push((source, target));
        }
    }

    // Successive shortest augmenting paths with Johnson potentials.
    let mut potentials = vec![0i128; node_bound];
    let mut found = 0;
    while found < k {
        let mut distances = vec![i128::MAX; node_bound];
        let mut parent: Vec<Option<usize>> = vec![None; node_bound];
        let mut visit_next = BinaryHeap::new();
        distances[start] = 0;
        visit_next.push(MinScored(0i128, start));

        while let Some(MinScored(node_distance, node)) = visit_next.pop() {
            if node_distance > distances[node] {
                continue;
            }
            for &arc_index in &adjacency[node] {
                let arc = arcs[arc_index];
                if arc.capacity == 0 {
                    continue;
                }
                // Reduced costs are kept non-negative by the potentials.
                let reduced = arc.cost + potentials[node] - potentials[arc.to];
                let next_distance = node_distance.saturating_add(reduced);
                if next_distance < distances[arc.to] {
                    distances[arc.to] = next_distance;
                    parent[arc.to] = Some(arc_index);
                    visit_next.push(MinScored(next_distance, arc.to));
                }
            }
        }

        if distances[goal] == i128::MAX {
            // Fewer than `k` edge-disjoint paths exist.
            return None;
        }

        for node in 0..node_bound {
            if distances[node] != i128::MAX {
                potentials[node] = potentials[node].saturating_add(distances[node]);
            }
        }

        let mut node = goal;
        while node != start {
            let arc_index = parent[node]?;
            arcs[arc_index].capacity -= 1;
            let reverse = arcs[arc_index].reverse;
            arcs[reverse].capacity += 1;
            node = arcs[reverse].to;
        }
        found += 1;
    }

    // Decompose the flow into `k` simple paths, cancelling any cycles so
    // that no path visits a node twice.
    let mut used: Vec<Vec<usize>> = vec![Vec::new(); node_bound];
    let mut total_cost = 0u64;
    for arc in &arcs {
        if let Some(edge) = arc.edge {
            if arc.capacity == 0 {
                used[edge_endpoints[edge].0].push(edge);
                total_cost = total_cost.checked_add(edge_costs[edge])?;
            }
        }
    }

    let mut paths = Vec::with_capacity(k);
    for _ in 0..k {
        let mut path = Vec::new();
        let mut path_nodes = vec![start];
        let mut position = vec![usize::MAX; node_bound];
        position[start] = 0;
        let mut node = start;
        while node != goal {
            let edge = used[node].pop()?;
            let target = edge_endpoints[edge].1;
            if position[target] != usize::MAX {
                // Found a cycle: cancel it by dropping its edges and nodes.
                let cycle_start = position[target];
                for &removed in &path_nodes[cycle_start + 1..] {
                    position[removed] = usize::MAX;
                }
                path.truncate(cycle_start);
                path_nodes.truncate(cycle_start + 1);
                node = target;
            } else {
                position[target] = path_nodes.len();
                path.push(edge_ids[edge]);
                path_nodes.push(target);
                node = target;
            }
        }
        paths.push(path);
    }

    Some((total_cost, paths))
}
