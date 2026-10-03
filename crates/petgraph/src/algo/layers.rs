//! Iterative layer-by-layer partitioning of a directed graph (Kahn-style
//! topological layers).
//!
//! This is *not* a shortest-path or scheduling computation: it peels the
//! graph into breadth "batches", where every node in a layer has all of its
//! predecessors in earlier layers.

use alloc::vec::Vec;

use crate::{
    Direction,
    visit::{EdgeRef, GraphBase, IntoEdgesDirected, IntoNodeIdentifiers, NodeIndexable},
};

/// Iterative layer-by-layer partitioning of a directed graph.
///
/// `Layers` is an iterator that yields the nodes of a directed graph in
/// topological *layers* (batches): the first batch contains every node
/// without predecessors, the second batch contains every node whose
/// predecessors all appeared in the first batch, and so on. In general, a
/// node appears in a layer only once **all** of its predecessors have
/// appeared in earlier layers. Within one layer the nodes are ordered by
/// their node index.
///
/// The layers are computed lazily, one batch per [`Iterator::next`] call;
/// dropping the iterator early does not perform any remaining work, and the
/// input graph is only borrowed, never modified.
///
/// Nodes that cannot be assigned to any layer because they sit on — or
/// downstream of — a cycle are **not** silently dropped: after the last
/// acyclic layer they are returned together as one final, non-empty batch
/// (the cycle closure), so every node of the graph is yielded exactly once.
/// Use [`is_cyclic_directed`](crate::algo::is_cyclic_directed) to tell such
/// a tail batch apart from a regular layer.
///
/// Parallel (duplicate) edges are handled correctly: each edge decrements
/// the in-degree of its target exactly once, so they never change the
/// resulting layering. Vacant slots of a [`StableGraph`](crate::stable_graph::StableGraph)
/// are skipped.
///
/// # Complexity
/// * Time complexity: **O(|V| + |E|)** overall (amortized over the whole
///   iteration); the constructor only counts in-degrees in **O(|V| + |E|)**.
/// * Auxiliary space: **O(|V|)**.
///
/// where **|V|** is the number of nodes and **|E|** the number of edges.
///
/// # Example
/// ```rust
/// use petgraph::{Graph, algo::layers};
///
/// // a, b both point to c, which points to d.
/// let mut g = Graph::<&str, ()>::new();
/// let a = g.add_node("a");
/// let b = g.add_node("b");
/// let c = g.add_node("c");
/// let d = g.add_node("d");
/// g.extend_with_edges([(a, c), (b, c), (c, d)]);
///
/// // Print the batches: layer 0: [a, b] / layer 1: [c] / layer 2: [d]
/// for (i, layer) in layers(&g).enumerate() {
///     let names: Vec<_> = layer.iter().map(|&n| g[n]).collect();
///     println!("layer {i}: {names:?}");
/// }
///
/// let batches: Vec<Vec<_>> = layers(&g).collect();
/// assert_eq!(batches, vec![vec![a, b], vec![c], vec![d]]);
/// ```
#[derive(Clone, Debug)]
pub struct Layers<G>
where
    G: GraphBase,
{
    graph: G,
    /// Remaining in-degree per node index; a node joins the frontier when
    /// its counter reaches zero.
    in_degree: Vec<usize>,
    /// Nodes whose predecessors are all emitted; yielded by the next
    /// `next()` call. Kept sorted by node index.
    frontier: Vec<G::NodeId>,
    /// Number of nodes not yet yielded.
    remaining: usize,
    /// Whether the cycle-closure tail batch has already been yielded.
    cycle_emitted: bool,
}

/// Create a new [`Layers`] iterator over the topological layers of `graph`.
///
/// See the [`Layers`] documentation for the exact layering semantics.
pub fn layers<G>(graph: G) -> Layers<G>
where
    G: IntoEdgesDirected + IntoNodeIdentifiers + NodeIndexable,
{
    let mut in_degree = alloc::vec![0_usize; graph.node_bound()];
    // Count each edge exactly once, so parallel edges cannot over-decrement
    // the in-degree later: every increment here is matched by exactly one
    // decrement when the source node is emitted.
    for edge in graph.edge_references() {
        in_degree[graph.to_index(edge.target())] += 1;
    }

    let mut remaining = 0;
    let mut frontier: Vec<G::NodeId> = graph
        .node_identifiers()
        .inspect(|_| remaining += 1)
        .filter(|&n| in_degree[graph.to_index(n)] == 0)
        .collect();
    frontier.sort_by_key(|&n| graph.to_index(n));

    Layers {
        graph,
        in_degree,
        frontier,
        remaining,
        cycle_emitted: false,
    }
}

impl<G> Iterator for Layers<G>
where
    G: IntoEdgesDirected + IntoNodeIdentifiers + NodeIndexable,
{
    /// One layer of nodes; the final item is the cycle-closure batch if the
    /// graph contains a cycle.
    type Item = Vec<G::NodeId>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.frontier.is_empty() {
            // No acyclic layer remains. If unemitted nodes are left, they
            // form the cycle closure: yield them as one real, non-empty
            // final batch instead of an empty tail.
            if self.remaining > 0 && !self.cycle_emitted {
                self.cycle_emitted = true;
                self.remaining = 0;
                let mut closure: Vec<G::NodeId> = self
                    .graph
                    .node_identifiers()
                    .filter(|&n| self.in_degree[self.graph.to_index(n)] > 0)
                    .collect();
                closure.sort_by_key(|&n| self.graph.to_index(n));
                return Some(closure);
            }
            return None;
        }

        let layer = core::mem::take(&mut self.frontier);
        self.remaining -= layer.len();

        let mut next_frontier: Vec<G::NodeId> = Vec::new();
        for &node in &layer {
            for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                let target = edge.target();
                let slot = &mut self.in_degree[self.graph.to_index(target)];
                *slot -= 1;
                if *slot == 0 {
                    next_frontier.push(target);
                }
            }
        }
        next_frontier.sort_by_key(|&n| self.graph.to_index(n));
        self.frontier = next_frontier;

        Some(layer)
    }
}
