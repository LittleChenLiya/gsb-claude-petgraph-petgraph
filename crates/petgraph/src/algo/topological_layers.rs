//! Iterative partitioning of a directed acyclic graph into topological layers.
//!
//! This module provides [`topological_layers`], a lazy iterator that yields one
//! *layer* (batch) of nodes at a time. A layer contains exactly the nodes whose
//! predecessors have all already been emitted in earlier layers; nodes within a
//! layer are listed in their original node-index order.
//!
//! The partitioning is purely structural (it is neither a shortest-path nor a
//! scheduling computation). When the graph contains a directed cycle, the
//! iterator stops and reports a *witness*: the actual closed loop of nodes that
//! prevents further layering, rather than an empty tail batch.

use alloc::{vec, vec::Vec};

use fixedbitset::FixedBitSet;

use crate::{
    prelude::*,
    visit::{EdgeRef, IntoEdgesDirected, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};

/// An error returned by [`TopologicalLayers`] when the graph contains a directed
/// cycle.
///
/// The tuple field holds the nodes of the cycle in order: there is an edge from
/// each node to the next, and an edge from the last node back to the first. This
/// is a *witness* of the obstruction, not an empty tail batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopologicalCycle<N>(pub Vec<N>);

impl<N> TopologicalCycle<N> {
    /// The nodes that make up the cycle, in order.
    pub fn cycle(&self) -> &[N] {
        &self.0
    }

    /// Consume the error and return the cycle's nodes.
    pub fn into_cycle(self) -> Vec<N> {
        self.0
    }
}

/// A lazy iterator over the topological layers of a directed graph.
///
/// Created by [`topological_layers`]. Each call to [`next`](Iterator::next)
/// yields one layer: a `Vec` of nodes all of whose predecessors were emitted in
/// earlier layers, ordered by original node index.
///
/// If the graph is acyclic, the iterator ends with `None` once every node has
/// been emitted. If the graph contains a directed cycle, the iterator yields a
/// single `Err` holding a witness of the cycle and then ends.
///
/// The iterator owns no background work: dropping it early simply frees its
/// small internal workspace.
pub struct TopologicalLayers<G>
where
    G: NodeIndexable + IntoEdgesDirected + IntoNodeIdentifiers + NodeCount,
{
    g: G,
    /// Remaining number of distinct predecessors for each node index.
    indegree: Vec<usize>,
    /// Number of valid nodes not yet emitted in a layer.
    remaining: usize,
    /// The layer to emit on the next call.
    current: Vec<G::NodeId>,
    /// Workspace for the layer after `current`.
    next: Vec<G::NodeId>,
    /// Scratch bitset used to deduplicate parallel edges and to track the
    /// on-path set while extracting a cycle witness.
    seen: FixedBitSet,
    /// Set once the iterator has finished (successfully or with a cycle).
    done: bool,
}

/// Partition a directed graph into topological layers.
///
/// Returns a lazy iterator that yields one layer at a time. A layer is a
/// maximal set of nodes (in original node-index order) whose predecessors have
/// all already been emitted in previous layers. The first layer consists of the
/// nodes with no predecessors.
///
/// Parallel edges between the same pair of nodes count as a single predecessor
/// relationship, so duplicated edges cannot inflate or deflate the in-degree.
/// Holes in [`StableGraph`](crate::stable_graph) node slots are skipped. The
/// input graph is only borrowed, never modified.
///
/// If the graph contains a directed cycle, the iterator yields a single `Err`
/// holding a witness of the actual closed loop (see [`TopologicalCycle`]) and
/// then ends; it never yields an empty tail batch.
///
/// # Arguments
/// * `g`: a directed graph.
///
/// # Returns
/// An iterator over `Result<Vec<NodeId>, TopologicalCycle<NodeId>>`.
///
/// # Complexity
/// * Time complexity: **O(|V| + |E|)**.
/// * Auxiliary space: **O(|V|)**.
///
/// # Examples
/// ```
/// use petgraph::{Graph, algo::topological_layers, prelude::*};
///
/// // a and b both point to c, c points to d.
/// let mut graph: Graph<(), (), Directed> = Graph::new();
/// let a = graph.add_node(());
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// let d = graph.add_node(());
/// graph.extend_with_edges([(a, c), (b, c), (c, d)]);
///
/// let layers: Vec<_> = topological_layers(&graph).collect();
/// assert_eq!(layers.len(), 3);
/// assert_eq!(layers[0].as_ref().unwrap(), &[a, b]);
/// assert_eq!(layers[1].as_ref().unwrap(), &[c]);
/// assert_eq!(layers[2].as_ref().unwrap(), &[d]);
/// ```
///
/// Parallel edges do not change the layering:
///
/// ```
/// use petgraph::{Graph, algo::topological_layers, prelude::*};
///
/// let mut graph: Graph<(), (), Directed> = Graph::new();
/// let a = graph.add_node(());
/// let c = graph.add_node(());
/// // Two parallel edges a -> c.
/// graph.extend_with_edges([(a, c), (a, c)]);
///
/// let layers: Vec<_> = topological_layers(&graph).collect();
/// assert_eq!(layers.len(), 2);
/// assert_eq!(layers[0].as_ref().unwrap(), &[a]);
/// assert_eq!(layers[1].as_ref().unwrap(), &[c]);
/// ```
pub fn topological_layers<G>(g: G) -> TopologicalLayers<G>
where
    G: NodeIndexable + IntoEdgesDirected + IntoNodeIdentifiers + NodeCount,
{
    let bound = g.node_bound();
    let mut indegree = vec![0usize; bound];
    let mut seen = FixedBitSet::with_capacity(bound);

    // Count distinct predecessors for every valid node. Parallel edges from
    // the same source are counted only once, so duplicated edges cannot inflate
    // the in-degree.
    for v in g.node_identifiers() {
        let vi = g.to_index(v);
        seen.clear();
        for edge in g.edges_directed(v, Incoming) {
            let source = edge.source();
            // `put` returns the previous value: `false` on the first visit,
            // `true` once the source has already been counted.
            if !seen.put(g.to_index(source)) {
                indegree[vi] += 1;
            }
        }
    }

    // The first layer is the set of nodes with no predecessors, in original
    // node-index order.
    let mut current: Vec<_> = g
        .node_identifiers()
        .filter(|&v| indegree[g.to_index(v)] == 0)
        .collect();
    current.sort_by_key(|&v| g.to_index(v));

    let remaining = g.node_count();

    TopologicalLayers {
        g,
        indegree,
        remaining,
        current,
        next: Vec::new(),
        seen,
        done: false,
    }
}

impl<G> TopologicalLayers<G>
where
    G: NodeIndexable + IntoEdgesDirected + IntoNodeIdentifiers + NodeCount,
{
    /// Extract a witness of a directed cycle in the remaining subgraph.
    ///
    /// Every node left unprocessed has at least one unprocessed predecessor
    /// (otherwise its in-degree would have reached zero and it would have been
    /// emitted), so walking backwards through predecessors must eventually
    /// repeat a node. The segment between the two occurrences is a closed loop;
    /// we return it in forward order.
    fn find_cycle(&mut self) -> Vec<G::NodeId> {
        let start = self
            .g
            .node_identifiers()
            .find(|&v| self.indegree[self.g.to_index(v)] > 0)
            .expect("remaining set is non-empty but no node has positive in-degree");

        let mut path: Vec<G::NodeId> = Vec::new();
        self.seen.clear();
        let mut cur = start;

        loop {
            let ci = self.g.to_index(cur);
            // `put` returns the previous value: `true` means `cur` was already
            // on the path, so we have found a closed loop.
            if self.seen.put(ci) {
                // `cur` is already on the path: the nodes from its first
                // occurrence to here form a closed loop.
                let i = path
                    .iter()
                    .position(|&v| self.g.to_index(v) == ci)
                    .expect("repeated node is on the path");
                let mut cycle = Vec::with_capacity(path.len() - i);
                cycle.push(cur);
                for j in (i + 1..path.len()).rev() {
                    cycle.push(path[j]);
                }
                return cycle;
            }
            path.push(cur);

            // Move to a predecessor that is still unprocessed (positive
            // in-degree). Such a predecessor always exists for a remaining
            // node.
            let mut predecessor = None;
            for edge in self.g.edges_directed(cur, Incoming) {
                let source = edge.source();
                if self.indegree[self.g.to_index(source)] > 0 {
                    predecessor = Some(source);
                    break;
                }
            }
            cur = predecessor.expect("remaining node has no remaining predecessor");
        }
    }
}

impl<G> Iterator for TopologicalLayers<G>
where
    G: NodeIndexable + IntoEdgesDirected + IntoNodeIdentifiers + NodeCount,
{
    type Item = Result<Vec<G::NodeId>, TopologicalCycle<G::NodeId>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.current.is_empty() {
            if self.remaining == 0 {
                self.done = true;
                return None;
            }
            // Stuck with unprocessed nodes: a cycle is preventing further
            // layering. Report the actual closed loop as a witness.
            let cycle = self.find_cycle();
            self.done = true;
            return Some(Err(TopologicalCycle(cycle)));
        }

        // Process the current layer, producing the next one.
        let layer = core::mem::take(&mut self.current);
        self.next.clear();
        for &u in &layer {
            self.remaining -= 1;
            self.seen.clear();
            for edge in self.g.edges(u) {
                let target = edge.target();
                let ti = self.g.to_index(target);
                // Decrement once per distinct successor (`put` returns the
                // previous value, so `!put` is true on the first visit), so
                // parallel edges cannot over-deduct the in-degree.
                if !self.seen.put(ti) {
                    self.indegree[ti] -= 1;
                    if self.indegree[ti] == 0 {
                        self.next.push(target);
                    }
                }
            }
        }

        // Emit the next layer in original node-index order.
        self.next.sort_by_key(|&v| self.g.to_index(v));
        core::mem::swap(&mut self.current, &mut self.next);

        Some(Ok(layer))
    }
}

impl<G> core::iter::FusedIterator for TopologicalLayers<G> where
    G: NodeIndexable + IntoEdgesDirected + IntoNodeIdentifiers + NodeCount
{
}
