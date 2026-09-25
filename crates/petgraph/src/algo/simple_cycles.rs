/*!
Pausable enumeration of all simple directed cycles, with support for parallel
edges and self loops.
*/

use alloc::{vec, vec::Vec};

use crate::{
    Direction::Outgoing,
    visit::{EdgeRef, IntoEdgeReferences, IntoEdgesDirected, IntoNodeIdentifiers, NodeIndexable},
};

/// Fingerprint of the graph structure at the time a [`SimpleCycleEnumerator`]
/// (or its [`CycleCursor`]) was created.
///
/// Used to reject resuming enumeration after the graph has changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GraphVersion {
    node_bound: usize,
    edge_count: usize,
    fingerprint: u64,
}

/// Order-independent mix of one edge's endpoint indices into the graph
/// fingerprint (FNV-1a over the index pair).
fn edge_mix(u: usize, v: usize) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in u.to_le_bytes().into_iter().chain(v.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn graph_version<G>(graph: G) -> GraphVersion
where
    G: IntoEdgeReferences + NodeIndexable,
{
    let mut edge_count = 0usize;
    let mut fingerprint = 0u64;
    for edge in graph.edge_references() {
        let u = graph.to_index(edge.source());
        let v = graph.to_index(edge.target());
        edge_count += 1;
        fingerprint = fingerprint.wrapping_add(edge_mix(u, v));
    }
    GraphVersion {
        node_bound: graph.node_bound(),
        edge_count,
        fingerprint,
    }
}

/// Error returned when a [`CycleCursor`] is resumed against a graph whose
/// structure differs from the one the cursor was created from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphChangedError;

impl core::fmt::Display for GraphChangedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("the graph changed since the cycle cursor was created")
    }
}

impl core::error::Error for GraphChangedError {}

/// Whether a [`SimpleCycleEnumerator`] paused (and can be resumed) or finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CycleStatus {
    /// The batch limits (result capacity or edge-visit budget) were reached
    /// before enumeration completed. More cycles may follow; resume with the
    /// returned [`CycleCursor`] or by calling
    /// [`SimpleCycleEnumerator::next_batch`] again.
    Paused,
    /// Enumeration is complete; no more cycles exist.
    Done,
}

/// Resumable position of a [`SimpleCycleEnumerator`].
///
/// A cursor is bound to the version of the graph it was created from:
/// [`SimpleCycleEnumerator::resume`] rejects it with [`GraphChangedError`] if
/// the graph has changed in the meantime. Resuming continues exactly where the
/// enumeration paused; it never rescans the already-enumerated prefix.
#[derive(Clone, Debug)]
pub struct CycleCursor<E> {
    version: GraphVersion,
    /// Number of self-loop cycles already emitted.
    self_loops_done: usize,
    /// Current start node (the minimum node index of the cycles being searched).
    start: usize,
    /// DFS stack of `(node, next edge position in the snapshot adjacency list)`.
    stack: Vec<(usize, usize)>,
    /// Edges of the current DFS path (one per stack frame except the start).
    path_edges: Vec<E>,
    done: bool,
}

impl<E> CycleCursor<E> {
    /// Returns `true` if the enumeration this cursor belongs to is complete.
    pub fn is_done(&self) -> bool {
        self.done
    }
}

/// One batch of results from [`SimpleCycleEnumerator::next_batch`].
#[derive(Clone, Debug)]
pub struct CycleBatch<E> {
    /// Simple cycles found in this batch. Each cycle is the sequence of its
    /// edge identifiers, rotated so that it starts at the smallest node index
    /// of the cycle. Parallel edges yield distinct cycles (different edge
    /// identifiers); each self loop is reported as its own one-edge cycle.
    pub cycles: Vec<Vec<E>>,
    /// `Paused` if enumeration can continue, `Done` if it is complete.
    pub status: CycleStatus,
    /// Number of edge visits performed while producing this batch.
    pub edge_visits: usize,
    /// Cursor that continues the enumeration where this batch stopped.
    pub cursor: CycleCursor<E>,
}

/// Pausable enumerator over all simple cycles of a directed graph snapshot.
///
/// The enumerator snapshots the graph's structure at creation
/// ([`SimpleCycleEnumerator::new`]) or at [`SimpleCycleEnumerator::resume`]
/// time and enumerates every simple directed cycle of that snapshot exactly
/// once:
///
/// * every cycle is rotated so that it starts at its smallest node index, so
///   each node-simple cycle is reported exactly once;
/// * parallel edges produce distinct cycles, because cycles are reported as
///   sequences of edge identifiers;
/// * every self loop is reported as its own one-edge cycle.
///
/// Enumeration proceeds in batches via [`next_batch`], which accepts a maximum
/// number of results and an edge-visit budget. Exhausting either limit yields
/// [`CycleStatus::Paused`] (never `Done`) together with a [`CycleCursor`] that
/// resumes exactly at the pause position, without rescanning the
/// already-enumerated prefix. The cursor is bound to the graph version;
/// resuming after the graph changed is rejected with [`GraphChangedError`].
///
/// [`next_batch`]: SimpleCycleEnumerator::next_batch
///
/// # Complexity
/// * Creating the snapshot costs **O(|V| + |E|)** time and space.
/// * Each edge visit costs **O(1)**; recording a cycle costs **O(|V|)**. The
///   total number of edge visits of a full enumeration is proportional to the
///   number of simple paths explored, which is output-sensitive but can be
///   exponential in **|V|** in the worst case (as for any exact simple-cycle
///   enumeration).
/// * A cursor occupies **O(|V|)** space; resuming costs **O(|V| + |E|)** to
///   rebuild the snapshot and never repeats earlier edge visits.
///
/// where **|V|** is the number of nodes and **|E|** the number of edges.
///
/// # Example
/// ```
/// use petgraph::{
///     algo::{CycleStatus, SimpleCycleEnumerator},
///     prelude::*,
/// };
///
/// let mut graph = DiGraph::<(), ()>::new();
/// let a = graph.add_node(());
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// graph.extend_with_edges([(a, b), (b, c), (c, a), (a, a)]);
///
/// let mut enumerator = SimpleCycleEnumerator::new(&graph);
/// let mut cycles = Vec::new();
/// loop {
///     let batch = enumerator.next_batch(1, 8);
///     cycles.extend(batch.cycles);
///     if batch.status == CycleStatus::Done {
///         break;
///     }
/// }
/// // One triangle cycle and one self-loop cycle.
/// assert_eq!(cycles.len(), 2);
/// ```
pub struct SimpleCycleEnumerator<E> {
    /// Snapshot adjacency: node index -> (target index, edge id); self loops excluded.
    adj: Vec<Vec<(usize, E)>>,
    /// Edge identifiers of self loops, each reported as its own cycle.
    self_loops: Vec<E>,
    cursor: CycleCursor<E>,
    total_edge_visits: usize,
}

impl<E: Copy> SimpleCycleEnumerator<E> {
    /// Creates an enumerator over a fixed snapshot of `graph`'s structure.
    pub fn new<G>(graph: G) -> Self
    where
        G: IntoEdgesDirected<EdgeId = E>
            + IntoEdgeReferences<EdgeId = E>
            + IntoNodeIdentifiers
            + NodeIndexable,
    {
        let version = graph_version(graph);
        let (adj, self_loops) = snapshot(graph);
        SimpleCycleEnumerator {
            adj,
            self_loops,
            cursor: CycleCursor {
                version,
                self_loops_done: 0,
                start: 0,
                stack: Vec::new(),
                path_edges: Vec::new(),
                done: false,
            },
            total_edge_visits: 0,
        }
    }

    /// Resumes enumeration from `cursor`, re-snapshotting `graph`.
    ///
    /// Returns [`GraphChangedError`] if the graph's structure changed since
    /// the cursor was created. Resuming continues at the pause position; it
    /// does not rescan the already-enumerated prefix.
    pub fn resume<G>(cursor: CycleCursor<E>, graph: G) -> Result<Self, GraphChangedError>
    where
        G: IntoEdgesDirected<EdgeId = E>
            + IntoEdgeReferences<EdgeId = E>
            + IntoNodeIdentifiers
            + NodeIndexable,
    {
        let version = graph_version(graph);
        if version != cursor.version {
            return Err(GraphChangedError);
        }
        let (adj, self_loops) = snapshot(graph);
        Ok(SimpleCycleEnumerator {
            adj,
            self_loops,
            cursor,
            total_edge_visits: 0,
        })
    }

    /// Returns the current cursor, which can later be passed to
    /// [`SimpleCycleEnumerator::resume`].
    pub fn cursor(&self) -> CycleCursor<E> {
        self.cursor.clone()
    }

    /// Returns `true` once enumeration is complete.
    pub fn is_done(&self) -> bool {
        self.cursor.done
    }

    /// Total number of edge visits performed by this enumerator so far.
    pub fn total_edge_visits(&self) -> usize {
        self.total_edge_visits
    }

    /// Enumerates the next batch of at most `max_results` cycles, performing
    /// at most `edge_budget` edge visits.
    ///
    /// Returns [`CycleStatus::Paused`] when either limit stopped the
    /// enumeration before completion; [`CycleStatus::Done`] only when every
    /// simple cycle of the snapshot has been reported.
    pub fn next_batch(&mut self, max_results: usize, edge_budget: usize) -> CycleBatch<E> {
        let mut cycles = Vec::new();
        let mut budget = edge_budget;
        let mut visits = 0usize;

        macro_rules! finish {
            ($status:expr) => {{
                self.total_edge_visits += visits;
                return CycleBatch {
                    cycles,
                    status: $status,
                    edge_visits: visits,
                    cursor: self.cursor.clone(),
                };
            }};
        }

        if self.cursor.done {
            finish!(CycleStatus::Done);
        }

        // Report each self loop as its own one-edge cycle first.
        while self.cursor.self_loops_done < self.self_loops.len() {
            if cycles.len() >= max_results || budget == 0 {
                finish!(CycleStatus::Paused);
            }
            budget -= 1;
            visits += 1;
            let edge = self.self_loops[self.cursor.self_loops_done];
            self.cursor.self_loops_done += 1;
            cycles.push(vec![edge]);
        }

        // Depth-first search for cycles whose minimum node index is the start node.
        let bound = self.adj.len();
        let mut on_path = vec![false; bound];
        for &(node, _) in &self.cursor.stack {
            on_path[node] = true;
        }

        while self.cursor.start < bound {
            if self.cursor.stack.is_empty() {
                let start = self.cursor.start;
                self.cursor.stack.push((start, 0));
                on_path[start] = true;
            }
            while let Some(&(node, pos)) = self.cursor.stack.last() {
                if pos == self.adj[node].len() {
                    self.cursor.stack.pop();
                    on_path[node] = false;
                    if !self.cursor.stack.is_empty() {
                        self.cursor.path_edges.pop();
                    }
                    continue;
                }
                if cycles.len() >= max_results || budget == 0 {
                    finish!(CycleStatus::Paused);
                }
                budget -= 1;
                visits += 1;
                let (target, edge) = self.adj[node][pos];
                self.cursor.stack.last_mut().unwrap().1 = pos + 1;
                if target == self.cursor.start {
                    // Found a cycle: current path plus this closing edge.
                    let mut cycle = self.cursor.path_edges.clone();
                    cycle.push(edge);
                    cycles.push(cycle);
                } else if target > self.cursor.start && !on_path[target] {
                    on_path[target] = true;
                    self.cursor.stack.push((target, 0));
                    self.cursor.path_edges.push(edge);
                }
            }
            self.cursor.start += 1;
        }

        self.cursor.done = true;
        finish!(CycleStatus::Done);
    }
}

/// Snapshots the graph's structure into index-based adjacency lists, with
/// self loops split out.
fn snapshot<G>(graph: G) -> (Vec<Vec<(usize, G::EdgeId)>>, Vec<G::EdgeId>)
where
    G: IntoEdgesDirected + IntoNodeIdentifiers + NodeIndexable,
{
    let bound = graph.node_bound();
    let mut adj: Vec<Vec<(usize, G::EdgeId)>> = (0..bound).map(|_| Vec::new()).collect();
    let mut self_loops = Vec::new();
    for node in graph.node_identifiers() {
        let u = graph.to_index(node);
        for edge in graph.edges_directed(node, Outgoing) {
            let v = graph.to_index(edge.target());
            if v == u {
                self_loops.push(edge.id());
            } else {
                adj[u].push((v, edge.id()));
            }
        }
    }
    (adj, self_loops)
}
