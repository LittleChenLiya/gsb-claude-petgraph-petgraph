/*!
Pausable enumeration of all simple cycles of a directed graph.

This module provides [SimpleCycleEnumerator], an explicit-state enumerator
that lists every simple directed cycle of a graph exactly once and can be
paused and resumed between batches.

The enumerator fixes a **snapshot** of the graph's structure at construction
time. Every batch additionally checks that the live graph still matches that
snapshot; resuming after the graph was structurally modified is rejected with
[CycleEnumeratorError::GraphChanged]. Resuming never restarts the scan from
the first cycle: the traversal state (an explicit DFS stack) is kept between
batches, so a batch only spends the work it is budgeted for.

Cycle normalization:

* Every simple cycle is emitted exactly once.
* Cycles are rotated so that the node with the smallest index comes first,
  i.e. the first edge of each emitted cycle leaves the minimum-index node of
  that cycle.
* Parallel edges produce distinct cycles, because cycles are emitted as
  sequences of *edge* ids.
* Self-loops are emitted as single-edge cycles.

# Complexity

Let **|V|** be the number of nodes, **|E|** the number of edges and **C** the
number of simple cycles of the graph.

* Time complexity: **O((|V| + |E|) * (C + 1))** total across all batches;
  a single batch with an edge-visit budget **B** and result limit **R** costs
  **O(B + R * |V|)**. Resuming a paused enumerator costs **O(1)** of extra
  bookkeeping and never rescans already visited edges.
* Auxiliary space: **O(|V| + |E|)** for the structural snapshot, plus
  **O(|V|)** for the traversal stack.

# Example

```
use petgraph::{
    algo::simple_cycles::{CycleBatchStatus, SimpleCycleEnumerator},
    prelude::*,
};

let mut graph = DiGraph::<(), ()>::new();
let a = graph.add_node(());
let b = graph.add_node(());
let c = graph.add_node(());
graph.extend_with_edges([(a, b), (b, c), (c, a), (a, b), (a, a)]);

let mut enumerator = SimpleCycleEnumerator::new(&graph);
let mut cursor = enumerator.cursor();
let mut cycles = Vec::new();
loop {
    // Emit at most one cycle per batch, with a generous edge-visit budget.
    let batch = enumerator.next_batch(&graph, &cursor, 1, usize::MAX).unwrap();
    cycles.extend(batch.cycles.iter().cloned());
    cursor = batch.cursor;
    if batch.status == CycleBatchStatus::Done {
        break;
    }
}
// Two triangle cycles (one per parallel a -> b edge) and one self-loop.
assert_eq!(cycles.len(), 3);
```
*/

use alloc::{vec, vec::Vec};
use core::fmt;

use fixedbitset::FixedBitSet;

use crate::visit::{EdgeRef, GraphRef, IntoEdgeReferences, NodeIndexable};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_push(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute a structural fingerprint of "graph".
///
/// The fingerprint covers the node index bound, the number of edges and the
/// ordered sequence of edge endpoints (by node index). Any structural
/// mutation (adding or removing nodes or edges) changes the fingerprint,
/// while changing node or edge *weights* does not.
fn fingerprint<G>(graph: G) -> u64
where
    G: IntoEdgeReferences + NodeIndexable,
{
    let mut hash = FNV_OFFSET;
    hash = fnv_push(hash, graph.node_bound() as u64);
    let mut edge_count = 0u64;
    for edge in graph.edge_references() {
        hash = fnv_push(hash, graph.to_index(edge.source()) as u64);
        hash = fnv_push(hash, graph.to_index(edge.target()) as u64);
        edge_count += 1;
    }
    fnv_push(hash, edge_count)
}

/// Status of a [CycleBatch]: whether the enumeration is paused or finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CycleBatchStatus {
    /// The batch stopped early because the result limit or the edge-visit
    /// budget was exhausted. More cycles may remain; resume with the returned
    /// [CycleCursor].
    Paused,
    /// The enumeration is complete; no more cycles exist.
    Done,
}

/// One batch of results produced by [SimpleCycleEnumerator::next_batch].
#[derive(Clone, Debug)]
pub struct CycleBatch<EId> {
    /// Cycles found in this batch. Each cycle is a sequence of edge ids,
    /// rotated so that the first edge leaves the minimum-index node of the
    /// cycle.
    pub cycles: Vec<Vec<EId>>,
    /// Whether the enumeration paused or finished.
    pub status: CycleBatchStatus,
    /// Cursor that must be passed to the next [SimpleCycleEnumerator::next_batch]
    /// call to resume the enumeration.
    pub cursor: CycleCursor,
    /// Number of edge visits performed by this batch. Summed over all
    /// batches, this equals the number of edge visits a single unpaused
    /// enumeration would perform: resuming never rescans edges.
    pub edge_visits: u64,
}

/// Opaque cursor identifying a resumable position of a [SimpleCycleEnumerator].
///
/// The cursor is bound to the graph version captured when the enumerator was
/// created. It is returned by [SimpleCycleEnumerator::cursor] and by every
/// [CycleBatch]; pass the latest cursor back to
/// [SimpleCycleEnumerator::next_batch] to continue the enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CycleCursor {
    version: u64,
    cycles_emitted: u64,
    batches: u64,
}

impl CycleCursor {
    /// Structural version of the graph snapshot this cursor is bound to.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Total number of cycles emitted before this cursor's position.
    pub fn cycles_emitted(&self) -> u64 {
        self.cycles_emitted
    }
}

/// Error returned by [SimpleCycleEnumerator::next_batch].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CycleEnumeratorError {
    /// The graph passed to [SimpleCycleEnumerator::next_batch] no longer
    /// matches the structural snapshot taken when the enumerator was created.
    GraphChanged,
    /// The cursor does not match the enumerator's current position (it is
    /// stale, or belongs to a different enumerator or graph version).
    StaleCursor,
}

impl fmt::Display for CycleEnumeratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CycleEnumeratorError::GraphChanged => {
                f.write_str("graph changed since the cycle enumerator was created")
            }
            CycleEnumeratorError::StaleCursor => {
                f.write_str("cursor does not match the enumerator's current position")
            }
        }
    }
}

impl core::error::Error for CycleEnumeratorError {}

/// One frame of the explicit DFS stack used by the enumerator.
#[derive(Clone, Copy, Debug)]
struct Frame<EId> {
    /// Node index this frame is positioned at.
    node: usize,
    /// Position of the next outgoing edge of "node" to examine.
    next_edge: usize,
    /// Edge through which this frame was entered ("None" for the start node).
    enter_edge: Option<EId>,
}

/// Pausable enumerator of all simple cycles of a directed graph.
///
/// See the [module documentation](self) for normalization rules, complexity
/// and an example.
///
/// The enumerator works on any graph reference implementing
/// [IntoEdgeReferences] + [NodeIndexable], including "&Graph",
/// "&StableGraph" (vacant node slots are skipped) and "&GraphMap".
///
/// The enumerator owns its snapshot and does not borrow the graph, so the
/// graph may be mutated between batches; the next batch then reports
/// [CycleEnumeratorError::GraphChanged].
pub struct SimpleCycleEnumerator<EId> {
    /// Structural fingerprint of the snapshot taken at construction.
    version: u64,
    /// Snapshot adjacency: for each node index, "(target, edge id)" pairs.
    adjacency: Vec<Vec<(usize, EId)>>,
    /// Nodes currently on the DFS path.
    on_path: FixedBitSet,
    /// Explicit DFS stack for the current start node.
    stack: Vec<Frame<EId>>,
    /// Index of the current start node (valid while "stack" is non-empty).
    start: usize,
    /// Next node index to try as a start node.
    next_start: usize,
    /// Whether the enumeration has finished.
    done: bool,
    /// Total number of cycles emitted so far.
    cycles_emitted: u64,
    /// Number of batches produced so far.
    batches: u64,
    /// Total number of edge visits performed so far.
    total_edge_visits: u64,
}

impl<EId: Copy> SimpleCycleEnumerator<EId> {
    /// Create a new enumerator over a structural snapshot of "graph".
    ///
    /// The snapshot fixes the graph version: later batches must be passed a
    /// graph with the same structure, otherwise they fail with
    /// [CycleEnumeratorError::GraphChanged].
    pub fn new<G>(graph: G) -> Self
    where
        G: IntoEdgeReferences + NodeIndexable + GraphRef<EdgeId = EId>,
    {
        let bound = graph.node_bound();
        let mut adjacency = vec![Vec::new(); bound];
        for edge in graph.edge_references() {
            let source = graph.to_index(edge.source());
            let target = graph.to_index(edge.target());
            adjacency[source].push((target, edge.id()));
        }
        SimpleCycleEnumerator {
            version: fingerprint(graph),
            adjacency,
            on_path: FixedBitSet::with_capacity(bound),
            stack: Vec::new(),
            start: 0,
            next_start: 0,
            done: false,
            cycles_emitted: 0,
            batches: 0,
            total_edge_visits: 0,
        }
    }

    /// Structural version of the graph snapshot this enumerator is bound to.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The cursor identifying the enumerator's current position.
    pub fn cursor(&self) -> CycleCursor {
        CycleCursor {
            version: self.version,
            cycles_emitted: self.cycles_emitted,
            batches: self.batches,
        }
    }

    /// Total number of cycles emitted so far.
    pub fn cycles_emitted(&self) -> u64 {
        self.cycles_emitted
    }

    /// Total number of edge visits performed so far, across all batches.
    pub fn total_edge_visits(&self) -> u64 {
        self.total_edge_visits
    }

    /// Whether the enumeration has finished.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Emit the next batch of simple cycles.
    ///
    /// # Arguments
    /// * "graph": the same graph the enumerator was created from. If its
    ///   structure changed since construction,
    ///   [CycleEnumeratorError::GraphChanged] is returned.
    /// * "cursor": the cursor from the previous batch (or
    ///   [SimpleCycleEnumerator::cursor] before the first batch). A cursor
    ///   that does not match the current position is rejected with
    ///   [CycleEnumeratorError::StaleCursor].
    /// * "max_results": maximum number of cycles to emit in this batch.
    /// * "edge_visit_budget": maximum number of edge visits to perform in
    ///   this batch.
    ///
    /// The returned batch has status [CycleBatchStatus::Paused] when it
    /// stopped because "max_results" or "edge_visit_budget" was exhausted,
    /// and [CycleBatchStatus::Done] when the enumeration is complete.
    /// Budget exhaustion is a pause, not the end of the enumeration: resume
    /// with the returned cursor to continue exactly where the batch stopped,
    /// without rescanning any edge.
    pub fn next_batch<G>(
        &mut self,
        graph: G,
        cursor: &CycleCursor,
        max_results: usize,
        edge_visit_budget: usize,
    ) -> Result<CycleBatch<EId>, CycleEnumeratorError>
    where
        G: IntoEdgeReferences + NodeIndexable + GraphRef<EdgeId = EId>,
    {
        if fingerprint(graph) != self.version {
            return Err(CycleEnumeratorError::GraphChanged);
        }
        if *cursor != self.cursor() {
            return Err(CycleEnumeratorError::StaleCursor);
        }

        let mut cycles = Vec::new();
        let mut edge_visits = 0u64;
        let status = self.run(&mut cycles, max_results, edge_visit_budget, &mut edge_visits);
        self.batches += 1;

        Ok(CycleBatch {
            cycles,
            status,
            cursor: self.cursor(),
            edge_visits,
        })
    }

    /// Drive the enumeration until it finishes, the result limit is reached
    /// or the edge-visit budget is exhausted.
    fn run(
        &mut self,
        cycles: &mut Vec<Vec<EId>>,
        max_results: usize,
        edge_visit_budget: usize,
        edge_visits: &mut u64,
    ) -> CycleBatchStatus {
        if self.done {
            return CycleBatchStatus::Done;
        }
        if max_results == 0 {
            return CycleBatchStatus::Paused;
        }
        loop {
            if self.stack.is_empty() {
                // Advance to the next start node. Cycles are enumerated from
                // their minimum-index node, so start nodes are tried in
                // increasing index order and each cycle is emitted once.
                if self.next_start >= self.adjacency.len() {
                    self.done = true;
                    return CycleBatchStatus::Done;
                }
                self.start = self.next_start;
                self.next_start += 1;
                self.on_path.put(self.start);
                self.stack.push(Frame {
                    node: self.start,
                    next_edge: 0,
                    enter_edge: None,
                });
            }

            let frame = self.stack.last_mut().unwrap();
            let node = frame.node;
            if frame.next_edge == self.adjacency[node].len() {
                // All outgoing edges of "node" examined: backtrack.
                self.on_path.set(node, false);
                self.stack.pop();
                continue;
            }
            if *edge_visits >= edge_visit_budget as u64 {
                // Budget exhausted: pause, do not finish the enumeration.
                return CycleBatchStatus::Paused;
            }

            let edge_pos = frame.next_edge;
            frame.next_edge += 1;
            let (target, edge_id) = self.adjacency[node][edge_pos];
            *edge_visits += 1;
            self.total_edge_visits += 1;

            if target == self.start {
                // Found a cycle back to the start node. The path edges are
                // stored in the stack frames, so the cycle is emitted
                // starting at the minimum-index node.
                let mut cycle = Vec::with_capacity(self.stack.len());
                for frame in &self.stack[1..] {
                    cycle.push(frame.enter_edge.unwrap());
                }
                cycle.push(edge_id);
                cycles.push(cycle);
                self.cycles_emitted += 1;
                if cycles.len() >= max_results {
                    return CycleBatchStatus::Paused;
                }
            } else if target > self.start && !self.on_path.contains(target) {
                // Only visit nodes with a larger index than the start node,
                // so that every cycle is found exactly once, from its
                // minimum-index node.
                self.on_path.put(target);
                self.stack.push(Frame {
                    node: target,
                    next_edge: 0,
                    enter_edge: Some(edge_id),
                });
            }
        }
    }
}
