//! Dependency closure planning with disjunctive satisfaction groups.
//!
//! Given a set of target nodes and, for each node, a list of *satisfaction
//! groups* (each group is a set of alternative predecessor nodes), the planner
//! chooses exactly one predecessor from each group so that every target is
//! satisfied, the total number of distinct required nodes is minimized, and
//! the chosen dependency graph is acyclic.
//!
//! Shared dependencies are counted only once. Cycles are rejected as
//! non-executable. The search is bounded by an explicit node budget; if the
//! budget is exhausted before a solution is found, [`PlannerResult::Unknown`]
//! is returned (distinct from [`PlannerResult::NoSolution`], which means the
//! planner proved no solution exists).
//!
//! The input graph is not mutated.

use alloc::{vec, vec::Vec};
use core::hash::Hash;
use hashbrown::{HashMap, HashSet};

/// Result of planning a dependency closure.
///
/// This enum distinguishes three outcomes that are important for a bounded
/// search: a valid plan, a proven impossibility, and an unknown result (the
/// budget was exhausted before a solution could be established).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannerResult<N> {
    /// A valid plan was found.
    Solution(Plan<N>),
    /// No valid plan exists: every branch is cyclic or unsatisfiable.
    NoSolution,
    /// The search budget was exhausted before a solution could be found.
    ///
    /// This is distinct from [`NoSolution`]: the planner does not claim that
    /// no solution exists, only that it could not find one within the budget.
    /// A larger budget may yield a solution.
    Unknown,
}

/// A complete dependency closure plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan<N> {
    /// The minimum required set of nodes (includes the targets).
    pub required: Vec<N>,
    /// The chosen predecessor for each (node, group).
    pub choices: Vec<Choice<N>>,
    /// A topological ordering of [`required`](Self::required) (dependencies
    /// first).
    pub build_order: Vec<N>,
}

/// A single satisfaction choice in the plan.
///
/// This records the *selection basis* for one group: which predecessor was
/// chosen, what alternatives were available, and which nodes were introduced
/// by this choice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice<N> {
    /// The node being satisfied.
    pub node: N,
    /// The group index (0-based) within `node`'s satisfaction groups.
    pub group_index: usize,
    /// The predecessor chosen from the group.
    pub chosen: N,
    /// The alternatives that were available in the group.
    pub alternatives: Vec<N>,
    /// The nodes in the required set introduced by this choice.
    ///
    /// A node is "introduced" by a choice if it was first added to the
    /// required set as a direct result of that choice. Targets are
    /// self-introduced and are not listed here.
    pub added_nodes: Vec<N>,
}

/// Internal search state.
struct State<N> {
    /// The set of nodes selected so far (targets + chosen predecessors).
    required: HashSet<N>,
    /// For each node, the chosen predecessor per group (in group order).
    choices: HashMap<N, Vec<N>>,
    /// Maps each required node to the choice that introduced it.
    /// `None` means the node is a target (self-introduced).
    introduced_by: HashMap<N, Option<(N, usize)>>,
}

impl<N> State<N>
where
    N: Copy + Eq + Hash,
{
    fn new(targets: &[N]) -> Self {
        let mut required = HashSet::with_capacity(targets.len());
        let mut introduced_by = HashMap::with_capacity(targets.len());
        for &t in targets {
            required.insert(t);
            introduced_by.insert(t, None);
        }
        State {
            required,
            choices: HashMap::new(),
            introduced_by,
        }
    }

    fn clone(&self) -> Self {
        State {
            required: self.required.clone(),
            choices: self.choices.clone(),
            introduced_by: self.introduced_by.clone(),
        }
    }
}

/// Outcome of a single search step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Solution,
    NoSolution,
    Unknown,
}

/// Plans a minimum-size dependency closure.
///
/// Given `targets` and a map of satisfaction `groups` (each group is a list of
/// alternative predecessor nodes), the planner chooses exactly one predecessor
/// from each group so that every target is satisfied, the total number of
/// distinct required nodes is minimized, and the chosen dependency graph is
/// acyclic.
///
/// The input graph is not mutated (the planner does not take a graph
/// parameter; the satisfaction groups fully define the dependency structure).
///
/// # Arguments
/// * `targets`: the nodes that must be satisfied.
/// * `groups`: for each node, its satisfaction groups. Each group is a list of
///   alternative predecessor nodes; exactly one is chosen per group. Nodes
///   absent from the map are treated as having no groups (trivially satisfied).
/// * `budget`: an explicit limit on the number of search nodes. If exceeded,
///   [`PlannerResult::Unknown`] is returned.
///
/// # Returns
/// * [`PlannerResult::Solution`] with the minimum plan, or
/// * [`PlannerResult::NoSolution`] if every branch is cyclic/unsatisfiable, or
/// * [`PlannerResult::Unknown`] if the budget was exhausted.
///
/// # Complexity
/// The search is exponential in the number of distinct choices, bounded by
/// `budget`. For small graphs this is exact; for larger graphs the budget
/// provides an explicit cutoff.
pub fn plan_dependency_closure<N>(
    targets: &[N],
    groups: &HashMap<N, Vec<Vec<N>>>,
    budget: usize,
) -> PlannerResult<N>
where
    N: Copy + Eq + Hash + Ord,
{
    if targets.is_empty() {
        return PlannerResult::NoSolution;
    }

    let mut budget_remaining = budget;
    let mut best: Option<State<N>> = None;

    let state = State::new(targets);
    match search(state, groups, &mut budget_remaining, &mut best) {
        Outcome::Solution => {}
        Outcome::NoSolution => return PlannerResult::NoSolution,
        Outcome::Unknown => return PlannerResult::Unknown,
    }

    match best {
        Some(state) => PlannerResult::Solution(build_plan(&state, groups)),
        None => PlannerResult::NoSolution,
    }
}

/// Recursive branch-and-bound search.
fn search<N>(
    state: State<N>,
    groups: &HashMap<N, Vec<Vec<N>>>,
    budget: &mut usize,
    best: &mut Option<State<N>>,
) -> Outcome
where
    N: Copy + Eq + Hash + Ord,
{
    if *budget == 0 {
        return Outcome::Unknown;
    }
    *budget -= 1;

    // Prune: if the current required set is already at least as large as the
    // best solution found, it cannot improve.
    if let Some(b) = best.as_ref() {
        if state.required.len() >= b.required.len() {
            return Outcome::NoSolution;
        }
    }

    // Find an unsatisfied required node (one with unprocessed groups).
    let x = match state.required.iter().copied().find(|&n| {
        let gi = state.choices.get(&n).map(|c| c.len()).unwrap_or(0);
        gi < groups.get(&n).map(|g| g.len()).unwrap_or(0)
    }) {
        Some(x) => x,
        None => {
            // All required nodes are satisfied. Update the best solution.
            if best
                .as_ref()
                .map_or(true, |b| state.required.len() < b.required.len())
            {
                *best = Some(state);
            }
            return Outcome::Solution;
        }
    };

    let gi = state.choices.get(&x).map(|c| c.len()).unwrap_or(0);
    let gx = groups.get(&x).map(|g| g.as_slice()).unwrap_or(&[]);

    let mut result = Outcome::NoSolution;
    for y in gx[gi].iter().copied() {
        // Cycle check: if `y` reaches `x` in the chosen dependency graph,
        // adding the edge `x -> y` would create a cycle `x -> y -> ... -> x`.
        if reaches(&state, y, x) {
            continue;
        }

        let mut s = state.clone();
        if !s.required.contains(&y) {
            s.required.insert(y);
            s.introduced_by.insert(y, Some((x, gi)));
        }
        s.choices.entry(x).or_insert_with(Vec::new).push(y);

        match search(s, groups, budget, best) {
            Outcome::Solution => result = Outcome::Solution,
            Outcome::Unknown => return Outcome::Unknown,
            Outcome::NoSolution => {}
        }
    }

    result
}

/// Returns `true` if `target` is reachable from `from` in the chosen
/// dependency graph (edges `n -> pred` mean `n` depends on `pred`).
fn reaches<N>(state: &State<N>, from: N, target: N) -> bool
where
    N: Copy + Eq + Hash,
{
    if from == target {
        return true;
    }
    let mut visited: HashSet<N> = HashSet::new();
    let mut stack = vec![from];
    while let Some(n) = stack.pop() {
        if n == target {
            return true;
        }
        if visited.insert(n) {
            if let Some(preds) = state.choices.get(&n) {
                for &p in preds {
                    stack.push(p);
                }
            }
        }
    }
    false
}

/// Builds the final plan from the best search state.
fn build_plan<N>(state: &State<N>, groups: &HashMap<N, Vec<Vec<N>>>) -> Plan<N>
where
    N: Copy + Eq + Hash + Ord,
{
    let required: Vec<N> = state.required.iter().copied().collect();

    // Build choices with selection rationale.
    let mut choices: Vec<Choice<N>> = Vec::new();
    for (&x, preds) in &state.choices {
        let gx = groups.get(&x).map(|g| g.as_slice()).unwrap_or(&[]);
        for (gi, &y) in preds.iter().enumerate() {
            let alternatives = if gi < gx.len() {
                gx[gi].clone()
            } else {
                Vec::new()
            };
            let added_nodes: Vec<N> = state
                .introduced_by
                .iter()
                .filter_map(|(&n, &intro)| {
                    if intro == Some((x, gi)) {
                        Some(n)
                    } else {
                        None
                    }
                })
                .collect();
            choices.push(Choice {
                node: x,
                group_index: gi,
                chosen: y,
                alternatives,
                added_nodes,
            });
        }
    }
    choices.sort_by_key(|c| (c.node, c.group_index));

    // Topological order (dependencies first).
    let build_order = topological_order(state);

    Plan {
        required,
        choices,
        build_order,
    }
}

/// Computes a topological ordering of the required nodes (dependencies first)
/// using Kahn's algorithm with deterministic tie-breaking.
fn topological_order<N>(state: &State<N>) -> Vec<N>
where
    N: Copy + Eq + Hash + Ord,
{
    // Build the prerequisite graph: edge `y -> x` means `y` is a prerequisite
    // of `x` (reverse of the dependency edge `x -> y`).
    let mut in_degree: HashMap<N, usize> = HashMap::new();
    let mut adj: HashMap<N, Vec<N>> = HashMap::new();

    for &n in &state.required {
        in_degree.entry(n).or_insert(0);
    }
    for (&x, preds) in &state.choices {
        for &y in preds {
            // x depends on y, so y is a prerequisite of x: edge y -> x.
            adj.entry(y).or_insert_with(Vec::new).push(x);
            *in_degree.entry(x).or_insert(0) += 1;
        }
    }

    let mut queue: Vec<N> = in_degree
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&n, _)| n)
        .collect();
    queue.sort();

    let mut order = Vec::with_capacity(state.required.len());
    while let Some(n) = queue.pop() {
        order.push(n);
        if let Some(dependents) = adj.get(&n) {
            for &d in dependents {
                let deg = in_degree.get_mut(&d).expect("node in in_degree");
                *deg -= 1;
                if *deg == 0 {
                    queue.push(d);
                }
            }
        }
        queue.sort();
    }

    order
}
