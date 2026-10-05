//! # Dependency closure planner
//!
//! Given a set of **target nodes** and, for every node, a list of
//! **requirement groups** (an AND of ORs: every group must be satisfied by
//! picking exactly *one* available provider predecessor), compute the
//! **required node set with the fewest nodes** that closes over all targets.
//!
//! Design rules implemented here:
//!
//! * **Shared dependencies are counted once.** The planner searches over
//!   whole assignments instead of greedily picking the locally smallest
//!   branch per target, so two targets that can share a provider produce a
//!   strictly smaller joint plan than the sum of their independent plans.
//! * **Cycles are explicit, non-executable diagnostics.** A candidate
//!   provider that would close a dependency cycle is rejected for that
//!   branch and recorded as a [`Diagnostic`]; it never counts as a valid
//!   satisfaction.
//! * **Explicit search budget for small graphs.**
//!   [`Planner::plan_with_budget`] bounds the number of requirement
//!   expansions. Exhausting the budget yields [`PlanOutcome::Unknown`] —
//!   never a false "unsatisfiable".
//! * **Explainable output.** A successful [`Plan`] carries the selected node
//!   set, one [`Choice`] per satisfied group with a human-readable
//!   rationale, and a topological `execution_order` (providers before
//!   dependents).
//! * **The input problem is immutable.** Planning takes `&self`; the graph
//!   is never modified.
//!
//! ## Local cargo commands
//!
//! ```text
//! cargo test -p dep-closure-planner                 # unit + integration tests
//! cargo test -p dep-closure-planner --test planner  # acceptance tests only
//! cargo run  -p dep-closure-planner --example six_node
//! ```

use core::fmt;

/// Default expansion budget used by [`Planner::plan`].
pub const DEFAULT_BUDGET: u64 = 100_000;

/// Cap on collected diagnostics so pathological inputs stay cheap.
const DIAGNOSTIC_CAP: usize = 128;

/// Identifier of a node inside a [`Planner`].
pub type NodeId = usize;

/// One requirement group of a node: an OR over candidate providers.
///
/// All groups of a selected node must be satisfied (AND semantics), and each
/// group is satisfied by choosing exactly one provider from
/// [`providers`](RequirementGroup::providers).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequirementGroup {
    name: String,
    providers: Vec<NodeId>,
}

impl RequirementGroup {
    /// Human-readable group name (e.g. `"database"`).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Candidate providers, sorted and deduplicated.
    #[must_use]
    pub fn providers(&self) -> &[NodeId] {
        &self.providers
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NodeDef {
    name: String,
    groups: Vec<RequirementGroup>,
}

/// An immutable dependency-closure problem.
///
/// Build it once, then run [`plan`](Planner::plan) /
/// [`plan_with_budget`](Planner::plan_with_budget) as often as needed; the
/// problem graph is never mutated by planning.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Planner {
    nodes: Vec<NodeDef>,
}

impl Planner {
    /// Create an empty problem.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node and return its id.
    pub fn add_node(&mut self, name: impl Into<String>) -> NodeId {
        self.nodes.push(NodeDef {
            name: name.into(),
            groups: Vec::new(),
        });
        self.nodes.len() - 1
    }

    /// Add a requirement group to `node`: exactly one of `providers` must be
    /// selected to satisfy it. Returns the group index.
    ///
    /// Provider ids are sorted and deduplicated. A group with an empty
    /// provider list can never be satisfied.
    ///
    /// # Panics
    ///
    /// Panics if `node` or any provider id is out of range.
    pub fn add_group(
        &mut self,
        node: NodeId,
        name: impl Into<String>,
        providers: &[NodeId],
    ) -> usize {
        assert!(
            node < self.nodes.len(),
            "add_group: node id {node} out of range"
        );
        let mut providers = providers.to_vec();
        providers.sort_unstable();
        providers.dedup();
        for &provider in &providers {
            assert!(
                provider < self.nodes.len(),
                "add_group: provider id {provider} out of range"
            );
        }
        self.nodes[node].groups.push(RequirementGroup {
            name: name.into(),
            providers,
        });
        self.nodes[node].groups.len() - 1
    }

    /// Number of nodes in the problem.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Name of a node.
    #[must_use]
    pub fn node_name(&self, id: NodeId) -> &str {
        &self.nodes[id].name
    }

    /// Requirement groups of a node.
    #[must_use]
    pub fn groups(&self, id: NodeId) -> &[RequirementGroup] {
        &self.nodes[id].groups
    }

    /// Plan with the [`DEFAULT_BUDGET`].
    #[must_use]
    pub fn plan(&self, targets: &[NodeId]) -> PlanOutcome {
        self.plan_with_budget(targets, DEFAULT_BUDGET)
    }

    /// Plan with an explicit expansion `budget`.
    ///
    /// The budget counts requirement expansions (search steps). Exhausting
    /// it yields [`PlanOutcome::Unknown`] — never a false `Unsatisfiable`.
    ///
    /// # Panics
    ///
    /// Panics if any target id is out of range.
    #[must_use]
    pub fn plan_with_budget(&self, targets: &[NodeId], budget: u64) -> PlanOutcome {
        let mut unique = targets.to_vec();
        unique.sort_unstable();
        unique.dedup();
        for &target in &unique {
            assert!(
                target < self.nodes.len(),
                "plan: target id {target} out of range"
            );
        }

        let mut search = Search::new(self, &unique, budget);
        search.run();
        search.finish()
    }
}

/// A recorded reason why a candidate branch was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// Machine-readable kind.
    pub kind: DiagnosticKind,
    /// Human-readable explanation.
    pub message: String,
}

/// Machine-readable diagnostic kinds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiagnosticKind {
    /// Choosing this provider would close a dependency cycle; the branch is
    /// non-executable. `loop_path` lists the cycle as node ids, first ==
    /// last.
    Cycle { loop_path: Vec<NodeId> },
    /// A requirement group has no candidate providers at all.
    EmptyGroup { node: NodeId, group: usize },
}

/// One satisfied requirement group inside a [`Plan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    /// Node whose group is satisfied.
    pub node: NodeId,
    /// Index of the satisfied group within the node.
    pub group: usize,
    /// Name of the satisfied group.
    pub group_name: String,
    /// Chosen provider.
    pub provider: NodeId,
    /// `true` when the provider was already part of the plan (shared
    /// dependency, adds zero nodes).
    pub reused: bool,
    /// Human-readable selection rationale.
    pub rationale: String,
}

/// A minimal executable plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Selected node set, sorted by id. Shared dependencies appear once.
    pub nodes: Vec<NodeId>,
    /// One entry per satisfied requirement group, with rationale.
    pub choices: Vec<Choice>,
    /// Topological execution order: every provider appears before the nodes
    /// that depend on it.
    pub execution_order: Vec<NodeId>,
    /// Diagnostics for branches that were explored and rejected (e.g. cyclic
    /// alternatives that lost to the chosen provider).
    pub diagnostics: Vec<Diagnostic>,
}

impl Plan {
    /// Number of nodes in the required set.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.nodes.len()
    }
}

/// Why a plan could not be decided within the given budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownReason {
    /// The explicit search budget was exhausted before the search space was
    /// fully explored. This is *not* a proof of unsatisfiability.
    SearchBudgetExhausted,
}

/// Result of planning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanOutcome {
    /// A minimal plan was found and proven minimal within the budget.
    Planned(Plan),
    /// The search space was fully explored: no executable closure exists.
    Unsatisfiable {
        /// Why candidate branches failed (cycles, empty groups, ...).
        diagnostics: Vec<Diagnostic>,
    },
    /// The budget ran out; the answer is unknown — distinct from
    /// `Unsatisfiable`. Carries the best plan found so far, if any.
    Unknown {
        /// Always [`UnknownReason::SearchBudgetExhausted`].
        reason: UnknownReason,
        /// Best (not necessarily minimal) plan found before the budget ran
        /// out, if any.
        best_so_far: Option<Plan>,
        /// Diagnostics collected before the budget ran out.
        diagnostics: Vec<Diagnostic>,
    },
}

impl PlanOutcome {
    /// `true` when a proven-minimal plan was found.
    #[must_use]
    pub const fn is_planned(&self) -> bool {
        matches!(self, Self::Planned(_))
    }
}

impl fmt::Display for UnknownReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SearchBudgetExhausted => {
                write!(
                    f,
                    "search budget exhausted before the space was fully explored"
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Search internals
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ChoiceRec {
    node: NodeId,
    group: usize,
    provider: NodeId,
    reused: bool,
}

/// Depth-first path from `from` to `to` following depends-on `edges`.
fn dependency_path(edges: &[Vec<NodeId>], from: NodeId, to: NodeId) -> Option<Vec<NodeId>> {
    fn dfs(
        edges: &[Vec<NodeId>],
        node: NodeId,
        to: NodeId,
        visited: &mut [bool],
        path: &mut Vec<NodeId>,
    ) -> bool {
        path.push(node);
        if node == to {
            return true;
        }
        visited[node] = true;
        for &next in &edges[node] {
            if !visited[next] && dfs(edges, next, to, visited, path) {
                return true;
            }
        }
        path.pop();
        false
    }

    let mut visited = vec![false; edges.len()];
    let mut path = Vec::new();
    dfs(edges, from, to, &mut visited, &mut path).then_some(path)
}

struct Search<'p> {
    problem: &'p Planner,
    targets: Vec<NodeId>,
    budget_left: u64,
    out_of_budget: bool,

    selected: Vec<bool>,
    selected_count: usize,
    /// depends-on edges: `depends_on[n]` = chosen providers of `n`.
    depends_on: Vec<Vec<NodeId>>,
    /// Pending (node, group) requirements, used as a stack.
    pending: Vec<(NodeId, usize)>,
    choices: Vec<ChoiceRec>,

    best: Option<Vec<ChoiceRec>>,
    best_count: usize,
    diagnostics: Vec<Diagnostic>,
}

impl<'p> Search<'p> {
    fn new(problem: &'p Planner, targets: &[NodeId], budget: u64) -> Self {
        let mut selected = vec![false; problem.node_count()];
        let mut pending = Vec::new();
        for &target in targets {
            selected[target] = true;
            for group_index in 0..problem.groups(target).len() {
                pending.push((target, group_index));
            }
        }
        Search {
            problem,
            targets: targets.to_vec(),
            budget_left: budget,
            out_of_budget: false,
            selected_count: targets.len(),
            selected,
            depends_on: vec![Vec::new(); problem.node_count()],
            pending,
            choices: Vec::new(),
            best: None,
            best_count: problem.node_count() + 1,
            diagnostics: Vec::new(),
        }
    }

    fn run(&mut self) {
        self.expand();
    }

    fn diagnostic(&mut self, diagnostic: Diagnostic) {
        if self.diagnostics.len() < DIAGNOSTIC_CAP {
            self.diagnostics.push(diagnostic);
        }
    }

    fn expand(&mut self) {
        if self.out_of_budget {
            return;
        }
        // Branch and bound: cannot beat the incumbent any more.
        if self.selected_count >= self.best_count {
            return;
        }
        if self.pending.is_empty() {
            // All requirements satisfied: record the incumbent.
            self.best_count = self.selected_count;
            self.best = Some(self.choices.clone());
            return;
        }
        // The pending worklist is shared mutable state, and the recursion
        // below consumes entries pushed by ancestor frames. Snapshot it and
        // restore it on every exit path so backtracking never loses
        // requirements of sibling branches.
        let saved = self.pending.clone();
        self.expand_inner();
        self.pending = saved;
    }

    fn expand_inner(&mut self) {
        let Some((node, group_index)) = self.pending.pop() else {
            return; // unreachable: expand() checked non-empty
        };
        if self.budget_left == 0 {
            self.out_of_budget = true;
            return;
        }
        self.budget_left -= 1;

        let group = &self.problem.nodes[node].groups[group_index];
        if group.providers.is_empty() {
            let message = format!(
                "group '{}' of '{}' has no candidate providers; branch is dead",
                group.name, self.problem.nodes[node].name
            );
            self.diagnostic(Diagnostic {
                kind: DiagnosticKind::EmptyGroup {
                    node,
                    group: group_index,
                },
                message,
            });
            return;
        }

        // Try already-selected (shared) providers first: this finds good
        // incumbents early and improves pruning. Optimality does not depend
        // on the order because the whole space is still explored.
        let mut providers = group.providers.clone();
        providers.sort_by_key(|&provider| !self.selected[provider]);

        // Stack length after the current requirement was popped; the
        // recursive callee restores `pending` to its own entry state, so
        // truncating back to `base` undoes this iteration's pushes.
        let base = self.pending.len();

        for provider in providers {
            if self.out_of_budget {
                return;
            }
            if let Some(mut loop_path) = dependency_path(&self.depends_on, provider, node) {
                // `provider` already depends on `node`: choosing it would
                // close a cycle, so this branch is non-executable.
                loop_path.push(provider);
                let names: Vec<&str> = loop_path
                    .iter()
                    .map(|&id| self.problem.nodes[id].name.as_str())
                    .collect();
                let message = format!(
                    "cycle {}: provider '{}' for group '{}' of '{}' is non-executable",
                    names.join(" -> "),
                    self.problem.nodes[provider].name,
                    group.name,
                    self.problem.nodes[node].name
                );
                self.diagnostic(Diagnostic {
                    kind: DiagnosticKind::Cycle { loop_path },
                    message,
                });
                continue;
            }

            let reused = self.selected[provider];
            self.depends_on[node].push(provider);
            self.choices.push(ChoiceRec {
                node,
                group: group_index,
                provider,
                reused,
            });
            self.pending.truncate(base);
            if !reused {
                self.selected[provider] = true;
                self.selected_count += 1;
                for sub_index in 0..self.problem.groups(provider).len() {
                    self.pending.push((provider, sub_index));
                }
            }
            self.expand();
            if !reused {
                self.selected[provider] = false;
                self.selected_count -= 1;
            }
            self.choices.pop();
            self.depends_on[node].pop();
        }
    }

    fn finish(self) -> PlanOutcome {
        let Search {
            problem,
            targets,
            out_of_budget,
            best,
            diagnostics,
            ..
        } = self;
        if out_of_budget {
            return PlanOutcome::Unknown {
                reason: UnknownReason::SearchBudgetExhausted,
                best_so_far: best
                    .as_deref()
                    .map(|choices| build_plan(problem, &targets, choices, Vec::new())),
                diagnostics,
            };
        }
        match best {
            Some(choices) => {
                PlanOutcome::Planned(build_plan(problem, &targets, &choices, diagnostics))
            }
            None => PlanOutcome::Unsatisfiable { diagnostics },
        }
    }
}

fn build_plan(
    problem: &Planner,
    targets: &[NodeId],
    choices: &[ChoiceRec],
    diagnostics: Vec<Diagnostic>,
) -> Plan {
    let node_total = problem.node_count();
    let mut selected = vec![false; node_total];
    for &target in targets {
        selected[target] = true;
    }
    for choice in choices {
        selected[choice.provider] = true;
    }
    let nodes: Vec<NodeId> = (0..node_total).filter(|&id| selected[id]).collect();

    let plan_choices: Vec<Choice> = choices
        .iter()
        .map(|choice| {
            let group = &problem.groups(choice.node)[choice.group];
            let node_name = problem.node_name(choice.node);
            let provider_name = problem.node_name(choice.provider);
            let rationale = if choice.reused {
                format!(
                    "satisfy '{}' of '{node_name}' with '{provider_name}': reuses a node \
                     already required by the plan (shared dependency, adds 0 nodes)",
                    group.name()
                )
            } else {
                format!(
                    "satisfy '{}' of '{node_name}' with '{provider_name}': adds \
                     '{provider_name}' as a new node (+1 node)",
                    group.name()
                )
            };
            Choice {
                node: choice.node,
                group: choice.group,
                group_name: group.name().to_string(),
                provider: choice.provider,
                reused: choice.reused,
                rationale,
            }
        })
        .collect();

    // Topological order (providers before dependents) over the selected set,
    // using the chosen edges. Deterministic: smallest id first on ties.
    let mut indegree = vec![0usize; node_total];
    let mut dependents: Vec<Vec<NodeId>> = vec![Vec::new(); node_total];
    for choice in choices {
        dependents[choice.provider].push(choice.node);
        indegree[choice.node] += 1;
    }
    let mut ready: Vec<NodeId> = nodes
        .iter()
        .copied()
        .filter(|&id| indegree[id] == 0)
        .collect();
    let mut execution_order = Vec::with_capacity(nodes.len());
    while let Some(&next) = ready.iter().min() {
        ready.retain(|&queued| queued != next);
        execution_order.push(next);
        for &dependent in &dependents[next] {
            indegree[dependent] -= 1;
            if indegree[dependent] == 0 {
                ready.push(dependent);
            }
        }
    }
    debug_assert_eq!(
        execution_order.len(),
        nodes.len(),
        "chosen edges are acyclic by construction"
    );

    Plan {
        nodes,
        choices: plan_choices,
        execution_order,
        diagnostics,
    }
}
