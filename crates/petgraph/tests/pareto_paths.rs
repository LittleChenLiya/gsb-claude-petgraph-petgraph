use hashbrown::HashMap;
use petgraph::{
    algo::{pareto_paths, ParetoPathError},
    prelude::*,
    stable_graph::StableGraph,
    Directed, Graph,
};

type W = u32;

fn brute_force(
    g: &Graph<(), (W, W), Directed>,
    start: NodeIndex,
    goal: NodeIndex,
    budget: W,
) -> Vec<(W, W, Vec<EdgeIndex>)> {
    let mut best: HashMap<(W, W), Vec<EdgeIndex>> = HashMap::new();
    let mut visited = vec![false; g.node_count()];
    let mut path_nodes: Vec<NodeIndex> = vec![start];
    let mut path_edges: Vec<EdgeIndex> = Vec::new();
    visited[start.index()] = true;
    dfs(
        g,
        start,
        goal,
        budget,
        0,
        0,
        &mut visited,
        &mut path_nodes,
        &mut path_edges,
        &mut best,
    );

    let entries: Vec<(W, W, Vec<EdgeIndex>)> = best
        .into_iter()
        .map(|((c, r), edges)| (c, r, edges))
        .collect();
    let mut keep = vec![true; entries.len()];
    for i in 0..entries.len() {
        for j in 0..entries.len() {
            if i == j {
                continue;
            }
            let (ci, ri) = (entries[i].0, entries[i].1);
            let (cj, rj) = (entries[j].0, entries[j].1);
            if cj <= ci && rj <= ri && (cj < ci || rj < ri) {
                keep[i] = false;
            }
        }
    }
    let mut result: Vec<_> = entries
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, e)| e)
        .collect();
    result.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    result
}

#[allow(clippy::too_many_arguments)]
fn dfs(
    g: &Graph<(), (W, W), Directed>,
    node: NodeIndex,
    goal: NodeIndex,
    budget: W,
    cost: W,
    resource: W,
    visited: &mut [bool],
    path_nodes: &mut Vec<NodeIndex>,
    path_edges: &mut Vec<EdgeIndex>,
    best: &mut HashMap<(W, W), Vec<EdgeIndex>>,
) {
    if node == goal {
        let key = (cost, resource);
        match best.get(&key) {
            Some(existing)
                if existing.as_slice().cmp(path_edges.as_slice()) <= core::cmp::Ordering::Equal => {}
            _ => {
                best.insert(key, path_edges.clone());
            }
        }
        return;
    }
    let mut walk = g.neighbors(node).detach();
    while let Some((edge_index, next)) = walk.next(g) {
        let (ec, er) = g[edge_index];
        let next_cost = cost + ec;
        let next_resource = resource + er;
        if next_resource > budget || visited[next.index()] {
            continue;
        }
        visited[next.index()] = true;
        path_nodes.push(next);
        path_edges.push(edge_index);
        dfs(
            g,
            next,
            goal,
            budget,
            next_cost,
            next_resource,
            visited,
            path_nodes,
            path_edges,
            best,
        );
        path_edges.pop();
        path_nodes.pop();
        visited[next.index()] = false;
    }
}

fn assert_matches_oracle(
    g: &Graph<(), (W, W), Directed>,
    start: NodeIndex,
    goal: NodeIndex,
    budget: W,
) {
    let oracle = brute_force(g, start, goal, budget);
    let result =
        pareto_paths(g, start, goal, budget, 1024, |e| e.weight().0, |e| e.weight().1)
            .expect("search should succeed with a generous bound");
    let got: Vec<(W, W, Vec<EdgeIndex>)> = result
        .labels
        .iter()
        .map(|label| (label.cost, label.resource, label.edges.clone()))
        .collect();
    assert_eq!(got, oracle);
}

#[test]
fn acceptance_three_parallel_edges() {
    let mut g = Graph::<(), (W, W), Directed>::new();
    let s = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, t, (2, 8));
    g.add_edge(s, t, (5, 3));
    g.add_edge(s, t, (6, 9));

    let res = pareto_paths(&g, s, t, 10, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    let frontier: Vec<(W, W)> = res
        .labels
        .iter()
        .map(|label| (label.cost, label.resource))
        .collect();
    assert_eq!(frontier, vec![(2, 8), (5, 3)]);
    assert_eq!(res.labels[0].edges, vec![EdgeIndex::new(0)]);
    assert_eq!(res.labels[1].edges, vec![EdgeIndex::new(1)]);

    let tight = pareto_paths(&g, s, t, 5, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    let frontier: Vec<(W, W)> = tight
        .labels
        .iter()
        .map(|label| (label.cost, label.resource))
        .collect();
    assert_eq!(frontier, vec![(5, 3)]);

    // Budget 8 is an inclusive boundary: (2, 8) qualifies, (5, 3) too.
    let boundary = pareto_paths(&g, s, t, 8, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    assert_eq!(boundary.labels.len(), 2);

    // Below every path's resource the frontier is empty but not an error.
    let none = pareto_paths(&g, s, t, 2, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    assert!(none.labels.is_empty());
}

#[test]
fn zero_weight_cycle_terminates() {
    let mut g = Graph::<(), (W, W), Directed>::new();
    let s = g.add_node(());
    let a = g.add_node(());
    let b = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, a, (1, 1));
    g.add_edge(a, b, (0, 0));
    g.add_edge(b, a, (0, 0));
    g.add_edge(a, t, (1, 1));

    let res = pareto_paths(&g, s, t, 10, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    let frontier: Vec<(W, W)> = res
        .labels
        .iter()
        .map(|label| (label.cost, label.resource))
        .collect();
    assert_eq!(frontier, vec![(2, 2)]);
    assert_eq!(res.labels[0].edges.len(), 2);
}

#[test]
fn label_limit_is_hard_error() {
    let mut g = Graph::<(), (W, W), Directed>::new();
    let s = g.add_node(());
    let m = g.add_node(());
    let t = g.add_node(());
    // Many incomparable labels at m (equal resource, increasing cost is
    // dominated, so use distinct incomparable pairs along the s -> m fan-in).
    g.add_edge(s, m, (10, 1));
    g.add_edge(s, m, (9, 2));
    g.add_edge(s, m, (8, 3));
    g.add_edge(s, m, (7, 4));
    g.add_edge(m, t, (0, 0));

    let err = pareto_paths(&g, s, t, 100, 2, |e| e.weight().0, |e| e.weight().1).unwrap_err();
    match err {
        ParetoPathError::LabelLimitReached {
            max_labels, stats, ..
        } => {
            assert_eq!(max_labels, 2);
            assert!(stats.labels_generated >= 3);
        }
        other => panic!("expected LabelLimitReached, got {other:?}"),
    }

    // With a sufficient bound the same graph returns the complete frontier.
    let ok = pareto_paths(&g, s, t, 100, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    assert_eq!(ok.labels.len(), 4);
}

#[test]
fn parallel_edges_keep_distinct_identities() {
    let mut g = Graph::<(), (W, W), Directed>::new();
    let s = g.add_node(());
    let t = g.add_node(());
    // Identical metrics on parallel edges: lexicographically smallest edge id
    // must win.
    g.add_edge(s, t, (1, 1));
    g.add_edge(s, t, (1, 1));
    g.add_edge(s, t, (1, 1));

    let res = pareto_paths(&g, s, t, 10, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    assert_eq!(res.labels.len(), 1);
    assert_eq!(res.labels[0].cost, 1);
    assert_eq!(res.labels[0].resource, 1);
    assert_eq!(res.labels[0].edges, vec![EdgeIndex::new(0)]);
    assert_eq!(res.labels[0].nodes, vec![s, t]);

    // Two-step paths differ only by which parallel edge is used; the smallest
    // edge id sequence is retained for the shared metric.
    let mut g2 = Graph::<(), (W, W), Directed>::new();
    let a = g2.add_node(());
    let b = g2.add_node(());
    let c = g2.add_node(());
    g2.add_edge(a, b, (0, 0)); // edge 0
    g2.add_edge(a, b, (0, 0)); // edge 1
    g2.add_edge(b, c, (0, 0)); // edge 2
    g2.add_edge(b, c, (0, 0)); // edge 3
    let res = pareto_paths(&g2, a, c, 10, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    assert_eq!(res.labels.len(), 1);
    assert_eq!(res.labels[0].edges, vec![EdgeIndex::new(0), EdgeIndex::new(2)]);
}

#[test]
fn stable_graph_with_holes() {
    let mut g = StableGraph::<(), (W, W), Directed>::new();
    let s = g.add_node(());
    let gone = g.add_node(());
    let t = g.add_node(());
    g.remove_node(gone); // leaves an index hole

    let edge = g.add_edge(s, t, (4, 4));
    g.add_edge(s, t, (3, 9));

    let res =
        pareto_paths(&g, s, t, 10, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    let frontier: Vec<(W, W)> = res
        .labels
        .iter()
        .map(|label| (label.cost, label.resource))
        .collect();
    assert_eq!(frontier, vec![(3, 9), (4, 4)]);
    assert!(res.labels.iter().any(|label| label.edges == vec![edge]));

    // A removed node cannot be used as a source.
    assert_eq!(
        pareto_paths(&g, gone, t, 10, 64, |e| e.weight().0, |e| e.weight().1).unwrap_err(),
        ParetoPathError::NodeNotFound
    );
}

#[test]
fn source_equals_target() {
    let mut g = Graph::<(), (W, W), Directed>::new();
    let s = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, t, (1, 1));

    let res = pareto_paths(&g, s, s, 10, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    assert_eq!(res.labels.len(), 1);
    assert_eq!(res.labels[0].cost, 0);
    assert_eq!(res.labels[0].resource, 0);
    assert!(res.labels[0].edges.is_empty());
    assert_eq!(res.labels[0].nodes, vec![s]);
}

#[test]
fn cost_overflow_is_reported() {
    let mut g = Graph::<(), (u8, u8), Directed>::new();
    let s = g.add_node(());
    let a = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, a, (200, 0));
    g.add_edge(a, t, (100, 0));

    let err = pareto_paths(&g, s, t, 10, 64, |e| e.weight().0, |e| e.weight().1).unwrap_err();
    assert_eq!(err, ParetoPathError::WeightOverflow);
}

#[test]
fn negative_weight_is_rejected() {
    let mut g = Graph::<(), i32, Directed>::new();
    let s = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, t, -1);

    let err = pareto_paths(&g, s, t, 10, 64, |e| *e.weight(), |_| 0i32).unwrap_err();
    assert_eq!(err, ParetoPathError::NegativeWeight);
}

// Simple pseudo-random linear congruential generator for deterministic tests.
struct Rng(u64);
impl Rng {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound
    }
}

#[test]
fn matches_brute_force_on_random_small_graphs() {
    let mut rng = Rng(0x1234_5678);
    for _ in 0..200 {
        let node_count = 1 + rng.next(5);
        let mut g = Graph::<(), (W, W), Directed>::new();
        let nodes: Vec<NodeIndex> = (0..node_count).map(|_| g.add_node(())).collect();
        let edge_count = rng.next(node_count * node_count + 1);
        for _ in 0..edge_count {
            let from = nodes[rng.next(node_count)];
            let to = nodes[rng.next(node_count)];
            let cost = (rng.next(4) as W) + 1;
            let resource = (rng.next(4) as W) + 1;
            g.add_edge(from, to, (cost, resource));
        }
        let start = nodes[rng.next(node_count)];
        let goal = nodes[rng.next(node_count)];
        let budget = (rng.next(10) as W) + 1;
        assert_matches_oracle(&g, start, goal, budget);
    }
}

#[test]
fn matches_brute_force_with_frontier_churn() {
    // Dense small graphs force many frontier replacements via dominance and
    // lexicographic ties; this exercises lazy-deletion bookkeeping.
    let mut rng = Rng(0xABCD_EF01);
    for _ in 0..60 {
        let node_count = 5;
        let mut g = Graph::<(), (W, W), Directed>::new();
        let nodes: Vec<NodeIndex> = (0..node_count).map(|_| g.add_node(())).collect();
        for from in &nodes {
            for to in &nodes {
                // Mostly zero weights to create equal metrics and cycles.
                let cost = if rng.next(3) == 0 { rng.next(3) as W } else { 0 };
                let resource = if rng.next(3) == 0 { rng.next(3) as W } else { 0 };
                g.add_edge(*from, *to, (cost, resource));
            }
        }
        assert_matches_oracle(&g, nodes[0], nodes[node_count - 1], 6);
    }
}

#[test]
fn report_shows_edge_indices_and_frontier() {
    // The acceptance scenario printed as an example report.
    let mut g = Graph::<(), (W, W), Directed>::new();
    let s = g.add_node(());
    let t = g.add_node(());
    g.add_edge(s, t, (2, 8));
    g.add_edge(s, t, (5, 3));
    g.add_edge(s, t, (6, 9));

    let res =
        pareto_paths(&g, s, t, 10, 64, |e| e.weight().0, |e| e.weight().1).unwrap();
    let mut report = String::from("budget=10");
    for label in &res.labels {
        let edges: Vec<String> = label
            .edges
            .iter()
            .map(|edge| edge.index().to_string())
            .collect();
        report.push_str(&format!(
            " | (cost={}, resource={}, edges=[{}])",
            label.cost,
            label.resource,
            edges.join(",")
        ));
    }
    assert_eq!(
        report,
        "budget=10 | (cost=2, resource=8, edges=[0]) | (cost=5, resource=3, edges=[1])"
    );
}
