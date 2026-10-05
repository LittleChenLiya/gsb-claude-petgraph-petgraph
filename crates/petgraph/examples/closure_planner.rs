//! Dependency closure planner example.
//!
//! This example demonstrates the six-node sharing problem:
//!
//! ```text
//!   a       b
//!  / \     / \
//! c   d   d   e
//!  \ /     \ /
//!   f       (none)
//! ```
//!
//! Targets `a` and `b` each need to choose a predecessor from a satisfaction
//! group. `a` can pick `c` or `d`; `b` can pick `d` or `e`. `c` and `d` each
//! require `f`. The shared solution (`a -> d`, `b -> d`) requires only
//! `{a, b, d, f}` (4 nodes), while independent optimization picks `c` for `a`
//! and `e` for `b`, requiring 5 nodes.
//!
//! Run with:
//! ```sh
//! cargo run -p petgraph --example closure_planner
//! ```

use hashbrown::HashMap;
use petgraph::algo::{PlannerResult, plan_dependency_closure};

fn main() {
    // Build the six-node satisfaction groups.
    let mut groups: HashMap<&str, Vec<Vec<&str>>> = HashMap::new();
    groups.insert("a", vec![vec!["c", "d"]]);
    groups.insert("b", vec![vec!["d", "e"]]);
    groups.insert("c", vec![vec!["f"]]);
    groups.insert("d", vec![vec!["f"]]);
    groups.insert("e", vec![]);
    groups.insert("f", vec![]);

    let targets = ["a", "b"];

    println!("Six-node dependency closure planner");
    println!("====================================");
    println!();
    println!("Targets: {:?}", targets);
    println!();
    println!("Satisfaction groups:");
    for (node, gs) in &groups {
        if gs.is_empty() {
            println!("  {}: (no dependencies)", node);
        } else {
            for (i, g) in gs.iter().enumerate() {
                println!("  {} group {}: choose one of {:?}", node, i, g);
            }
        }
    }
    println!();

    // Plan with a generous budget (exact for this small graph).
    let result = plan_dependency_closure(&targets, &groups, 1000);

    match result {
        PlannerResult::Solution(plan) => {
            println!("Result: Solution");
            println!("  Required set: {:?}", plan.required);
            println!("  Build order:  {:?}", plan.build_order);
            println!();
            println!("  Selection rationale:");
            for choice in &plan.choices {
                println!(
                    "    {} group {}: chose {} (alternatives: {:?}, introduced: {:?})",
                    choice.node,
                    choice.group_index,
                    choice.chosen,
                    choice.alternatives,
                    choice.added_nodes
                );
            }
            println!();
            println!(
                "The shared solution requires {} nodes: {:?}",
                plan.required.len(),
                plan.required
            );
            println!("Independent optimization would need 5 nodes (c for a, e for b).");
            println!("Sharing beats independent optimization!");
        }
        PlannerResult::NoSolution => {
            println!("Result: NoSolution (all branches cyclic or unsatisfiable)");
        }
        PlannerResult::Unknown => {
            println!("Result: Unknown (budget exhausted)");
        }
    }
}
