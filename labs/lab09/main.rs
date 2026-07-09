use optimization::{
    multivariate::branch_bound::{BranchAndBound, MILPProblem},
    optimizer::Optimize,
};
use plotly::{Plot, Scatter};

fn load_knapsack() -> (Vec<f64>, Vec<f64>) {
    let content =
        std::fs::read_to_string("lab-instructions/11-lab7-Files for lab of May, 8th/knapsack.csv")
            .or_else(|_| std::fs::read_to_string("knapsack.csv"))
            .expect("Failed to read knapsack.csv");
    let mut weights = Vec::new();
    let mut values = Vec::new();
    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 2 {
            let w: f64 = parts[0].trim().parse().unwrap();
            let v: f64 = parts[1].trim().parse().unwrap();
            weights.push(w);
            values.push(v);
        }
    }
    (weights, values)
}

fn main() {
    println!("========================================");
    println!("  LAB 9: Branch and Bound Algorithm");
    println!("========================================");

    // -------------------------------------------------------------------------
    // Exercise 1: Small MILP Problem
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1: 4-Variable Binary MILP ---");
    // Minimize -2.5x1 - 1.1x2 - 0.9x3 - 1.5x4
    // Subject to:
    //   4.3x1 + 3.8x2 + 1.6x3 + 2.1x4 <= 9.2
    //   4x1 + 2x2 + 1.9x3 + 3x4 <= 9
    // xi in {0, 1}

    let c = [-2.5, -1.1, -0.9, -1.5];
    let a = [[4.3, 3.8, 1.6, 2.1], [4.0, 2.0, 1.9, 3.0]];
    let b = [9.2, 9.0];

    let problem1 = MILPProblem::<4, 2> {
        objective_coeffs: c,
        constraint_matrix: a,
        constraint_rhs: b,
    };

    let solver = BranchAndBound;

    let mut nodes_visited = Vec::new();
    let mut best_y_vals = Vec::new();
    let mut step_count = 0;
    let mut final_step1 = None;

    for step in solver.optimize(problem1, ()) {
        nodes_visited.push(step_count as f64);
        best_y_vals.push(step.best_y);
        step_count += 1;
        println!(
            "  Node {}: best_y = {:.2}, active_nodes = {}",
            step_count, step.best_y, step.active_nodes
        );
        final_step1 = Some(step);
    }

    let final_1 = final_step1.expect("B&B should run at least one node");
    println!(
        "  Optimal solution y*: {:.2} (Target = -4.9)",
        final_1.best_y
    );
    if let Some(ref x) = final_1.best_x {
        println!("  Optimal x*:          {:.2?}", x);
    }
    println!("  Nodes evaluated:     {}", step_count);

    let mut plot1 = Plot::new();
    plot1.add_trace(Scatter::new(nodes_visited, best_y_vals).name("Best Cost (Minimization)"));
    plot1.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Nodes Evaluated"))
            .y_axis(plotly::layout::Axis::new().title("Current Best Cost")),
    );

    // -------------------------------------------------------------------------
    // Exercise 2: Knapsack Problem with Branch & Bound
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 2: Knapsack with Branch and Bound ---");
    let (weights, values) = load_knapsack();

    // Minimize: -sum(v_i * x_i)
    let c_knap_vec: Vec<f64> = values.iter().map(|&v| -v).collect();
    let c_knap: [f64; 100] = c_knap_vec.try_into().unwrap();
    let weights_arr: [f64; 100] = weights.try_into().unwrap();
    let a_knap = [weights_arr];
    let b_knap = [1965.0];

    let problem2 = MILPProblem::<100, 1> {
        objective_coeffs: c_knap,
        constraint_matrix: a_knap,
        constraint_rhs: b_knap,
    };

    println!("Running Branch & Bound for Knapsack (100 variables)...");

    let mut nodes_visited_knap = Vec::new();
    let mut best_y_knap = Vec::new();
    let mut step_count_knap = 0;
    let mut final_step2 = None;

    for step in solver.optimize(problem2, ()) {
        // Print status updates periodically to avoid huge terminal output
        if step_count_knap % 100 == 0 {
            println!(
                "  Evaluated {} nodes... best_value = {:.1}, active_nodes_in_stack = {}",
                step_count_knap, -step.best_y, step.active_nodes
            );
        }
        nodes_visited_knap.push(step_count_knap as f64);
        best_y_knap.push(-step.best_y); // Plot as positive value
        step_count_knap += 1;
        final_step2 = Some(step);
    }

    let final_2 = final_step2.expect("B&B Knapsack should run");
    let opt_val = -final_2.best_y;
    println!("  Optimal value achieved: {:.1} (Target = 4966.0)", opt_val);
    println!("  Total nodes evaluated:  {}", step_count_knap);

    let mut plot2 = Plot::new();
    plot2.add_trace(Scatter::new(nodes_visited_knap, best_y_knap).name("Best Knapsack Value"));
    plot2.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Nodes Evaluated"))
            .y_axis(plotly::layout::Axis::new().title("Best Knapsack Value")),
    );

    optimization::helpers::save_dashboard(
        "labs/lab09/plot.html",
        "LAB 9: Branch & Bound Solver",
        &[
            ("Exercise 1 MILP Convergence", &plot1),
            ("Exercise 2 Knapsack B&B Convergence", &plot2),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab09/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab09/plot.html");
}
