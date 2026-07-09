use optimization::{
    multivariate::dynamic_programming::{DPProblem, DynamicProgramming},
    optimizer::Optimize,
};
use plotly::{Plot, Scatter};

fn load_knapsack() -> (Vec<f64>, Vec<usize>) {
    let content = std::fs::read_to_string("lab-instructions/11-lab7-Files for lab of May, 8th/knapsack.csv")
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
            // DP table requires integer capacities, so we round/cast weights to usize
            weights.push(w.round() as usize);
            values.push(v);
        }
    }
    (values, weights)
}

fn main() {
    println!("========================================");
    println!("  LAB 10: Dynamic Programming");
    println!("========================================");

    // -------------------------------------------------------------------------
    // Exercise 1: Small Knapsack Problem
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1: Small Knapsack Test Case ---");
    let v_small = [4.0, 3.0, 3.0, 7.0, 2.0];
    let w_small = [4, 5, 2, 6, 1];
    let w_max_small = 10;

    let problem1 = DPProblem {
        state_space_size: w_max_small + 1,
        transition_function: move |item_idx: usize, capacity: usize, prev_row: &[f64]| {
            let val = v_small[item_idx];
            let wt = w_small[item_idx];
            if wt <= capacity {
                let take_val = prev_row[capacity - wt] + val;
                let skip_val = prev_row[capacity];
                if take_val > skip_val { take_val } else { skip_val }
            } else {
                prev_row[capacity]
            }
        },
        backtrack_function: move |stage: usize, capacity: &mut usize, dp_table: &[Vec<f64>]| {
            let prev_val = dp_table[stage - 1][*capacity];
            let curr_val = dp_table[stage][*capacity];
            if (curr_val - prev_val).abs() > 1e-7 {
                *capacity -= w_small[stage - 1];
                true
            } else {
                false
            }
        },
    };

    let solver = DynamicProgramming;

    let mut final_step1 = None;
    let mut stages_small = Vec::new();
    let mut values_small = Vec::new();

    for (k, step) in solver.optimize(problem1, ()).enumerate() {
        stages_small.push(k as f64 + 1.0);
        values_small.push(step.max_value_found);
        final_step1 = Some(step);
    }

    let final_1 = final_step1.expect("DP Small should run");
    println!("  Optimal Value: {:.1} (Target = 12.0)", final_1.max_value_found);
    let decisions1: [bool; 5] = final_1.optimal_decisions.unwrap();
    println!("  Chosen items (0-indexed):");
    for i in 0..decisions1.len() {
        if decisions1[i] {
            println!("    Item {} (weight: {}, value: {})", i + 1, [4, 5, 2, 6, 1][i], [4.0, 3.0, 3.0, 7.0, 2.0][i]);
        }
    }

    let mut plot1 = Plot::new();
    plot1.add_trace(Scatter::new(stages_small, values_small).name("Small DP Max Value"));
    plot1.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Stage (Items Considered)"))
            .y_axis(plotly::layout::Axis::new().title("Max Value")),
    );

    // -------------------------------------------------------------------------
    // Exercise 2: 100-Item Knapsack Problem
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 2: Large Knapsack (100 items) ---");
    let (v_large_vec, w_large_vec) = load_knapsack();
    let v_large: [f64; 100] = v_large_vec.try_into().unwrap();
    let w_large: [usize; 100] = w_large_vec.try_into().unwrap();
    let w_max_large = 1965;

    let problem2 = DPProblem {
        state_space_size: w_max_large + 1,
        transition_function: move |item_idx: usize, capacity: usize, prev_row: &[f64]| {
            let val = v_large[item_idx];
            let wt = w_large[item_idx];
            if wt <= capacity {
                let take_val = prev_row[capacity - wt] + val;
                let skip_val = prev_row[capacity];
                if take_val > skip_val { take_val } else { skip_val }
            } else {
                prev_row[capacity]
            }
        },
        backtrack_function: move |stage: usize, capacity: &mut usize, dp_table: &[Vec<f64>]| {
            let prev_val = dp_table[stage - 1][*capacity];
            let curr_val = dp_table[stage][*capacity];
            if (curr_val - prev_val).abs() > 1e-7 {
                *capacity -= w_large[stage - 1];
                true
            } else {
                false
            }
        },
    };

    println!("Running Dynamic Programming...");

    let mut final_step2 = None;
    let mut stages_large = Vec::new();
    let mut values_large = Vec::new();

    for (k, step) in solver.optimize(problem2, ()).enumerate() {
        stages_large.push(k as f64 + 1.0);
        values_large.push(step.max_value_found);
        final_step2 = Some(step);
    }

    let final_2 = final_step2.expect("DP Large should run");
    println!("  Optimal Value: {:.1} (Target = 4966.0)", final_2.max_value_found);

    let decisions2: [bool; 100] = final_2.optimal_decisions.unwrap();
    let mut chosen_count = 0;
    let mut total_weight = 0;
    for i in 0..decisions2.len() {
        if decisions2[i] {
            chosen_count += 1;
            // Load original weights again to output exact weight sum
            let (_, original_w) = load_knapsack();
            total_weight += original_w[i];
        }
    }
    println!("  Items chosen:  {} / 100", chosen_count);
    println!("  Total weight:  {} / {}", total_weight, w_max_large);

    let mut plot2 = Plot::new();
    plot2.add_trace(Scatter::new(stages_large, values_large).name("Large DP Max Value"));
    plot2.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Stage (Items Considered)"))
            .y_axis(plotly::layout::Axis::new().title("Max Value")),
    );

    optimization::helpers::save_dashboard(
        "labs/lab10/plot.html",
        "LAB 10: Dynamic Programming for Knapsack",
        &[
            ("Exercise 1 Small Knapsack Progression", &plot1),
            ("Exercise 2 Large Knapsack Progression", &plot2),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab10/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab10/plot.html");
}
