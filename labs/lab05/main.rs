#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use optimization::{
    helpers::Iterations,
    linalg::{Column, Matrix, Row},
    multivariate::simplex::{AugmentedLP, LinearProgram, PermutedLP, Simplex, SimplexGuess},
    optimizer::TryOptimize,
};
use plotly::{Plot, Scatter};

fn main() {
    println!("========================================");
    println!("  LAB 5: Linear Programming & Simplex");
    println!("========================================");

    // -------------------------------------------------------------------------
    // Exercise 1: Standard Simplex Algorithm
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1: Standard Simplex Algorithm ---");
    let problem1 = LinearProgram {
        a: Matrix([
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ]),
        b: Column::new_column([8.0, 5.0, 2.0, 6.0]),
        c: Row::new_row([5.0, 5.0, 3.0, 6.0, 4.0, 1.0]),
    };
    let opt = Simplex::new(Iterations(100));

    let start_guess1 = SimplexGuess {
        base_idx: [0, 1, 2, 3], // Columns 0, 1, 2, 3 as initial basis
        x: Column::zeros(),
        is_optimal: false,
    };

    println!(
        "Starting basis indices (0-indexed): {:?}",
        start_guess1.base_idx
    );

    let mut steps_ex1 = Vec::new();
    let mut costs_ex1 = Vec::new();
    let mut step = 0;
    let mut last_guess1 = None;

    for res in opt.try_optimize(problem1.clone(), start_guess1.clone()) {
        match res {
            Ok(guess) => {
                let cost = (problem1.c * guess.x).into_value();
                steps_ex1.push(step as f64);
                costs_ex1.push(cost);
                println!(
                    "  Iteration {}: basis={:?}, cost={:.2}, x={:.2?}",
                    step,
                    guess.base_idx,
                    cost,
                    guess.x.into_column()
                );
                step += 1;
                last_guess1 = Some(guess);
            }
            Err(err) => {
                println!("  Simplex Error: {}", err);
                break;
            }
        }
    }

    let final_guess1 = last_guess1.expect("Simplex should have run at least one iteration");
    let final_cost1 = (problem1.c * final_guess1.x).into_value();

    println!("\nMethod: Standard Simplex Solver");
    println!(
        "  Starting Guess: Zeros (basis: {:?})",
        start_guess1.base_idx
    );
    println!("  Final Estimate: {:.5?}", final_guess1.x.into_column());
    println!("  Optimal Cost:   {:.5}", final_cost1);
    println!("  Steps:          {}", step);
    println!(
        "  Convergence:    {}",
        if final_guess1.is_optimal {
            "Success"
        } else {
            "Failed"
        }
    );

    // Create Plot 1
    let mut plot1 = Plot::new();
    let scatter1 = Scatter::new(steps_ex1, costs_ex1)
        .name("Cost vs Iterations")
        .x_axis("x")
        .y_axis("y");
    plot1.add_trace(scatter1);
    plot1.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Iteration Step"))
            .y_axis(plotly::layout::Axis::new().title("Objective Value")),
    );

    // -------------------------------------------------------------------------
    // Exercise 2: Two-Phase Simplex Algorithm
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 2: Two-Phase Simplex Algorithm ---");
    let a = Matrix([
        [1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.0],
        [6.0, 5.0, 3.0, 2.0, 0.0, 1.0, 0.0],
        [3.0, 4.0, 9.0, 12.0, 0.0, 0.0, 1.0],
    ]);
    let b = Column::new_column([20.0, 19.0, 75.0]);
    let c = Row::new_row([-6.0, -4.0, -7.0, -5.0, 0.0, 0.0, 0.0]);

    println!("Constraint Matrix A:");
    println!("{a}");
    println!("Vector b:");
    println!("{b}");
    println!("Cost Vector c:");
    println!("{c}");

    println!("\n>> Phase 1: Auxiliary Problem (driving artificial variables to zero) <<");
    let phase1_problem = AugmentedLP::new(&a, &b);
    let start_guess_p1 = SimplexGuess {
        base_idx: [0, 1, 2], // Artificial variables form the initial identity basis
        x: Column::zeros(),
        is_optimal: false,
    };

    let mut steps_p1 = Vec::new();
    let mut costs_p1 = Vec::new();
    let mut step_p1 = 0;
    let mut last_guess_p1 = None;

    for res in opt.try_optimize(phase1_problem, start_guess_p1.clone()) {
        match res {
            Ok(guess) => {
                let mut virtual_cost = 0.0;
                // Artificial variables are variables at index 0, 1, 2
                for i in 0..3 {
                    virtual_cost += guess.x.0[i][0];
                }
                steps_p1.push(step_p1 as f64);
                costs_p1.push(virtual_cost);
                println!(
                    "  Phase 1 Iteration {}: basis={:?}, virtual_cost={:.4}, x={:.2?}",
                    step_p1,
                    guess.base_idx,
                    virtual_cost,
                    guess.x.into_column()
                );
                step_p1 += 1;
                last_guess_p1 = Some(guess);
            }
            Err(err) => {
                println!("  Phase 1 Error: {}", err);
                break;
            }
        }
    }

    let final_guess_p1 = last_guess_p1.expect("Phase 1 should have run iterations");
    let mut final_virtual_cost = 0.0;
    for i in 0..3 {
        final_virtual_cost += final_guess_p1.x.0[i][0];
    }

    if final_virtual_cost > 1e-6 {
        println!(
            "Phase 1 terminated with positive virtual cost: {:.4}. Original LP is infeasible.",
            final_virtual_cost
        );
        return;
    } else {
        println!(
            "Phase 1 optimal virtual cost is {:.4}. A Basic Feasible Solution is found!",
            final_virtual_cost
        );
    }

    println!("\n>> Phase 2: Solve permuted system using BFS from Phase 1 <<");
    println!(
        "Final Phase 1 basis indices (in augmented system): {:?}",
        final_guess_p1.base_idx
    );

    let permuted_lp = PermutedLP::new(&a, &b, &c, &final_guess_p1.base_idx)
        .expect("Failed to perform coordinate change and permute LP");

    println!("Permuted constraint matrix A_perm:");
    println!("{}", permuted_lp.a_perm);
    println!("Permuted vector b_perm:");
    println!("{}", permuted_lp.b_perm);
    println!(
        "Permutation mapping (indices map to original variables): {:?}",
        permuted_lp.perm
    );

    let start_guess_p2 = SimplexGuess {
        base_idx: [0, 1, 2], // First C columns are identity in permuted system
        x: Column::zeros(),
        is_optimal: false,
    };

    let mut steps_p2 = Vec::new();
    let mut costs_p2 = Vec::new();
    let mut step_p2 = 0;
    let mut last_guess_p2 = None;

    for res in opt.try_optimize(permuted_lp.to_linear_program(), start_guess_p2.clone()) {
        match res {
            Ok(guess) => {
                let cost = (permuted_lp.c_perm * guess.x).into_value();
                steps_p2.push(step_p2 as f64);
                costs_p2.push(cost);
                println!(
                    "  Phase 2 Iteration {}: basis={:?}, cost={:.4}, x={:.2?}",
                    step_p2,
                    guess.base_idx,
                    cost,
                    guess.x.into_column()
                );
                step_p2 += 1;
                last_guess_p2 = Some(guess);
            }
            Err(err) => {
                println!("  Phase 2 Error: {}", err);
                break;
            }
        }
    }

    let final_guess_p2 = last_guess_p2.expect("Phase 2 should have run iterations");

    // Reconstruct solution in original variable space
    let mut x_orig = [0.0; 7];
    for i in 0..7 {
        x_orig[permuted_lp.perm[i]] = final_guess_p2.x.0[i][0];
    }

    let final_cost_ex2 = (c * Column::new_column(x_orig)).into_value();

    println!("\nMethod: Two-Phase Simplex Solver");
    println!("  Final Estimate (Original Space): {:.5?}", x_orig);
    println!("  Optimal Cost:                   {:.5}", final_cost_ex2);
    println!("  Phase 1 Steps:                  {}", step_p1);
    println!("  Phase 2 Steps:                  {}", step_p2);
    println!(
        "  Convergence:                    {}",
        if final_guess_p2.is_optimal {
            "Success"
        } else {
            "Failed"
        }
    );

    // Create Plots for Phase 1 & 2
    let mut plot2 = Plot::new();
    let scatter_p1 = Scatter::new(steps_p1, costs_p1)
        .name("Phase 1 (Virtual Cost)")
        .x_axis("x")
        .y_axis("y");
    let scatter_p2 = Scatter::new(steps_p2, costs_p2)
        .name("Phase 2 (Real Cost)")
        .x_axis("x")
        .y_axis("y");
    plot2.add_trace(scatter_p1);
    plot2.add_trace(scatter_p2);
    plot2.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Iteration Step"))
            .y_axis(plotly::layout::Axis::new().title("Objective / Virtual Cost")),
    );

    // Save dashboard
    optimization::helpers::save_dashboard(
        "labs/lab05/plot.html",
        "LAB 5: Linear Programming & Simplex",
        &[
            ("Exercise 1 Simplex Cost Trajectory", &plot1),
            ("Exercise 2 Two-Phase Simplex Trajectory", &plot2),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab05/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab05/plot.html");
}
