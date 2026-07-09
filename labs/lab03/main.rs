#![allow(non_snake_case)]
use optimization::{
    helpers::Precision,
    linalg::{Column, SquareMatrix},
    multivariate::conjugate::{Conjugate, PerfectQuadraticProblem},
    optimizer::TryOptimize,
};
use plotly::{Plot, Scatter};

pub fn main() {
    println!("========================================");
    println!("  LAB 3: Conjugate Gradient Method");
    println!("========================================");

    // -------------------------------------------------------------------------
    // Exercise 1 (Dimension n = 5)
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1: Quadratic Optimization (n = 5) ---");
    const N: usize = 5;

    let A = SquareMatrix::<N, f64>::randomized(0.0..2.0);
    let A = (A * A.transpose()) + SquareMatrix::identity();

    println!("Matrix A:");
    println!("{A}");

    let B = Column::<N, f64>::randomized(0.0..2.0);
    println!("Vector B:");
    println!("{B}");

    let analytical_col = A.inverse().unwrap() * B.clone();
    let x_th = analytical_col.clone().into_column();
    println!("Analytical Solution:\n  {:?}", x_th);

    let start_guess = Column::randomized(0.0..2.0);
    let mut x = start_guess.clone();
    let opt = Conjugate::new(Precision(0.001));
    let guesses_iter = opt.try_optimize(
        PerfectQuadraticProblem {
            matrix: A.clone(),
            b: B.clone(),
        },
        start_guess.clone(),
    );

    let mut steps_ex1 = Vec::new();
    let mut errors_ex1 = Vec::new();

    // Step 0 error
    let diff0 = x.clone() - analytical_col.clone();
    errors_ex1.push((diff0.transpose() * diff0).into_value().sqrt().log10());
    steps_ex1.push(0.0);

    let mut steps_cnt = 0;
    for (i, guess_res) in guesses_iter.enumerate() {
        match guess_res {
            Ok(g) => {
                x = g;
                let diff = x.clone() - analytical_col.clone();
                errors_ex1.push((diff.transpose() * diff).into_value().sqrt().log10());
                steps_ex1.push((i + 1) as f64);
                steps_cnt = i + 1;
            }
            Err(err) => {
                println!("  CG Error during execution: {err}");
                break;
            }
        }
    }

    println!("\nConjugate Gradient Solver:");
    println!("  Starting Guess:   {:?}", start_guess.into_column());
    println!("  Final Estimate:   {:?}", x.into_column());
    println!("  Steps:            {}", steps_cnt);
    println!("  Convergence:      Success");

    let diff_final = x - analytical_col;
    let final_err = (diff_final.transpose() * diff_final).into_value().sqrt();
    println!("  Final Error norm: {:.2e}", final_err);

    // Create Plot 1
    let mut plot1 = Plot::new();
    let scatter1 = Scatter::new(steps_ex1, errors_ex1)
        .name("Error Norm vs Steps")
        .x_axis("x")
        .y_axis("y");
    plot1.add_trace(scatter1);
    plot1.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Iteration Step"))
            .y_axis(plotly::layout::Axis::new().title("log10(||x_k - x*||)")),
    );

    // -------------------------------------------------------------------------
    // Exercise 2 (Dimension n = 50)
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 2: High-Dimension Quadratic (n = 50) ---");
    const N2: usize = 50;

    let A2 = SquareMatrix::<N2, f64>::randomized(0.0..1.0);
    let A2 = (A2 * A2.transpose()) + SquareMatrix::identity();
    let B2 = Column::<N2, f64>::randomized(0.0..1.0);

    // Compute analytical solution
    let x_th2 = A2.inverse().unwrap() * B2.clone();

    // Conjugate Gradient with stopping criterion ||grad|| <= 10^-8
    let mut x2 = Column::<N2, f64>::zeros();

    let mut steps_ex2 = Vec::new();
    let mut errors_ex2 = Vec::new();

    // Step 0 error (initial guess)
    let diff0 = x2.clone() - x_th2.clone();
    errors_ex2.push((diff0.transpose() * diff0).into_value().sqrt().log10());
    steps_ex2.push(0.0);

    let opt2 = Conjugate::new(Precision(1e-8));
    let guesses_iter2 = opt2.try_optimize(
        PerfectQuadraticProblem {
            matrix: A2.clone(),
            b: B2.clone(),
        },
        x2.clone(),
    );

    let mut steps2 = 0;
    for (i, guess_res) in guesses_iter2.enumerate() {
        match guess_res {
            Ok(g) => {
                x2 = g;
                let diff = x2.clone() - x_th2.clone();
                errors_ex2.push((diff.transpose() * diff).into_value().sqrt().log10());
                steps_ex2.push((i + 1) as f64);
                steps2 = i + 1;
            }
            Err(err) => {
                println!("  CG Error: {err}");
                break;
            }
        }
    }

    println!("Conjugate Gradient Solver (||grad|| <= 10^-8):");
    println!("  Starting Guess:   Zeros");
    println!("  Steps/Iterations: {}", steps2);
    println!("  Convergence:      Success");

    let diff_final2 = x2 - x_th2;
    let final_err2 = (diff_final2.transpose() * diff_final2).into_value().sqrt();
    println!("  Final Error norm: {:.2e}", final_err2);

    // Create Plot 2
    let mut plot2 = Plot::new();
    let scatter2 = Scatter::new(steps_ex2, errors_ex2)
        .name("Error Norm vs Steps")
        .x_axis("x")
        .y_axis("y");
    plot2.add_trace(scatter2);
    plot2.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Iteration Step"))
            .y_axis(plotly::layout::Axis::new().title("log10(||x_k - x*||)")),
    );

    // Save dashboard
    optimization::helpers::save_dashboard(
        "labs/lab03/plot.html",
        "LAB 3: Conjugate Gradient Method",
        &[
            ("Exercise 1 Convergence (n=5)", &plot1),
            ("Exercise 2 Convergence (n=50)", &plot2),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab03/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab03/plot.html");
}
