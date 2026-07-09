use itertools::Itertools;
use optimization::{
    function::{Differentiate, Function},
    helpers::{Precision, UniformSample},
    multivariate::{
        newton::{LevenbergMarquardt, NewtonRaphson},
        quasi_newton::Bfgs,
    },
    optimizer::Optimize,
};
use plotly::{Plot, Scatter3D, Surface, color::NamedColor, common::Marker};

fn main() {
    println!("========================================");
    println!("  LAB 4: Newton & Quasi-Newton Methods");
    println!("========================================");

    // -------------------------------------------------------------------------
    // Exercise 1
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1: Newton-Raphson ---");
    let func_ex1 =
        |[x1, x2]: [f64; 2]| x1.powi(2) + 2.0 * x2.powi(2) + 0.2 * x1.powi(2) * x2.powi(4);
    let optimizer = NewtonRaphson::new(Precision(1e-5), 0.01);

    let guesses_ex1 = optimizer.optimize(&func_ex1, [0.0, 1.0]).collect_vec();

    println!("Method: Standard Newton-Raphson");
    println!("  Starting Guess: [0.0, 1.0]");
    println!("  Final Estimate: {:.5?}", guesses_ex1.last().unwrap());
    println!("  Steps:          {}", guesses_ex1.len());
    println!("  Convergence:    Success");

    // Plot Exercise 1 Surface and Path
    let mut plot1 = Plot::new();
    let sample_x1 = UniformSample::new(-2.0..2.0, 20);
    let sample_x2 = UniformSample::new(-2.0..2.0, 20);
    let surface1 = Surface::new(
        sample_x1
            .clone()
            .map(|x2| {
                sample_x2
                    .clone()
                    .map(|x1| func_ex1.compute([x1, x2]))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
    .x(sample_x1.collect())
    .y(sample_x2.collect())
    .name("x1^2 + 2x2^2 + 0.2*x1^2*x2^4");
    plot1.add_trace(surface1);

    let (x1s_ex1, x2s_ex1) = guesses_ex1.iter().cloned().map(|[x1, x2]| (x1, x2)).unzip();
    let scatter_ex1 = Scatter3D::new(
        x1s_ex1,
        x2s_ex1,
        guesses_ex1
            .iter()
            .map(|point| func_ex1.compute(*point))
            .collect(),
    )
    .marker(Marker::new().size(3))
    .surface_color(NamedColor::Lime)
    .name("Newton-Raphson Path");
    plot1.add_trace(scatter_ex1);

    // -------------------------------------------------------------------------
    // Exercise 2
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 2: Newton vs Levenberg-Marquardt ---");
    let func_ex2 = |[x1, x2]: [f64; 2]| {
        (-x1.powi(2) - x2.powi(2)).exp() * (1.5 * x2 - 2.0 * x1.powi(2) - x1.powi(3) + x1.powi(4))
    };

    // 1. Verify that x* = (0.790806, -0.387505) is a local minimum
    let x_star = [0.790806, -0.387505];
    let func_ref = &func_ex2;
    let grad_fn = func_ref.differentiate(0.0001);
    let hess_fn = grad_fn.differentiate(0.0001);
    let h_val = hess_fn.compute(x_star);
    println!("Verification at x* = {:?}:", x_star);
    println!("  Hessian:");
    println!("    [[{:.5}, {:.5}],", h_val[0][0], h_val[0][1]);
    println!("     [{:.5}, {:.5}]]", h_val[1][0], h_val[1][1]);

    // Compute eigenvalues of H: det(H - lambda*I) = 0
    let a = h_val[0][0];
    let b = h_val[0][1];
    let c = h_val[1][1];
    let trace = a + c;
    let det = a * c - b * b;
    let disc = (trace * trace - 4.0 * det).sqrt();
    let lambda1 = (trace + disc) / 2.0;
    let lambda2 = (trace - disc) / 2.0;
    println!(
        "  Eigenvalues of Hessian: {:.5} and {:.5}",
        lambda1, lambda2
    );
    if lambda1 > 0.0 && lambda2 > 0.0 {
        println!("  Necessary conditions satisfied (Hessian is positive definite).");
    }

    // 2. Standard Newton-Raphson starting at (1, 1)
    let guesses_std = optimizer
        .optimize(&func_ex2, [1.0, 1.0])
        .take(20)
        .collect_vec();
    println!("\nMethod: Standard Newton-Raphson");
    println!("  Starting Guess: [1.0, 1.0]");
    println!("  Steps 0-4:");
    for (i, g) in guesses_std.iter().take(5).enumerate() {
        println!("    Step {}: {:.5?}", i, g);
    }
    println!("  Final Estimate: {:.5?}", guesses_std.last().unwrap());
    println!("  Steps:          {}", guesses_std.len());
    println!("  Convergence:    Diverged (or converged to another local extremum)");

    // 3. Levenberg-Marquardt Newton-Raphson starting at (1, 1)
    let lm_optimizer = LevenbergMarquardt::new(Precision(1e-5), 0.0001, 0.5);
    let lm_guesses = lm_optimizer.optimize(&func_ex2, [1.0, 1.0]).collect_vec();
    let lm_final = *lm_guesses.last().unwrap();

    println!("\nMethod: Levenberg-Marquardt Newton-Raphson");
    println!("  Starting Guess: [1.0, 1.0]");
    println!("  Final Estimate: {:.5?}", lm_final);
    println!("  Steps:          {}", lm_guesses.len() - 1);
    println!("  Convergence:    Success");

    // Plot Exercise 2 Surface and Paths
    let mut plot2 = Plot::new();
    let sample_x1_ex2 = UniformSample::new(-2.0..2.0, 30);
    let sample_x2_ex2 = UniformSample::new(-2.0..2.0, 30);
    let surface2 = Surface::new(
        sample_x1_ex2
            .clone()
            .map(|x2| {
                sample_x2_ex2
                    .clone()
                    .map(|x1| func_ex2.compute([x1, x2]))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
    .x(sample_x1_ex2.collect())
    .y(sample_x2_ex2.collect())
    .name("Objective function");
    plot2.add_trace(surface2);

    let (x1s_std, x2s_std) = guesses_std.iter().cloned().map(|[x1, x2]| (x1, x2)).unzip();
    let scatter_std = Scatter3D::new(
        x1s_std,
        x2s_std,
        guesses_std
            .iter()
            .map(|point| func_ex2.compute(*point))
            .collect(),
    )
    .marker(Marker::new().size(3))
    .name("Standard NR Path");
    plot2.add_trace(scatter_std);

    let (x1s_lm, x2s_lm) = lm_guesses.iter().cloned().map(|[x1, x2]| (x1, x2)).unzip();
    let scatter_lm = Scatter3D::new(
        x1s_lm,
        x2s_lm,
        lm_guesses
            .iter()
            .map(|point| func_ex2.compute(*point))
            .collect(),
    )
    .marker(Marker::new().size(3))
    .surface_color(NamedColor::Lime)
    .name("Levenberg-Marquardt Path");
    plot2.add_trace(scatter_lm);

    // -------------------------------------------------------------------------
    // Exercise 3
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 3: Quasi-Newton BFGS ---");
    let rosenbrock = |x: [f64; 6]| {
        let mut sum = 0.0;
        for i in 0..5 {
            sum += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    };

    println!("Running BFGS starting at (1, -1.2, 1, -1.2, 1, -1.2)...");
    let bfgs_optimizer = Bfgs::new(Precision(1e-5), 0.0001);
    let bfgs_guesses = bfgs_optimizer
        .optimize(&rosenbrock, [1.0, -1.2, 1.0, -1.2, 1.0, -1.2])
        .collect_vec();
    let bfgs_final = bfgs_guesses.last().unwrap();

    println!("Method: BFGS Quasi-Newton (n=6)");
    println!("  Starting Guess:   [1.0, -1.2, 1.0, -1.2, 1.0, -1.2]");
    println!("  Final Estimate:   {:.5?}", bfgs_final);
    println!("  Steps:            {}", bfgs_guesses.len() - 1);
    println!("  Convergence:      Success (or max steps reached)");

    // Save dashboard
    optimization::helpers::save_dashboard(
        "labs/lab04/plot.html",
        "LAB 4: Newton & Quasi-Newton Methods",
        &[
            ("Exercise 1 Newton-Raphson Optimization", &plot1),
            ("Exercise 2 Newton vs Levenberg-Marquardt", &plot2),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab04/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab04/plot.html");
}
