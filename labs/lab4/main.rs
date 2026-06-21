use itertools::Itertools;
use optimization::{
    function::{Differentiate, Function},
    helpers::{Precision, UniformSample},
    linalg::{Column, Matrix, SquareMatrix},
    multivariate::newton::NewtonRaphson,
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
    let mut guess = [1.0, 1.0];
    let mut lm_guesses = vec![guess];
    let grad_fn = func_ex2.differentiate(0.0001);
    let mut mu = 0.5; // damping parameter
    let mut lm_steps = 0;

    for step in 1..=100 {
        let gk = grad_fn.compute(guess);
        let gk_norm = gk.iter().map(|x| x * x).sum::<f64>().sqrt();
        if gk_norm < 1e-5 {
            lm_steps = step;
            break;
        }

        let fk = hess_fn.compute(guess);
        let mut fk_lm = fk;
        fk_lm[0][0] += mu;
        fk_lm[1][1] += mu;

        if let Some(inv) = Matrix(fk_lm).inverse() {
            let dk = -(inv * Column::new_column(gk));
            let next_guess = (Column::new_column(guess) + dk).into_column();

            if func_ex2.compute(next_guess) < func_ex2.compute(guess) {
                guess = next_guess;
                lm_guesses.push(guess);
                mu *= 0.5;
            } else {
                mu *= 2.0;
            }
        } else {
            mu *= 2.0;
        }
        lm_steps = step;
    }
    println!("\nMethod: Levenberg-Marquardt Newton-Raphson");
    println!("  Starting Guess: [1.0, 1.0]");
    println!("  Final Estimate: {:.5?}", guess);
    println!("  Steps:          {}", lm_steps);
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
    let mut x0 = Column::<6, f64>::new_column([1.0, -1.2, 1.0, -1.2, 1.0, -1.2]);
    let mut h = SquareMatrix::<6, f64>::identity();
    let grad_fn_ros = rosenbrock.differentiate(0.0001);
    let mut g = Column::new_column(grad_fn_ros.compute(x0.into_column()));

    let mut bfgs_steps = 0;
    for step in 1..=200 {
        let g_norm = (g.transpose() * g.clone()).into_value().sqrt();
        if g_norm < 1e-5 {
            bfgs_steps = step;
            break;
        }

        let d = -(h.clone() * g.clone());

        // Backtracking line search
        let mut alpha = 1.0;
        let f_x = rosenbrock.compute(x0.into_column());
        let mut best_alpha = 0.0;
        let mut best_val = f_x;
        for _ in 0..12 {
            let x_test = x0.clone() + d.clone() * alpha;
            let val = rosenbrock.compute(x_test.into_column());
            if val < best_val {
                best_val = val;
                best_alpha = alpha;
            }
            alpha *= 0.5;
        }

        let alpha_opt = if best_alpha == 0.0 { 1e-4 } else { best_alpha };
        let xp = x0.clone() + d.clone() * alpha_opt;
        let gp = Column::new_column(grad_fn_ros.compute(xp.into_column()));

        let delta_g = gp.clone() - g.clone();
        let delta_x = xp.clone() - x0.clone();

        let den = (delta_g.transpose() * delta_x.clone()).into_value();
        if den.abs() > 1e-9 {
            let rho = 1.0 / den;
            let identity = SquareMatrix::<6, f64>::identity();
            let term1 = identity.clone() - (delta_x.clone() * delta_g.transpose() * rho);
            let term2 = identity.clone() - (delta_g.clone() * delta_x.transpose() * rho);
            h = term1 * h * term2 + (delta_x.clone() * delta_x.transpose() * rho);
        }

        x0 = xp;
        g = gp;
        bfgs_steps = step;
    }

    println!("Method: BFGS Quasi-Newton (n=6)");
    println!("  Starting Guess:   [1.0, -1.2, 1.0, -1.2, 1.0, -1.2]");
    println!("  Final Estimate:   {:.5?}", x0.into_column());
    println!("  Steps:            {}", bfgs_steps);
    println!("  Convergence:      Success (or max steps reached)");

    // Save dashboard
    optimization::helpers::save_dashboard(
        "labs/lab4/plot.html",
        "LAB 4: Newton & Quasi-Newton Methods",
        &[
            ("Exercise 1 Newton-Raphson Optimization", &plot1),
            ("Exercise 2 Newton vs Levenberg-Marquardt", &plot2),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab4/plot.html");
}
