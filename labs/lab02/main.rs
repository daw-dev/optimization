use itertools::Itertools;
use optimization::{
    function::Function,
    helpers::{Precision, UniformSample},
    multivariate::gradient::{FixedStepGradientDescent, SteepestGradientDescent},
    optimizer::{Optimize, TryOptimize},
    univariate::dichotomy::Dichotomy,
};
use plotly::{Layout, Plot, Scatter3D, Surface, color::NamedColor, common::Marker};

fn main() {
    println!("========================================");
    println!("  LAB 2: Gradient Descent Methods");
    println!("========================================");

    // -------------------------------------------------------------------------
    // Exercise 1 (Part 1): Quadratic Function
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1: Quadratic Function ---");
    let func1 = |[x1, x2]: [f64; 2]| x1.powi(2) + x2.powi(2) - 1.5 * x1 * x2;
    let mut plot1 = Plot::new();
    plot1.set_layout(Layout::new().auto_size(true));

    let sample_x1 = UniformSample::new(-10.0..10.0, 20);
    let sample_x2 = UniformSample::new(-10.0..10.0, 20);
    let surface1 = Surface::new(
        sample_x1
            .clone()
            .map(|x2| {
                sample_x2
                    .clone()
                    .map(|x1| func1.compute([x1, x2]))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
    .x(sample_x1.collect())
    .y(sample_x2.collect())
    .name("f(x1, x2) = x1^2 + x2^2 - 1.5*x1*x2");
    plot1.add_trace(surface1);

    println!("Method: Fixed Step Gradient Descent");
    let opt_fixed1 = FixedStepGradientDescent::new(0.001, 0.1, Precision(1e-4));
    let guesses_fixed1 = opt_fixed1.optimize(&func1, [4.0, 8.0]).collect_vec();
    println!("  Starting Guess: [4.0, 8.0]");
    println!("  Final Guess:    {:.5?}", guesses_fixed1.last().unwrap());
    println!("  Steps:          {}", guesses_fixed1.len());
    println!("  Convergence:    Success");

    let (x1s_f1, x2s_f1) = guesses_fixed1
        .iter()
        .cloned()
        .map(|[x1, x2]| (x1, x2))
        .unzip();
    let scatter_fixed1 = Scatter3D::new(
        x1s_f1,
        x2s_f1,
        guesses_fixed1
            .iter()
            .map(|point| func1.compute(*point))
            .collect(),
    )
    .marker(Marker::new().size(3))
    .name("Fixed Step Path");
    plot1.add_trace(scatter_fixed1);

    println!("\nMethod: Steepest Gradient Descent");
    let opt_steep1 = SteepestGradientDescent::new(
        0.001,
        Dichotomy::new(Precision(1e-5)),
        0.0..2.0,
        Precision(1e-4),
    );
    let result_steep1 = opt_steep1
        .try_optimize(&func1, [6.0, 5.0])
        .process_results(|iter| iter.collect_vec());

    match result_steep1 {
        Ok(guesses_steep1) => {
            println!("  Starting Guess: [6.0, 5.0]");
            println!("  Final Guess:    {:.5?}", guesses_steep1.last().unwrap());
            println!("  Steps:          {}", guesses_steep1.len());
            println!("  Convergence:    Success");

            let (x1s_s1, x2s_s1) = guesses_steep1
                .iter()
                .cloned()
                .map(|[x1, x2]| (x1, x2))
                .unzip();
            let scatter_steep1 = Scatter3D::new(
                x1s_s1,
                x2s_s1,
                guesses_steep1
                    .iter()
                    .map(|point| func1.compute(*point))
                    .collect(),
            )
            .marker(Marker::new().size(3))
            .surface_color(NamedColor::Lime)
            .name("Steepest Path");
            plot1.add_trace(scatter_steep1);
        }
        Err(error) => println!("  Error: {error}"),
    }

    // -------------------------------------------------------------------------
    // Exercise 1 (Part 2): Non-Quadratic Function
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1 (Part 2): Non-Quadratic Function ---");
    let func2 = |[x1, x2]: [f64; 2]| {
        (1.0 - x1).powi(2) + (1.0 - x2).powi(2) + 0.5 * (2.0 * x2 - x1.powi(2)).powi(2)
    };
    let mut plot2 = Plot::new();
    plot2.set_layout(Layout::new().auto_size(true));

    let sample_x1 = UniformSample::new(-10.0..10.0, 20);
    let sample_x2 = UniformSample::new(-10.0..10.0, 20);
    let surface2 = Surface::new(
        sample_x1
            .clone()
            .map(|x2| {
                sample_x2
                    .clone()
                    .map(|x1| func2.compute([x1, x2]))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
    .x(sample_x1.collect())
    .y(sample_x2.collect())
    .name("f(x1,x2) = (1-x1)^2 + (1-x2)^2 + 0.5*(2x2-x1^2)^2");
    plot2.add_trace(surface2);

    println!("Method: Fixed Step Gradient Descent");
    let opt_fixed2 = FixedStepGradientDescent::new(0.001, 0.001, Precision(1e-4));
    let guesses_fixed2 = opt_fixed2.optimize(&func2, [6.0, 5.0]).collect_vec();
    println!("  Starting Guess: [6.0, 5.0]");
    println!("  Final Guess:    {:.5?}", guesses_fixed2.last().unwrap());
    println!("  Steps:          {}", guesses_fixed2.len());
    println!("  Convergence:    Success");

    let (x1s_f2, x2s_f2) = guesses_fixed2
        .iter()
        .cloned()
        .map(|[x1, x2]| (x1, x2))
        .unzip();
    let scatter_fixed2 = Scatter3D::new(
        x1s_f2,
        x2s_f2,
        guesses_fixed2
            .iter()
            .map(|point| func2.compute(*point))
            .collect(),
    )
    .marker(Marker::new().size(3))
    .name("Fixed Step Path");
    plot2.add_trace(scatter_fixed2);

    println!("\nMethod: Steepest Gradient Descent");
    let opt_steep2 = SteepestGradientDescent::new(
        0.001,
        Dichotomy::new(Precision(1e-5)),
        0.0..2.0,
        Precision(1e-4),
    );
    let result_steep2 = opt_steep2
        .try_optimize(&func2, [6.0, 5.0])
        .process_results(|iter| iter.collect_vec());

    match result_steep2 {
        Ok(guesses_steep2) => {
            println!("  Starting Guess: [6.0, 5.0]");
            println!("  Final Guess:    {:.5?}", guesses_steep2.last().unwrap());
            println!("  Steps:          {}", guesses_steep2.len());
            println!("  Convergence:    Success");

            let (x1s_s2, x2s_s2) = guesses_steep2
                .iter()
                .cloned()
                .map(|[x1, x2]| (x1, x2))
                .unzip();
            let scatter_steep2 = Scatter3D::new(
                x1s_s2,
                x2s_s2,
                guesses_steep2
                    .iter()
                    .map(|point| func2.compute(*point))
                    .collect(),
            )
            .marker(Marker::new().size(3))
            .surface_color(NamedColor::Lime)
            .name("Steepest Path");
            plot2.add_trace(scatter_steep2);
        }
        Err(error) => println!("  Error: {error}"),
    }

    // -------------------------------------------------------------------------
    // Exercise 2: Rosenbrock Function
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 2: Rosenbrock Function ---");
    let func3 = |[x1, x2]: [f64; 2]| (x1 - 1.0).powi(2) + 100.0 * (x1.powi(2) - x2).powi(2);
    let mut plot3 = Plot::new();
    plot3.set_layout(Layout::new().auto_size(true));

    let sample_x1 = UniformSample::new(-10.0..10.0, 20);
    let sample_x2 = UniformSample::new(-10.0..10.0, 20);
    let surface3 = Surface::new(
        sample_x1
            .clone()
            .map(|x2| {
                sample_x2
                    .clone()
                    .map(|x1| func3.compute([x1, x2]))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
    .x(sample_x1.collect())
    .y(sample_x2.collect())
    .name("Rosenbrock Surface");
    plot3.add_trace(surface3);

    println!("Method: Fixed Step Gradient Descent");
    println!("  Starting Guess: [-1.8, 2.0]");

    for step in [1e-1, 1e-2, 1e-3] {
        let opt_fixed3 = FixedStepGradientDescent::new(0.001, step, Precision(1e-4));
        let guesses_fixed3 = opt_fixed3.optimize(&func3, [-1.8, 2.0]).collect_vec();

        let is_diverged = guesses_fixed3
            .iter()
            .any(|g| g[0].is_nan() || g[0].is_infinite() || g[0].abs() > 1e6);
        let status = if is_diverged {
            "Diverged (or truncated)"
        } else {
            "Success"
        };

        println!("  Step Size: {}", step);
        println!("    Final Guess: {:.5?}", guesses_fixed3.last().unwrap());
        println!("    Steps:        {}", guesses_fixed3.len());
        println!("    Convergence:  {}", status);

        // Filter out extreme values to prevent plot distortion
        let plot_guesses: Vec<_> = guesses_fixed3
            .iter()
            .filter(|g| g[0].abs() < 20.0 && g[1].abs() < 20.0)
            .cloned()
            .collect();

        if !plot_guesses.is_empty() {
            let (x1s_f3, x2s_f3) = plot_guesses
                .iter()
                .cloned()
                .map(|[x1, x2]| (x1, x2))
                .unzip();
            let scatter_fixed3 = Scatter3D::new(
                x1s_f3,
                x2s_f3,
                plot_guesses
                    .iter()
                    .map(|point| func3.compute(*point))
                    .collect(),
            )
            .marker(Marker::new().size(2))
            .name(format!("Step size {}", step));
            plot3.add_trace(scatter_fixed3);
        }
    }

    // Save all plots to one dashboard HTML
    optimization::helpers::save_dashboard(
        "labs/lab02/plot.html",
        "LAB 2: Gradient Descent Methods",
        &[
            ("Exercise 1: Quadratic Function", &plot1),
            ("Exercise 1 (Part 2): Non-Quadratic Function", &plot2),
            ("Exercise 2: Rosenbrock Function Trajectories", &plot3),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab02/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab02/plot.html");
}
