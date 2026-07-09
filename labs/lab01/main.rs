use optimization::{
    helpers::{Iterations, Precision, UniformSample},
    optimizer::{Optimize, TryOptimize},
    univariate::{dichotomy::Dichotomy, fibonacci::Fibonacci, golden::GoldenRatio, newton::Newton},
};
use plotly::{Plot, Scatter};

fn main() {
    println!("========================================");
    println!("  LAB 1: Line Search Methods");
    println!("========================================");

    println!("\n--- Exercise 1 ---");
    let func = |x| 8.0 * f64::exp(1.0 - x) + 7.0 * f64::ln(x);

    let mut plot_ex1 = Plot::new();
    let x_sample = UniformSample::new(1.0..2.0, 100);
    let scatter = Scatter::new(x_sample.clone().collect(), x_sample.map(func).collect())
        .name("8e^(1-x) + 7ln(x)");
    plot_ex1.add_trace(scatter);

    let iter_dich = Dichotomy::<()>::iterations_from_precision(0.23, &(1.0..2.0));
    println!(
        "Dichotomy: {} iterations needed for uncertainty 0.23",
        iter_dich
    );
    let opt_dich = Dichotomy::new(Iterations(iter_dich));
    let min_dich = opt_dich.try_solution(&func, 1.0..2.0);
    println!("  Minimizer: {:?}", min_dich);

    let iter_gold = GoldenRatio::<()>::iterations_from_precision(0.23, &(1.0..2.0));
    println!(
        "Golden Section: {} iterations needed for uncertainty 0.23",
        iter_gold
    );
    let opt_gold = GoldenRatio::new(Iterations(iter_gold));
    let min_gold = opt_gold.try_solution(&func, 1.0..2.0);
    println!("  Minimizer: {:?}", min_gold);

    const N: usize = 4;
    println!(
        "Fibonacci (n = {}): precision is {}",
        N,
        Fibonacci::<N>::GAMMAS.iter().product::<f64>()
    );
    let opt_fib = Fibonacci::<N>;
    let min_fib = opt_fib.try_solution(&func, 1.0..2.0);
    println!("  Minimizer: {:?}", min_fib);

    println!("\n--- Exercise 1 (Part 4): Unimodal Test Function ---");
    let func_unimodal = |x: f64| x.powi(4) - 3.0 * x.powi(2) + x + 1.0;

    let mut plot_ex1_part4 = Plot::new();
    let x_sample_unimodal = UniformSample::new(-1.0..3.0, 100);
    let scatter_unimodal = Scatter::new(
        x_sample_unimodal.clone().collect(),
        x_sample_unimodal.map(func_unimodal).collect(),
    )
    .name("x^4 - 3x^2 + x + 1");
    plot_ex1_part4.add_trace(scatter_unimodal);

    let min_d = Dichotomy::new(Precision(0.23)).try_solution(&func_unimodal, -1.0..3.0);
    println!("Dichotomy: {:?}", min_d);
    let min_g = GoldenRatio::new(Precision(0.23)).try_solution(&func_unimodal, -1.0..3.0);
    println!("Golden Section: {:?}", min_g);
    let min_f = Fibonacci::<4>.try_solution(&func_unimodal, -1.0..3.0);
    println!("Fibonacci: {:?}", min_f);

    println!("\n--- Exercise 2: Newton-Raphson Method ---");
    let func_quad = |x: f64| x.powi(2) - 4.0 * x + 2.0;
    let opt_newton = Newton::new(Precision(1e-5), 0.001);
    for &x0 in &[3.0, 6.0, 8.0, -15.0] {
        let guess = opt_newton.clone().solution(&func_quad, x0);
        println!("  x0 = {:^5} -> minimizer: {}", x0, guess);
    }

    println!("\nNewton-Raphson for f(x) = 0.5 * x^2 - sin(x):");
    let func_trig = |x: f64| 0.5 * x.powi(2) - x.sin();
    for &x0 in &[2.0, 3.0, 0.0] {
        let guess = opt_newton.clone().solution(&func_trig, x0);
        println!("  x0 = {:^5} -> minimizer: {}", x0, guess);
    }

    println!("\n--- Exercise 3: Function f(x) = arctan(x)^2 ---");
    let func_atan = |x: f64| x.atan().powi(2);
    let opt_dich_ex3 = Dichotomy::new(Precision(1e-5));
    let opt_newton_ex3 = Newton::new(Precision(1e-5), 0.001);

    println!("Case 1: interval [0.0, 0.5], x0 = 0.25");
    println!(
        "  Dichotomy: {:?}",
        opt_dich_ex3.try_solution(&func_atan, 0.0..0.5)
    );
    println!("  Newton: {:?}", opt_newton_ex3.solution(&func_atan, 0.25));

    println!("Case 2: interval [0.0, 3.0], x0 = 1.5");
    println!(
        "  Dichotomy: {:?}",
        opt_dich_ex3.try_solution(&func_atan, 0.0..3.0)
    );
    println!("  Newton (first 10 steps):");
    for (i, guess) in opt_newton_ex3
        .optimize(&func_atan, 1.5)
        .take(10)
        .enumerate()
    {
        println!("    Step {}: {}", i, guess);
    }
    println!("  Newton diverges!");

    optimization::helpers::save_dashboard(
        "labs/lab01/plot.html",
        "LAB 1: Line Search Methods",
        &[
            ("Exercise 1 Test Function", &plot_ex1),
            ("Exercise 1 Unimodal Test Function", &plot_ex1_part4),
        ],
    )
    .unwrap();
    println!("\nSaved plots to: labs/lab01/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab01/plot.html");
}
