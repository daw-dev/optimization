use optimization::{
    dicothomic::Dicothomic, fibonacci::Fibonacci, golden::GoldenRatio, helpers::UniformSample,
    optimizer::Optimizer,
};
// use plotters::prelude::*;
use plotly::{Plot, Scatter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let func = |x| 8.0 * f64::exp(1.0 - x) + 7.0 * f64::ln(x);

    let mut plot = Plot::new();
    let x_sample = UniformSample::new(1.0..2.0, 100);
    let scatter = Scatter::new(
        x_sample.clone().collect(),
        x_sample.map(func.clone()).collect(),
    );
    plot.add_trace(scatter);
    plot.show_html("labs/lab1/plot.html");

    let iterations = Dicothomic::iterations_from_precision(0.23, 1.0..2.0);
    println!("{iterations} iterations needed");
    let optimizer = Dicothomic::with_iterations(iterations, 1.0..2.0);
    let min = optimizer.optimize(&func, 1.5f64);
    println!("{min}");

    let iterations = GoldenRatio::iterations_from_precision(0.23, 1.0..2.0);
    println!("{iterations} iterations needed");
    let optimizer = GoldenRatio::with_iterations(iterations, 1.0..2.0);
    let min = optimizer.optimize(&func, 1.5f64);
    println!("{min}");

    const N: usize = 4;
    type Fib = Fibonacci<N>;

    println!(
        "precision with {} iterations: {}",
        N,
        Fib::GAMMAS.iter().product::<f64>()
    );
    println!("gammas: {:?}", Fib::GAMMAS);
    let optimizer = Fib::new(1.0..2.0);
    let min = optimizer.optimize(&func, 1.5f64);
    println!("{min}");

    Ok(())
}
