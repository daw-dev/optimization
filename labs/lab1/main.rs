use optimization::{
    dicothomic::{self, Dicothomic},
    fibonacci::Fibonacci,
    golden::{self, GoldenRatio},
    helpers::UniformSample,
    optimizer::Optimizer,
};
use plotly::{Layout, Plot, Scatter, layout::{GridPattern, LayoutGrid}};

fn main() {
    let func = |x| 8.0 * f64::exp(1.0 - x) + 7.0 * f64::ln(x);

    let mut plot = Plot::new();
    plot.set_layout(
        Layout::new().grid(
            LayoutGrid::new()
                .rows(1)
                .columns(3)
                .pattern(GridPattern::Independent),
        ),
    );
    let x_sample = UniformSample::new(1.0..2.0, 100);
    let scatter = Scatter::new(
        x_sample.clone().collect(),
        x_sample.map(func.clone()).collect(),
    )
    .x_axis("x1")
    .y_axis("y1")
    .name(r"$8 \cdot e^{1 - x} + 7 \cdot \ln(x)$");
    plot.add_trace(scatter);

    let iterations = Dicothomic::iterations_from_precision(0.23, &(1.0..2.0));
    println!("{iterations} iterations needed");
    let optimizer = Dicothomic::new(dicothomic::StoppingCondition::Iterations(iterations));
    let min = optimizer.optimize(&func, 1.0..2.0);
    println!("{min:?}");

    let iterations = GoldenRatio::iterations_from_precision(0.23, &(1.0..2.0));
    println!("{iterations} iterations needed");
    let optimizer = GoldenRatio::new(golden::StoppingCondition::Iterations(iterations));
    let min = optimizer.optimize(&func, 1.0..2.0);
    println!("{min:?}");

    const N: usize = 4;

    println!(
        "precision with {} iterations: {}",
        N,
        Fibonacci::<N>::GAMMAS.iter().product::<f64>()
    );
    println!("gammas: {:?}", Fibonacci::<N>::GAMMAS);
    let optimizer = Fibonacci::<N>;
    let min = optimizer.optimize(&func, 1.0..2.0);
    println!("{min:?}");

    let func = |x: f64| {
        x.powi(4) - 3.0 * x.powi(2) + x
        // + 0.0 * ((0..rand::random_range(1..10000000)).sum::<usize>() as f64)
    };

    let x_sample = UniformSample::new(-1.0..3.0, 100);
    let scatter = Scatter::new(
        x_sample.clone().collect(),
        x_sample.map(func.clone()).collect(),
    )
    .x_axis("x2")
    .y_axis("y2")
    .name(r"$x^4-3x^2+x+0\cdot\sum_{i=0}^{rand(1, 1e7)})$");
    plot.add_trace(scatter);

    let guess = Dicothomic::new(dicothomic::StoppingCondition::Precision(0.23)).optimize(&func, -1.0..3.0);
    println!("{guess:?}");
    let guess = GoldenRatio::new(golden::StoppingCondition::Precision(0.23)).optimize(&func, -1.0..3.0);
    println!("{guess:?}");
    let guess = Fibonacci::<4>.optimize(&func, -1.0..3.0);
    println!("{guess:?}");

    plot.write_html("labs/lab1/plot.html");
}
