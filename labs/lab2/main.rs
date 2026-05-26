use optimization::{
    functions::Function,
    helpers::{Precision, UniformSample},
    linear::gradient_descent::{FixedStepGradientDescent, SteepestGradientDescent},
    optimizer::Optimizer,
};
use plotly::{Layout, Plot, Scatter3D, Surface, color::NamedColor, common::Marker};

fn main() {
    let func = |[x1, x2]: [f64; 2]| x1.powi(2) + x2.powi(2) - 1.5 * x1 * x2;
    let mut plot = Plot::new();
    plot.set_layout(Layout::new().height(800).auto_size(true));
    let sample_x1 = UniformSample::new(-10.0..10.0, 20);
    let sample_x2 = UniformSample::new(-10.0..10.0, 20);
    let scatter = Surface::new(
        sample_x1
            .clone()
            .map(|x1| {
                sample_x2
                    .clone()
                    .map(|x2| func.compute([x1, x2]))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
    .x(sample_x1.collect())
    .y(sample_x2.collect());
    plot.add_trace(scatter);

    println!("FixedStepGradientDescent:");

    let optimizer = FixedStepGradientDescent::new(0.001, 0.1, Precision(1e-4));

    let guesses: Vec<[f64; 2]> = optimizer.optimize(&func, [4.0, 8.0]);

    println!(
        "final guess is {:.5?} in {} steps",
        guesses.last().unwrap(),
        guesses.len()
    );

    let (x1s, x2s) = guesses.iter().cloned().map(|[x1, x2]| (x1, x2)).unzip();

    let scatter = Scatter3D::new(
        x1s,
        x2s,
        guesses.iter().map(|point| func.compute(*point)).collect(),
    )
    .marker(Marker::new().size(2));

    plot.add_trace(scatter);

    println!("SteepestGradientDescent:");

    let optimizer = SteepestGradientDescent::new(0.001, Precision(1e-4));

    let guesses: Vec<[f64; 2]> = optimizer.optimize(&func, [4.0, 8.0]);

    println!(
        "final guess is {:.5?} in {} steps",
        guesses.last().unwrap(),
        guesses.len()
    );

    let (x1s, x2s) = guesses.iter().cloned().map(|[x1, x2]| (x1, x2)).unzip();

    let scatter = Scatter3D::new(
        x1s,
        x2s,
        guesses.iter().map(|point| func.compute(*point)).collect(),
    )
    .marker(Marker::new().size(2))
    .surface_color(NamedColor::Lime);

    plot.add_trace(scatter);

    plot.write_html("labs/lab2/plot1.html");

    let func = |[x1, x2]: [f64; 2]| {
        (1.0 - x1).powi(2) + (1.0 - x2).powi(2) + 0.5 * (2.0 * x2 - x1.powi(2)).powi(2)
    };
    let mut plot = Plot::new();
    plot.set_layout(Layout::new().height(800).auto_size(true));
    let sample_x1 = UniformSample::new(-10.0..10.0, 20);
    let sample_x2 = UniformSample::new(-10.0..10.0, 20);
    let scatter = Surface::new(
        sample_x1
            .clone()
            .map(|x1| {
                sample_x2
                    .clone()
                    .map(|x2| func.compute([x1, x2]))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
    .x(sample_x1.collect())
    .y(sample_x2.collect());
    plot.add_trace(scatter);

    println!("FixedStepGradientDescent:");

    let optimizer = FixedStepGradientDescent::new(0.001, 0.1, Precision(1e-4));

    let guesses: Vec<[f64; 2]> = optimizer.optimize(&func, [4.0, 8.0]);

    println!(
        "final guess is {:.5?} in {} steps",
        guesses.last().unwrap(),
        guesses.len()
    );

    let (x1s, x2s) = guesses.iter().cloned().map(|[x1, x2]| (x1, x2)).unzip();

    let scatter = Scatter3D::new(
        x1s,
        x2s,
        guesses.iter().map(|point| func.compute(*point)).collect(),
    )
    .marker(Marker::new().size(2));

    plot.add_trace(scatter);

    println!("SteepestGradientDescent:");

    let optimizer = SteepestGradientDescent::new(0.001, Precision(1e-4));

    let guesses: Vec<[f64; 2]> = optimizer.optimize(&func, [4.0, 8.0]);

    println!(
        "final guess is {:.5?} in {} steps",
        guesses.last().unwrap(),
        guesses.len()
    );

    let (x1s, x2s) = guesses.iter().cloned().map(|[x1, x2]| (x1, x2)).unzip();

    let scatter = Scatter3D::new(
        x1s,
        x2s,
        guesses.iter().map(|point| func.compute(*point)).collect(),
    )
    .marker(Marker::new().size(2))
    .surface_color(NamedColor::Lime);

    plot.add_trace(scatter);
    plot.write_html("labs/lab2/plot2.html");
}
