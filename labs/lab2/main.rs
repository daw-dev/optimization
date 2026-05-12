use optimization::{functions::Function, helpers::UniformSample};
use plotly::{Plot, Surface};

fn main() {
    let func = |[x1, x2]: [f64; 2]| x1.powi(2) + x2.powi(2) - 1.5 * x1 * x2;
    let mut plot = Plot::new();
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

    

    plot.write_html("labs/lab2/plot.html");
}
