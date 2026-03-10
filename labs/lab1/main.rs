use optimization::{dicothomic::Dicothomic, golden::GoldenRatio, optimizer::Optimizer};
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let func = |x| 8.0 * f64::exp(1.0 - x) + 7.0 * f64::ln(x);

    let root = BitMapBackend::new("labs/lab1/plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("y=8e^{1−x} +7 log(x)", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(1f32..2f32, 7.5f32..8.5f32)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (-100..=200)
                .map(|x| x as f32 / 100.0)
                .map(|x| (x, func(x as f64) as f32)),
            &RED,
        ))?
        .label("y = 8e^{1−x} +7 log(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    let iterations = Dicothomic::iterations_from_precision(0.23, 1.0..2.0);
    println!("{iterations} iterations needed");
    let optimizer = Dicothomic::with_iterations(iterations, 1.0..2.0);
    let [min] = optimizer.optimize(&func, [1.5f64]);
    println!("{min}");

    let iterations = GoldenRatio::iterations_from_precision(0.23, 1.0..2.0);
    println!("{iterations} iterations needed");
    let optimizer = GoldenRatio::with_iterations(iterations, 1.0..2.0);
    let [min] = optimizer.optimize(&func, [1.5f64]);
    println!("{min}");

    Ok(())
}
