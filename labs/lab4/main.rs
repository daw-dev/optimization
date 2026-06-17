use optimization::{
    helpers::Precision, optimizer::Optimizer, quadratic::newton_raphson::NewtonRaphson,
};

fn main() {
    let func = |[x1, x2]: [f64; 2]| x1.powi(2) + 2.0 * x2.powi(2) + 0.2 * x1.powi(2) * x2.powi(4);

    let optimizer = NewtonRaphson::new(Precision(1e-5), 0.01);
    let guesses = optimizer.optimize(&func, [0.0, 1.0]);

    println!(
        "final guess is {:.5?} in {} steps",
        guesses.last().unwrap(),
        guesses.len()
    );

    println!("{guesses:?}");
}
