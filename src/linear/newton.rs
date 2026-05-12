use crate::{functions::{Derivative, Function}, helpers::{Iterations, Precision}, optimizer::Optimizer};

#[derive(Debug, Clone)]
pub struct Newton<S> {
    stop_condition: S,
    diff_precision: f64,
}

impl<S> Newton<S> {
    pub fn new(stop_condition: S, diff_precision: f64) -> Self {
        Self { stop_condition, diff_precision }
    }
}

impl Optimizer<f64, f64, f64> for Newton<Iterations> {
    fn optimize<F: Function<f64, f64>>(self, func: &F, starting_guess: f64) -> f64 {
        let deriv1 = func.derivative(self.diff_precision);
        let deriv2 = deriv1.derivative(self.diff_precision);
        let mut guess = starting_guess;
        for _ in 0..self.stop_condition.0 {
            guess = guess - deriv1.compute(guess) / deriv2.compute(guess);
        }
        guess
    }
}

impl Optimizer<f64, f64, f64> for Newton<Precision> {
    fn optimize<F: Function<f64, f64>>(self, func: &F, starting_guess: f64) -> f64 {
        let deriv1 = func.derivative(self.diff_precision);
        let deriv2 = deriv1.derivative(self.diff_precision);
        let mut guess = starting_guess;
        loop {
            let diff = deriv1.compute(guess) / deriv2.compute(guess);
            guess = guess - diff;
            if diff.abs() < self.stop_condition.0 {
                return guess;
            }
        }
    }
}
