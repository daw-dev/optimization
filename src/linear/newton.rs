use crate::{
    functions::{Derivative, Function},
    helpers::{Iterations, Precision},
    optimizer::Optimize,
};

#[derive(Debug, Clone)]
pub struct Newton<S> {
    stop_condition: S,
    diff_precision: f64,
}

impl<S> Newton<S> {
    pub fn new(stop_condition: S, diff_precision: f64) -> Self {
        Self {
            stop_condition,
            diff_precision,
        }
    }
}

impl<F: Function<f64, f64>> Optimize<&F, f64> for Newton<Iterations> {
    fn optimize(&self, func: &F, starting_guess: f64) -> impl Iterator<Item = f64> {
        let mut guess = starting_guess;
        std::iter::once(starting_guess).chain((0..self.stop_condition.0).map(move |_| {
            let deriv1 = func.derivative(self.diff_precision);
            let deriv2_source = func.derivative(self.diff_precision);
            let deriv2 = deriv2_source.derivative(self.diff_precision);
            guess = guess - deriv1.compute(guess) / deriv2.compute(guess);
            guess
        }))
    }
}

impl<F: Function<f64, f64>> Optimize<&F, f64> for Newton<Precision> {
    fn optimize(&self, func: &F, starting_guess: f64) -> impl Iterator<Item = f64> {
        let mut guess = starting_guess;
        std::iter::once(starting_guess).chain(std::iter::from_fn(move || {
            let deriv1 = func.derivative(self.diff_precision);
            let deriv2_source = func.derivative(self.diff_precision);
            let deriv2 = deriv2_source.derivative(self.diff_precision);
            let diff = deriv1.compute(guess) / deriv2.compute(guess);
            guess = guess - diff;
            if diff.abs() < self.stop_condition.0 {
                None
            } else {
                Some(guess)
            }
        }))
    }
}
