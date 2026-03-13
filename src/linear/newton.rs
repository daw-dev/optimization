use crate::{functions::{Derivative, Function}, optimizer::Optimizer};

#[derive(Debug, Clone)]
pub struct Newton {
    iterations: usize,
}

impl Newton {
    pub fn new(iterations: usize) -> Self {
        Self { iterations }
    }
}

impl Optimizer<f64, f64, f64> for Newton {
    fn optimize<F: Function<f64, f64>>(self, func: &F, starting_guess: f64) -> f64 {
        let deriv1 = func.derivative(0.001);
        let deriv2 = deriv1.derivative(0.001);
        let mut guess = starting_guess;
        for _ in 0..self.iterations {
            guess = guess - deriv1.compute(guess) / deriv2.compute(guess);
        }
        guess
    }
}
