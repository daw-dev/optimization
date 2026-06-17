use crate::{
    functions::{Function, Gradient, Hessian},
    helpers::{Iterations, Precision},
    optimizer::Optimizer,
    quadratic::{Column, Matrix},
};

pub struct NewtonRaphson<S> {
    stopping_condition: S,
    difference: f64,
}

impl<S> NewtonRaphson<S> {
    pub fn new(stopping_condition: S, difference: f64) -> Self {
        Self {
            stopping_condition,
            difference,
        }
    }
}

impl<const N: usize> Optimizer<[f64; N], f64, [f64; N], [f64; N]> for NewtonRaphson<Iterations> {
    fn optimize<F: crate::functions::Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> [f64; N] {
        let mut guess = starting_guess;
        let gradient = func.gradient(self.difference);
        let hessian = func.hessian(self.difference);
        for _ in 0..self.stopping_condition.0 {
            let gk = gradient.compute(guess);
            let fk = hessian.compute(guess);
            guess = (Column::new_column(guess)
                - (Matrix(fk).inverse().unwrap() ^ Column::new_column(gk)))
            .into_column();
        }
        guess
    }
}

impl<const N: usize> Optimizer<[f64; N], f64, [f64; N], Vec<[f64; N]>>
    for NewtonRaphson<Precision>
{
    fn optimize<F: crate::functions::Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> Vec<[f64; N]> {
        let mut guess = starting_guess;
        let mut guesses = vec![guess];
        let gradient = func.gradient(self.difference);
        let hessian = func.hessian(self.difference);
        loop {
            let gk = gradient.compute(guess);
            let fk = hessian.compute(guess);
            let next_guess = (Column::new_column(guess)
                - (Matrix(fk).inverse().unwrap() ^ Column::new_column(gk)))
            .into_column();
            guesses.push(next_guess);
            if gk.map(|x| x * x).iter().sum::<f64>() < self.stopping_condition.0.powi(2) {
                return guesses;
            }
            guess = next_guess;
        }
    }
}
