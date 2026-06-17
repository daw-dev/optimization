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
    ) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;
        (0..self.stopping_condition.0).map(move |_| {
            let gradient = func.gradient(self.difference);
            let hessian = func.hessian(self.difference);
            let gk = gradient.compute(guess);
            let fk = hessian.compute(guess);
            guess = (Column::new_column(guess)
                - (Matrix(fk).inverse().unwrap() ^ Column::new_column(gk)))
            .into_column();
            guess
        })
    }
}

impl<const N: usize> Optimizer<[f64; N], f64, [f64; N], [f64; N]> for NewtonRaphson<Precision> {
    fn optimize<F: crate::functions::Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;
        std::iter::repeat_with(move || {
            let gradient = func.gradient(self.difference);
            let hessian = func.hessian(self.difference);
            let gk = gradient.compute(guess);
            let fk = hessian.compute(guess);
            let next_guess = (Column::new_column(guess)
                - (Matrix(fk).inverse().unwrap() ^ Column::new_column(gk)))
            .into_column();
            if gk.map(|x| x * x).iter().sum::<f64>() < self.stopping_condition.0.powi(2) {
                return None;
            }
            guess = next_guess;
            Some(guess)
        })
        .flatten()
    }
}
