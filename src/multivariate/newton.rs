use crate::{
    function::{Function, Gradient, Hessian},
    helpers::{Iterations, Precision},
    optimizer::Optimize,
    linalg::{Column, Matrix},
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

impl<const N: usize, F: crate::function::Function<[f64; N], f64>> Optimize<&F, [f64; N]>
    for NewtonRaphson<Iterations>
{
    fn optimize(&self, func: &F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;
        std::iter::once(starting_guess).chain((0..self.stopping_condition.0).map(move |_| {
            let gradient = func.gradient(self.difference);
            let hessian = func.hessian(self.difference);
            let gk = gradient.compute(guess);
            let fk = hessian.compute(guess);
            guess = (Column::new_column(guess)
                - (Matrix(fk).inverse().unwrap() * Column::new_column(gk)))
            .into_column();
            guess
        }))
    }
}

impl<const N: usize, F: crate::function::Function<[f64; N], f64>> Optimize<&F, [f64; N]>
    for NewtonRaphson<Precision>
{
    fn optimize(&self, func: &F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;
        std::iter::once(starting_guess).chain(std::iter::from_fn(move || {
            let gradient = func.gradient(self.difference);
            let hessian = func.hessian(self.difference);
            let gk = gradient.compute(guess);
            let fk = hessian.compute(guess);
            let next_guess = (Column::new_column(guess)
                - (Matrix(fk).inverse().unwrap() * Column::new_column(gk)))
            .into_column();
            if gk.map(|x| x * x).iter().sum::<f64>() < self.stopping_condition.0.powi(2) {
                None
            } else {
                guess = next_guess;
                Some(guess)
            }
        }))
    }
}
