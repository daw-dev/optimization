use crate::{
    function::{Differentiate, Function},
    helpers::{Iterations, Precision},
    linalg::{Column, Matrix},
    optimizer::Optimize,
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

impl<'a, const N: usize, F: crate::function::Function<[f64; N], f64> + 'a> Optimize<&'a F, [f64; N]>
    for NewtonRaphson<Iterations>
{
    fn optimize(&self, func: &'a F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;
        std::iter::once(starting_guess).chain((0..self.stopping_condition.0).map(move |_| {
            let gradient = func.differentiate(self.difference);
            let hessian = gradient.differentiate(self.difference);
            let gk = gradient.compute(guess);
            let fk = hessian.compute(guess);
            guess = (Column::new_column(guess)
                - (Matrix(fk).inverse().unwrap() * Column::new_column(gk)))
            .into_column();
            guess
        }))
    }
}

impl<'a, const N: usize, F: crate::function::Function<[f64; N], f64> + 'a> Optimize<&'a F, [f64; N]>
    for NewtonRaphson<Precision>
{
    fn optimize(&self, func: &'a F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;
        std::iter::once(starting_guess).chain(std::iter::from_fn(move || {
            let gradient = func.differentiate(self.difference);
            let hessian = gradient.differentiate(self.difference);
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

pub struct LevenbergMarquardt<S> {
    stopping_criterion: S,
    difference: f64,
    initial_mu: f64,
}

impl<S> LevenbergMarquardt<S> {
    pub fn new(stopping_criterion: S, difference: f64, initial_mu: f64) -> Self {
        Self {
            stopping_criterion,
            difference,
            initial_mu,
        }
    }
}

impl<'a, const N: usize, F: crate::function::Function<[f64; N], f64> + 'a> Optimize<&'a F, [f64; N]>
    for LevenbergMarquardt<Iterations>
{
    fn optimize(&self, func: &'a F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;
        let mut mu = self.initial_mu;
        let mut count = 0;
        let limit = self.stopping_criterion.0;

        std::iter::once(starting_guess).chain(std::iter::from_fn(move || {
            if count >= limit {
                return None;
            }
            let gradient = func.differentiate(self.difference);
            let hessian = gradient.differentiate(self.difference);
            let gk = gradient.compute(guess);
            let mut trials = 0;
            while trials < 100 {
                let fk = hessian.compute(guess);
                let mut fk_lm = fk;
                for i in 0..N {
                    fk_lm[i][i] += mu;
                }

                if let Some(inv) = Matrix(fk_lm).inverse() {
                    let dk = -(inv * Column::new_column(gk));
                    let next_guess = (Column::new_column(guess) + dk).into_column();

                    if func.compute(next_guess) < func.compute(guess) {
                        guess = next_guess;
                        mu *= 0.5;
                        count += 1;
                        return Some(guess);
                    } else {
                        mu *= 2.0;
                    }
                } else {
                    mu *= 2.0;
                }
                trials += 1;
            }
            None
        }))
    }
}

impl<'a, const N: usize, F: crate::function::Function<[f64; N], f64> + 'a> Optimize<&'a F, [f64; N]>
    for LevenbergMarquardt<Precision>
{
    fn optimize(&self, func: &'a F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;
        let mut mu = self.initial_mu;

        std::iter::once(starting_guess).chain(std::iter::from_fn(move || {
            let gradient = func.differentiate(self.difference);
            let gk = gradient.compute(guess);
            let gk_norm = gk.iter().map(|x| x * x).sum::<f64>().sqrt();
            if gk_norm < self.stopping_criterion.0 {
                return None;
            }

            let hessian = gradient.differentiate(self.difference);
            let mut trials = 0;
            while trials < 100 {
                let fk = hessian.compute(guess);
                let mut fk_lm = fk;
                for i in 0..N {
                    fk_lm[i][i] += mu;
                }

                if let Some(inv) = Matrix(fk_lm).inverse() {
                    let dk = -(inv * Column::new_column(gk));
                    let next_guess = (Column::new_column(guess) + dk).into_column();

                    if func.compute(next_guess) < func.compute(guess) {
                        guess = next_guess;
                        mu *= 0.5;
                        return Some(guess);
                    } else {
                        mu *= 2.0;
                    }
                } else {
                    mu *= 2.0;
                }
                trials += 1;
            }
            None
        }))
    }
}
