use crate::{
    function::{Differentiate, Function},
    helpers::{Iterations, Precision},
    optimizer::{Optimize, TryOptimize},
};
use std::{array, ops::Range};

pub struct FixedStepGradientDescent<S> {
    gradient_precision: f64,
    step: f64,
    stopping_criterion: S,
}

impl<S> FixedStepGradientDescent<S> {
    pub fn new(gradient_precision: f64, step: f64, stopping_criterion: S) -> Self {
        Self {
            gradient_precision,
            step,
            stopping_criterion,
        }
    }
}

impl<const N: usize, F: crate::function::Function<[f64; N], f64>> Optimize<&F, [f64; N]>
    for FixedStepGradientDescent<Precision>
{
    fn optimize(&self, func: &F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;
        let gradient = func.differentiate(self.gradient_precision);

        std::iter::once(starting_guess).chain(std::iter::from_fn(move || {
            let computed = gradient.compute(guess);
            let norm: f64 = computed.iter().map(|x| x * x).sum();
            if norm < self.stopping_criterion.0.powi(2) {
                None
            } else {
                for i in 0..N {
                    guess[i] = guess[i] - self.step * computed[i];
                }
                Some(guess)
            }
        }))
    }
}

impl<const N: usize, F: crate::function::Function<[f64; N], f64>> Optimize<&F, [f64; N]>
    for FixedStepGradientDescent<Iterations>
{
    fn optimize(&self, func: &F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;

        std::iter::once(starting_guess).chain((1..self.stopping_criterion.0).map(move |_| {
            let gradient = func.differentiate(self.gradient_precision);
            let computed = gradient.compute(guess);
            for i in 0..N {
                guess[i] = guess[i] - self.step * computed[i];
            }
            guess
        }))
    }
}

pub struct SteepestGradientDescent<LS, S> {
    gradient_precision: f64,
    line_search: LS,
    line_search_starting_guess: Range<f64>,
    stopping_criterion: S,
}

impl<LS, S> SteepestGradientDescent<LS, S> {
    pub fn new(
        gradient_precision: f64,
        line_search: LS,
        line_search_starting_guess: Range<f64>,
        stopping_criterion: S,
    ) -> Self {
        Self {
            gradient_precision,
            line_search,
            line_search_starting_guess,
            stopping_criterion,
        }
    }
}

impl<const N: usize, F: Function<[f64; N], f64>, LS, Error> TryOptimize<&F, [f64; N]>
    for SteepestGradientDescent<LS, Precision>
where
    LS: for<'a> TryOptimize<&'a dyn Function<f64, f64>, Range<f64>, Error = Error>,
{
    type Error = Error;

    fn try_optimize(
        &self,
        func: &F,
        starting_guess: [f64; N],
    ) -> impl Iterator<Item = Result<[f64; N], Self::Error>> {
        let mut guess = starting_guess;
        let gradient = func.differentiate(self.gradient_precision);
        std::iter::once(Ok(starting_guess)).chain(std::iter::from_fn(move || {
            let computed = gradient.compute(guess);
            let norm: f64 = computed.iter().map(|x| x * x).sum();
            if norm < self.stopping_criterion.0.powi(2) {
                None
            } else {
                let line_search_func = move |step: f64| {
                    func.compute(array::from_fn(|i| guess[i] - step * computed[i]))
                };
                let step_range = match self
                    .line_search
                    .try_solution(&line_search_func, self.line_search_starting_guess.clone())
                {
                    Ok(step_range) => step_range,
                    Err(reason) => return Some(Err(reason)),
                };
                let step = (step_range.start + step_range.end) / 2.0;
                for i in 0..N {
                    guess[i] = guess[i] - step * computed[i];
                }
                Some(Ok(guess))
            }
        }))
    }
}

impl<const N: usize, F: Function<[f64; N], f64>, LS> Optimize<&F, [f64; N]>
    for SteepestGradientDescent<LS, Precision>
where
    LS: for<'a> Optimize<&'a dyn Function<f64, f64>, Range<f64>>,
{
    fn optimize(&self, func: &F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut guess = starting_guess;

        let gradient = func.differentiate(self.gradient_precision);

        let sqr_prec = self.stopping_criterion.0.powi(2);

        std::iter::from_fn(move || {
            let computed = gradient.compute(guess);
            let norm: f64 = computed.iter().map(|x| x * x).sum();
            if norm < sqr_prec {
                None
            } else {
                let line_search_func = move |step: f64| {
                    func.compute(array::from_fn(|i| guess[i] - step * computed[i]))
                };
                let step_range = self
                    .line_search
                    .solution(&line_search_func, self.line_search_starting_guess.clone());
                let step = (step_range.start + step_range.end) / 2.0;
                for i in 0..N {
                    guess[i] = guess[i] - step * computed[i];
                }
                Some(guess)
            }
        })
    }
}
