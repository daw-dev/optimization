use crate::{
    functions::{Function, Gradient},
    helpers::{Iterations, Precision},
    optimizer::{Optimization, Optimizer, TryOptimization, TryOptimizationResult, TryOptimizer},
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

impl<const N: usize> Optimizer<[f64; N], f64, [f64; N], [f64; N]>
    for FixedStepGradientDescent<Precision>
{
    fn optimize<F: crate::functions::Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> impl crate::optimizer::OptimizationResult<Guess = [f64; N]> {
        let mut guess = starting_guess;
        Optimization::new(
            std::iter::once(starting_guess).chain(std::iter::from_fn(move || {
                let gradient = func.gradient(self.gradient_precision);
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
            })),
        )
    }
}

impl<const N: usize> Optimizer<[f64; N], f64, [f64; N], [f64; N]>
    for FixedStepGradientDescent<Iterations>
{
    fn optimize<F: crate::functions::Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> impl crate::optimizer::OptimizationResult<Guess = [f64; N]> {
        let mut guess = starting_guess;

        Optimization::new(std::iter::once(starting_guess).chain(
            (1..self.stopping_criterion.0).map(move |_| {
                let gradient = func.gradient(self.gradient_precision);
                let computed = gradient.compute(guess);
                for i in 0..N {
                    guess[i] = guess[i] - self.step * computed[i];
                }
                guess
            }),
        ))
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

impl<const N: usize, LS, Error> TryOptimizer<[f64; N], f64, [f64; N], Error>
    for SteepestGradientDescent<LS, Precision>
where
    LS: TryOptimizer<f64, f64, Range<f64>, Error> + Clone,
{
    fn try_optimize<F: Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> impl crate::optimizer::TryOptimizationResult<Guess = [f64; N], Error = Error> {
        let mut guess = starting_guess;
        let gradient = func.gradient(self.gradient_precision);
        TryOptimization::new(
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
                        .clone()
                        .try_optimize(&line_search_func, self.line_search_starting_guess.clone())
                        .guess()
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
            })),
        )
    }
}
