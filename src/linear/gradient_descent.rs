use std::{array, ops::Range};
use crate::{
    functions::{Function, Gradient},
    helpers::{Iterations, Precision},
    optimizer::Optimizer,
};

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

impl<const N: usize> Optimizer<[f64; N], f64, [f64; N]> for FixedStepGradientDescent<Precision> {
    fn optimize<F: crate::functions::Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> [f64; N] {
        Optimizer::<_, _, _, Vec<[f64; N]>>::optimize(self, func, starting_guess)
            .pop()
            .unwrap()
    }
}

impl<const N: usize> Optimizer<[f64; N], f64, [f64; N], Vec<[f64; N]>>
    for FixedStepGradientDescent<Precision>
{
    fn optimize<F: crate::functions::Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> Vec<[f64; N]> {
        let mut guess = starting_guess;
        let mut guesses = vec![starting_guess];
        loop {
            let gradient = func.gradient(self.gradient_precision);
            let computed = gradient.compute(guess);
            let norm: f64 = computed.iter().map(|x| x * x).sum();
            if norm < self.stopping_criterion.0.powi(2) {
                break guesses;
            }
            for i in 0..N {
                guess[i] = guess[i] - self.step * computed[i];
            }
            guesses.push(guess);
        }
    }
}

impl<const N: usize> Optimizer<[f64; N], f64, [f64; N], Vec<[f64; N]>>
    for FixedStepGradientDescent<Iterations>
{
    fn optimize<F: crate::functions::Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> Vec<[f64; N]> {
        let mut guess = starting_guess;
        let mut guesses = Vec::with_capacity(self.stopping_criterion.0);
        guesses.push(starting_guess);

        for _ in 1..self.stopping_criterion.0 {
            let gradient = func.gradient(self.gradient_precision);
            let computed = gradient.compute(guess);
            for i in 0..N {
                guess[i] = guess[i] - self.step * computed[i];
            }
            guesses.push(guess);
        }

        guesses
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

impl<const N: usize, LS>
    Optimizer<[f64; N], f64, [f64; N], Result<Vec<[f64; N]>, (String, Vec<[f64; N]>)>>
    for SteepestGradientDescent<LS, Precision>
where
    LS: Optimizer<f64, f64, Range<f64>, Result<f64, String>> + Clone,
{
    fn optimize<F: Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> Result<Vec<[f64; N]>, (String, Vec<[f64; N]>)> {
        let mut guess = starting_guess;
        let mut guesses = vec![starting_guess];
        let gradient = func.gradient(self.gradient_precision);
        loop {
            let computed = gradient.compute(guess);
            let norm: f64 = computed.iter().map(|x| x * x).sum();
            if norm < self.stopping_criterion.0.powi(2) {
                break Ok(guesses);
            }
            let line_search_func =
                |step: f64| func.compute(array::from_fn(|i| guess[i] - step * computed[i]));
            let step = match self
                .line_search
                .clone()
                .optimize(&line_search_func, self.line_search_starting_guess.clone())
            {
                Ok(step) => step,
                Err(reason) => return Err((reason, guesses)),
            };
            for i in 0..N {
                guess[i] = guess[i] - step * computed[i];
            }
            guesses.push(guess);
        }
    }
}
