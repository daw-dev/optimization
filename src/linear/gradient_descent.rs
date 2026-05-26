use std::array;

use crate::{
    functions::{Function, Gradient},
    helpers::{Average, Precision},
    linear::{dicothomic::Dicothomic, gradient_descent},
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
        let mut guesses = Vec::new();
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

pub struct SteepestGradientDescent<S> {
    gradient_precision: f64,
    stopping_criterion: S,
}

impl<S> SteepestGradientDescent<S> {
    pub fn new(gradient_precision: f64, stopping_criterion: S) -> Self {
        Self {
            gradient_precision,
            stopping_criterion,
        }
    }
}

impl<const N: usize> Optimizer<[f64; N], f64, [f64; N], Vec<[f64; N]>>
    for SteepestGradientDescent<Precision>
{
    fn optimize<F: Function<[f64; N], f64>>(
        self,
        func: &F,
        starting_guess: [f64; N],
    ) -> Vec<[f64; N]> {
        let mut guess = starting_guess;
        let mut guesses = Vec::new();
        loop {
            let gradient = func.gradient(self.gradient_precision);
            let computed = gradient.compute(guess);
            let norm: f64 = computed.iter().map(|x| x * x).sum();
            if norm < self.stopping_criterion.0.powi(2) {
                break guesses;
            }
            let line_search_func =
                |step: f64| func.compute(array::from_fn(|i| guess[i] - step * computed[i]));
            let optimizer = Dicothomic::new(Precision(0.001)).chain(Average);
            let step = optimizer.optimize(&line_search_func, 0.0..1.0).unwrap();
            for i in 0..N {
                guess[i] = guess[i] - step * computed[i];
            }
            guesses.push(guess);
        }
    }
}
