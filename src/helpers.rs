use crate::functions::Function;
use crate::optimizer::{Optimization, Optimizer};
use std::ops::Range;

#[derive(Clone)]
pub struct Iterations(pub usize);

#[derive(Clone)]
pub struct Precision(pub f64);

#[derive(Clone)]
pub struct Average;

#[derive(Debug, Clone)]
pub struct UniformSample {
    range: Range<f64>,
    points: usize,
    current_point: usize,
}

impl UniformSample {
    pub fn new(range: Range<f64>, points: usize) -> Self {
        Self {
            range,
            points,
            current_point: 0,
        }
    }
}

impl Iterator for UniformSample {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        (self.current_point <= self.points).then(|| {
            let res = self.range.start
                + (self.range.end - self.range.start) * self.current_point as f64
                    / self.points as f64;
            self.current_point += 1;
            res
        })
    }
}

impl<X, Y> Optimizer<X, Y, Range<f64>, f64> for Average {
    fn optimize<F: Function<X, Y>>(
        self,
        _func: &F,
        starting_guess: Range<f64>,
    ) -> impl crate::optimizer::OptimizationResult<Guess = f64> {
        Optimization::new(std::iter::once((starting_guess.start + starting_guess.end) / 2.0))
    }
}
