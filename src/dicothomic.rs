use crate::optimizer::{Function, Optimizer};

pub struct Dicothomic {
    iterations: usize,
    starting_interval: f64,
}

impl Optimizer<1> for Dicothomic {
    fn optimize<F: Function<1>>(self, func: &F, starting_guess: [f64; 1]) -> [f64; 1] {
        [starting_guess[0]]
    }
}
