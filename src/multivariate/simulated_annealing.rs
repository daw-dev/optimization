use crate::optimizer::TryOptimize;
use rand::RngExt;

/// Simulated Annealing optimization problem
pub struct SAProblem<F, const N: usize> {
    /// Cost function to minimize (often called energy function)
    pub cost_function: F,
    /// Starting temperature
    pub initial_temperature: f64,
    /// Multiplicative cooling rate parameter (usually in [0.8, 0.9999])
    pub cooling_rate: f64,
}

/// A search step inside the Simulated Annealing solver
#[derive(Clone, Debug)]
pub struct SAStep<const N: usize> {
    /// The current state/path of the tour
    pub current_tour: [usize; N],
    /// The cost of the current state/path
    pub current_cost: f64,
    /// The best state/path found during the search so far
    pub best_tour: [usize; N],
    /// The cost of the best state/path found so far
    pub best_cost: f64,
    /// The current search temperature
    pub temperature: f64,
}

impl<const N: usize> SAStep<N> {
    /// Create a new starting search step
    pub fn new(initial_tour: [usize; N], initial_cost: f64, initial_temperature: f64) -> Self {
        Self {
            current_tour: initial_tour,
            current_cost: initial_cost,
            best_tour: initial_tour,
            best_cost: initial_cost,
            temperature: initial_temperature,
        }
    }
}

pub struct SimulatedAnnealing;

impl<F, const N: usize> TryOptimize<SAProblem<F, N>, SAStep<N>, SAStep<N>> for SimulatedAnnealing
where
    F: Fn(&[usize; N]) -> f64 + Clone + 'static,
{
    type Error = String;

    fn try_optimize(
        &self,
        problem: SAProblem<F, N>,
        starting_guess: SAStep<N>,
    ) -> impl Iterator<Item = Result<SAStep<N>, Self::Error>> {
        let mut current = starting_guess;

        std::iter::from_fn(move || {
            let mut rng = rand::rng();
            if N <= 1 {
                return Some(Err("Tour length must be greater than 1".into()));
            }

            // Generate neighbor tour via 2-opt (reversing a segment between two random indices)
            let mut neighbor_tour = current.current_tour;
            let city_idx_1 = rng.random_range(0..N);
            let city_idx_2 = rng.random_range(0..N);
            if city_idx_1 != city_idx_2 {
                let (start, end) = if city_idx_1 < city_idx_2 {
                    (city_idx_1, city_idx_2)
                } else {
                    (city_idx_2, city_idx_1)
                };
                neighbor_tour[start..=end].reverse();
            }

            let neighbor_cost = (problem.cost_function)(&neighbor_tour);
            let cost_difference = neighbor_cost - current.current_cost;

            // Acceptance probability check (Metropolis criterion)
            let should_accept_move = if cost_difference < 0.0 {
                true // Always accept improvements
            } else {
                let random_probability: f64 = rng.random();
                random_probability < (-cost_difference / current.temperature).exp()
            };

            if should_accept_move {
                current.current_tour = neighbor_tour;
                current.current_cost = neighbor_cost;

                if neighbor_cost < current.best_cost {
                    current.best_cost = neighbor_cost;
                    current.best_tour = current.current_tour;
                }
            }

            // Apply exponential cooling schedule step
            current.temperature *= problem.cooling_rate;

            Some(Ok(current.clone()))
        })
    }
}
