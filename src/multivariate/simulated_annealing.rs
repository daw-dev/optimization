use crate::optimizer::Optimize;
use rand::RngExt;

/// Simulated Annealing optimization problem
pub struct SAProblem<F, N> {
    /// Cost function to minimize (often called energy function)
    pub cost_function: F,
    /// Neighbor generation function
    pub neighbor_function: N,
}

/// A search step inside the Simulated Annealing solver
#[derive(Clone, Debug)]
pub struct SAStep<State, Cost> {
    /// The current state of the search
    pub current_state: State,
    /// The cost of the current state
    pub current_cost: Cost,
    /// The best state found during the search so far
    pub best_state: State,
    /// The cost of the best state found so far
    pub best_cost: Cost,
    /// The current search temperature
    pub temperature: f64,
}

pub struct SimulatedAnnealing {
    /// Starting temperature
    pub initial_temperature: f64,
    /// Multiplicative cooling rate parameter (usually in [0.8, 0.9999])
    pub cooling_rate: f64,
}

impl SimulatedAnnealing {
    pub fn new(initial_temperature: f64, cooling_rate: f64) -> Self {
        Self {
            initial_temperature,
            cooling_rate,
        }
    }
}

impl<F, N, State, Cost> Optimize<SAProblem<F, N>, State, SAStep<State, Cost>> for SimulatedAnnealing
where
    F: Fn(&State) -> Cost + 'static,
    N: Fn(&State) -> State + 'static,
    State: Clone + 'static,
    Cost: PartialOrd + Copy + Into<f64> + 'static,
{
    fn optimize(
        &self,
        problem: SAProblem<F, N>,
        starting_guess: State,
    ) -> impl Iterator<Item = SAStep<State, Cost>> {
        let initial_cost = (problem.cost_function)(&starting_guess);
        let mut current = SAStep {
            current_state: starting_guess.clone(),
            current_cost: initial_cost,
            best_state: starting_guess,
            best_cost: initial_cost,
            temperature: self.initial_temperature,
        };
        let cooling_rate = self.cooling_rate;

        std::iter::from_fn(move || {
            let mut rng = rand::rng();

            let neighbor_state = (problem.neighbor_function)(&current.current_state);
            let neighbor_cost = (problem.cost_function)(&neighbor_state);
            let cost_difference = neighbor_cost.into() - current.current_cost.into();

            // Acceptance probability check (Metropolis criterion)
            let should_accept_move = if cost_difference < 0.0 {
                true // Always accept improvements
            } else {
                let random_probability: f64 = rng.random();
                random_probability < (-cost_difference / current.temperature).exp()
            };

            if should_accept_move {
                current.current_state = neighbor_state;
                current.current_cost = neighbor_cost;

                if neighbor_cost < current.best_cost {
                    current.best_cost = neighbor_cost;
                    current.best_state = current.current_state.clone();
                }
            }

            // Apply exponential cooling schedule step
            current.temperature *= cooling_rate;

            Some(current.clone())
        })
    }
}
