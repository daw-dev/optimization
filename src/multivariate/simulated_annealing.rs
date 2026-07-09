use crate::optimizer::Optimize;
use rand::RngExt;

/// Simulated Annealing optimization problem
pub struct SAProblem<F, N> {
    /// Cost function to minimize (often called energy function)
    pub cost_function: F,
    /// Neighbor generation function
    pub neighbor_function: N,
}

#[derive(Clone, Copy)]
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

impl<F, N, State, Cost> Optimize<SAProblem<F, N>, State, State> for SimulatedAnnealing
where
    F: Fn(&State) -> Cost + 'static,
    N: Fn(&State) -> State + 'static,
    State: Clone + 'static,
    Cost: PartialOrd + Copy + Into<f64> + 'static,
{
    fn optimize(
        self,
        problem: SAProblem<F, N>,
        starting_guess: State,
    ) -> impl Iterator<Item = State> {
        let mut current_state = starting_guess;
        let mut current_cost = (problem.cost_function)(&current_state);
        let mut temperature = self.initial_temperature;
        let cooling_rate = self.cooling_rate;

        std::iter::from_fn(move || {
            let mut rng = rand::rng();

            let neighbor_state = (problem.neighbor_function)(&current_state);
            let neighbor_cost = (problem.cost_function)(&neighbor_state);
            let cost_difference = neighbor_cost.into() - current_cost.into();

            // Acceptance probability check (Metropolis criterion)
            let should_accept_move = if cost_difference < 0.0 {
                true // Always accept improvements
            } else {
                let random_probability: f64 = rng.random();
                random_probability < (-cost_difference / temperature).exp()
            };

            if should_accept_move {
                current_state = neighbor_state;
                current_cost = neighbor_cost;
            }

            // Apply exponential cooling schedule step
            temperature *= cooling_rate;

            Some(current_state.clone())
        })
    }
}
