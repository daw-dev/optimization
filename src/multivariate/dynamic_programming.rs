use crate::optimizer::Optimize;

/// A generic Dynamic Programming problem definition
pub struct DPProblem<F, B> {
    /// The size of the state space per stage
    pub state_space_size: usize,
    /// Transition function to compute row values: (stage_idx, state_idx, prev_row) -> value
    pub transition_function: F,
    /// Backtracking function to recover decisions: (stage, current_state_idx, dp_table) -> Decision
    pub backtrack_function: B,
}

/// A search step inside the generic Dynamic Programming solver
#[derive(Clone, Debug)]
pub struct DPStep<Decision, const N: usize> {
    /// The current stage of the DP (number of items/stages considered so far)
    pub current_stage: usize,
    /// 2D dynamic programming value table of size (num_stages + 1) x state_space_size
    pub dp_table: Vec<Vec<f64>>,
    /// Maximum value found at the current stage
    pub max_value_found: f64,
    /// The optimal decisions recovered via backtracking (filled at the final stage)
    pub optimal_decisions: Option<[Decision; N]>,
}

pub struct DynamicProgramming;

impl<F, B, Decision, const N: usize> Optimize<DPProblem<F, B>, (), DPStep<Decision, N>> for DynamicProgramming
where
    F: Fn(usize, usize, &[f64]) -> f64 + 'static,
    B: Fn(usize, &mut usize, &[Vec<f64>]) -> Decision + 'static,
    Decision: Default + Clone + Copy + 'static,
{
    fn optimize(
        &self,
        problem: DPProblem<F, B>,
        _starting_guess: (),
    ) -> impl Iterator<Item = DPStep<Decision, N>> {
        let state_space_size = problem.state_space_size;

        let mut current = DPStep {
            current_stage: 0,
            dp_table: vec![vec![0.0; state_space_size]; N + 1],
            max_value_found: 0.0,
            optimal_decisions: None,
        };

        let mut current_stage = 0;

        std::iter::from_fn(move || {
            if current_stage >= N {
                return None; // DP table is fully computed
            }

            // Fill row `current_stage + 1` of the DP table
            for s in 0..state_space_size {
                let prev_row = &current.dp_table[current_stage];
                current.dp_table[current_stage + 1][s] = (problem.transition_function)(current_stage, s, prev_row);
            }

            current_stage += 1;
            current.current_stage = current_stage;
            current.max_value_found = current.dp_table[current_stage][state_space_size - 1];

            // If we have considered all stages, perform backtracking to recover optimal decisions
            if current_stage == N {
                let mut decisions = [Decision::default(); N];
                let mut current_state = state_space_size - 1;
                for stage in (1..=N).rev() {
                    decisions[stage - 1] = (problem.backtrack_function)(stage, &mut current_state, &current.dp_table);
                }
                current.optimal_decisions = Some(decisions);
            }

            Some(current.clone())
        })
    }
}
