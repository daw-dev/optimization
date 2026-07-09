use crate::optimizer::TryOptimize;

/// 0-1 Knapsack Problem definition for Dynamic Programming
pub struct DPProblem<const N: usize> {
    /// Values of the items
    pub item_values: [f64; N],
    /// Weights of the items (must be integers for indexing the DP table)
    pub item_weights: [usize; N],
    /// Maximum capacity of the knapsack
    pub knapsack_capacity: usize,
}

/// A search step inside the Dynamic Programming knapsack solver
#[derive(Clone, Debug)]
pub struct DPStep<const N: usize> {
    /// The current stage of the DP (number of items considered so far)
    pub current_item_stage: usize,
    /// 2D dynamic programming value table of size (num_items + 1) x (knapsack_capacity + 1).
    /// `dp_value_table[k][w]` stores the maximum value achievable considering first `k` items
    /// with capacity limit `w`.
    pub dp_value_table: Vec<Vec<f64>>,
    /// Maximum value found at the current stage
    pub max_value_found: f64,
    /// A mask representing which items were chosen in the optimal solution (only filled at the final step)
    pub chosen_items_mask: [bool; N],
}

impl<const N: usize> DPStep<N> {
    /// Create a new starting search step
    pub fn new(knapsack_capacity: usize) -> Self {
        Self {
            current_item_stage: 0,
            dp_value_table: vec![vec![0.0; knapsack_capacity + 1]; 1],
            max_value_found: 0.0,
            chosen_items_mask: [false; N],
        }
    }
}

pub struct DynamicProgramming;

impl<const N: usize> TryOptimize<DPProblem<N>, DPStep<N>, DPStep<N>> for DynamicProgramming {
    type Error = String;

    fn try_optimize(
        &self,
        problem: DPProblem<N>,
        starting_guess: DPStep<N>,
    ) -> impl Iterator<Item = Result<DPStep<N>, Self::Error>> {
        let knapsack_capacity = problem.knapsack_capacity;

        let mut current = starting_guess;
        // Ensure the dp_value_table has enough rows to store values for all items: (num_items + 1)
        if current.dp_value_table.len() < N + 1 {
            current.dp_value_table = vec![vec![0.0; knapsack_capacity + 1]; N + 1];
        }

        let mut current_item_stage = current.current_item_stage;

        std::iter::from_fn(move || {
            if current_item_stage >= N {
                return None; // DP table is fully computed
            }

            let item_idx = current_item_stage;
            let item_value = problem.item_values[item_idx];
            let item_weight = problem.item_weights[item_idx];

            // Fill row `current_item_stage + 1` of the DP table
            for current_capacity in 0..=knapsack_capacity {
                if item_weight <= current_capacity {
                    let take_item_val = current.dp_value_table[current_item_stage][current_capacity - item_weight] + item_value;
                    let skip_item_val = current.dp_value_table[current_item_stage][current_capacity];
                    current.dp_value_table[current_item_stage + 1][current_capacity] = if take_item_val > skip_item_val {
                        take_item_val
                    } else {
                        skip_item_val
                    };
                } else {
                    current.dp_value_table[current_item_stage + 1][current_capacity] = current.dp_value_table[current_item_stage][current_capacity];
                }
            }

            current_item_stage += 1;
            current.current_item_stage = current_item_stage;
            current.max_value_found = current.dp_value_table[current_item_stage][knapsack_capacity];

            // If we have considered all items, perform backtracking to recover the chosen items
            if current_item_stage == N {
                let mut backtracked_choices = [false; N];
                let mut remaining_capacity = knapsack_capacity;
                for backtrack_stage in (1..=N).rev() {
                    let prev_val = current.dp_value_table[backtrack_stage - 1][remaining_capacity];
                    let curr_val = current.dp_value_table[backtrack_stage][remaining_capacity];
                    if (curr_val - prev_val).abs() > 1e-7 {
                        backtracked_choices[backtrack_stage - 1] = true;
                        remaining_capacity -= problem.item_weights[backtrack_stage - 1];
                    }
                }
                current.chosen_items_mask = backtracked_choices;
            }

            Some(Ok(current.clone()))
        })
    }
}
