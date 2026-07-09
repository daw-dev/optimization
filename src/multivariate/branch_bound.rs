use crate::optimizer::Optimize;

/// Mixed-Integer Linear Program (MILP) Problem
pub struct MILPProblem<const N: usize, const M: usize> {
    /// Coefficients of the objective function to minimize (minimize c^T * x)
    pub objective_coeffs: [f64; N],
    /// Constraint coefficient matrix (A * x <= b)
    pub constraint_matrix: [[f64; N]; M],
    /// Right-hand side bounds vector for constraints (A * x <= b)
    pub constraint_rhs: [f64; M],
}

/// A search step inside the Branch and Bound algorithm
#[derive(Clone, Debug)]
pub struct BBStep<const N: usize> {
    /// Best integer solution vector found so far
    pub best_x: Option<[f64; N]>,
    /// Objective function value of the best integer solution found so far
    pub best_y: f64,
    /// Number of active unexplored nodes remaining in the search stack
    pub active_nodes: usize,
    /// The bounds of the current subproblem: Some((lower_bounds, upper_bounds))
    pub current_node: Option<([f64; N], [f64; N])>,
}

impl<const N: usize> BBStep<N> {
    /// Create a new starting search step
    pub fn new() -> Self {
        Self {
            best_x: None,
            best_y: f64::INFINITY,
            active_nodes: 0,
            current_node: None,
        }
    }
}

pub struct BranchAndBound;

/// Solves the LP relaxation: Minimize objective_coeffs^T * x subject to constraint_matrix * x <= constraint_rhs,
/// and lower_bounds <= x <= upper_bounds.
/// This implementation assumes all coefficients in constraint_matrix are non-negative.
pub fn solve_lp<const N: usize, const M: usize>(
    objective_coeffs: &[f64; N],
    constraint_matrix: &[[f64; N]; M],
    constraint_rhs: &[f64; M],
    lower_bounds: &[f64; N],
    upper_bounds: &[f64; N],
) -> Option<(f64, [f64; N])> {
    // 1. Detect fixed variables and check bounds feasibility
    let mut fixed_values = [None; N];
    let mut free_variable_indices = [0; N];
    let mut num_free = 0;
    for i in 0..N {
        if lower_bounds[i] > upper_bounds[i] + 1e-7 {
            return None; // Infeasible bound range
        }
        if (lower_bounds[i] - upper_bounds[i]).abs() < 1e-7 {
            fixed_values[i] = Some(lower_bounds[i]);
        } else {
            fixed_values[i] = None;
            free_variable_indices[num_free] = i;
            num_free += 1;
        }
    }

    // 2. Reduce constraints by substituting fixed variables
    // Original: constraint_matrix * x <= constraint_rhs
    // Reduced: reduced_matrix * x_free <= constraint_rhs - constraint_matrix_fixed * x_fixed
    let mut reduced_rhs = vec![0.0; M + num_free];
    let mut reduced_matrix = vec![vec![0.0; num_free]; M + num_free];

    for row_idx in 0..M {
        let mut rhs_val = constraint_rhs[row_idx];
        for var_idx in 0..N {
            if let Some(fixed_val) = fixed_values[var_idx] {
                rhs_val -= constraint_matrix[row_idx][var_idx] * fixed_val;
            }
        }
        if rhs_val < -1e-7 {
            // Since coefficients and variables are positive, a negative RHS is infeasible.
            return None;
        }
        reduced_rhs[row_idx] = rhs_val;
        for (col_idx, &orig_idx) in free_variable_indices[..num_free].iter().enumerate() {
            reduced_matrix[row_idx][col_idx] = constraint_matrix[row_idx][orig_idx];
        }
    }

    // Add simple upper bound constraints: x_j <= 1 for free variables
    for (col_idx, &_orig_idx) in free_variable_indices[..num_free].iter().enumerate() {
        let constraint_row = M + col_idx;
        reduced_rhs[constraint_row] = 1.0;
        reduced_matrix[constraint_row][col_idx] = 1.0;
    }

    // Calculate objective constant offset contributed by fixed variables
    let mut objective_constant_offset = 0.0;
    for var_idx in 0..N {
        if let Some(fixed_val) = fixed_values[var_idx] {
            objective_constant_offset += objective_coeffs[var_idx] * fixed_val;
        }
    }

    // Prepare objective coefficients for free variables
    let mut reduced_objective = vec![0.0; num_free];
    for (col_idx, &orig_idx) in free_variable_indices[..num_free].iter().enumerate() {
        reduced_objective[col_idx] = objective_coeffs[orig_idx];
    }

    // If there are no free variables left, the solution is fully determined by bounds
    if num_free == 0 {
        let mut x_sol = [0.0; N];
        for i in 0..N {
            x_sol[i] = fixed_values[i].unwrap();
        }
        return Some((objective_constant_offset, x_sol));
    }

    // 3. Build the Simplex Tableau
    // M constraints and N free variables
    let total_constraints = M + num_free;
    let total_free_vars = num_free;
    // Columns: variables + slack variables + RHS column
    let num_tableau_cols = total_free_vars + total_constraints + 1;
    let mut tableau = vec![vec![0.0; num_tableau_cols]; total_constraints + 1];

    // Track which variable is basic for each constraint row.
    // Row i initially has slack variable total_free_vars + i.
    let mut basis = (0..total_constraints)
        .map(|i| total_free_vars + i)
        .collect::<Vec<usize>>();

    for i in 0..total_constraints {
        for j in 0..total_free_vars {
            tableau[i][j] = reduced_matrix[i][j];
        }
        tableau[i][total_free_vars + i] = 1.0;
        tableau[i][num_tableau_cols - 1] = reduced_rhs[i];
    }

    // Set objective row in tableau (negated for minimization setup)
    for j in 0..total_free_vars {
        tableau[total_constraints][j] = -reduced_objective[j];
    }

    // 4. Run Simplex Iterations
    let mut iteration = 0;
    while iteration < 2000 {
        iteration += 1;

        // Find entering variable (largest positive coefficient in objective row)
        let mut entering_col_idx = 0;
        let mut max_coeff = 0.0;
        for col_idx in 0..(total_free_vars + total_constraints) {
            if tableau[total_constraints][col_idx] > max_coeff {
                max_coeff = tableau[total_constraints][col_idx];
                entering_col_idx = col_idx;
            }
        }

        if max_coeff <= 1e-7 {
            // Optimal solution reached!
            break;
        }

        // Ratio test to find leaving basic variable
        let mut leaving_row_idx = None;
        let mut min_ratio = f64::MAX;
        for row_idx in 0..total_constraints {
            if tableau[row_idx][entering_col_idx] > 1e-7 {
                let ratio = tableau[row_idx][num_tableau_cols - 1] / tableau[row_idx][entering_col_idx];
                if ratio < min_ratio {
                    min_ratio = ratio;
                    leaving_row_idx = Some(row_idx);
                }
            }
        }

        let leaving_row_idx = match leaving_row_idx {
            Some(idx) => idx,
            None => return None, // Problem is unbounded
        };

        // Pivot on (leaving_row_idx, entering_col_idx)
        let pivot_value = tableau[leaving_row_idx][entering_col_idx];
        for col_idx in 0..num_tableau_cols {
            tableau[leaving_row_idx][col_idx] /= pivot_value;
        }

        for row_idx in 0..=total_constraints {
            if row_idx != leaving_row_idx {
                let factor = tableau[row_idx][entering_col_idx];
                for col_idx in 0..num_tableau_cols {
                    tableau[row_idx][col_idx] -= factor * tableau[leaving_row_idx][col_idx];
                }
            }
        }

        basis[leaving_row_idx] = entering_col_idx;
    }

    // Extract values for free variables
    let mut free_variable_solutions = vec![0.0; total_free_vars];
    for row_idx in 0..total_constraints {
        let var_idx = basis[row_idx];
        if var_idx < total_free_vars {
            free_variable_solutions[var_idx] = tableau[row_idx][num_tableau_cols - 1];
        }
    }

    // Reconstruct full solution vector
    let mut full_variable_solutions = [0.0; N];
    for var_idx in 0..N {
        if let Some(fixed_val) = fixed_values[var_idx] {
            full_variable_solutions[var_idx] = fixed_val;
        } else {
            let free_idx = free_variable_indices[..num_free]
                .iter()
                .position(|&idx| idx == var_idx)
                .unwrap();
            full_variable_solutions[var_idx] = free_variable_solutions[free_idx];
        }
    }

    let optimal_lp_value = tableau[total_constraints][num_tableau_cols - 1] + objective_constant_offset;
    Some((optimal_lp_value, full_variable_solutions))
}

impl<const N: usize, const M: usize> Optimize<MILPProblem<N, M>, (), BBStep<N>> for BranchAndBound {
    fn optimize(
        &self,
        problem: MILPProblem<N, M>,
        _starting_guess: (),
    ) -> impl Iterator<Item = BBStep<N>> {
        let mut stack = Vec::new();
        stack.push(([0.0; N], [1.0; N]));

        let mut current_step = BBStep::<N>::new();

        std::iter::from_fn(move || {
            let (lower_bounds, upper_bounds) = match stack.pop() {
                Some(node) => node,
                None => return None, // Search completed
            };

            current_step.active_nodes = stack.len();
            current_step.current_node = Some((lower_bounds.clone(), upper_bounds.clone()));

            let lp_res = solve_lp(
                &problem.objective_coeffs,
                &problem.constraint_matrix,
                &problem.constraint_rhs,
                &lower_bounds,
                &upper_bounds,
            );

            match lp_res {
                None => {
                    // Infeasible node, prune branch
                    Some(current_step.clone())
                }
                Some((relaxed_objective_val, relaxed_variable_values)) => {
                    // Prune by bound (relaxed objective val is worse than best integer solution found)
                    if relaxed_objective_val >= current_step.best_y - 1e-7 {
                        Some(current_step.clone())
                    } else {
                        // Find the most fractional variable index
                        let mut split_variable_idx = None;
                        let mut min_dist_from_half = f64::MAX;

                        for var_idx in 0..N {
                            let distance_to_half = (relaxed_variable_values[var_idx] - 0.5).abs();
                            if distance_to_half < min_dist_from_half {
                                min_dist_from_half = distance_to_half;
                                split_variable_idx = Some(var_idx);
                            }
                        }

                        // If distance to half is > 0.499, the variables are practically integer
                        if min_dist_from_half > 0.499 {
                            current_step.best_y = relaxed_objective_val;
                            current_step.best_x = Some(relaxed_variable_values);
                            Some(current_step.clone())
                        } else {
                            // Branch on the split variable index
                            if let Some(j) = split_variable_idx {
                                // Right branch: force xj = 1
                                let mut right_lower_bounds = lower_bounds.clone();
                                right_lower_bounds[j] = 1.0;
                                stack.push((right_lower_bounds, upper_bounds.clone()));

                                // Left branch: force xj = 0
                                let mut left_upper_bounds = upper_bounds.clone();
                                left_upper_bounds[j] = 0.0;
                                stack.push((lower_bounds.clone(), left_upper_bounds));
                            }
                            current_step.active_nodes = stack.len();
                            Some(current_step.clone())
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_lp() {
        let objective_coeffs = [-2.5, -1.1, -0.9, -1.5];
        let constraint_matrix = [
            [4.3, 3.8, 1.6, 2.1],
            [4.0, 2.0, 1.9, 3.0],
        ];
        let constraint_rhs = [9.2, 9.0];
        let lower_bounds = [0.0, 0.0, 0.0, 0.0];
        let upper_bounds = [1.0, 1.0, 1.0, 1.0];

        let res = solve_lp(&objective_coeffs, &constraint_matrix, &constraint_rhs, &lower_bounds, &upper_bounds);
        println!("LP relaxation result: {:?}", res);
        
        // Also test with x2 fixed to 0
        let lower_bounds2 = [0.0, 0.0, 0.0, 0.0];
        let upper_bounds2 = [1.0, 0.0, 1.0, 1.0];
        let res2 = solve_lp(&objective_coeffs, &constraint_matrix, &constraint_rhs, &lower_bounds2, &upper_bounds2);
        println!("LP relaxation with x2=0 result: {:?}", res2);
    }

    #[test]
    fn test_branch_and_bound() {
        let objective_coeffs = [-2.5, -1.1, -0.9, -1.5];
        let constraint_matrix = [
            [4.3, 3.8, 1.6, 2.1],
            [4.0, 2.0, 1.9, 3.0],
        ];
        let constraint_rhs = [9.2, 9.0];

        let problem = MILPProblem::<4, 2> {
            objective_coeffs,
            constraint_matrix,
            constraint_rhs,
        };
        let solver = BranchAndBound;

        for (i, step) in solver.optimize(problem, ()).enumerate() {
            println!(
                "Step {}: current_node={:?}, best_y={:?}, best_x={:?}, active_nodes={}",
                i + 1,
                step.current_node,
                step.best_y,
                step.best_x,
                step.active_nodes
            );
        }
    }
}

