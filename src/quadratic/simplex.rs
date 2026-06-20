use super::*;
use crate::{helpers::Iterations, optimizer::TryOptimize};

#[derive(Clone, Debug)]
pub struct SimplexGuess<const V: usize, const C: usize> {
    pub base_idx: [usize; C],
    pub x: Column<V, f64>,
    pub is_optimal: bool,
}

#[derive(Clone)]
pub struct Simplex<S> {
    stopping_condition: S,
}

impl<S> Simplex<S> {
    pub fn new(stopping_condition: S) -> Self {
        Self { stopping_condition }
    }
}

/// Exposes the static LP matrices using the const-generic Matrix types.
pub trait LinearProgram<const V: usize, const C: usize> {
    fn a(&self) -> Matrix<C, V, f64>;
    fn b(&self) -> Column<C, f64>;
    fn c(&self) -> Row<V, f64>;
}

impl<const V: usize, const C: usize, P> TryOptimize<P, SimplexGuess<V, C>> for Simplex<Iterations>
where
    P: LinearProgram<V, C>,
{
    type Error = String;

    fn try_optimize(
        &self,
        problem: P,
        starting_guess: SimplexGuess<V, C>,
    ) -> impl Iterator<Item = Result<SimplexGuess<V, C>, String>> {
        // Extract matrix definitions once
        let a = problem.a();
        let b = problem.b();
        let c = problem.c();

        let mut current_guess = starting_guess;

        (0..self.stopping_condition.0).map(move |_| {
            if current_guess.is_optimal {
                return None;
            }

            // 1. Extract current basic matrix B from A
            let mut b_mat: SquareMatrix<C, f64> = Matrix(core::array::from_fn(|_| [0.0; C]));
            for i in 0..C {
                for j in 0..C {
                    b_mat.0[i][j] = a.0[i][current_guess.base_idx[j]];
                }
            }

            // 2. Compute inverse B_inv utilizing built-in method
            let b_inv = match b_mat.inverse() {
                Some(inv) => inv,
                None => return Some(Err("Singular basis matrix encountered".to_string())),
            };

            // 3. Extract costs of basic variables and compute y = c_B * B_inv
            let mut c_b = Row::<C, f64>::new_row([0.0; C]);
            for i in 0..C {
                c_b.0[0][i] = c.0[0][current_guess.base_idx[i]];
            }
            let y = c_b * b_inv;

            // 4. Compute reduced costs vector r = c - y * A
            let y_a = y * a;
            let r = c - y_a;

            // 5. Optimality test & find entering variable
            let mut min_r = f64::MAX;
            let mut j_en = 0;

            for j in 0..V {
                if r.0[0][j] < min_r {
                    min_r = r.0[0][j];
                    j_en = j;
                }
            }

            if min_r >= -1e-9 {
                // Calculate optimal coordinates x_opt = B_inv * b
                let x_b = b_inv * b;
                let mut x_opt = Column::<V, f64>::new_column([0.0; V]);
                for (i, &idx) in current_guess.base_idx.iter().enumerate() {
                    x_opt.0[idx][0] = x_b.0[i][0];
                }
                current_guess.x = x_opt;
                current_guess.is_optimal = true;
                return Some(Ok(current_guess.clone()));
            }

            // 6. Compute a_tilde and b_tilde for the entering column
            let mut a_j_en = Column::<C, f64>::new_column([0.0; C]);
            for i in 0..C {
                a_j_en.0[i][0] = a.0[i][j_en];
            }

            let a_tilde = b_inv * a_j_en;
            let b_tilde = b_inv * b;

            // 7. Unboundedness test
            let mut is_unbounded = true;
            for i in 0..C {
                if a_tilde.0[i][0] > 1e-9 {
                    is_unbounded = false;
                    break;
                }
            }
            if is_unbounded {
                return Some(Err("Problem is unbounded".to_string()));
            }

            // 8. Find leaving variable (minimal ratio test)
            let mut min_theta = f64::MAX;
            let mut q = 0;

            for i in 0..C {
                if a_tilde.0[i][0] > 1e-9 {
                    let theta = b_tilde.0[i][0] / a_tilde.0[i][0];
                    if theta < min_theta {
                        min_theta = theta;
                        q = i;
                    }
                }
            }

            // 9. Update basis indices
            current_guess.base_idx[q] = j_en;

            Some(Ok(current_guess.clone()))
        }).flatten()
    }
}
