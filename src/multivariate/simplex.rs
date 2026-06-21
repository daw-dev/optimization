use crate::linalg::{Column, Matrix, Row, SquareMatrix};
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

        (0..self.stopping_condition.0)
            .map(move |_| {
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
            })
            .flatten()
    }
}

#[derive(Clone, Debug)]
pub struct AugmentedLP<const V: usize, const C: usize>
where
    [(); C + V]:,
{
    pub a_aug: Matrix<C, { C + V }, f64>,
    pub b: Column<C, f64>,
    pub c_virt: Row<{ C + V }, f64>,
}

impl<const V: usize, const C: usize> LinearProgram<{ C + V }, C> for AugmentedLP<V, C>
where
    [(); C + V]:,
{
    fn a(&self) -> Matrix<C, { C + V }, f64> {
        self.a_aug
    }
    fn b(&self) -> Column<C, f64> {
        self.b
    }
    fn c(&self) -> Row<{ C + V }, f64> {
        self.c_virt
    }
}

impl<const V: usize, const C: usize> AugmentedLP<V, C>
where
    [(); C + V]:,
{
    pub fn new(a: &Matrix<C, V, f64>, b: &Column<C, f64>) -> Self {
        let mut a_aug = [[0.0; C + V]; C];
        // Set first C columns to identity
        for i in 0..C {
            for j in 0..C {
                a_aug[i][j] = if i == j { 1.0 } else { 0.0 };
            }
        }
        // Set remaining columns to A
        for i in 0..C {
            for j in 0..V {
                a_aug[i][C + j] = a.0[i][j];
            }
        }

        let mut c_virt = [0.0; C + V];
        // Minimize artificial variables: first C variables have cost 1.0
        for i in 0..C {
            c_virt[i] = 1.0;
        }

        Self {
            a_aug: Matrix(a_aug),
            b: b.clone(),
            c_virt: Row::new_row(c_virt),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PermutedLP<const V: usize, const C: usize> {
    pub a_perm: Matrix<C, V, f64>,
    pub b_perm: Column<C, f64>,
    pub c_perm: Row<V, f64>,
    pub perm: [usize; V],
}

impl<const V: usize, const C: usize> LinearProgram<V, C> for PermutedLP<V, C> {
    fn a(&self) -> Matrix<C, V, f64> {
        self.a_perm
    }
    fn b(&self) -> Column<C, f64> {
        self.b_perm
    }
    fn c(&self) -> Row<V, f64> {
        self.c_perm
    }
}

impl<const V: usize, const C: usize> PermutedLP<V, C> {
    pub fn new(
        a: &Matrix<C, V, f64>,
        b: &Column<C, f64>,
        c: &Row<V, f64>,
        idx_init: &[usize; C],
    ) -> Result<Self, String> {
        let mut perm = [0; V];
        let mut used = [false; V];

        for i in 0..C {
            let orig_idx = idx_init[i].checked_sub(C).ok_or_else(|| {
                "Artificial variable remains in basis; problem might be infeasible".to_string()
            })?;
            perm[i] = orig_idx;
            used[orig_idx] = true;
        }

        let mut fill_idx = C;
        for i in 0..V {
            if !used[i] {
                perm[fill_idx] = i;
                fill_idx += 1;
            }
        }

        let mut a_new = [[0.0; V]; C];
        let mut c_new = [0.0; V];
        for j in 0..V {
            let orig_j = perm[j];
            for i in 0..C {
                a_new[i][j] = a.0[i][orig_j];
            }
            c_new[j] = c.0[0][orig_j];
        }

        let mut cch = SquareMatrix::<C, f64>::identity();
        for i in 0..C {
            for j in 0..C {
                cch.0[i][j] = a_new[i][j];
            }
        }

        let cch_inv = cch
            .inverse()
            .ok_or_else(|| "Coordinate change matrix is singular".to_string())?;

        let a_perm = cch_inv * Matrix(a_new);
        let b_perm = cch_inv * b.clone();

        Ok(Self {
            a_perm,
            b_perm,
            c_perm: Row::new_row(c_new),
            perm,
        })
    }
}
