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

#[derive(Clone, Debug)]
pub struct LinearProgram<const V: usize, const C: usize> {
    pub a: Matrix<C, V, f64>,
    pub b: Column<C, f64>,
    pub c: Row<V, f64>,
}

impl<const V: usize, const C: usize> TryOptimize<LinearProgram<V, C>, SimplexGuess<V, C>>
    for Simplex<Iterations>
{
    type Error = String;

    fn try_optimize(
        &self,
        problem: LinearProgram<V, C>,
        starting_guess: SimplexGuess<V, C>,
    ) -> impl Iterator<Item = Result<SimplexGuess<V, C>, String>> {
        let mut current_guess = starting_guess;

        (0..self.stopping_condition.0)
            .map(move |_| {
                if current_guess.is_optimal {
                    return None;
                }

                let mut b_mat: SquareMatrix<C, f64> = Matrix(core::array::from_fn(|_| [0.0; C]));
                for i in 0..C {
                    for j in 0..C {
                        b_mat.0[i][j] = problem.a.0[i][current_guess.base_idx[j]];
                    }
                }

                let b_inv = match b_mat.inverse() {
                    Some(inv) => inv,
                    None => return Some(Err("Singular basis matrix encountered".to_string())),
                };

                let mut c_b = Row::<C, f64>::new_row([0.0; C]);
                for i in 0..C {
                    c_b.0[0][i] = problem.c.0[0][current_guess.base_idx[i]];
                }
                let y = c_b * b_inv;

                let y_a = y * problem.a;
                let r = problem.c - y_a;

                let mut min_r = f64::MAX;
                let mut j_en = 0;

                for j in 0..V {
                    if r.0[0][j] < min_r {
                        min_r = r.0[0][j];
                        j_en = j;
                    }
                }

                if min_r >= -1e-9 {
                    let x_b = b_inv * problem.b;
                    let mut x_opt = Column::<V, f64>::new_column([0.0; V]);
                    for (i, &idx) in current_guess.base_idx.iter().enumerate() {
                        x_opt.0[idx][0] = x_b.0[i][0];
                    }
                    current_guess.x = x_opt;
                    current_guess.is_optimal = true;
                    return Some(Ok(current_guess.clone()));
                }

                let mut a_j_en = Column::<C, f64>::new_column([0.0; C]);
                for i in 0..C {
                    a_j_en.0[i][0] = problem.a.0[i][j_en];
                }

                let a_tilde = b_inv * a_j_en;
                let b_tilde = b_inv * problem.b;

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

                current_guess.base_idx[q] = j_en;

                Some(Ok(current_guess.clone()))
            })
            .flatten()
    }
}

pub struct AugmentedLP;

impl AugmentedLP {
    pub fn new<const V: usize, const C: usize>(
        a: &Matrix<C, V, f64>,
        b: &Column<C, f64>,
    ) -> LinearProgram<{ C + V }, C>
    where
        [(); C + V]:,
    {
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

        LinearProgram {
            a: Matrix(a_aug),
            b: b.clone(),
            c: Row::new_row(c_virt),
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

impl<const V: usize, const C: usize> PermutedLP<V, C> {
    pub fn to_linear_program(&self) -> LinearProgram<V, C> {
        LinearProgram {
            a: self.a_perm,
            b: self.b_perm.clone(),
            c: self.c_perm,
        }
    }

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

impl<const V: usize, const C: usize> From<PermutedLP<V, C>> for LinearProgram<V, C> {
    fn from(p: PermutedLP<V, C>) -> Self {
        LinearProgram {
            a: p.a_perm,
            b: p.b_perm,
            c: p.c_perm,
        }
    }
}
