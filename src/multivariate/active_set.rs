use crate::helpers::{Iterations, Precision};
use crate::linalg::{Column, Matrix, SquareMatrix};
use crate::optimizer::TryOptimize;

pub struct InequalityConstrainedQP<const N: usize, const M: usize> {
    pub q: SquareMatrix<N, f64>,
    pub c: Column<N, f64>,
    pub a: Matrix<M, N, f64>,
    pub b: Column<M, f64>,
}

#[derive(Clone, Debug)]
pub struct ActiveSetGuess<const N: usize, const M: usize> {
    pub x: Column<N, f64>,
    pub w: Vec<usize>,
}

impl<const N: usize, const M: usize> From<Column<N, f64>> for ActiveSetGuess<N, M> {
    fn from(x: Column<N, f64>) -> Self {
        Self { x, w: Vec::new() }
    }
}

#[derive(Clone, Copy)]
pub struct ActiveSetMethod<S> {
    pub stopping_criterion: S,
}

impl<S> ActiveSetMethod<S> {
    pub fn new(stopping_criterion: S) -> Self {
        Self { stopping_criterion }
    }
}

fn solve_dynamic(mat: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = mat.len();
    let mut a = mat.to_owned();
    let mut x = b.to_owned();

    for i in 0..n {
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for j in (i + 1)..n {
            if a[j][i].abs() > max_val {
                max_val = a[j][i].abs();
                max_row = j;
            }
        }

        if max_val < 1e-9 {
            return None;
        }

        if max_row != i {
            a.swap(i, max_row);
            x.swap(i, max_row);
        }

        let pivot = a[i][i];
        for j in i..n {
            a[i][j] /= pivot;
        }
        x[i] /= pivot;

        for j in 0..n {
            if j != i {
                let factor = a[j][i];
                for k in i..n {
                    a[j][k] -= factor * a[i][k];
                }
                x[j] -= factor * x[i];
            }
        }
    }
    Some(x)
}

impl<const N: usize, const M: usize>
    TryOptimize<InequalityConstrainedQP<N, M>, ActiveSetGuess<N, M>, ActiveSetGuess<N, M>>
    for ActiveSetMethod<Precision>
{
    type Error = String;

    fn try_optimize(
        self,
        problem: InequalityConstrainedQP<N, M>,
        starting_guess: ActiveSetGuess<N, M>,
    ) -> impl Iterator<Item = Result<ActiveSetGuess<N, M>, Self::Error>> {
        let mut current = starting_guess.clone();
        let precision = self.stopping_criterion.0;
        let mut converged = false;

        std::iter::once(Ok(starting_guess)).chain(std::iter::from_fn(move || {
            if converged {
                return None;
            }

            let m_w = current.w.len();
            let kkt_size = N + m_w;
            let mut kkt_mat = vec![vec![0.0; kkt_size]; kkt_size];

            // Fill Q
            for i in 0..N {
                for j in 0..N {
                    kkt_mat[i][j] = problem.q.0[i][j];
                }
            }
            // Fill A_W^T and A_W
            for idx in 0..m_w {
                let constraint_idx = current.w[idx];
                for i in 0..N {
                    let val = problem.a.0[constraint_idx][i];
                    kkt_mat[i][N + idx] = val;
                    kkt_mat[N + idx][i] = val;
                }
            }

            // Fill RHS: [ -(Q x + c); 0 ]
            let qx = problem.q * current.x;
            let grad = qx + problem.c;
            let mut rhs = vec![0.0; kkt_size];
            for i in 0..N {
                rhs[i] = -grad.0[i][0];
            }

            // Solve using dynamic KKT solver
            let solution = match solve_dynamic(&kkt_mat, &rhs) {
                Some(sol) => sol,
                None => return Some(Err("Singular KKT matrix encountered in Active Set Method".to_string())),
            };

            let mut p = [0.0; N];
            p[..N].copy_from_slice(&solution[..N]);
            let lambdas = &solution[N..];

            // Check if step p is close to zero
            let p_norm = p.iter().map(|val| val * val).sum::<f64>().sqrt();

            if p_norm <= precision {
                // Stationary point check
                let mut min_lambda = f64::MAX;
                let mut j_leave_idx = 0;
                for idx in 0..m_w {
                    if lambdas[idx] < min_lambda {
                        min_lambda = lambdas[idx];
                        j_leave_idx = idx;
                    }
                }

                if min_lambda >= -1e-9 {
                    converged = true;
                    return None;
                } else {
                    current.w.remove(j_leave_idx);
                }
            } else {
                // Find maximum feasible step length theta
                let mut min_theta = f64::INFINITY;
                let mut j_block = -1;

                for i in 0..M {
                    if !current.w.contains(&i) {
                        let mut ap = 0.0;
                        for j in 0..N {
                            ap += problem.a.0[i][j] * p[j];
                        }

                        if ap > 1e-9 {
                            let mut ax = 0.0;
                            for j in 0..N {
                                ax += problem.a.0[i][j] * current.x.0[j][0];
                            }
                            let theta = (problem.b.0[i][0] - ax) / ap;
                            if theta < min_theta {
                                min_theta = theta;
                                j_block = i as i32;
                            }
                        }
                    }
                }

                if min_theta >= 1.0 {
                    for i in 0..N {
                        current.x.0[i][0] += p[i];
                    }
                } else {
                    for i in 0..N {
                        current.x.0[i][0] += min_theta * p[i];
                    }
                    current.w.push(j_block as usize);
                }
            }

            Some(Ok(current.clone()))
        }))
    }
}

impl<const N: usize, const M: usize>
    TryOptimize<InequalityConstrainedQP<N, M>, ActiveSetGuess<N, M>, ActiveSetGuess<N, M>>
    for ActiveSetMethod<Iterations>
{
    type Error = String;

    fn try_optimize(
        self,
        problem: InequalityConstrainedQP<N, M>,
        starting_guess: ActiveSetGuess<N, M>,
    ) -> impl Iterator<Item = Result<ActiveSetGuess<N, M>, Self::Error>> {
        let mut current = starting_guess.clone();
        let limit = self.stopping_criterion.0;
        let mut count = 0;
        let mut converged = false;

        std::iter::once(Ok(starting_guess)).chain(std::iter::from_fn(move || {
            if converged || count >= limit {
                return None;
            }

            let m_w = current.w.len();
            let kkt_size = N + m_w;
            let mut kkt_mat = vec![vec![0.0; kkt_size]; kkt_size];

            // Fill Q
            for i in 0..N {
                for j in 0..N {
                    kkt_mat[i][j] = problem.q.0[i][j];
                }
            }
            // Fill A_W^T and A_W
            for idx in 0..m_w {
                let constraint_idx = current.w[idx];
                for i in 0..N {
                    let val = problem.a.0[constraint_idx][i];
                    kkt_mat[i][N + idx] = val;
                    kkt_mat[N + idx][i] = val;
                }
            }

            // Fill RHS: [ -(Q x + c); 0 ]
            let qx = problem.q * current.x;
            let grad = qx + problem.c;
            let mut rhs = vec![0.0; kkt_size];
            for i in 0..N {
                rhs[i] = -grad.0[i][0];
            }

            // Solve KKT dynamically
            let solution = match solve_dynamic(&kkt_mat, &rhs) {
                Some(sol) => sol,
                None => return Some(Err("Singular KKT matrix encountered in Active Set Method".to_string())),
            };

            let mut p = [0.0; N];
            p[..N].copy_from_slice(&solution[..N]);
            let lambdas = &solution[N..];

            // Check if step p is close to zero
            let p_norm = p.iter().map(|val| val * val).sum::<f64>().sqrt();

            if p_norm <= 1e-7 {
                let mut min_lambda = f64::MAX;
                let mut j_leave_idx = 0;
                for idx in 0..m_w {
                    if lambdas[idx] < min_lambda {
                        min_lambda = lambdas[idx];
                        j_leave_idx = idx;
                    }
                }

                if min_lambda >= -1e-9 {
                    converged = true;
                    return None;
                } else {
                    current.w.remove(j_leave_idx);
                }
            } else {
                let mut min_theta = f64::INFINITY;
                let mut j_block = -1;

                for i in 0..M {
                    if !current.w.contains(&i) {
                        let mut ap = 0.0;
                        for j in 0..N {
                            ap += problem.a.0[i][j] * p[j];
                        }

                        if ap > 1e-9 {
                            let mut ax = 0.0;
                            for j in 0..N {
                                ax += problem.a.0[i][j] * current.x.0[j][0];
                            }
                            let theta = (problem.b.0[i][0] - ax) / ap;
                            if theta < min_theta {
                                min_theta = theta;
                                j_block = i as i32;
                            }
                        }
                    }
                }

                if min_theta >= 1.0 {
                    for i in 0..N {
                        current.x.0[i][0] += p[i];
                    }
                } else {
                    for i in 0..N {
                        current.x.0[i][0] += min_theta * p[i];
                    }
                    current.w.push(j_block as usize);
                }
            }

            count += 1;
            Some(Ok(current.clone()))
        }))
    }
}
