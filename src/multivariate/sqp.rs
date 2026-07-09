use crate::helpers::{Iterations, Precision};
use crate::linalg::{Column, Matrix};
use crate::optimizer::TryOptimize;

pub struct EqualityConstrainedProblem<const N: usize, const M: usize, F, H> {
    pub f: F,
    pub h: H,
}

#[derive(Clone, Debug)]
pub struct SqpState<const N: usize, const M: usize> {
    pub x: [f64; N],
    pub lambda: [f64; M],
}

#[derive(Clone, Copy)]
pub struct LocalSqpMethod<S> {
    pub stopping_criterion: S,
    pub finite_difference_step: f64,
}

impl<S> LocalSqpMethod<S> {
    pub fn new(stopping_criterion: S) -> Self {
        Self {
            stopping_criterion,
            finite_difference_step: 1e-5,
        }
    }

    pub fn with_step(stopping_criterion: S, step: f64) -> Self {
        Self {
            stopping_criterion,
            finite_difference_step: step,
        }
    }
}

// Helper functions for numerical differentiation
fn num_grad_f<const N: usize>(
    f: &impl Fn(Column<N, f64>) -> f64,
    x: &Column<N, f64>,
    h_step: f64,
) -> Column<N, f64> {
    let mut grad = [0.0; N];
    for i in 0..N {
        let mut x_plus = *x;
        x_plus.0[i][0] += h_step;
        let mut x_minus = *x;
        x_minus.0[i][0] -= h_step;

        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * h_step);
    }
    Column::new_column(grad)
}

fn num_jac_h<const N: usize, const M: usize>(
    h: &impl Fn(Column<N, f64>) -> Column<M, f64>,
    x: &Column<N, f64>,
    h_step: f64,
) -> Matrix<M, N, f64> {
    let mut jac = [[0.0; N]; M];
    for j in 0..N {
        let mut x_plus = *x;
        x_plus.0[j][0] += h_step;
        let mut x_minus = *x;
        x_minus.0[j][0] -= h_step;

        let h_plus = h(x_plus);
        let h_minus = h(x_minus);

        for i in 0..M {
            jac[i][j] = (h_plus.0[i][0] - h_minus.0[i][0]) / (2.0 * h_step);
        }
    }
    Matrix(jac)
}

fn grad_lagrangian<const N: usize, const M: usize>(
    f: &impl Fn(Column<N, f64>) -> f64,
    h: &impl Fn(Column<N, f64>) -> Column<M, f64>,
    x: &Column<N, f64>,
    lambda: &Column<M, f64>,
    h_step: f64,
) -> Column<N, f64> {
    let mut grad = [0.0; N];
    for i in 0..N {
        let mut x_plus = *x;
        x_plus.0[i][0] += h_step;
        let mut x_minus = *x;
        x_minus.0[i][0] -= h_step;

        let f_plus = f(x_plus);
        let h_plus = h(x_plus);
        let mut l_plus = f_plus;
        for j in 0..M {
            l_plus += lambda.0[j][0] * h_plus.0[j][0];
        }

        let f_minus = f(x_minus);
        let h_minus = h(x_minus);
        let mut l_minus = f_minus;
        for j in 0..M {
            l_minus += lambda.0[j][0] * h_minus.0[j][0];
        }

        grad[i] = (l_plus - l_minus) / (2.0 * h_step);
    }
    Column::new_column(grad)
}

fn num_hess_lagrangian<const N: usize, const M: usize>(
    f: &impl Fn(Column<N, f64>) -> f64,
    h: &impl Fn(Column<N, f64>) -> Column<M, f64>,
    x: &Column<N, f64>,
    lambda: &Column<M, f64>,
    h_step: f64,
) -> Matrix<N, N, f64> {
    let mut hess = [[0.0; N]; N];
    for j in 0..N {
        let mut x_plus = *x;
        x_plus.0[j][0] += h_step;
        let mut x_minus = *x;
        x_minus.0[j][0] -= h_step;

        let grad_plus = grad_lagrangian(f, h, &x_plus, lambda, h_step);
        let grad_minus = grad_lagrangian(f, h, &x_minus, lambda, h_step);

        for i in 0..N {
            hess[i][j] = (grad_plus.0[i][0] - grad_minus.0[i][0]) / (2.0 * h_step);
        }
    }
    Matrix(hess)
}

impl<const N: usize, const M: usize, F, H>
    TryOptimize<EqualityConstrainedProblem<N, M, F, H>, [f64; N], SqpState<N, M>>
    for LocalSqpMethod<Precision>
where
    F: Fn([f64; N]) -> f64,
    H: Fn([f64; N]) -> [f64; M],
    [(); N + M]:,
{
    type Error = String;

    fn try_optimize(
        self,
        problem: EqualityConstrainedProblem<N, M, F, H>,
        starting_guess: [f64; N],
    ) -> impl Iterator<Item = Result<SqpState<N, M>, Self::Error>> {
        let mut x_col = Column::new_column(starting_guess);
        let mut lambda_col = Column::new_column([0.0; M]);
        let precision = self.stopping_criterion.0;
        let h_step = self.finite_difference_step;
        let mut converged = false;

        let f_wrapped = move |xc: Column<N, f64>| (problem.f)(xc.into_column());
        let h_wrapped = move |xc: Column<N, f64>| Column::new_column((problem.h)(xc.into_column()));

        let initial_state = SqpState {
            x: starting_guess,
            lambda: [0.0; M],
        };

        std::iter::once(Ok(initial_state)).chain(std::iter::from_fn(move || {
            if converged {
                return None;
            }

            // Calculate objective gradient numerically
            let g_f = num_grad_f(&f_wrapped, &x_col, h_step);

            // Calculate Hessian of Lagrangian numerically
            let q_m = num_hess_lagrangian(&f_wrapped, &h_wrapped, &x_col, &lambda_col, h_step);

            // Calculate constraint Jacobian numerically
            let a_m = num_jac_h(&h_wrapped, &x_col, h_step);

            // Constraint value
            let h_val = h_wrapped(x_col);

            // Solve QP subproblem:
            // [ Q_m   A_m^T ] [ p ]   [ -g_f ]
            // [ A_m     0   ] [ l ] = [ -h   ]
            let h_l = q_m.block_concat(&a_m);
            let rhs = (-g_f).stack(&(-h_val));

            let solution = match h_l.solve(&rhs) {
                Ok(sol) => sol,
                Err(_) => return Some(Err("Singular KKT matrix encountered in SQP subproblem".to_string())),
            };

            let p = solution.extract_p();
            let lambda_new = solution.extract_lambda();

            let p_norm = (p.transpose() * p).into_value().sqrt();

            if p_norm <= precision {
                converged = true;
                return None;
            }

            x_col += p;
            lambda_col = lambda_new;

            Some(Ok(SqpState {
                x: x_col.into_column(),
                lambda: lambda_col.into_column(),
            }))
        }))
    }
}

impl<const N: usize, const M: usize, F, H>
    TryOptimize<EqualityConstrainedProblem<N, M, F, H>, [f64; N], SqpState<N, M>>
    for LocalSqpMethod<Iterations>
where
    F: Fn([f64; N]) -> f64,
    H: Fn([f64; N]) -> [f64; M],
    [(); N + M]:,
{
    type Error = String;

    fn try_optimize(
        self,
        problem: EqualityConstrainedProblem<N, M, F, H>,
        starting_guess: [f64; N],
    ) -> impl Iterator<Item = Result<SqpState<N, M>, Self::Error>> {
        let mut x_col = Column::new_column(starting_guess);
        let mut lambda_col = Column::new_column([0.0; M]);
        let limit = self.stopping_criterion.0;
        let h_step = self.finite_difference_step;
        let mut count = 0;

        let f_wrapped = move |xc: Column<N, f64>| (problem.f)(xc.into_column());
        let h_wrapped = move |xc: Column<N, f64>| Column::new_column((problem.h)(xc.into_column()));

        let initial_state = SqpState {
            x: starting_guess,
            lambda: [0.0; M],
        };

        std::iter::once(Ok(initial_state)).chain(std::iter::from_fn(move || {
            if count >= limit {
                return None;
            }

            let g_f = num_grad_f(&f_wrapped, &x_col, h_step);
            let q_m = num_hess_lagrangian(&f_wrapped, &h_wrapped, &x_col, &lambda_col, h_step);
            let a_m = num_jac_h(&h_wrapped, &x_col, h_step);
            let h_val = h_wrapped(x_col);

            let h_l = q_m.block_concat(&a_m);
            let rhs = (-g_f).stack(&(-h_val));

            let solution = match h_l.solve(&rhs) {
                Ok(sol) => sol,
                Err(_) => return Some(Err("Singular KKT matrix encountered in SQP subproblem".to_string())),
            };

            let p = solution.extract_p();
            let lambda_new = solution.extract_lambda();

            x_col += p;
            lambda_col = lambda_new;

            count += 1;
            Some(Ok(SqpState {
                x: x_col.into_column(),
                lambda: lambda_col.into_column(),
            }))
        }))
    }
}
