use crate::linalg::{Column, Matrix, SquareMatrix};
use crate::optimizer::TryOptimize;

pub struct EqualityConstrainedQP<const N: usize, const M: usize> {
    pub q: SquareMatrix<N, f64>,
    pub c: Column<N, f64>,
    pub a: Matrix<M, N, f64>,
    pub b: Column<M, f64>,
}

#[derive(Clone, Debug)]
pub struct QPStep<const N: usize, const M: usize> {
    pub x: Column<N, f64>,
    pub lambda: Column<M, f64>,
}

pub struct NewtonRaphsonQP<const N: usize, const M: usize>;

impl<const N: usize, const M: usize>
    TryOptimize<EqualityConstrainedQP<N, M>, QPStep<N, M>, QPStep<N, M>> for NewtonRaphsonQP<N, M>
where
    [(); N + M]:,
{
    type Error = String;

    fn try_optimize(
        &self,
        problem: EqualityConstrainedQP<N, M>,
        starting_guess: QPStep<N, M>,
    ) -> impl Iterator<Item = Result<QPStep<N, M>, Self::Error>> {
        let mut current = starting_guess;
        let mut opt = false;

        let max_iter = 20;

        (0..max_iter).map(move |_| {
            if opt {
                return None;
            }

            // 1. Construct KKT Matrix H_L (Hessian of the Lagrangian)
            // [ Q   A^T ]
            // [ A    0  ]
            let h_l = problem.q.block_concat(&problem.a);

            // 2. Construct gradient vector g
            // [ Qx + c ]
            // [ Ax - b ]
            let qx_c = (problem.q * current.x) + problem.c;
            let ax_b = (problem.a * current.x) - problem.b;
            let g = qx_c.stack(&ax_b);

            // 3. Solve H_L * [p, lambda] = -g
            let solution = match h_l.solve(&(-g)) {
                Ok(sol) => sol,
                Err(_) => return Some(Err("Failed to solve KKT system: Linearly dependent constraints or singular matrix".into())),
            };

            // Extract the step p (size N) and the new lambda (size M)
            let p = solution.extract_p();
            let lambda_new = solution.extract_lambda();

            // 4. Tolerance check for convergence
            let p_norm = (p.transpose() * p).into_value().sqrt();
            if p_norm <= 1e-7 {
                opt = true;
                current.lambda = lambda_new;
                return Some(Ok(current.clone()));
            }

            current.x += p;
            current.lambda = lambda_new;

            Some(Ok(current.clone()))
        }).flatten()
    }
}

impl<const N: usize, const M: usize> From<Column<N, f64>> for QPStep<N, M> {
    fn from(x0: Column<N, f64>) -> Self {
        Self {
            x: x0,
            lambda: Column::zeros(), // Initialize multipliers to 0
        }
    }
}
