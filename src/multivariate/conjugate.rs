use crate::{
    helpers::{Iterations, Precision},
    linalg::{Column, SquareMatrix},
    optimizer::TryOptimize,
};

pub struct PerfectQuadraticProblem<const N: usize> {
    pub matrix: SquareMatrix<N, f64>,
    pub b: Column<N, f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct Conjugate<const N: usize, S = Precision> {
    stopping_criterion: S,
}

impl<const N: usize, S> Conjugate<N, S> {
    pub fn new(stopping_criterion: S) -> Self {
        Self { stopping_criterion }
    }
}

impl<const N: usize> TryOptimize<PerfectQuadraticProblem<N>, Column<N, f64>>
    for Conjugate<N, Precision>
{
    type Error = String;

    fn try_optimize(
        self,
        problem: PerfectQuadraticProblem<N>,
        starting_guess: Column<N, f64>,
    ) -> impl Iterator<Item = Result<Column<N, f64>, String>> {
        let matrix = problem.matrix;
        let b = problem.b;
        let mut x = starting_guess;

        let qx = matrix * x;
        let mut g = qx - b;

        let mut d = -g;

        (0..N)
            .map(move |_| {
                let g_norm = (g.transpose() * g).into_value().sqrt();

                if g_norm <= self.stopping_criterion.0 {
                    return None;
                }

                let qd = matrix * d;

                let den = (d.transpose() * qd).into_value();

                if den.abs() < f64::EPSILON {
                    return Some(Err(
                        "Division by zero: Q matrix might not be positive definite".into(),
                    ));
                }

                let alpha = -(g.transpose() * d).into_value() / den;

                x += d * alpha;

                let g_next = g + qd * alpha;

                let beta = (g_next.transpose() * qd).into_value() / den;

                d = -g_next + d * beta;

                g = g_next;
                Some(Ok(x))
            })
            .flatten()
    }
}

impl<const N: usize> TryOptimize<PerfectQuadraticProblem<N>, Column<N, f64>>
    for Conjugate<N, Iterations>
{
    type Error = String;

    fn try_optimize(
        self,
        problem: PerfectQuadraticProblem<N>,
        starting_guess: Column<N, f64>,
    ) -> impl Iterator<Item = Result<Column<N, f64>, String>> {
        let matrix = problem.matrix;
        let b = problem.b;
        let mut x = starting_guess;

        let qx = matrix * x;
        let mut g = qx - b;

        let mut d = -g;
        let limit = self.stopping_criterion.0;
        let mut count = 0;

        std::iter::from_fn(move || {
            if count >= limit {
                return None;
            }

            let qd = matrix * d;

            let den = (d.transpose() * qd).into_value();

            if den.abs() < f64::EPSILON {
                return Some(Err(
                    "Division by zero: Q matrix might not be positive definite".into(),
                ));
            }

            let alpha = -(g.transpose() * d).into_value() / den;

            x += d * alpha;

            let g_next = g + qd * alpha;

            let beta = (g_next.transpose() * qd).into_value() / den;

            d = -g_next + d * beta;

            g = g_next;
            count += 1;
            Some(Ok(x))
        })
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::helpers::Precision;
    use crate::linalg::Matrix;

    #[test]
    fn test_cg_convergence() {
        const N: usize = 3;
        let A = Matrix([[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]);
        let B = Matrix([[1.0], [2.0], [3.0]]);

        let opt = Conjugate::new(Precision(0.001));
        let res = opt
            .try_optimize(
                PerfectQuadraticProblem {
                    matrix: A.clone(),
                    b: B.clone(),
                },
                Column::zeros(),
            )
            .last()
            .unwrap()
            .unwrap();

        let analytical = A.inverse().unwrap() * B.clone();

        // Assert commutative property: scalar * matrix == matrix * scalar
        let B_left = 2.0 * B.clone();
        let B_right = B.clone() * 2.0;
        for i in 0..N {
            assert!((B_left.0[i][0] - B_right.0[i][0]).abs() < 1e-9);
        }

        println!("CG: {:?}", res.0);
        println!("Analytical: {:?}", analytical.0);

        for i in 0..N {
            assert!((res.0[i][0] - analytical.0[i][0]).abs() < 1e-3);
        }
    }
}
