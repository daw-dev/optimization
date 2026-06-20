use crate::{
    optimizer::TryOptimize,
    quadratic::{Column, SquareMatrix},
};

pub struct PerfectQuadraticProblem<const N: usize> {
    pub matrix: SquareMatrix<N, f64>,
    pub b: Column<N, f64>,
}

pub struct Conjugate<const N: usize>;

impl<const N: usize> TryOptimize<PerfectQuadraticProblem<N>, Column<N, f64>> for Conjugate<N> {
    type Error = String;

    fn try_optimize(
        &self,
        problem: PerfectQuadraticProblem<N>,
        starting_guess: Column<N, f64>,
    ) -> impl Iterator<Item = Result<Column<N, f64>, String>> {
        let matrix = problem.matrix;
        let b = problem.b;
        let mut x = starting_guess;

        let qx = matrix ^ x;
        let mut g = qx - b;

        let mut d = -g;

        (0..N)
            .map(move |_| {
                let g_norm = (g.transpose() ^ g).into_value().sqrt();

                if g_norm <= 0.001 {
                    return None;
                }

                let qd = matrix ^ d;

                let den = (d.transpose() ^ qd).into_value();

                if den.abs() < f64::EPSILON {
                    return Some(Err(
                        "Division by zero: Q matrix might not be positive definite".into(),
                    ));
                }

                let alpha = -(g.transpose() ^ d).into_value() / den;

                x += d * alpha;

                let g_next = g + qd * alpha;

                let beta = (g_next.transpose() ^ qd).into_value() / den;

                d = -g_next + d * beta;

                g = g_next;
                Some(Ok(x))
            })
            .flatten()
    }
}
