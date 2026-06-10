use crate::quadratic::{Column, SquareMatrix};

pub struct Conjugate<const N: usize>;

impl<const N: usize> Conjugate<N> {
    pub fn optimize(
        self,
        matrix: SquareMatrix<N, f64>,
        b: Column<N, f64>,
        starting_guess: Column<N, f64>,
    ) -> Result<Vec<Column<N, f64>>, String> {
        let mut x = starting_guess;
        let mut guesses = vec![x];

        let qx = matrix ^ x;
        let mut g = qx - b;

        let mut d = -g;

        for _ in 0..N {
            let g_norm = (g.transpose() ^ g).into_value().sqrt();

            if g_norm <= 0.001 {
                break;
            }

            let qd = matrix ^ d;

            let den = (d.transpose() ^ qd).into_value();

            if den.abs() < f64::EPSILON {
                return Err("Division by zero: Q matrix might not be positive definite".into());
            }

            let alpha = -(g.transpose() ^ d).into_value() / den;

            x += d * alpha;
            guesses.push(x);

            let g_next = g + qd * alpha;

            let beta = (g_next.transpose() ^ qd).into_value() / den;

            d = -g_next + d * beta;

            g = g_next;
        }

        Ok(guesses)
    }
}
