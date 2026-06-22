use crate::{
    function::{Differentiate, Function},
    helpers::{Iterations, Precision},
    linalg::{Column, SquareMatrix},
    optimizer::Optimize,
};

pub struct Bfgs<S> {
    stopping_criterion: S,
    difference: f64,
}

impl<S> Bfgs<S> {
    pub fn new(stopping_criterion: S, difference: f64) -> Self {
        Self {
            stopping_criterion,
            difference,
        }
    }
}

impl<'a, const N: usize, F: crate::function::Function<[f64; N], f64> + 'a> Optimize<&'a F, [f64; N]>
    for Bfgs<Iterations>
{
    fn optimize(&self, func: &'a F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut x0 = Column::<N, f64>::new_column(starting_guess);
        let mut h = SquareMatrix::<N, f64>::identity();
        let grad_fn = func.differentiate(self.difference);
        let mut g = Column::new_column(grad_fn.compute(x0.into_column()));
        let mut count = 0;
        let limit = self.stopping_criterion.0;

        std::iter::once(starting_guess).chain(std::iter::from_fn(move || {
            if count >= limit {
                return None;
            }
            let d = -(h.clone() * g.clone());

            // Backtracking line search
            let mut alpha = 1.0;
            let f_x = func.compute(x0.into_column());
            let mut best_alpha = 0.0;
            let mut best_val = f_x;
            for _ in 0..12 {
                let x_test = x0.clone() + d.clone() * alpha;
                let val = func.compute(x_test.into_column());
                if val < best_val {
                    best_val = val;
                    best_alpha = alpha;
                }
                alpha *= 0.5;
            }

            let alpha_opt = if best_alpha == 0.0 { 1e-4 } else { best_alpha };
            let xp = x0.clone() + d.clone() * alpha_opt;
            let gp = Column::new_column(grad_fn.compute(xp.into_column()));

            let delta_g = gp.clone() - g.clone();
            let delta_x = xp.clone() - x0.clone();

            let den = (delta_g.transpose() * delta_x.clone()).into_value();
            if den.abs() > 1e-9 {
                let rho = 1.0 / den;
                let identity = SquareMatrix::<N, f64>::identity();
                let term1 = identity.clone() - (delta_x.clone() * delta_g.transpose() * rho);
                let term2 = identity.clone() - (delta_g.clone() * delta_x.transpose() * rho);
                h = term1 * h * term2 + (delta_x.clone() * delta_x.transpose() * rho);
            }

            x0 = xp;
            g = gp;
            count += 1;
            Some(x0.into_column())
        }))
    }
}

impl<'a, const N: usize, F: crate::function::Function<[f64; N], f64> + 'a> Optimize<&'a F, [f64; N]>
    for Bfgs<Precision>
{
    fn optimize(&self, func: &'a F, starting_guess: [f64; N]) -> impl Iterator<Item = [f64; N]> {
        let mut x0 = Column::<N, f64>::new_column(starting_guess);
        let mut h = SquareMatrix::<N, f64>::identity();
        let grad_fn = func.differentiate(self.difference);
        let mut g = Column::new_column(grad_fn.compute(x0.into_column()));

        std::iter::once(starting_guess).chain(std::iter::from_fn(move || {
            let g_norm = (g.transpose() * g.clone()).into_value().sqrt();
            if g_norm < self.stopping_criterion.0 {
                return None;
            }

            let d = -(h.clone() * g.clone());

            // Backtracking line search
            let mut alpha = 1.0;
            let f_x = func.compute(x0.into_column());
            let mut best_alpha = 0.0;
            let mut best_val = f_x;
            for _ in 0..12 {
                let x_test = x0.clone() + d.clone() * alpha;
                let val = func.compute(x_test.into_column());
                if val < best_val {
                    best_val = val;
                    best_alpha = alpha;
                }
                alpha *= 0.5;
            }

            let alpha_opt = if best_alpha == 0.0 { 1e-4 } else { best_alpha };
            let xp = x0.clone() + d.clone() * alpha_opt;
            let gp = Column::new_column(grad_fn.compute(xp.into_column()));

            let delta_g = gp.clone() - g.clone();
            let delta_x = xp.clone() - x0.clone();

            let den = (delta_g.transpose() * delta_x.clone()).into_value();
            if den.abs() > 1e-9 {
                let rho = 1.0 / den;
                let identity = SquareMatrix::<N, f64>::identity();
                let term1 = identity.clone() - (delta_x.clone() * delta_g.transpose() * rho);
                let term2 = identity.clone() - (delta_g.clone() * delta_x.transpose() * rho);
                h = term1 * h * term2 + (delta_x.clone() * delta_x.transpose() * rho);
            }

            x0 = xp;
            g = gp;
            Some(x0.into_column())
        }))
    }
}
