use rand::RngExt;

use crate::optimizer::TryOptimize;
use crate::linalg::Column;

pub struct RealGAProblem<F> {
    pub objective: F,
    pub bounds_min: f64,
    pub bounds_max: f64,
    pub mutation_rate: f64,
}

#[derive(Clone, Debug)]
pub struct GAStep<const N: usize> {
    pub population: Vec<Column<N, f64>>,
    pub best_x: Option<Column<N, f64>>,
    pub best_f: f64,
}

impl<const N: usize> GAStep<N> {
    pub fn new(pop_size: usize, bounds_min: f64, bounds_max: f64) -> Self {
        let mut rng = rand::rng();
        let size = if pop_size % 2 != 0 { pop_size + 1 } else { pop_size };
        
        let mut population = Vec::with_capacity(size);
        for _ in 0..size {
            let mut ind = Column::default(); 
            for j in 0..N {
                ind[(0, j)] = bounds_min + (bounds_max - bounds_min) * rng.random::<f64>(); 
            }
            population.push(ind);
        }

        Self {
            population,
            best_x: None,
            best_f: f64::INFINITY, 
        }
    }
}

pub struct GeneticAlgorithm<const N: usize>;

impl<const N: usize, F> TryOptimize<RealGAProblem<F>, GAStep<N>, GAStep<N>> for GeneticAlgorithm<N>
where
    F: Fn(&Column<N, f64>) -> f64 + Clone + 'static,
{
    type Error = String;

    fn try_optimize(
        &self,
        problem: RealGAProblem<F>,
        starting_guess: GAStep<N>,
    ) -> impl Iterator<Item = Result<GAStep<N>, Self::Error>> {
        let mut current = starting_guess;
        let l_scale = (problem.bounds_max - problem.bounds_min).abs();

        std::iter::from_fn(move || {
            let mut rng = rand::rng();
            let pop_size = current.population.len();
            let mut fitness = Vec::with_capacity(pop_size);

            let mut phi_min = f64::INFINITY;
            let mut idx_min = 0;
            let mut phi_high = f64::NEG_INFINITY;

            for (i, ind) in current.population.iter().enumerate() {
                let f_val = (problem.objective)(ind);
                fitness.push(f_val);

                if f_val < phi_min {
                    phi_min = f_val;
                    idx_min = i;
                }
                if f_val > phi_high {
                    phi_high = f_val;
                }
            }

            if phi_min < current.best_f {
                current.best_f = phi_min;
                current.best_x = Some(current.population[idx_min].clone());
            }

            let mut sum_w = 0.0;
            let mut weights = Vec::with_capacity(pop_size);
            for f_val in &fitness {
                let w = phi_high - f_val;
                weights.push(w);
                sum_w += w;
            }

            let mut s_cumsum = Vec::with_capacity(pop_size);
            let mut current_sum = 0.0;
            for w in weights {
                current_sum += w / sum_w;
                s_cumsum.push(current_sum);
            }

            let mut selected = Vec::with_capacity(pop_size);
            for _ in 0..(pop_size - 1) {
                let r: f64 = rng.random();
                let mut chosen_idx = pop_size - 1;
                for (j, &s_val) in s_cumsum.iter().enumerate() {
                    if s_val >= r {
                        chosen_idx = j;
                        break;
                    }
                }
                selected.push(current.population[chosen_idx].clone());
            }

            if let Some(best) = &current.best_x {
                selected.push(best.clone());
            }

            let mut next_pop = Vec::with_capacity(pop_size);
            for i in (0..pop_size - 1).step_by(2) {
                let p1 = &selected[i];
                let p2 = &selected[i + 1];
                let lambda: f64 = rng.random();

                let mut c1 = Column::default();
                let mut c2 = Column::default();

                for j in 0..N {
                    c1[(0, j)] = lambda * p2[(0, j)] + (1.0 - lambda) * p1[(0, j)];
                    c2[(0, j)] = lambda * p1[(0, j)] + (1.0 - lambda) * p2[(0, j)];
                }

                for child in [&mut c1, &mut c2] {
                    if rng.random::<f64>() <= problem.mutation_rate {
                        for j in 0..N {
                            child[(0, j)] += l_scale * 0.05 * (rng.random::<f64>() - 0.5);
                        }
                    }
                    next_pop.push(child.clone());
                }
            }

            current.population = next_pop;

            Some(Ok(current.clone()))
        })
    }
}
