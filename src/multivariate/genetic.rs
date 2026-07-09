use rand::RngExt;

use crate::linalg::Column;
use crate::optimizer::Optimize;

pub struct RealGAProblem<F> {
    pub objective: F,
    pub bounds_min: f64,
    pub bounds_max: f64,
}

#[derive(Clone, Debug)]
pub struct GAStep<const N: usize, const POP_SIZE: usize> {
    pub population: [Column<N, f64>; POP_SIZE],
    pub best_x: Option<Column<N, f64>>,
    pub best_f: f64,
}

impl<const N: usize, const POP_SIZE: usize> GAStep<N, POP_SIZE> {
    pub fn new(bounds_min: f64, bounds_max: f64) -> Self {
        let mut rng = rand::rng();
        let mut population = [Column::default(); POP_SIZE];
        for i in 0..POP_SIZE {
            for j in 0..N {
                population[i][(j, 0)] = bounds_min + (bounds_max - bounds_min) * rng.random::<f64>();
            }
        }

        Self {
            population,
            best_x: None,
            best_f: f64::INFINITY,
        }
    }
}

pub struct GeneticAlgorithm<const N: usize, const POP_SIZE: usize> {
    pub mutation_rate: f64,
}

impl<const N: usize, const POP_SIZE: usize> GeneticAlgorithm<N, POP_SIZE> {
    pub fn new(mutation_rate: f64) -> Self {
        Self { mutation_rate }
    }
}

impl<const N: usize, const POP_SIZE: usize, F> Optimize<RealGAProblem<F>, (), GAStep<N, POP_SIZE>>
    for GeneticAlgorithm<N, POP_SIZE>
where
    F: Fn(&Column<N, f64>) -> f64 + 'static,
{
    fn optimize(
        &self,
        problem: RealGAProblem<F>,
        _starting_guess: (),
    ) -> impl Iterator<Item = GAStep<N, POP_SIZE>> {
        let mut current = GAStep::new(problem.bounds_min, problem.bounds_max);
        let l_scale = (problem.bounds_max - problem.bounds_min).abs();
        let mutation_rate = self.mutation_rate;

        std::iter::from_fn(move || {
            let mut rng = rand::rng();
            let mut fitness = [0.0; POP_SIZE];

            let mut phi_min = f64::INFINITY;
            let mut idx_min = 0;
            let mut phi_high = f64::NEG_INFINITY;

            for (i, ind) in current.population.iter().enumerate() {
                let f_val = (problem.objective)(ind);
                fitness[i] = f_val;

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
            let mut weights = [0.0; POP_SIZE];
            for i in 0..POP_SIZE {
                let w = phi_high - fitness[i];
                weights[i] = w;
                sum_w += w;
            }

            let mut s_cumsum = [0.0; POP_SIZE];
            let mut current_sum = 0.0;
            for i in 0..POP_SIZE {
                current_sum += weights[i] / sum_w;
                s_cumsum[i] = current_sum;
            }

            let mut selected = [Column::<N, f64>::default(); POP_SIZE];
            for i in 0..(POP_SIZE - 1) {
                let r: f64 = rng.random();
                let mut chosen_idx = POP_SIZE - 1;
                for (j, &s_val) in s_cumsum.iter().enumerate() {
                    if s_val >= r {
                        chosen_idx = j;
                        break;
                    }
                }
                selected[i] = current.population[chosen_idx].clone();
            }

            if let Some(best) = &current.best_x {
                selected[POP_SIZE - 1] = best.clone();
            } else {
                selected[POP_SIZE - 1] = selected[0].clone();
            }

            let mut next_pop = [Column::<N, f64>::default(); POP_SIZE];
            let mut i = 0;
            while i < POP_SIZE - 1 {
                let p1 = &selected[i];
                let p2 = &selected[i + 1];
                let lambda: f64 = rng.random();

                let mut c1 = Column::default();
                let mut c2 = Column::default();

                for j in 0..N {
                    c1[(j, 0)] = lambda * p2[(j, 0)] + (1.0 - lambda) * p1[(j, 0)];
                    c2[(j, 0)] = lambda * p1[(j, 0)] + (1.0 - lambda) * p2[(j, 0)];
                }

                for child in [&mut c1, &mut c2] {
                    if rng.random::<f64>() <= mutation_rate {
                        for j in 0..N {
                            child[(j, 0)] += l_scale * 0.05 * (rng.random::<f64>() - 0.5);
                        }
                    }
                }

                next_pop[i] = c1;
                next_pop[i + 1] = c2;
                i += 2;
            }

            current.population = next_pop;

            Some(current.clone())
        })
    }
}

pub struct BinaryGAProblem<F> {
    /// Fitness function to maximize
    pub fitness_function: F,
}

#[derive(Clone, Debug)]
pub struct BinaryGAStep<const N: usize, const POP_SIZE: usize> {
    /// Current generation's population of binary chromosomes
    pub population: [[bool; N]; POP_SIZE],
    /// Best binary chromosome found so far
    pub best_x: Option<[bool; N]>,
    /// Best fitness value achieved so far
    pub best_fitness: f64,
}

impl<const N: usize, const POP_SIZE: usize> BinaryGAStep<N, POP_SIZE> {
    pub fn new() -> Self {
        let mut rng = rand::rng();
        let mut population = [[false; N]; POP_SIZE];
        for i in 0..POP_SIZE {
            for j in 0..N {
                population[i][j] = rng.random::<bool>();
            }
        }
        Self {
            population,
            best_x: None,
            best_fitness: f64::NEG_INFINITY,
        }
    }
}

pub struct BinaryGeneticAlgorithm<const N: usize, const POP_SIZE: usize> {
    pub mutation_probability: f64,
}

impl<const N: usize, const POP_SIZE: usize> BinaryGeneticAlgorithm<N, POP_SIZE> {
    pub fn new(mutation_probability: f64) -> Self {
        Self {
            mutation_probability,
        }
    }
}

impl<F, const N: usize, const POP_SIZE: usize> Optimize<BinaryGAProblem<F>, (), BinaryGAStep<N, POP_SIZE>>
    for BinaryGeneticAlgorithm<N, POP_SIZE>
where
    F: Fn(&[bool; N]) -> f64 + 'static,
{
    fn optimize(
        &self,
        problem: BinaryGAProblem<F>,
        _starting_guess: (),
    ) -> impl Iterator<Item = BinaryGAStep<N, POP_SIZE>> {
        let mut current = BinaryGAStep::new();
        let mutation_probability = self.mutation_probability;

        std::iter::from_fn(move || {
            let mut rng = rand::rng();

            // 1. Selection (Tournament of size 2)
            let mut mating_pool = [[false; N]; POP_SIZE];
            for i in 0..POP_SIZE {
                let candidate_idx_1 = rng.random_range(0..POP_SIZE);
                let candidate_idx_2 = rng.random_range(0..POP_SIZE);
                let fitness_1 = (problem.fitness_function)(&current.population[candidate_idx_1]);
                let fitness_2 = (problem.fitness_function)(&current.population[candidate_idx_2]);
                mating_pool[i] = if fitness_1 >= fitness_2 {
                    current.population[candidate_idx_1]
                } else {
                    current.population[candidate_idx_2]
                };
            }

            // 2. Crossover (Single-point crossover between adjacent parents in mating pool)
            let mut offspring = [[false; N]; POP_SIZE];
            let mut i = 0;
            while i < POP_SIZE {
                let parent_1 = &mating_pool[i];
                let parent_2 = &mating_pool[i + 1];
                let crossover_split_point = rng.random_range(1..N);

                let mut child_1 = [false; N];
                let mut child_2 = [false; N];
                child_1[..crossover_split_point].copy_from_slice(&parent_1[..crossover_split_point]);
                child_1[crossover_split_point..].copy_from_slice(&parent_2[crossover_split_point..]);

                child_2[..crossover_split_point].copy_from_slice(&parent_2[..crossover_split_point]);
                child_2[crossover_split_point..].copy_from_slice(&parent_1[crossover_split_point..]);

                offspring[i] = child_1;
                offspring[i + 1] = child_2;
                i += 2;
            }

            // 3. Mutation (Bit-flip mutation based on mutation probability)
            for individual in &mut offspring {
                for gene in individual.iter_mut() {
                    if rng.random::<f64>() <= mutation_probability {
                        *gene = !*gene;
                    }
                }
            }

            current.population = offspring;

            // 4. Update Best Solution
            let mut current_generation_best_fitness = f64::NEG_INFINITY;
            let mut current_generation_best_index = 0;
            for idx in 0..POP_SIZE {
                let fitness_value = (problem.fitness_function)(&current.population[idx]);
                if fitness_value > current_generation_best_fitness {
                    current_generation_best_fitness = fitness_value;
                    current_generation_best_index = idx;
                }
            }

            if current_generation_best_fitness > current.best_fitness {
                current.best_fitness = current_generation_best_fitness;
                current.best_x = Some(current.population[current_generation_best_index]);
            }

            Some(current.clone())
        })
    }
}

