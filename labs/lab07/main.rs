use optimization::{
    multivariate::genetic::{
        BinaryGAProblem, BinaryGeneticAlgorithm, GeneticAlgorithm, RealGAProblem,
    },
    optimizer::Optimize,
    linalg::Column,
};
use plotly::{Plot, Scatter};

fn load_knapsack() -> (Vec<f64>, Vec<f64>) {
    let content = std::fs::read_to_string("lab-instructions/11-lab7-Files for lab of May, 8th/knapsack.csv")
        .or_else(|_| std::fs::read_to_string("knapsack.csv"))
        .expect("Failed to read knapsack.csv");
    let mut weights = Vec::new();
    let mut values = Vec::new();
    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 2 {
            let w: f64 = parts[0].trim().parse().unwrap();
            let v: f64 = parts[1].trim().parse().unwrap();
            weights.push(w);
            values.push(v);
        }
    }
    (weights, values)
}

fn main() {
    println!("========================================");
    println!("  LAB 7: Genetic Algorithms");
    println!("========================================");

    // -------------------------------------------------------------------------
    // Exercise 1: Binary GA for 0-1 Knapsack Problem
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1: 0-1 Knapsack with Binary GA ---");
    let (weights_vec, values_vec) = load_knapsack();
    assert_eq!(weights_vec.len(), 100);
    let weights: [f64; 100] = weights_vec.try_into().unwrap();
    let values: [f64; 100] = values_vec.try_into().unwrap();
    println!("Loaded 100 items from CSV.");

    let max_weight = 1965.0;
    let alpha = 100.0; // Penalty multiplier for constraint violation

    // Define penalized fitness function
    let fitness_fn = move |chromosome: &[bool; 100]| -> f64 {
        let mut total_w = 0.0;
        let mut total_v = 0.0;
        for i in 0..100 {
            if chromosome[i] {
                total_w += weights[i];
                total_v += values[i];
            }
        }
        let penalty = alpha * (total_w - max_weight).max(0.0);
        total_v - penalty
    };

    let pmut = 0.01;
    let n_generations = 300;

    // Test with mating pool size 50
    println!("Running GA with mating pool size 50...");
    let solver_binary = BinaryGeneticAlgorithm::<100, 50>::new(pmut);
    let problem_50 = BinaryGAProblem {
        fitness_function: fitness_fn.clone(),
    };

    let mut generations_50 = Vec::new();
    let mut best_fitness_50 = Vec::new();
    let mut step_cnt = 0;
    let mut final_best_x_50 = None;
    let mut final_best_f_50 = 0.0;

    for step in solver_binary.optimize(problem_50, ()).take(n_generations) {
        generations_50.push(step_cnt as f64);
        best_fitness_50.push(step.best_fitness);
        final_best_x_50 = step.best_x;
        final_best_f_50 = step.best_fitness;
        step_cnt += 1;
    }

    let mut final_weight_50 = 0.0;
    if let Some(ref x) = final_best_x_50 {
        for i in 0..100 {
            if x[i] {
                final_weight_50 += weights[i];
            }
        }
    }

    println!("  Mating Pool 50 Summary:");
    println!("    Best Fitness: {:.1}", final_best_f_50);
    println!("    Total Weight: {:.1} / {}", final_weight_50, max_weight);

    // Test with mating pool size 500
    println!("Running GA with mating pool size 500...");
    let solver_binary_500 = BinaryGeneticAlgorithm::<100, 500>::new(pmut);
    let problem_500 = BinaryGAProblem {
        fitness_function: fitness_fn.clone(),
    };

    let mut generations_500 = Vec::new();
    let mut best_fitness_500 = Vec::new();
    let mut step_cnt_500 = 0;
    let mut final_best_x_500 = None;
    let mut final_best_f_500 = 0.0;

    for step in solver_binary_500.optimize(problem_500, ()).take(n_generations) {
        generations_500.push(step_cnt_500 as f64);
        best_fitness_500.push(step.best_fitness);
        final_best_x_500 = step.best_x;
        final_best_f_500 = step.best_fitness;
        step_cnt_500 += 1;
    }

    let mut final_weight_500 = 0.0;
    if let Some(ref x) = final_best_x_500 {
        for i in 0..100 {
            if x[i] {
                final_weight_500 += weights[i];
            }
        }
    }

    println!("  Mating Pool 500 Summary:");
    println!("    Best Fitness: {:.1} (Optimum target = 4966)", final_best_f_500);
    println!("    Total Weight: {:.1} / {}", final_weight_500, max_weight);

    let mut plot1 = Plot::new();
    plot1.add_trace(Scatter::new(generations_50, best_fitness_50).name("Mating Pool: 50"));
    plot1.add_trace(Scatter::new(generations_500, best_fitness_500).name("Mating Pool: 500"));
    plot1.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Generation"))
            .y_axis(plotly::layout::Axis::new().title("Best Fitness (Value)")),
    );

    // -------------------------------------------------------------------------
    // Exercise 2: Real GA for Continuous Function Minimization
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 2: Continuous Minimization with Real GA ---");
    let target_function = |x: &Column<2, f64>| -> f64 {
        let x1 = x[(0, 0)];
        let x2 = x[(1, 0)];
        x1.powi(4) + x2.powi(4) - 4.0 * x1.powi(3) - 3.0 * x2.powi(3) + 2.0 * x1.powi(2) + 2.0 * x1 * x2
    };

    println!("Stage 1: Initial broad search in [-10.0, 10.0] x [-10.0, 10.0]");
    let problem_real = RealGAProblem {
        objective: target_function,
        bounds_min: -10.0,
        bounds_max: 10.0,
    };
    let solver_real = GeneticAlgorithm::<2, 100>::new(0.15);

    let mut real_generations = Vec::new();
    let mut real_best_costs = Vec::new();
    let mut step_real = 0;
    let mut final_real_x = None;
    let mut final_real_f = f64::INFINITY;

    for step in solver_real.optimize(problem_real, ()).take(150) {
        real_generations.push(step_real as f64);
        real_best_costs.push(step.best_f);
        final_real_x = step.best_x.clone();
        final_real_f = step.best_f;
        step_real += 1;
    }

    println!("  Stage 1 Results:");
    println!("    Best Cost: {:.5} (Target = -13.532)", final_real_f);
    if let Some(ref x) = final_real_x {
        println!("    Best Position: [{:.5}, {:.5}] (Target = [2.67321, -0.675885])", x[(0, 0)], x[(1, 0)]);
    }

    println!("\nStage 2: Refined search in [0.0, 4.0] x [-2.0, 2.0]");
    let problem_refined = RealGAProblem {
        objective: target_function,
        bounds_min: -2.0, // using the helper bounds
        bounds_max: 4.0,
    };
    let solver_refined = GeneticAlgorithm::<2, 100>::new(0.1);

    let mut refined_generations = Vec::new();
    let mut refined_best_costs = Vec::new();
    let mut step_ref = 0;
    let mut final_ref_x = None;
    let mut final_ref_f = f64::INFINITY;

    for step in solver_refined.optimize(problem_refined, ()).take(100) {
        refined_generations.push(step_ref as f64);
        refined_best_costs.push(step.best_f);
        final_ref_x = step.best_x.clone();
        final_ref_f = step.best_f;
        step_ref += 1;
    }

    println!("  Stage 2 Refined Results:");
    println!("    Best Cost: {:.6} (Target = -13.5320)", final_ref_f);
    if let Some(ref x) = final_ref_x {
        println!("    Best Position: [{:.6}, {:.6}] (Target = [2.67321, -0.675885])", x[(0, 0)], x[(1, 0)]);
    }

    let mut plot2 = Plot::new();
    plot2.add_trace(Scatter::new(real_generations, real_best_costs).name("Initial [-10, 10]"));
    plot2.add_trace(Scatter::new(refined_generations, refined_best_costs).name("Refined [-2, 4]"));
    plot2.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Generation"))
            .y_axis(plotly::layout::Axis::new().title("Best Cost (Objective)")),
    );

    optimization::helpers::save_dashboard(
        "labs/lab07/plot.html",
        "LAB 7: Genetic Algorithms",
        &[
            ("Exercise 1: Knapsack GA Convergence", &plot1),
            ("Exercise 2: Real GA Function Minimization", &plot2),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab07/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab07/plot.html");
}
