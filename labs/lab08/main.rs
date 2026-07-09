use optimization::{
    multivariate::simulated_annealing::{SAProblem, SAStep, SimulatedAnnealing},
    optimizer::TryOptimize,
};
use plotly::{Plot, Scatter};

fn load_capitals() -> Vec<(f64, f64)> {
    let content = std::fs::read_to_string("lab-instructions/12-lab8-Files for lab of May 13th/USA_cap.txt")
        .or_else(|_| std::fs::read_to_string("USA_cap.txt"))
        .expect("Failed to read USA_cap.txt");
    let mut coords = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            let x: f64 = parts[0].trim().parse().unwrap();
            let y: f64 = parts[1].trim().parse().unwrap();
            coords.push((x, y));
        }
    }
    coords
}

fn main() {
    println!("========================================");
    println!("  LAB 8: Simulated Annealing");
    println!("========================================");

    // -------------------------------------------------------------------------
    // Exercise 1: 5-Cities Test Case
    // -------------------------------------------------------------------------
    println!("\n--- Test Case: 5-Cities TSP ---");
    let dist_5 = vec![
        vec![0.0, 3.0, 4.0, 2.0, 7.0],
        vec![3.0, 0.0, 4.0, 6.0, 3.0],
        vec![4.0, 4.0, 0.0, 5.0, 8.0],
        vec![2.0, 6.0, 5.0, 0.0, 6.0],
        vec![7.0, 3.0, 8.0, 6.0, 0.0],
    ];

    let dist_5_clone = dist_5.clone();
    let energy_5 = move |tour: &[usize; 5]| -> f64 {
        let mut d = 0.0;
        let n = tour.len();
        for i in 0..n {
            d += dist_5_clone[tour[i]][tour[(i + 1) % n]];
        }
        d
    };

    let start_tour_5 = [0, 1, 2, 3, 4];
    let start_energy_5 = energy_5(&start_tour_5);
    let start_step_5 = SAStep::<5>::new(start_tour_5, start_energy_5, 100.0);

    let problem_5 = SAProblem::<_, 5> {
        cost_function: energy_5,
        initial_temperature: 100.0,
        cooling_rate: 0.95,
    };

    let solver = SimulatedAnnealing;
    let mut final_step_5 = None;

    // Run for 200 iterations
    for res in solver.try_optimize(problem_5, start_step_5).take(200) {
        if let Ok(step) = res {
            final_step_5 = Some(step);
        }
    }

    let final_5 = final_step_5.unwrap();
    println!("  Starting Tour: [0, 1, 2, 3, 4] with length: {}", start_energy_5);
    println!("  Optimal Tour:  {:?} with length: {:.1} (Target = 19.0)", final_5.best_tour, final_5.best_cost);

    // -------------------------------------------------------------------------
    // Exercise 2: 48 USA State Capitals
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1: 48 USA Capitals TSP ---");
    let coords = load_capitals();
    let n_cities = coords.len();
    println!("Loaded {} state capitals.", n_cities);

    // Compute distance matrix
    let mut dist = vec![vec![0.0; n_cities]; n_cities];
    for i in 0..n_cities {
        for j in 0..n_cities {
            let dx = coords[i].0 - coords[j].0;
            let dy = coords[i].1 - coords[j].1;
            dist[i][j] = (dx * dx + dy * dy).sqrt();
        }
    }

    let dist_clone = dist.clone();
    let energy_cap = move |tour: &[usize; 48]| -> f64 {
        let mut d = 0.0;
        let n = tour.len();
        for i in 0..n {
            d += dist_clone[tour[i]][tour[(i + 1) % n]];
        }
        d
    };

    // Initial tour: 0..48 sequentially
    let mut start_tour_vec: Vec<usize> = (0..n_cities).collect();
    // Shuffle the tour slightly to avoid initial bias
    let mut rng = rand::rng();
    use rand::seq::SliceRandom;
    start_tour_vec.shuffle(&mut rng);

    let start_tour: [usize; 48] = start_tour_vec.try_into().unwrap();
    let start_energy = energy_cap(&start_tour);

    let t0: f64 = 1000.0;
    let t_end: f64 = 1e-4;
    let n_steps = 150_000;
    let tau: f64 = t_end / t0;
    let alpha: f64 = tau.powf(1.0 / n_steps as f64);

    println!("Starting Simulated Annealing...");
    println!("  t0 = {:.1}, alpha = {:.6}, steps = {}", t0, alpha, n_steps);

    let start_step = SAStep::<48>::new(start_tour, start_energy, t0);
    let problem_cap = SAProblem::<_, 48> {
        cost_function: energy_cap,
        initial_temperature: t0,
        cooling_rate: alpha,
    };

    let mut iterations = Vec::new();
    let mut current_lengths = Vec::new();
    let mut best_lengths = Vec::new();
    let mut final_step = None;

    // We can sample plots every 500 steps to keep plotly light
    for (i, res) in solver.try_optimize(problem_cap, start_step).take(n_steps).enumerate() {
        if let Ok(step) = res {
            if i % 500 == 0 {
                iterations.push(i as f64);
                current_lengths.push(step.current_cost);
                best_lengths.push(step.best_cost);
            }
            final_step = Some(step);
        }
    }

    let final_step = final_step.unwrap();
    println!("  Initial Tour Length: {:.1}", start_energy);
    println!("  Final Tour Length:   {:.1} (Minimal target = 33523.0)", final_step.best_cost);
    println!("  Best Tour Order:     {:?}", final_step.best_tour);

    // Plotting
    let mut plot1 = Plot::new();
    plot1.add_trace(Scatter::new(iterations.clone(), current_lengths).name("Current Tour Length"));
    plot1.add_trace(Scatter::new(iterations, best_lengths).name("Best Tour Length"));
    plot1.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("Iteration"))
            .y_axis(plotly::layout::Axis::new().title("Length / Energy")),
    );

    // Coordinate Plot for best tour
    let mut tour_x = Vec::new();
    let mut tour_y = Vec::new();
    for &idx in &final_step.best_tour {
        tour_x.push(coords[idx].0);
        tour_y.push(coords[idx].1);
    }
    // Return to start
    if !final_step.best_tour.is_empty() {
        let first_idx = final_step.best_tour[0];
        tour_x.push(coords[first_idx].0);
        tour_y.push(coords[first_idx].1);
    }

    let mut plot_map = Plot::new();
    let capitals_scatter = Scatter::new(
        coords.iter().map(|c| c.0).collect(),
        coords.iter().map(|c| c.1).collect(),
    )
    .mode(plotly::common::Mode::Markers)
    .name("Capitals");
    let tour_line = Scatter::new(tour_x, tour_y)
        .mode(plotly::common::Mode::Lines)
        .name("Best Tour");
    plot_map.add_trace(capitals_scatter);
    plot_map.add_trace(tour_line);
    plot_map.set_layout(
        plotly::Layout::new()
            .x_axis(plotly::layout::Axis::new().title("X Coordinate"))
            .y_axis(plotly::layout::Axis::new().title("Y Coordinate")),
    );

    optimization::helpers::save_dashboard(
        "labs/lab08/plot.html",
        "LAB 8: Simulated Annealing for TSP",
        &[
            ("USA Capitals SA Convergence Trajectory", &plot1),
            ("USA Capitals Final Tour Route Map", &plot_map),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab08/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab08/plot.html");
}
