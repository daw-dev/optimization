#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use optimization::{
    helpers::Precision,
    linalg::{Column, Matrix},
    multivariate::{
        active_set::{ActiveSetGuess, ActiveSetMethod, InequalityConstrainedQP},
        constrained::{EqualityConstrainedQP, NewtonRaphsonQP},
        sqp::{EqualityConstrainedProblem, LocalSqpMethod, SqpState},
    },
    optimizer::TryOptimize,
};
use plotly::{
    Layout, Plot, Scatter,
    common::{Marker, Mode},
    layout::Axis,
};

fn main() {
    println!("========================================");
    println!("  LAB 6: Quadratic & Constrained Programming");
    println!("========================================");

    // -------------------------------------------------------------------------
    // Exercise 1: 5D QP with 3 Equality Constraints
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1: 5D QP with 3 Equality Constraints ---");
    
    let q1 = Matrix([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, -0.5, 0.0, 0.0],
        [0.0, -0.5, 2.0, 2.0, 0.2],
        [0.0, 0.0, 2.0, 3.0, 0.0],
        [0.0, 0.0, 0.2, 0.0, 0.5],
    ]);
    let c1 = Column::new_column([2.0, 1.0, 0.0, 0.0, -1.0]);
    let a1 = Matrix([
        [0.0, 1.0, 0.0, -1.0, 1.0],
        [2.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 1.0, 0.0],
    ]);
    let b1 = Column::new_column([3.0, 4.0, 3.0]);

    let problem1 = EqualityConstrainedQP {
        q: q1,
        c: c1,
        a: a1,
        b: b1,
    };

    let start_guess1 = Column::zeros();
    let solver1 = NewtonRaphsonQP::<5, 3, Precision>::new(Precision(1e-7));

    let final_step1 = itertools::process_results(
        solver1.try_optimize(problem1, start_guess1),
        |iter| {
            iter.fold(
                (0, None),
                |(step, _), guess| {
                    println!(
                        "  Iteration {}: x = {:.5?}, lambda = {:.5?}",
                        step,
                        guess.x.into_column(),
                        guess.lambda.into_column()
                    );
                    (step + 1, Some(guess))
                }
            ).1
        }
    )
    .expect("NewtonRaphsonQP failed");

    let sol1 = final_step1.expect("NewtonRaphsonQP should converge");
    let x_star = sol1.x;
    let lambda_star = sol1.lambda;
    println!("\nExercise 1 Results:");
    println!("  Optimal Solution x*:     {:.5?}", x_star.into_column());
    println!("  Lagrange Multipliers:    {:.5?}", lambda_star.into_column());

    // Verify Lagrange's Theorem is satisfied: Qx* + c + A^T lambda* = 0
    let grad_lagrangian = (q1 * x_star) + c1 + (a1.transpose() * lambda_star);
    println!("  Verification (gradient of Lagrangian, target = [0.0; 5]):");
    println!("    {:.5?}", grad_lagrangian.into_column());
    let constr_check = (a1 * x_star) - b1;
    println!("  Verification (Ax* - b, target = [0.0; 3]):");
    println!("    {:.5?}", constr_check.into_column());


    // -------------------------------------------------------------------------
    // Exercise 2: Local SQP for nonlinear function
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 2: Local SQP for Nonlinear Equality Constraints ---");
    // Minimize f(x) = 1.5 * x2 - 2 * x1^2 - x1^3 + x1^4
    // Subject to: x1^2 + x2^2 = 3

    let problem2 = EqualityConstrainedProblem {
        f: |[x1, x2]: [f64; 2]| {
            1.5 * x2 - 2.0 * x1.powi(2) - x1.powi(3) + x1.powi(4)
        },
        h: |[x1, x2]: [f64; 2]| {
            [x1.powi(2) + x2.powi(2) - 3.0]
        },
    };

    let start_guess2 = [1.0, 5.0];
    let solver2 = LocalSqpMethod::new(Precision(1e-6));

    println!("Starting Local SQP from x0 = {:?}", start_guess2);

    let (path_ex2, final_state) = itertools::process_results(
        solver2.try_optimize(problem2, start_guess2),
        |iter| {
            iter.fold(
                (Vec::<[f64; 2]>::new(), None),
                |(mut path, _), state: SqpState<2, 1>| {
                    let x = state.x;
                    let lambda = state.lambda;
                    
                    let p_norm = if path.is_empty() {
                        0.0
                    } else {
                        let prev = path.last().unwrap();
                        ((x[0] - prev[0]).powi(2) + (x[1] - prev[1]).powi(2)).sqrt()
                    };

                    path.push(x);
                    let step_num = path.len();
                    
                    println!(
                        "  Step {}: x = [{:.5}, {:.5}], lambda = {:.5}, stop_crit (||p||) = {:.3e}",
                        step_num, x[0], x[1], lambda[0], p_norm
                    );
                    
                    (path, Some(state))
                }
            )
        }
    )
    .expect("SQP optimization failed");

    let sol2 = final_state.expect("SqpState iterator should yield at least one state");
    println!("Local SQP converged in {} iterations.", path_ex2.len());
    println!("\nExercise 2 Results:");
    println!("  Optimal Solution x*:     [{:.5}, {:.5}]", sol2.x[0], sol2.x[1]);
    println!("  Lagrange Multiplier:     {:.5}", sol2.lambda[0]);

    let (path_ex2_x1, path_ex2_x2) = path_ex2.into_iter().map(|[x1, x2]| (x1, x2)).unzip();

    // -------------------------------------------------------------------------
    // Exercise 3: Active Set Method for QP with Inequality Constraints
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 3: Active Set Method for Convex QP ---");
    
    let q_ex3 = Matrix([
        [1.0, 1.0],
        [1.0, 2.0]
    ]);
    let c_ex3 = Column::new_column([4.0, -5.0]);
    
    // Constraints Ax <= b
    let a_ex3 = Matrix([
        [-1.0, 0.0],
        [1.0, 0.0],
        [1.0, -1.0],
        [0.0, 1.0],
    ]);
    let b_ex3 = Column::new_column([0.0, 1.0, 0.0, 3.0]);

    let problem3 = InequalityConstrainedQP {
        q: q_ex3,
        c: c_ex3,
        a: a_ex3,
        b: b_ex3,
    };

    let start_guess3 = ActiveSetGuess {
        x: Column::new_column([1.0, 1.0]),
        w: vec![1, 2], // Initial active set (constraints 2 and 3)
    };

    let solver3 = ActiveSetMethod::new(Precision(1e-7));

    println!("Starting Active Set Method from x0 = [1.0, 1.0], W0 = {:?}", start_guess3.w);

    let (path_ex3, final_guess3) = itertools::process_results(
        solver3.try_optimize(problem3, start_guess3),
        |iter| {
            iter.fold(
                (Vec::<[f64; 2]>::new(), None),
                |(mut path, _), guess| {
                    let x = [guess.x.0[0][0], guess.x.0[1][0]];
                    path.push(x);
                    let step_num = path.len();
                    println!(
                        "  Iteration {}: x = [{:.4}, {:.4}], W = {:?}",
                        step_num,
                        x[0],
                        x[1],
                        guess.w
                    );
                    (path, Some(guess))
                }
            )
        }
    )
    .expect("Active Set Method failed");

    let sol3 = final_guess3.expect("ActiveSetMethod should converge");
    println!("\nExercise 3 Results:");
    println!("  Optimal Solution x*:     [{:.5}, {:.5}]", sol3.x.0[0][0], sol3.x.0[1][0]);

    let (path_ex3_x1, path_ex3_x2): (Vec<f64>, Vec<f64>) = path_ex3.into_iter().map(|[x1, x2]| (x1, x2)).unzip();


    // -------------------------------------------------------------------------
    // Plotting & Dashboard saving
    // -------------------------------------------------------------------------
    
    // Plot 1: SQP Path and Circle constraint for Exercise 2
    let mut plot1 = Plot::new();
    
    // Circle constraint: x1^2 + x2^2 = 3
    let mut circle_x = Vec::new();
    let mut circle_y = Vec::new();
    let r_circle = 3.0_f64.sqrt();
    for i in 0..=100 {
        let theta = (i as f64) * 2.0 * std::f64::consts::PI / 100.0;
        circle_x.push(r_circle * theta.cos());
        circle_y.push(r_circle * theta.sin());
    }
    let trace_constraint = Scatter::new(circle_x, circle_y)
        .mode(Mode::Lines)
        .name("Constraint: x1^2 + x2^2 = 3");
    plot1.add_trace(trace_constraint);

    let trace_path_ex2 = Scatter::new(path_ex2_x1, path_ex2_x2)
        .mode(Mode::LinesMarkers)
        .marker(Marker::new().size(8).color("lime"))
        .name("LSQP Optimization Path");
    plot1.add_trace(trace_path_ex2);

    let trace_opt_ex2 = Scatter::new(vec![1.2801], vec![-1.1668])
        .mode(Mode::Markers)
        .marker(
            Marker::new()
                .size(12)
                .color("red")
                .symbol(plotly::common::MarkerSymbol::Star),
        )
        .name("Optimal Solution (1.2801, -1.1668)");
    plot1.add_trace(trace_opt_ex2);
    
    plot1.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("x1").range(vec![-3.0, 3.0]))
            .y_axis(Axis::new().title("x2").range(vec![-6.0, 6.0])),
    );


    // Plot 2: Active Set Path and Feasible region for Exercise 3
    let mut plot2 = Plot::new();

    // Draw constraints boundaries
    // c1: x1 = 0
    let c1_trace = Scatter::new(vec![0.0, 0.0], vec![-1.0, 4.5]).mode(Mode::Lines).name("x1 >= 0");
    // c2: x1 = 1
    let c2_trace = Scatter::new(vec![1.0, 1.0], vec![-1.0, 4.5]).mode(Mode::Lines).name("x1 <= 1");
    // c3: x1 = x2
    let c3_trace = Scatter::new(vec![-0.5, 2.0], vec![-0.5, 2.0]).mode(Mode::Lines).name("x1 <= x2");
    // c4: x2 = 3
    let c4_trace = Scatter::new(vec![-0.5, 2.0], vec![3.0, 3.0]).mode(Mode::Lines).name("x2 <= 3");

    plot2.add_trace(c1_trace);
    plot2.add_trace(c2_trace);
    plot2.add_trace(c3_trace);
    plot2.add_trace(c4_trace);

    let trace_path_ex3 = Scatter::new(path_ex3_x1, path_ex3_x2)
        .mode(Mode::LinesMarkers)
        .marker(Marker::new().size(8).color("lime"))
        .name("Active Set Path");
    plot2.add_trace(trace_path_ex3);

    let trace_opt_ex3 = Scatter::new(vec![0.0], vec![2.5])
        .mode(Mode::Markers)
        .marker(
            Marker::new()
                .size(12)
                .color("red")
                .symbol(plotly::common::MarkerSymbol::Star),
        )
        .name("Optimal Solution (0, 2.5)");
    plot2.add_trace(trace_opt_ex3);

    plot2.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("x1").range(vec![-0.5, 2.0]))
            .y_axis(Axis::new().title("x2").range(vec![-0.5, 4.5])),
    );

    optimization::helpers::save_dashboard(
        "labs/lab06/plot.html",
        "LAB 6: Quadratic & Nonlinear Programming (QP / LSQP / Active Set)",
        &[
            ("Exercise 2: LSQP for Nonlinear Constraints", &plot1),
            ("Exercise 3: Active Set Method Path", &plot2),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab06/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab06/plot.html");
}
