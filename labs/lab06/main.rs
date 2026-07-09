#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use optimization::{
    helpers::{Iterations, Precision},
    linalg::{Column, Matrix},
    multivariate::{
        active_set::{ActiveSetGuess, ActiveSetMethod, InequalityConstrainedQP},
        constrained::{EqualityConstrainedQP, NewtonRaphsonQP},
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
        q: q1.clone(),
        c: c1.clone(),
        a: a1.clone(),
        b: b1.clone(),
    };

    let start_guess1 = Column::zeros();
    let solver1 = NewtonRaphsonQP::<5, 3, Precision>::new(Precision(1e-7));

    println!("Solving KKT System for 5D problem...");
    let mut final_step1 = None;
    let mut step = 0;

    for res in solver1.try_optimize(problem1, start_guess1) {
        match res {
            Ok(guess) => {
                println!(
                    "  Iteration {}: x = {:.5?}, lambda = {:.5?}",
                    step,
                    guess.x.clone().into_column(),
                    guess.lambda.clone().into_column()
                );
                step += 1;
                final_step1 = Some(guess);
            }
            Err(err) => {
                println!("  Error: {}", err);
                break;
            }
        }
    }

    let sol1 = final_step1.expect("NewtonRaphsonQP should converge");
    let x_star = sol1.x.clone();
    let lambda_star = sol1.lambda.clone();
    println!("\nExercise 1 Results:");
    println!("  Optimal Solution x*:     {:.5?}", x_star.clone().into_column());
    println!("  Lagrange Multipliers:    {:.5?}", lambda_star.clone().into_column());

    // Verify Lagrange's Theorem is satisfied: Qx* + c + A^T lambda* = 0
    let grad_lagrangian = (q1 * x_star.clone()) + c1 + (a1.transpose() * lambda_star.clone());
    println!("  Verification (gradient of Lagrangian, target = [0.0; 5]):");
    println!("    {:.5?}", grad_lagrangian.into_column());
    let constr_check = (a1 * x_star.clone()) - b1;
    println!("  Verification (Ax* - b, target = [0.0; 3]):");
    println!("    {:.5?}", constr_check.into_column());


    // -------------------------------------------------------------------------
    // Exercise 2: Local SQP for nonlinear function
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 2: Local SQP for Nonlinear Equality Constraints ---");
    // Minimize f(x) = 1.5 * x2 - 2 * x1^2 - x1^3 + x1^4
    // Subject to: x1^2 + x2^2 = 3

    let mut x_k = [1.0, 5.0];
    let mut lambda_k = 0.0;
    let epsilon = 1e-6;
    let max_iter = 400;

    let mut path_ex2_x1 = vec![x_k[0]];
    let mut path_ex2_x2 = vec![x_k[1]];

    println!("Starting Local SQP from x0 = {:?}, lambda0 = {}", x_k, lambda_k);
    let mut k = 0;
    while k < max_iter {
        let x1: f64 = x_k[0];
        let x2: f64 = x_k[1];

        // 1. Calculate gradient of objective f: g = [ df/dx1, df/dx2 ]
        let g_ex2 = [
            -4.0 * x1 - 3.0 * x1.powi(2) + 4.0 * x1.powi(3),
            1.5
        ];

        // 2. Calculate Hessian of Lagrangian: Qm = [ [d^2 L/dx1^2, 0], [0, d^2 L/dx2^2] ]
        let qm_11 = -4.0 - 6.0 * x1 + 12.0 * x1.powi(2) + 2.0 * lambda_k;
        let qm_22 = 2.0 * lambda_k;
        let qm = Matrix([
            [qm_11, 0.0],
            [0.0, qm_22]
        ]);

        // 3. Calculate constraint Jacobian: Am = [2*x1, 2*x2]
        let am = Matrix([
            [2.0 * x1, 2.0 * x2]
        ]);

        // 4. Constraint value
        let h_val = x1.powi(2) + x2.powi(2) - 3.0;

        // Solve QP subproblem using library's NewtonRaphsonQP
        let subproblem = EqualityConstrainedQP {
            q: qm,
            c: Column::new_column(g_ex2),
            a: am,
            b: Column::new_column([-h_val]), // Ax = b constraint is Am * p = -h
        };

        let sub_solver = NewtonRaphsonQP::<2, 1, Iterations>::new(Iterations(1));
        let mut sub_result = None;
        for res in sub_solver.try_optimize(subproblem, Column::zeros()) {
            if let Ok(guess) = res {
                sub_result = Some(guess);
            }
        }

        let sub_step = sub_result.expect("Subproblem should be solved");
        let p = sub_step.x.into_column();
        let lambda_new = sub_step.lambda.into_column()[0];

        let p_norm = (p[0]*p[0] + p[1]*p[1]).sqrt();
        
        x_k[0] += p[0];
        x_k[1] += p[1];
        lambda_k = lambda_new;

        path_ex2_x1.push(x_k[0]);
        path_ex2_x2.push(x_k[1]);

        println!(
            "  Step {}: x = [{:.5}, {:.5}], lambda = {:.5}, stop_crit (||p||) = {:.3e}",
            k + 1, x_k[0], x_k[1], lambda_k, p_norm
        );

        if p_norm <= epsilon {
            println!("Local SQP converged in {} iterations.", k + 1);
            break;
        }
        k += 1;
    }

    println!("\nExercise 2 Results:");
    println!("  Optimal Solution x*:     {:.5?}", x_k);
    println!("  Lagrange Multiplier:     {:.5}", lambda_k);


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
    let mut path_ex3_x1 = vec![start_guess3.x.0[0][0]];
    let mut path_ex3_x2 = vec![start_guess3.x.0[1][0]];

    let mut iter_ex3 = 0;
    let mut final_guess3 = None;

    for res in solver3.try_optimize(problem3, start_guess3) {
        match res {
            Ok(guess) => {
                println!(
                    "  Iteration {}: x = [{:.4}, {:.4}], W = {:?}",
                    iter_ex3 + 1,
                    guess.x.0[0][0],
                    guess.x.0[1][0],
                    guess.w
                );
                path_ex3_x1.push(guess.x.0[0][0]);
                path_ex3_x2.push(guess.x.0[1][0]);
                iter_ex3 += 1;
                final_guess3 = Some(guess);
            }
            Err(err) => {
                println!("  Error: {}", err);
                break;
            }
        }
    }

    let sol3 = final_guess3.expect("ActiveSetMethod should converge");
    println!("\nExercise 3 Results:");
    println!("  Optimal Solution x*:     [{:.5}, {:.5}]", sol3.x.0[0][0], sol3.x.0[1][0]);


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
