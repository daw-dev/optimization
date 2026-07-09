#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use optimization::{
    linalg::{Column, Matrix},
    multivariate::constrained::{EqualityConstrainedQP, NewtonRaphsonQP, QPStep},
    optimizer::TryOptimize,
};
use plotly::{
    Layout, Plot, Scatter,
    common::{Marker, Mode},
    layout::Axis,
};

fn main() {
    println!("========================================");
    println!("  LAB 6: Equality Constrained QP");
    println!("========================================");

    // -------------------------------------------------------------------------
    // Exercise 1: 3D Quadratic Programming Problem (3 variables, 2 constraints)
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 1: 3D QP with 2 Equality Constraints ---");
    // Minimize f(x) = x1^2 + x2^2 + x3^2
    // subject to:
    //   x1 + 2x2 - x3 = 4
    //   x1 - x2 + x3 = 2
    //
    // Q = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    // c = [0, 0, 0]
    // A = [[1, 2, -1], [1, -1, 1]]
    // b = [4, 2]

    let q1 = Matrix([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
    let c1 = Column::new_column([0.0, 0.0, 0.0]);
    let a1 = Matrix([[1.0, 2.0, -1.0], [1.0, -1.0, 1.0]]);
    let b1 = Column::new_column([4.0, 2.0]);

    let problem1 = EqualityConstrainedQP {
        q: q1,
        c: c1,
        a: a1,
        b: b1,
    };

    let start_guess1 = QPStep::from(Column::new_column([0.0, 0.0, 0.0]));
    let solver1 = NewtonRaphsonQP::<3, 2>;

    println!("Starting Guess x0: [0.0, 0.0, 0.0]");
    println!("Solving KKT System...");

    let mut steps_ex1 = Vec::new();
    let mut x1_val = Vec::new();
    let mut x2_val = Vec::new();
    let mut x3_val = Vec::new();

    let mut step = 0;
    let mut final_step1 = None;

    for res in solver1.try_optimize(problem1, start_guess1) {
        match res {
            Ok(guess) => {
                let x = guess.x.clone().into_column();
                steps_ex1.push(step as f64);
                x1_val.push(x[0]);
                x2_val.push(x[1]);
                x3_val.push(x[2]);
                println!(
                    "  Iteration {}: x = {:.5?}, lambda = {:.5?}",
                    step,
                    x,
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

    let sol1 = final_step1.expect("NewtonRaphsonQP should converge in at least one iteration");
    println!("\nExercise 1 Summary:");
    println!(
        "  Optimal Solution x*:     {:.5?}",
        sol1.x.clone().into_column()
    );
    println!(
        "  Lagrange Multipliers:    {:.5?}",
        sol1.lambda.clone().into_column()
    );
    println!("  Steps to Converge:       {}", step);

    let mut plot1 = Plot::new();
    plot1.add_trace(
        Scatter::new(steps_ex1.clone(), x1_val)
            .mode(Mode::LinesMarkers)
            .name("x1"),
    );
    plot1.add_trace(
        Scatter::new(steps_ex1.clone(), x2_val)
            .mode(Mode::LinesMarkers)
            .name("x2"),
    );
    plot1.add_trace(
        Scatter::new(steps_ex1.clone(), x3_val)
            .mode(Mode::LinesMarkers)
            .name("x3"),
    );
    plot1.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("Iteration Step"))
            .y_axis(Axis::new().title("Coordinate Value")),
    );

    // -------------------------------------------------------------------------
    // Exercise 2: 2D Quadratic Programming Problem (2 variables, 1 constraint)
    // -------------------------------------------------------------------------
    println!("\n--- Exercise 2: 2D QP with 1 Equality Constraint ---");
    // Minimize f(x) = x1^2 + 2x2^2 - 2x1x2 - 2x1 - 6x2
    // subject to:
    //   x1 + x2 = 3
    //
    // Q = [[2, -2], [-2, 4]]
    // c = [-2, -6]
    // A = [[1, 1]]
    // b = [3]

    let q2 = Matrix([[2.0, -2.0], [-2.0, 4.0]]);
    let c2 = Column::new_column([-2.0, -6.0]);
    let a2 = Matrix([[1.0, 1.0]]);
    let b2 = Column::new_column([3.0]);

    let problem2 = EqualityConstrainedQP {
        q: q2,
        c: c2,
        a: a2,
        b: b2,
    };

    let start_guess2 = QPStep::from(Column::new_column([0.0, 0.0]));
    let solver2 = NewtonRaphsonQP::<2, 1>;

    let mut path_x1 = vec![0.0];
    let mut path_x2 = vec![0.0];
    let mut step_ex2 = 0;
    let mut final_step2 = None;

    for res in solver2.try_optimize(problem2, start_guess2) {
        match res {
            Ok(guess) => {
                let x = guess.x.clone().into_column();
                path_x1.push(x[0]);
                path_x2.push(x[1]);
                println!(
                    "  Iteration {}: x = {:.5?}, lambda = {:.5?}",
                    step_ex2,
                    x,
                    guess.lambda.clone().into_column()
                );
                step_ex2 += 1;
                final_step2 = Some(guess);
            }
            Err(err) => {
                println!("  Error: {}", err);
                break;
            }
        }
    }

    let sol2 = final_step2.expect("NewtonRaphsonQP should converge in at least one iteration");
    println!("\nExercise 2 Summary:");
    println!(
        "  Optimal Solution x*:     {:.5?}",
        sol2.x.clone().into_column()
    );
    println!(
        "  Lagrange Multipliers:    {:.5?}",
        sol2.lambda.clone().into_column()
    );
    println!("  Steps to Converge:       {}", step_ex2);

    let mut plot2 = Plot::new();

    let c_line_x = vec![-0.5, 3.5];
    let c_line_y = vec![3.5, -0.5];
    let trace_constraint = Scatter::new(c_line_x, c_line_y)
        .mode(Mode::Lines)
        .name("Constraint: x1 + x2 = 3");
    plot2.add_trace(trace_constraint);

    let trace_path = Scatter::new(path_x1, path_x2)
        .mode(Mode::LinesMarkers)
        .marker(Marker::new().size(8).color("lime"))
        .name("Newton-Raphson Path");
    plot2.add_trace(trace_path);

    let trace_opt = Scatter::new(vec![1.4], vec![1.6])
        .mode(Mode::Markers)
        .marker(
            Marker::new()
                .size(12)
                .color("red")
                .symbol(plotly::common::MarkerSymbol::Star),
        )
        .name("Analytical Solution (1.4, 1.6)");
    plot2.add_trace(trace_opt);

    plot2.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("x1").range(vec![-0.5, 3.5]))
            .y_axis(Axis::new().title("x2").range(vec![-0.5, 3.5])),
    );

    optimization::helpers::save_dashboard(
        "labs/lab06/plot.html",
        "LAB 6: Quadratic Programming (QP)",
        &[
            ("Exercise 1 Coordinate Convergence (n=3, m=2)", &plot1),
            (
                "Exercise 2 Optimization Path & Constraint (n=2, m=1)",
                &plot2,
            ),
        ],
    )
    .unwrap();

    println!("\nSaved plots to: labs/lab06/plot.html");
    optimization::helpers::prompt_and_open_dashboard("labs/lab06/plot.html");
}
