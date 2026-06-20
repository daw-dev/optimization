#![allow(non_snake_case)]

use optimization::{
    optimizer::TryOptimize,
    quadratic::{
        Column, SquareMatrix,
        conjugate::{Conjugate, PerfectQuadraticProblem},
    },
};

pub fn main() {
    const N: usize = 5;

    let A = SquareMatrix::<N, f64>::randomized(0.0..2.0);
    let A = (A * A.transpose()) + SquareMatrix::identity();

    println!("A");
    println!("{A}");

    let B = Column::<N, f64>::randomized(0.0..2.0);

    println!("B");
    println!("{B}");

    let min = (A.inverse().unwrap() * B).into_column();
    println!("{min:?}");

    let result = Conjugate.try_solution(
        PerfectQuadraticProblem { matrix: A, b: B },
        Column::randomized(0.0..2.0),
    );

    match result {
        Ok(min) => println!("result is {:?}", min),
        Err(err) => eprintln!("{err}"),
    }
}
