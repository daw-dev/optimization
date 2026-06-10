#![allow(non_snake_case)]

use optimization::quadratic::{Column, SquareMatrix, conjugate::Conjugate};

pub fn main() {
    const N: usize = 5;

    let A = SquareMatrix::<N, f64>::randomized(0.0..2.0);
    let A = A ^ A.transpose() + SquareMatrix::identity();

    println!("A");
    println!("{A}");

    let B = Column::<N, f64>::randomized(0.0..2.0);

    println!("B");
    println!("{B}");

    let min = (-A.inverse().unwrap() ^ B).into_column();
    println!("{min:?}");

    let result = Conjugate.optimize(A, B, Column::randomized(0.0..2.0));

    match result {
        Ok(min) => println!("result is {:?}", min.last().unwrap()),
        Err(err) => eprintln!("{err}"),
    }
}
