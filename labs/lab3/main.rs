use optimization::quadratic::SquareMatrix;

pub fn main() {
    let matrix = SquareMatrix::<5, f64>::randomized(0.0..2.0);
    let matrix = matrix * matrix.transpose() + SquareMatrix::identity();
    println!("{matrix}");
}
