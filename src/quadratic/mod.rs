use std::{
    array,
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign},
};

#[derive(Debug, Clone, Copy)]
pub struct Matrix<const N: usize, const M: usize, T>([[T; M]; N]);

pub type SquareMatrix<const N: usize, T> = Matrix<N, N, T>;

impl<const N: usize, const M: usize, T> Matrix<N, M, T> {
    pub fn randomized<R>(range: R) -> Self
    where
        T: rand::distr::uniform::SampleUniform,
        R: rand::distr::uniform::SampleRange<T> + Clone,
    {
        Self(array::from_fn(|_| {
            array::from_fn(|_| rand::random_range(range.clone()))
        }))
    }

    pub fn transpose(self) -> Matrix<M, N, T>
    where
        T: Copy,
    {
        let mut result = [[{ None }; N]; M];
        let mut i = 0;
        while i < N {
            let mut j = 0;
            while j < M {
                result[j][i] = Some(self.0[i][j]);
                j += 1;
            }
            i += 1;
        }
        Matrix(result.map(|row| row.map(Option::unwrap)))
    }
}

impl<const N: usize> SquareMatrix<N, f64> {
    pub fn identity() -> Self {
        Self(array::from_fn(|i| {
            array::from_fn(|j| if i == j { 1.0 } else { 0.0 })
        }))
    }
}

impl<const N: usize, const M: usize, T, U, V> Add<Matrix<N, M, U>> for Matrix<N, M, T>
where
    T: Add<U, Output = V>,
{
    type Output = Matrix<N, M, V>;

    fn add(self, rhs: Matrix<N, M, U>) -> Self::Output {
        let mut left_rows = self.0.into_iter();
        let mut right_rows = rhs.0.into_iter();
        Matrix(array::from_fn(|_| {
            let mut left_values = left_rows.next().unwrap().into_iter();
            let mut right_values = right_rows.next().unwrap().into_iter();

            array::from_fn(|_| {
                let left = left_values.next().unwrap();
                let right = right_values.next().unwrap();

                left + right
            })
        }))
    }
}

impl<const N: usize, const M: usize, T, U> AddAssign<Matrix<N, M, U>> for Matrix<N, M, T>
where
    T: AddAssign<U>,
{
    fn add_assign(&mut self, rhs: Matrix<N, M, U>) {
        self.0
            .iter_mut()
            .zip(rhs.0.into_iter())
            .for_each(|(dst, src)| {
                dst.iter_mut().zip(src.into_iter()).for_each(|(dst, src)| {
                    *dst += src;
                })
            });
    }
}

impl<'a, 'b, const A: usize, const B: usize, const C: usize, T, U, V> Mul<&'b Matrix<B, C, U>>
    for &'a Matrix<A, B, T>
where
    &'a T: Mul<&'b U, Output = V>,
    V: Add<V, Output = V>,
{
    type Output = Matrix<A, C, V>;

    fn mul(self, rhs: &'b Matrix<B, C, U>) -> Self::Output {
        Matrix(array::from_fn(|i| {
            array::from_fn(|j| {
                let mut acc = &self.0[i][0] * &rhs.0[0][j];
                for k in 1..B {
                    acc = acc + (&self.0[i][k] * &rhs.0[k][j]);
                }
                acc
            })
        }))
    }
}

impl<const A: usize, const B: usize, const C: usize, T, U, V> Mul<Matrix<B, C, U>>
    for Matrix<A, B, T>
where
    for<'a, 'b> &'a T: Mul<&'b U, Output = V>,
    V: Add<V, Output = V>,
{
    type Output = Matrix<A, C, V>;
    fn mul(self, rhs: Matrix<B, C, U>) -> Self::Output {
        &self * &rhs // Just calls the reference version
    }
}

impl<const N: usize, T> MulAssign for SquareMatrix<N, T>
where
    for<'a, 'b> &'a Matrix<N, N, T>: Mul<&'b Matrix<N, N, T>, Output = Matrix<N, N, T>>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(&rhs);
    }
}

impl<const N: usize, T> MulAssign<&Self> for SquareMatrix<N, T>
where
    for<'a, 'b> &'a Matrix<N, N, T>: Mul<&'b Matrix<N, N, T>, Output = Matrix<N, N, T>>,
{
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self.mul(rhs);
    }
}

impl<const N: usize, const M: usize, T: Display> Display for Matrix<N, M, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..N {
            for j in 0..M {
                write!(f, "{:^5.2}", self.0[i][j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
