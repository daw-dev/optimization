use std::{
    array,
    fmt::Display,
    ops::{
        Add, AddAssign, BitXor, BitXorAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
    },
};

pub mod conjugate;
pub mod constrained;
pub mod genetic;
pub mod newton_raphson;
pub mod simplex;

#[derive(Debug, Clone, Copy, Default)]
pub struct Matrix<const N: usize, const M: usize, T>(pub [[T; M]; N]);

pub type SquareMatrix<const N: usize, T> = Matrix<N, N, T>;

pub type Column<const N: usize, T> = Matrix<N, 1, T>;

pub type Row<const M: usize, T> = Matrix<1, M, T>;

pub type Value<T> = SquareMatrix<1, T>;

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
    pub fn inverse(&self) -> Option<Self> {
        let mut a = self.0.clone();
        let mut inv = Self::identity().0;

        for i in 0..N {
            let mut pivot_row = i;
            let mut max_val = a[i][i].abs();

            for j in (i + 1)..N {
                let val = a[j][i].abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = j;
                }
            }

            if max_val < f64::EPSILON {
                return None;
            }

            if i != pivot_row {
                a.swap(i, pivot_row);
                inv.swap(i, pivot_row);
            }

            let pivot_val = a[i][i];
            for j in 0..N {
                a[i][j] /= pivot_val;
                inv[i][j] /= pivot_val;
            }

            for j in 0..N {
                if i != j {
                    let factor = a[j][i];
                    for k in 0..N {
                        a[j][k] -= factor * a[i][k];
                        inv[j][k] -= factor * inv[i][k];
                    }
                }
            }
        }

        Some(Self(inv))
    }
}

impl<const N: usize, T> Column<N, T> {
    pub fn new_column(col: [T; N]) -> Self {
        Self(col.map(|v| [v]))
    }

    pub fn into_column(self) -> [T; N] {
        self.0.map(|[v]| v)
    }
}

impl<const M: usize, T> Row<M, T> {
    pub fn new_row(col: [T; M]) -> Self {
        Self([col])
    }

    pub fn into_row(self) -> [T; M] {
        let [vec] = self.0;
        vec
    }
}

impl<T> Value<T> {
    pub fn new_value(val: T) -> Self {
        Self([[val]])
    }

    pub fn into_value(self) -> T {
        let [[val]] = self.0;
        val
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

impl<const N: usize, const M: usize, T, U, V> Sub<Matrix<N, M, U>> for Matrix<N, M, T>
where
    T: Sub<U, Output = V>,
{
    type Output = Matrix<N, M, V>;

    fn sub(self, rhs: Matrix<N, M, U>) -> Self::Output {
        let mut left_rows = self.0.into_iter();
        let mut right_rows = rhs.0.into_iter();
        Matrix(array::from_fn(|_| {
            let mut left_values = left_rows.next().unwrap().into_iter();
            let mut right_values = right_rows.next().unwrap().into_iter();

            array::from_fn(|_| {
                let left = left_values.next().unwrap();
                let right = right_values.next().unwrap();

                left - right
            })
        }))
    }
}

impl<const N: usize, const M: usize, T, U> SubAssign<Matrix<N, M, U>> for Matrix<N, M, T>
where
    T: SubAssign<U>,
{
    fn sub_assign(&mut self, rhs: Matrix<N, M, U>) {
        self.0
            .iter_mut()
            .zip(rhs.0.into_iter())
            .for_each(|(dst, src)| {
                dst.iter_mut().zip(src.into_iter()).for_each(|(dst, src)| {
                    *dst -= src;
                })
            });
    }
}

impl<const N: usize, const M: usize, T> Neg for Matrix<N, M, T>
where
    T: Neg,
{
    type Output = Matrix<N, M, T::Output>;

    fn neg(self) -> Self::Output {
        Matrix(self.0.map(|row| row.map(Neg::neg)))
    }
}

impl<const N: usize, const M: usize, T, U> Mul<U> for Matrix<N, M, T>
where
    T: Mul<U>,
    U: Copy,
{
    type Output = Matrix<N, M, T::Output>;

    fn mul(self, rhs: U) -> Self::Output {
        Matrix(self.0.map(|row| row.map(|val| val * rhs)))
    }
}

impl<const N: usize, const M: usize, T, U> MulAssign<Matrix<N, M, U>> for Matrix<N, M, T>
where
    T: MulAssign<U>,
{
    fn mul_assign(&mut self, rhs: Matrix<N, M, U>) {
        self.0
            .iter_mut()
            .zip(rhs.0.into_iter())
            .for_each(|(dst, src)| {
                dst.iter_mut().zip(src.into_iter()).for_each(|(dst, src)| {
                    *dst *= src;
                })
            });
    }
}

impl<const A: usize, const B: usize, const C: usize, T, U> BitXor<Matrix<B, C, U>>
    for Matrix<A, B, T>
where
    T: Mul<U> + Copy,
    U: Copy,
    T::Output: Add<Output = T::Output> + Default,
{
    type Output = Matrix<A, C, T::Output>;
    fn bitxor(self, rhs: Matrix<B, C, U>) -> Self::Output {
        Matrix(core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                (0..B)
                    .map(|k| self.0[i][k] * rhs.0[k][j])
                    .fold(T::Output::default(), |acc, val| acc + val)
            })
        }))
    }
}

impl<const N: usize, T> BitXorAssign for SquareMatrix<N, T>
where
    Self: BitXor<Output = Self> + Default,
{
    fn bitxor_assign(&mut self, rhs: Self) {
        let owned = std::mem::take(self);
        *self = owned ^ rhs;
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

impl<const N: usize, const M: usize, T> IndexMut<(usize, usize)> for Matrix<N, M, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1]
    }
}

impl<const N: usize, const M: usize, T> Index<(usize, usize)> for Matrix<N, M, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1]
    }
}
