use std::array;

pub trait Function<X, Y> {
    fn compute(&self, point: X) -> Y;
}

impl<'a, X, Y> Function<X, Y> for Box<dyn Function<X, Y> + 'a> {
    fn compute(&self, point: X) -> Y {
        self.as_ref().compute(point)
    }
}

impl<F, X, Y> Function<X, Y> for F
where
    F: Fn(X) -> Y,
{
    fn compute(&self, point: X) -> Y {
        self(point)
    }
}

impl<const N: usize, F, X, Y> Function<X, [Y; N]> for [F; N]
where 
    F: Function<X, Y>,
    X: Copy,
{
    fn compute(&self, point: X) -> [Y; N] {
        array::from_fn(|i| self[i].compute(point))
    }
}

pub trait Derivative<X>: Function<X, X> {
    fn derivative(&self, difference: X) -> impl Function<X, X>;
}

impl<F> Derivative<f64> for F
where
    F: Function<f64, f64>,
{
    fn derivative(&self, difference: f64) -> impl Function<f64, f64> {
        move |x| (self.compute(x + difference) - self.compute(x - difference)) / difference / 2.0
    }
}

pub trait Gradient<const N: usize, X> {
    fn gradient(&self, difference: X) -> [Box<dyn Function<[X; N], X> + '_>; N];
}

// impl<const N: usize, F> Gradient<N, f64> for F
// where
//     F: Function<[f64; N], f64>,
// {
//     fn gradient(&self, difference: f64) -> [Box<dyn Function<[f64; N], f64> + '_>; N] {
//         array::from_fn(|i| {
//             Box::new(move |point: [f64; N]| -> f64 {
//                 (|x: f64| {
//                     let mut point = point.clone();
//                     point[i] = x;
//                     self.compute(point)
//                 })
//                 .derivative(difference)
//                 .compute(point[i])
//             }) as Box<dyn Function<[f64; N], f64> + '_>
//         })
//     }
// }

impl<const N: usize, F> Gradient<N, f64> for &F
where
    F: Function<[f64; N], f64>,
{
    fn gradient(&self, difference: f64) -> [Box<dyn Function<[f64; N], f64> + '_>; N] {
        array::from_fn(|i| {
            Box::new(move |point: [f64; N]| -> f64 {
                (|x: f64| {
                    let mut point = point.clone();
                    point[i] = x;
                    self.compute(point)
                })
                .derivative(difference)
                .compute(point[i])
            }) as Box<dyn Function<[f64; N], f64> + '_>
        })
    }
}
