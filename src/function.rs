use std::array;

pub trait Function<X, Y> {
    fn compute(&self, point: X) -> Y;
}

impl<'a, X, Y> Function<X, Y> for Box<dyn Function<X, Y> + 'a> {
    fn compute(&self, point: X) -> Y {
        self.as_ref().compute(point)
    }
}

impl<'a, X, Y> Function<X, Y> for &'a dyn Function<X, Y> {
    fn compute(&self, x: X) -> Y {
        (**self).compute(x)
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

pub trait Gradient<'a, const N: usize, X>: Function<[X; N], X> {
    type Result: Function<[X; N], [X; N]> + 'a;

    fn gradient(&'a self, difference: X) -> Self::Result;
}

impl<'a, const N: usize, F> Gradient<'a, N, f64> for F
where
    F: Function<[f64; N], f64>,
{
    type Result = [Box<dyn Function<[f64; N], f64> + 'a>; N];
    fn gradient(&'a self, difference: f64) -> Self::Result {
        array::from_fn(|i| {
            Box::new(move |point: [f64; N]| -> f64 {
                (|x: f64| {
                    let mut point = point;
                    point[i] = x;
                    self.compute(point)
                })
                .derivative(difference)
                .compute(point[i])
            }) as Box<dyn Function<[f64; N], f64> + '_>
        })
    }
}

pub trait Hessian<'a, const N: usize, X> {
    type Result: Function<[X; N], [[X; N]; N]> + 'a;

    fn hessian(&'a self, difference: X) -> Self::Result;
}

impl<'a, const N: usize, F> Hessian<'a, N, f64> for &'a F
where
    F: Function<[f64; N], f64> + 'a,
{
    type Result = Box<dyn Function<[f64; N], [[f64; N]; N]> + 'a>;

    fn hessian(&'a self, difference: f64) -> Self::Result {
        let grad = self.gradient(difference);

        Box::new(move |point: [f64; N]| -> [[f64; N]; N] {
            array::from_fn(|i| {
                let grad_i = &grad[i];
                array::from_fn(|j| {
                    (|x: f64| {
                        let mut p = point;
                        p[j] = x;
                        grad_i.compute(p)
                    })
                    .derivative(difference)
                    .compute(point[j])
                })
            })
        })
    }
}
