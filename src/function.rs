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

pub trait Differentiate<'a, Point, X> {
    type Result;
    fn differentiate(&'a self, difference: X) -> Self::Result;
}

impl<'a, F> Differentiate<'a, f64, f64> for F
where
    F: Function<f64, f64> + 'a,
{
    type Result = Box<dyn Function<f64, f64> + 'a>;
    fn differentiate(&'a self, difference: f64) -> Self::Result {
        Box::new(move |x| {
            (self.compute(x + difference) - self.compute(x - difference)) / difference / 2.0
        })
    }
}

impl<'a, const N: usize, F> Differentiate<'a, [f64; N], f64> for F
where
    F: Function<[f64; N], f64> + 'a,
{
    type Result = [Box<dyn Function<[f64; N], f64> + 'a>; N];
    fn differentiate(&'a self, difference: f64) -> Self::Result {
        array::from_fn(|i| {
            Box::new(move |point: [f64; N]| -> f64 {
                (|x: f64| {
                    let mut point = point;
                    point[i] = x;
                    self.compute(point)
                })
                .differentiate(difference)
                .compute(point[i])
            }) as Box<dyn Function<[f64; N], f64> + '_>
        })
    }
}

impl<'a, const N: usize, F, Point> Differentiate<'a, Point, f64> for [F; N]
where
    F: Differentiate<'a, Point, f64> + 'a,
{
    type Result = [F::Result; N];
    fn differentiate(&'a self, difference: f64) -> Self::Result {
        array::from_fn(|i| self[i].differentiate(difference))
    }
}
