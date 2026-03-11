pub struct Then<T, U> {
    first: T,
    second: U,
}

impl<T, U, X, Y> Optimizer<X, Y> for Then<T, U>
where
    T: Optimizer<X, Y>,
    U: Optimizer<X, Y>,
{
    fn optimize<F: Function<X, Y>>(self, func: &F, starting_guess: X) -> X {
        self.second
            .optimize(func, self.first.optimize(func, starting_guess))
    }
}

pub trait Function<X, Y> {
    fn compute(&self, point: X) -> Y;
}

impl<F, X, Y> Function<X, Y> for F
where
    F: Fn(X) -> Y,
{
    fn compute(&self, point: X) -> Y {
        self(point)
    }
}

pub trait Optimizer<X, Y> {
    fn optimize<F: Function<X, Y>>(self, func: &F, starting_guess: X) -> X;

    fn then<U, YPrime>(self, other: U) -> Then<Self, U>
    where
        Self: Sized,
        U: Optimizer<X, Y>,
    {
        Then {
            first: self,
            second: other,
        }
    }
}
