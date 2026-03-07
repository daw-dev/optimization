pub struct Then<T, U> {
    first: T,
    second: U,
}

impl<const DIM: usize, T, U> Optimizer<DIM> for Then<T, U>
where
    T: Optimizer<DIM>,
    U: Optimizer<DIM>,
{
    fn optimize<F: Function<DIM>>(self, func: &F, starting_guess: [f64; DIM]) -> [f64; DIM] {
        self.second
            .optimize(func, self.first.optimize(func, starting_guess))
    }
}

pub trait Function<const DIM: usize> {
    fn compute(&self, point: [f64; DIM]) -> f64;
}

impl<F> Function<1> for F
where
    F: Fn(f64) -> f64,
{
    fn compute(&self, point: [f64; 1]) -> f64 {
        self(point[0])
    }
}

// pub trait Continuous<const ORDER: usize>
// where
//     [(); ORDER]:,
//     Self: Continuous<{ ORDER - 1 }>,
// {
//
// }

pub trait Optimizer<const DIM: usize> {
    fn optimize<F: Function<DIM>>(self, func: &F, starting_guess: [f64; DIM]) -> [f64; DIM];

    fn then<U: Optimizer<DIM>>(self, other: U) -> Then<Self, U>
    where
        Self: Sized,
    {
        Then {
            first: self,
            second: other,
        }
    }
}
