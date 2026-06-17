use std::marker::PhantomData;

use crate::functions::Function;

#[derive(Debug, Clone)]
pub struct Chain<T, U, IntermediateGuess> {
    first: T,
    second: U,
    intermediate_guess: PhantomData<IntermediateGuess>,
}

impl<T, U, X, Y, StartingGuess, IntermediateGuess, FinalGuess>
    Optimizer<X, Y, StartingGuess, FinalGuess> for Chain<T, U, IntermediateGuess>
where
    T: Optimizer<X, Y, StartingGuess, IntermediateGuess>,
    U: Optimizer<X, Y, IntermediateGuess, FinalGuess>,
{
    fn optimize<F: Function<X, Y>>(
        self,
        func: &F,
        starting_guess: StartingGuess,
    ) -> impl Iterator<Item = FinalGuess> {
        self.second.optimize(
            func,
            self.first.optimize(func, starting_guess).last().unwrap(),
        )
    }
}

pub trait Optimizer<X, Y, StartingGuess, FinalGuess = StartingGuess> {
    fn optimize<F: Function<X, Y>>(
        self,
        func: &F,
        starting_guess: StartingGuess,
    ) -> impl Iterator<Item = FinalGuess>;

    fn chain<U>(self, other: U) -> Chain<Self, U, FinalGuess>
    where
        Self: Sized,
    {
        Chain {
            first: self,
            second: other,
            intermediate_guess: PhantomData,
        }
    }
}
