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
    fn optimize<F: Function<X, Y>>(self, func: &F, starting_guess: StartingGuess) -> FinalGuess {
        self.second
            .optimize(func, self.first.optimize(func, starting_guess))
    }
}

pub struct Map<T, F, IntermediateGuess> {
    optimizer: T,
    mapper: F,
    intermediate_guess: PhantomData<IntermediateGuess>,
}

impl<T, M, X, Y, StartingGuess, IntermediateGuess, FinalGuess>
    Optimizer<X, Y, StartingGuess, FinalGuess> for Map<T, M, IntermediateGuess>
where
    T: Optimizer<X, Y, StartingGuess, IntermediateGuess>,
    M: FnOnce(IntermediateGuess) -> FinalGuess,
{
    fn optimize<F: Function<X, Y>>(self, func: &F, starting_guess: StartingGuess) -> FinalGuess {
        (self.mapper)(self.optimizer.optimize(func, starting_guess))
    }
}

pub trait Optimizer<X, Y, StartingGuess, FinalGuess = StartingGuess> {
    fn optimize<F: Function<X, Y>>(self, func: &F, starting_guess: StartingGuess) -> FinalGuess;

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

    fn map<F>(self, mapper: F) -> Map<Self, F, FinalGuess>
    where
        Self: Sized,
    {
        Map {
            optimizer: self,
            mapper,
            intermediate_guess: PhantomData,
        }
    }
}
