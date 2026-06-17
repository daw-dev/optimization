use std::marker::PhantomData;

use crate::functions::Function;

#[derive(Debug, Clone)]
pub struct Optimization<I> {
    iter: I,
}

impl<I> Optimization<I> {
    pub fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I: Iterator> Iterator for Optimization<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub trait OptimizationResult: Iterator<Item = Self::Guess> + Sized {
    type Guess;

    fn guess(self) -> Self::Guess;

    fn guesses(self) -> Self {
        self
    }
}

impl<I> OptimizationResult for Optimization<I>
where
    I: Iterator,
{
    type Guess = I::Item;

    fn guess(self) -> Self::Guess {
        self.iter.last().expect("optimizer produced no guesses")
    }
}

#[derive(Debug, Clone)]
pub struct TryOptimization<I> {
    iter: I,
}

impl<I> TryOptimization<I> {
    pub fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I: Iterator> Iterator for TryOptimization<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub trait TryOptimizationResult:
    Iterator<Item = Result<Self::Guess, <Self as TryOptimizationResult>::Error>> + Sized
{
    type Guess;
    type Error;

    fn guess(self) -> Result<Self::Guess, <Self as TryOptimizationResult>::Error>;

    fn guesses(self) -> Self {
        self
    }
}

impl<I, Guess, Error> TryOptimizationResult for TryOptimization<I>
where
    I: Iterator<Item = Result<Guess, Error>>,
{
    type Guess = Guess;
    type Error = Error;

    fn guess(self) -> Result<Self::Guess, <Self as TryOptimizationResult>::Error> {
        let mut last = None;
        for item in self.iter {
            match item {
                Ok(guess) => last = Some(guess),
                Err(error) => return Err(error),
            }
        }
        Ok(last.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct Chain<T, U, IntermediateGuess> {
    first: T,
    second: U,
    intermediate_guess: PhantomData<IntermediateGuess>,
}

pub trait Optimizer<X, Y, StartingGuess, FinalGuess = StartingGuess> {
    fn optimize<F: Function<X, Y>>(
        self,
        func: &F,
        starting_guess: StartingGuess,
    ) -> impl OptimizationResult<Guess = FinalGuess>;

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

pub trait TryOptimizer<X, Y, StartingGuess, Error, FinalGuess = StartingGuess> {
    fn try_optimize<F: Function<X, Y>>(
        self,
        func: &F,
        starting_guess: StartingGuess,
    ) -> impl TryOptimizationResult<Guess = FinalGuess, Error = Error>;
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
    ) -> impl OptimizationResult<Guess = FinalGuess> {
        let intermediate_guess = self.first.optimize(func, starting_guess).guess();
        self.second.optimize(func, intermediate_guess)
    }
}

impl<T, X, Y, StartingGuess, Error, FinalGuess> TryOptimizer<X, Y, StartingGuess, Error, FinalGuess>
    for T
where
    T: Optimizer<X, Y, StartingGuess, FinalGuess>,
{
    fn try_optimize<F: Function<X, Y>>(
        self,
        func: &F,
        starting_guess: StartingGuess,
    ) -> impl TryOptimizationResult<Guess = FinalGuess, Error = Error> {
        TryOptimization::new(self.optimize(func, starting_guess).guesses().map(Ok))
    }
}
