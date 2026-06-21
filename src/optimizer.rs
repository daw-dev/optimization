use itertools::Itertools;

pub trait Optimize<Problem, StartingGuess, Step = StartingGuess> {
    fn optimize(
        &self,
        problem: Problem,
        starting_guess: StartingGuess,
    ) -> impl Iterator<Item = Step>;

    fn ith_guess(&self, problem: Problem, starting_guess: StartingGuess, iterations: usize) -> Step
    where
        StartingGuess: Clone,
        Step: From<StartingGuess>,
    {
        self.optimize(problem, starting_guess.clone())
            .take(iterations)
            .last()
            .unwrap_or(starting_guess.into())
    }

    fn solution(&self, problem: Problem, starting_guess: StartingGuess) -> Step
    where
        StartingGuess: Clone,
        Step: From<StartingGuess>,
    {
        self.optimize(problem, starting_guess.clone())
            .last()
            .unwrap_or(starting_guess.into())
    }
}

pub trait TryOptimize<Problem, StartingGuess, Step = StartingGuess> {
    type Error;

    fn try_optimize(
        &self,
        problem: Problem,
        starting_guess: StartingGuess,
    ) -> impl Iterator<Item = Result<Step, Self::Error>>;

    fn try_ith_guess(
        &self,
        problem: Problem,
        starting_guess: StartingGuess,
        iterations: usize,
    ) -> Result<Step, Self::Error>
    where
        StartingGuess: Clone,
        Step: From<StartingGuess>,
    {
        self.try_optimize(problem, starting_guess.clone())
            .process_results(|iter| {
                iter.take(iterations)
                    .last()
                    .unwrap_or(starting_guess.into())
            })
    }

    fn try_solution(
        &self,
        problem: Problem,
        starting_guess: StartingGuess,
    ) -> Result<Step, Self::Error>
    where
        StartingGuess: Clone,
        Step: From<StartingGuess>,
    {
        self.try_optimize(problem, starting_guess.clone())
            .process_results(|iter| iter.last().unwrap_or(starting_guess.into()))
    }
}
