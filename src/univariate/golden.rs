use crate::helpers::Iterations;
use crate::optimizer::TryOptimize;
use crate::{function::Function, helpers::Precision};
use std::{cmp::Ordering, f64, ops::Range};

#[derive(Clone)]
pub struct GoldenRatio<S> {
    stopping_condition: S,
}

impl<S> GoldenRatio<S> {
    pub const GAMMA: f64 = f64::consts::GOLDEN_RATIO - 1f64;

    pub fn iterations_from_precision(precision: f64, starting_interval: &Range<f64>) -> usize {
        (precision / (starting_interval.end - starting_interval.start))
            .log(Self::GAMMA)
            .ceil() as usize
    }

    pub fn new(stopping_condition: S) -> Self {
        Self { stopping_condition }
    }
}

impl<F> TryOptimize<&F, Range<f64>> for GoldenRatio<Precision>
where
    F: Function<f64, f64> + ?Sized,
{
    type Error = String;
    fn try_optimize(
        &self,
        func: &F,
        starting_guess: Range<f64>,
    ) -> impl Iterator<Item = Result<Range<f64>, String>> {
        fn find_points(start: f64, end: f64) -> [f64; 4] {
            let first = end - GoldenRatio::<Precision>::GAMMA * (end - start);
            let second = start + GoldenRatio::<Precision>::GAMMA * (end - start);
            [start, first, second, end]
        }

        let iterations =
            Self::iterations_from_precision(self.stopping_condition.0, &starting_guess);

        let mut points = find_points(starting_guess.start, starting_guess.end).map(|x| (x, None));

        std::iter::once(Ok(starting_guess.clone())).chain((0..iterations).map(move |_| {
            let [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] =
                points.map(|(x, y)| (x, y.unwrap_or_else(|| func.compute(x))));
            match (y1.total_cmp(&y2), y2.total_cmp(&y3), y3.total_cmp(&y4)) {
                (Ordering::Less, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x1, x2);
                    points = [(x1, Some(y1)), (x2, None), (x3, None), (x4, Some(y2))];
                }
                (Ordering::Greater, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x1, x3);
                    points = [(x1, Some(y1)), (x2, None), (x3, Some(y2)), (x4, Some(y3))];
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x2, x4);
                    points = [(x1, Some(y2)), (x2, Some(y3)), (x3, None), (x4, Some(y4))];
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Greater) => {
                    let [x1, x2, x3, x4] = find_points(x3, x4);
                    points = [(x1, Some(y3)), (x2, None), (x3, None), (x4, Some(y4))];
                }
                (Ordering::Greater, Ordering::Equal, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x2, x3);
                    points = [(x1, Some(y2)), (x2, None), (x3, None), (x4, Some(y3))];
                }
                t => {
                    return Err(format!("this function is not unimodal: {t:?}"));
                }
            }
            Ok(points[1].0..points[2].0)
        }))
    }
}

impl<F> TryOptimize<&F, Range<f64>> for GoldenRatio<Iterations>
where
    F: Function<f64, f64> + ?Sized,
{
    type Error = String;
    fn try_optimize(
        &self,
        func: &F,
        starting_guess: Range<f64>,
    ) -> impl Iterator<Item = Result<Range<f64>, String>> {
        fn find_points(start: f64, end: f64) -> [f64; 4] {
            let first = end - GoldenRatio::<Precision>::GAMMA * (end - start);
            let second = start + GoldenRatio::<Precision>::GAMMA * (end - start);
            [start, first, second, end]
        }

        let iterations = self.stopping_condition.0;

        let mut points = find_points(starting_guess.start, starting_guess.end).map(|x| (x, None));

        std::iter::once(Ok(starting_guess.clone())).chain((0..iterations).map(move |_| {
            let [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] =
                points.map(|(x, y)| (x, y.unwrap_or_else(|| func.compute(x))));
            match (y1.total_cmp(&y2), y2.total_cmp(&y3), y3.total_cmp(&y4)) {
                (Ordering::Less, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x1, x2);
                    points = [(x1, Some(y1)), (x2, None), (x3, None), (x4, Some(y2))];
                }
                (Ordering::Greater, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x1, x3);
                    points = [(x1, Some(y1)), (x2, None), (x3, Some(y2)), (x4, Some(y3))];
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x2, x4);
                    points = [(x1, Some(y2)), (x2, Some(y3)), (x3, None), (x4, Some(y4))];
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Greater) => {
                    let [x1, x2, x3, x4] = find_points(x3, x4);
                    points = [(x1, Some(y3)), (x2, None), (x3, None), (x4, Some(y4))];
                }
                (Ordering::Greater, Ordering::Equal, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x2, x3);
                    points = [(x1, Some(y2)), (x2, None), (x3, None), (x4, Some(y3))];
                }
                t => {
                    return Err(format!("this function is not unimodal: {t:?}"));
                }
            }
            Ok(points[1].0..points[2].0)
        }))
    }
}

