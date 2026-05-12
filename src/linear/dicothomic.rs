use crate::helpers::{Iterations, Precision};
use crate::optimizer::{Optimizer};
use crate::functions::Function;
use std::{cmp::Ordering, ops::Range};

#[derive(Clone)]
pub struct Dicothomic<S> {
    stopping_condition: S,
}

impl<S> Dicothomic<S> {
    pub fn iterations_from_precision(precision: f64, starting_interval: &Range<f64>) -> usize {
        ((starting_interval.end - starting_interval.start) / precision)
            .log2()
            .ceil() as usize
    }

    pub fn new(stopping_condition: S) -> Self {
        Self { stopping_condition }
    }
}

impl Optimizer<f64, f64, Range<f64>, Result<Range<f64>, String>> for Dicothomic<Iterations> {
    fn optimize<F: Function<f64, f64>>(
        self,
        func: &F,
        starting_guess: Range<f64>,
    ) -> Result<Range<f64>, String> {
        fn find_points(start: f64, end: f64) -> [f64; 5] {
            let mid = (start + end) / 2.0;
            let left_quarter = (start + mid) / 2.0;
            let right_quarter = (mid + end) / 2.0;
            [start, left_quarter, mid, right_quarter, end]
        }

        let iterations = self.stopping_condition.0;

        let mut points = find_points(starting_guess.start, starting_guess.end).map(|x| (x, None));

        for _ in 0..iterations {
            let [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)] =
                points.map(|(x, y)| (x, y.unwrap_or_else(|| func.compute(x))));
            match (
                y1.total_cmp(&y2),
                y2.total_cmp(&y3),
                y3.total_cmp(&y4),
                y4.total_cmp(&y5),
            ) {
                (Ordering::Less, Ordering::Less, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4, x5] = find_points(x1, x2);
                    points = [
                        (x1, Some(y1)),
                        (x2, None),
                        (x3, None),
                        (x4, None),
                        (x5, Some(y2)),
                    ]
                }
                (Ordering::Greater, Ordering::Less, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4, x5] = find_points(x1, x3);
                    points = [
                        (x1, Some(y1)),
                        (x2, None),
                        (x3, Some(y2)),
                        (x4, None),
                        (x5, Some(y3)),
                    ]
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4, x5] = find_points(x2, x4);
                    points = [
                        (x1, Some(y2)),
                        (x2, None),
                        (x3, Some(y3)),
                        (x4, None),
                        (x5, Some(y4)),
                    ]
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Greater, Ordering::Less) => {
                    let [x1, x2, x3, x4, x5] = find_points(x3, x5);
                    points = [
                        (x1, Some(y3)),
                        (x2, None),
                        (x3, Some(y4)),
                        (x4, None),
                        (x5, Some(y5)),
                    ]
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Greater, Ordering::Greater) => {
                    let [x1, x2, x3, x4, x5] = find_points(x4, x5);
                    points = [
                        (x1, Some(y4)),
                        (x2, None),
                        (x3, None),
                        (x4, None),
                        (x5, Some(y5)),
                    ]
                }
                (Ordering::Greater, Ordering::Equal, Ordering::Less, Ordering::Less) => {
                    return Ok(x2..x3);
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Equal, Ordering::Less) => {
                    return Ok(x3..x4);
                }
                t => {
                    return Err(format!("this function is not unimodal: {t:?}"));
                }
            }
        }

        Ok(points[0].0..points[4].0)
    }
}

impl Optimizer<f64, f64, Range<f64>, Result<Range<f64>, String>> for Dicothomic<Precision> {
    fn optimize<F: Function<f64, f64>>(
        self,
        func: &F,
        starting_guess: Range<f64>,
    ) -> Result<Range<f64>, String> {
        fn find_points(start: f64, end: f64) -> [f64; 5] {
            let mid = (start + end) / 2.0;
            let left_quarter = (start + mid) / 2.0;
            let right_quarter = (mid + end) / 2.0;
            [start, left_quarter, mid, right_quarter, end]
        }

        let iterations = Self::iterations_from_precision(self.stopping_condition.0, &starting_guess);

        let mut points = find_points(starting_guess.start, starting_guess.end).map(|x| (x, None));

        for _ in 0..iterations {
            let [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)] =
                points.map(|(x, y)| (x, y.unwrap_or_else(|| func.compute(x))));
            match (
                y1.total_cmp(&y2),
                y2.total_cmp(&y3),
                y3.total_cmp(&y4),
                y4.total_cmp(&y5),
            ) {
                (Ordering::Less, Ordering::Less, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4, x5] = find_points(x1, x2);
                    points = [
                        (x1, Some(y1)),
                        (x2, None),
                        (x3, None),
                        (x4, None),
                        (x5, Some(y2)),
                    ]
                }
                (Ordering::Greater, Ordering::Less, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4, x5] = find_points(x1, x3);
                    points = [
                        (x1, Some(y1)),
                        (x2, None),
                        (x3, Some(y2)),
                        (x4, None),
                        (x5, Some(y3)),
                    ]
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4, x5] = find_points(x2, x4);
                    points = [
                        (x1, Some(y2)),
                        (x2, None),
                        (x3, Some(y3)),
                        (x4, None),
                        (x5, Some(y4)),
                    ]
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Greater, Ordering::Less) => {
                    let [x1, x2, x3, x4, x5] = find_points(x3, x5);
                    points = [
                        (x1, Some(y3)),
                        (x2, None),
                        (x3, Some(y4)),
                        (x4, None),
                        (x5, Some(y5)),
                    ]
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Greater, Ordering::Greater) => {
                    let [x1, x2, x3, x4, x5] = find_points(x4, x5);
                    points = [
                        (x1, Some(y4)),
                        (x2, None),
                        (x3, None),
                        (x4, None),
                        (x5, Some(y5)),
                    ]
                }
                (Ordering::Greater, Ordering::Equal, Ordering::Less, Ordering::Less) => {
                    return Ok(x2..x3);
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Equal, Ordering::Less) => {
                    return Ok(x3..x4);
                }
                t => {
                    return Err(format!("this function is not unimodal: {t:?}"));
                }
            }
        }

        Ok(points[0].0..points[4].0)
    }
}
