use std::{cmp::Ordering, ops::Range};

use crate::optimizer::{Function, Optimizer};

pub struct Dicothomic {
    iterations: usize,
    interval: Range<f64>,
}

impl Dicothomic {
    pub fn with_iterations(iterations: usize, starting_interval: Range<f64>) -> Self {
        Self {
            iterations,
            interval: starting_interval,
        }
    }

    pub fn iterations_from_precision(precision: f64, starting_interval: Range<f64>) -> usize {
        ((starting_interval.end - starting_interval.start) / precision)
            .log2()
            .ceil() as usize
    }

    pub fn with_precision(precision: f64, starting_interval: Range<f64>) -> Self {
        Self {
            iterations: Dicothomic::iterations_from_precision(precision, starting_interval.clone()),
            interval: starting_interval,
        }
    }
}

impl Optimizer<f64, f64> for Dicothomic {
    fn optimize<F: Function<f64, f64>>(self, func: &F, starting_guess: f64) -> f64 {
        fn find_points(start: f64, end: f64) -> [f64; 5] {
            let mid = (start + end) / 2.0;
            let left_quarter = (start + mid) / 2.0;
            let right_quarter = (mid + end) / 2.0;
            [start, left_quarter, mid, right_quarter, end]
        }

        let mut points = find_points(self.interval.start, self.interval.end).map(|x| (x, None));

        for _ in 0..self.iterations {
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
                    return (x2 + x3) / 2.0;
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Equal, Ordering::Less) => {
                    return (x3 + x4) / 2.0;
                }
                t => {
                    unreachable!("this function is not unimodal: {t:?}")
                }
            }
        }

        points[2].0
    }
}
