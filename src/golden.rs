use std::{cmp::Ordering, f64, ops::Range};

use crate::optimizer::{Function, Optimizer};

pub struct GoldenRatio {
    iterations: usize,
    interval: Range<f64>,
}

impl GoldenRatio {
    pub const GAMMA: f64 = 1f64 - f64::consts::GOLDEN_RATIO;

    pub fn with_iterations(iterations: usize, starting_interval: Range<f64>) -> Self {
        Self {
            iterations,
            interval: starting_interval,
        }
    }

    pub fn iterations_from_precision(precision: f64, starting_interval: Range<f64>) -> usize {
        (precision / (starting_interval.end - starting_interval.start))
            .log(GoldenRatio::GAMMA)
            .ceil() as usize
    }

    pub fn with_precision(precision: f64, starting_interval: Range<f64>) -> Self {
        Self {
            iterations: GoldenRatio::iterations_from_precision(
                precision,
                starting_interval.clone(),
            ),
            interval: starting_interval,
        }
    }
}

impl Optimizer<1> for GoldenRatio {
    fn optimize<F: Function<1>>(self, func: &F, starting_guess: [f64; 1]) -> [f64; 1] {
        fn find_points(start: f64, end: f64) -> [f64; 4] {
            let first = start + GoldenRatio::GAMMA * (end - start);
            let second = end - GoldenRatio::GAMMA * (end - start);
            [start, first, second, end]
        }

        let mut points = find_points(self.interval.start, self.interval.end).map(|x| (x, None));

        for _ in 0..self.iterations {
            let [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] =
                points.map(|(x, y)| (x, y.unwrap_or_else(|| func.compute([x]))));
            match (y1.total_cmp(&y2), y2.total_cmp(&y3), y3.total_cmp(&y4)) {
                (Ordering::Less, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x1, x2);
                    points = [(x1, Some(y1)), (x2, None), (x3, None), (x4, Some(y2))]
                }
                (Ordering::Greater, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x1, x3);
                    points = [(x1, Some(y1)), (x2, None), (x3, Some(y2)), (x4, Some(y3))]
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points(x2, x4);
                    points = [(x1, Some(y2)), (x2, Some(y3)), (x3, None), (x4, Some(y4))]
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Greater) => {
                    let [x1, x2, x3, x4] = find_points(x3, x4);
                    points = [(x1, Some(y3)), (x2, None), (x3, None), (x4, Some(y4))]
                }
                _ => {
                    unreachable!("this function is not unimodal")
                }
            }
        }

        [points[2].0]
    }
}

