use crate::functions::Function;
use crate::optimizer::Optimizer;
use std::{cmp::Ordering, f64, ops::Range};

#[derive(Clone)]
pub enum StoppingCondition {
    Precision(f64),
    Iterations(usize),
}

#[derive(Clone)]
pub struct GoldenRatio {
    stopping_condition: StoppingCondition,
}

impl GoldenRatio {
    pub const GAMMA: f64 = f64::consts::GOLDEN_RATIO - 1f64;

    pub fn iterations_from_precision(precision: f64, starting_interval: &Range<f64>) -> usize {
        (precision / (starting_interval.end - starting_interval.start))
            .log(GoldenRatio::GAMMA)
            .ceil() as usize
    }

    pub fn new(stopping_condition: StoppingCondition) -> Self {
        Self { stopping_condition }
    }
}

impl Optimizer<f64, f64, Range<f64>, Result<Range<f64>, String>> for GoldenRatio {
    fn optimize<F: Function<f64, f64>>(
        self,
        func: &F,
        starting_guess: Range<f64>,
    ) -> Result<Range<f64>, String> {
        fn find_points(start: f64, end: f64) -> [f64; 4] {
            let first = end - GoldenRatio::GAMMA * (end - start);
            let second = start + GoldenRatio::GAMMA * (end - start);
            [start, first, second, end]
        }

        let iterations = match self.stopping_condition {
            StoppingCondition::Precision(prec) => {
                Self::iterations_from_precision(prec, &starting_guess)
            }
            StoppingCondition::Iterations(iter) => iter,
        };

        let mut points = find_points(starting_guess.start, starting_guess.end).map(|x| (x, None));

        for _ in 0..iterations {
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
        }

        Ok(points[1].0..points[2].0)
    }
}
