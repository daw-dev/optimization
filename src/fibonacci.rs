use std::{cmp::Ordering, ops::Range};

use crate::optimizer::{Function, Optimizer};

pub struct Fibonacci<const N: usize> {
    interval: Range<f64>,
}

impl<const N: usize> Fibonacci<N> {
    pub const GAMMAS: [f64; N] = {
        let mut fibs: [usize; N] = [0; N];
        let mut prev = 1;
        let mut pprev = 1;
        let mut i = 0;
        while i < N {
            fibs[i] = prev + pprev;
            pprev = prev;
            prev = fibs[i];
            i += 1;
        }
        let mut res = [0.0; N];
        res[N - 1] = 0.5;
        let mut i = 0;
        while i < N - 1 {
            res[i] = fibs[N - i - 2] as f64 / fibs[N - i - 1] as f64;
            i += 1;
        }
        res
    };

    pub fn new(interval: Range<f64>) -> Self {
        Self { interval }
    }
}

impl<const N: usize> Optimizer<f64, f64> for Fibonacci<N> {
    fn optimize<F: Function<f64, f64>>(self, func: &F, starting_guess: f64) -> f64 {
        fn find_points<const N: usize>(start: f64, end: f64, iteration: usize) -> [f64; 4] {
            let first = end - Fibonacci::<N>::GAMMAS[iteration] * (end - start);
            let second = start + Fibonacci::<N>::GAMMAS[iteration] * (end - start);
            [start, first, second, end]
        }

        let mut points =
            find_points::<N>(self.interval.start, self.interval.end, 0).map(|x| (x, None));

        for i in 0..N {
            let [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] =
                points.map(|(x, y)| (x, y.unwrap_or_else(|| func.compute(x))));
            match (y1.total_cmp(&y2), y2.total_cmp(&y3), y3.total_cmp(&y4)) {
                (Ordering::Less, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points::<N>(x1, x2, i + 1);
                    points = [(x1, Some(y1)), (x2, None), (x3, None), (x4, Some(y2))]
                }
                (Ordering::Greater, Ordering::Less, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points::<N>(x1, x3, i + 1);
                    points = [(x1, Some(y1)), (x2, None), (x3, Some(y2)), (x4, Some(y3))]
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Less) => {
                    let [x1, x2, x3, x4] = find_points::<N>(x2, x4, i + 1);
                    points = [(x1, Some(y2)), (x2, Some(y3)), (x3, None), (x4, Some(y4))]
                }
                (Ordering::Greater, Ordering::Greater, Ordering::Greater) => {
                    let [x1, x2, x3, x4] = find_points::<N>(x3, x4, i + 1);
                    points = [(x1, Some(y3)), (x2, None), (x3, None), (x4, Some(y4))]
                }
                (Ordering::Greater, Ordering::Equal, Ordering::Less) => {
                    return (x2 + x3) / 2.0;
                }
                t => {
                    unreachable!("this function is not unimodal: {t:?}")
                }
            }
        }

        (points[1].0 + points[2].0) / 2.0
    }
}
