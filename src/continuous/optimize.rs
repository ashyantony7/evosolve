//! Optimization algorithms for problems in continuous domains.
//!
//! The trait handles the common methods for all optimization algorithms in continuous domains.
//!
use rand::{distributions::Standard, prelude::Distribution};
use std::ops::{Add, Sub};

pub trait OptimizeContinuous<T> {
    fn set_bounds(&mut self, lower_bound: Vec<T>, upper_bound: Vec<T>) -> Result<(), String>
    where
        T: PartialOrd;

    fn optimize(&mut self) -> Result<(), String>
    where
        T: Add + Sub + Sub<Output = T> + Copy + PartialOrd + std::convert::From<f64> + rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>,
        Vec<T>: FromIterator<<T as std::ops::Add>::Output>,
        f64: From<<T as Sub>::Output>;

    fn get_solution(&self) -> &Vec<T>
    where
        T: Clone;

    fn get_value(&self) -> T
    where
        T: Copy;

    fn get_value_history(&self) -> &Vec<T>;
}
