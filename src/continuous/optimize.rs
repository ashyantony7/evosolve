//! Optimization algorithms for problems in continuous domains.
//!
//! The trait handles the common methods for all optimization algorithms in continuous domains.
//!
use rand::{distributions::Standard, prelude::Distribution};
use std::ops::{Add, Sub};

pub trait OptimizeContinuous<T> {
    /// Set the lower and upper bounds for the search space.
    ///
    /// # Arguments
    /// * lower_bound - The lower bound for each dimension of the search space.
    /// * upper_bound - The upper bound for each dimension of the search space.
    fn set_bounds(&mut self, lower_bound: Vec<T>, upper_bound: Vec<T>) -> Result<(), String>
    where
        T: PartialOrd;

    /// Run the optimization algorithm.
    fn optimize(&mut self) -> Result<(), String>
    where
        T: Add + Sub + Sub<Output = T> + Copy + PartialOrd + std::convert::From<f64> + rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>,
        Vec<T>: FromIterator<<T as std::ops::Add>::Output>,
        f64: From<<T as Sub>::Output>;

    /// Get the best solution found by the optimization algorithm.
    ///
    /// # Returns
    /// * The best solution found by the optimization algorithm.
    fn get_solution(&self) -> &Vec<T>
    where
        T: Clone;

    /// Get the value of the best solution found by the optimization algorithm.
    ///
    /// # Returns
    /// * The value of the best solution found by the optimization algorithm.
    fn get_value(&self) -> T
    where
        T: Copy;

    /// Get the history of the best value found by the optimization algorithm.
    ///
    /// # Returns
    /// * The history of the best value found by the optimization algorithm.
    fn get_value_history(&self) -> &Vec<T>;
}
