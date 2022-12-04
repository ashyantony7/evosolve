use rand::{distributions::Standard, prelude::Distribution};

pub trait OptimizeContinuous<T> {
    fn set_bounds(&mut self, lower_bound: Vec<T>, upper_bound: Vec<T>) -> Result<(), String>
    where
        T: PartialOrd;

    fn set_equality_constraint(&mut self, constraint: fn(&Vec<T>) -> T)
    where
        T: PartialOrd;

    fn set_inequality_constraint(&mut self, constraint: fn(&Vec<T>) -> T)
    where
        T: PartialOrd;

    fn get_equality_constraint_weights(&self) -> &Vec<T>;

    fn set_equality_constraint_weights(&mut self, weights: Vec<T>) -> Result<(), String>
    where
        T: PartialOrd;

    fn get_inequality_constraint_weights(&self) -> &Vec<T>;

    fn set_inequality_constraint_weights(&mut self, weights: Vec<T>) -> Result<(), String>
    where
        T: PartialOrd;

    fn optimize(&mut self) -> Result<(), String>
    where
        T: Copy + PartialOrd + rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>;

    fn get_solution(&self) -> &Vec<T>
    where
        T: Clone;

    fn get_value(&self) -> T
    where
        T: Copy;
}
