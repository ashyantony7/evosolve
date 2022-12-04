use crate::continuous::optimize;
use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};

pub struct Particle<T> {
    pub position: Vec<T>,
    pub velocity: Vec<T>,
    pub best_position: Vec<T>,
    pub best_fitness: T,
}

impl<T> Clone for Particle<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Particle {
            position: self.position.clone(),
            velocity: self.velocity.clone(),
            best_position: self.best_position.clone(),
            best_fitness: self.best_fitness.clone(),
        }
    }
}

pub struct PSO<T> {
    max_iterations: usize,
    lower_bound: Vec<T>,
    upper_bound: Vec<T>,
    pub particles: Vec<Particle<T>>,
    fn_objective: fn(&Vec<T>) -> T,
    equality_constraints: Vec<fn(&Vec<T>) -> T>,
    inequality_constraints: Vec<fn(&Vec<T>) -> T>,
    equality_constraint_weights: Vec<T>,
    inequality_constraint_weights: Vec<T>,
    best_solution: Vec<T>,
    best_value: T,
    pub c1: f64,
    pub c2: f64,
    pub w: f64,
    best_value_history: Vec<T>,
}

impl<T> PSO<T> {
    pub fn new(dimensions: usize, max_iterations: usize, number_particles: usize, objective_function: fn(&Vec<T>) -> T) -> PSO<T>
    where
        T: Copy + Default + Ord + rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>,
    {
        PSO {
            max_iterations,
            lower_bound: Vec::with_capacity(dimensions),
            upper_bound: Vec::with_capacity(dimensions),
            particles: Vec::with_capacity(number_particles),
            fn_objective: objective_function,
            equality_constraints: Vec::new(),
            inequality_constraints: Vec::new(),
            equality_constraint_weights: Vec::new(),
            inequality_constraint_weights: Vec::new(),
            best_solution: Vec::with_capacity(dimensions),
            best_value: T::default(),
            c1: 2.0,
            c2: 2.0,
            w: 0.7,
            best_value_history: Vec::with_capacity(max_iterations),
        }
    }

    fn range_limit(&self, solution: &mut Vec<T>)
    where
        T: PartialOrd + Copy,
    {
        for i in 0..solution.len() {
            if solution[i] < self.lower_bound[i] {
                solution[i] = self.lower_bound[i];
            } else if solution[i] > self.upper_bound[i] {
                solution[i] = self.upper_bound[i];
            }
        }
    }

    fn evaluate(&self, solution: Vec<T>) -> (T, bool)
    where
        T: Default
            + PartialOrd
            + Copy
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Neg
            + std::ops::SubAssign
            + std::ops::AddAssign,
    {
        let mut objective = (self.fn_objective)(&solution);
        let mut constraint_success = true;

        for i in 0..self.equality_constraints.len() {
            let constraint = (self.equality_constraints[i])(&solution) * self.equality_constraint_weights[i];
            if constraint > T::default() {
                constraint_success = false;
                objective += constraint;
            }
        }

        for i in 0..self.inequality_constraints.len() {
            let constraint = (self.inequality_constraints[i])(&solution);
            if constraint < T::default() {
                objective -= constraint * self.inequality_constraint_weights[i];
            }
        }

        (objective, constraint_success)
    }

    fn init(&mut self)
    where
        T: Copy + PartialOrd + rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>,
    {
        let unconstrained: bool = self.lower_bound.is_empty() || self.upper_bound.is_empty();

        for _ in 0..self.particles.capacity() {
            let mut position = Vec::with_capacity(self.lower_bound.len());
            let mut velocity = Vec::with_capacity(self.lower_bound.len());

            for i in 0..self.lower_bound.len() {
                let mut rng = thread_rng();
                let p_i: T = if unconstrained {
                    rng.gen()
                } else {
                    rng.gen_range(self.lower_bound[i]..=self.upper_bound[i])
                };
                let v_i: T = if unconstrained {
                    rng.gen()
                } else {
                    rng.gen_range(self.lower_bound[i]..=self.upper_bound[i])
                };
                position.push(p_i);
                velocity.push(v_i);
            }

            let best_position = position.clone();
            let best_fitness = (self.fn_objective)(&position);

            self.particles.push(Particle {
                position,
                velocity,
                best_position,
                best_fitness,
            });
        }
    }
}

impl<T> optimize::OptimizeContinuous<T> for PSO<T> {
    fn set_bounds(&mut self, lower_bound: Vec<T>, upper_bound: Vec<T>) -> Result<(), String>
    where
        T: PartialOrd,
    {
        if lower_bound.len() != upper_bound.len() {
            return Err(String::from("Lower and upper bounds must have the same dimensions"));
        }
        for i in 0..lower_bound.len() {
            if lower_bound[i] > upper_bound[i] {
                return Err(String::from("Lower bound must be less than upper bound"));
            }
        }

        self.lower_bound = lower_bound;
        self.upper_bound = upper_bound;

        Ok(())
    }

    fn set_equality_constraint(&mut self, constraint: fn(&Vec<T>) -> T) {
        self.equality_constraints.push(constraint);
    }

    fn set_inequality_constraint(&mut self, constraint: fn(&Vec<T>) -> T) {
        self.inequality_constraints.push(constraint);
    }

    fn get_equality_constraint_weights(&self) -> &Vec<T> {
        &self.equality_constraint_weights
    }

    fn set_equality_constraint_weights(&mut self, weights: Vec<T>) -> Result<(), String> {
        if weights.len() != self.equality_constraints.len() {
            return Err(String::from("Number of weights must equal number of constraints"));
        }
        self.equality_constraint_weights = weights;
        Ok(())
    }

    fn get_inequality_constraint_weights(&self) -> &Vec<T> {
        &self.inequality_constraint_weights
    }

    fn set_inequality_constraint_weights(&mut self, weights: Vec<T>) -> Result<(), String> {
        if weights.len() != self.equality_constraints.len() {
            return Err(String::from("Number of weights must equal number of constraints"));
        }
        self.inequality_constraint_weights = weights;
        Ok(())
    }

    fn optimize(&mut self) -> Result<(), String>
    where
        T: Copy + PartialOrd + rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>,
    {
        // TODO
        self.init();

        Ok(())
    }

    fn get_solution(&self) -> &Vec<T> {
        self.best_solution.as_ref()
    }

    fn get_value(&self) -> T
    where
        T: Copy,
    {
        self.best_value
    }
}
