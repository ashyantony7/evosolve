use crate::continuous::optimize::OptimizeContinuous;
use crate::utils::misc;
use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};
use std::ops::{Add, Sub};
use std::thread;

struct Particle<T> {
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

/// Particle Swarm Optimization (PSO) algorithm.
///
/// # Example
///
/// ```
/// use evosolve::prelude::*;
///
/// // Given a function f(x) = x^2 + y^2
/// let sphere = |x: &Vec<f64>| -> f64 { x.iter().map(|x| x.powi(2)).sum() };
///
/// // When we run the PSO algorithm
/// let max_iterations = 200;
/// let mut pso = PSO::<f64>::new(2, 200, 100, sphere);
/// pso.set_bounds(vec![-100.0, -100.0], vec![100.0, 100.0]).unwrap();
/// pso.optimize().unwrap();
///
/// // Expect the solution to be close to the origin (0, 0)
/// assert!(pso.get_value() < 1e-5);
/// assert!(pso.get_solution().iter().all(|x| x.abs() < 1e-4));
///```
pub struct PSO<T>
where
    T: Clone,
{
    dimensions: usize,
    max_iterations: usize,
    lower_bound: Vec<T>,
    upper_bound: Vec<T>,
    particles: Vec<Particle<T>>,
    fn_objective: fn(&Vec<T>) -> T,
    best_solution: Vec<T>,
    best_value: T,
    pub c1: f64,
    pub c2: f64,
    pub w: f64,
    best_value_history: Vec<T>,
}

impl<T> PSO<T>
where
    T: Clone + PartialOrd + 'static,
{
    /// Create a new PSO instance.
    ///
    /// # Arguments
    /// * `dimensions` - Number of dimensions of the problem.
    /// * `max_iterations` - Maximum number of iterations.
    /// * `number_particles` - Number of particles.
    /// * `objective_function` - Objective function to minimize.
    pub fn new(dimensions: usize, max_iterations: usize, number_particles: usize, objective_function: fn(&Vec<T>) -> T) -> PSO<T>
    where
        T: Copy + Default + PartialOrd + rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>,
    {
        PSO {
            dimensions,
            max_iterations,
            lower_bound: Vec::with_capacity(dimensions),
            upper_bound: Vec::with_capacity(dimensions),
            particles: Vec::with_capacity(number_particles),
            fn_objective: objective_function,
            best_solution: Vec::with_capacity(dimensions),
            best_value: T::default(),
            c1: 2.0,
            c2: 2.0,
            w: 0.7,
            best_value_history: Vec::with_capacity(max_iterations),
        }
    }

    /// Set the parameters of the PSO algorithm.
    ///
    /// # Arguments
    /// * `c1` - Cognitive parameter.
    /// * `c2` - Social parameter.
    /// * `w` - Inertia parameter.
    pub fn set_parameters(&mut self, c1: f64, c2: f64, w: f64) {
        self.c1 = c1;
        self.c2 = c2;
        self.w = w;
    }

    fn split_iter_among_threads(&self, num_threads: usize) -> Vec<usize> {
        let mut iter_per_thread = vec![self.max_iterations / num_threads; num_threads];
        let remainder = self.max_iterations % num_threads;
        for i in 0..remainder {
            iter_per_thread[i as usize] += 1;
        }
        iter_per_thread
    }

    fn random_init_particles(
        number_particles: usize,
        lower_bound: Vec<T>,
        upper_bound: Vec<T>,
        fn_objective: fn(&Vec<T>) -> T,
    ) -> Vec<Particle<T>>
    where
        T: Copy + PartialOrd + rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>,
    {
        let mut particles = Vec::with_capacity(number_particles);

        let unconstrained: bool = lower_bound.is_empty() || upper_bound.is_empty();

        for _ in 0..particles.capacity() {
            let mut position = Vec::with_capacity(lower_bound.len());
            let mut velocity = Vec::with_capacity(lower_bound.len());

            for i in 0..lower_bound.len() {
                let mut rng = thread_rng();
                let p_i: T = if unconstrained {
                    rng.gen()
                } else {
                    rng.gen_range(lower_bound[i]..=upper_bound[i])
                };
                let v_i: T = if unconstrained {
                    rng.gen()
                } else {
                    rng.gen_range(lower_bound[i]..=upper_bound[i])
                };
                position.push(p_i);
                velocity.push(v_i);
            }

            let best_fitness = (fn_objective)(&position);

            particles.push(Particle {
                position: position.clone(),
                velocity,
                best_position: position,
                best_fitness,
            });
        }
        particles
    }

    fn init_best_solution(&mut self) {
        let mut best_value = self.particles[0].best_fitness.clone();
        let mut best_solution = self.particles[0].best_position.clone();
        for particle in self.particles.iter() {
            if particle.best_fitness < best_value {
                best_value = particle.best_fitness.clone();
                best_solution = particle.best_position.clone();
            }
        }
        self.best_value = best_value;
        self.best_solution = best_solution;
    }

    fn init(&mut self, thread_iters: Vec<usize>)
    where
        T: Copy + PartialOrd + rand::distributions::uniform::SampleUniform + Send,
        Standard: Distribution<T>,
    {
        if thread_iters.len() == 1 {
            self.particles = PSO::<T>::random_init_particles(
                self.particles.capacity(),
                self.lower_bound.clone(),
                self.upper_bound.clone(),
                self.fn_objective,
            );
            self.init_best_solution();
            return;
        }

        let mut handles = Vec::with_capacity(thread_iters.len());
        for i in 0..thread_iters.len() {
            let num_iters = thread_iters[i].clone();
            let lower_bound = self.lower_bound.clone();
            let upper_bound = self.upper_bound.clone();
            let fn_objective = self.fn_objective.clone();
            let handle = thread::spawn(move || PSO::<T>::random_init_particles(num_iters, lower_bound, upper_bound, fn_objective));
            handles.push(handle);
        }

        for handle in handles {
            let result = handle.join().unwrap();
            self.particles.extend(result);
        }
        self.init_best_solution();
    }

    fn main_loop(&mut self)
    where
        T: Add + Sub + Sub<Output = T> + Copy + PartialOrd + From<f64>,
        Standard: Distribution<T>,
        Vec<T>: FromIterator<<T as Add>::Output>,
        f64: From<<T as Sub>::Output>,
    {
        let mut rng = thread_rng();
        for _ in 0..self.max_iterations {
            for particle in self.particles.iter_mut() {
                for i in 0..particle.velocity.len() {
                    let r1: f64 = rng.gen_range(0_f64..=1_f64);
                    let r2: f64 = rng.gen_range(0_f64..=1_f64);
                    let cognitive: f64 = self.c1 * r1 * f64::from(particle.best_position[i] - particle.position[i]);
                    let social: f64 = self.c2 * r2 * f64::from(self.best_solution[i] - particle.position[i]);
                    let new_velocity: f64 = self.w * f64::from(particle.velocity[i]) + cognitive + social;
                    particle.velocity[i] = new_velocity.into();
                }
                particle.position = misc::elementwise_addition(&particle.position, &particle.velocity);

                for i in 0..particle.position.len() {
                    if particle.position[i] < self.lower_bound[i] {
                        particle.position[i] = self.lower_bound[i];
                    } else if particle.position[i] > self.upper_bound[i] {
                        particle.position[i] = self.upper_bound[i];
                    }
                }

                let new_fitness = (self.fn_objective)(&particle.position);
                if new_fitness < particle.best_fitness {
                    particle.best_fitness = new_fitness;
                    particle.best_position = particle.position.clone();
                    if new_fitness < self.best_value {
                        self.best_value = new_fitness;
                        self.best_solution = particle.position.clone();
                    }
                }
            }

            self.best_value_history.push(self.best_value);
        }
    }
}

impl<T> OptimizeContinuous<T> for PSO<T>
where
    T: Clone + 'static,
{
    fn set_bounds(&mut self, lower_bound: Vec<T>, upper_bound: Vec<T>) -> Result<(), String>
    where
        T: PartialOrd,
    {
        if lower_bound.len() != upper_bound.len() {
            return Err(String::from("Lower and upper bounds must have the same dimensions"));
        }

        if lower_bound.len() != self.dimensions {
            return Err(String::from("Bound dimensions must match the number of dimensions of the problem"));
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

    fn optimize(&mut self) -> Result<(), String>
    where
        T: Add + Sub + Sub<Output = T> + Copy + PartialOrd + From<f64> + rand::distributions::uniform::SampleUniform + Send,
        Standard: Distribution<T>,
        Vec<T>: FromIterator<<T as Add>::Output>,
        f64: From<<T as Sub>::Output>,
    {
        let thread_iter = vec![self.max_iterations];
        self.init(thread_iter);

        self.main_loop();

        Ok(())
    }

    fn optimize_parallel(&mut self, num_threads: usize) -> Result<(), String>
    where
        T: Add + Sub + Sub<Output = T> + Copy + PartialOrd + From<f64> + rand::distributions::uniform::SampleUniform + Send + Sync,
        Standard: Distribution<T>,
        Vec<T>: FromIterator<<T as Add>::Output>,
        f64: From<<T as Sub>::Output>,
    {
        let thread_iters = self.split_iter_among_threads(num_threads);

        self.init(thread_iters);

        self.main_loop();

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

    fn get_value_history(&self) -> &Vec<T> {
        self.best_value_history.as_ref()
    }
}
