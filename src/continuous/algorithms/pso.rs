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
    fn_fitness: fn(&Vec<T>) -> T,
    best_position: Vec<T>,
    best_fitness: T,
    pub c1: f64,
    pub c2: f64,
    pub w: f64,
    best_fitness_history: Vec<T>,
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
            fn_fitness: objective_function,
            best_position: Vec::with_capacity(dimensions),
            best_fitness: T::default(),
            c1: 2.0,
            c2: 2.0,
            w: 0.7,
            best_fitness_history: Vec::with_capacity(max_iterations),
        }
    }

    pub fn set_bounds(&mut self, lower_bound: Vec<T>, upper_bound: Vec<T>)
    where
        T: PartialOrd,
    {
        if lower_bound.len() != upper_bound.len() {
            panic!("Lower and upper bounds must have the same dimensions");
        }
        for i in 0..lower_bound.len() {
            if lower_bound[i] > upper_bound[i] {
                panic!("Lower bound must be less than upper bound");
            }
        }

        self.lower_bound = lower_bound;
        self.upper_bound = upper_bound;
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
            let best_fitness = (self.fn_fitness)(&position);

            self.particles.push(Particle {
                position,
                velocity,
                best_position: best_position,
                best_fitness: best_fitness,
            });
        }
    }

    pub fn optimize(&mut self)
    where
        T: Copy + PartialOrd + rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>,
    {
        // TODO
        self.init();
    }
}
