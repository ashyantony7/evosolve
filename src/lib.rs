//! Evolutionary algorithms for continuous and combinatorial optimization.
//!
//! evolove is a Rust library for evolutionary algorithms to solve optimization problems.
//!
//! ## Quickstart
//!
//! ```rust
//! use evosolve::prelude::*;
//!
//! // Sphere function f(x) = x^2 + y^2
//! let sphere = |x: &Vec<f64>| -> f64 { x.iter().map(|x| x.powi(2)).sum() };
//!
//! // Initialize PSO algorithm
//! let max_iterations = 200;
//! let number_particles = 100;
//! let mut pso = PSO::<f64>::new(2, max_iterations, number_particles, sphere);
//!
//! // Set bounds
//! pso.set_bounds(vec![-100.0, -100.0], vec![100.0, 100.0]).unwrap();
//!
//! // Run optimization
//! pso.optimize().unwrap();
//!
//! println!("Solution: {:?}", pso.get_solution());
//! println!("Value: {}", pso.get_value());
//! ```
//!

pub mod continuous;
pub mod prelude;
pub mod utils;
