//! Optimization algorithms for problems in continuous domains.
//!
//!
//!
pub mod optimize;
pub use optimize::OptimizeContinuous;

mod algorithms {
    pub mod pso;
}

pub use algorithms::pso;
