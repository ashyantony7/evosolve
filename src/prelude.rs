//! Convenience re-export of common members
//!
//! Like the standard library's prelude, this module simplifies importing of
//! common items. Unlike the standard prelude, the contents of this module must
//! be imported manually:
//!
//! ```
//! use evosolve::prelude::*;
//! ```
//!
pub use crate::continuous::optimize::OptimizeContinuous;
pub use crate::continuous::pso::PSO;
pub use crate::utils::misc;
