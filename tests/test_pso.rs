use evosolve::prelude::*;

#[test]
fn pso_bounds() {
    // Given a function f(x) = x^2 + y^2
    let mut pso = PSO::<f64>::new(2, 200, 100, |x| x.iter().map(|x| x.powi(2)).sum());

    // Expect error when upper bound is smaller than lower bound
    assert!(pso.set_bounds(vec![-100.0, -100.0], vec![100.0, 100.0]).is_ok());

    // Expect error when lower bound and upper bound have different dimensions
    assert!(pso.set_bounds(vec![-100.0, -100.0], vec![100.0]).is_err());

    // Expect error when length of lower bound and upper bound do not match the number of dimensions
    assert!(pso.set_bounds(vec![-100.0], vec![100.0]).is_err());
}

#[test]
fn pso_f64() {
    // Given a function f(x) = x^2 + y^2
    let sphere = |x: &Vec<f64>| -> f64 { x.iter().map(|x| x.powi(2)).sum() };

    // When we run the PSO algorithm
    let max_iterations = 200;
    let mut pso = PSO::<f64>::new(2, max_iterations, 100, sphere);
    pso.set_bounds(vec![-100.0, -100.0], vec![100.0, 100.0]).unwrap();
    pso.optimize().unwrap();

    // Expect the solution to be close to the origin (0, 0)
    assert!(pso.get_value() < 1e-5);
    assert!(pso.get_solution().iter().all(|x| x.abs() < 1e-4));

    // Expect length of value history to be equal to the number of iterations
    assert_eq!(pso.get_value_history().len(), max_iterations);
}

#[test]
fn pso_parallel() {
    // Given a function f(x) = x^2 + y^2
    let sphere = |x: &Vec<f64>| -> f64 { x.iter().map(|x| x.powi(2)).sum() };

    // When we run the PSO algorithm in parallel
    let max_iterations = 200;
    let mut pso = PSO::<f64>::new(2, max_iterations, 100, sphere);
    pso.set_bounds(vec![-100.0, -100.0], vec![100.0, 100.0]).unwrap();
    pso.optimize_parallel(8).unwrap();

    // Expect the solution to be close to the origin (0, 0)
    assert!(pso.get_value() < 1e-5);
    assert!(pso.get_solution().iter().all(|x| x.abs() < 1e-4));

    // Expect length of value history to be equal to the number of iterations
    assert_eq!(pso.get_value_history().len(), max_iterations);
}
