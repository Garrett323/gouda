mod linear_regression;
mod model;
pub use linear_regression::SolverType;
pub use model::Mice;

trait Solver {
    fn solve();
}
