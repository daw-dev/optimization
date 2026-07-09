# Copilot Instructions

## Build, test, and examples

- `cargo check` — fast compile pass for the library.
- `cargo test` — runs the crate test/build pass.
- `cargo test <filter>` — run a single test by name once tests exist.
- `cargo fmt` — format the workspace.
- `cargo clippy` — run the Rust linter.
- `cargo build --examples` — builds the `labs/` examples declared in `Cargo.toml`.
- `cargo run --example lab01` / `lab02` / `lab03` / `lab04` — run an individual lab example.

## Architecture

- The crate is split into a generic optimization core plus two domains: `linear` for 1D/gradient methods and `quadratic` for matrix-based multivariate methods.
- `src/optimizer.rs` defines the `Optimizer` trait. Optimizers return iterators of intermediate guesses, and `chain()` composes one optimizer’s output into another.
- `src/functions.rs` defines the `Function`, `Derivative`, `Gradient`, and `Hessian` traits. Closures and boxed functions work directly through blanket impls, and derivatives are approximated numerically.
- `src/helpers.rs` contains small wrapper types for stopping criteria (`Iterations`, `Precision`) plus `UniformSample` for evenly spaced plotting samples.
- `src/linear/` contains interval search methods (`dicothomic`, `golden`, `fibonacci`), Newton-style 1D search, and gradient descent variants.
- `src/quadratic/` defines the fixed-size `Matrix` type and aliases like `SquareMatrix`, `Column`, and `Row`, plus Newton-Raphson and conjugate-style solvers.
- The `labs/` examples are demonstration programs; several generate Plotly HTML output under `labs/lab*/`.

## Conventions

- Prefer the existing iterator-based optimizer pattern: `optimize(...)` should yield every intermediate guess rather than only the final answer.
- Keep stopping logic in the `Iterations` / `Precision` wrapper types instead of adding ad hoc booleans or flags.
- Reuse the `Function` trait instead of hard-coding closures; most algorithms accept any `Function` implementation.
- For multivariate code, use the `Column` / `Row` / `Value` wrappers and the overloaded matrix operators (`^`, `+`, `-`, `*`) already used in `quadratic/`.
- Line searches and multi-stage methods are commonly built with `Optimizer::chain(...)`; keep new composition logic consistent with that pattern.
- `labs/lab02` currently references `helpers::Average`, which is commented out in `src/helpers.rs`, so that example is incomplete as-is.
