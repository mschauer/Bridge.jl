# Home

## Summary
 
Stochastic calculus and univariate and multivariate stochastic processes/Markov processes in continuous time.

The key objects introduced are the abstract type `ContinuousTimeProcess{T}` parametrised by the state space of the path, for example `T == Float64` and various `structs` suptyping it, for example `Wiener{Float64}` for a real Brownian motion. These play roughly a similar role as types subtyping `Distribution` in the Distributions.jl package.

Secondly, the struct 
```julia
struct SamplePath{T}
    tt::Vector{Float64}
    yy::Vector{T}
    SamplePath{T}(tt, yy) where {T} = new(tt, yy)
end
```
serves as container for sample path returned by direct and approximate samplers (`sample`, `euler`, ...).
`tt` is the vector of the grid points of the simulation and `yy` the corresponding vector of states.

Help is available at the REPL:
```
help?> euler
search: euler euler! eulergamma default_worker_pool schedule @schedule

  euler(u, W, P) -> X

  Solve stochastic differential equation ``dX_t = b(t, X_t)dt + σ(t, X_t)dW_t, X_0 = u``
  using the Euler scheme.
```

Pre-defined processes defined are
`Wiener`, `WienerBridge`, `Gamma`, `LinPro` (linear diffusion/generalized Ornstein-Uhlenbeck) and others.

## Features

- Define and simulate diffusion processes in one or more dimension
- Continuous and discrete likelihood using Girsanovs theorem and transition densities
- Monte Carlo sample diffusion bridges, diffusion processes conditioned to hit a point v at a prescribed time T
- Brownian motion in one and more dimensions
- Ornstein-Uhlenbeck processes
- Bessel processes
- Gamma processes
- Basic stochastic calculus functionality (Ito integral, quadratic variation)
- Euler-Scheme and implicit methods (Runge-Kutta)

The layout/api was originally written to be compatible with Simon Danisch's package [FixedSizeArrays.jl](https://github.com/SimonDanisch/FixedSizeArrays.jl). It was refactored to be compatible with [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) by Dan Getz.

The example programs in the example/directory have additional dependencies: ConjugatePriors and a plotting library.


