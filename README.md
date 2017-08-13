[![Build Status](https://travis-ci.org/mschauer/Bridge.jl.svg?branch=master)](https://travis-ci.org/mschauer/Bridge.jl)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://mschauer.github.io/Bridge.jl/latest/)

# Bridge.jl
 
Stochastic calculus and univariate and multivariate stochastic processes/Markov processes in continuous time.
See [./example/tutorial.jl](./example/tutorial.jl) for an introduction. I am personally interested in simulating diffusion bridges and doing Bayesian inference on discretely observed diffusion processes, but this package is written to be of general use and contributions are welcome. 

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


It is also quite transparent how to add a new process:

```julia
# Define a diffusion process
struct OrnsteinUhlenbeck  <: ContinuousTimeProcess{Float64}
    β::Float64 # drift parameter (also known as inverse relaxation time)
    σ::Float64 # diffusion parameter
    function OrnsteinUhlenbeck(β::Float64, σ::Float64)
        isnan(β) || β > 0. || error("Parameter λ must be positive.")
        isnan(σ) || σ > 0. || error("Parameter σ must be positive.")
        new(β, σ)
    end
end

# define drift and diffusion coefficient of OrnsteinUhlenbeck
import Bridge: b, σ, a, transitionprob
Bridge.b(t,x, P::OrnsteinUhlenbeck) = -P.β*x
Bridge.σ(t, x, P::OrnsteinUhlenbeck) = P.σ
Bridge.a(t, x, P::OrnsteinUhlenbeck) = P.σ^2

# simulate OrnsteinUhlenbeck using Euler scheme
W = sample(0:0.01:10, Wiener()) 
X = euler(0.1, W, OrnsteinUhlenbeck(20.0, 1.0))
```

- [x] Define and simulate diffusion processes in one or more dimension
- [x] Continuous and discrete likelihood using Girsanovs theorem and transition densities
- [x] Monte Carlo sample diffusion bridges, diffusion processes conditioned to hit a point v at a prescribed time T
- [x] Brownian motion in one and more dimensions
- [x] Ornstein-Uhlenbeck processes
- [ ] Geometric Brownian motion 
- [x] Bessel processes
- [x] Gamma processes
- [x] Basic stochastic calculus functionality (Ito integral, quadratic variation)
- [x] Euler-Scheme and implicit methods (Rungekutta)

The layout/api was originally written to be compatible with Simon Danisch's package [FixedSizeArrays.jl](https://github.com/SimonDanisch/FixedSizeArrays.jl). It was refactored to be compatible with [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) by Dan Getz.

The example programs in the example/ directory have additional dependencies: ConjugatePriors and a plotting library.


