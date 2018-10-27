[![Build Status](https://travis-ci.org/mschauer/Bridge.jl.svg?branch=master)](https://travis-ci.org/mschauer/Bridge.jl)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://mschauer.github.io/Bridge.jl/latest/)
[![Latest](https://img.shields.io/badge/docs-stable-blue.svg)](https://mschauer.github.io/Bridge.jl/stable/)

![Logo](https://mschauer.github.io/Bridge.jl/bridgelogo.gif)

# Bridge.jl

Stochastic calculus and univariate and multivariate stochastic processes/Markov processes in continuous time.
See [./example/tutorial.jl](./example/tutorial.jl) for an introduction. I am personally interested in simulating diffusion bridges and doing Bayesian inference on discretely observed diffusion processes, but this package is written to be of general use and contributions are welcome.

- [x] Define and simulate diffusion processes in one or more dimension
- [x] Continuous and discrete likelihood using Girsanovs theorem and transition densities
- [x] Monte Carlo sample diffusion bridges, diffusion processes conditioned to hit a point v at a prescribed time T
- [x] Brownian motion in one and more dimensions
- [x] Ornstein-Uhlenbeck processes and Ornstein-Uhlenbeck bridges
- [x] Bessel processes
- [x] Gamma processes
- [x] Inhomogenous poisson process
- [x] Basic stochastic calculus functionality (Ito integral, quadratic variation)
- [x] Euler-Scheme and implicit methods (Runge-Kutta)
- [x] Levy-driven SDEs
- [x] Continuous-discrete filtering for partially observed diffusion processes

The layout/api was originally written to be compatible with Simon Danisch's package [FixedSizeArrays.jl](https://github.com/SimonDanisch/FixedSizeArrays.jl). It was refactored to be compatible with [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) by Dan Getz.
Some SDE and ODE solvers in Bridge are accessible with the `JuliaDiffEq` common interface via [BridgeDiffEq.jl](https://github.com/JuliaDiffEq/BridgeDiffEq.jl).

The example programs in the example/ directory have additional dependencies: ConjugatePriors and a plotting library.


## Introduction

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
help?> GammaProcess
search: GammaProcess LocalGammaProcess VarianceGammaProcess
```

| <code>GammaProcess</code>    |
| --- |    
| <p>A <em>GammaProcess</em> with jump rate <code>γ</code> and inverse jump size <code>λ</code> has increments <code>Gamma(t*γ, 1/λ)</code> and Levy measure</p><p>ν(x) = γ x⁻¹ exp(-λ x),</p><p>Here <code>Gamma(α,θ)</code> is the Gamma distribution in julia&#39;s parametrization with shape parameter <code>α</code> and scale <code>θ</code>.</p> <p>&emsp;<b> Examples </b></p><p><code> julia> sample(linspace(0.0, 1.0),  GammaProcess(1.0, 0.5)) </code>

Pre-defined processes defined are
`Wiener`, `WienerBridge`, `Gamma`, `LinPro` (linear diffusion/generalized Ornstein-Uhlenbeck) and others.


It is also quite transparent how to add a new process:

```julia
using Bridge
using Plots

# Define a diffusion process
struct OrnsteinUhlenbeck  <: ContinuousTimeProcess{Float64}
    β::Float64 # drift parameter (also known as inverse relaxation time)
    σ::Float64 # diffusion parameter
end

# define drift and diffusion coefficient of OrnsteinUhlenbeck
Bridge.b(t, x, P::OrnsteinUhlenbeck) = -P.β*x
Bridge.σ(t, x, P::OrnsteinUhlenbeck) = P.σ

# simulate OrnsteinUhlenbeck using Euler scheme
W = sample(0:0.01:10, Wiener())
X = solve(EulerMaruyama(), 0.1, W, OrnsteinUhlenbeck(2.0, 1.0))
plot(X, label="X")
```

![OrnsteinUhlenbeck](https://mschauer.github.io/Bridge.jl/latest/assets/ou.png)

```julia
# Levy (Difference-Gamma process) driven OrnsteinUhlenbeck
Z = sample(0:0.01:10, GammaProcess(100.0,10.0))
Z.yy .-= sample(0:0.01:10, GammaProcess(100.0,10.0)).yy
Y = solve(EulerMaruyama(), 0.1, Z, OrnsteinUhlenbeck(2.0, 1.0))
plot(Y, label="Y")
```

![Levy OrnsteinUhlenbeck](https://mschauer.github.io/Bridge.jl/latest/assets/levyou.png)


## Feedback and Contributing

See the [documentation](https://mschauer.github.io/Bridge.jl/latest/) for more functionality and [issue #12 (Feedback and Contribution)](https://github.com/mschauer/Bridge.jl/issues/12) for coordination of the development.
Bridge is free software under the MIT licence. If you use Bridge.jl in a closed environment I’d be happy to hear about your use case in a mail to moritzschauer@web.de and able to give some support.

## Literature

F. v. d. Meulen, M. Schauer: Bayesian estimation of discretely observed multi-dimensional diffusion processes using guided proposals. *Electronic Journal of Statistics* 11 (1), 2017, [doi:10.1214/17-EJS1290](http://dx.doi.org/10.1214/17-EJS1290).

M. Schauer, F. v. d. Meulen, H. v. Zanten: Guided proposals for simulating multi-dimensional diffusion bridges. *Bernoulli* 23 (4A), 2017, [doi:10.3150/16-BEJ833](http://dx.doi.org/10.3150/16-BEJ833).
