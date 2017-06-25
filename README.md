# Bridge.jl
 
Stochastic calculus and univariate and multivariate stochastic processes/Markov processes in continuous time.
See [./example/tutorial.jl](./example/tutorial.jl) for an introduction.
This package is going to replace my package https://github.com/mschauer/SDE.jl . I am personally interested in simulating diffusion bridges and doing bayesian inference on discretely observed diffusion processes, but this package is written to be of general use and contributions are welcome. It is also quite transparent how to add a new process:

```Julia
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
W = sample(0:0.01:10, Wiener{Float64}()) 
X = euler(0.1, W, OrnsteinUhlenbeck(20., 1.))
```

- [x] Define and simulate diffusion processes in one or more dimension
- [x] Continuous and discrete likelihood using Girsanovs theorem and transition densities
- [x] Monte Carlo sample diffusion bridges, diffusion processes conditioned to hit a point v at a prescribed time T
- [x] Brownian motion in one and more dimensions
- [x] Ornstein-Uhlenbeck processes
- [ ] Geometric Brownian motion 
- [ ] Fractional Brownian motion
- [x] Basic stochastic calculus functionality (Ito integral, quadratic variation) 

The layout/api was originally written to be compatible with SimonDanisch's package [FixedSizeArrays.jl](https://github.com/SimonDanisch/FixedSizeArrays.jl). It was refactored to be compatible with [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).





