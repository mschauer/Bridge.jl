# Manual

## Define and simulate a stochastic process

In this section, an Ornstein-Uhlenbeck process is defined by the
stochastic differential equation

```math
    \mathrm{d} X_t = -β\, \mathrm{d}t + σ\, \mathrm{d} W_t\qquad(1)
```

and a sample path is generated in three steps.
`β::Float64` is the mean reversion parameter 
and `σ::Float64` is the diffusion parameter.

### Step 1. Define a diffusion process `OrnsteinUhlenbeck`.

The new struct `OrnsteinUhlenbeck` is a subtype `ContinuousTimeProcess{Float64}` indicating that the Ornstein-Uhlenbeck process has
`Float64`-valued trajectories.

```jldoctest OrnsteinUhlenbeck
using Bridge
struct OrnsteinUhlenbeck  <: ContinuousTimeProcess{Float64}
    β::Float64
    σ::Float64 
    function OrnsteinUhlenbeck(β::Float64, σ::Float64)
        isnan(β) || β > 0. || error("Parameter β must be positive.")
        isnan(σ) || σ > 0. || error("Parameter σ must be positive.")
        new(β, σ)
    end
end

# output

```

### Step 2. Define drift and diffusion coefficient.

`b` is the dependend drift, `σ` the dispersion coefficient and `a` the
diffusion coefficient. These functions expect a time `t`, a location `x`
and are dispatch on the type of the process `P`. In this case their values are constants provided by the `P` argument.

```jldoctest OrnsteinUhlenbeck
Bridge.b(t, x, P::OrnsteinUhlenbeck) = -P.β * x
Bridge.σ(t, x, P::OrnsteinUhlenbeck) = P.σ
Bridge.a(t, x, P::OrnsteinUhlenbeck) = P.σ^2

# output

```

### Step 3. Simulate `OrnsteinUhlenbeck` process using the Euler scheme.

Generate the driving Brownian motion `W` of the stochastic differential equation (1) with `sample`. Thefirst argument is the time grid, the second arguments specifies a `Float64`-valued Brownian motion/Wiener process.

```jldoctest OrnsteinUhlenbeck
using Random
Random.seed!(1)
W = sample(0:0.1:1, Wiener())

# output

SamplePath{Float64}([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.0940107, 0.214935, 0.0259463, 0.0226432, -0.24268, -0.144298, 0.581472, -0.135443, 0.0321464, 0.168574])
```

The output is a `SamplePath` object assigned to `W`. It contains time grid `W.tt` and the sampled values `W.yy`.

Generate a solution `X` using the `Euler()`-scheme, using time grid `W.tt`. The arguments are
starting point `0.1`, driving Brownian motion `W` and the `OrnsteinUhlenbeck` object with parameters `β = 20.0` and
`σ = 1.0`.

```jldoctest OrnsteinUhlenbeck
X = Bridge.solve(Euler(), 0.1, W, OrnsteinUhlenbeck(20.0, 1.0));

# output

SamplePath{Float64}([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.1, -0.00598928, 0.126914, -0.315902, 0.312599, -0.577923, 0.676305, 0.0494658, -0.766381, 0.933971, -0.797544])
```

This returns a `SamplePath` of the solution.

```@meta
DocTestSetup = quote
    using Bridge
end
```

## Tutorials and Notebooks

A detailed tutorial script:
[./example/tutorial.jl](https://www.github.com/mschauer/Bridge.jl/blob/master/example/tutorial.jl)

A nice notebook detailing the generation of the logo using ordinary and stochastic differential equations (and, in fact, *diffusion bridges* (sic) to create a seamless loop):
[./example/Bridge+Logo.ipynb](https://github.com/mschauer/Bridge.jl/blob/master/example/Bridge%2BLogo.ipynb)