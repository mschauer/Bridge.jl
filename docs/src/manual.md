# Manual

## Defining and simulating a new process

In this section, a Ornstein-Uhlenbeck process

```math
    \mathrm{d} X_t = -β \mathrm{d}t + \mathrm{d} W_t
```

is defined and a sample path generated in three steps.

### Step 1. Define a diffusion process `OrnsteinUhlenbeck`.

```jldoctest OrnsteinUhlenbeck
using Bridge
struct OrnsteinUhlenbeck  <: ContinuousTimeProcess{Float64}
    β::Float64 # drift parameter (also known as inverse relaxation time)
    σ::Float64 # diffusion parameter
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
and are dispatch on the type of the process `P`.

```jldoctest OrnsteinUhlenbeck
Bridge.b(t, x, P::OrnsteinUhlenbeck) = -P.β*x
Bridge.σ(t, x, P::OrnsteinUhlenbeck) = P.σ
Bridge.a(t, x, P::OrnsteinUhlenbeck) = P.σ^2

# output

```

### Step 3. Simulate `OrnsteinUhlenbeck` process using the Euler scheme.

Generate a driving Brownian motion `W`.

```jldoctest OrnsteinUhlenbeck
srand(1)
W = sample(0:0.1:1, Wiener())

# output

Bridge.SamplePath{Float64}([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.0940107, 0.214935, 0.0259463, 0.0226432, -0.24268, -0.144298, 0.581472, -0.135443, 0.0321464, 0.168574])
```

Generate a solution `X` using the `Euler()`-scheme.

```jldoctest OrnsteinUhlenbeck
X = Bridge.solve(Euler(), 0.1, W, OrnsteinUhlenbeck(20.0, 1.0));

# output

Bridge.SamplePath{Float64}([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.1, -0.00598928, 0.126914, -0.315902, 0.312599, -0.577923, 0.676305, 0.0494658, -0.766381, 0.933971, -0.797544])
```

```@meta
DocTestSetup = quote
    using Bridge
end
```

## Tutorials and Notebooks

A detailed tutorial script:
[./example/tutorial.jl](https://www.github.com/mschauer/Bridge.jl/example/tutorial.jl)

A nice notebook detailing the generation of the logo: 
[./example/Bridge+Logo.ipynb](https://github.com/mschauer/Bridge.jl/blob/master/example/Bridge%2BLogo.ipynb)