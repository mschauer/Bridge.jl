
"""
    InhomogPoisson(λ)

Inhomogenous Poisson process with intensity function `λ(t)`.
See also `ThinningAlg`.
"""
struct InhomogPoisson{T<:Function}
    λ::T
end

abstract type PoissonSampler
end

"""
    ThinningAlg(λmax)

Sampling method for `InhomogPoisson` by the 'thinning' algorithm. 

#### Examples:

```
sample(ThinningAlg(λmax), T, InhomogPoisson(λ))
```
"""
struct ThinningAlg <: PoissonSampler
    λmax::Float64
end

"""
    sample(::Thinning, T, P::InhomogPoisson) -> tt
"""
function sample(method::ThinningAlg, T, P::InhomogPoisson)
    λmax = method.λmax
    t = 0.0
    tt = zeros(0)
    while t <= T
        t = t - log(rand())/λmax
        if rand() ≤ P.λ(t)/λmax
            push!(tt, t)
        end
    end
    tt
end