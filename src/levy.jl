abstract type LevyProcess{T} <: ContinuousTimeProcess{T} end

"""
    CompoundPoisson{T} <: LevyProcess{T}

Abstract type. For a compound Poisson process define `rjumpsize(P) -> T` and
 `arrival(P) -> Distribution`.
"""
abstract type CompoundPoisson{T} <: LevyProcess{T} end


"""
    ExpCounting(λ)

Counting process with arrival times `arrival(P) = Exponential(1/λ)` and unit jumps.
"""
struct ExpCounting <: CompoundPoisson{Int}
    λ::Float64
end

arrival(P::ExpCounting) = Exponential(inv(P.λ))

function rjumpsize(::ExpCounting)
    1
end

function sample(T, P::CompoundPoisson)
    dt = rand(arrival(P))
    t = 0.0
    y = 0.0
    tt = [t]
    yy = [y]

    while t + dt <= T
        t = t + dt
        y = y + rjumpsize(P)
        append!(tt, t)
        append!(yy, y)

        dt = rand(arrival(P))
    end
    SamplePath(tt, yy)
end


"""
    GammaProcess

A *GammaProcess* with jump rate `γ` and inverse jump size `λ` has increments `Gamma(t*γ, 1/λ)` and Levy measure

```math
ν(x)=γ x^{-1}\\exp(-λ x), 
```

Here `Gamma(α,θ)` is the Gamma distribution in julia's parametrization with shape parameter `α` and scale `θ`.
"""
struct GammaProcess <: LevyProcess{Float64}
    γ::Float64
    λ::Float64
end


struct VarianceGammaProcess <: LevyProcess{Float64}
    θ::Float64
    σ::Float64
    ν::Float64
end

struct VarianceGamma 
    θ::Float64
    σ::Float64
    t::Float64
    ν::Float64
end

"""
    GammaBridge(t, v, P)

A `GammaProcess` `P` conditional on htting `v` at time `t`.
"""
struct GammaBridge  <: ContinuousTimeProcess{Float64}
    t::Float64
    v::Float64
    P::GammaProcess
end



sample(tt::AbstractVector{Float64}, P::LevyProcess{T}, x1=zero(T)) where {T} =  
    sample!(SamplePath{T}(collect(tt), zeros(T, length(tt))), P, x1)
 

function sample!(X, P::LevyProcess{T}, x1=zero(T)) where {T}
    tt = X.tt
    yy = X.yy

    yy[1] = x = x1
    for i in 2:length(tt)
        x = x + rand(increment(tt[i]-tt[i-1], P))
        yy[i] = x
    end
    X
end


increment(t, P::GammaProcess) = Gamma(t*P.γ, 1/P.λ)

lp(s, x, t, y, P::GammaProcess) = logpdf(increment(t-s, P), y-x)

increment(t, P::VarianceGammaProcess) = VarianceGamma(P.θ, P.σ, t, P.ν)

function rand(P::VarianceGamma) 
    Z = randn()
    G = rand(Gamma(P.t/P.ν, P.ν))
    P.θ*G + P.σ*sqrt(G)*Z
end


function sample(tt::AbstractVector{Float64}, P::GammaBridge, x1::Float64 = 0.0)
    tt = collect(tt) 
    t = P.t
    r = searchsorted(tt, t)
    if isempty(r) # add t between n = last(r) and first(r)=n+1
       tt = Float64[tt[1:last(r)]; t; tt[first(r):end]] # t now at first(r)
    end
    X = sample(tt, P.P, zero(x1))    
    dx = P.v - x1
    yy = X.yy
    yy[:] =  yy .* (dx/yy[first(r)]) .+ x1
    if isempty(r) # remove (t,x1)
        tt = [tt[1:last(r)]; tt[first(r)+1:end]]                
        yy = [yy[1:last(r)]; yy[first(r)+1:end]]   
    end
    SamplePath{Float64}(tt, yy)
end

function sample!(X, P::GammaBridge, x1::Float64 = 0.0)
    tt = X.tt 
    t = P.t
    r = searchsorted(tt, t)
    if isempty(r) 
        throw(BoundsError("$tt $t"))
    end
    sample!(X, P.P, zero(x1))    
    dx = P.v - x1
    yy = X.yy
    yy[:] =  yy .* (dx/yy[first(r)]) .+ x1
    X
end


"""
    LocalGammaProcess
"""
struct LocalGammaProcess
    P::GammaProcess
    θ::Vector{Float64} # parameter vector
    b::Vector{Float64} # grid points b1, ..., bn where n = length(θ)
    LocalGammaProcess(P, θ, b) = length(θ) != length(b) ? 
        throw(ArgumentError("θ and b differ in length")) : new(P, θ, b)

end

"""
    θ(x, P::LocalGammaProcess)

Inverse jump size compared to gamma process with same alpha and beta.
"""
function θ(x, P::LocalGammaProcess)
    N = length(P.θ)
    N == 0 && return 0
    x <= P.b[1] && return 0.
    x > P.b[N] && return P.θ[N] * x
    
    #k = Int(div(x, P.b1))
    k = first(searchsorted(P.b, x)) - 1
    P.θ[k] * x
end

"""
     nu(k,P)

(Bin-wise) integral of the Levy measure ``\\nu(B_k)``.
"""
function nu(k,P)
    if k == 0
        P.P.γ*(-log(P.P.λ) - expint1((P.P.λ)*P.b[1])) # up to constant
    elseif k == length(P.θ) 
        assert((P.P.λ + P.θ[k]) > 0.0)
        P.P.γ*(expint1((P.P.λ + P.θ[k])*P.b[k])) # - 0
    else
        P.P.γ*(expint1((P.P.λ + P.θ[k])*P.b[k]) - expint1((P.P.λ + P.θ[k])*P.b[k+1]))
    end
end

"""
    compensator(kstart, P::LocalGammaProcess)

Compensator of LocalGammaProcess 
For `kstart = 1`, this is ``\\sum_{k=1}^N \\nu(B_k)``,
for `kstart = 0`, this is ``\\sum_{k=0}^N \\nu(B_k) - C`` (where ``C`` is a constant).
"""
function compensator(kstart, P::LocalGammaProcess)
    s = 0.0
    for k in kstart:length(P.θ)
        s = s + nu(k, P)
    end
    s
end

"""
    compensator0(kstart, P::LocalGammaProcess)

Compensator of GammaProcess approximating the LocalGammaProcess.
For `kstart == 1` (only choice) this is ``\\nu_0([b_1,\\infty)``.
"""
function compensator0(kstart, P::LocalGammaProcess)
    if kstart == 1
        return P.P.γ*(expint1(P.P.λ*P.b[1]))
    else
        throw(ArgumentError("k != 1 or 2"))
    end
end


"""
    llikelihood(X::SamplePath, Pº::LocalGammaProcess, P::LocalGammaProcess)

Log-likelihood `dPº/dP`. (Up to proportionality.)
"""
function llikelihood(X::SamplePath, Pº::LocalGammaProcess, P::LocalGammaProcess, c = 0.0)::Float64
    assert(Pº.P.γ == P.P.γ)
    if Pº.P.λ == P.P.λ # same on the first bin
        ll = 0.
        for i in 2:length(X.tt)
            x = X.yy[i]-X.yy[i-1] - c
            ll = ll - (θ(x, Pº)-θ(x, P)) # θ(x, P) ≈ θ_k x
        end
        ll = ll - (X.tt[end]-X.tt[1])*(compensator(1, Pº)-compensator(1, P))
        return ll
    elseif Pº.θ === P.θ
        ll = -(Pº.P.λ-(P.P.λ))*(X.yy[end]-X.yy[1])
        return ll - (X.tt[end]-X.tt[1])*(compensator(0, Pº)-compensator(0, P))
    else
        throw(ArgumentError(""))
    end
end


"""
    llikelihood(X::SamplePath, P::LocalGammaProcess)

Bridge log-likelihood with respect to reference measure `P.P`.
(Up to proportionality.)
"""
function llikelihood(X::SamplePath, P::LocalGammaProcess, c = 0.0)::Float64
    ll = 0.
    for i in 2:length(X.tt)
        x = X.yy[i]-X.yy[i-1] - c
        ll = ll - θ(x, P)
    end
    ll = ll - (X.tt[end]-X.tt[1])*(compensator(1, P)-compensator0(1, P))
    return ll
end



