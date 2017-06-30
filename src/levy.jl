abstract type LevyProcess{T} <: ContinuousTimeProcess{T} end


"""
    GammaProcess

A *GammaProcess* with jump rate `γ` and inverse jump size `λ` has increments `Gamma(t*γ, 1/λ)` and Levy measure

```math
ν(x)=γ x^{-1}\\exp(-λ x), 
```

Here `Gamma(α,θ)` is the Gamma distribution in julia's parametrization with shape parameter `α` and scale `θ`
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


struct GammaBridge  <: ContinuousTimeProcess{Float64}
    t::Float64;v::Float64
    P::GammaProcess
end



function sample{T}(tt::AbstractVector{Float64}, P::LevyProcess{T}, x1=zero(T))
    tt = collect(tt)
    yy = zeros(T,length(tt))

    yy[1] = x = x1
    for i in 2:length(tt)
        x = x + rand(increment(tt[i]-tt[i-1], P))
        yy[i] = x
    end
    SamplePath{T}(tt, yy)
end


increment(t, P::GammaProcess) = Gamma(t*P.γ, 1/P.λ)

lp(s, x, t, y, P::GammaProcess) = logpdf(increment(t-s, P), y-x)

increment(t, P::VarianceGammaProcess) = VarianceGamma(P.θ, P.σ, t, P.ν)

function rand(P::VarianceGamma) 
    Z = randn()
    G = rand(Gamma(P.t/P.ν, P.ν))
    P.θ*G + P.σ*sqrt(G)*Z
end


function sample(tt::AbstractVector{Float64}, P::GammaBridge, x1::Float64 = 0.)
    tt = collect(tt) 
    t = P.t
    r = searchsorted(tt, t)
    if isempty(r)
       tt = Float64[tt[1:last(r)]; t; tt[first(r):end]]
    end
    X = sample(tt, P.P, x1)    
    yy = X.yy
    yy[:] = yy ./ yy[first(r)]
    if isempty(r)
        tt = [tt[1:last(r)]; tt[first(r)+1:end]]                
        yy = [yy[1:last(r)]; yy[first(r)+1:end]]   
    end             
    SamplePath{Float64}(tt, yy)
end


"""
LocalGammaProcess
"""
struct LocalGammaProcess
    P::GammaProcess
    ϵ
    alpha
    x
    k
end

"""
inverse jump size compared to gamma process
"""
function bigλ(x, P::LocalGammaProcess)
    
    x <= P.ϵ && return 0.
    x >= P.x && return -alpha[i]
    
    i = floor(Int, P.k*(x-P.ϵ)/(P.x-P.ϵ))
    -alpha[i]
end

function comp(P::LocalGammaProcess)
    s = 0.0
    for k in 1:P.k-1
        dx = (P.x- P.ϵ)/(k-1)
        s = s + P.γ*(expint(1, P.alpha[k]*( P.ϵ + (k-1)*dx)) - expint(1, P.alpha[k+1]*( P.ϵ + k*dx)))
    end
    s = s + P.γ*(expint(1, P.alpha[P.k]*(P.x))) # might be removed
end


"""
Up to proportionality
"""
function llikelihood(X::SamplePath, P::LocalGammaProcess)
    ll = 0.
    for i in 2:length(X.tt)
        dt = X.tt[i]-X.tt[i]
        ll += bigλ(X.yy-X.xx, P)
    end
    ll - (X.tt[end]-X.tt[1])*comp(P)
end

export LocalGamma


