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
    t::Float64
    v::Float64
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


"""
LocalGammaProcess
"""
struct LocalGammaProcess
    P::GammaProcess
    θ # parameter vector
    b1 # grid points b1*0, ..., b1*length(θ), Inf
end

"""
Inverse jump size compared to gamma process
"""
function θ(x, P::LocalGammaProcess)
    N = length(P.θ)
    x <= P.b1 && return 0.
    x >= N*P.b1 && return θ[N]*x
    
    k = Int(div(x,P.b1))
    θ[k]*x
end

"""
(Bin-wise) integral of the Levy measure
"""
function nu(k,P)
    if k == 0
        P.P.γ*(-log(P.P.λ) - expint(1, (P.P.λ)*P.b1))
    elseif k == length(P.θ) 
        P.P.γ*(expint(1, (P.P.λ + P.θ[k])*k*P.b1))
    else
        P.P.γ*(expint(1, (P.P.λ + P.θ[k])*(k)*P.b1) - expint(1, (P.P.λ + P.θ[k])*(k+1)*P.b1))
    end
end

"""
Compensator of LocalGammaProcess 

for kstart = 1, this is sum_k=1^N nu(B_k)
for kstart = 0, this is sum_k=0^N nu(B_k) - C (where C is a constant)
"""
function compensator(kstart, P::LocalGammaProcess)
    s = 0.0
    for k in kstart:length(P.θ)
        s = s + nu(k,P)
    end
end



"""
Log-likelihood with respect to reference measure P.P

Up to proportionality
"""
function llikelihood(X::SamplePath, Pº::LocalGammaProcess, P::LocalGammaProcess)::Float64
    if Pº.P.λ == P.P.λ # same on the first bin
        ll = 0.
        for i in 2:length(X.tt)
            x = X.yy[i]-X.yy[i-1]
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

export LocalGamma


