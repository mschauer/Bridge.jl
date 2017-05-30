abstract type LevyProcess{T} <: ContinuousTimeProcess{T} end

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


increment(t, P::GammaProcess) = Gamma(t*P.γ, P.λ)
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
