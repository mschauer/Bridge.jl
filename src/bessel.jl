"""
    Bessel3Bridge(t, v, σ)

Bessel(3) bridge from below or above to the point `v` at time `t`,
not crossing `v`, with dispersion σ.
"""
struct Bessel3Bridge <: ContinuousTimeProcess{Float64}
    t::Float64
    v::Float64
    σ::Float64
end

"""
    Bessel{N}(σ)

N-dimensional Bessel process with dispersion σ.
Sample with
```
u = 0.0
t = 0:0.1:1
σ = 1.0
sample(u, t, Bridge.Bessel{3}(σ))
```
"""
struct Bessel{N} <: ContinuousTimeProcess{Float64}
    σ::Float64
end

function sample(u, tt, P::Bessel{N}) where N
    tt2 = tt*P.σ^2
    w = sample(tt2, Wiener(), u).yy.^2
    for i in 2:N
        w .+= (sample(tt2, Wiener()).yy).^2
    end
    SamplePath(tt, sqrt.(w))
end

function bessel3(u, tt, t, v, si)
    @assert u != v
    tt2 = tt*si^2
    w2 = sample(tt2, WienerBridge(t*si^2, 0.), 0.).yy
    w3 = sample(tt2, WienerBridge(t*si^2, 0.), 0.).yy

    if u > v
        w1 = sample(tt2, WienerBridge(t*si^2, 0.), u - v).yy
        SamplePath(tt, v .+ sqrt.(w1.^2 + w2.^2 + w3.^2))
    else
        w1 = sample(tt2, WienerBridge(t*si^2, 0.), v - u).yy
        SamplePath(tt, v .- sqrt.(w1.^2 + w2.^2 + w3.^2))
    end

end



b(t, x, P::Bessel3Bridge) = outer(P.σ)*inv(x - P.v) + inv(P.t - t)*(P.v - x)
σ(t, x, P::Bessel3Bridge) = P.σ
a(t, x, P::Bessel3Bridge) = outer(P.σ)
Γ(t, x, P::Bessel3Bridge) = inv(outer(P.σ))
constdiff(P::Bessel3Bridge) = true

sample(u, tt, P::Bessel3Bridge) = bessel3(u, tt, P.t, P.v, P.σ)




"""
    BesselProp

Bessel type proposal
"""
mutable struct BesselProp <: ContinuousTimeProcess{Float64}
    Target
    t; v
    BesselProp(Target::ContinuousTimeProcess{Float64}, t, v) = new(Target, t, v)
end

r(t, x, P::BesselProp) = inv(x - P.v) + inv((P.t - t)*a(P.t, P.v, P.Target))*(P.v-x)
H(t, x, P::BesselProp) =  1/(x - P.v)^2 + inv((P.t - t)*a(P.t, P.v, P.Target))


b(t, x, P::BesselProp) = b(t, x, P.Target) + a(t, x, P.Target)*r(t, x, P)
σ(t, x, P::BesselProp) = σ(t, x, P.Target)
a(t, x, P::BesselProp) = a(t, x, P.Target)
Γ(t, x, P::BesselProp) = Γ(t, x, P.Target)
constdiff(P::BesselProp) = constdiff(P.Target)

btilde(t, x, P::BesselProp) = 0*x
atilde(t, x, P::BesselProp) = a(P.t,P.v,P.Target)



function lptilde(s,u, P::BesselProp)
        v = P.v
        t = P.t
        at = a(t,v,P.Target)
        #abs(v-u)/sqrt(2pi*at*(t-s)^3)*exp(-(v-u)^2/(2*at*(t-s)))

        1/2*((u - v)^2/(at * (s - t)) - log(2π*at) - 3log(t - s) + 2log(abs(v - u)))
end



"""
    aeuler(u, s:dtmax:t, P, tau=0.5)

Adaptive Euler-Maruyama scheme from https://arxiv.org/pdf/math/0601029.pdf
sampling a path from u at s to t with adaptive stepsize of 2.0^(-k)*dtmax
"""
function aeuler(u, r, P::ContinuousTimeProcess{Float64}, tau = 0.5, kmax = 10)

    s = first(r)
    t = last(r)
    dtmax = step(r)
    k = 1
    yy = zeros(0)
    tt = zeros(0)

    y = u
    dt = 2.0^(-k)*dtmax
    while s < t
        append!(yy, y)
        append!(tt, s)
        k = max(0,k - 1)
        B = b(s, y, P)
        while k < kmax && abs(B-b(s, y + B* 2.0^(-k)*dtmax, P)) > tau
            k = k + 1
        end
        dt = 2.0^(-k)*dtmax
        y = y + B*dt + σ(s, y, P)*sqrt(dt)*randn()
        s = s + dt
    end
    SamplePath(tt,yy)
end

llikelihood(Xcirc::SamplePath{T}, Po::BesselProp) where {T} = llikelihoodleft(Xcirc, Po)
