abstract type LinearProcess{T} <: ContinuousTimeProcess{T}
end

b(s, x, P::LinearProcess) = B(s, P)*x + β(s, P)
σ(s, x, P::LinearProcess) = σ(s, P)
a(s, P::LinearProcess) = outer(σ(s, P))



symmetrize(A) = (A + A')/2
#####################

mutable struct Ptilde{T} <: ContinuousTimeProcess{T}
    cs::CSpline{T}
    σ
    a
    Γ
    Ptilde{T}(cs, σ) where T = new(cs, σ, σ*σ', inv(σ*σ'))
end
b(t::Float64, x, P::Ptilde) = P.cs(t)

B(t::Float64, P::Ptilde) = 0.0
β(t::Float64, P::Ptilde) = P.cs(t)

mu(s, x, t, P::Ptilde) = x + integrate(P.cs, s, t)
σ(t, x, P::Ptilde) = P.σ
a(t, x, P::Ptilde) = P.a
a(t, P::Ptilde) = P.a
gamma(t, x, P::Ptilde) = P.Γ
constdiff(::Ptilde) = true

"""
    Ptilde(cs::CSpline, σ)

Affine diffusion ``dX = cs(t) dt + σdW``
with cs a cubic spline `::CSpline`.

"""
Ptilde(cs::CSpline{T}, σ) where {T} = Ptilde{T}(cs, σ)


function lp(s, x, t, y, P::Ptilde{T}) where T
    logpdfnormal(y - mu(s,x,t,P), (t-s)*P.a)
end

transitionprob(s,x,t,P::Ptilde) = Gaussian(mu(s,x,t,P), (t-s)*P.a)


function V(t, T, v, P::Ptilde)
    v - integrate(P.cs, t, T)
end

function dotV(t, T, v, P::Ptilde)
     P.cs(t)
end


function H(t, T, P::Ptilde)
    P.Γ/(T-t)
end
function H(t, T, P::Ptilde, x)
    P.Γ*x/(T-t)
end

mutable struct LinPro{S,T,U} <: ContinuousTimeProcess{T}
    B::S
    μ::T
    σ::U
    a::U
    Γ::U
    lambda::S # stationary covariance
    function LinPro(B::S, μ::T, σ::U) where {S,T,U}
        a = σ*σ'
        return new{S,T,U}(B, μ, σ, a, inv(a), symmetrize(lyap(B, a)))
    end
end

B(t, P::LinPro) = P.B
β(t, P::LinPro) = -P.B*P.μ
b(t::Float64, x, P::LinPro) = P.B*(x .- P.μ)
σ(t, x, P::LinPro) = P.σ
bderiv(t, x, P::LinPro) = P.B
σderiv(t, x, P::LinPro) = 0*(P.σ)

a(t, x, P::LinPro) = P.a
a(t, P::LinPro) = P.a
constdiff(::LinPro) = true

"""
    LinPro(B, μ::T, σ)

Linear diffusion ``dX = B(X - μ)dt + σdW``.

"""
LinPro


function lp(s, x, t, y, P::LinPro{T}) where T
    logpdfnormal(y - mu(s,x,t,P), K(s, t, P::LinPro))
end

transitionprob(s,x,t,P::LinPro) = Gaussian(mu(s,x,t,P), K(s, t, P::LinPro))

function Phi(t, T, P::LinPro)
    exp((T-t)*P.B)
end

function mu(t, x, T, P::LinPro)
    phi = exp((T-t)*P.B)
    phi*(x - P.μ) + P.μ
end

function K(t, T, P::LinPro)
    phi = exp((T-t)*P.B)
    P.lambda - phi*P.lambda*phi'
end

function H(t, T, P::LinPro, x)
     phim = exp(-(T-t)*P.B)
     (phim*P.lambda*phim'-P.lambda)\x
end
function H(t, T, P::LinPro)
     phim = exp(-(T-t)*P.B)
     inv(phim*P.lambda*phim'-P.lambda)
end

function V(t, T, v, P::LinPro)
    phim = exp(-(T-t)*P.B)
    phim*(v - P.μ) + P.μ
end

function dotV(t, T, v, P::LinPro)
    exp(-(T-t)*P.B)*P.B*(v - P.μ)
end

"""
    LinProBridge

Bridge process of `P::LinPro` with `μ == 0` conditional on ending in `v` at time `t`.
"""
struct LinProBridge{T,S<:LinPro} <: ContinuousTimeProcess{T}
    t::Float64  # end time
    v::T        # end point
    P::S
    LinProBridge(t, v::T, P::S) where {T,S<:LinPro} = !iszero(P.μ) ? throw(ArgumentError("μ ≠ 0")) : new{T,S}(t,v,P)
end

b(s, x, P::LinProBridge) = P.P.B * (x - P.P.μ) + P.P.a * H(s, P.t, P.P, V(s, P.t, P.v, P.P) - x)
B(s, P::LinProBridge) = P.P.B - P.P.a * H(s, P.t, P.P)

σ(t, x, P::LinProBridge) = P.P.σ
bderiv(t, x, P::LinProBridge) = B(t, P)
σderiv(t, x, P::LinProBridge) = 0*(P.P.σ)

a(t, x, P::LinProBridge) = P.P.a
a(t, P::LinProBridge) = P.P.a
constdiff(::LinProBridge) = true

Phi(t, P::LinProBridge) = P.P.lambda*exp((P.t - t)*P.P.B)' - exp(-(P.t - t)*P.P.B)*P.P.lambda

Phi(t, T, Ps::LinProBridge) = Phi(T, Ps)*inv(Phi(t, Ps))

mu(s, x, t, P::LinProBridge{Float64}) = inv(sinh(P.P.B*(P.t - s)))*( P.v*sinh(P.P.B*(t - s)) + x*sinh(P.P.B*(P.t - t)) )
K(s, t, P::LinProBridge{Float64}) = 2 * P.P.lambda * csch(-P.P.B*(P.t - s)) * sinh(-P.P.B*(t-s))*sinh(-P.P.B*(P.t - t))
# Wolfram alpha: integral_(1/e)^ gamma (sinh^2(-(π - gamma )))/(sinh^2(-(π - s))) ds = sinh(1/e - gamma ) sinh( gamma - π) (-csch(1/e - π))≈0.170692

#function mu(s, x, t, Ps::LinProBridge)
#    T = Ps.t
#    P = Ps.P
#    inv(P.a - expm(P.B*(T-s))*P.a*expm(P.B*(T-s))')*(P.a - expm(P.B*(T-t))*P.a*expm(P.B*(T-t))')*x
#end

transitionprob(s, x, t, P::LinProBridge{Float64}) = Gaussian(mu(s, x, t, P), K(s, t, P))


#################################################

"""
    LinearAppr(tt, B, β, a)
"""
struct LinearAppr{R,S,T} <: ContinuousTimeProcess{T}
    tt::Vector{Float64}
    xx::Vector{T}
    B::Vector{S}
    b::Vector{T}
    Σ::Vector{R}
end

_b((i,s)::IndexedTime, x, P::LinearAppr) = P.B[i]*(x - P.xx[i]) + P.b[i]
B((i,s)::IndexedTime, P::LinearAppr) = P.B[i]
β((i,s)::IndexedTime, P::LinearAppr) = P.b[i] - P.B[i]*P.xx[i]
a((i,s)::IndexedTime, x, P::LinearAppr) = outer(P.Σ[i])
a((i,s)::IndexedTime, P::LinearAppr) = outer(P.Σ[i])
constdiff(::LinearAppr) = false

linearappr(Y, P) = LinearAppr(Y.tt, Y.yy, map((t,x) -> Bridge.bderiv(t, x, P), Y.tt, Y.yy), map((t,x) -> Bridge.b(t, x, P), Y.tt, Y.yy), map((t,x) -> Bridge.σ(t, x, P), Y.tt, Y.yy))
function linearappr!(Pt::LinearAppr, Y, P)
    Pt.tt[:] = Y.tt
    Pt.xx[:] = Y.yy
    Pt.B[:] = map((t,x) -> Bridge.bderiv(t, x, P), Y.tt, Y.yy)
    Pt.b[:] = map((t,x) -> Bridge.b(t, x, P), Y.tt, Y.yy)
    Pt.Σ[:] = map((t,x) -> Bridge.σ(t, x, P), Y.tt, Y.yy)
    Pt
end
