symmetrize(A) = (A + A')/2
#####################

mutable struct Ptilde{T} <: ContinuousTimeProcess{T}
    cs::CSpline{T}
    σ
    a
    Γ
    Ptilde{T}(cs, σ) where T = new(cs, σ, σ*σ', inv(σ*σ'))
end
b(t, x, P::Ptilde) = P.cs(t) 

B(t, P::Ptilde) = 0.0
β(t, P::Ptilde) = P.cs(t)

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
b(t, x, P::LinPro) = P.B*(x .- P.μ)
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

function mu(t, x, T, P::LinPro)
    phi = expm((T-t)*P.B)
    phi*(x - P.μ) + P.μ
end

function K(t, T, P::LinPro)
    phi = expm((T-t)*P.B)
    P.lambda - phi*P.lambda*phi'
end

function H(t, T, P::LinPro, x)
     phim = expm(-(T-t)*P.B)
     (phim*P.lambda*phim'-P.lambda)\x
end
function H(t, T, P::LinPro)
     phim = expm(-(T-t)*P.B)
     inv(phim*P.lambda*phim'-P.lambda)
end

function V(t, T, v, P::LinPro)
    phim = expm(-(T-t)*P.B)
    phim*(v - P.μ) + P.μ
end

function dotV(t, T, v, P::LinPro)
    expm(-(T-t)*P.B)*P.B*(v - P.μ)
end


