symmetrize(A) = (A + A')/2


#####################

type Ptilde{T} <: ContinuousTimeProcess{T}
    cs::CSpline{T}
    σ
    a
    Γ
    Ptilde(cs, σ) = new(cs, σ, σ*σ', inv(σ*σ'))
end
b(t, x, P::Ptilde) = P.cs(t) 
mu(s, x, t, P::Ptilde) = x + integrate(P.cs, s, t)
σ(t, x, P::Ptilde) = P.σ
a(t, x, P::Ptilde) = P.a
gamma(t, x, P::Ptilde) = P.Γ

"""
    Ptilde(cs::CSpline, σ) 
    
Affine diffusion ``dX = cs(t) dt + σdW`` 
with cs a 
    
"""    
Ptilde{T}(cs::CSpline{T}, σ) = Ptilde{T}(cs, σ)


function lp{T}(s, x, t, y, P::Ptilde{T}) 
    logpdfnormal(y - mu(s,x,t,P), (t-s)*P.a)
end

transitionprob(s,x,t,P::Ptilde) = Gaussian(mu(s,x,t,P), (t-s)*P.a)


function V(t, T, v, P::Ptilde)
    v - integrate(P.cs, t, T)
end

function H(t, T, P::Ptilde)
    P.Γ/(T-t)
end
function H(t, T, P::Ptilde, x)
    P.Γ*x/(T-t)
end

type LinPro{T} <: ContinuousTimeProcess{T}
    B
    μ
    σ
    a
    Γ
    lambda # stationary covariance
    LinPro(B, μ, σ) = let 
        a = σ*σ'
        new(B, μ, σ, σ*σ', inv(a), symmetrize(lyap(B, a)))
        end
end

b(t, x, P::LinPro) = P.B*(x .- P.μ)
σ(t, x, P::LinPro) = P.σ
a(t, x, P::LinPro) = P.a
"""
    LinPro(B, μ::T, σ) 
    
Linear diffusion ``dX = B(X - μ)dt + σdW``
    
"""    
LinPro{T}(B, μ::T, σ) = LinPro{T}(B, μ, σ)


function lp{T}(s, x, t, y, P::LinPro{T}) 
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


