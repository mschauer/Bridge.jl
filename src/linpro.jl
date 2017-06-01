symmetrize(A) = (A + A')/2

lyap{T}(a::SMatrix{1, 1, T}, c::SMatrix{1, 1, T}) =
  SMatrix{1,1,T}(lyap(a[1,1],c[1,1]))

function lyap{T}(a::SMatrix{2, 2, T}, c::SMatrix{2, 2, T})
    d = det(a)
    t = trace(a)
     -(d*c  + (a - t*I)*c*(a-t*I)')/(2*d*t) # http://www.nber.org/papers/w8956.pdf
end
lyap{m,T}(a::SMatrix{m,m,T},c::SMatrix{m,m,T}) =
  SMatrix(lyap(Matrix(a),Matrix(c)))

#####################

mutable struct Ptilde{T} <: ContinuousTimeProcess{T}
    cs::CSpline{T}
    σ
    a
    Γ
    Ptilde{T}(cs, σ) where T = new(cs, σ, σ*σ', inv(σ*σ'))
end
b(t, x, P::Ptilde) = P.cs(t) 
mu(s, x, t, P::Ptilde) = x + integrate(P.cs, s, t)
σ(t, x, P::Ptilde) = P.σ
a(t, x, P::Ptilde) = P.a
gamma(t, x, P::Ptilde) = P.Γ
constdiff(::Ptilde) = true

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

function dotV(t, T, v, P::Ptilde)
     P.cs(t)
end


function H(t, T, P::Ptilde)
    P.Γ/(T-t)
end
function H(t, T, P::Ptilde, x)
    P.Γ*x/(T-t)
end

mutable struct LinPro{T} <: ContinuousTimeProcess{T}
    B
    μ
    σ
    a
    Γ
    lambda # stationary covariance
    function LinPro{T}(B, μ, σ) where T
        a = σ*σ'
        return new(B, μ, σ, σ*σ', inv(a), symmetrize(lyap(B, a)))
    end
end

b(t, x, P::LinPro) = P.B*(x .- P.μ)
σ(t, x, P::LinPro) = P.σ
bderiv(t, x, P::LinPro) = P.B
σderiv(t, x, P::LinPro) = 0.*P.σ

a(t, x, P::LinPro) = P.a
constdiff(::LinPro) = true

"""
    LinPro(B, μ::T, σ) 
    
Linear diffusion ``dX = B(X - μ)dt + σdW``
    
"""    
LinPro(B, μ::T, σ) where T = LinPro{T}(B, μ, σ)


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


function dotV(t, T, v, P::LinPro)
    expm(-(T-t)*P.B)*P.B*(v - P.μ)
end

function mustar(s, x, t, T, P::LinPro)
    inv(P.a - expm(P.B*(T-s))*a*expm(B*(T-s))')*(a - expm(B*(T-t))*a*expm(B*(T-t))')*x
end

#(1-exp(2*(T-t)*b))/(1-exp(2*(T-s)*b))*exp(b*(t-s))
#msinh(b) = (expm(b) - expm(-b))/2
#mu(s, t, T, B) = inv(msinh((T-s)*B))*msinh((T-t)*B)


function Mu(t, T, P::LinPro)
    # phim = expm(-(T-t)*P.B)
    # phi = expm((T-t)*P.B)
    # phim*(phi*P.lambda*phi'-P.lambda)
    P.lambda*expm((T-t)*P.B)'- expm(-(T-t)*P.B)*P.lambda
end

#Mu(tt[n-1], T, P)*inv(Mu(0, T, P))*x0
#@SVector [1.1596192644448247,0.9449866351942955]

#xx = zeros(n);x = x0;for i in 1:n
#       x = x + (P.B -  P.a*Bridge.H(tt[i], tt[end], P))*x*(tt[i+1]-tt[i]) 
#      xx[i] = x[1];end;x
#@SVector [1.1594605378744431,0.9448833540533151]


