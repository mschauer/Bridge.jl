symmetrize(A) = (A + A')/2
type LinPro{T} <: ContinuousTimeProcess{T}
    B
    μ
    σ
    a
    Γ
    lambda
    LinPro(B, μ, σ) = let 
        a = σ*σ'
        new(B, μ, σ, σ*σ', inv(a'), -symmetrize(lyap(B, a)))
        end
end

b(t, x, P::LinPro) = P.B*(x .- P.μ)
σ(t, x, P::LinPro) = P.σ
a(t, x, P::LinPro) = P.a
LinPro{T}(B, cs::CSpline{T}, σ) = LinPro{T}(cs, σ)


function lp{T}(s, x, t, y, P::LinPro{T}) 
    logpdfnormal(y - mu(s,x,t,P), (t-s)*P.a)
end

function mu(t, x, T, P::LinPro)
    phi = expm((T-t)*P.B)
    phi*(x - P.μ) - P.μ
end

function H(t, T, P::LinPro)
     phim = expm(-(T-t)*P.B)
     inv(phim*P.lambda*phim'-P.lambda)
end


function K(t, T, P::LinPro)
    phi = expm((T-t)*P.B)
    P.lambda - phi*P.lambda*phi'
end

 
function V(t, T, v, P::LinPro)
    phim = expm(-(T-t)*P.B)
    phim*(v - P.μ) + P.μ
end


""" 
    r(t, x, T, v, P)

Returns ``r(t,x) = \operatorname{grad}_x \log p(t,x; T, v)`` where
``p`` is the transition density of the process ``P``.
""" 
function r(t, x, T, v, P)
    H(t, T, P, V(t, T, v, P)-x)
end


"""
    LinBridgeProp

General bridge proposal process
"""    
type LinBridgeProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0; t1; v1
    Pt::ContinuousTimeProcess{T}

    LinBridgeProp(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, a, cs) = new(Target, 
        t0, v0, t1, v1, 
        Pt,
        a, inv(a))

end
LinBridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, a, cs=CSpline(t0, t1, zero(T))) = LinBridgeProp{T}(Target, t0, v0, t1, v1, a, cs)

h(t,x, P::LinBridgeProp) = P.v1 - x -  integrate(P.cs, t,  P.t1)
b(t, x, P::LinBridgeProp) = b(t, x, P.Target) + a(t, x, P.Target)*r(t, x, P) 
σ(t, x, P::LinBridgeProp) = σ(t, x, P.Target)
a(t, x, P::LinBridgeProp) = a(t, x, P.Target)
Γ(t, x, P::LinBridgeProp) = Γ(t, x, P.Target)

btilde(t, x, P::LinBridgeProp) = b(t,x,P.Pt)
atilde(t, x, P::LinBridgeProp) = a(t,x,P.Pt)
function r(t, x, P::LinBridgeProp) 
    r(t, x, P.t1, P.v1, P.Pt) 
end
function H(t, x, P::LinBridgeProp) 
    H(t, P.t1, P.v1, P.Pt) 
end
