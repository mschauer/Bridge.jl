""" 
    r(t, x, T, v, P)

Returns ``r(t,x) = \operatorname{grad}_x \log p(t,x; T, v)`` where
``p`` is the transition density of the process ``P``.
""" 
function r(t, x, T, v, P)
    H(t, T, P, V(t, T, v, P)-x)
end

tau(s, t, T) = t + (s.-t).*(2-(s-t)/(T-t))
tau(ss::Vector) = tau(ss, ss[1], ss[end])

#####################



mutable struct BridgeProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0; t1; v1
    cs::CSpline{T}
    a
    Γ
    BridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, a, cs) where T = 
        new(Target, t0, v0, t1, v1, cs, a, inv(a))
end
BridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, a, cs=CSpline(t0, t1, zero(T))) = BridgeProp{T}(Target, t0, v0, t1, v1, a, cs)

h(t,x, P::BridgeProp) = P.v1 - x -  integrate(P.cs, t,  P.t1)
b(t, x, P::BridgeProp) = b(t, x, P.Target) + a(t, x, P.Target)*r(t, x, P) 
function bderiv(t, x, P::BridgeProp) 
    assert(constdiff(P))
    bderiv(t, x, P.Target) - a(t, x, P.Target)*P.Γ/(P.t1 - t)
end    

σ(t, x, P::BridgeProp) = σ(t, x, P.Target)
a(t, x, P::BridgeProp) = a(t, x, P.Target)
Γ(t, x, P::BridgeProp) = Γ(t, x, P.Target)
constdiff(P::BridgeProp) = constdiff(P.Target)

btilde(t, x, P::BridgeProp) = P.cs(t)
atilde(t, x, P::BridgeProp) = P.a
ptilde(P::BridgeProp) = Ptilde(P.cs, σ(P.t1, P.v1, P.Target))

 
function r(t, x, P::BridgeProp) 
    P.Γ*h(t, x, P)/(P.t1 - t)
end
function H(t, x, P::BridgeProp) 
    P.Γ/(P.t1 - t)
end

function lptilde{T}(P::BridgeProp{T}) 
    logpdfnormal(P.v1 - (P.v0 + integrate(P.cs, P.t0, P.t1)), (P.t1 -P.t0)*P.a)
end


"""
    GuidedProp

General bridge proposal process
"""    
mutable struct GuidedProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0; t1; v1
    Pt::ContinuousTimeProcess{T}
    GuidedProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, Pt) where T = 
        new(Target, t0, v0, t1, v1, Pt)
end

GuidedProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, Pt::ContinuousTimeProcess{T}) = GuidedProp{T}(Target, t0, v0, t1, v1, Pt)


V(t, P::GuidedProp) = V(t, P.t1, P.v1, P.Pt) 
dotV(t, P::GuidedProp) = dotV(t, P.t1, P.v1, P.Pt) 
r(t, x, P::GuidedProp) = r(t, x, P.t1, P.v1, P.Pt) 
b(t, x, P::GuidedProp) = b(t, x, P.Target) + a(t, x, P.Target)*r(t, x, P.t1, P.v1, P.Pt) 
σ(t, x, P::GuidedProp) = σ(t, x, P.Target)
a(t, x, P::GuidedProp) = a(t, x, P.Target)
Γ(t, x, P::GuidedProp) = Γ(t, x, P.Target)
constdiff(P::GuidedProp) = constdiff(P.Target) && constdiff(P.Pt)

btilde(t, x, P::GuidedProp) = b(t,x,P.Pt)
atilde(t, x, P::GuidedProp) = a(t,x,P.Pt)
ptilde(P::GuidedProp) = P.Pt

function lptilde{T}(P::GuidedProp{T}) 
     lp( P.t0, P.v0, P.t1, P.v1, P.Pt) 
end




#####################


mutable struct PBridgeProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0; tm; vm; t1; v1
    L; Lt; Σ
    cs::CSpline{T}    
    a
    Γ
    PBridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, tm, vm, t1, v1,  L, Σ, a, cs) where T = 
        new(Target, t0, v0, tm, vm, t1, v1, L, L', Σ, cs, a, inv(a))
end
PBridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, tm, vm, t1, v1,  L, Σ, a, cs=CSpline(t0, t1, zero(T))) = PBridgeProp{T}(Target, t0, v0, tm, vm, t1, v1,  L, Σ, a, cs)
        
h1(t,x, P::PBridgeProp) = P.vm - x - integrate(P.cs, t,  P.tm) 
h2(t,x, P::PBridgeProp) = P.v1 - x - integrate(P.cs, t,  P.t1) 

N(t, P::PBridgeProp) = inv(P.L*P.a*P.Lt*(P.tm - t) + (P.t1 - t)/(P.t1 - P.tm)*P.Σ)
Q(t, P::PBridgeProp) = P.Lt*N(t, P)*P.L 

function b(t, x, P::PBridgeProp) 
    if t >= P.tm
        b(t, x, P.Target) + a(t, x, P.Target)*P.Γ*h2(t, x, P)/(P.t1 -t)
    else    
        b(t, x, P.Target) + a(t, x, P.Target)*(Q(t, P)*h1(t, x, P)  +    (P.Γ - Q(t, P)*(P.tm -t))*h2(t, x, P)/(P.t1 -t))
    end
end
function r(t, x, P::PBridgeProp) 
    if t >= P.tm
        P.Γ*h2(t, x, P)/(P.t1 -t)
    else    
        (Q(t, P)*h1(t, x, P)  +    (P.Γ - Q(t, P)*(P.tm -t))*h2(t, x, P)/(P.t1 -t))
    end
end
function H(t, x, P::PBridgeProp) 
    if t >= P.tm
        P.Γ/(P.t1 - t)
    else    
        P.Γ/(P.t1 - t) + Q(t, P)*(P.t1-P.tm)/(P.t1 - t)
    end
end

function lptilde(P::PBridgeProp) 
	n = N(P.t0, P)*(P.tm-P.t0)
	U = Any[	(P.t1-P.t0)/(P.t1-P.tm)/(P.tm-P.t0)*n 		-n*P.L/(P.t1-P.tm)
			-P.L'*n/(P.t1-P.tm) 				(P.Γ + P.L'*n*P.L*(P.tm-P.t0)/(P.t1-P.tm))/(P.t1-P.t0)]
	ldm = sumlogdiag(chol(U[1,1])') +sumlogdiag(chol(U[2, 2] - (U[2,1]*inv(U[1,1])*U[1,2]))')
		 	 				
	mu = [P.L*h1(P.t0, P.v0, P); h2(P.t0, P.v0, P)]
	-length(mu)/2*log(2pi) + ldm - 0.5*dot(mu,U*mu)
end

btilde(t, x, P::PBridgeProp) = P.cs(t)
atilde(t, x, P::PBridgeProp) = P.a
σ(t, x, P::PBridgeProp) = σ(t, x, P.Target)
a(t, x, P::PBridgeProp) = a(t, x, P.Target)
constdiff(P::PBridgeProp) = constdiff(P.Target)


#####################


mutable struct FilterProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0; t1; v1
    L; Lt; Σ
    cs
    a
    Γ
    FilterProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1,  L, Σ, a, cs) where T = 
        new(Target, t0, v0, t1, v1, L, L', Σ, cs, a, inv(a))
end
FilterProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1,  L, Σ, a, cs=CSpline(t0, t1, zero(T))) = FilterProp{T}(Target, t0, v0, t1, v1,  L, Σ, a, cs)
        
h(t,x, P::FilterProp) = P.v1 - x - integrate(P.cs, t, P.t1)

H(t, P::FilterProp) = P.Lt*inv(P.L*P.a*P.Lt*(P.t1 -t) + P.Σ)*P.L

function b(t, x, P::FilterProp) 
    b(t, x, P.Target) + a(t, x, P.Target)*r(t, x, P)
end
function r(t, x, P::FilterProp) 
       H(t, P)*h(t, x, P)
end

btilde(t, x, P::FilterProp) = P.cs(t)
atilde(t, x, P::FilterProp) = P.a
σ(t, x, P::FilterProp) = σ(t, x, P.Target)
a(t, x, P::FilterProp) = a(t, x, P.Target)
constdiff(P::FilterProp) = constdiff(P.Target)

function bderiv(t, x, P::FilterProp) 
    assert(constdiff(P))
    bderiv(t, x, P.Target) - a(t, x, P.Target)*H(t, P)
end    

################################################################


mutable struct DHBridgeProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0::T; t1; v1::T
    DHBridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1) where T = 
        new(Target, t0, v0, t1, v1)

end
DHBridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1) = DHBridgeProp{T}(Target, t0, v0, t1, v1)

b(t, x, P::DHBridgeProp) = (P.v1 - x )/(P.t1 - t)
σ(t, x, P::DHBridgeProp) = σ(t, x, P.Target)
a(t, x, P::DHBridgeProp) = a(t, x, P.Target)
Γ(t, x, P::DHBridgeProp) = Γ(t, x, P.Target)
constdiff(P::DHBridgeProp) = constdiff(P.Target)

function llikelihood{T}(Xcirc::SamplePath{T}, P::DHBridgeProp{T})
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    N = length(tt)
    for i in 1:N-1 #skip last value, summing over n-1 elements
        s = tt[i]
        sh = tt[i+1]
        x = xx[i]
        xh = xx[i+1] 
        m = b(s,x,P.Target)
        G = Γ(s,x,P.Target)
        Gh = Γ(sh,xh,P.Target)
        
        som += dot(m,G*(xh - x - 0.5*m*(sh-s))) # girsanov(Xcirc, P.Target, Wiener{Float64}())
        if i < N-1 
            y = xh - P.v1
            som -= 0.5*dot(y, (Gh-G)*y)/(P.t1 - tt[i+1])
        end
    end
    som
end

function lptilde{T}(P::DHBridgeProp{T}) 
    dv = P.v1-P.v0
    -length(P.v1)/2*log(2pi*(P.t1-P.t0)) -0.5*logdet(a(P.t1,P.v1,P)) - (0.5/(P.t1-P.t0))*dot(dv, Γ(P.t0,P.v0,P)*dv)
end




################################################################

# using left approximation
function llikelihoodleft{T}(Xcirc::SamplePath{T}, Po::Union{GuidedProp{T},BridgeProp{T},PBridgeProp{T},FilterProp{T}})
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    for i in 1:length(tt)-1 #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r = Bridge.r(s, x, Po)
        som += (dot(b(s,x, Po.Target) - btilde(s,x, Po), r)  ) * (tt[i+1]-tt[i])
        if !constdiff(Po)
            som += trace((a(s,x, Po.Target) - atilde(s, x, Po))*(H(s,x,Po) -  r*r')) * (tt[i+1]-tt[i])
        end
    end
    som
end

#using trapezoidal rule
function llikelihoodtrapez{T}(Xcirc::SamplePath{T}, Po::Union{GuidedProp{T},BridgeProp{T},PBridgeProp{T},FilterProp{T}})
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    i = 1
    s = tt[i]
    x = xx[i]
    r = Bridge.r(s, x, Po)
    som += 0.5*(dot(b(s,x, Po.Target) - btilde(s,x, Po), r)  ) * (tt[i+1]-tt[i])
    if !constdiff(Po)
        som += 0.5*trace((a(s,x, Po.Target) - atilde(s, x, Po))*(H(s,x,Po) -  r*r')) * (tt[i+1]-tt[i])
    end

    for i in 2:length(tt)-1 #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r = Bridge.r(s, x, Po)
        som += 0.5*(dot(b(s,x, Po.Target) - btilde(s,x, Po), r)  ) * (tt[i+1]-tt[i-1])
        if !constdiff(Po)
            som += 0.5*trace((a(s,x, Po.Target) - atilde(s, x, Po))*(H(s,x,Po) -  r*r')) * (tt[i+1]-tt[i-1])
        end
    end
    som
end
llikelihood{T}(Xcirc::SamplePath{T}, Po::Union{GuidedProp{T},BridgeProp{T},PBridgeProp{T},FilterProp{T}}) = llikelihoodleft(Xcirc, Po)
#llikelihood{T}(Xcirc::SamplePath{T}, Po::Union{GuidedProp{T},BridgeProp{T},PBridgeProp{T},FilterProp{T}}) = llikelihoodtrapez(Xcirc, Po) 

