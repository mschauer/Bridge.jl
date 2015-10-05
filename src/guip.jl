function cspline(s, t1, t2, p1, p2, m1, m2)
    d = t2-t1
    t = (s-t1)/(t2-t1)
    t2 = t*t
    t3 = t2*t
    z = @fsa([2. -3. 0. 1.; -2. 3. 0. 0.; 1. -2. 1. 0.; 1. -1. 0. 0.])* @fsa([t3, t2, t, 1.])
    z[1]*p1 + z[2]*p2 + z[3]*d*m1 + z[4]*d*m2
end
function intcspline(s, t1, t2, p1, p2, m1, m2)
        d = t2-t1
        t = (s-t1)/(t2-t1)
        t2 = t*t
        t3 = t2*t
        t4 = t2*t2
        t4, t3, t2 = t4/4, t3/3, t2/2
        z = @fsa([2. -3. 0. 1.; -2. 3. 0. 0.; 1. -2. 1. 0.; 1. -1. 0. 0.])* @fsa([t4, t3, t2, t])
        (z[1]*p1 + z[2]*p2 + z[3]*d*m1 + z[4]*d*m2)*d
end
intcspline(s, T, t1, t2, p1, p2, m1, m2) = intcspline(T, t1, t2, p1, p2, m1, m2) - intcspline(s, t1, t2, p1, p2, m1, m2)

type CSpline{T}
    s; t; x::T; y::T; mx; my
end
CSpline{T}(s, t, x::T, y = x, m0 = (y-x)/(t-s), m1 = m0) = CSpline{T}(s, t, x, y, mx, my)
call(cs::CSpline, t) =  cspline(t, cs.s, cs.t, cs.x, cs.y, cs.mx, cs.my)
integrate(cs::CSpline, s, t) = intcspline(s,t, cs.s, cs.t, cs.x, cs.y, cs.mx, cs.my)

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
Ptilde{T}(cs::CSpline{T}, σ) = Ptilde{T}(cs, σ)


function lp{T}(s, x, t, y, P::Ptilde{T}) 
    logpdfnormal(y - mu(s,x,t,P), (t-s)*P.a)
end

#####################



type BridgeProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0; t1; v1
    cs::CSpline{T}
    
    
    a
    Γ

    BridgeProp(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, a, cs) = new(Target, 
        t0, v0, t1, v1, 
        cs,
        a, inv(a))

end
BridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, a, cs=CSpline(t0, t1, zero(T))) = BridgeProp{T}(Target, t0, v0, t1, v1, a, cs)

h(t,x, P::BridgeProp) = P.v1 - x -  integrate(P.cs, t,  P.t1)
b(t, x, P::BridgeProp) = b(t, x, P.Target) + a(t, x, P.Target)*r(t, x, P) 
σ(t, x, P::BridgeProp) = σ(t, x, P.Target)
a(t, x, P::BridgeProp) = a(t, x, P.Target)
btilde(t, x, P::BridgeProp) = P.cs(t)
atilde(t, x, P::BridgeProp) = P.a
function r(t, x, P::BridgeProp) 
    P.Γ*h(t, x, P)/(P.t1 - t)
end
function H(t, x, P::BridgeProp) 
    P.Γ/(P.t1 - t)
end


#####################


type PBridgeProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0; tm; vm; t1; v1
    L; Lt; Σ
    cs::CSpline{T}
    
    a
    Γ

    PBridgeProp(Target::ContinuousTimeProcess{T}, t0, v0, tm, vm, t1, v1,  L, Σ, a, cs) = new(Target, 
        t0, v0, tm, vm, t1, v1, 
        L, L', Σ,
        cs,
        a, inv(a))
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
	ldm = sumlogdiag(chol(U[1,1], Val{:L})) +sumlogdiag(chol(U[2, 2] - (U[2,1]*inv(U[1,1])*U[1,2]), Val{:L}))
		 	 				
	mu = [P.L*h1(P.t0, P.v0, P); h2(P.t0, P.v0, P)]
	-length(mu)/2*log(2pi) + ldm - 0.5*dot(mu,U*mu)
end

btilde(t, x, P::PBridgeProp) = P.cs(t)
atilde(t, x, P::PBridgeProp) = P.a
σ(t, x, P::PBridgeProp) = σ(t, x, P.Target)
a(t, x, P::PBridgeProp) = a(t, x, P.Target)



#####################


type FilterProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0; t1; v1
    L; Lt; Σ
    cs
    
    a
    Γ

    FilterProp(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1,  L, Σ, a, cs) = new(Target, 
        t0, v0, t1, v1, 
        L, L', Σ,
        cs,
        a, inv(a))
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



function llikelihood{T}(Xcirc::SamplePath{T}, Pt::Union{BridgeProp{T},PBridgeProp{T},FilterProp{T}}; consta = true)
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    for i in 1:length(tt)-1 #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r = Bridge.r(s, x, Pt)
        som += (dot(b(s,x, Pt.Target) - btilde(s,x, Pt), r)  ) * (tt[i+1]-tt[i])
        if consta == false
            som += trace((a(s,x, Pt.Target) - atilde(s, x, Pt))*(H(s,x,Pt) -  r*r')) * (tt[i+1]-tt[i])
        end
    end
    som
end

function lptilde{T}(P::BridgeProp{T}) 
    logpdfnormal(P.v1 - (P.v0 + integrate(P.cs, P.t0, P.t1)), (P.t1 -P.t0)*P.a)
end



