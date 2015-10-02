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

type BridgeProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0; t1; v1
    p0; p1; m0; m1 #beta(t0), beta(t2), beta'(t0), beta'(t2)
    
    
    a
    Γ

    BridgeProp(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, a, p0, p1, m0, m1) = new(Target, 
        t0, v0, t1, v1, 
        p0, p1, m0, m1,
        a, inv(a))

end
BridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, a, p0 = zero(T), p1 = p0, m0 = zero(T), m1 = zero(T)) = BridgeProp{T}(Target, t0, v0, t1, v1, a, p0, p1, m0, m1)

h(t,x, P::BridgeProp) = P.v1 - x - intcspline(t, P.t1, P.t0, P.t1, P.p0, P.p1, P.m0, P.m1)
b(t, x, P::BridgeProp) = b(t, x, P.Target) + a(t, x, P.Target)*r(t, x, P) 
σ(t, x, P::BridgeProp) = σ(t, x, P.Target)
a(t, x, P::BridgeProp) = a(t, x, P.Target)
btilde(t, x, P::BridgeProp) = cspline(t, P.t0, P.t1, P.p0, P.p1, P.m0, P.m1) 
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
    p0; p1; m0; m1 #beta(t0), beta(t1), beta'(t0), beta'(t1)
    
    a
    Γ

    PBridgeProp(Target::ContinuousTimeProcess{T}, t0, v0, tm, vm, t1, v1,  L, Σ, a, p0, p1, m0 , m1) = new(Target, 
        t0, v0, tm, vm, t1, v1, 
        L, L', Σ,
        p0, p1, m0, m1,
        a, inv(a))
end
PBridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, tm, vm, t1, v1,  L, Σ, a, p0 = zero(T), p1 = p0, m0 = zero(T), m1 = zero(T)) = PBridgeProp{T}(Target, t0, v0, tm, vm, t1, v1,  L, Σ, a, p0, p1, m0 , m1)
        
h1(t,x, P::PBridgeProp) = P.vm - x - intcspline(t, P.tm, P.t0, P.t1, P.p0, P.p1, P.m0, P.m1)
h2(t,x, P::PBridgeProp) = P.v1 - x - intcspline(t, P.t1, P.t0, P.t1, P.p0, P.p1, P.m0, P.m1)

N(t, P::PBridgeProp) = inv(P.L*P.a*P.Lt + (P.t1 - t)/(P.tm - t)/(P.t1 - P.tm)*P.Σ)
Q(t, P::PBridgeProp) = P.Lt*N(t, P)*P.L 

function b(t, x, P::PBridgeProp) 
    if t >= P.tm
        b(t, x, P.Target) + a(t, x, P.Target)*P.Γ*h2(t, x, P)/(P.t1 -t)
    else    
        b(t, x, P.Target) + a(t, x, P.Target)*(Q(t, P)*h1(t, x, P)/(P.tm -t)  +    (P.Γ - Q(t, P))*h2(t, x, P)/(P.t1 -t))
    end
end
function r(t, x, P::PBridgeProp) 
    if t >= P.tm
        P.Γ*h2(t, x, P)/(P.t1 -t)
    else    
        (Q(t, P)*h1(t, x, P)/(P.tm -t)  +    (P.Γ - Q(t, P))*h2(t, x, P)/(P.t1 -t))
    end
end
function H(t, x, P::PBridgeProp) 
    if t >= P.tm
        P.Γ/(P.t1 - t)
    else    
        P.Γ/(P.t1 - t) + Q(t, P)*(P.t1-P.tm)/(P.tm - t)/(P.t1 - t)
    end
end


btilde(t, x, P::PBridgeProp) = cspline(t, P.t0, P.t1, P.p0, P.p1, P.m0, P.m1)
atilde(t, x, P::PBridgeProp) = P.a
σ(t, x, P::PBridgeProp) = σ(t, x, P.Target)
a(t, x, P::PBridgeProp) = a(t, x, P.Target)



function llikelihood{T}(Xcirc::SamplePath{T}, Pt::Union{BridgeProp{T},PBridgeProp{T}}; consta = true)
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    for i in 1:length(tt)-1 #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r = Bridge.r(s, x, Pt)
        som += (dot(b(s,x, Pt.Target) - btilde(s,x, Pt), r)  ) * (tt[i+1]-tt[i])
        if consta == false
            som += trace((a(s,x, Pt.Target) - atilde(s, x, Pt))*(H(s,x,Pt) -  r*r'))
        end
    end
    som
end

function lptilde{T}(P::BridgeProp{T}) 
    logpdfnormal(P.v1 - (P.v0 + intcspline(P.t0, P.t1, P.t0, P.t1, P.p0, P.p1, P.m0, P.m1)), (P.t1 -P.t0)*P.a)
end




