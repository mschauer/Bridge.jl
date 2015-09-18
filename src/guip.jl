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
    #z = @fsa([2. -3. 0. 1.; -2. 3. 0. 0.; 1. -2. 1. 0.; 1. -1. 0. 0.])* @fsa([t4/4, t3/3, t2/2, t])
    #z[1]*p1 + z[2]*p2 + z[3]*d*m1 + z[4]*d*m2
    t4, t3, t2 = t4/4, t3/3, t2/2
    p2*(3.*t3-2.*t4)+d*m1*(+1.*t2-2.*t3+1.*t4)+d*m2*(-1.* t3+1.*t4)+ p1*(1.*t1-3.*t3+2.*t4)
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
b(t, x, P::BridgeProp) = b(t, x, P.Target) + a(t, x, P.Target)*P.Γ*h(t, x, P::BridgeProp)/(P.t1 -t)
σ(t, x, P::BridgeProp) = σ(t, x, P.Target)
a(t, x, P::BridgeProp) = a(t, x, P.Target)


type PBridgeProp{T} <: ContinuousTimeProcess{T}
    Target
    t0; v0; t1; v1; t2; v2
    L; Lt; Σ
    p0; p2; m0; m2 #beta(t0), beta(t2), beta'(t0), beta'(t2)
    
    a
    Γ

    PBridgeProp(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, t2, v2,  L, Σ, a, p0, p2, m0 , m2) = new(Target, 
        t0, v0, t1, v1, t2, v2, 
        L, L', Σ,
        p0, p2, m0, m2,
        a, inv(a))
end
PBridgeProp{T}(Target::ContinuousTimeProcess{T}, t0, v0, t1, v1, t2, v2,  L, Σ, a, p0 = zero(T), p2 = p0, m0 = zero(T), m2 = zero(T)) = PBridgeProp{T}(Target, t0, v0, t1, v1, t2, v2,  L, Σ, a, p0, p2, m0 , m2)
        
h1(t,x, P::PBridgeProp) = P.v1 - x - intcspline(t, P.t1, P.t0, P.t2, P.p0, P.p2, P.m0, P.m2)
h2(t,x, P::PBridgeProp) = P.v2 - x - intcspline(t, P.t2, P.t0, P.t2, P.p0, P.p2, P.m0, P.m2)

N(t, P::PBridgeProp) = inv(P.L*P.a*P.Lt + (P.t2 - t)/(P.t1 - t)/(P.t2 - P.t1)*P.Σ)
Q(t, P::PBridgeProp) = P.Lt*N(t, P)*P.L 

function b(t, x, P::PBridgeProp) 
    if t >= P.t1
        b(t, x, P.Target) + a(t, x, P.Target)*P.Γ*h2(t, x, P)/(P.t2 -t)
    else    
        b(t, x, P.Target) + a(t, x, P.Target)*(Q(t, P)*h1(t, x, P)/(P.t1 -t)  +    (P.Γ - Q(t, P))*h2(t, x, P)/(P.t2 -t))
    end
end
function r(t, x, P::PBridgeProp) 
    if t >= P.t1
        P.Γ*h2(t, x, P)/(P.t2 -t)
    else    
        (Q(t, P)*h1(t, x, P)/(P.t1 -t)  +    (P.Γ - Q(t, P))*h2(t, x, P)/(P.t2 -t))
    end
end

btilde(t, x, P::PBridgeProp) = cspline(t, P.t0, P.t2, P.p0, P.p2, P.m0, P.m2) 


σ(t, x, P::PBridgeProp) = σ(t, x, P.Target)
a(t, x, P::PBridgeProp) = a(t, x, P.Target)


#function girsanov{T}(Y::SamplePath{T}, P1, P2)
#    for i in 1:length(Y.tt)
#
#end


function llikelihood{T}(Xcirc::SamplePath{T}, Pt::PBridgeProp{T})
    tt = Xcirc.tt
    xx = Xcirc.yy

    som::Float64 = 0.
    for i in 1:length(tt)-1 #skip last value, summing over n-1 elements
      s = tt[i]
      x = xx[i]
      som += (dot(b(s,x, Pt.Target) - btilde(s,x, Pt), Bridge.r(s, x, Pt))  ) * (tt[i+1]-tt[i])
    end
    som
end





