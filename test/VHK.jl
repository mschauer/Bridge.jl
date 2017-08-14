using Bridge, StaticArrays, Base.Test
srand(5)
n, m = 200, 1000
T = 2.
tt = linspace(0, T, n)

u = 0.5
v = 0.1
β = 0.8
mu = 0.2
a = 0.7

P1 = LinPro(-β, 0*mu, sqrt(a))

cs = Bridge.CSpline(tt[1], tt[end],  
    Bridge.b(tt[1], u, P1), 
    Bridge.b(tt[end], v, P1), 
    (Bridge.b(tt[2], u + Bridge.b(tt[1], u, P1)*(tt[2]-tt[1]), P1)-Bridge.b(tt[1], u, P1))/(tt[2]-tt[1]), # -P1.β*u*(1-exp(-P1.β*dt))/dt
    (Bridge.b(tt[end], v, P1) - Bridge.b(tt[end-1], v - Bridge.b(tt[end], v, P1)*(tt[end]-tt[end-1]), P1))/(tt[end]-tt[end-1])
)

Ptarget = Bridge.Ptilde(cs, sqrt(a))
Pt = LinPro(-β, mu, sqrt(a))

Po = GuidedProp(Ptarget, tt[1], u, T, v, Pt)
GP = Bridge.GuidedBridge(tt, (u,v), Ptarget, Pt)

@test norm(GP.K-[inv(Bridge.H(t, T, Pt)) for t in tt], Inf) < 1e-5
@test norm(GP.V-[Bridge.V(t, T, v, Pt) for t in tt], Inf) < 1e-5

t, x = 0.0, v
@test norm(Bridge.r(t, x, T, v, Pt) - GP.K[1]\(GP.V[1] - x)) < 1e-5
@test norm(Bridge.b(t, x, Po) -  Bridge.b(t, x, Ptarget) - Bridge.a(t, x, Ptarget)*Bridge.r(t, x, T, v, Pt)) < 1e-10
@test norm(Bridge.b(t, x, Po) - Bridge.b(t, x, Ptarget) - a*(GP.K[1]\(GP.V[1] - x))) < 1e-5



S = SVector{1, Float64}
M = SArray{Tuple{1,1},Float64,2,1}
u = S(u)
v = S(v)
β = M(β)
mu = S(mu)
a = M(a)

cs = Bridge.CSpline(tt[1], tt[end],  
    Bridge.b(tt[1], u, P1), 
    Bridge.b(tt[end], v, P1), 
    (Bridge.b(tt[2], u + Bridge.b(tt[1], u, P1)*(tt[2]-tt[1]), P1)-Bridge.b(tt[1], u, P1))/(tt[2]-tt[1]), # -P1.β*u*(1-exp(-P1.β*dt))/dt
    (Bridge.b(tt[end], v, P1) - Bridge.b(tt[end-1], v - Bridge.b(tt[end], v, P1)*(tt[end]-tt[end-1]), P1))/(tt[end]-tt[end-1])
)


Ptarget = Bridge.Ptilde(cs, sqrtm(a))
Pt = LinPro(-β, mu, sqrtm(a))

Po = GuidedProp(Ptarget, tt[1], u, T, v, Pt)
GP = Bridge.GuidedBridge(tt, (u,v), Ptarget, Pt)

@test typeof(GP.K[1]) == M
@test typeof(GP.V[1]) == S

@test norm(GP.K-[inv(Bridge.H(t, T, Pt)) for t in tt], Inf) < 1e-5
@test norm(GP.V-[Bridge.V(t, T, v, Pt) for t in tt], Inf) < 1e-5

t, x = 0.0, v
@test norm(Bridge.r(t, x, T, v, Pt) - GP.K[1]\(GP.V[1] - x)) < 1e-4
@test norm(Bridge.b(t, x, Po) -  Bridge.b(t, x, Ptarget) - Bridge.a(t, x, Ptarget)*Bridge.r(t, x, T, v, Pt)) < 1e-10
@test norm(Bridge.b(t, x, Po) - Bridge.b(t, x, Ptarget) - a*(GP.K[1]\(GP.V[1] - x))) < 1e-5
@test norm(Bridge.b(t, x, Po) - Bridge.bi(1, x, GP)) < 1e-5

