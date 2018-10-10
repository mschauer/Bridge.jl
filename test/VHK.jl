using Bridge, StaticArrays, Distributions
using Test, Random, LinearAlgebra
Random.seed!(5)
n, m = 200, 1000
T = 2.
tt = range(0, stop=T, length=n)

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
GP = Bridge.GuidedBridge(tt, Ptarget, Pt, v)

@test norm(GP.H♢-[inv(Bridge.H(t, T, Pt)) for t in tt], Inf) < 1e-5
@test norm(GP.V-[Bridge.V(t, T, v, Pt) for t in tt], Inf) < 1e-5

t, x = 0.0, v
@test norm(Bridge.r(t, x, T, v, Pt) - GP.H♢[1]\(GP.V[1] - x)) < 1e-5
@test norm(Bridge.b(t, x, Po) -  Bridge.b(t, x, Ptarget) - Bridge.a(t, x, Ptarget)*Bridge.r(t, x, T, v, Pt)) < 1e-10
@test norm(Bridge.b(t, x, Po) - Bridge.b(t, x, Ptarget) - a*(GP.H♢[1]\(GP.V[1] - x))) < 1e-5


@test abs(Bridge.mu(t, u, T, Pt) - (exp(-β*(T-t))*(u-mu) + mu)) < 1e-5
@test abs(solve(Bridge.R3(), Bridge._traceB, tt, 0.0, GP.Pt) - log(det(exp(-β*(T-t))))) < 1e-5

P = GP
x1 = (P.V[1] - u)'*inv(P.H♢[1])*(P.V[1] - u)
x2 = (P.V[end] - Bridge.gpmu(P.tt, u, P.Pt))'*inv(Bridge.gpK(P.tt, Bridge.outer(zero(u)), P.Pt))*( P.V[end] - Bridge.gpmu(P.tt, u, P.Pt))
@test norm(x1 - x2) < 1e-7

Phi = exp(-β*(T-t))

y0 = log(det(P.H♢[1]))
y1 = log(det(Phi*P.H♢[1]*Phi'))
y2 = log(det(Bridge.gpK(P.tt, Bridge.outer(zero(u)), P.Pt)))
y3 = solve(Bridge.R3(), Bridge._traceB, tt, 0.0, GP.Pt)
y4 = log(det(Phi*Phi))/2
@test norm(y1 - y2) < 1e-5
@test norm(y3 - y4) < 1e-5

@test norm( -x2/2 - log(2pi)/2 - y2/2 - Bridge.lptilde2(GP, u) ) < 1e-5
@test norm( -x2/2 - log(2pi)/2 - y0/2 - y3 - Bridge.lptilde2(GP, u) ) < 1e-5
@test norm( -x2/2 - log(2pi)/2 - y2/2  - log(pdf(Normal(P.V[end] - Bridge.gpmu(P.tt, u, P.Pt), sqrt(Bridge.gpK(P.tt, Bridge.outer(zero(u)), P.Pt))), 0.0))) < 1e-5
@test norm( -x2/2 - log(2pi)/2 - y0/2  - log(pdf(Normal(P.V[1], sqrt(P.H♢[1])), u))) < 1e-5

@test norm(Bridge.logpdfnormal(P.V[1] - u, P.H♢[1]) - log(pdf(Normal(P.V[1], (sqrt(P.H♢[1]))), u))) < 1e-5

@test norm(Bridge.lptilde2(GP, u) - Bridge.lp(t, u, T, v, Pt)) < 1e-5

@test norm(Bridge.lptilde(GP, u) - Bridge.lp(t, u, T, v, Pt)) < 1e-5


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


Ptarget = Bridge.Ptilde(cs, sqrt(a))
Pt = LinPro(-β, mu, sqrt(a))

Po = GuidedProp(Ptarget, tt[1], u, T, v, Pt)
GP = Bridge.GuidedBridge(tt, Ptarget, Pt, v)

@test typeof(GP.H♢[1]) == M
@test typeof(GP.V[1]) == S

@test norm(GP.H♢-[inv(Bridge.H(t, T, Pt)) for t in tt], Inf) < 1e-5
@test norm(GP.V-[Bridge.V(t, T, v, Pt) for t in tt], Inf) < 1e-5

t, x = 0.0, v
@test norm(Bridge.r(t, x, T, v, Pt) - GP.H♢[1]\(GP.V[1] - x)) < 1e-4
@test norm(Bridge.b(t, x, Po) -  Bridge.b(t, x, Ptarget) - Bridge.a(t, x, Ptarget)*Bridge.r(t, x, T, v, Pt)) < 1e-10
@test norm(Bridge.b(t, x, Po) - Bridge.b(t, x, Ptarget) - a*(GP.H♢[1]\(GP.V[1] - x))) < 1e-5
@test norm(Bridge.b(t, x, Po) - Bridge._b((1,NaN), x, GP)) < 1e-5

@test norm(Bridge.solvebackward!(Bridge.R3(), Bridge.F, SamplePath(tt,zeros(length(tt))), 2.0, ((t,x)->-x,)).yy[1] -
    2exp(tt[end]-tt[1]))<1e-5
