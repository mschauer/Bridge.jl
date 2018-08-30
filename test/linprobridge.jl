using Bridge, Distributions, StaticArrays
using Test

n = 10000
T = 2.0
TT = 3.0
tt = 0.0:T/n:T
m = 1000
S = SVector{2,Float64}
M = SArray{Tuple{2,2},Float64,2,4}
B = SMatrix{2,2,Float64}([-1 0.1; -0.2 -1])
mu = 0.0*S([0.2, 0.3])
sigma = SMatrix{2,2,Float64}(2*[-0.212887  0.0687025
  0.193157  0.388997 ])
a = sigma*sigma'
  
P = LinPro(B, mu, sigma)

u = S([1.0, -0.2])
v = S([.5, 0.0])

@test norm(solve(Bridge.R3(), Bridge.b, tt, u, P) - Bridge.mu(0.0, u, T, P)) < 1e-8
@test norm(solve(Bridge.R3(), Bridge._dK, tt, 0*a, P) - Bridge.K(0.0, T, P)) < 1e-8
@test norm(inv(Bridge.solvebackward(Bridge.R3(), Bridge._dHinv, tt, 0*a, P)) - (Bridge.H(0.0, T, P))) < 1e-8
@test norm(solve(Bridge.R3(), Bridge._dPhi, tt, one(a), P) - Bridge.Phi(0.0, T, P)) < 1e-8

Ps = Bridge.LinProBridge(TT, v, P)
#@test norm(solve(Bridge.R3(), Bridge.b, tt, u, Ps) - Bridge.mu(0.0, u, T, Ps)) < 1e-8
#@test norm(solve(Bridge.R3(), Bridge._dPhi, tt, one(a), Ps) - Bridge.Phi(0.0, T, Ps))< 1e-8
@test norm(solve(Bridge.R3(), Bridge._dPhi, tt, one(a), Ps) - Bridge.Phi(0.0, T, Ps)) < 1e-8

#@show solve(Bridge.R3(), (t,x,P)->Bridge.B(t,P)*x, tt, one(a), Ps)*u

P2 = LinPro(B[1,1], mu[1], sqrt(a[1,1]))
Ps2 = Bridge.LinProBridge(TT, v[1], P2)
@test norm(solve(Bridge.R3(), Bridge.b, tt, u[1], Ps2) - Bridge.mu(0.0, u[1], T, Ps2)) < 1e-8
@test norm(solve(Bridge.R3(), Bridge._dK, tt, 0.0, Ps2) - Bridge.K(0.0, T, Ps2)) < 1e-8