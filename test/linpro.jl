using Bridge, Distributions, StaticArrays
using Test
#import Bridge: b, σ, a, transitionprob
n = 1000
TT = 5
tt = 0.0:TT/n:TT
m = 1000
S = SVector{2,Float64}
M = SArray{Tuple{2,2},Float64,2,4}
B = SMatrix{2,2,Float64}([-1 0.1; -0.2 -1])
mu = 0.1*S([0.2, 0.3])
sigma = SMatrix{2,2,Float64}(2*[-0.212887  0.0687025
  0.193157  0.388997 ])
a = sigma*sigma'
  
P = LinPro(B, mu, sigma)

u = S([1.0, 0.0])
v = S([.5, 0.0])

@test (norm(Matrix(-P.lambda*B' - B*P.lambda - a))) < eps()


t = 0.5
T = 2.0



@test Bridge.bderiv(t, v, P) == B
@test Bridge.σderiv(t, v, P) == zero(sigma)



dt = 1e-6
@test norm( (Bridge.V(t+dt, T, v, P)-Bridge.V(t, T, v, P))/dt - Bridge.dotV(t, T, v, P) ) < 10*dt


n2 = 150
tt = range(t, stop=T, length=n2)
K = SamplePath(tt, zeros(M, length(tt)))
Bridge.gpHinv!(K, P)
V = SamplePath(tt, zeros(S, length(tt)))
Bridge.gpHinv!(K, P) # warm up
Bridge.gpV!(V, P, v)
Mu = SamplePath(tt, zeros(S, length(tt)))
Mu2 = SamplePath(tt, zeros(S, length(tt)))

solve!(Bridge.R3(), Bridge.b, Mu, u, P)
solve!(BS3(), Bridge.b, Mu2, u, P)
@test (@allocated Bridge.gpHinv!(K, P)) == 0
@test (@allocated Bridge.gpV!(V, P, v)) == 0
@test norm(K.yy[1]*Bridge.H(t, T, P) - I) < 10/n2^3
@test norm(V.yy[1] - Bridge.V(t, T, v, P)) < 10/n2^3


@test norm(Mu.yy[end] - Bridge.mu(t, u, T, P)) < 10/n2^3
@test norm(Mu2.yy[end] - Bridge.mu(t, u, T, P)) < 10/n2^3

@test norm(Bridge.K(t, T, P) - solve(Bridge.R3(), Bridge._dK, tt, zero(M), P)) < 10/n2^3

# Normal(mu, lambda) is the stationary distribution. check by starting in stationary distribution and evolve 20 time units
X = Bridge.mat(S[solve(EulerMaruyama(), mu + cholupper(P.lambda)*randn(S), sample(tt, Wiener{S}()),P).yy[end] - mu for i in 1:m])

@test supnorm(cov(X, dims=2) - Matrix(P.lambda)) < .1


n = 1000
TT = 0.5
tt = 0.0:TT/n:TT

X = solve(EulerMaruyama(), mu, sample(tt, Wiener{S}()),P)
@test supnorm(quvar(X)-TT*a) < 0.06


theta = 0.7
@test_skip  begin X = Bridge.mat(S[thetamethod(mu + chol(P.lambda)*randn(S), sample(tt, Wiener{S}()), P, theta).yy[end] - mu for i in 1:m])
  supnorm(cov(X,2) - Matrix(P.lambda)) < 1.0
end


Pt = Bridge.Ptilde(Bridge.CSpline(tt[1], tt[end], 1.0, 0.0, 0.0, 1.0), sigma)

@test Bridge.gamma(t, v, Pt) ≈ inv(sigma*sigma')
