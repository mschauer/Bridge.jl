using Bridge, Distributions, StaticArrays
using Base.Test
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
n2 = 150
tt = linspace(t, T, n2)
K = SamplePath(tt, zeros(M, length(tt)))
Bridge.gpK!(K, P)
V = SamplePath(tt, zeros(S, length(tt)))
Bridge.gpK!(K, P) # warm up
Bridge.gpV!(V, v, P)
Mu = SamplePath(tt, zeros(S, length(tt)))
Mu2 = SamplePath(tt, zeros(S, length(tt)))

solve!(Bridge.R3(), Bridge._F, Mu, u, P)
solve!(BS3(), Bridge._F, Mu2, u, P)
@test (@allocated Bridge.gpK!(K, P)) == 0
@test (@allocated Bridge.gpV!(V, v, P)) == 0
@test norm(K.yy[1]*Bridge.H(t, T, P) - I) < 10/n2^3
@test norm(V.yy[1] - Bridge.V(t, T, v, P)) < 10/n2^3

@test norm(Mu.yy[end] - Bridge.mu(t, u, T, P)) < 10/n2^3
@test norm(Mu2.yy[end] - Bridge.mu(t, u, T, P)) < 10/n2^3


# Normal(mu, lambda) is the stationary distribution. check by starting in stationary distribution and evolve 20 time units
X = Bridge.mat(S[solve(EulerMaruyama(), mu + chol(P.lambda)*randn(S), sample(tt, Wiener{S}()),P).yy[end] - mu for i in 1:m])

@test supnorm(cov(X,2) - Matrix(P.lambda)) < .1


n = 1000
TT = 0.5
tt = 0.0:TT/n:TT

X = solve(EulerMaruyama(), mu, sample(tt, Wiener{S}()),P)
@test supnorm(quvar(X)-TT*a) < 0.06


theta = 0.7
@test_skip  begin X = Bridge.mat(S[thetamethod(mu + chol(P.lambda)*randn(S), sample(tt, Wiener{S}()), P, theta).yy[end] - mu for i in 1:m])
  supnorm(cov(X,2) - Matrix(P.lambda)) < 1.0
end
