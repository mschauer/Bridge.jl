using Bridge, Distributions
using Base.Test
#import Bridge: b, σ, a, transitionprob
n = 1000
TT = 5
tt = 0.:TT/n:TT
m = 1000
S = SVector{2,Float64}
B = SMatrix{2,2,Float64}([-1 0.1; -0.2 -1])
mu = 0*S([0.2, 0.3])
sigma = SMatrix{2,2,Float64}(2*[-0.212887  0.0687025
  0.193157  0.388997 ])
a = sigma*sigma'
  
P = LinPro{S}(B, mu, sigma)

u = S([1., 0.])
v = S([.5, 0.])

@test (norm(Matrix(-P.lambda*B' - B*P.lambda - a))) < eps()

# Normal(mu, lambda) is the stationary distribution. check by starting in stationary distribution and evolve 20 time units
X = Bridge.mat(S[euler(mu + chol(P.lambda)*randn(S), sample(tt, Wiener{S}()),P).yy[end] - mu for i in 1:m])

@test supnorm(cov(X,2) - Matrix(P.lambda)) < .1
println("Fixme: supnorm(cov(X,2) - Matrix(P.lambda)) < .1")

n = 1000
TT = 0.5
tt = 0.:TT/n:TT

X = euler(mu, sample(tt, Wiener{S}()),P)
@test supnorm(quvar(X)-TT*a) < 0.06


theta = 0.7
X = Bridge.mat(S[thetamethod(mu + chol(P.lambda)*randn(S), sample(tt, Wiener{S}()), P, theta).yy[end] - mu for i in 1:m])

@test supnorm(cov(X,2) - Matrix(P.lambda)) < 1.0
