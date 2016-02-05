using Bridge, FixedSizeArrays, Distributions
using Base.Test
#import Bridge: b, σ, a, transitionprob
qu(x) = x*x'
S = Vec{2,Float64}
B = Mat{2,2,Float64}([-1 0.1; -0.2 -1])
mu = S([0.2, 0.3])
sigma = Mat{2,2,Float64}([-0.212887  0.0687025
  0.193157  0.388997 ])
a = qu(sigma)
@test supnorm(qu(chol(a)') - a) < eps()

@test norm(cov(Bridge.mat(S[ chol(a)'*randn(S) for i in 1:100])') - Matrix(a)) < 0.1