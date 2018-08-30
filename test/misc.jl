using Bridge, Distributions
using Test
#import Bridge: b, σ, a, transitionprob
qu(x) = x*x'
S = SVector{2,Float64}
B = SMatrix{2,2,Float64}([-1 0.1; -0.2 -1])
mu = S([0.2, 0.3])
sigma = SMatrix{2,2,Float64}([-0.212887  0.0687025
  0.193157  0.388997 ])
a = qu(sigma)
@test supnorm(qu(cholupper(a)') - a) < eps()

@test norm(cov(Bridge.mat(S[ cholupper(a)'*randn(S) for i in 1:100])') - Matrix(a)) < 0.1

x = [0.289639+0.649503im, 0.724532+0.963363im]

@test dot(x,x) == Bridge.inner(x)
@test dot(x, 1 .- x) == Bridge.inner(x, 1 .- x)
@test x*x' == Bridge.outer(x)
@test x*(1 .- x)' == Bridge.outer(x, 1 .- x)


A = reshape(collect(1:5*4*3), 5,4,3)
@test A[.., 1] == reshape(1:5*4, 5,4)
A[.., 1] = A[.., 3]
@test A[.., 1] == reshape((3-1) * 5 * 4 .+ (1:5*4), 5, 4)
