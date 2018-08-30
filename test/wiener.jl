using Bridge, StaticArrays
using Test, Statistics, LinearAlgebra, Random
SV2 = SVector{2,Float64}

@test let # functions callable
    tt = range(0, stop=1, length=100)
   
    sample(tt, Wiener{SV2}())
    sample!(SamplePath{SV2}(collect(tt), zeros(SV2, 100)), Wiener{SV2}())
    sample(tt, WienerBridge{SV2}(1.0, SVector(2.0, -2.0)))
    sample!(SamplePath{SV2}(collect(tt), zeros(SV2, 100)), WienerBridge{SV2}(1.0, SVector(2.0, -2.0)))
    solve(Euler(), SVector(1.1, 1.0), sample(tt, Wiener{SV2}()), Wiener{SV2}())

    sample(tt, Wiener())
    sample(tt, WienerBridge(1.0, 1.0))
    true
end

# tests with alpha 0.01
# test fail 1% of the time


r = 2.576
n = 1000


# varmu(x, mu) = dot(x-mu, x-mu)/length(x)
# var0(x) = dot(x, x)/length(x)



#testing mean and variance of Brownian motion sampled at few points
X = [sample(range(0.0, stop=2.0, length=5), Wiener()).yy[end] for i in 1:n]

@test abs(mean(X)) < r * sqrt(2/n)
# see above, fails 1% of the time
# if you want to check: 99% ≈ mean([abs(mean([sample(linspace(0.0, 2.0, 5), Wiener()).yy[end] for i in 1:n]))< r*sqrt(2/n) for k in 1:10000]) 


chiupper = 1118.95 #upper 0.005 percentile for n = 1000
chilower = 888.56 #lower 0.005 percentile for n = 1000
@test chiupper > 1000.0 * var(X)/2 > chilower

# mean and covariance of vector brownian motion

@test norm(SVector(5.0,2.0) - mean([sample(range(0.0, stop=2.0, length=5),Wiener{SV2}(),SVector(5.0,2.0)).yy[end] for i = 1:div(n,2)])) < r *sqrt(2*2.82 / n)
@test norm(2I - cov(Bridge.mat([sample(range(0.0, stop=2.0, length=5),Wiener{SV2}(),SVector(5.0,2.0)).yy[end] for i = 1:1000]), dims=2, corrected=true))  < 0.3

@test Bridge.a(0.0, 0.0, Wiener()) == inv(Bridge.Γ(0.0, 0.0, Wiener())) == Bridge.σ(0.0, 0.0, Wiener())
@test Bridge.a(0.0, 0.0, WienerBridge(1.0, 0.0)) == inv(Bridge.Γ(0.0, 0.0, WienerBridge(1.0, 0.0))) == 
    Bridge.σ(0.0, 0.0, WienerBridge(1.0, 0.0))


@test mean(transitionprob(0.0, 0.0, 2.0, Wiener())) == 0
@test var(transitionprob(0.0, 0.0, 2.0, Wiener())) ≈ 2.0