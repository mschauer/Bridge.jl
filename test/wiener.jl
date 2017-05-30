using Bridge, StaticArrays
using Base.Test
srand(8)


@test begin # functions callable
    sample(linspace(0,1,100),  Wiener{SVector{2,Float64}}())
    sample!(SamplePath{SVector{2,Float64}}(collect(linspace(0,1,100)), zeros(SVector{2,Float64},100)), Wiener{SVector{2,Float64}}())
    sample(linspace(0,1,100),  WienerBridge{SVector{2,Float64}}(1., SVector(2.,-2.)))
    sample!(SamplePath{SVector{2,Float64}}(collect(linspace(0,1,100)), zeros(SVector{2,Float64},100)),  WienerBridge{SVector{2,Float64}}(1., SVector(2.,-2.)))
    euler!(sample(linspace(0,1,100),  Wiener{SVector{2,Float64}}()), SVector(1.1,1.), sample(linspace(0,1,100),  Wiener{SVector{2,Float64}}()), Wiener{SVector{2,Float64}}())

    sample(linspace(0,1,100),  Wiener{Float64}())
    sample(linspace(0,1,100),  WienerBridge{Float64}(1., 1.))
    true
    end

# tests with alpha 0.01
# test fail 1% of the time


r = 2.576
n = 1000


# varmu(x, mu) = dot(x-mu, x-mu)/length(x)
# var0(x) = dot(x, x)/length(x)



#testing mean and variance of Brownian motion sampled at few points
X = [sample(linspace(0., 2., 5), Wiener{Float64}()).yy[end] for i in 1:n]

@test abs(mean(X)) < r*sqrt(2/n)
# see above, fails 1% of the time
# if you want to check: 99% ≈ mean([abs(mean([sample(linspace(0., 2., 5), Wiener{Float64}()).yy[end] for i in 1:n]))< r*sqrt(2/n) for k in 1:10000]) 


chiupper = 1118.95 #upper 0.005 percentile for n = 1000
chilower = 888.56 #lower 0.005 percentile for n = 1000
@test chiupper >1000.*var(X)/2 > chilower

# mean and covariance of vector brownian motion

@test norm(SVector(5.,2.) - mean([sample(linspace(0.0,2.0,5),Wiener{SVector{2,Float64}}(),SVector(5.,2.)).yy[end] for i = 1:div(n,2)])) < r *sqrt(2*2.82 / n)
@test norm(2I - Base.cov(Bridge.mat([sample(linspace(0.0,2.0,5),Wiener{SVector{2,Float64}}(),SVector(5.0,2.0)).yy[end] for i = 1:1000]), 2, true))  < 0.3
