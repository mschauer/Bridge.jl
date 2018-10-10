using Bridge
using Test, Statistics, LinearAlgebra, Random

# tests with alpha 0.01
# test fail 1% of the time
r = 2.576
n = 1000
chiupper = 1118.95 #upper 0.005 percentile for n = 1000
chilower = 888.56 #lower 0.005 percentile for n = 1000

# varmu(x, mu) = dot(x-mu, x-mu)/length(x)
var0(x) = dot(x, x)/length(x)

brown1(s, t, n) = sample(range(s, stop=t, length=n), Wiener())
bb(u,v,t,n) = sample(range(0, stop=t, length=n), WienerBridge{Float64}(t,v), u)
X = SamplePath([0.0, 1.0, 3.0], zeros(3))
@test abs(mean(Bridge.sample!(Bridge.TransitionProb(), X,  Wiener(), 0.0).yy[end] for i in 1:n) -  
    mean(Bridge.transitionprob(0.0, 0.0, 3.0, Wiener()))) <
    3*std(Bridge.transitionprob(0.0, 0.0, 3.0, Wiener()))/sqrt(n)


X = sample(range(0.0, stop=2.0, length=1000), Wiener())
@test bracket(X)[end] == bracket(X,X)[end] == quvar(X)

#quadratic variation of Brownian motion is proportional to time plus sampling bias
quv = [quvar(sample(range(0.0, stop=2.0, length=1000), Wiener())) for j in 1:1000]
s2 = var(quv)
@test abs(mean(quv)-2)/sqrt(s2)*sqrt(1000) < 3


# int0^T w_t dw_t = w_T^2/2 - T/2
@test abs((b -> (ito(b, b).yy[end] - (0.5b.yy[end]^2 - 1)))(brown1(0, 2, 10000))) < 0.1

struct Diff <: Bridge.ContinuousTimeProcess{Float64}
end
import Bridge: b, σ
Bridge.b(t,x, ::Diff) = -5x
Bridge.σ(t, x, ::Diff) = 1.


X = [solve(EulerMaruyama(), 0.0, brown1(0.0,1.0,1000), WienerBridge(2.0, 1.0)).yy[end] for i in 1:1000]
@test abs(mean(X)-0.5) < r*sqrt(2/n)


#test (roughly) the quadratic variance of Euler approximation
quv = [quvar(solve(EulerMaruyama(), 0.0, brown1(0.0,1.0,1000), Diff())) for j in 1: 1000]
s2 = var(quv)
@test abs(mean(quv)-1)/sqrt(s2)*sqrt(1000) < 5.


#  B(t) = (1-t) W(t/(1-t)).

@test abs(mean([bb(0,1,1,3).yy[2] for i in 1:1000])-0.5)/sqrt(0.25)*sqrt(1000) < r
@test abs(mean([bb(0,1,1,5).yy[3] for i in 1:1000])-0.5)/sqrt(0.25)*sqrt(1000) < r
@test chilower < 1000*abs(var0([bb(0,1,1,3).yy[2] for i in 1:1000] .- 0.5))/.25 < chiupper
@test chilower < 1000*abs(var0([bb(0,1,1,5).yy[3] for i in 1:1000] .- 0.5))/.25 < chiupper


# Covariance of a Brownian bridge from t_1 to t_2 (t_2-t)(s-t_1)/(t_2-t_1)
# here (2-1)(1)/2 = 1/2

@test chilower < 1000*abs(var0([bb(0,1,2,5).yy[3] for i in 1:1000] .- 0.5))*2 < chiupper
