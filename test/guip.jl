using Bridge, FixedSizeArrays, Distributions
using Base.Test
#import Bridge: b, σ, a, transitionprob

# Define a diffusion process
if !isdefined(:OrnsteinUhlenbeck)
immutable OrnsteinUhlenbeck  <: ContinuousTimeProcess{Float64}
    β::Float64 # drift parameter (also known as inverse relaxation time)
    σ::Float64 # diffusion parameter
    function OrnsteinUhlenbeck(β::Float64, σ::Float64)
        isnan(β) || β > 0. || error("Parameter λ must be positive.")
        isnan(σ) || σ > 0. || error("Parameter σ must be positive.")
        new(β, σ)
    end
end
end

# define drift and sigma of OrnsteinUhlenbeck

Bridge.b(t,x, P::OrnsteinUhlenbeck) = -P.β*x
Bridge.σ(t, x, P::OrnsteinUhlenbeck) = P.σ
Bridge.a(t, x, P::OrnsteinUhlenbeck) = P.σ^2
Bridge.Γ(t, x, P::OrnsteinUhlenbeck) = inv(P.σ^2)
Bridge.transitionprob(s, x, t, P::OrnsteinUhlenbeck) = Normal(x*exp(-P.β*(t-s)), sqrt((0.5P.σ^2/P.β) *(1-exp(-2*P.β*(t-s)))))

if !isdefined(:VOrnsteinUhlenbeck)
immutable VOrnsteinUhlenbeck{d}  <: ContinuousTimeProcess{Vec{d,Float64}}
    β 
    σ 
    function VOrnsteinUhlenbeck(β, σ)
           new(β, σ)
    end
end
end

# define drift and sigma of VOrnsteinUhlenbeck
 
Bridge.b(t, x, P::VOrnsteinUhlenbeck) = -P.β*x
Bridge.σ(t, x, P::VOrnsteinUhlenbeck) = P.σ*I
Bridge.a(t, x, P::VOrnsteinUhlenbeck) = P.σ*P.σ'*I
kernel(x, a=0.001) = 1/sqrt(2pi*a)* exp(-abs2(x)/(2a))
@vectorize_1arg Float64 kernel

n = 700
tt = 0.:1/n:1.
m = 6000
P = VOrnsteinUhlenbeck{2}(2., 1.)
P1 = OrnsteinUhlenbeck(2., 1.)

u = Vec(0., 0.)
v = Vec(.5, 0.)
L = Mat(((1.,),(0.,)))
S = Mat(((1.,),))

convolution(N1::Normal,N2::Normal) = Normal(mean(N1)+mean(N2), sqrt(var(N1)+var(N2)))
EXgivenXpY(X, Y, z) = (z) * var(X) / (var(X) + var(Y))

X = [euler(u, sample(tt, Wiener{Vec{2,Float64}}()),P).yy[end][1] for i in 1:m]
Xstat = mean(X),var(X)
p1(s, x, t, P::VOrnsteinUhlenbeck) = Normal(x*exp(-P.β*(t-s)), sqrt((0.5P.σ^2/P.β) *(1-exp(-2*P.β*(t-s)))))
PX1 = p1(tt[1],u[1], tt[end], P)
Peta = Normal(0,1.)
Pobs = convolution(PX1, Peta)
p = pdf(PX1,v[1])
pt= pdf(Normal(0., 1.),v[1])
p2 = pdf(Pobs,v[1])
pt2 = pdf(Normal(0., sqrt(2.)),v[1])

X = euler(0., sample(tt, Wiener{Float64}()),P1)
@test abs(girsanov(X, P1, Wiener{Float64}()) - llikelihood(X, P1) + llikelihood(X,Wiener{Float64}())) < 0.5


y = EXgivenXpY(PX1, Peta, v[1])

Po = FilterProp(P, tt[1], u, tt[end], v, L, S, Bridge.a(tt[end],v,P))
Po3 = PBridgeProp(P, tt[1], u, (tt[end]-tt[1])/2, 1.2v,tt[end], v, L, S, Bridge.a(tt[end],v,P))



Y = Float64[
begin
 X = euler(Vec(0., 0.), sample(tt, Wiener{Vec{2,Float64}}()),Po)
 
 X.yy[end][1]*exp(llikelihood(X, Po))*pt2/p2
 end
 
 for i in 1:m]
Z = Float64[
begin
 X = euler(Vec(0., 0.), sample(tt, Wiener{Vec{2,Float64}}()),Po)
 exp(llikelihood(X, Po))*pt2/p2
 end
 for i in 1:m]
 
@test abs(mean(Y-y)*sqrt(m)/std(Y)) < 1.96
@test abs(mean(Z-1)*sqrt(m)/std(Z)) < 1.96


# BridgeProp

T = 2.
tt = linspace(0, T, n)
u = 1.
v = 0.5
a = .7
P1 = OrnsteinUhlenbeck(0.8, sqrt(a))
cs = 1*[u, u*exp(-P1.β*T), -P1.β*u, -exp(-P1.β*T)*P1.β*u]
cs2 = Bridge.CSpline(tt[1], tt[end], cs...)
Po = BridgeProp(P1, tt[1], u, tt[end], v, a, cs2)
Z = Float64[
    begin
    X = euler(u, sample(tt, Wiener{Float64}()),Po)
    exp(llikelihood(X, Po)) 
    end
    for i in 1:m]

p = pdf(transitionprob(0., u, T, P1), v)

Pt = Bridge.Ptilde(cs2, sqrt(a))
pt = exp(lp(0., u, T, v, Pt))
@test_approx_eq pt exp(lptilde(Po))
@test abs(mean(Z*pt/p-1)*sqrt(m)/std(Z*pt/p)) < 1.96

# DHBridgeProp

Po3 = DHBridgeProp(P1, tt[1], u, tt[end], v)
Z = Float64[
    begin
    X = euler(u, sample(tt, Wiener{Float64}()),Po3)
    exp(llikelihood(X, Po3)) 
    end
    for i in 1:m]
pt = exp(lptilde(Po3))
@test_approx_eq lptilde(Po3) Bridge.logpdfnormal(v-u, T*a)
@test abs(mean(Z*pt/p-1)*sqrt(m)/std(Z*pt/p)) < 1.96


# PBridgeProp
tm, vm = 0.5, 0.7
si = 1.
L = 1.
Po2 = PBridgeProp(P1, tt[1], u, tm, vm, tt[end], v, L, si^2, a, cs2)
Z2 = Float64[
    begin
    X = euler(u, sample(tt, Wiener{Float64}()),Po2)
    exp(llikelihood(X, Po2)) 
    end
    for i in 1:m]

f(x) = pdf(transitionprob(0., u, tm, P1), x)*pdf(transitionprob(tm,x,T, P1), v)*kernel(x-vm,si^2)
ft(x) = exp(Bridge.lp(0., u, tm, x, Pt) + Bridge.lp(tm,x,T, v, Pt))*kernel(x-vm,si^2)
p2 = sum(map(f,linspace(-20,20,1001)))*40/1000
pt2 = exp(Bridge.lptilde(Po2))
@test_approx_eq pt2 sum(map(ft,linspace(-20,20,1001)))*40/1000
@test abs(mean(Z2*pt2/p2-1)*sqrt(m)/std(Z2*pt2/p2)) < 1.96

