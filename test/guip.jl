using Bridge, StaticArrays, Distributions
using Test, LinearAlgebra, Random

const percentile = 3.0

# Define a diffusion process
if ! @isdefined(OrnsteinUhlenbeck)
struct OrnsteinUhlenbeck  <: ContinuousTimeProcess{Float64}
    β::Float64 # drift parameter (also known as inverse relaxation time)
    σ::Float64 # diffusion parameter
    function OrnsteinUhlenbeck(β::Float64, σ::Float64)
        isnan(β) || β > 0.0 || error("Parameter λ must be positive.")
        isnan(σ) || σ > 0.0 || error("Parameter σ must be positive.")
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
Bridge.constdiff(::OrnsteinUhlenbeck) = true

if ! @isdefined(VOrnsteinUhlenbeck)
struct VOrnsteinUhlenbeck{d}  <: ContinuousTimeProcess{SVector{d,Float64}}
    β
    σ
end
end

# define drift and sigma of VOrnsteinUhlenbeck

Bridge.b(t, x, P::VOrnsteinUhlenbeck) = -P.β*x
Bridge.σ(t, x, P::VOrnsteinUhlenbeck) = P.σ*I
Bridge.bderiv(t, x, P::VOrnsteinUhlenbeck) = -P.β*I
Bridge.σderiv(t, x, P::VOrnsteinUhlenbeck) = 0.0*I
Bridge.a(t, x, P::VOrnsteinUhlenbeck) = P.σ*P.σ'*I
Bridge.constdiff(::VOrnsteinUhlenbeck) = true

kernel(x, a=0.001) = 1/sqrt(2pi*a)* exp(-abs2(x)/(2a))

n = 500
tt = 1.0:1/n:2.0
m = 1000
P = VOrnsteinUhlenbeck{2}(2.0, 1.0)
P1 = OrnsteinUhlenbeck(2.0, 1.0)

u = @SVector [0.0, 0.0]
v = @SVector [0.5, 0.0]
L = @SMatrix [1.0 0.0]
S = @SMatrix [1.0]

convolution(N1::Normal,N2::Normal) = Normal(mean(N1)+mean(N2), sqrt(var(N1)+var(N2)))
EXgivenXpY(X, Y, z) = (z) * var(X) / (var(X) + var(Y))

X = [solve(EulerMaruyama(), u, sample(tt, Wiener{SVector{2,Float64}}()),P).yy[end][1] for i in 1:m]
Xstat = mean(X),var(X)
p1(s, x, t, P::VOrnsteinUhlenbeck) = Normal(x*exp(-P.β*(t-s)), sqrt((0.5P.σ^2/P.β) *(1-exp(-2*P.β*(t-s)))))
PX1 = p1(tt[1],u[1], tt[end], P)
Peta = Normal(0,1.0)
Pobs = convolution(PX1, Peta)
p = pdf(PX1,v[1])
pt= pdf(Normal(0.0, 1.0),v[1])
p2 = pdf(Pobs,v[1])
pt2 = pdf(Normal(0.0, sqrt(2.0)),v[1])

X = solve(EulerMaruyama(), 0.0, sample(tt, Wiener{Float64}()),P1)
@test abs(girsanov(X, P1, Wiener{Float64}()) - llikelihood(X, P1) + llikelihood(X,Wiener{Float64}())) < 0.5


y = EXgivenXpY(PX1, Peta, v[1])

Po = FilterProp(P, tt[1], u, tt[end], v, L, S, Bridge.a(tt[end],v,P))
Po3 = PBridgeProp(P, tt[1], u, (tt[end]-tt[1])/2, 1.2v,tt[end], v, L, S, Bridge.a(tt[end],v,P))


Y = Float64[
let
 X = solve(EulerMaruyama(), (@SVector [0.0, 0.0]), sample(tt, Wiener{SVector{2,Float64}}()),Po)
 X.yy[end][1]*exp(llikelihood(X, Po))*pt2/p2
 end

 for i in 1:m]


Z = Float64[
let
 X = solve(EulerMaruyama(), (@SVector [0.0, 0.0]), sample(tt, Wiener{SVector{2,Float64}}()),Po)
 exp(llikelihood(X, Po))*pt2/p2
 end
 for i in 1:m]

@test abs(mean(Y .- y)*sqrt(m)/std(Y)) < percentile
@test abs(mean(Z .- 1)*sqrt(m)/std(Z)) < percentile


##########################################

Cnames = []
C = []


# BridgeProp
push!(Cnames, "BridgeProp")
Random.seed!(5)
n, m = 200, 1000
T = 2.
ss = range(0, stop=T, length=n)
tau(s, T) = s.*(2-s/T)
#tt = tau(ss, T)
tt = ss

u = 0.5
v = 0.1
a = 0.7
P1 = OrnsteinUhlenbeck(0.8, sqrt(a))
#cs = Bridge.CSpline(tt[1], tt[end], u, u*exp(-P1.β*T), -P1.β*u, -exp(-P1.β*T)*P1.β*u)
#cs = Bridge.CSpline(tt[1], tt[end],  -P1.β*u, -exp(-P1.β*T)*P1.β*u)
h = 0.01

cs = Bridge.CSpline(tt[1], tt[end],
    Bridge.b(tt[1], u, P1),
    Bridge.b(tt[end], v, P1),
    (Bridge.b(tt[2], u + Bridge.b(tt[1], u, P1)*(tt[2]-tt[1]), P1)-Bridge.b(tt[1], u, P1))/(tt[2]-tt[1]), # -P1.β*u*(1-exp(-P1.β*dt))/dt
    (Bridge.b(tt[end], v, P1) - Bridge.b(tt[end-1], v - Bridge.b(tt[end], v, P1)*(tt[end]-tt[end-1]), P1))/(tt[end]-tt[end-1])
)

Po = BridgeProp(P1, tt, (u, v), a, cs)
Z = Float64[
    let
    X = solve(EulerMaruyama(), u, sample(tt, Wiener{Float64}()),Po)
    exp(llikelihood(X, Po))
    end
    for i in 1:m]

p = pdf(transitionprob(0.0, u, T, P1), v)

Pt = Bridge.Ptilde(cs, sqrt(a))
pt = exp(lp(0.0, u, T, v, Pt))
@test pt ≈ exp(lptilde(Po))
push!(C, abs(mean(Z*pt/p .- 1)*sqrt(m)/std(Z*pt/p)))

# Scaling for BridgeProp
push!(Cnames, "ScaledBridgeProp")
Z = Float64[
    let
        X = Bridge.ubridge(sample(tt, Wiener{Float64}()), Po)
        exp(llikelihood(X, Po))
    end
    for i in 1:m]

p = pdf(transitionprob(0.0, u, T, P1), v)

Pt = Bridge.ptilde(Po)
pt = exp(lptilde(Po))
push!(C, abs(mean(Z*pt/p .- 1)*sqrt(m)/std(Z*pt/p)))
#error("end")

# GuidedProp
push!(Cnames, "GuidedProp")
β = 0.8
Ptarget = LinPro(-β, 0.0, sqrt(a))
Po = GuidedProp(Ptarget, tt[1], u, tt[end], v, Pt)

z = Float64[
    let
    W = sample(tt, Wiener{Float64}())
    X = Bridge.bridgeold!(copy(W), W, Po)
    llikelihood(X, Po)
    end
    for i in 1:m]

p2 = pdf(transitionprob(0.0, u, T, Ptarget), v)
p = exp(lp(0.0, u, T, v, Ptarget))
pt = exp(lp(0.0, u, T, v, Pt))
@test p == p2
@test pt ≈ exp(lptilde(Po))
push!(C, abs(mean(exp.(z)*pt/p .- 1)*sqrt(m)/std(exp.(z)*pt/p)))


# DHBridgeProp
push!(Cnames, "DHBridgeProp")

Po3 = DHBridgeProp(P1, tt[1], u, tt[end], v)
Z = Float64[
    let
    X = solve(EulerMaruyama(), u, sample(tt, Wiener{Float64}()),Po3)
    exp(llikelihood(X, Po3))
    end
    for i in 1:m]

p = pdf(transitionprob(0.0, u, T, P1), v)
pt = exp(lptilde(Po3))
@test lptilde(Po3) ≈ Bridge.logpdfnormal(v-u, T*a)
push!(C, abs(mean(Z*pt/p .- 1)*sqrt(m)/std(Z*pt/p)))


# PBridgeProp
push!(Cnames, "PBridgeProp")
tm, vm = 0.5, 0.7
si = 1.
L = 1.
Po2 = PBridgeProp(P1, tt[1], u, tm, vm, tt[end], v, L, si^2, a, cs)
Z2 = Float64[
    let
    X = solve(EulerMaruyama(), u, sample(tt, Wiener{Float64}()),Po2)
    exp(llikelihood(X, Po2))
    end
    for i in 1:m]

f(x) = pdf(transitionprob(0.0, u, tm, P1), x)*pdf(transitionprob(tm,x,T, P1), v)*kernel.(x-vm,si^2)
ft(x) = exp(Bridge.lp(0.0, u, tm, x, Pt) + Bridge.lp(tm,x,T, v, Pt))*kernel.(x-vm,si^2)
p2 = sum(map(f,range(-20, stop=20, length=1001)))*40/1000
pt2 = exp(Bridge.lptilde(Po2))
@test pt2 ≈ sum(map(ft,range(-20, stop=20, length=1001)))*40/1000
push!(C, abs(mean(Z2*pt2/p2 .- 1)*sqrt(m)/std(Z2*pt2/p2)))


# GuidedProp
push!(Cnames, "GuidedProp")
Ptarget = Bridge.Ptilde(cs, sqrt(a))
Pt = LinPro(-β, 0.2, sqrt(a))
Po = GuidedProp(Ptarget, tt[1], u, tt[end], v, Pt)

z = Float64[
    let
    W = sample(tt, Wiener{Float64}())
    X = Bridge.bridgeold!(copy(W), W, Po)
    llikelihood(X, Po)
    end
    for i in 1:m]

p2 = pdf(transitionprob(0.0, u, T, Ptarget), v)
p = exp(lp(0.0, u, T, v, Ptarget))
pt = exp(lp(0.0, u, T, v, Pt))
@test p ≈ p2
@test pt ≈ exp(lptilde(Po))
push!(C, abs(mean(exp.(z)*pt/p .- 1)*sqrt(m)/std(exp.(z)*pt/p)))


# GuidedProposal
push!(Cnames, "GuidedProposal")
Ptarget = Bridge.Ptilde(cs, sqrt(a))
Pt = LinPro(-β, 0.2, sqrt(a))
GP = Bridge.GuidedBridge(tt, Ptarget, Pt, v)

@test norm(GP.H♢-[inv(Bridge.H(t, T, Pt)) for t in tt], Inf) < 1e-5
@test norm(GP.V-[Bridge.V(t, T, v, Pt) for t in tt], Inf) < 1e-5


z = Float64[
    let
    W = sample(tt, Wiener{Float64}())
    X = copy(W)
    Bridge.solve!(Euler(), X, u, W, GP)
    llikelihood(LeftRule(), X, GP)
    end
    for i in 1:m]

p2 = pdf(transitionprob(0.0, u, T, Ptarget), v)
p = exp(lp(0.0, u, T, v, Ptarget))
pt = exp(lp(0.0, u, T, v, Pt))
@test p ≈ p2
@test pt ≈ exp(lptilde(Po))
push!(C, abs(mean(exp.(z)*pt/p .- 1)*sqrt(m)/std(exp.(z)*pt/p)))

println(Cnames)
println(C)

@test  all(C .< percentile)
