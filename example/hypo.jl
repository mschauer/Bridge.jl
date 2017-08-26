using Bridge, StaticArrays, Distributions, PyPlot
using Base.Test
import Base.Math.gamma
#import Bridge: b, σ, a, transitionprob
const percentile = 3.0
SV = SVector{2,Float64}
SM = SMatrix{2,2,Float64,4}
kernel(x, a=0.001) = exp(Bridge.logpdfnormal(x, a*I))


@inline _traceB(t, K, P) = trace(Bridge.B(t, P))

traceB(tt, u::T, P) where {T} = solve(Bridge.R3(), _traceB, tt, u, P)


runmean(x) = [mean(x[1:n]) for n in 1:length(x)]

using Bridge.outer

# Define a diffusion process
if ! @_isdefined(Target)
struct Target  <: ContinuousTimeProcess{SV}
    
end
end

if ! @_isdefined(Linear)
struct Linear  <: ContinuousTimeProcess{SV}
    T::Float64
    v::SV
end
end

g(t, x) = sin(x)
gamma(t, x) = 1.2 #+ atan(x)/8
c() = 0.1
const κ = 3.0

# define drift and sigma of Target

Bridge.b(t,x, P::Target) = SV(κ*x[2] - c()*x[1],  -c()*x[2] + g(t, x[2]))
Bridge.σ(t, x, P::Target) = SV(0.0, gamma(t, x[2]))
Bridge.a(t, x, P::Target) = SM(0, 0, 0, outer(gamma(t, x[2])))
Bridge.constdiff(::Target) = false


# define drift and sigma of Linear approximation
 
Bridge.b(t, x, P::Linear) = SV(κ*x[2]-c()*x[1], -c()*x[2] + g(P.T, P.v[2]))
Bridge.B(t, P::Linear) = SM(-c(), 0.0, κ, -c())

Bridge.β(t, P::Linear) = SV(0, g(P.T, P.v[2]))

Bridge.σ(t, x, P::Linear) = SV(0.0, gamma(P.T, P.v[2]))
Bridge.a(t, x, P::Linear) = SM(0, 0, 0, outer(gamma(P.T, P.v[2])))
Bridge.a(t, P::Linear) = SM(0, 0, 0, outer(gamma(P.T, P.v[2])))
Bridge.constdiff(::Linear) = false

t = 1.0
T = 1.5
n = 1000
dt = 1/n
tt = t:1/n:T
m = 30000

u = @SVector [0.1, 0.1]
v = @SVector [0.3, -0.6]


P = Target()
Pt = Linear(T, v)

B = Bridge.B(0, Pt)
β = Bridge.β(0, Pt)
a = Bridge.a(0, Pt)
σ = sqrtm(Bridge.a(0, Pt))

Phi = expm(B*(T-t))
Λ = lyap(B, a)

Pt2 = LinPro(B, -B\β, σ)
lpt2 = lp(t, u, T, v, Pt2)


GP = Bridge.GuidedBridge(tt, (u,v), P, Pt)
lpt = Bridge.logpdfnormal(v - Bridge.gpmu(tt, u, Pt), Bridge.gpK(tt, zero(SM), Pt))

@test norm(Bridge.a(0,0, Pt) - Bridge.a(0,0, Pt2)) < eps()



@test norm(Bridge.K(t, T, Pt2) - Bridge.gpK(tt, zero(SM), Pt)) < 1e-7
# norm(Bridge.gpmu(tt, u, Pt) - Bridge.mu(t, u, T,  Pt2))

@test norm(Phi*u + sum(expm(B*(T-t))*β*dt for t in tt) - Bridge.gpmu(tt, u, Pt)) < 5e-2

@test norm(Bridge.gpmu(tt, u, Pt) - ( Phi*u + (Phi-I)*(B\β))) < 1e-7

@test norm(Bridge.mu(t, u, T,  Pt2) - ( Phi*u + (Phi-I)*(B\β))) < 1e-7


W = sample(tt, Wiener{Float64}())

# Xtilde

Yt = SV[]
Xt = SamplePath(tt, zeros(SV, length(tt)))
Xts = SamplePath(tt, zeros(SV, length(tt)))
best = Inf
for i in 1:m
    W = sample!(W, Wiener{Float64}())
    Bridge.solve!(Euler(), Xt, u, W, Pt)
    push!(Yt, Xt.yy[end])
    n = norm(v-Xt.yy[end])
    
    if n < best
        best = n
        Xts.yy .= Xt.yy
    end
end

# Target

Y = SV[]
X = SamplePath(tt, zeros(SV, length(tt)))
Xs = SamplePath(tt, zeros(SV, length(tt)))
best = Inf
for i in 1:m
    W = sample!(W, Wiener{Float64}())
    Bridge.solve!(Euler(), X, u, W, P)
    push!(Y, X.yy[end])
    nrm = norm(v-X.yy[end])
    
    if n < best
        best = nrm
        Xs.yy .= X.yy
    end
end

# Proposal
lpthat = log(mean(kernel.(collect(y - v for y in Yt))))

Z = Float64[]
Xo = SamplePath(tt, zeros(SV, length(tt)))
@time for i in 1:m
    W = sample!(W, Wiener{Float64}())
    Bridge.bridge!(Bridge.Euler(), Xo, W, GP)
    z = llikelihood(LeftRule(), Xo, GP) + lpt
    push!(Z, z)
end


# compare

GP2 = GuidedProp(P, tt[1], u, tt[end], v, Pt2)

Bridge.H(t, x, P::GuidedProp) = Bridge.H(t, P.t1, P.Pt)

Z2 = Float64[]
Xo2 = SamplePath(tt, zeros(SV, length(tt)))
@time for i in 1:m
    W = sample!(W, Wiener{Float64}())
    Bridge.bridge!(Xo2, W, GP2)
    z = llikelihood(Xo2, GP2) + lpt
    push!(Z2, z)
end

@test norm(Bridge.bridge!(Xo2, W, GP2).yy  - Bridge.bridge!(Bridge.Euler(), Xo, W, GP).yy) < 1e-4
@test norm(llikelihood(Xo2, GP2) - llikelihood(LeftRule(), Xo, GP)) < 1e-3

# Some tests

@test norm(lpt2 - lpt) < sqrt(eps())

@test norm(mean(Yt)[2] - Bridge.mu(t, u, T, Pt2)[2]) < 1.9*std(last.(Yt))/sqrt(m)


Ytv = collect(y - v for y in Yt)
Yv = collect(y - v for y in Y)


@show mean(kernel.(Ytv))
@show exp(lpt)
@show std(kernel.(Ytv))
 

@show mean(kernel.(Yv))
@show std(kernel.(Yv))
 
@show mean(exp.(Z))
@show std(exp.(Z))



# Target

figure()
subplot(211)
X = SamplePath(tt, zeros(SV, length(tt)))
for i in 1:10
    W = sample!(W, Wiener{Float64}())
    Bridge.solve!(Euler(), X, u, W, P)
    display(plot(X.tt, X.yy))
end    
subplot(212)
plot(first.(Y[1:1000]), last.(Y[1:1000]), ".")

figure()
subplot(411)
plot(Xs.tt, Xs.yy, label="X*")
legend()

subplot(412)
plot(Xts.tt, Xts.yy, label="Xt*")
legend()

subplot(413)
plot(Xo.tt, Xo.yy, label="Xo")
legend()

subplot(414)
plot(runmean(exp.(Z)), label="Xo")
plot(runmean(kernel.(Yv)), label="X")
plot(runmean(kernel.(Ytv)), label="Xt")
legend()
axis([1, m, 0, 2*exp(lpthat)])



    
