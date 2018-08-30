using Bridge, StaticArrays, Distributions, PyPlot
using Test
import Base.Math.gamma # just to use the name
#import Bridge: b, σ, a, transitionprob
using Bridge: runmean, sqmahal, Gaussian

const percentile = 3.0
const SV = SVector{2,Float64}
const SM = SMatrix{2,2,Float64,4}

kernel(x, a=0.0025) = exp(Bridge.logpdfnormal(x, a*I))

refine(tt, n) =  first(tt):(Base.step(tt)/n):last(tt)

TEST = false
CLASSIC = false


using Bridge.outer

# Define a diffusion process
if ! @isdefined(Target)
struct Target  <: ContinuousTimeProcess{SV}
    c::Float64
    κ::Float64
end
end

if ! @isdefined(Linear)
struct Linear  <: ContinuousTimeProcess{SV}
    T::Float64
    v::SV
    b11::Float64
    b21::Float64
    b12::Float64
    b22::Float64
end
end


g(t, x) = sin(x)
gamma(t, x) = 1.2 - sech(x)/2

# define drift and sigma of Target

Bridge.b(t, x, P::Target) = SV(P.κ*x[2] - P.c*x[1],  -P.c*x[2] + g(t, x[2]))::SV
Bridge.σ(t, x, P::Target) = SM(0.5, 0.0, 0.0, gamma(t, x[2]))
Bridge.a(t, x, P::Target) = SM(0.25, 0, 0, outer(gamma(t, x[2])))
Bridge.constdiff(::Target) = false


# define drift and sigma of Linear approximation
 
Bridge.b(t, x, P::Linear) = SV(P.b11*x[1] + P.b12*x[2], P.b21*x[1] + P.b22*x[2] + g(P.T, P.v[2]))
Bridge.B(t, P::Linear) = SM(P.b11, P.b21, P.b12, P.b22)

Bridge.β(t, P::Linear) = SV(0, g(P.T, P.v[2]))

Bridge.σ(t, x, P::Linear) = SM(0.5, 0, 0, gamma(P.T, P.v[2]))
Bridge.a(t, x, P::Linear) = SM(0.25, 0, 0, outer(gamma(P.T, P.v[2])))
Bridge.a(t, P::Linear) = SM(0.25, 0, 0, outer(gamma(P.T, P.v[2])))
Bridge.constdiff(::Linear) = false



c = 0.0
κ = 3.0

# times

t = 0.7
T = 1.5
S = (t + T)/2

# grid

n = 401
dt = (T-t)/(n-1)
tt = t:dt:T
S = tt[n÷2 + 1]
tt1 = t:dt:S
tt2 = S:dt:T

m = 1_800_000

Ti = n
Si = n÷2

# observations

Σ = 0.8 # observation noise
Q = Normal(0, sqrt(Σ)) # noise distribution
L = @SMatrix [1.0 0.0]

xt = @SVector [0.1, 0.0]
vS = @SVector [0.1]
xS = @SVector [0.1, -0.2]
xT = @SVector [0.3, -0.4]

# processes

P = Target(c, κ)
P̃ = Pt = Linear(T, xT, -c-0.1, -0.1, κ-0.1, -c/2)

# parameters

B = Bridge.B(0, Pt)
β = Bridge.β(0, Pt)
a = Bridge.a(0, Pt)
σ = sqrt(Bridge.a(0, Pt))



W = sample(tt, Wiener{SV}())
W1 = sample(tt1, Wiener{SV}())
W2 = sample(tt2, Wiener{SV}())


# Target and log probability density (forward simulation)

YT = SV[]
YS = Float64[]
p = Float64[]
X = SamplePath(tt, zeros(SV, length(tt)))
Xs = SamplePath(tt, zeros(SV, length(tt)))
best = Inf
for i in 1:m
    W = sample!(W, Wiener{SV}())
    Bridge.solve!(Euler(), X, xt, W, P)
    push!(YT, X.yy[end])
    push!(YS, X.yy[end÷2][2]) # depends on L
    
    eta = rand(Q)
    nrm = norm(xT - X.yy[Ti]) + norm(vS - eta - L*X.yy[Si])
    l = kernel(xT - X.yy[Ti]) * kernel(vS - L*X.yy[Si] - eta)
    push!(p, l)
    if nrm < best
        best = nrm
        Xs.yy .= X.yy
    end
end

lphat = log(mean(p))

# Proposal log probability density (forward simulation)

YtT = SV[]
p̃ = pt = Float64[]
Xt = SamplePath(tt, zeros(SV, length(tt)))
l = 0.0
for i in 1:m
    W = sample!(W, Wiener{SV}())
    Bridge.solve!(Euler(), Xt, xt, W, Pt)
    push!(YtT, Xt.yy[end])
    eta = rand(Q)
    l =  kernel(xT - Xt.yy[Ti]) * kernel(vS - L*Xt.yy[Si] - eta) # likelihood
    push!(pt, l)
end
lpthat = log(mean(pt))



@show lpthat




# Proposal

Z = Float64[]
Xo1 = SamplePath(tt1, zeros(SV, length(tt)))
Xo2 = SamplePath(tt2, zeros(SV, length(tt)))

GP2 = GuidedBridge(tt2, P, Pt, xT)
H♢, V = Bridge.gpupdate(GP2, L, Σ, vS)
GP1 = GuidedBridge(tt1, P, Pt, V, H♢)

@time for i in 1:m
    
    sample!(W1, Wiener{SV}())
    sample!(W2, Wiener{SV}())
    y = Bridge.bridge!(Xo1, xt, W1, GP1)
    Bridge.bridge!(Xo2, y, W2, GP2)

    
    ll = llikelihood(LeftRule(), Xo1, GP1) + llikelihood(LeftRule(), Xo2, GP2)
    push!(Z, exp(ll))
end

@show log(mean(exp.(lpthat))), lphat


@show lpthat
#@show lpt
@show lptilde(GP1, xt) - Bridge.traceB(tt2, Pt)


PhiS = Bridge.fundamental_matrix(tt1, Pt)
PhiT = Bridge.fundamental_matrix(tt, Pt)
PhiTS = Bridge.fundamental_matrix(tt2, Pt)

KS = Bridge.gpK(tt1, zero(SM), Pt)
KT = Bridge.gpK(tt, zero(SM), Pt)
KTS = Bridge.gpK(tt2, zero(SM), Pt)

muS = Bridge.gpmu(tt1, xt, Pt)
muT = Bridge.gpmu(tt, xt, Pt)
muTS = Bridge.gpmu(tt2, xS, Pt)

hS = Bridge.gpmu(tt1, 0*xt, Pt)
hT = Bridge.gpmu(tt, 0*xt, Pt)
hTS = Bridge.gpmu(tt2, 0*xS, Pt)


@test norm(PhiTS\(xT - muTS) - (GP2.V[1] - xS)) < 1e-9

#@test norm(muT - mean(YtT)) < 1/sqrt(m)


H♢ = GP1.H♢[1]
H♢S_ = GP1.H♢[end]
H♢S = GP2.H♢[1]
V = GP1.V[1]
VS = GP2.V[1]
VS_ = GP1.V[end]

u = [vS - L*muS; xT - muT]

Upsilon = [
L*KS*L' + Σ     L*KS*PhiTS' 
PhiTS*KS*L'     KT
]

Upsilon2 = [
    L*(PhiS*H♢*PhiS' - GP1.H♢[end])*L' + Σ     L*(PhiS*H♢*PhiT' - GP1.H♢[end]*PhiTS') 
    (PhiT*H♢*PhiS' - PhiTS*GP1.H♢[end])*L'     KT
    ]
    
Upsilon2 = [
    L*KS*L' + Σ     L*KS 
    KS*L'     KS + GP2.H♢[1]
    ]
@test norm(cat((1,2),eye(1),PhiTS)*Upsilon2*cat((1,2),eye(1),PhiTS)' - Upsilon)<1e-8
U = inv(Upsilon)
#0 = logdet(Upsilon2) + 2*logdet(PhiTS) + logdet(U)

A = [L*PhiS; PhiT ]

z = u + A*xt


lpt = Bridge.logpdfnormal(u, Upsilon)

@test norm(logdet(U) + Bridge.logdetU(GP1, GP2, L, Σ)) < 1e-8

@test norm(PhiTS*xS + (hTS - muTS)) < 1e-7

@test norm(PhiTS*(VS-xS) - ( xT-PhiTS*xS - hTS ) ) < 1e-7

@test_broken norm(sqmahal(Gaussian( A*xt, Upsilon), z) - sqmahal(Gaussian(GP1.V[1], GP1.H♢[1]) , xt)) < 1e-8
@show lpt
@show lpthat

@show logdet(Upsilon)
@show logdet(KT) + logdet(Σ) + 2*logdet(PhiTS)
@test -logdet(U) ≈ logdet( L*KS*L' + Σ - L*KS*PhiTS'*inv(KT)*PhiTS*KS*L' ) + 
    logdet(KT)
@test -logdet(U) ≈ logdet( KT - PhiTS*KS*L'*inv(L*KS*L' + Σ)*L*KS*PhiTS' ) + 
    logdet(L*KS*L' + Σ )

@test 1/det(U) ≈  det(L*KS*L' + Σ - 1)*det(KT) +  det(KT -  PhiTS*KS*L'*L*KS*PhiTS')

@test norm(logdet(KTS) - (logdet(GP2.H♢[1]) + 2logdet(PhiTS))) < 1e-8

@test norm(A'*U*A - inv(GP1.H♢[1])) < 1e-7

@test norm(PhiTS\(xT - muTS) - (GP2.V[1] - xS)) < 1e-9
@test norm(inv(GP1.H♢[end]) - inv(GP2.H♢[1]) -  L'*inv(Σ)*L) < 1e-10
@test norm(KS - PhiS*H♢*PhiS' + GP1.H♢[end]) < 1e-8
@test norm(KT - PhiT*H♢*PhiT' - PhiTS*(GP2.H♢[1]- GP1.H♢[end])*PhiTS' ) < 1e-8

@test norm(xt'*inv(H♢)*xt - xt'*A'*U*A*xt) < 1e-7
@test norm(xt'*inv(H♢)*V - xt'*A'*U*z) < 1e-7

@test norm(PhiT*(V-xt) + PhiTS*(VS-VS_)  - ( xT-PhiT*xt - hT ) ) < 1e-7

mh1 = [vS - L*muS;xT - muT]'*U*[vS - L*muS;xT - muT]
mh2 = (V - xt)'*inv(H♢)*(V - xt)

r1, r2 = H♢\(V - xt), A'*U*[vS - L*muS;xT - muT]
@test norm(r1 - r2)  < 1e-7


# Plot best "bridge"

clf()
subplot(121)
plot(Xs.tt, first.(Xs.yy), label="X₁˟")
plot(Xs.tt, last.(Xs.yy), label="X₂˟")
plot.(t, xt, "ro")
plot(S, vS, "ro")
plot.(T, xT, "ro")
legend()

subplot(122)
step = 10
plot(mean(pt)*runmean(Z)[1:step:end], label="Ψ p̃")
plot(runmean(p)[1:step:end], label="p")
plot(runmean(pt)[1:step:end], label="p̃")
plot(fill(exp(lpt),length(1:step:m)), label="p̃ theor.")
legend()
axis([1, div(m,step), 0, 3*exp(lpthat)])


error("done")




figure()
subplot(411)
plot(Xs.tt, Xs.yy, label="X*")
legend()


subplot(413)
plot(Xo.tt, Xo.yy, label="Xo")
legend()

subplot(414)
step = 10
plot(runmean(exp.(Z))[1:step:end], label="Xo")
plot(runmean(kernel.(Yv))[1:step:end], label="X")
plot(runmean(kernel.(Ytv))[1:step:end], label="Xt")
legend()
axis([1, div(m,step), 0, 2*exp(lpt)])



ex = Dict(
"u" => u,    
"v" => v,    
"Xo" => Xo,
"Xs" => Xs,
"Xts" => Xts,
"Yt" => Yt,
"Y" => Y,
"Z" => Z,
"lpt" => lpt
)