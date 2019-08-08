using Test
using Bridge

VERBOSE = false

using Bridge: Bessel3Bridge, BesselProp, aeuler

T = 1.0
N = 10^3
tt = 0:1/N:(T-1/N)
K = 2000
h = 0.08


ahat(X) = var(diff(X.yy)./sqrt.(diff(X.tt)))



"""
    hit{T}(u, v, dt, tmax, P::ContinuousTimeProcess{T})

Hitting time samples
"""
function hit(u, v, dt, tmax, P::ContinuousTimeProcess{T}) where T

    t = 0.
    rdt = sqrt(dt)
  
    y::T = u
    
    
    while sign(v-y) == sign(v-u) && t < tmax
        t += dt
        y += Bridge.b(t, y, P)*dt + Bridge.σ(t, y, P)*rdt*randn(T)
    end
    t
end

struct Target <: ContinuousTimeProcess{Float64}
    mu
end

 

Bridge.b(t, x, P::Target) = -P.mu*x
Bridge.σ(t, x, P::Target) = sqrt(2.)
Bridge.a(t, x, P::Target) = 2.
Bridge.Γ(t, x, P::Target) = 0.5
 
Bridge.constdiff(P::Target) = true

x0 = 0.5

Pt = Target(0.)
Pto =  Bessel3Bridge(1., 0, sqrt(2))
P = Target(1.)
Po = Bridge.BesselProp(P, 1., 0.)

VERBOSE && println("pt")


tau0 = [hit(x0, 0, 1/N, 10, Target(0.)) for i in 1:10K]
pt_ = mean(1 .< tau0 .< 1 + h)/h

VERBOSE && println("p")

tau1 = [hit(x0, 0, 1/N, Inf, Target(1.)) for i in 1:10K]
pr = mean(1 .< tau1 .< 1. + h)
p=pr/h
pse = sqrt(pr*(1-pr)/length(tau1))/h

Xo = aeuler(x0, 0:1/100:0.99, Po, 0.5)
B3 = sample(x0, tt, Pto)


VERBOSE && println("ll (B3)")
ll = zeros(K)
ll2 = zeros(K)
for k in 1:K
    global B3 = sample(x0, tt, Pto)
    ll[k] = girsanov(B3, P, Pt)
    ll2[k] = girsanov(B3, Po, Pto) +  llikelihood(B3, Po)
end

VERBOSE && println("ll (Xo)")
ll3 = zeros(K)
for k in 1:K
    k % 100 == 0 && VERBOSE && print(" $k ")
    global Xo = aeuler(x0, tt, Po, 0.5)
    ll3[k] = llikelihood(Xo, Po)
end
VERBOSE && println(".")


pt = exp(lptilde(0., x0, Po))


phat = pt*mean(exp.(ll))
phat3 = pt*mean(exp.(ll3))




VERBOSE && println("X")

X = solve(Euler(), x0, Bridge.sample(tt,Bridge.Wiener{Float64}()), P)
while any(X.yy .< 0) || X.yy[end] > 0.1
    global X = solve(Euler(), x0, Bridge.sample(tt,Bridge.Wiener{Float64}()), P)
end    
VERBOSE && println("Xo")



@test abs(p - 0.1788) < 0.015
@test abs(pt - pt_) < 0.015
@test abs(phat - 0.1788)<0.015
@test abs(phat3 - 0.1788)<0.015
@test abs(ahat(B3) - 2) < 0.175
@test abs(ahat(Xo) - 2) < 0.15;
