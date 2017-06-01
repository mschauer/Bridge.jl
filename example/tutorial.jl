using Bridge, Distributions, StaticArrays

PLOT = :winston
include("plot.jl")

#
t = 2. 
n = 100
dt = t / n
x0 = 1.

# define a Float64 Wiener process
P = Wiener{Float64}() #returns object representing standard Brownian motion

typeof(P)
#prints Bridge.Wiener{Float64}

supertype(typeof(P))
#prints Bridge.ContinuousTimeProcess{Float64}

 

# sample Brownian motion on an equidistant grid
W = sample(linspace(0., t, n), P)
W = sample(0:dt:t, P) # similar way

print(W.tt) 
# prints [0.0,0.03,0.06,..., 0.93.96,0.99]
plot(W.tt, W.yy) 
# plots path 



# sample complex Brownian motion on a nonequidistant grid
X = sample(sort(rand(1000)), Wiener{Complex{Float64}}())
plot(real(X.yy), imag(X.yy))

# sample a standard Brownian bridge ending in v at time s
s = 1.
v = 0.

B = sample(0:dt:s, WienerBridge(s, v)) 
plot(W.tt, W.yy, "b")
oplot(B.tt, B.yy, "r")
# displays X and  Brownian bridge B in red


# Define a diffusion process
struct OrnsteinUhlenbeck  <: ContinuousTimeProcess{Float64}
    β::Float64 # drift parameter (also known as inverse relaxation time)
    σ::Float64 # diffusion parameter
    function OrnsteinUhlenbeck(β::Float64, σ::Float64)
        isnan(β) || β > 0. || error("Parameter λ must be positive.")
        isnan(σ) || σ > 0. || error("Parameter σ must be positive.")
        new(β, σ)
    end
end

# define drift and sigma of OrnsteinUhlenbeck
import Bridge: b, σ, a, transitionprob
Bridge.b(t,x, P::OrnsteinUhlenbeck) = -P.β*x
Bridge.σ(t, x, P::OrnsteinUhlenbeck) = P.σ
Bridge.a(t, x, P::OrnsteinUhlenbeck) = P.σ^2

# simulate OrnsteinUhlenbeck using Euler scheme
W = sample(0:0.01:10, Wiener{Float64}()) 
X = euler(0.1, W, OrnsteinUhlenbeck(20., 1.))

plot(X.tt, X.yy)

# define transition density
transitionprob(s, x, t, P::OrnsteinUhlenbeck) = Normal(x*exp(-P.β*(t-s)), sqrt((0.5P.σ^2/P.β) *(1-exp(-2*P.β*(t-s)))))

# plot likelihood of β 
LL = [(β, llikelihood(X, OrnsteinUhlenbeck(β, 1.))) for β in 1.:30.]
for (β, ll) in LL
    println("β $β loglikelihood ", ll )
end
plot(Float64[β for (β, ll) in LL], Float64[ll for (β, ll) in LL])


# sample OrnsteinUhlenbeck exactly. compare with euler scheme which degrates as dt = 0.07
X = euler(0.1, sample(0:0.07:10, Wiener{Float64}()), OrnsteinUhlenbeck(20., 1.))
X2 =  sample(0:0.07:10, OrnsteinUhlenbeck(20., 1.), 0.1)
plot(X.tt, 1+X.yy)
oplot(X2.tt, X2.yy)

# sample vector Brownian motion
W2 = sample(0:0.1:10, Wiener{SVector{4,Float64}}())
llikelihood(W2,Wiener{SVector{4,Float64}}())

P = BridgeProp(OrnsteinUhlenbeck(3., 1.), 0., 0., 1., 1., 1.)
X = euler(0.1, sample(0:0.01:1, Wiener{Float64}()),P)

P2 = PBridgeProp(OrnsteinUhlenbeck(3., 1.), 0., 0., 1., 2., 2., 0., 1., 0.3, 1.)
Y = euler(0.1, sample(0:0.01:2, Wiener{Float64}()),P2)
plot(Y.tt, Y.yy, xrange=(0, 2), yrange=(-3,3))

for i in 1:100
    Y = euler(0.1, sample(0:0.01:2, Wiener{Float64}()),P2)
    oplot(Y.tt, Y.yy, xrange=(0, 2), yrange=(-3,3), linewidth=0.2)
end
oplot()


# Define a diffusion process
struct VOrnsteinUhlenbeck{d}  <: ContinuousTimeProcess{SVector{d,Float64}}
    β # drift parameter (also known as inverse relaxation time)
    σ # diffusion parameter
    function VOrnsteinUhlenbeck(β, σ)
           new(β, σ)
    end
end

# define drift and sigma of VOrnsteinUhlenbeck
 
Bridge.b(t, x, P::VOrnsteinUhlenbeck) = -P.β*x
Bridge.σ(t, x, P::VOrnsteinUhlenbeck) = P.σ*I
Bridge.a(t, x, P::VOrnsteinUhlenbeck) = P.σ*P.σ'*I

# Simulate
X = euler(SVector(0., 0.), sample(0.:0.003:10., Wiener{SVector{2,Float64}}()),VOrnsteinUhlenbeck{2}(3., 1.)) 
yy = Bridge.mat(X.yy)
plot(yy[1,:], yy[2,:], xrange=(-2, 2), yrange=(-2,2),  linewidth=0.5)


