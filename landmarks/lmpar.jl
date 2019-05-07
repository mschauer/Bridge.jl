# reminder, to type H*, do H\^+
#outdir="output/landmarks/"
#cd("/Users/Frank/.julia/dev/Bridge/landmarks")

using Bridge, StaticArrays, Distributions
using Bridge:logpdfnormal
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using RCall
using Base.Iterators
using SparseArrays
using LowRankApprox

models = [:ms, :ahs]
model = models[1]
TEST = false#true
partialobs = true


const d = 2
const itostrat = false

n = 100 # nr of landmarks

θ = 2π/5# π/6 0#π/5  # angle to rotate endpoint

σobs = 10^(-2)   # noise on observations
println(model)

T = 0.75#1.0#0.5
t = 0.0:0.01:T  # time grid

#Random.seed!(5)
include("state.jl")
include("models.jl")
include("patches.jl")
include("lmguiding.jl")

### Specify landmarks models

### Specify landmarks models
a = 3.0 # the larger, the stronger landmarks behave similarly
λ = 0.0; #= not the lambda of noise fields  =# γ = 8.0
db = 6.0 # domainbound
nfstd = 2.0 # tau , widht of noisefields
r1 = -db:nfstd:db
r2 = -db:nfstd:db
nfloc = Point.(collect(product(r1, r2)))[:]
nfscales = [.1Point(1.0, 1.0) for x in nfloc]  # intensity

nfs = [Noisefield(δ, λ, nfstd) for (δ, λ) in zip(nfloc, nfscales)]
Pms = MarslandShardlow(a, γ, λ, n)
Pahs = Landmarks(a, λ, n, nfs)
###

StateW = Point{Float64}
if model == :ms
    dwiener = n
    P = Pms
else
    dwiener = length(nfloc)
    P = Pahs
end

w0 = zeros(StateW, dwiener)
W = SamplePath(t, [copy(w0) for s in t])
sample!(W, Wiener{Vector{StateW}}())

# specify initial landmarks configuration
q0 = [Point(2.0cos(t), sin(t)) for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
p_ = 2*Point(-0.1, 0.1)
p0 = [p_ for i in 1:n]  #
#p0 = [randn(Point) for i in 1:n]
x0 = State(q0, p0)

#Random.seed!(1234)
X = SamplePath(t, [copy(x0) for s in t])
println("Solve for forward provess:")
@time solve!(EulerMaruyama!(), X, x0, W, P)
    #@time solve!(StratonovichHeun!(), X, x0, W, P)

tc(t,T) = t.* (2 .-t/T)
tt_ =  tc(t,T)#tc(t,T)# 0:dtimp:(T)
#tt_ = t

# observe positions WITH noise
v0 = q(X.yy[1]) + σobs * randn(PointF,n)
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), 2*cos(θ))
#vT = [rot * X.yy[end].q[i] for i in 1:P.n ]
trans = SMatrix{2,2}(1.5, 1.0, 0.0, 1.0)
vT = [rot * trans * X.yy[end].q[i] for i in 1:P.n ]



####################
# solve backward recursion on [0,T]
if partialobs==true
    L = [(i==j)*one(Unc) for i in 1:2:2n, j in 1:2n]
    Σ = [(i==j)*σobs^2*one(Unc) for i in 1:n, j in 1:n]
    #Σ = 10 * reshape(rand(Unc,n^2),n,n)
    μend = zeros(Point{Float64},P.n)
    xobs = vT
    mT = zero(vT)#rand(Point, n)#
    Pahsaux = LandmarksAux(Pahs, State(vT, mT))
    Pmsaux = MarslandShardlowAux(Pms, State(vT, mT))
else
    # full observation case
    L = [(i==j)*one(Unc) for i in 1:2n, j in 1:2n]
    Σ = [(i==j)*σobs^2*one(Unc) for i in 1:2n, j in 1:2n]
    μend = zeros(Point{Float64}, 2P.n)
    xobs = vec(X.yy[end])
    Pahsaux = LandmarksAux(Pahs, X.yy[end])
    Pmsaux = MarslandShardlowAux(Pms, X.yy[end])
end

if model == :ms
    Paux = Pmsaux
else
    Paux = Pahsaux
end

# initialise Lt and M⁺t
Lt =  [copy(L) for s in tt_]
Mt⁺ = [copy(Σ) for s in tt_]
μt = [copy(μend) for s in tt_]
println("compute guiding term:")
@time (Lend, Mend⁺, μend) =  guidingbackwards!(Lm(), tt_, (Lt, Mt⁺,μt), Paux, (L, Σ, μend))
(L0, M0⁺, μ0, V) = lmgpupdate(Lend, Mend⁺, μend, vT, Σ, L, v0)
# issymmetric(deepmat(Bridge.a(0,Paux)))
# isposdef(deepmat(Bridge.a(0,Paux)))
# map(x->minimum(eigen(deepmat(x)).values),Mt⁺)
#Mt = map(X -> deepmat2unc(inv(deepmat(X))),Mt⁺)
println("Compute Cholesky for Mt⁺:")
@time Mt = map(X -> InverseCholesky(lchol(X)),Mt⁺)

Q = GuidedProposall!(P, Paux, tt_, Lt, Mt, μt, xobs)
if partialobs
    xinit = State(v0, rand(Point{Float64}, P.n))
else
    xinit = x0
end
winit = zeros(StateW, dwiener)
XX = SamplePath(tt_, [copy(xinit) for s in tt_])
WW = SamplePath(tt_, [copy(winit) for s in tt_])
sample!(WW, Wiener{Vector{StateW}}())

println("Sample guided proposal:")
@time Bridge.solve!(EulerMaruyama!(), XX, xinit, WW, Q)
#guid = guidingterms(XX,Q)

include("plotlandmarks.jl")


if model==:ms
    @time llikelihood(LeftRule(), XX, Q; skip = 0)  # won't work for AHS because matrix multilication for Htilde is not defined yet
end



using ForwardDiff
dual(x, i, n) = ForwardDiff.Dual(x, ForwardDiff.Chunk{n}(), Val(i))
dual(x, n) = ForwardDiff.Dual(x, ForwardDiff.Chunk{n}(), Val(0))
#=
#using Flux
xinitv = deepvec(xinit)

xinitv = map(i->dual(xinitv[i], i <= 2 ? i : 0, 2), 1:length(xinitv))

xinitnew = deepvec2state(xinitv)
x = copy(xinitnew)

#lux.Tracker.gradient(x -> Bridge._b!((1,0.0), deepvec2state(x), deepvec2state(x), P), deepvec(xinit))
Bridge.b!(0.0, x, copy(x), P)

import Bridge;

#include(joinpath(dirname(pathof(Bridge)), "..", "landmarks/patches.jl"))
#include(joinpath(dirname(pathof(Bridge)), "..", "landmarks/models.jl"))

XX = Bridge.solve(EulerMaruyama!(), xinitnew, WW, P)
=#



function obj(xinitv)
    xinit = deepvec2state(xinitv)
    sample!(WW, Wiener{Vector{StateW}}())
    XXᵒ = Bridge.solve(EulerMaruyama!(), xinit, WW, Q)
    (
    (lptilde(xinit, Lend, Mend⁺, μend, V, Q) - lptilde(x0, Lend, Mend⁺, μend, V, Q))
     + llikelihood(LeftRule(), XXᵒ, Q; skip = 1)
    )
end
using Makie, Random
Random.seed!(2)
let
    x = deepvec(x0)
    x = x .* (1 .+ 0.2*randn(length(x)))

    x = deepvec(State(x0.q, deepvec2state(x).p))

    s = deepvec2state(x)
    n = Node(s.q)
    n2 = Node(s.p)
    sc = scatter(x0.q, color=:red)

    scatter!(sc, n2)
    scatter!(sc, n, color=:blue)
    display(sc)

    # only optimize momenta
    mask = deepvec(State(0 .- 0*xinit.q, 1 .- 0*(xinit.p)))
    ϵ = 6.e-4
    @show o =  obj(x)

    for i in 1:1000
    #record(sc, "output/gradientdescent.mp4", 1:100) do i
        #i % 10 == 0 && (o =  obj(x))
        for k in 1:1
            ∇x = ForwardDiff.gradient(obj, x)
            x .+= ϵ*mask.*∇x
        end
        s = deepvec2state(x)
        n[] = s.q
        n2[] = s.p
        display(s-x0)
        println("$i d(x,xtrue) = ", norm(deepvec(x0)-x))#, " ", o)
    end
end
