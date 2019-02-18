# reminder, to type H*, do H\^+
#outdir="output/landmarks/"
cd("/Users/Frank/.julia/dev/Bridge/landmarks")

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

models = [:ms, :ahs]
model = models[1]
TEST = false#true
partialobs = true


const d = 2
const itostrat = false

n = 75 # nr of landmarks

θ = π/3# π/6 0#π/5  # angle to rotate endpoint

σobs = 10^(-3)   # noise on observations
println(model)

T = 0.3#1.0#0.5
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
db = 3.0 # domainbound
nfstd = 1.0 # tau , widht of noisefields
r1 = -db:nfstd:db
r2 = -db:nfstd:db
nfloc = Point.(collect(product(r1, r2)))[:]
nfscales = [.05Point(1.0, 1.0) for x in nfloc]  # intensity

nfs = [Noisefield(δ, λ, nfstd) for (δ, λ) in zip(nfloc, nfscales)]
Pms = MarslandShardlow(a, γ, λ, n)
Pahs = Landmarks(a, λ, n, nfs)
###

StateW = Point
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
q0 = [Point(2.5cos(t), sin(t)) for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
p_ = 2*Point(-0.1, 0.1)
p0 = [p_ for i in 1:n]  #
#p0 = [randn(Point) for i in 1:n]
x0 = State(q0, p0)

#Random.seed!(1234)
X = SamplePath(t, [copy(x0) for s in t])
println("Solve for forward provess:")
@time solve!(EulerMaruyama!(), X, x0, W, P)
    #@time solve!(StratonovichHeun!(), X, x0, W, P)

tc(t,T) = t.*(2-t/T)
tt_ =  tc(t,T)#tc(t,T)# 0:dtimp:(T)


# observe positions without noise
v0 = q(X.yy[1])
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
vT = [rot * X.yy[end].q[i] for i in 1:P.n ]


#extractcomp(v,i) = map(x->x[i], v)
####################
# solve backward recursion on [0,T]
if partialobs==true
    L = [(i==j)*one(Unc) for i in 1:2:2n, j in 1:2n]
    Σ = [(i==j)*σobs^2*one(Unc) for i in 1:n, j in 1:n]
    #Σ = 10 * reshape(rand(Unc,n^2),n,n)
    μend = zeros(Point,P.n)
    xobs = vT
    Pahsaux = LandmarksAux(Pahs, State(vT, zero(vT)))
    Pmsaux = MarslandShardlowAux(Pms, State(vT, zero(vT)))
    Pahsaux = LandmarksAux(Pahs, State(vT, rand(Point,Pahs.n)))
else
    # full observation case
    L = [(i==j)*one(Unc) for i in 1:2n, j in 1:2n]
    Σ = [(i==j)*σobs^2*one(Unc) for i in 1:2n, j in 1:2n]
    μend = zeros(Point,2P.n)
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
@time (Lend, Mend⁺,μend) =  guidingbackwards!(Lm(), tt_, (Lt, Mt⁺,μt), Paux, (L, Σ, μend))

# issymmetric(deepmat(Bridge.a(0,Paux)))
# isposdef(deepmat(Bridge.a(0,Paux)))
# map(x->minimum(eigen(deepmat(x)).values),Mt⁺)
#Mt = map(X -> deepmat2unc(inv(deepmat(X))),Mt⁺)
println("Compute Cholesky for Mt⁺:")
@time Mt = map(X -> InverseCholesky(lchol(X)),Mt⁺)

Q = GuidedProposall!(P, Paux, tt_, Lt, Mt, μt, xobs)
if partialobs
    xinit = State(v0, rand(Point,P.n))
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
