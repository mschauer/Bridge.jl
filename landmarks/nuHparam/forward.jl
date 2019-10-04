# reminder, to type H*, do H\^+
#outdir="output/landmarks/"
#cd("/Users/Frank/.julia/dev/Bridge/landmarks")
#cd("landmarks")


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
using Trajectories
using LowRankApprox

models = [:ms, :ahs]
model = models[2]
TEST = false#true

discrmethods = [:ralston, :lowrank, :psd, :lm, :marcin] # not a good name anymore
discrmethod = discrmethods[1]
LM = discrmethod == :lm

obsschemes =[:full, :partial]
obsscheme = obsschemes[2]

const d = 2
const itostrat = false

n = 15 # nr of landmarks
ldim = 40   # dimension of low-rank approximation to H\^+

cheat = false #true#false # if cheat is true, then we initialise at x0 (true value) and
# construct the guiding term based on xT (true value)

θ = 0.0 #-π/6# π/6 0#π/5  # angle to rotate endpoint


ϵ = 10.0^(-4)   # parameter for initialising Hend⁺
σobs = 10.0^(-2)   # noise on observations

println(model)
println(discrmethod)
println(obsscheme)

T = 1.0#1.0#0.5
t = 0.0:0.01:T  # time grid

Random.seed!(5)
#include("nstate.jl")
include("ostate.jl")
include("state.jl")

include("models.jl")
include("patches.jl")
if discrmethod == :lm
    include("lmguiding.jl")
else
    include("guiding.jl")
end
include("LowrankRiccati.jl")
using .LowrankRiccati


### Specify landmarks models
a = 3.0 # the larger, the stronger landmarks behave similarly
λ = 0.0 # not the lambda of noise fields, but the mean reversion
γ = 1.0
db = 3.0 # domainbound
nfstd = 1.0 # tau , width of noisefields
r1 = -db:nfstd:db
r2 = -db:nfstd:db
nfloc = PointF.(collect(product(r1, r2)))[:]
nfscales = [.1PointF(1.0, 1.0) for x in nfloc]  # intensity

nfs = [Noisefield(δ, λ, nfstd) for (δ, λ) in zip(nfloc, nfscales)]
Pms = MarslandShardlow(a, γ, λ, n)
Pahs = Landmarks(a, λ, n, nfs)
###

StateW = PointF
if model == :ms
    dwiener = n
    P = Pms
else
    dwiener = length(nfloc)
    P = Pahs
end

# specify initial landmarks configuration
q0 = [PointF(2.5cos(t), sin(t)) for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
p_ = 10*PointF(0.1, 0.1)
p0 = [p_ for i in 1:n]  #
#p0 = [randn(PointF) for i in 1:n]
x0 = State(q0, p0)

#Random.seed!(1234)
w0 = zeros(StateW, dwiener)
W = SamplePath(t, [copy(w0) for s in t])
X = SamplePath(t, [copy(x0) for s in t])
sample!(W, Wiener{Vector{StateW}}())
println("Sample forward process:")
@time solve!(EulerMaruyama!(), X, x0, W, P)
#@time solve!(StratonovichHeun!(), X, x0, W, P)

# compute Hamiltonian along path
ham = [hamiltonian(X.yy[i], Pms) for i in 1:length(t)]

tc(t,T) = t.*(2 .-t/T)
tt_ =  tc(t,T)#tc(t,T)# 0:dtimp:(T)



####################
if obsscheme==:partial
    L = deepmat( [(i==j)*one(Unc) for i in 1:2:2n, j in 1:2n])
    Σ = Diagonal(σobs^2*ones(n*d))

    # observe positions
    v0 = q(X.yy[1])  + σobs * randn(PointF,n)
    rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
    shft = 0.
    vT = [shft .+ rot * X.yy[end].q[i] + σobs * randn(d)    for i in 1:P.n ]

    Pmsaux = MarslandShardlowAux(Pms, State(vT, zero(vT)))
    if cheat
        Pahsaux = LandmarksAux(Pahs, X.yy[end])
    else
        Pahsaux = LandmarksAux(Pahs, State(vT, zero(vT)))
    #Pahsaux = LandmarksAux(Pahs, State(vT, rand(PointF,Pahs.n)))
    end
elseif obsscheme==:full
    L = deepmat( [(i==j)*one(Unc) for i in 1:2n, j in 1:2n])
    Σ = Diagonal(σobs^2*ones(2n*d))
    Pmsaux = MarslandShardlowAux(Pms,X.yy[end])
    Pahsaux = LandmarksAux(Pahs, X.yy[end])
    v0 = vec(X.yy[1])
    vT = vec(X.yy[end])
end



#Paux = (model==:ms) ? Pmsaux : Pahsaux
# solve backward recursion on [0,T]
if LM

    if obsscheme == :partial
        Lend = deepmat2unc(L)
        Σend = deepmat2unc(Matrix(Σ))
        μend = zeros(PointF, n)
        xobs = vT
    else
        # full observation case
#        L = [(i==j)*one(Unc) for i in 1:2n, j in 1:2n]
#        Σ = [(i==j)*σobs^2*one(Unc) for i in 1:2n, j in 1:2n]
        μend = zeros(PointF, 2n)
    end
else
    if obsscheme == :partial
        #νT = State(zero(vT), zero(vT))
        νT = State(randn(PointF,Pahs.n), 0randn(PointF,Pahs.n))
    elseif obsscheme == :full
        νT = X.yy[end]
    end
    HT⁺ = reshape(zeros(Unc,4n^2),2n,2n)
    for i in 1:n
        HT⁺[2i-1,2i-1] = one(Unc)/ϵ  # high variance on positions
        if discrmethod==:lowrank
            HT⁺[2i,2i] = one(Unc)*10^(-4) # still need to figure out why
        else
            HT⁺[2i,2i] = one(Unc)/10^(-4)
        end
    end
    #### perform gpupdate step
    νT , HT⁺, CT = gpupdate(νT, HT⁺, 0.0, Σ, L, vT)
    C = CT
    ν = copy(νT)
    H⁺  = copy(HT⁺)
    Pahsaux = LandmarksAux(Pahs, copy(ν))   # this might be a good idea
    Pmsaux = MarslandShardlowAux(Pms, copy(ν))   # this might be a good idea
end

if model == :ms
    Paux = Pmsaux
else
    Paux = Pahsaux
end


# L, and Σ are ordinary matrices, vT an array of Points
# ν is a state , H⁺ a  UncMat
if LM

    # initialise Lt and M⁺t
    Lt =  [copy(Lend) for s in tt_]
    Mt⁺ = [Matrix(Σend) for s in tt_]
    μt = [copy(μend) for s in tt_]
    println("compute guiding term:")
    @time (Lend, Mend⁺, μend) =  guidingbackwards!(Lm(), tt_, (Lt, Mt⁺,μt), Paux, (L, Σ, μend))

    # issymmetric(deepmat(Bridge.a(0,Paux)))
    # isposdef(deepmat(Bridge.a(0,Paux)))
    # map(x->minimum(eigen(deepmat(x)).values),Mt⁺)
    #Mt = map(X -> deepmat2unc(inv(deepmat(X))),Mt⁺)
    println("Compute Cholesky for Mt⁺:")
    @time Mt = map(X -> InverseCholesky(lchol(X)),Mt⁺)

    Q = GuidedProposall!(P, Paux, tt_, Lt, Mt, μt, xobs)
    (Lstart, Mstart⁺, μstart) = lmgpupdate(Lend, Mend⁺, μend, Σ, L, v0)
    xinit = x0



else
    νt =  [copy(ν) for s in tt_]
    println("Compute guiding term:")
    if discrmethod==:lowrank
        M0 = eigen(deepmat(H⁺))
        largest = sortperm(M0.values)[end-ldim+1:end]
        S = Matrix(Diagonal(M0.values[largest]))
        U = M0.vectors[:,largest]
        St = [copy(S) for s in tt_]
        Ut = [copy(U) for s in tt_]
        @time ν, (S, U) = bucybackwards!(LRR(), tt_, νt, (St, Ut), Paux, ν, (S, U))
        H⁺ = deepmat2unc(U * S * U')
        Ht = map((S,U) -> deepmat2unc(U * inv(S) * U'), St, Ut)  # directly compute Mt
        #Ht = map((S,U) -> LowRank(S,U), St,Ut)
    elseif discrmethod==:ralston
        H⁺t = [copy(H⁺) for s in tt_]
        Ht = map(H⁺ -> InverseCholesky(lchol(H⁺ + I)), H⁺t)
        @time ν , H⁺, H, C = bucybackwards!(Bridge.R3!(), tt_, νt, H⁺t, Ht, Paux, ν, H⁺, C)
#        Ht = map(H⁺ -> InverseCholesky(lchol(H⁺)),H⁺t)
    elseif discrmethod==:psd
        H⁺t = [copy(H⁺) for s in tt_]
        @time ν , H⁺, C = bucybackwards!(Lyap(), tt_, νt, H⁺t, Paux, ν, H⁺)
    #    println(map(x->isposdef(deepmat(x)),H⁺t))
        Ht = map(H⁺ -> InverseCholesky(lchol(H⁺)),H⁺t)
    end

# careful, not a state
    νstart , Hstart⁺, Cstart = gpupdate(ν , H⁺, C, Σ, L, v0)
    #also update C??
    Q = GuidedProposal!(P, Paux, tt_, νt, Ht, νstart, InverseCholesky(lchol(Hstart⁺)), Cstart)
    xinit = cheat ? x0 : State(νstart.q, 0νstart.q)#νstart  # or xinit ~ N(νstart, Hstart⁺)
end
winit = zeros(StateW, dwiener)
XX = SamplePath(tt_, [copy(xinit) for s in tt_])
WW = SamplePath(tt_, [copy(winit) for s in tt_])
sample!(WW, Wiener{Vector{StateW}}())

# adjust xinit as test
xinit = State(v0, [Point(3,3) for i in 1:P.n] + rand(Point{Float64}, P.n))

println("Sample guided bridge proposal:")
@time Bridge.solve!(EulerMaruyama!(), XX, xinit, WW, Q)
#error("STOP EARLY")
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
    if !isdefined(Main, :XXᵒ_)
        XXᵒ_ = Bridge.solve(EulerMaruyama!(), xinit, WW, Q)
    else
        Bridge.solve!(EulerMaruyama!(), XXᵒ_, xinit, WW, Q)
    end
    (
    (lptilde(xinit, Q) - lptilde(x0, Q))  + llikelihood(LeftRule(), XXᵒ_, Q; skip = 1)
    )
end

MAKIE = false
if MAKIE
    using Makie
end
using Random, Profile
Random.seed!(2)
let
    x = deepvec(x0)
    #x = deepvec(State(x0.q, 0.5 * x0.p))
    x = x .* (1 .+ 0.2*randn(length(x)))

    x = deepvec(State(x0.q, deepvec2state(x).p))


    s = deepvec2state(x)
    if MAKIE
        n = Node(s.q)
        n2 = Node(s.p)
        sc = scatter(x0.q, color=:red)

        scatter!(sc, n2)
        scatter!(sc, n, color=:blue)
        display(sc)
    end
    # only optimize momenta
    mask = deepvec(State(0 .- 0*xinit.q, 1 .- 0*(xinit.p)))
    ϵ = 6.e-2
    #@show o =  obj(x)
    ∇x = ForwardDiff.gradient(obj, x)
    x .+= ϵ*mask.*∇x
    s = deepvec2state(x)
    if MAKIE
        n[] = s.q
        n2[] = s.p
    end
    display(s-x0)


    @profile for i in 1:1000
    #record(sc, "output/gradientdescent.mp4", 1:100) do i
        #i % 10 == 0 && (o =  obj(x))
        for k in 1:1
            ∇x = ForwardDiff.gradient(obj, x)
            x .+= ϵ*mask.*∇x
        end
        s = deepvec2state(x)
        if MAKIE
            n[] = s.q
            n2[] = s.p
        end
        display(s-x0)
        println("$i d(x,xtrue) = ", norm(deepvec(x0)-x))#, " ", o)
    end
end
