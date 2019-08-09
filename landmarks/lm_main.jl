workdir = "/Users/Frank/.julia/dev/Bridge/landmarks/"
outdir = "/Users/Frank/.julia/dev/Bridge/landmarks/figs/"
cd(workdir)
# THIS SCRIPT REPLACES THE OLDER 'lmpar.jl'
using Bridge, StaticArrays, Distributions
using Bridge:logpdfnormal
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles,  DataFrames,  CSV, RCall
using Base.Iterators, SparseArrays, LowRankApprox, Trajectories
using ForwardDiff #: GradientConfig, Chunk, gradient!, gradient, Dual, value
using ReverseDiff #: GradientConfig,  gradient!, gradient, Dual, value
using DiffResults
using TimerOutputs #undeclared
using Plots,  PyPlot #using Makie
using RecursiveArrayTools
using DataFrames
using NPZ # for reading python datafiles

pyplot()

const sk=1  # entries to skip for likelihood evaluation
const itostrat = true#false#true                    #false#true#false#true
const d = 2
const inplace = true  # if true inplace updates on the path when doing autodifferentiation
const TEST = false

#include("ostate.jl")
include("nstate.jl")
include("state.jl")
include("models.jl")
include("patches.jl")
include("lmguiding.jl")
include("plotlandmarks.jl")
include("automaticdiff_lm.jl")
include("generatedata.jl")
include("lm_mcmc.jl")

################################# start settings #################################
n = 10#35 # nr of landmarks
models = [:ms, :ahs]
model = models[1]
println(model)

ITER = 1000 # nr of sgd iterations
subsamples = 0:1:ITER


startPtrue = false  # start from true P?
showplotσq = false # only for ahs model

samplers =[:sgd, :sgld, :mcmc]
sampler = samplers[3]

ρ = 0.9
if model==:ms
    δ = 0.1
elseif model==:ahs
    δ = 0.005
end

σ_a = 0.1  # update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())
σ_γ = 0.1  # update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())

ϵ = 0.01  # sgd step size
ϵstep(i) = 1/(1+i)^(0.7)


datasets =["forwardsimulated", "shifted","shiftedextreme",
        "bear", "heart","peach", "generatedstefan"]
dataset = datasets[2]


σobs = 0.01   # noise on observations

prior_a = Uniform(0.1,10)
prior_γ = Exponential(1.0)

# set time grids
T = 1.0
dt = 0.01
t = 0.0:dt:T  # time grid
tt_ =  tc(t,T)                          #tc(t,T)# 0:dtimp:(T)

################################# end settings #################################

### Specify landmarks models
a = 2.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
γ = 1.0     # Noise level

if model == :ms
    λ = 0.0;    # Mean reversion par in MS-model = not the lambda of noise fields  =#
    nfs = 0 # needs to have value for plotting purposes
    Ptrue = MarslandShardlow(a, γ, λ, n)
else
    db = 5.0 # domainbound
    nfstd = 2.5#2.5#  1.25 # tau , width of noisefields
    nfs = construct_nfs(db, nfstd, γ) # 3rd argument gives average noise of positions (with superposition)
    Ptrue = Landmarks(a, n, db, nfstd, nfs)
end

if (model == :ahs) & showplotσq
    plotσq(db, nfs)
end

x0, xobs0, xobsT, Xf, Ptrue = generatedata(dataset,Ptrue,t,σobs)

if startPtrue
    P = Ptrue
else
    ainit = 0.3
    γinit = 2.0
    if model == :ms
        P = MarslandShardlow(ainit, γinit, Ptrue.λ, Ptrue.n)
    elseif model == :ahs
        nfsinit = construct_nfs(Ptrue.db, Ptrue.nfstd, γinit)
        P = Landmarks(ainit, Ptrue.n, Ptrue.db, Ptrue.nfstd, nfsinit)
    end
end

mT = zeros(PointF,P.n)   # vector of momenta at time T used for constructing guiding term
#xinit = State(xobs0, [Point(-1.0,3.0)/P.n for i in 1:P.n])
xinit = State(xobs0, zeros(PointF,P.n)) # xinit = State(xobs0, rand(PointF,P.n))# xinit = x0

start = time() # to compute elapsed time
Xsave, parsave, objvals, perc_acc = lm_mcmc(tt_, (xobs0,xobsT), mT, P, model, sampler,
                                        dataset, xinit, δ, ITER, subsamples,
                                        prior_a, prior_γ, σ_a, σ_γ, outdir)
elapsed = time() - start

include("/Users/Frank/.julia/dev/Bridge/landmarks/postprocessing.jl")
