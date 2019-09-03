# THIS SCRIPT REPLACES THE OLDER 'lmpar.jl'
#] add StaticArrays Distributions DelimitedFiles DataFrames CSV RCall SparseArrays LowRankApprox Trajectories
#] add ForwardDiff DiffResults TimerOutputs Plots RecursiveArrayTools NPZ
using Bridge, StaticArrays, Distributions
using Bridge:logpdfnormal
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles, DataFrames, CSV, RCall
# install.packages(ggforce)
using Base.Iterators, SparseArrays, LowRankApprox, Trajectories
using ForwardDiff
using DiffResults
#using TimerOutputs #undeclared
using Plots,  PyPlot #using Makie
using RecursiveArrayTools

using NPZ # for reading python datafiles

workdir = @__DIR__
println(workdir)
cd(workdir)

pyplot()

const sk=1  # entries to skip for likelihood evaluation
const itostrat = true
const d = 2
const TEST = false

include("nstate.jl")
include("state.jl")
include("models.jl")
include("patches.jl")
include("lmguiding_mv.jl")
include("plotlandmarks.jl")
include("generatedata.jl")
include("plotting.jl")

################################# start settings #################################
n = 6  # nr of landmarks
models = [:ms, :ahs]
model = models[2]
println("model: ",model)

ITER = 50
subsamples = 0:1:ITER

startPtrue = false # start from true P?
showplotσq = false # only for ahs model

samplers =[:sgd, :sgld, :mcmc]
sampler = samplers[3]
println("sampler: ",sampler)

#------------------------------------------------------------------
# prior on θ = (a, c, γ)
prior_a = Exponential(5.0)
prior_c = Exponential(5.0)
prior_γ = Exponential(5.0)
#------------------------------------------------------------------

datasets =["forwardsimulated", "shifted","shiftedextreme","bear",
            "heart","peach", "generatedstefan", "forwardsimulated_multiple"]
dataset = datasets[8]
println("dataset: ",dataset)

#------------------------------------------------------------------
# for sgd (FIX LATER)
ϵ = 0.01  # sgd step size
ϵstep(i) = 1/(1+i)^(0.7)

#------------------------------------------------------------------
### MCMC tuning pars
# pcN-step
ρ = 0.9

# step-size on initial state
δ = [0.0, 0.25] # in this case first comp is not used

# proposal for θ = (a, c, γ)
σ_a = 0.1  # update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())
σ_c = 0.1  # update c to cᵒ as cᵒ = c * exp(σ_c * rnorm())
σ_γ = 0.1  # update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())

#------------------------------------------------------------------
σobs = 0.05   # noise on observations

#------------------------------------------------------------------
# set time grids
dt = 0.01
T = 1.0; t = 0.0:dt:T; tt_ =  tc(t,T)

outdir = "./figs/"
if false # to use later on, when plotting is transferred from R to Julia
    outdir = "./figs/"* string(model) * "_" * string(sampler) *"_" * string(dataset) * "/"
    if !isdir(outdir) mkdir(outdir) end
end
################################# end settings #################################

### Specify landmarks models
a = 2.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
c = 0.1     # multiplicative factor in kernel
γ = 1.0     # Noise level


if model == :ms
    λ = 0.0;    # Mean reversion par in MS-model = not the lambda of noise fields  =#
    nfs = 0 # needs to have value for plotting purposes
    Ptrue = MarslandShardlow(a, c, γ, λ, n)
else
    db = 4.0 # domainbound
    nfstd = 1.25 # tau , width of noisefields
    nfs = construct_nfs(db, nfstd, γ) # 3rd argument gives average noise of positions (with superposition)
    Ptrue = Landmarks(a, c, n, db, nfstd, nfs)
end

if (model == :ahs) & showplotσq
    plotσq(db, nfs)
end

x0, xobs0, xobsT, Xf, Ptrue, pb, obs_atzero  = generatedata(dataset,Ptrue,t,σobs)
if !obs_atzero
    δ = [0.0025, 0.001]
end

δ = 0.1*δ

if startPtrue
    P = Ptrue
else
    ainit = 0.1
    cinit = 0.1
    γinit = 0.1
    if model == :ms
        P = MarslandShardlow(ainit, cinit, γinit, Ptrue.λ, Ptrue.n)
    elseif model == :ahs
        nfsinit = construct_nfs(Ptrue.db, Ptrue.nfstd, γinit)
        P = Landmarks(ainit, cinit, Ptrue.n, Ptrue.db, Ptrue.nfstd, nfsinit)
    end
end

mT = zeros(PointF,P.n)   # vector of momenta at time T used for constructing guiding term

start = time() # to compute elapsed time
# anim, Xsave, parsave, objvals, perc_acc = lm_mcmc(tt_, (xobs0,xobsT), σobs, mT, P,
#                                     sampler, dataset,
#                                     xinit, ITER, subsamples,
#                                     (δ, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ),
#                                       outdir,pb;makefig=true)

if obs_atzero
    xobsTvec = [xobsT]
        xinit = State(xobs0, zeros(PointF,P.n))
else
    xobsTvec = xobsT #xobsT, xobsT + 0.1*rand(PointF,n)] # just a simple example
    xinit = State(xobsTvec[1], zeros(PointF,P.n))
end

anim, Xsave, parsave, objvals, perc_acc = lm_mcmc(tt_, (xobs0,xobsTvec), σobs, mT, P,
         sampler, dataset, obs_atzero,
         xinit, ITER, subsamples,
        (δ, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ),
        outdir, pb; updatepars = true, makefig=true, showmomenta=false)

elapsed = time() - start
println("Average acceptance percentage: ",perc_acc,"\n")
println("Elapsed    time: ",round(elapsed/60;digits=2), " minutes")

include("./postprocessing.jl")
