#] add StaticArrays Distributions DelimitedFiles DataFrames CSV RCall SparseArrays LowRankApprox Trajectories
#] add ForwardDiff DiffResults TimerOutputs Plots RecursiveArrayTools NPZ
using Bridge, StaticArrays, Distributions
using Bridge:logpdfnormal
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles, DataFrames, RCall
using CSV
# install.packages(ggforce)
using Base.Iterators, SparseArrays, LowRankApprox, Trajectories
using ForwardDiff
using Plots,  PyPlot #using Makie
using RecursiveArrayTools
using NPZ # for reading python datafiles
using Random

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
include("plotlandmarks.jl")  # keep, but presently unused as all is transferred to plotting in R
include("generatedata.jl")
include("plotting.jl")
#include("lmguiding_mv.jl")
#include("update_initialstate.jl")
include("lmguid.jl")  # replacing lmguiding_mv and update_initialstate



Random.seed!(3)

################################# start settings #################################
n = 10  # nr of landmarks
models = [:ms, :ahs]
model = models[1]
println("model: ",model)

ITER = 15
subsamples = 0:1:ITER

showplotσq = false # only for ahs model

sampler =[:sgd, :mcmc][2]
println("sampler: ",sampler)

#------------------------------------------------------------------
# prior on θ = (a, c, γ)
prior_a = Exponential(5.0)
prior_c = Exponential(5.0)
prior_γ = Exponential(5.0)
#------------------------------------------------------------------

datasets =["forwardsimulated", "shifted","shiftedextreme",
            "bear","heart","peach",
            "generatedstefan","forwardsimulated_multiple", "cardiac"]
dataset = datasets[9]
println("dataset: ",dataset)

fixinitmomentato0 = true
#------------------------------------------------------------------
# for sgd (FIX LATER)
ϵ = 0.01  # sgd step size
ϵstep(i) = 1/(1+i)^(0.7)

#------------------------------------------------------------------
### MCMC tuning pars
ρ = 0.8              # pcN-step

# proposal for θ = (a, c, γ)
σ_a = 0.2  # update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())
σ_c = 0.2  # update c to cᵒ as cᵒ = c * exp(σ_c * rnorm())
σ_γ = 0.2  # update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())

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
    nfstd = 1.75 # tau , width of noisefields
    nfs = construct_nfs(db, nfstd, γ) # 3rd argument gives average noise of positions (with superposition)
    Ptrue = Landmarks(a, c, n, db, nfstd, nfs)
end

if (model == :ahs) & showplotσq
    plotσq(db, nfs)
end

x0, xobs0, xobsT, Xf, Ptrue, pb, obs_atzero  = generatedata(dataset,Ptrue,t,σobs)

# step-size on mala steps for initial state
if obs_atzero
    δ = [0.0, 0.1] # in this case first comp is not used
else
    δ = [0.01, 0.001]
end

(ainit, cinit, γinit) = (0.1, 0.1, 0.1)
if model == :ms
    P = MarslandShardlow(ainit, cinit, γinit, Ptrue.λ, Ptrue.n)
elseif model == :ahs
    nfsinit = construct_nfs(Ptrue.db, Ptrue.nfstd, γinit)
    P = Landmarks(ainit, cinit, Ptrue.n, Ptrue.db, Ptrue.nfstd, nfsinit)
end


mT = zeros(PointF,P.n)   # vector of momenta at time T used for constructing guiding term
#mT = randn(PointF,P.n)

start = time() # to compute elapsed time


if obs_atzero     # only update mom
    xobsT = [xobsT]  # should be a vector
    xinit = State(xobs0, zeros(PointF,P.n))
    initstate_updatetypes = [:mala_mom]
elseif !obs_atzero & !fixinitmomentato0 # multiple shapes: update both pos and mom
    θ = π/6
    rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
    xinit = 1.3*State([rot * xobsT[1][i] for i in 1:P.n], zeros(PointF,P.n))
    #xinit = 1.3*State([rot * xobsTvec[1][i] for i in 1:n], randn(PointF,P.n))
    xinit = State(xobsT[1], zeros(PointF, P.n))
    initstate_updatetypes = [:mala_mom, :rmmala_pos]
elseif !obs_atzero & fixinitmomentato0   # multiple shapes only update pos, so initialised momenta (equal toz zero) are maintained throughout the iterations
    xinit = 1.2*State(xobsT[1], zeros(PointF,P.n))
    θ = π/6
    rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
    xinit = 1.3*State([rot * xobsT[1][i] for i in 1:P.n], zeros(PointF,P.n))
    xinit = State(xobsT[1], zeros(PointF, P.n))
    initstate_updatetypes = [:rmmala_pos]
end


anim, Xsave, parsave, objvals, perc_acc_pcn, accinfo = lm_mcmc(tt_, (xobs0,xobsT), σobs, mT, P,
         sampler, obs_atzero, fixinitmomentato0,
         xinit, ITER, subsamples,
        (ρ, δ, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ), initstate_updatetypes,
        outdir,  pb; updatepars = true, makefig=true, showmomenta=false)

elapsed = time() - start

println("Elapsed    time: ",round(elapsed/60;digits=2), " minutes")
println("Acceptance percentage pCN step: ",perc_acc_pcn)

include("./postprocessing.jl")
