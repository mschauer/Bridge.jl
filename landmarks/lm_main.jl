 cd("/Users/Frank/.julia/dev/Bridge/landmarks/")
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

outdir = "/Users/Frank/.julia/dev/Bridge/landmarks/figs/"

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

startPtrue = false  # start from true P?

showplotσq = false

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

ITER = 5 # nr of sgd iterations
subsamples = 0:1:ITER

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

x0, xobs0, xobsT, Xf, Ptrue = generatedata(dataset,P,t,σobs)

if startPtrue
    P = Ptrue
else
    if model == :ms
        P = MarslandShardlow(.3, 2.0, P.λ, P.n)
    elseif model == :ahs
        P = Landmarks(.3, P.n, P.db, P.nfstd, P.nfs)
    end
end

# plotlandmarkpositions(Xf,P.n,model,xobs0,xobsT,P.nfs,db=6)#2.6)
# ham = [hamiltonian(Xf.yy[i],P) for i in 1:length(t)]
# Plots.plot(1:length(t),ham)
# print(ham)

# vector of momenta at time T used for constructing guiding term
mT = zeros(PointF,P.n)   #

# initialise guided path
#xinit = State(xobs0, [Point(-1.0,3.0)/P.n for i in 1:P.n])
xinit = State(xobs0, zeros(PointF,P.n)) # xinit = State(xobs0, rand(PointF,P.n))# xinit = x0

start = time() # to compute elapsed time
Xsave, parsave, objvals, perc_acc = lm_mcmc(tt_, (xobs0,xobsT), mT, P, model, sampler,
                                        dataset, xinit, δ, ITER, subsamples,
                                        prior_a, prior_γ, σ_a, σ_γ, outdir)


elapsed = time() - start

# write mcmc iterates to csv file
iterates = reshape(vcat(Xsave...),2*d*length(tt_)*P.n, length(subsamples)) # each column contains samplepath of an iteration
# Ordering in each column is as follows:
# 1) time
# 2) landmark nr
# 3) for each landmark: q1, q2 p1, p2
pqtype = repeat(["pos1", "pos2", "mom1", "mom2"], length(tt_)*P.n)
times = repeat(tt_,inner=2d*P.n)
landmarkid = repeat(1:P.n, inner=2d, outer=length(tt_))

out = hcat(times,pqtype,landmarkid,iterates)
head = "time " * "pqtype " * "landmarkid " * prod(map(x -> "iter"*string(x)*" ",subsamples))
head = chop(head,tail=1) * "\n"

fn = outdir*"iterates.csv"
f = open(fn,"w")
write(f, head)
writedlm(f,out)
close(f)

println("Average acceptance percentage: ",perc_acc,"\n")
println("Elapsed time: ",round(elapsed;digits=3))



# write info to txt file
fn = outdir*"info.txt"
f = open(fn,"w")
write(f, "Dataset: ", string(dataset),"\n")
write(f, "Sampler: ", string(sampler), "\n")

write(f, "Number of iterations: ",string(ITER),"\n")
write(f, "Number of landmarks: ",string(P.n),"\n")
write(f, "Length time grid: ", string(length(tt_)),"\n")
write(f, "Mesh width: ",string(dt),"\n")
write(f, "Noise Sigma: ",string(σobs),"\n")
write(f, "rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
write(f, "MALA parameter (delta): ",string(δ),"\n")
write(f, "skip in evaluation of loglikelihood: ",string(sk),"\n")
write(f, "Average acceptance percentage (path - initial state): ",string(perc_acc),"\n\n")
#write(f, "Backward type parametrisation in terms of nu and H? ",string(Î½Hparam),"\n")
close(f)



pardf = DataFrame(a=extractcomp(parsave,1),gamma=extractcomp(parsave,2), subsamples=subsamples)
@rput pardf
R"""
library(ggplot2)
pardf %>% ggplot(aes(x=a,y=gamma,colour=subsamples)) + geom_point()
"""

pp1 = Plots.plot(subsamples, extractcomp(parsave,1),label="")
xlabel!(pp1,"iteration")
pp2 = Plots.plot(subsamples, extractcomp(parsave,2),label="")
xlabel!(pp2,"iteration")
pp3 = Plots.plot(extractcomp(parsave,1), extractcomp(parsave,2),seriestype=:scatter,label="")
xlabel!(pp3,"a")
ylabel!(pp3,"γ")
l = @layout [a  b c]
pp = Plots.plot(pp1,pp2,pp3,background_color = :ivory,layout=l , size = (900, 500) )



################ following is probably obsolete
if false
    ########### grad desc for pars

    # also do gradient descent on parameters a (in kernel of Hamiltonian)
    # first for MS model
    get_targetpars(Q::GuidedProposall!) = [Q.target.a, Q.target.γ]
    get_auxpars(Q::GuidedProposall!) = [Q.aux.a, Q.aux.γ]

    put_targetpars = function(pars,Q)
        if isa(Q.target,MarslandShardlow)
            P = MarslandShardlow(pars[1],pars[2],Q.target.λ, Q.target.n)
        end
        if isa(Q.target,Landmarks)
            nfs = construct_nfs(Q.target.db, Q.target.nfstd, pars[2]) # need ot add db and nfstd to struct Landmarks
            P = Landmarks(pars[1],P.n,nfs)
        end
        GuidedProposall!(P, Q.aux, Q.tt, Q.Lt, Q.Mt, Q.μt,Q.Ht, Q.xobs0, Q.xobsT, Q.Lt0,Q.Mt⁺0,Q.μt0)
    end


    if TEST
        get_targetpars(Q)
        QQ = put_targetpars([3.0, 300.0],Q)
        QQ.target.a
        QQ.target.γ
    end
end
