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

outdir = "/Users/Frank/.julia/dev/Bridge/landmarks/figs/"

pyplot()

#Base.Float64(d::Dual{T,V,N}) where {T,V,N} = Float64(d.value)
#Base.float(d::Dual{T,V,N}) where {T,V,N} = Float64(d.value)

deepvalue(x::Float64) = x
deepvalue(x::ForwardDiff.Dual) = ForwardDiff.value(x)
deepvalue(x) = deepvalue.(x)
function deepvalue(x::State)
    State(deepvalue.(x.x))
end




n = 6#35 # nr of landmarks
models = [:ms, :ahs]
model = models[1]
println(model)

TEST = false#true
partialobs = true  #false
rotation = false  # rotate configuration at time T
showplotσq = false

samplers =[:sgd, :sgld, :mcmc]
sampler = samplers[3]



ρ = 0.9#0.99999#1.0 #0.9999 #CN par
if model==:ms
    δ = 0.1#0.00001 # MALA par
else
    δ = 0.005
end
ϵ = 0.01  # sgd step size
ϵstep(i) = 1/(1+i)^(0.7)



datasets =["forwardsimulated", "shifted","shiftedextreme", "bear", "heart","peach"]
dataset = datasets[1]

ITER = 10 # nr of sgd iterations
subsamples = 0:2:ITER

const sk=1  # entries to skip for likelihood evaluation
const itostrat = true#false#true                    #false#true#false#true
const d = 2
const inplace = true  # if true inplace updates on the path when doing autodifferentiation



σobs = 0.01   # noise on observations

T = 1.0#1.0#0.5
t = 0.0:0.005:T  # time grid

#Random.seed!(5)
#include("ostate.jl")
include("nstate.jl")
include("state.jl")
#include("state_localversion.jl")
include("models.jl")
include("patches.jl")
include("lmguiding.jl")
include("plotlandmarks.jl")
include("automaticdiff_lm.jl")
include("generatedata.jl")


### Specify landmarks models
a = 5.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
#a = 5.0

if model == :ms
    λ = 0.0;    # Mean reversion par in MS-model = not the lambda of noise fields  =#
    γ = .5 #2.0     # Noise level in for MS-model
    dwiener = n
    nfs = 0 # needs to have value for plotting purposes
    P = MarslandShardlow(a, γ, λ, n)
else
    db = 5.0 # domainbound
    nfstd = 2.5#2.5#  1.25 # tau , width of noisefields
    γ = 0.2
    nfs = construct_nfs(db, nfstd, γ) # 3rd argument gives average noise of positions (with superposition)
    dwiener = length(nfs)
    P = Landmarks(a, n, nfs)
end



if (model == :ahs) & showplotσq
    plotσq(db, nfs)
end

StateW = PointF

# set time grid for guided process
tt_ =  tc(t,T)#tc(t,T)# 0:dtimp:(T)

# generate data
x0, xobs0, xobsT, Xf, P = generatedata(dataset,P,t,σobs)

# plotlandmarkpositions(Xf,P.n,model,xobs0,xobsT,nfs,db=6)#2.6)
 # ham = [hamiltonian(Xf.yy[i],P) for i in 1:length(t)]
 # Plots.plot(1:length(t),ham)
 # print(ham)

if partialobs
    L0 = LT = [(i==j)*one(UncF) for i in 1:2:2P.n, j in 1:2P.n]
    Σ0 = ΣT = [(i==j)*σobs^2*one(UncF) for i in 1:P.n, j in 1:P.n]
    μT = zeros(PointF,P.n)
    mT = zeros(PointF,P.n)   #
else
    LT = [(i==j)*one(UncF) for i in 1:2P.n, j in 1:2P.n]
    ΣT = [(i==j)*σobs^2*one(UncF) for i in 1:2P.n, j in 1:2P.n]
    μT = zeros(PointF,2P.n)
    xobsT = vec(X.yy[end])
    mT = Xf.yy[end].p
    L0 = [(i==j)*one(UncF) for i in 1:2:2P.n, j in 1:2P.n]
    Σ0 = [(i==j)*σobs^2*one(UncF) for i in 1:P.n, j in 1:P.n]
end

 #


if model == :ms
    Paux = MarslandShardlowAux(P, State(xobsT, mT))
else
    Paux = LandmarksAux(P, State(xobsT, mT))
end

# initialise guided path
xinit = State(xobs0, [Point(-1.0,3.0)/P.n for i in 1:P.n])
# xinit = State(xobs0, rand(PointF,n))# xinit = x0#xinit = State(xobs0, zeros(PointF,n))#xinit=State(x0.q, 30*x0.p)

Xsave, objvals, perc_acc = lm_mcmc(tt_, (LT,ΣT,μT), (L0,Σ0), (xobs0,xobsT), P, Paux, model, sampler,
                                        dataset, xinit, δ, ITER, outdir; makefig=true)


########### grad desc for pars

# also do gradient descent on parameters a (in kernel of Hamiltonian)
# first for MS model
get_targetpars(Q::GuidedProposall!) = [Q.target.a, Q.target.γ]
get_auxpars(Q::GuidedProposall!) = [Q.aux.a, Q.aux.γ]

put_targetpars = function(pars,Q)
    GuidedProposall!(MarslandShardlow(pars[1],pars[2],Q.target.λ, Q.target.n), Q.aux, Q.tt, Q.Lt, Q.Mt, Q.μt,Q.Ht, Q.xobs)
end

put_auxpars(pars,Q) = GuidedProposall!(Q.target,MarslandShardlowAux(pars[1],pars[2],Q.aux.λ, Q.aux.xT,Q.aux.n), Q.tt, Q.Lt, Q.Mt, Q.μt,Q.Ht, Q.xobs)

QQ = put_targetpars([3.0, 300.0],Q)
QQ.target.a
QQ.target.γ




if false
    # write mcmc iterates to csv file
    iterates = reshape(vcat(Xsave...),2*d*length(tt_)*P.n, length(subsamples)) # each column contains samplepath of an iteration
    # Ordering in each column is as follows:
    # 1) time
    # 2) landmark nr
    # 3) for each landmark: q1, q2 p1, p2
    pqtype = repeat(["pos1", "pos2", "mom1", "mom2"], length(tt_)*P.n* length(subsamples))

    fn = outdir*"iterates.csv"
    iterates = [Any[s, Xsave[i].tt[j], d, Xsave[i].yy[j][d]] for d in 1:1, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
    f = open(fn,"w")
    head = "iteration, time, component, value \n"
    write(f, head)
    writedlm(f,iterates)
    close(f)

    ave_acc_perc = 100*round(acc/iterations,2)

    # write info to txt file
    fn = outdir*"info.txt"
    f = open(fn,"w")
    write(f, "Number of iterations: ",string(iterations),"\n")
    write(f, "Skip every ",string(skip_it)," iterations, when saving to csv","\n\n")
    write(f, "Starting point: ",string(x0),"\n")
    write(f, "End time T: ", string(T),"\n")
    write(f, "Endpoint v: ",string(v),"\n")
    write(f, "Noise Sigma: ",string(Î£),"\n")
    write(f, "L: ",string(L),"\n\n")
    write(f, "Mesh width: ",string(dt),"\n")
    write(f, "rho (Crank-Nicholsen parameter: ",string(Ï),"\n")
    write(f, "Average acceptance percentage: ",string(ave_acc_perc),"\n\n")
    write(f, "Backward type parametrisation in terms of nu and H? ",string(Î½Hparam),"\n")
    close(f)


    println("Average acceptance percentage: ",ave_acc_perc,"\n")
    println("Parametrisation of nu and H? ", Î½Hparam)
    println("Elapsed time: ",elapsed_time)
end
