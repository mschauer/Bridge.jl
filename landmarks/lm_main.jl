# THIS SCRIPT REPLACES THE OLDER 'lmpar.jl'
using Bridge, StaticArrays, Distributions
using Bridge:logpdfnormal
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles,  DataFrames,  CSV, RCall
using Base.Iterators, SparseArrays, LowRankApprox, Trajectories
using ForwardDiff: GradientConfig, Chunk, gradient!, gradient
using TimerOutputs #undeclared
using Plots,  PyPlot #using Makie

pyplot()

models = [:ms, :ahs]
model = models[2]
println(model)

TEST = false#true
partialobs = true  #false
rotation = false  # rotate configuration at time T
showplotσq = false

samplers =[:sgd, :sgld, :mcmc]
sampler = samplers[3]

ρ = 0.99 #CN par
δ = 0.8 # MALA par
ϵ = 0.01  # sgd step size
ϵstep(i) = 1/(1+i)^(0.7)


datasets =["forwardsimulated", "shifted","shiftedextreme", "bear", "heart","peach"]
dataset = datasets[2]

ITER = 10 # nr of sgd iterations
subsamples = 0:2:ITER


const itostrat = true                    #false#true#false#true
const d = 2

n = 10#35 # nr of landmarks

σobs = 0.01   # noise on observations

T = 2.0#1.0#0.5
t = 0.0:0.005:T  # time grid

#Random.seed!(5)
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
a = 2.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)

if model == :ms
    λ = 0.0;    # Mean reversion par in MS-model = not the lambda of noise fields  =#
    γ = 10.0     # Noise level in for MS-model
    dwiener = n
    nfs = 0 # needs to have value for plotting purposes
    P = MarslandShardlow(a, γ, λ, n)
else
    db = 5.0 # domainbound
    nfstd = 2.5#  1.25 # tau , width of noisefields
    nfs = construct_nfs(db, nfstd, .2) # 3rd argument gives average noise of positions (with superposition)
    dwiener = length(nfs)
    P = Landmarks(a, 0.0, n, nfs)
end

if (model == :ahs) & showplotσq
    plotσq(db, nfs)
end

StateW = PointF

# set time grid for guided process
tt_ =  tc(t,T)#tc(t,T)# 0:dtimp:(T)

# generate data
x0, xobs0, xobsT, Xf, P = generatedata(dataset,P,t,σobs)
plotlandmarkpositions(Xf,P.n,model,xobs0,xobsT,nfs,db=3)#2.6)

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



if model == :ms
    Paux = MarslandShardlowAux(P, State(xobsT, mT))
else
    Paux = LandmarksAux(P, State(xobsT, mT))
end

# compute guided prposal
println("compute guiding term:")
Lt, Mt⁺ , μt = initLMμ(tt_,(LT,ΣT,μT))
(Lt, Mt⁺ , μt), Q, (Lt0, Mt⁺0, μt0, xobst0) = construct_guidedproposal!(tt_, (Lt, Mt⁺ , μt), (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P, Paux)

# initialise guided path
xinit = State(xobs0, [Point(-6.,6.) for i in 1:P.n])
xinit = State(xobs0, rand(PointF,n))
#xinit = State(xobs0, zeros(PointF,n))

# sample guided path
println("Sample guided proposal:")
Xᵒ = initSamplePath(tt_, xinit)
Wᵒ = initSamplePath(tt_,  zeros(StateW, dwiener))
sample!(Wᵒ, Wiener{Vector{StateW}}())
@time Bridge.solve!(EulerMaruyama!(), Xᵒ, xinit, Wᵒ, Q)
#guid = guidingterms(Xᵒ,Q)
# plot forward path and guided path
plotlandmarkpositions(Xf,Xᵒ,P.n,model,xobs0,xobsT,nfs,db=3)#2.6)

@time llikelihood(LeftRule(), Xᵒ, Q; skip = 1)
@time lptilde(vec(xinit), Lt0, Mt⁺0, μt0, xobst0)


objvals =   Float64[]  # keep track of (sgd approximation of the) loglikelihood


mask = deepvec(State(0 .- 0*xinit.q, 1 .- 0*(xinit.p)))  # only optimize momenta
mask_id = (mask .> 0.1) # get indices that correspond to momenta


sk = 1
acc = zeros(2) # keep track of mcmc accept probs (first comp is for CN update; 2nd component for langevin update on initial momenta)

#Xsave = SamplePath{State{PointF}}[]
Xsave = typeof(Xᵒ)[]

# initialisation
W = copy(Wᵒ)
Wnew = copy(Wᵒ)
X = SamplePath(t, [copy(xinit) for s in tt_])
solve!(EulerMaruyama!(), X, x0, W, Q)
ll = llikelihood(Bridge.LeftRule(), X, Q,skip=sk)
Xᵒ = copy(X)
if 0 in subsamples
    push!(Xsave, copy(X))
end

x = deepvec(xinit)
xᵒ = copy(x)
∇x = copy(x)
∇xᵒ = copy(x)

# for plotting
xobs0comp1 = extractcomp(xobs0,1)
xobs0comp2 = extractcomp(xobs0,2)
xobsTcomp1 = extractcomp(xobsT,1)
xobsTcomp2 = extractcomp(xobsT,2)

showmomenta = false

anim =    @animate for i in 1:ITER
#for i in 1:ITER
    #
    global ll
    global acc
    global X
    global Xᵒ
    global W
    global Wᵒ
    global x
    global xᵒ
    global ∇x
    global ∇xᵒ
    println("iteration $i")

    δ = ϵstep(i)

    if sampler==:mcmc
        δ = 0.02 # for mala in this case
    end

     X,Xᵒ,W,Wᵒ,ll,x,xᵒ,∇x,∇xᵒ, obj,acc = updatepath!(X,Xᵒ,W,Wᵒ,Wnew,ll,x,xᵒ,∇x, ∇xᵒ,
            sampler,(Lt0,  Mt⁺0, μt0, xobst0, Q),
                mask, mask_id, δ, ρ, acc)

    if i in subsamples
        push!(Xsave, copy(X))
    end
    push!(objvals, obj)
end

error("FF STOPPPEN NU")

    # plotting
    s = deepvec2state(x).p
    s0 = x0.p # true momenta

    # plot initial and final shapes
    pp = Plots.plot(xobs0comp1, xobs0comp2,seriestype=:scatter, color=:black,label="q0", title="Landmark evolution")
    Plots.plot!(pp, repeat(xobs0comp1,2), repeat(xobs0comp2,2),seriestype=:path, color=:black,label="")
    Plots.plot!(pp, xobsTcomp1, xobsTcomp2,seriestype=:scatter , color=:orange,label="qT") # points move from black to orange
    Plots.plot!(pp, repeat(xobsTcomp1,2), repeat(xobsTcomp2,2),seriestype=:path, color=:orange,label="")

    if showmomenta
        Plots.plot!(pp, extractcomp(s,1), extractcomp(s,2),seriestype=:scatter ,
         color=:blue,label="p0 est") # points move from black to orange)
        Plots.plot!(pp, extractcomp(s0,1), extractcomp(s0,2),seriestype=:scatter ,
          color=:red,label="p0",markersize=5) # points move from black to orange)
          xlims!(-3,3)
          ylims!(-4,3)
    else
        xlims!(-3,3)
        ylims!(-2,3)
    end


    outg = [Any[X.tt[i], [X.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(X.tt) ][:]
    dfg = DataFrame(time=extractcomp(outg,1),pos1=extractcomp(outg,2),pos2=extractcomp(outg,3),mom1=extractcomp(outg,4),mom2=extractcomp(outg,5),pointID=extractcomp(outg,6))
    for j in 1:n
        #global pp
        el1 = dfg[:pointID].=="point"*"$j"
        dfg1 = dfg[el1,:]
        Plots.plot!(pp,dfg1[:pos1], dfg1[:pos2],label="")
    end

    pp2 = Plots.plot(collect(1:i), objvals[1:i],seriestype=:scatter ,color=:blue,markersize=1.5,label="",title="Loglikelihood approximation")
    Plots.plot!(pp2, collect(1:i), objvals[1:i] ,color=:blue,label="")
    xlabel!(pp2,"iteration")
    ylabel!(pp2,"stoch log likelihood")
    xlims!(0,ITER)

    l = @layout [a  b]
    Plots.plot(pp,pp2,background_color = :ivory,layout=l , size = (900, 500) )

    plotlandmarkpositions(Xf,X,P.n,model,xobs0,xobsT,nfs,db=2.6)
end


cd("/Users/Frank/.julia/dev/Bridge/landmarks/figs")
fn = "me"*"_" * string(model) * "_" * string(sampler) *"_" * string(dataset)
gif(anim, fn*".gif", fps = 20)
mp4(anim, fn*".mp4", fps = 20)

print(objvals)
#end

sc2 = Plots.plot(collect(1:ITER), objvals,seriestype=:scatter ,color=:blue,markersize=1.2,label="",title="Loglikelihood approximation")
Plots.plot!(sc2, collect(1:ITER), objvals ,color=:blue,label="")
xlabel!(sc2,"iteration")
ylabel!(sc2,"stoch log likelihood")
xlims!(0,ITER)
display(sc2)
png(sc2,"stochlogp.png")


error("STOP HERE")












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


function obj2(xinitv,pars)
    Q = put_targetpars(pars,Q)
    Q = put_auxpars(pars,Q)
    xinit = deepvec2state(xinitv)
    sample!(WW, Wiener{Vector{StateW}}())
    Xᵒ = Bridge.solve(EulerMaruyama!(), xinit, WW, Q)
    (
    (lptilde(vec(xinit  ), L0, M0⁺, μ0, V, Q) - lptilde(vec(x0), L0, M0⁺, μ0, V, Q))
     + llikelihood(LeftRule(), Xᵒ, Q; skip = 1)
    )
end



if false
    # write mcmc iterates to csv file

    fn = outdir*"iterates.csv"
    f = open(fn,"w")
    head = "iteration, time, component, value \n"
    write(f, head)
    iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:1, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
    writecsv(f,iterates)
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
