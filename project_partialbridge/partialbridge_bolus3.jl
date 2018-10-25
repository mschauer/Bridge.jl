# reminder, to type H*, do H\^+
#cd("/Users/Frank/.julia/dev/Bridge")
outdir="output/bolus/"

using Bridge, StaticArrays, Distributions
using Bridge:logpdfnormal
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using RCall

sk = 0 # skipped in evaluating loglikelihoodν

νHparam = true
simlongpath = true
obs_scheme =["full","firstcomponent"][2]

# settings in case of νH - parametrisation
ϵ = 10^(-3)
Σdiagel = 10^(-4)

# settings sampler
iterations = 25000
skip_it = 1000# 1000
subsamples = 0:skip_it:iterations

ρ = 0.0#95

L = @SMatrix [.5 .5 ]
#L = @SMatrix [1. 0. ; 0.0 1.0]

m, d = size(L)
Σ = SMatrix{m,m}(Σdiagel*I)
dp = 2 # dprime

################## specify target process
struct Diffusion <: ContinuousTimeProcess{ℝ{2}}
    α::Float64
    β::Float64
    λ::Float64
    μ::Float64
    σ1::Float64
    σ2::Float64
end

# pdfemg(x, μ, σ, λ) = λ/2*exp(λ/2*(2μ + λ*σ^2 - 2x)).*erfc((μ + λ*σ^2 - x)/(sqrt(2)*σ))
# dose(t, c) = pdfemg(t, c...)*c[3]
# dose(t, c) = 1. *(t > c)

Bridge.b(t, x, P::Diffusion) = ℝ{2}(P.α*dose(t) -(P.λ + P.β)*x[1] + P.μ*x[2],  P.λ*x[1] -P.μ*x[2])  # reminder mu = k-lambda
Bridge.σ(t, x, P::Diffusion) = @SMatrix [P.σ1 0.0 ;0.0  P.σ1]
Bridge.constdiff(::Diffusion) = true

################## specify auxiliary process
struct DiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    α::Float64
    β::Float64
    λ::Float64
    μ::Float64
    σ1::Float64
    σ2::Float64
end

#Random.seed!(42)
Bridge.B(t, P::DiffusionAux) = @SMatrix [ -P.λ - P.β P.μ ;  P.λ  -P.μ]
Bridge.β(t, P::DiffusionAux) = ℝ{2}(P.α*dose(t),0.0)
Bridge.σ(t, P::DiffusionAux) = @SMatrix [P.σ1 0.0 ;0.0   P.σ2]
Bridge.constdiff(::DiffusionAux) = true
Bridge.b(t, x, P::DiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
Bridge.a(t, P::DiffusionAux) = Bridge.outer(Bridge.σ(t, P))
DiffusionAux(P::Diffusion) = DiffusionAux(P.α, P.β,P.λ,P.μ,P.σ1,P.σ2)


dose(t) = 2*(t/2)/(1+(t/2)^2)
FT = 70.; VB = 20.; PS = VE = 15.; HE = 0.4; DT = 2.4 # Favetto-Samson
DT = 1. # out choice
#Ptrue = Diffusion(FT/(1-HE), FT/(VB*(1-HE)), PS/(VB*(1-HE)),  PS/VE, sqrt(2),0.2)
Ptrue = Diffusion(FT/(1-HE), FT/(VB*(1-HE)), PS/(VB*(1-HE)), 1.5* PS/VE, sqrt(2),0.2)

################## simulate discrete time data
if simlongpath
    # Simulate one long path
    # Random.seed!(2)
    x0 = ℝ{2}(0.0, 0.0)
    #x0 = ℝ{2}(-8.0, 1.0)
    T_long = 8.0#10.0
    dt = 0.0001
    tt_long = 0.:dt:T_long
    W_long = sample(tt_long, Wiener{ℝ{dp}}())
    X_long = solve(Euler(), x0, W_long, Ptrue)
    # Extract partial observations
    lt = length(tt_long)
    # obsnum = 10
    # if obsnum > 2
    #     obsind = sort(sample(2:lt-1,obsnum-2,replace=false))
    #     pushfirst!(obsind,1)
    #     push!(obsind,lt)
    # elseif obsnum==2
    #     obsind = [1, lt]
    # else
    #     error("provide valid number of observations ")
    # end
    obsnum = 10
    if obsnum > 2
        obsind = 1:(lt÷obsnum):lt
        obsnum = length(obsind)
    elseif obsnum==2
         obsind = [1, lt]
    else
         error("provide valid number of observations ")
     end
    _pairs(collection) = Base.Generator(=>, keys(collection), values(collection))
    V_ = SamplePath(collect(_pairs(X_long))[obsind])
    V = SamplePath(V_.tt, map(y -> (L*y)[1:m] .+ (cholesky(Σ)).U' * randn(m), V_.yy))
end

obsnum = length(V)
segnum = obsnum-1

longpath = [Any[tt_long[j], d, X_long.yy[j][d]] for d in 1:2, j in 1:5:length(X_long) ][:]
obs = [Any[V.tt[j], dind, V.yy[j][dind]] for dind in 1:1, j in 1:length(V) ][:]
obsDf = DataFrame(time=map(x->x[1],obs), component = map(x->x[2],obs),value=map(x->x[3],obs) )
longpathDf = DataFrame(time=map(x->x[1],longpath), component = map(x->x[2],longpath),value=map(x->x[3],longpath) )

################## define gpudate function
function gpupdate(ν::SVector, H⁺::SMatrix, Σ, L, v)
    if all(diag(H⁺) .== Inf)
        H⁺_ = SMatrix(inv(L' * inv(Σ) * L))
        V_ = (L' * inv(Σ) * L)\(L' * inv(Σ) *  v)
        return V_, H⁺_
    else
        Z = I - H⁺*L'*inv(Σ + L*H⁺*L')*L
        return SVector(Z*H⁺*L'*inv(Σ)*v + Z*ν), Z*H⁺
    end
end

################## initialisation
tX = ℝ{d}
tW = ℝ{dp} #Float64 #ℝ{dp}# #
typeX = SamplePath{tX}
typeW = SamplePath{tW}
typeQ = Bridge.PartialBridgeνH{ℝ{d},Diffusion,DiffusionAux,ℝ{d},SArray{Tuple{d,d},Float64,2,d^2}}
XX = Vector{typeX}(undef, segnum)
XXtemp = Vector{typeX}(undef, segnum)
WW = Vector{typeW}(undef, segnum)
Q = Vector{typeQ}(undef, segnum)
Qᵒ = Vector{typeQ}(undef, segnum) # needed when parameter estimation is done

βinit = 8.0#Ptrue.β#3.0 ##10.0 #0.7 * Ptrue.μ
σ1init = 0.5
P = Diffusion(Ptrue.α, βinit,Ptrue.λ,Ptrue.μ,σ1init,Ptrue.σ2)
Pᵒ = deepcopy(P)
Pt = DiffusionAux(P)
Ptᵒ = DiffusionAux(P)
#H⁺i = Vector{typeof(Hend⁺)}(undef, segnum)

dtimp = 0.001  # mesh width for imputed paths
τ(t, T0, Tend) = T0 +(t-T0)*(2-(t-T0)/(Tend-T0))

# solve backward recursion on [0,T]
νend = SVector{d}(zeros(d))
Hend⁺ = SMatrix{d,d}(I/ϵ)
νend, Hend⁺ = gpupdate(νend, Hend⁺, Σ, L, V.yy[end])
Hrightmost⁺ = deepcopy(Hend⁺)
νrightmost = deepcopy(νend)


for i in segnum:-1:1
    # update on interval (t[i-1],t[i])
    tt_ = τ.(V.tt[i]:dtimp:V.tt[i+1],V.tt[i],V.tt[i+1])
    #tt_ = V.tt[i]:dtimp:V.tt[i+1],V.tt[i],V.tt[i+1]
    XX[i] = Bridge.samplepath(tt_, zero(tX)) # initialise
    WW[i] = Bridge.samplepath(tt_, zero(tW)) # initialise
#    H⁺i[i] = Hend⁺
    Q[i], νend, Hend⁺ = Bridge.partialbridgeνH(tt_, P, Pt, νend, Hend⁺)
    Qᵒ[i] = Q[i]
    global νend, Hend⁺ = gpupdate(νend, Hend⁺, Σ, L, V.yy[i])
end
#H⁺i[1] = Hend⁺

#elapsed_time= @elapsed begin

# simulate forward initial path
xstart = νend #+ √(Hend⁺) * randn(d)  # note that this is really ν(0)
for i in 1:segnum
    tt = Q[i].tt
    WW[i] = sample(tt,Wiener{tW}())
    Bridge.solve!(Euler(), XX[i], xstart, WW[i], Q[i])
    xstart = XX[i].yy[end] # starting point for next segment
end

################## visualisation of generated data, discrete time data and initial path
XXinit = vcat(XX...)
longpath = [Any[tt_long[j], D, X_long.yy[j][D]] for D in 1:d, j in 1:5:length(X_long) ][:]
initpath = [Any[XXinit.tt[j], D, XXinit.yy[j][D]] for D in 1:d, j in 1:1:length(XXinit) ][:]
obs = [Any[V.tt[j], dind, V.yy[j][dind]] for dind in 1:m, j in 1:length(V) ][:]
obsDf = DataFrame(time=map(x->x[1],obs), component = map(x->x[2],obs),value=map(x->x[3],obs) )
longpathDf = DataFrame(time=map(x->x[1],longpath), component = map(x->x[2],longpath),value=map(x->x[3],longpath) )
initpathDf = DataFrame(time=map(x->x[1],initpath), component = map(x->x[2],initpath),value=map(x->x[3],initpath) )

@rput obsDf
@rput longpathDf
@rput initpathDf
R"""
library(ggplot2)
library(tidyverse)

initpathDf$component <- as.factor(initpathDf$component)
longpathDf$component <- as.factor(longpathDf$component)
p <- ggplot() +
  ylab("") + geom_path(aes(x=time,y=value,group=component),data=initpathDf)+
   geom_point(aes(x=time,y=value),data=obsDf,colour="red")+
  geom_path(aes(x=time,y=value,colour=component),data=longpathDf)+theme_minimal()+
  theme(legend.position="bottom")
  #facet_wrap(~component,ncol=1,scales='free_y')
"""

XXᵒ = deepcopy(XX)
XXtemp = deepcopy(XX)
WWᵒ = deepcopy(WW)

# save some of the paths
XXsave = Any[]
if 0 in subsamples
    push!(XXsave, deepcopy(XX))
end

acc = 0
accparams = 0
mhsteps = 0
mhstepsparams = 0

Hzero⁺ = SMatrix{d,d}(0.01*I)

param(P) = [P.β P.σ1]
logπ(P) = logpdf(Gamma(1,100),P.β) + logpdf(Gamma(1,100),P.σ1)
logq(P, Pᵒ) = 0.0
function propose(σ, P)
#    Diffusion(P.α,P.β,P.λ,P.μ + σ * randn(),P.σ1,P.σ2)
    Diffusion(P.α,P.β + σ * randn(), P.λ,P.μ,P.σ1 + σ * randn(),P.σ2)
end

################## MH-algorithm

C = [param(P)]
for iter in 1:iterations
    finished = false
    klow = 1
    ind = segnum:-1:1
    kup = obsnum

    xstart = XX[1].yy[1]
    xstartᵒ = xstart
    updateparams = rand(Bool)#false#
    while !finished
        if updateparams # update all segments
            ind = (segnum==2) ? (1:1) : (segnum:-1:1)
            kup = obsnum
        else
            segnum_update = sample(1:obsnum-klow)   # number of segments to update (could also do min(2,obsnum-klow))
            kup = klow + segnum_update  # update on interval [t(klow), t(kup)]
            ind = (segnum_update==1) ? (1:1) : ((kup-1):-1:klow) # indices of segments to update
        end
        #hasbegin = ind[end]==1
        hasend = ind[1]==segnum

        # initialise νend, Hend⁺, νendᵒ, Hend⁺ᵒ
        νend  = hasend ? νrightmost :  XX[ind[1]].yy[end]  # initialise on rightmost segment
        Hend⁺ = hasend ? Hrightmost⁺ : Hzero⁺ # initialise on rightmost segment
        νendᵒ = deepcopy(νend)
        Hend⁺ᵒ = deepcopy(Hend⁺)

        # propose new parameter value
        Pᵒ = updateparams ? propose(.02, P) : P
        Ptᵒ = DiffusionAux(Pᵒ)

        # compute guiding term
        for i in ind
            tt = Q[i].tt
            Q[i], νend, Hend⁺ = Bridge.partialbridgeνH(tt, P, Pt, νend, Hend⁺)
            if updateparams
                Qᵒ[i], νendᵒ, Hend⁺ᵒ = Bridge.partialbridgeνH(tt, Pᵒ, Ptᵒ, νendᵒ, Hend⁺ᵒ)
            else
                Qᵒ[i], νendᵒ, Hend⁺ᵒ = Q[i], νend, Hend⁺
            end
            if !(i==last(ind))  # on the left endpoint we don't need to do gupdate
                νend, Hend⁺ = gpupdate(νend, Hend⁺, Σ, L, V.yy[i])
                if updateparams
                    νendᵒ, Hend⁺ᵒ = gpupdate(νendᵒ, Hend⁺ᵒ, Σ, L, V.yy[i])
                else
                    νendᵒ, Hend⁺ᵒ = νend, Hend⁺
                end
            end
        end

        ### to get the starting point updated correctly, we need to omit the very last step of gpupdate

        global diffll = 0.0  # difference in loglikehood

        # simulate guided proposal
        for i in reverse(ind)
            #tt = Q[i].tt
            if !updateparams
                sample!(WWᵒ[i], Wiener{ℝ{dp}}())
                WWᵒ[i].yy .= ρ * WW[i].yy + sqrt(1-ρ^2) * WWᵒ[i].yy
            else
                WWᵒ[i].yy .= WW[i].yy
            end
            if i==1  # starting point
                xstart = XX[1].yy[1]
                if updateparams
                    u = randn()
                    xstartᵒ = rand(Bool) ? (xstart .+ 0.1 *  ℝ{2}(u, -u)) : xstart  # with prob 0.5 propose joint update on starting point and parameter
                else # propose new starting point
                    u = randn()
                    xstartᵒ = xstart .+ 0.1 *  ℝ{2}(u, -u)
                end
                diffll += logpdfnormal(xstartᵒ-νendᵒ, Bridge.symmetrize(Hend⁺ᵒ))-logpdfnormal(xstart-νend, Bridge.symmetrize(Hend⁺))               # plus possibly log q(X0|X0o) = log q(X0o|X0)
                # need to put  it here, as xstart and xstartᵒ change later on
            else
                xstart = XXtemp[i-1].yy[end]
                xstartᵒ = XXᵒ[i-1].yy[end]
            end
            solve!(Euler(), XXtemp[i], xstart, WW[i], Q[i])
            solve!(Euler(), XXᵒ[i], xstartᵒ, WWᵒ[i], Qᵒ[i])
        end

        # update loglikelihood
        for i in ind
            diffll += llikelihood(LeftRule(), XXᵒ[i],  Qᵒ[i]) - llikelihood(LeftRule(), XXtemp[i],  Q[i])
        end
        if updateparams # this term is still not correct
            diffll +=  (V.tt[end]-V.tt[1]) *(tr(Bridge.B(0.0, Ptᵒ)) - tr(Bridge.B(0.0,Pt)))        + logπ(Pᵒ) - logπ(P)
            #diffll +=  (V.tt[2]-V.tt[1]) *(tr(Bridge.B(0.0, Ptᵒ)) - tr(Bridge.B(0.0,Pt)))+ logπ(Pᵒ) - logπ(P)
        end

        println("ind:  ",ind,"  updateparms= ",updateparams,"  diff_ll:   ",round(diffll, digits=3))
        # MH step
        if log(rand()) <= diffll
            print("✓")
            for i in ind
                 XX[i], XXᵒ[i] = XXᵒ[i], XX[i]
                 WW[i], WWᵒ[i] = WWᵒ[i], WW[i]
            end
            P = Pᵒ
            Pt = Ptᵒ

            if updateparams
                accparams += 1
                push!(C, param(P))
            else
                acc += 1
            end
        end
        klow = kup # adjust counter
        finished = (klow==obsnum) # all segments have been updated (this is correct when a single sweep is done for parameter update)
        mhsteps += !updateparams
        mhstepsparams += updateparams
    end
    println("iteration finished")
    if iter in subsamples
        push!(XXsave, deepcopy(XX))
    end
end

@info "Done."*"\x7"^6

ave_acc_perc = 100*round(acc/mhsteps, digits=2)
ave_acc_percparams = 100*round(accparams/mhstepsparams, digits=2)
println("Average acceptance percentage: ",ave_acc_perc,"\n")
println("Average acceptance percentage params: ",ave_acc_percparams,"\n")

trueval = param(Ptrue)
est = DataFrame(iteration = 1:length(C), beta= map(x->x[1],C), sigma1=map(x->x[2],C))

@rput est
#@rput P
@rput trueval

R"""
plot.ts(est$beta,ylim=c(0,10)); abline(h=trueval[1],col='red')
lines(est$sigma1,col='blue'); abline(h=trueval[2],col='red')
"""

#error("STOP HERE")

limp = length(vcat(XXsave[1]...))
iterates = [Any[s,  vcat(XXsave[i]...).tt[j], dind, vcat(XXsave[i]...).yy[j][dind]] for dind in 1:d, j in 1:10:limp, (i,s) in enumerate(subsamples) ][:]
iteratesaverage = [Any[s,  vcat(XXsave[i]...).tt[j], mean(vcat(XXsave[i]...).yy[j])] for j in 1:10:limp, (i,s) in enumerate(subsamples) ][:]



write2csv = true
if write2csv
    # write long path to csv file
    f = open(outdir*"truepath.csv","w")
    headl = "time, component, value \n"
    write(f, headl)
    writedlm(f, longpath, ',')
    close(f)
    # write mcmc iterates to csv file
    fn = outdir*"iterates.csv"
    f = open(fn,"w")
    headl = "iteration, time, component, value \n"
    write(f, headl)
    writedlm(f,iterates,",")
    close(f)
    # also average of iterates
    fn = outdir*"iteratesaverage.csv"
    f = open(fn,"w")
    headl = "iteration, time,  value \n"
    write(f, headl)
    writedlm(f,iteratesaverage,",")
    close(f)
    # write observations to csv file
    fn = outdir*"observations.csv"
    f = open(fn,"w")
    headl = "time, component, value \n"
    write(f, headl)
    writedlm(f,obs,",")
    close(f)

    fn = outdir*"parestimates.csv"
    f = open(fn,"w")
    headl = "iterate, beta\n"
    write(f,headl)
    writedlm(f,hcat(1:length(C), C),",")
    close(f)
end


# plotting the results
iteratesDf = DataFrame(iteration = map(x->x[1],iterates), time=map(x->x[2],iterates), component = map(x->x[3],iterates),value=map(x->x[4],iterates) )
iteratesaverageDf = DataFrame(iteration = map(x->x[1],iteratesaverage), time=map(x->x[2],iteratesaverage), value = map(x->x[3],iteratesaverage) )
@rput iteratesDf
@rput iteratesaverageDf



R"""
library(ggplot2)
library(tidyverse)
iteratesDfsub <- iteratesDf #%>% filter(iteration > 5)
p <- ggplot() +
  geom_path(mapping=aes(x=time,y=value,colour=iteration,group=iteration),data=iteratesDfsub) +
  scale_colour_gradient(low='green',high='blue')+
   ylab("") + geom_point(aes(x=time,y=value),data=obsDf,colour="red")+
   geom_path(aes(x=time,y=value),data=longpathDf,colour="yellow")+
   facet_wrap(~component,ncol=1,scales='free_y') +
theme_minimal()

show(p)
"""

R"""
ggplot() +
  geom_path(mapping=aes(x=time,y=value,colour=iteration,group=iteration),data=iteratesaverageDf) +
  scale_colour_gradient(low='green',high='blue')+
   ylab("") + geom_point(aes(x=time,y=value),data=obsDf,colour="red")+
   #geom_path(aes(x=time,y=value),data=longpathDf,colour="yellow")+
   theme_minimal()
"""

writeinfo = true
if writeinfo
    # write info to txt file
    fn = outdir*"info.txt"
    f = open(fn,"w")
    #write(f, "Elapsed time: ", string(elapsed_time),"\n")
    #write(f, "Choice of observation schemes: ",obs_scheme,"\n")
    #write(f, "Easy conditioning (means going up to 1 for the rough component instead of 2): ",string(easy_conditioning),"\n")
    write(f, "Number of iterations: ",string(iterations),"\n")
    write(f, "Skip every ",string(skip_it)," iterations, when saving to csv","\n\n")
    write(f, "Starting point: ",string(x0),"\n")
    write(f, "End time T: ", string(V.tt[end]),"\n")
    #write(f, "Endpoint v: ",string(v),"\n")
    write(f, "Noise Sigma: ",string(Σ),"\n")
    write(f, "Regularisation parameter epsilon", string(ϵ),"\n")
    write(f, "L: ",string(L),"\n\n")
    write(f, "Mesh width: ",string(dtimp),"\n")
    write(f, "rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
    write(f, "Average acceptance percentage: ",string(ave_acc_perc),"\n\n")
    write(f, "Backward type parametrisation in terms of nu and H? ",string(νHparam),"\n")
    close(f)
end
