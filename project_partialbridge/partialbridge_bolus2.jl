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
Σdiagel = 10^(-5)


# settings sampler
iterations = 500
skip_it = 50# 1000
subsamples = 0:skip_it:iterations

ρ = 0.0#95

L = @SMatrix [.5 .5 ]
#L = @SMatrix [1. 0. ; 0.0 1.0]

m, d = size(L)
Σ = SMatrix{m,m}(Σdiagel*I)
dp = 2 # dprime

# specify target process
struct Diffusion <: ContinuousTimeProcess{ℝ{2}}
    α::Float64
    β::Float64
    λ::Float64
    k::Float64
    σ1::Float64
    σ2::Float64
end

# pdfemg(x, μ, σ, λ) = λ/2*exp(λ/2*(2μ + λ*σ^2 - 2x)).*erfc((μ + λ*σ^2 - x)/(sqrt(2)*σ))
# dose(t, c) = pdfemg(t, c...)*c[3]
# dose(t, c) = 1. *(t > c)


Bridge.b(t, x, P::Diffusion) = ℝ{2}(P.α*dose(t) -(P.λ + P.β)*x[1] + (P.k-P.λ)*x[2],  P.λ*x[1] - (P.k-P.λ)*x[2])
#Bridge.b(t, x, P::Diffusion) = ℝ{2}(0.0 , 0.0)#P.α*dose(t)
Bridge.σ(t, x, P::Diffusion) = @SMatrix [P.σ1 0.0 ;0.0  P.σ1]
Bridge.constdiff(::Diffusion) = true


# specify auxiliary process
struct DiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    α::Float64
    β::Float64
    λ::Float64
    k::Float64
    #c::Float64
    σ1::Float64
    σ2::Float64
end

Random.seed!(42)
Bridge.B(t, P::DiffusionAux) = @SMatrix [ -P.λ - P.β P.k-P.λ ;  P.λ  P.λ-P.k]
Bridge.β(t, P::DiffusionAux) = ℝ{2}(P.α*dose(t),0.0)

#Bridge.B(t, P::DiffusionAux) = @SMatrix [ 0.0  0.0 ;  0.0 0.0]
#Bridge.β(t, P::DiffusionAux) = ℝ{2}(0.0,0.0)#P.α*dose(t)
Bridge.σ(t, P::DiffusionAux) = @SMatrix [P.σ1 0.0 ;0.0   P.σ2]
Bridge.constdiff(::DiffusionAux) = true
Bridge.b(t, x, P::DiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
Bridge.a(t, P::DiffusionAux) = Bridge.outer(Bridge.σ(t, P))

#dose(t) = 2*(t/20)/(1+(t/20)^2)
dose(t) = 2*(t/2)/(1+(t/2)^2)
FT = 70.; VB = 20.; PS = VE = 15.; HE = 0.4; DT = 2.4 # Favetto-Samson
DT = 1. # out choice


P = Diffusion(FT/(1-HE), FT/(VB*(1-HE)), PS/(VB*(1-HE)), PS/(VB*(1-HE)) + PS/VE, sqrt(2),0.2)
Pt = DiffusionAux(P.α, P.β,P.λ,P.k,P.σ1,P.σ2)



if simlongpath
    # Simulate one long path
    # Random.seed!(2)
    x0 = ℝ{2}(0.0, 0.0)
    #x0 = ℝ{2}(-8.0, 1.0)
    T_long = 2.0
    dt = 0.001
    tt_long = 0.:dt:T_long
    W_long = sample(tt_long, Wiener{ℝ{dp}}())
    X_long = solve(Euler(), x0, W_long, P)
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
    obsind = 1:(lt÷obsnum):lt
    obsnum = length(obsind)

    _pairs(collection) = Base.Generator(=>, keys(collection), values(collection))
    V_ = SamplePath(collect(_pairs(X_long))[obsind])
    V = SamplePath(V_.tt, map(y -> (L*y)[1] .+ (cholesky(Σ)).U' * randn(m), V_.yy))
end

obsnum = length(V)
segnum = obsnum-1

longpath = [Any[tt_long[j], d, X_long.yy[j][d]] for d in 1:2, j in 1:5:length(X_long) ][:]
obs = [Any[V.tt[j], dind, V.yy[j][dind]] for dind in 1:1, j in 1:length(V) ][:]
obsDf = DataFrame(time=map(x->x[1],obs), component = map(x->x[2],obs),value=map(x->x[3],obs) )
longpathDf = DataFrame(time=map(x->x[1],longpath), component = map(x->x[2],longpath),value=map(x->x[3],longpath) )




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

tX = ℝ{d}
tW = ℝ{dp} #Float64 #ℝ{dp}# #
typeX = SamplePath{tX}
typeW = SamplePath{tW}
typePo = Bridge.PartialBridgeνH{ℝ{d},Diffusion,DiffusionAux,ℝ{d},SArray{Tuple{d,d},Float64,2,d^2}}
XX = Vector{typeX}(undef, segnum)
XX2 = Vector{typeX}(undef, segnum)
WW = Vector{typeW}(undef, segnum)
Po = Vector{typePo}(undef, segnum)
Poᵒ = Vector{typePo}(undef, segnum) # needed when parameter estimation is done


# initialise
νend = SVector{d}(zeros(d))
Hend⁺ = SMatrix{d,d}(I/ϵ)
H⁺i = Vector{typeof(Hend⁺)}(undef, segnum)

νend, Hend⁺ = gpupdate(νend, Hend⁺ , Σ,L,V.yy[end])

dtimp = 0.0001  # mesh width for imputed paths

τ(t, T0, Tend) = T0 +(t-T0)*(2-(t-T0)/(Tend-T0))

# solve backward recursion on [0,T]

for i in segnum:-1:1
    # update on interval (t[i-1],t[i])
    tt_ = τ.(V.tt[i]:dtimp:V.tt[i+1],V.tt[i],V.tt[i+1])
    XX[i] = Bridge.samplepath(tt_, zero(tX)) # initialise
    XX2[i] = Bridge.samplepath(tt_, zero(tX)) # initialise
    WW[i] = Bridge.samplepath(tt_, zero(tW)) # initialise
    H⁺i[i] = Hend⁺
    Po[i], νend, Hend⁺ = Bridge.partialbridgeνH(tt_, P, Pt, νend, Hend⁺)
    Poᵒ[i] = Po[i]
    global νend, Hend⁺ = gpupdate(νend, Hend⁺, Σ, L, V.yy[i])
end
H⁺i[1] = Hend⁺

#elapsed_time= @elapsed begin

####################### MH algorithm ###################

# Initialisation, forward simulate on [0,T] a bridge
xstart = νend #+ √(Hend⁺) * randn(d)  # note that this is really ν(0)
for i in 1:segnum
    tt = Po[i].tt
    WW[i] = sample(tt,Wiener{tW}())
    Bridge.solve!(Euler(), XX[i], xstart, WW[i], Po[i])
    xstart = XX[i].yy[end] # starting point for next segment
end

XXinit = vcat(XX...)
longpath = [Any[tt_long[j], d, X_long.yy[j][d]] for d in 1:2, j in 1:5:length(X_long) ][:]
initpath = [Any[XXinit.tt[j], d, XXinit.yy[j][d]] for d in 1:2, j in 1:5:length(XXinit) ][:]
obs = [Any[V.tt[j], dind, V.yy[j][dind]] for dind in 1:1, j in 1:length(V) ][:]
obsDf = DataFrame(time=map(x->x[1],obs), component = map(x->x[2],obs),value=map(x->x[3],obs) )
longpathDf = DataFrame(time=map(x->x[1],longpath), component = map(x->x[2],longpath),value=map(x->x[3],longpath) )
initpathDf = DataFrame(time=map(x->x[1],initpath), component = map(x->x[2],initpath),value=map(x->x[3],initpath) )


@rput obsDf
@rput longpathDf
@rput initpathDf


R"""
library(ggplot2)
library(tidyverse)

longpathDf$component <- as.factor(longpathDf$component)
p <- ggplot() +
  ylab("") + geom_path(aes(x=time,y=value,colour=as.factor(component)),data=initpathDf)+
   geom_point(aes(x=time,y=value),data=obsDf,colour="red")+
  geom_path(aes(x=time,y=value,colour=component),data=longpathDf)+theme_minimal()+
  theme(legend.position="bottom")
  #facet_wrap(~component,ncol=1,scales='free_y') +

"""




XXo = deepcopy(XX)
WWo = deepcopy(WW)
#WW2 = deepcopy(WW)

# save some of the paths
XXsave = Any[]
if 0 in subsamples
    push!(XXsave, deepcopy(XX))
end

acc = 0
accparams = 0
mhsteps = 0
mhstepsparams = 0
Hrightmost⁺ = H⁺i[segnum]
νrightmost = Po[segnum].ν[end]
Hzero⁺ = SMatrix{d,d}(0.01*I)



C = [0.0]


for iter in 1:iterations

    finished = false
    klow = 1
    ind = segnum:-1:1
    kup = obsnum

    xstart = XX[1].yy[1]
    xstarto = xstart
    updateparams =  false#
    while !finished
        # update a block

        if updateparams
            ind = (segnum==2) ? (1:1) : (segnum:-1:1)
            kup = obsnum
        else
            segnum_update = sample(1:obsnum-klow) #obsnum-1   ##obsnum-1#   # number of segments to update
            kup = klow + segnum_update  # update on interval [t(klow), t(kup)]
            ind = (segnum_update==1) ? (1:1) : ((kup-1):-1:klow) # indices of segments to update
        end
        hasbegin = ind[end]==1
        hasend = ind[1]==segnum

        νend  = hasend ? νrightmost :  XX[ind[1]].yy[end]  # initialise on rightmost segment
        Hend⁺ = hasend ? Hrightmost⁺ : Hzero⁺ # initialise on rightmost segment
        νendᵒ = νend
        Hend⁺ᵒ = Hend⁺

        α = P.α
        if updateparams
            αᵒ = α + 0.01 * randn()
        else
            αᵒ = α
        end

        Pᵒ = Diffusion(αᵒ,P.β,P.λ,P.k,P.σ1,P.σ2)
        Ptᵒ = DiffusionAux(αᵒ,P.β,P.λ,P.k,P.σ1,P.σ2)
        Pt = DiffusionAux(α, P.β,P.λ,P.k,P.σ1,P.σ2)

        # compute guiding term
        for i in ind
            tt = Po[i].tt
            Po[i], νend, Hend⁺ = Bridge.partialbridgeνH(tt, P, Pt, νend, Hend⁺)
            νend, Hend⁺ = gpupdate(νend, Hend⁺, Σ, L, V.yy[i])
            if updateparams
                Poᵒ[i], νendᵒ, Hend⁺ᵒ = Bridge.partialbridgeνH(tt, Pᵒ, Ptᵒ, νendᵒ, Hend⁺ᵒ)
                νendᵒ, Hend⁺ᵒ = gpupdate(νendᵒ, Hend⁺ᵒ, Σ, L, V.yy[i])
            else
                Poᵒ[i] = Po[i]
                νendᵒ, Hend⁺ᵒ = νend, Hend⁺
            end
        end


        # simulate guided proposal
        for i in reverse(ind)
            tt = Po[i].tt
            if !updateparams
                sample!(WWo[i], Wiener{ℝ{2}}())
                WWo[i].yy .= ρ * WW[i].yy + sqrt(1-ρ^2) * WWo[i].yy
            else
                WWo[i].yy .= WW[i].yy
            end
            if updateparams
                xstarto = xstart = XX[1].yy[1]
            elseif i==1
                xstart = XX[1].yy[1]
                u = randn()
                xstarto = xstart + 0.1 * ℝ{2}(u, -u)
            else
                xstarto = xstart = XX[i-1].yy[end]
            end

            # at this point, either WW = WWo or Po == Poᵒ (if updateparams=true)
            solve!(Euler(), XX2[i], xstart, WW[i], Po[i])
            solve!(Euler(), XXo[i], xstarto, WWo[i], Poᵒ[i])
        end
        # compute loglikelihood
        diffll = 0.0
        for i in ind
            diffll += llikelihood(LeftRule(), XXo[i],  Poᵒ[i]) - llikelihood(LeftRule(), XX2[i],  Po[i])
        end
        if hasbegin
             diffll += logpdfnormal(xstarto-νendᵒ, Bridge.symmetrize(Hend⁺ᵒ))-logpdfnormal(xstart-νend, Bridge.symmetrize(Hend⁺))               # plus possibly log q(X0|X0o) = log q(X0o|X0)
        end
        if updateparams
            diffll += -(V.tt[end]-V.tt[1]) *(trace(Bridge.B(0.0, Ptᵒ)) - trace(Bridge.B(0.0,Pt))) +
            logpdf(Gamma(2,1),cᵒ) - logpdf(Gamma(2,1),c)
        end

        print("iter  diff_ll: ",round(diffll, digits=3))
        # MH step
        if log(rand()) <= diffll
            print("✓")
            for i in ind
                 XX[i], XXo[i] = XXo[i], XX[i]
                 WW[i], WWo[i] = WWo[i], WW[i]
            end
            P = Pᵒ
            Pt = Ptᵒ
            push!(C, αᵒ)
            if updateparams
                accparams += 1
            else
                acc += 1

            end
        end
        klow = kup # adjust counter
        finished = (klow==obsnum) # all segments have been updated (this is correct when a single sweep is done for parameter update)
        mhsteps += !updateparams
        mhstepsparams += updateparams
    end
    println()
    if iter in subsamples
        push!(XXsave, deepcopy(XX))
    end






end

@info "Done."*"\x7"^6


ave_acc_perc = 100*round(acc/mhsteps, digits=2)

limp = length(vcat(XXsave[1]...))
iterates = [Any[s,  vcat(XXsave[i]...).tt[j], dind, vcat(XXsave[i]...).yy[j][dind]] for dind in 1:d, j in 1:10:limp, (i,s) in enumerate(subsamples) ][:]
iteratesaverage = [Any[s,  vcat(XXsave[i]...).tt[j], mean(vcat(XXsave[i]...).yy[j])] for j in 1:10:limp, (i,s) in enumerate(subsamples) ][:]

write2csv = false
if write2csv
    # write long path to csv file
    f = open(outdir*"longforward_pendulum.csv","w")
    headl = "time, component, value \n"
    write(f, headl)
    writedlm(f, longpath, ',')
    close(f)
    # write mcmc iterates to csv file
    fn = outdir*"iterates-"*obs_scheme*".csv"
    f = open(fn,"w")
    headl = "iteration, time, component, value \n"
    write(f, headl)
    writedlm(f,iterates,",")
    close(f)
    # write observations to csv file
    fn = outdir*"observations.csv"
    f = open(fn,"w")
    headl = "time, component, value \n"
    write(f, headl)
    writedlm(f,obs,",")
    close(f)
end

println("Average acceptance percentage: ",ave_acc_perc,"\n")


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
