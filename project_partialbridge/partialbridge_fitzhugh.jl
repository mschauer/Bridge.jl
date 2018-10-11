mkpath("output/out_fh")
outdir="output/out_fh/"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV

T = 2.0
dt = 1/5000
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood

generate_data = false
νHparam = false

# settings in case of νH - parametrisation
ϵ = 10^(-3)
Σdiagel = 10^(-10)#10^(-10)

# settings sampler
# iterations = 5*10^4
# skip_it = 1000
# subsamples = 0:skip_it:iterations

# specify observation scheme
L = @SMatrix [1. 0.]
Σdiagel = 10^(-10)
Σ = @SMatrix [Σdiagel]

# specify target process
struct FitzhughDiffusion <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ::Float64
end

Bridge.b(t, x, P::FitzhughDiffusion) = ℝ{2}((x[1]-x[2]-x[1]^3+P.s)/P.ϵ, P.γ*x[1]-x[2] +P.β)
Bridge.σ(t, x, P::FitzhughDiffusion) = ℝ{2}(0.0, P.σ)
Bridge.constdiff(::FitzhughDiffusion) = true

P = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 0.3) # Ditlevsen-Samson
#P = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 1.0)
x0 = ℝ{2}(-0.5, -0.6)

# Generate Data
if generate_data
     include("/Users/Frank/Sync/DOCUMENTS/onderzoek/code/diffbridges/truepaths_fh.jl")
end

# specify auxiliary process
struct FitzhughDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ::Float64
    t::Float64
    u::Float64
    T::Float64
    v::Float64
end

function uv(t, P::FitzhughDiffusionAux)
    λ = (t - P.t)/(P.T - P.t)
    P.v*λ + P.u*(1-λ)
end

for k1 in (1:3)
    for k2 in (1:2)
        Random.seed!(4)# this is what i used all the time
        Random.seed!(44)
        aux_choice = ["linearised_end" "linearised_startend"  "matching"][k1]
        endpoint = ["first", "extreme"][k2]

        # settings sampler
        iterations =  !(k1==3) ? 5*10^4 : 10*10^4
        skip_it = 1000
        subsamples = 0:skip_it:iterations
        printiter = 100

        if endpoint == "first"
            #v = -0.959
            v = -1
        elseif endpoint == "extreme"
            #v = 0.633
            v = 1.1
        else
            error("not implemented")
        end

        if aux_choice=="linearised_end"
            Bridge.B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ-3*P.v^2/P.ϵ  -1/P.ϵ; P.γ -1.0]
            Bridge.β(t, P::FitzhughDiffusionAux) = ℝ{2}(P.s/P.ϵ+2*P.v^3/P.ϵ, P.β)
            ρ = endpoint=="extreme" ? 0.9 : 0.0
        elseif aux_choice=="linearised_startend"
            Bridge.B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ-3*uv(t, P)^2/P.ϵ  -1/P.ϵ; P.γ -1.0]
            Bridge.β(t, P::FitzhughDiffusionAux) = ℝ{2}(P.s/P.ϵ+2*uv(t, P)^3/P.ϵ, P.β)
            ρ = endpoint=="extreme" ? 0.98 : 0.0
        else
            Bridge.B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ  -1/P.ϵ; P.γ -1.0]
            Bridge.β(t, P::FitzhughDiffusionAux) = ℝ{2}(P.s/P.ϵ-(P.v^3)/P.ϵ, P.β)
            ρ = 0.99
        end

        Bridge.σ(t, P::FitzhughDiffusionAux) = ℝ{2}(0.0, P.σ)
        Bridge.constdiff(::FitzhughDiffusionAux) = true

        Bridge.b(t, x, P::FitzhughDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
        Bridge.a(t, P::FitzhughDiffusionAux) = Bridge.σ(t,P) * Bridge.σ(t, P)'

        Pt = FitzhughDiffusionAux(P.ϵ, P.s, P.γ, P.β, P.σ, tt[1], x0[1], tt[end], v)

        # Solve Backward Recursion
        Po = νHparam ? Bridge.PartialBridgeνH(tt, P, Pt, L, ℝ{1}(v),ϵ, Σ) : Bridge.PartialBridge(tt, P, Pt, L, ℝ{1}(v), Σ)

        ####################### MH algorithm ###################
        # initalisation
        W = sample(tt, Wiener())
        X = solve(Euler(), x0, W, P)
        Xo = copy(X)
        solve!(Euler(),Xo, x0, W, Po)

        solve!(Euler(),X, x0, W, Po)
        ll = llikelihood(Bridge.LeftRule(), X, Po,skip=sk)

        # further initialisation
        Wo = copy(W)
        W2 = copy(W)
        XX = Any[]
        if 0 in subsamples
            push!(XX, copy(X))
        end

        acc = 0

        for iter in 1:iterations
            # Proposal
            sample!(W2, Wiener())

            Wo.yy .= ρ*W.yy + sqrt(1-ρ^2)*W2.yy

            #    i0 = length(W.yy)÷4
            #    Wo.yy[1:i0] .= W.yy[1:i0]
            #    Wo.yy[i0+1:end] .= W.yy[i0] + ρ*(W.yy[i0+1:end] - W.yy[i0+1]) + sqrt(1-ρ^2)*(W2.yy[i0+1:end]-W2.yy[i0+1])

            solve!(Euler(),Xo, x0, Wo, Po)

            llo = llikelihood(Bridge.LeftRule(), Xo, Po)

            if mod(iter, printiter) == 0
                print(iter," ll $ll $llo, diff_ll: ",round(llo-ll, digits=3))#, X[10], " ", Xo[10])
            end
            accept = false
            if log(rand()) <= llo - ll
                X.yy .= Xo.yy
                W.yy .= Wo.yy
                ll = llo
                accept = true
                acc +=1
            end

            if iter in subsamples
                push!(XX, copy(X))
            end
            if mod(iter, printiter) == 0
                accept && print("✓")
                println()
            end
        end

        @info "Done."*"\x7"^6

        # write mcmc iterates to csv file

        fn = outdir*"iterates-"*endpoint*"-"*aux_choice*".csv"
        f = open(fn,"w")
        head = "iteration, time, component, value \n"
        write(f, head)
        iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:2, j in 1:50:length(X), (i,s) in enumerate(subsamples) ][:]
        #writecsv(f,iterates)
        writedlm(f,iterates,',')
        close(f)

        ave_acc_perc = 100*round(acc/iterations, digits=2)

        # write info to txt file
        fn = outdir*"info-"*endpoint*"-"*aux_choice*".txt"
        f = open(fn,"w")
        write(f, "Choice of auxiliary process: ",aux_choice,"\n")
        write(f, "Choice of endpoint: ",endpoint,"\n\n")
        write(f, "Number of iterations: ",string(iterations),"\n")
        write(f, "Skip every ",string(skip_it)," iterations, when saving to csv","\n\n")
        write(f, "Starting point: ",string(x0),"\n")
        write(f, "End time T: ", string(T),"\n")
        write(f, "Endpoint v: ",string(v),"\n")
        write(f, "Noise Sigma: ",string(Σ),"\n")
        write(f, "L: ",string(L),"\n\n")
        write(f,"Mesh width: ",string(dt),"\n")
        write(f, "rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
        write(f, "Average acceptance percentage: ",string(ave_acc_perc),"\n")
        close(f)

       println("Average acceptance percentage: ",ave_acc_perc,"\n")

    end
end

println("Parametrisation of nu and H? ", νHparam)
