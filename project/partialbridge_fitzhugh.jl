cd("/Users/Frank/.julia/dev/Bridge")

using Revise
using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using RCall
using DataFrames
using CSV

@rlibrary ggplot2

T = 2.0
dt = 1/5000
tt = 0.:dt:T

sk = 10 # skipped in evaluating loglikelihood

generate_data = true

# settings sampler
iterations = 5*10^4
skip_it = 1000
subsamples = 0:skip_it:iterations

# specify observation scheme
L = @SMatrix [1. 0.]
Σ = @SMatrix [0.0]

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

P = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 0.3)
x0 = ℝ{2}(-0.5, -0.6)

# Generate Data
if generate_data
    Random.seed!(2)
    # generate one long path
    T_long = 10.0
    tt_long = 0.:dt:T_long
    W_long = sample(tt_long, Wiener())
    X_long = solve(Euler(), x0, W_long, P)    
    # write long forward to csv file 
    f = open("/Users/Frank/Dropbox/DiffBridges/simresults/out_fh/longforward_fh.csv","w")
    head = "time, component, value \n"
    write(f, head)
    longpath = [Any[tt_long[j], d, X_long.yy[j][d]] for d in 1:2, j in 1:length(X_long) ][:]
    writedlm(f, longpath, ',')
    close(f)         

    # simulate forwards, on the shorter interval
    Random.seed!(3)
    W = sample(tt, Wiener())
    X = solve(Euler(), x0, W, P)    
    XX = [X]
    samples = 100
    # draw more paths
    for j in 2:samples
        W = sample(tt, Wiener())
        X = solve(Euler(), x0, W, P)
        push!(XX, X)
    end

    # write forwards to csv file 
    f = open("/Users/Frank/Dropbox/DiffBridges/simresults/out_fh/forwards_fh.csv","w")
    head = "samplenr, time, component, value \n"
    write(f, head)
    iterates = [Any[i, tt[j], d, XX[i].yy[j][d]] for d in 1:2, j in 1:length(X), i in 1:samples ][:]
    writedlm(f, iterates, ',')
    close(f)         

    # write first simulated path to file
    f = open("/Users/Frank/Dropbox/DiffBridges/simresults/out_fh/firstpath_fh.csv","w")
    head = "iteration, time, component, value \n"
    write(f, head)
    iterates = [Any[1, tt[j], d, XX[1].yy[j][d]] for d in 1:2, j in 1:length(X)][:]
    writedlm(f, iterates, ',')
    close(f)         

    # write most extreme path to file   
    itrue = findmax([XX[i].yy[end][1] for i in 1:samples])[2]    # index of most extreme path
    f = open("/Users/Frank/Dropbox/DiffBridges/simresults/out_fh/extremepath_fh.csv","w")
    head = "iteration, time, component, value \n"
    write(f, head)
    iterates = [Any[itrue, tt[j], d, XX[itrue].yy[j][d]] for d in 1:2, j in 1:length(X)][:]
    writedlm(f, iterates, ',')
    close(f)         
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


for k2 in 1:2
    for k1 in 1:3
        Random.seed!(4)
        aux_choice = ["linearised_end" "linearised_startend" "matching"][k1]
        endpoint = ["first", "extreme"][k2]

        if endpoint == "first"
            v = -0.959 
        elseif endpoint == "extreme"
            v = 0.633
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

        Bridge.σ(t, x, P::FitzhughDiffusionAux) = ℝ{2}(0.0, P.σ)
        Bridge.constdiff(::FitzhughDiffusionAux) = true

        Bridge.b(t, x, P::FitzhughDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
        Bridge.a(t, P::FitzhughDiffusionAux) = Bridge.σ(t,0,P) * Bridge.σ(t, 0, P)'

        #Pt = IntegratedDiffusionAux(0.7)
        Pt = FitzhughDiffusionAux(0.1, 0.0, 1.5, 0.8, 0.3, tt[1], x0[1], tt[end], v)

        # Solve Backward Recursion
        Po = Bridge.PartialBridge(tt, P, Pt, L, ℝ{1}(v), Σ)

        # initalisation
        W = sample(tt, Wiener())
        Xo = copy(X)
        bridge!(Xo, x0, W, Po)
        sample!(W, Wiener())
        bridge!(X, x0, W, Po)
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
            bridge!(Xo, x0, Wo, Po)

            llo = llikelihood(Bridge.LeftRule(), Xo, Po,skip=sk)
            print("ll $ll $llo, diff_ll: ",round(llo-ll,3))#, X[10], " ", Xo[10])
            
            if log(rand()) <= llo - ll
                X.yy .= Xo.yy
                W.yy .= Wo.yy
                ll = llo
                print("✓")
                acc +=1 
            end    
            println()
            if iter in subsamples
                push!(XX, copy(X))
            end
        end

        @info "Done."*"\x7"^6

        # write mcmc iterates to csv file 

        fn = "/Users/Frank/Dropbox/DiffBridges/simresults/out_fh/iterates-"*endpoint*"-"*aux_choice*".csv"
        f = open(fn,"w")
        head = "iteration, time, component, value \n"
        write(f, head)
        iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:2, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
        writecsv(f,iterates)
        close(f)

        ave_acc_perc = 100*round(acc/iterations,2) 
        println("Average acceptance percentage: ",ave_acc_perc,"\n")

        # write info to txt file
        fn = "/Users/Frank/Dropbox/DiffBridges/simresults/out_fh/info-"*endpoint*"-"*aux_choice*".txt"
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
    end


    if generate_data
        # simulate true bridges by multiple times simulating forward
        #       simulate bridges forwards, on the shorter interval
        Random.seed!(3)
        W = sample(tt, Wiener())
        X = solve(Euler(), x0, W, P)    
        XX = []
        samples = 50
        s = 0
        # draw more paths
        while true
            sample!(W, Wiener())
            solve!(Euler(), X, x0, W, P)
            if norm(L*X.yy[end] .- v) < 0.001
                push!(XX, copy(X))
                s += 1
                s % 10 == 0 && println(".")
                s >= samples && break
            end
        end

        outdir="/Users/Frank/Dropbox/DiffBridges/simresults/out_fh/"
        # write forwards to csv file 
        f = open(outdir*"bridges_fh"*endpoint*".csv","w")
        head = "samplenr, time, component, value \n"
        write(f, head)
        iterates = [Any[i, tt[j], d, XX[i].yy[j][d]] for d in 1:2, j in 1:length(X), i in 1:samples ][:]
        writedlm(f, iterates, ',')
        close(f)
    end

end





