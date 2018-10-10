# reminder, to type H⁺, do H\^+

mkpath("output/out_nclar")
outdir="output/out_nclar/"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV

T = 0.5
dt = 1/5000
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood

obs_scheme =["full", "firstcomponent"][2]
easy_conditioning = false # if true, then path to 1, else to 2

νHparam = false
generate_data = false

# settings in case of νH - parametrisation
ϵ = 10^(-3)
Σdiagel = 10^(-10)

# settings sampler
iterations = 5*10^4
skip_it = 1000
subsamples = 0:skip_it:iterations

ρ = obs_scheme=="full" ? 0.85 : 0.95

if obs_scheme=="full"
    L = SMatrix{3, 3}(1.0I)
#    Σ = SMatrix{3, 3}(0.0I)
    v = easy_conditioning ?  ℝ{3}(1/32, 1/4, 1) :  ℝ{3}(5/128, 3/8, 2)
end
if obs_scheme=="firstcomponent"
    L = @SMatrix [1. 0. 0.]
#    Σ = @SMatrix [0.0]
    v = easy_conditioning ? 1/32 : 5/128
end

m, d = size(L)
Σ = SMatrix{m, m}(Σdiagel*I)

# specify target process
struct NclarDiffusion <: ContinuousTimeProcess{ℝ{3}}
    α::Float64
    ω::Float64
    σ::Float64
end

Bridge.b(t, x, P::NclarDiffusion) = ℝ{3}(x[2], x[3], -P.α * sin(P.ω * x[3]))
Bridge.σ(t, P::NclarDiffusion) = ℝ{3}(0.0, 0.0, P.σ)
Bridge.σ(t, x, P::NclarDiffusion) = Bridge.σ(t, P)
Bridge.constdiff(::NclarDiffusion) = true

P = NclarDiffusion(2*3.0, 2pi, 1.0)
x0 = ℝ{3}(0.0, 0.0, 0.0)

if generate_data
     include("/Users/Frank/Sync/DOCUMENTS/onderzoek/code/diffbridges/truepaths_nclar.jl")
end

# specify auxiliary process
struct NclarDiffusionAux <: ContinuousTimeProcess{ℝ{3}}
    α::Float64
    ω::Float64
    σ::Float64
end

Random.seed!(42)
Bridge.B(t, P::NclarDiffusionAux) = @SMatrix [0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 0.0 0.0 0.0]
Bridge.β(t, P::NclarDiffusionAux) = ℝ{3}(0.0, 0.0, 0)
Bridge.σ(t, P::NclarDiffusionAux) = ℝ{3}(0.0, 0.0, P.σ)
Bridge.σ(t, x, P::NclarDiffusionAux) = Bridge.σ(t, P)
Bridge.constdiff(::NclarDiffusionAux) = true
Bridge.b(t, x, P::NclarDiffusionAux) = Bridge.B(t, P) * x + Bridge.β(t, P)
Bridge.a(t, P::NclarDiffusionAux) = Bridge.σ(t, 0, P) * Bridge.σ(t, 0, P)'

Pt = NclarDiffusionAux(P.α, P.ω, P.σ)

# Solve Backward Recursion
Po = νHparam ? Bridge.PartialBridgeνH(tt, P, Pt, L, ℝ{m}(v), ϵ, Σ) : Bridge.PartialBridge(tt, P, Pt, L, ℝ{m}(v), Σ)

####################### MH algorithm ###################
W = sample(tt, Wiener())
X = solve(Euler(), x0, W, P)
Xo = copy(X)
solve!(Euler(),Xo, x0, W, Po)

solve!(Euler(),X, x0, W, Po)

# further initialisation
Wo = copy(W)
W2 = copy(W)
XX = Any[]
if 0 in subsamples
    push!(XX, copy(X))
end


let
    ll = llikelihood(Bridge.LeftRule(), X, Po, skip=sk)
    acc = 0
    for iter in 1:iterations
        # Proposal
        sample!(W2, Wiener())
        #ρ = rand(Uniform(0.95, 1.0))
        Wo.yy .= ρ*W.yy + sqrt(1-ρ^2)*W2.yy
        solve!(Euler(),Xo, x0, Wo, Po)

        llo = llikelihood(Bridge.LeftRule(), Xo, Po, skip=sk)
        print("ll $ll $llo, diff_ll: ", round(llo - ll, digits = 3))

        if log(rand()) <= llo - ll
            X.yy .= Xo.yy
            W.yy .= Wo.yy
            ll = llo
            print("✓")
            acc += 1
        end
        println()
        if iter in subsamples
            push!(XX, copy(X))
        end
    end
end

@info "Done."*"\x7"^6

# write mcmc iterates to csv file

fn = outdir*"iterates-"*obs_scheme*".csv"
f = open(fn, "w")
head = "iteration, time, component, value\n"
write(f, head)
iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:3, j in 1:length(X), (i, s) in enumerate(subsamples) ][:]
writedlm(f, iterates, ',')
close(f)

ave_acc_perc = 100*round(acc/iterations, digits=2)

# write info to txt file
fn = outdir*"info-"*obs_scheme*".txt"
f = open(fn, "w")
write(f, "Choice of observation schemes: ", obs_scheme, "\n")
write(f, "Easy conditioning (means going up to 1 for the rough component instead of 2): ", string(easy_conditioning), "\n")
write(f, "Number of iterations: ", string(iterations), "\n")
write(f, "Skip every ", string(skip_it), " iterations, when saving to csv", "\n\n")
write(f, "Starting point: ", string(x0), "\n")
write(f, "End time T: ", string(T), "\n")
write(f, "Endpoint v: ", string(v), "\n")
write(f, "Noise Sigma: ", string(Σ), "\n")
write(f, "L: ", string(L), "\n\n")
write(f, "Mesh width: ", string(dt), "\n")
write(f, "rho (Crank-Nicholsen parameter: ", string(ρ), "\n")
write(f, "Average acceptance percentage: ", string(ave_acc_perc), "\n\n")
write(f, "Backward type parametrisation in terms of nu and H? ", string(νHparam), "\n")
close(f)


println("Average acceptance percentage: ", ave_acc_perc, "\n")
println(obs_scheme)
println("Parametrisation of nu and H? ", νHparam)
