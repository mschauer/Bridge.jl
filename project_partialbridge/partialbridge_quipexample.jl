# reminder, to type H*, do H\^+
cd("/Users/Frank/.julia/dev/Bridge")
outdir="/Users/Frank/Sync/DOCUMENTS/onderzoek/code/diffbridges/out_quipexample/"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV

T = 1.0
dt = 1/5000
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood

νHparam = false#false
generate_data = false

# settings in case of νH - parametrisation
ϵ = 0#10^(-3)
Σdiagel = 10^(-10)

# settings sampler
iterations = 5*10^3
skip_it = 10 # 1000
subsamples = 0:skip_it:iterations

L = @SMatrix [1.0]

ρ = 0.5
#v = ℝ{1}(0.5pi)
v = ℝ{1}(0.75pi)

m, d = size(L)
Σ = SMatrix{m,m}(Σdiagel*I)

# specify target process b(x) = α - ω sin (8x), σ = 1/2
struct NclarDiffusion <: ContinuousTimeProcess{ℝ{1}}
    α::Float64
    ω::Float64
    σ::Float64
end

Bridge.b(t, x, P::NclarDiffusion) = ℝ{1}(P.α - P.ω * sin(8* x[1]))
Bridge.σ(t, x, P::NclarDiffusion) = ℝ{1}(P.σ)
Bridge.constdiff(::NclarDiffusion) = true

P = NclarDiffusion(2.0, 2.0, 0.5)
x0 = ℝ{1}(0.0)

if generate_data
     include("/Users/Frank/Sync/DOCUMENTS/onderzoek/code/diffbridges/truepaths_nclar.jl")
end

# specify auxiliary process
struct NclarDiffusionAux <: ContinuousTimeProcess{ℝ{1}}
    α::Float64
    ω::Float64
    σ::Float64
end

Random.seed!(42)
Bridge.B(t, P::NclarDiffusionAux) = @SMatrix [0.0]
Bridge.β(t, P::NclarDiffusionAux) = ℝ{1}(P.α)
Bridge.σ(t, x, P::NclarDiffusionAux) = ℝ{1}(P.σ)
Bridge.constdiff(::NclarDiffusionAux) = true
Bridge.b(t, x, P::NclarDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
Bridge.a(t, P::NclarDiffusionAux) = Bridge.σ(t,0,P) * Bridge.σ(t, 0, P)'

Pt = NclarDiffusionAux(P.α, P.ω, P.σ)

# set clock
tic()

# Solve Backward Recursion
Po = νHparam ? Bridge.PartialBridgeνH(tt, P, Pt, L, ℝ{m}(v),ϵ, Σ) : Bridge.PartialBridge(tt, P, Pt, L, ℝ{m}(v), Σ)

print(Po.M⁺[end-5:end])
####################### MH algorithm ###################
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
    #ρ = rand(Uniform(0.95,1.0))
    Wo.yy .= ρ*W.yy + sqrt(1-ρ^2)*W2.yy
    solve!(Euler(),Xo, x0, Wo, Po)

    llo = llikelihood(Bridge.LeftRule(), Xo, Po,skip=sk)
    print("ll $ll $llo, diff_ll: ",round(llo-ll,3))

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
elapsed_time = toc()

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
write(f, "Noise Sigma: ",string(Σ),"\n")
write(f, "L: ",string(L),"\n\n")
write(f,"Mesh width: ",string(dt),"\n")
write(f, "rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
write(f, "Average acceptance percentage: ",string(ave_acc_perc),"\n\n")
write(f, "Backward type parametrisation in terms of nu and H? ",string(νHparam),"\n")
close(f)


println("Average acceptance percentage: ",ave_acc_perc,"\n")
println("Parametrisation of nu and H? ", νHparam)
println("Elapsed time: ",elapsed_time)
