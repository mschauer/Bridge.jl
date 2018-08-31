using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles


T = 2.0
dt = 1/100

tt = 0.:dt:T
struct IntegratedDiffusion <: ContinuousTimeProcess{ℝ{2}}
    γ::Float64
end

βu(t, x::Float64, P::IntegratedDiffusion) = - (x+sin(x)) + 1/2
Bridge.b(t, x, P::IntegratedDiffusion) = ℝ{2}(x[2], βu(t, x[2], P))
Bridge.σ(t, x, P::IntegratedDiffusion) = ℝ{2}(0.0, P.γ)

Bridge.constdiff(::IntegratedDiffusion) = true

struct IntegratedDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    γ::Float64
end

βu(t, x::Float64, P::IntegratedDiffusionAux) = -x + 1/2
Bridge.b(t, x, P::IntegratedDiffusionAux) = ℝ{2}(x[2], βu(t, x[2], P))
Bridge.σ(t, x, P::IntegratedDiffusionAux) = ℝ{2}(0.0, P.γ)

Bridge.B(t, P::IntegratedDiffusionAux) = @SMatrix [0.0 1.0; 0.0 -1.0]
Bridge.β(t, P::IntegratedDiffusionAux) = ℝ{2}(0, 1/2)
Bridge.a(t, P::IntegratedDiffusionAux) = @SMatrix [0.0 0.0; 0.0 P.γ^2]

Bridge.constdiff(::IntegratedDiffusionAux) = true

# Generate Data
Random.seed!(1)

P = IntegratedDiffusion(0.7)
Pt = IntegratedDiffusionAux(0.7)

W = sample(tt, Wiener())
x0 = ℝ{2}(2.0, 1.0)
X = solve(Euler(), x0, W, P)

L = @SMatrix [1. 0.]
Σ = @SMatrix [0.0]
v = ℝ{1}(2.5)    #ℝ{1}(X.yy[end][1] + √Σ[1,1]*randn())

# Solve Backward Recursion

S2 = typeof(L)
S = typeof(L*L')
T = typeof(diag(L*L'))

N = length(tt)
Lt = zeros(S2, N)
M⁺t = zeros(S, N)
μt = zeros(T, N) 

Bridge.partialbridgeode!(Bridge.R3(), tt, L, Σ, Lt, M⁺t, μt, Pt)

j = 10

@test norm((μt[j+1] - μt[j])/dt - (-Lt[j+1]*Bridge.β(tt[j+1], Pt))) < 0.01
@test norm((M⁺t[j+1] - M⁺t[j])/dt - (-Lt[j+1]*Bridge.a(tt[j+1], Pt)*Lt[j+1]')) < 0.01

Po = Bridge.PartialBridge(tt, P, Pt, L, v, Σ)

@test Po.L == Lt

W = sample(tt, Wiener())
x0 = ℝ{2}(2.0, 1.0)
Xo = copy(X)
bridge!(Xo, x0, W, Po)


# Likelihood

ll = llikelihood(Bridge.LeftRule(), Xo, Po)


# parameter

iterations = 10000
subsamples = 0:100:iterations
ρ = 0.9


# initalization
sample!(W, Wiener())
bridge!(X, x0, W, Po)
ll = llikelihood(Bridge.LeftRule(), X, Po)


Wo = copy(W)
W2 = copy(W)

XX = Any[]
if 0 in subsamples
    push!(XX, copy(X))
end
for iter in 1:iterations
    # Proposal
    sample!(W2, Wiener())
    Wo.yy .= ρ*W.yy + sqrt(1-ρ^2)*W2.yy

    
    
    bridge!(Xo, x0, Wo, Po)
    llo = llikelihood(Bridge.LeftRule(), Xo, Po)
    print("ll $ll $llo ")#, X[10], " ", Xo[10])
    if log(rand()) <= llo - ll
        X.yy .= Xo.yy
        W.yy .= Wo.yy
        ll = llo
        print("✓")
    end    
    println()
    if iter in subsamples
        push!(XX, copy(X))
    end
end



#writedlm("/Users/Frank/Dropbox/DiffBridges/Rcode/integrated_diff/integrateddiff1.csv", [first(XX[i].yy[j]) for i in 1:iterations, j in 1:length(X)])
#writedlm("/Users/Frank/Dropbox/DiffBridges/Rcode/integrated_diff/integrateddiff2.csv", [last(XX[i].yy[j]) for i in 1:iterations, j in 1:length(X)])





# write mcmc iterates to csv file 
f = open("/Users/Frank/Dropbox/DiffBridges/Rcode/integrated_diff/iterates.csv","w")
head = "iteration, time, component, value \n"
write(f, head)
iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:2, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
writecsv(f,iterates)
close(f)

