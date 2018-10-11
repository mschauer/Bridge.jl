using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models


T = 2.0
dt = 1/100

tt = 0.:dt:T
struct IntegratedDiffusion <: ContinuousTimeProcess{ℝ{2}}
    γ::Float64
end

βu(t, x::Float64, P::IntegratedDiffusion) = - (x+sin(x)) + 1/2
Bridge.b(t::Float64, x, P::IntegratedDiffusion) = ℝ{2}(x[2], βu(t, x[2], P))
Bridge.σ(t, x, P::IntegratedDiffusion) = ℝ{2}(0.0, P.γ)

Bridge.constdiff(::IntegratedDiffusion) = true

struct IntegratedDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    γ::Float64
end

βu(t, x::Float64, P::IntegratedDiffusionAux) = -x + 1/2
Bridge.b(t::Float64, x, P::IntegratedDiffusionAux) = ℝ{2}(x[2], βu(t, x[2], P))
Bridge.σ(t, P::IntegratedDiffusionAux) =  ℝ{2}(0.0, P.γ)
Bridge.σ(t, x, P::IntegratedDiffusionAux) = Bridge.σ(t, P)

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
v = ℝ{1}(2.5)

# Solve Backward Recursion

S2 = typeof(L)
S = typeof(L*L')
T = typeof(diag(L*L'))

N = length(tt)
Lt = zeros(S2, N)
Mt = zeros(S, N)
μt = zeros(T, N)

Bridge.partialbridgeode!(Bridge.R3(), tt, L, Σ, Lt, Mt, μt, Pt)

j = 10

@test norm((μt[j+1] - μt[j])/dt - (-Lt[j+1]*Bridge.β(tt[j+1], Pt))) < 0.01
@test norm((inv(Mt[j+1]) - inv(Mt[j]))/dt - (-Lt[j+1]*Bridge.a(tt[j+1], Pt)*Lt[j+1]')) < 0.01

Po = Bridge.PartialBridge(tt, P, Pt, L, v, Σ)

@test Po.L == Lt

W = sample(tt, Wiener())
x0 = ℝ{2}(2.0, 1.0)
Xo = copy(X)
solve!(Euler(), Xo, x0, W, Po)


# Likelihood

ll = llikelihood(Bridge.LeftRule(), Xo, Po)

@testset "MCMC" begin

    # MCMC parameter

    iterations = 10000
    subsamples = 0:100:iterations
    ρ = 0.9


    # initalization
    sample!(W, Wiener())
    solve!(Euler(), X, x0, W, Po)
    ll = llikelihood(Bridge.LeftRule(), X, Po)

    acc = 0

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

        solve!(Euler(), Xo, x0, Wo, Po)
        llo = llikelihood(Bridge.LeftRule(), Xo, Po)
        if log(rand()) <= llo - ll
            X.yy .= Xo.yy
            W.yy .= Wo.yy
            ll = llo
            acc += 1
        end
        if iter in subsamples
            push!(XX, copy(X))
        end
    end
    @test 1 < acc < iterations

end
