using Bridge, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models

T = 2.0
dt = 1/1000

tt = 0.:dt:T
struct PBIntegratedDiffusion <: ContinuousTimeProcess{Float64}
    γ::Float64
end
struct PBIntegratedDiffusionAux <: ContinuousTimeProcess{Float64}
    γ::Float64
end

PorPtilde = Union{PBIntegratedDiffusion, PBIntegratedDiffusionAux}


βu(t, x::Float64, P::PBIntegratedDiffusion) = - (x+sin(x)) + 1/2
βu(t, x::Float64, P::PBIntegratedDiffusionAux) = -x + 1/2
# not really a 'beta'

Bridge.b(t::Float64, x, P::PorPtilde) = Bridge.b!(t, x, copy(x), P)
function Bridge.b!(t, x, out, P::PorPtilde)
    out[1] = x[2]
    out[2] = βu(t, x[2], P)
    out
end

function Bridge.σ!(t, x, dm, out, P::PorPtilde)
    out[1] = 0.0
    out[2] = dm*P.γ
    out
end
function Bridge.a(t, P::PorPtilde)
    [0.0 0.0; 0.0 P.γ^2]
end
Bridge.a(t, x, P::PorPtilde) = Bridge.a(t, P::PorPtilde)

Bridge.constdiff(::PorPtilde) = true

function Bridge.B!(t, arg, out, P::PBIntegratedDiffusionAux)
    B = [0.0 1.0; 0.0 -1.0]
    out .= (B*arg)
    out
end
function BBt!(t, arg, out, P::PBIntegratedDiffusionAux)
    B = [0.0 1.0; 0.0 -1.0]
    out .= (B*arg + arg*B')
    out
end

function Bridge.dP!(t, p, out, P)
    BBt!(t, p, out, P)
    out[2,2] -= P.γ^2
    out
end

# Generate Data
Random.seed!(1)

P = PBIntegratedDiffusion(0.7)
Pt = PBIntegratedDiffusionAux(0.7)

W1 = sample(tt, Wiener())
x0 = [2.0, 1.0]
X1 = solve(EulerMaruyama!(), x0, W1, P)


L = [1. 0.]
Σnoise = fill(0.01, 1, 1)
v = [2.5]

# Solve Backward Recursion

N = length(tt)
νt = [zero(x0) for i in 1:N]
Σt = [zero(Bridge.outer(x0)) for i in 1:N]
ϵ = 0.02

Bridge.partialbridgeode!(Bridge.R3!(), tt, L, Σnoise, v, νt, Σt, Pt, ϵ)


j = length(tt)÷2
B = [0.0 1.0; 0.0 -1.0]
A = Bridge.a(NaN, Pt)
@test norm((νt[j+1] - νt[j])/dt - (Bridge.b(tt[j+1],  νt[j], Pt))) < 0.01
j = length(tt) - 3
@test norm((B*Σt[j+1] + Σt[j+1]*B' - Bridge.a(tt[j+1], Pt)) - Bridge.dP!(tt[j+1], Σt[j+1], copy(Σt[j+1]), Pt)) < 0.01
@test_broken norm((Σt[j+1] - Σt[j])/dt - (B*Σt[j+1] + Σt[j+1]*B' - Bridge.a(tt[j+1], Pt))) < 0.01

Po1 = Bridge.PartialBridge!(tt, P, Pt, L, v, ϵ, Σnoise)

@test Po1.H == inv.(Σt)


x0 = [2.0, 1.0]
Xo1 = deepcopy(X1)

solve!(Bridge.EulerMaruyama!(), Xo1, x0, W1, Po1)


# Likelihood

ll1 = llikelihood(Bridge.LeftRule(), Xo1, Po1)


lls1 = Float64[]

function mcmc1(x0, tt, Po2)
    # MCMC parameter

    iterations = 1000
    subsamples = 0:100:iterations
    ρ = 0.9


    # initalization
    W = sample(tt, Wiener())
    X = solve(EulerMaruyama!(), x0, W, Po1)
    ll = llikelihood(Bridge.LeftRule(), X, Po1)

    acc = 0

    Wo = deepcopy(W)
    Wrho = deepcopy(W)

    Xo = deepcopy(X)

    @assert !(X.yy[1] === Xo.yy[1])

    XX = Any[]
    if 0 in subsamples
        push!(XX, deepcopy(X))
    end
    for iter in 1:iterations
        # Proposal
        sample!(Wrho, Wiener())
        Wo.yy .= ρ*W.yy + sqrt(1-ρ^2)*Wrho.yy


        solve!(EulerMaruyama!(), Xo, x0, Wo, Po1)
        llo = llikelihood(Bridge.LeftRule(), Xo, Po1)
        push!(lls1, llo)

        if log(rand()) <= llo - ll

            X, Xo = Xo, X
            W, Wo = Wo, W
            ll = llo
            acc += 1
        end
        if iter in subsamples
            push!(XX, deepcopy(X))
        end
    end


end

@time @testset "MCMC1" begin
    mcmc1(x0, tt, Po2)

end
