using Bridge, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models

include("partialparam.jl")
x0 = Vector(x0)
L = Matrix(L)
Σ = Matrix(Σ)
v = Vector(v)


# Generate Data
Random.seed!(1)

P = PBIntegratedDiffusion(γ)
Pt = PBIntegratedDiffusionAux(γ)

W1 = sample(tt, Wiener())

X1 = solve(EulerMaruyama!(), x0, W1, P)





# Solve Backward Recursion

N = length(tt)
νt = [zero(x0) for i in 1:N]
Σt = [zero(Bridge.outer(x0)) for i in 1:N]


Bridge.partialbridgeode!(Bridge.R3!(), tt, L, Σ, v, νt, Σt, Pt, ϵ)


j = length(tt)÷2
B = [0.0 1.0; 0.0 -1.0]
A = Bridge.a(NaN, Pt)
@test norm((νt[j+1] - νt[j])/dt - (Bridge.b(tt[j+1],  νt[j], Pt))) < 0.01
j = length(tt) - 3
@test norm((B*Σt[j+1] + Σt[j+1]*B' - Bridge.a(tt[j+1], Pt)) - Bridge.dP!(tt[j+1], Σt[j+1], copy(Σt[j+1]), Pt)) < 0.01
#@test_broken norm((Σt[j+1] - Σt[j])/dt - (B*Σt[j+1] + Σt[j+1]*B' - Bridge.a(tt[j+1], Pt))) < 0.01

Po1 = Bridge.PartialBridge!(tt, P, Pt, L, v, ϵ, Σ)

@test Po1.H == inv.(Σt)


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
    mcmc1(x0, tt, Po1)

end
