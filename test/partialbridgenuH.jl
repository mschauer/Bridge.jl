using StaticArrays

include("partialbridge!.jl")

T = 2.0
dt = 1/1000

tt = 0.:dt:T
#struct PBIntegratedDiffusion <: ContinuousTimeProcess{ℝ{2}}
#    γ::Float64
#end

Bridge.b(t::Float64, x, P::PBIntegratedDiffusion) = ℝ{2}(x[2], βu(t, x[2], P))

@test Bridge.b!(0.0, X1.yy[1], copy( X1.yy[1]),  P) == Bridge.b(0.0, X1.yy[1], P)

Bridge.σ(t, x, P::PBIntegratedDiffusion) = ℝ{2}(0.0, P.γ)

@test Bridge.σ!(0.0, X1.yy[1], 1.0, copy( X1.yy[1]),  P) == Bridge.σ(0.0, X1.yy[1], P)


Bridge.constdiff(::PBIntegratedDiffusion) = true

#struct PBIntegratedDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
#    γ::Float64
#end

Bridge.b(t::Float64, x, P::PBIntegratedDiffusionAux) = ℝ{2}(x[2], βu(t, x[2], P))

@test Bridge.b!(0.0, X1.yy[1], copy( X1.yy[1]),  Pt) == Bridge.b(0.0, X1.yy[1], Pt)


Bridge.σ(t, P::PBIntegratedDiffusionAux) =  ℝ{2}(0.0, P.γ)
Bridge.σ(t, x, P::PBIntegratedDiffusionAux) = Bridge.σ(t, P)

Bridge.B(t, P::PBIntegratedDiffusionAux) = @SMatrix [0.0 1.0; 0.0 -1.0]
Bridge.β(t, P::PBIntegratedDiffusionAux) = ℝ{2}(0, 1/2)
Bridge.a(t, P::PBIntegratedDiffusionAux) = @SMatrix [0.0 0.0; 0.0 P.γ^2]


# Generate Data
Random.seed!(1)

W2 = sample(tt, Wiener())

@test W2.yy == W1.yy

X2 = solve(Euler(), ℝ{2}(x0), W2, P)

@test X2.yy == X1.yy


L = @SMatrix [1. 0.]
Σ = @SMatrix [0.01]
v = ℝ{1}(2.5)

ϵ = 0.02
# Solve Backward Recursion

S2 = typeof(L)
S = typeof(L*L')
T = typeof(diag(L*L'))

N = length(tt)
Lt = zeros(S2, N)
Ht = zeros(S, N)
νt = zeros(T, N)


Po2 = Bridge.PartialBridgeνH(tt, P, Pt, L, v, ϵ, Σ)

Xo2 = copy(X2)
solve!(Euler(), Xo2, ℝ{2}(x0), W2, Po2)

@test norm(Xo1.yy - Xo2.yy) < 500*eps()

# Likelihood

ll2 = llikelihood(Bridge.LeftRule(), Xo2, Po2)

@test ll1 ≈ ll2

lls2 = Float64[]

    # MCMC parameter
function mcmc2(x0, tt, Po2)
    iterations = 1000
    subsamples = 0:100:iterations
    ρ = 0.9


    # initalization
    W = sample(tt, Wiener())
    X = solve(Euler(), x0, W, Po2)
    ll = llikelihood(Bridge.LeftRule(), X, Po2)

    acc = 0

    Wo = copy(W)
    Xo = copy(X)
    Wrho = copy(W)

    XX = Any[]
    if 0 in subsamples
        push!(XX, copy(X))
    end
    for iter in 1:iterations
        # Proposal
        sample!(Wrho, Wiener())
        Wo.yy .= ρ*W.yy + sqrt(1-ρ^2)*Wrho.yy

        solve!(Euler(), Xo, x0, Wo, Po2)
        llo = llikelihood(Bridge.LeftRule(), Xo, Po2)
        push!(lls2, llo)

        if log(rand()) <= llo - ll
            X, Xo = Xo, X
            W, Wo = Wo, W

            ll = llo
            acc += 1
        end
        if iter in subsamples
            push!(XX, copy(X))
        end
    end


end


@time @testset "MCMC" begin
    mcmc2(ℝ{2}(x0), tt, Po2)

    @test norm(lls1 - lls2) < sqrt(eps())

end
