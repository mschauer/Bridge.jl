using StaticArrays



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


Bridge.constdiff(::PBIntegratedDiffusion) = true
Bridge.b(t::Float64, x, P::PBIntegratedDiffusion) = ℝ{2}(x[2], βu(t, x[2], P))
Bridge.σ(t, x, P::PBIntegratedDiffusion) = ℝ{2}(0.0, P.γ)
Bridge.b(t::Float64, x, P::PBIntegratedDiffusionAux) = ℝ{2}(x[2], βu(t, x[2], P))

Bridge.σ(t, P::PBIntegratedDiffusionAux) =  ℝ{2}(0.0, P.γ)
Bridge.σ(t, x, P::PBIntegratedDiffusionAux) = Bridge.σ(t, P)

Bridge.B(t, P::PBIntegratedDiffusionAux) = @SMatrix [0.0 1.0; 0.0 -1.0]
Bridge.β(t, P::PBIntegratedDiffusionAux) = ℝ{2}(0, 1/2)
Bridge.a(t, P::PBIntegratedDiffusionAux) = @SMatrix [0.0 0.0; 0.0 P.γ^2]

SKIP! = true
if !SKIP!

include("partialbridge!.jl")


@test Bridge.b!(0.0, X1.yy[1], copy( X1.yy[1]),  P) == Bridge.b(0.0, X1.yy[1], P)


@test Bridge.σ!(0.0, X1.yy[1], 1.0, copy( X1.yy[1]),  P) == Bridge.σ(0.0, X1.yy[1], P)

@test Bridge.b!(0.0, X1.yy[1], copy( X1.yy[1]),  Pt) == Bridge.b(0.0, X1.yy[1], Pt)

end

include("partialparam.jl")


# Generate Data
Random.seed!(1)

W2 = sample(tt, Wiener())


X2 = solve(Euler(), x0, W2, P)



if !SKIP!
    @test W2.yy == W1.yy
    @test X2.yy == X1.yy

end


Po2 = Bridge.PartialBridgeνH(tt, P, Pt, L, v, ϵ, Σ)

Ft = copy(Po2.ν)
Ht = copy(Po2.H)

ν, H⁺, C_ = Bridge.updateνH⁺C(L, Σ, v, ϵ)

F, H, C = Bridge.updateFHC(L, Σ, v, zero(Ft[end]), zero(Ht[end]), ϵ)
@test C ≈ C2
@test F ≈ H*ν
@test H⁺ ≈ inv(H)


Ft, Ht, C = Bridge.partialbridgeodeHνH!(Bridge.R3(), tt, Ft, Ht, Pt, (F, H, C))

@test abs(C - Po2.C) < 0.02
@test maximum(norm.(Ht .- Po2.H)) < 1e-5
@test maximum(norm.(Po2.H .* Po2.ν .- Ft)) < 0.01 #not very precise, update if cond number becomes better
@test_broken cond(Ht[1]) < 1.e7 #

#LP2 = -0.5*(x0'*Po2.H[1]*x0 - 2*x0'*Po2.F)[] - Po2.C
LP2 = -0.5*(x0'*Po2.H[1]*x0 - 2*x0'*Po2.H[1]*Po2.ν[1])[] - Po2.C

@show LP, LP2
@test abs(LP - LP2) < 0.01

Xo2 = copy(X2)
solve!(Euler(), Xo2, x0, W2, Po2)

if !SKIP!
    @test norm(Xo1.yy - Xo2.yy) <sqrt(eps())
end
# Likelihood

ll2 = llikelihood(Bridge.LeftRule(), Xo2, Po2)

#@test ll1 ≈ ll2
if !SKIP!
    @test abs(ll1 - ll2) < 0.0002
end

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
    if !SKIP!
        @test norm(lls1 - lls2) < sqrt(eps())
    end
    #@test  norm(lls1 - lls2) < 0.005

end
