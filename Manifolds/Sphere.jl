using ForwardDiff
using LinearAlgebra
using StaticArrays, Distributions
using Plots
using Bridge

include("SpherePlots.jl")
include("Definitions.jl")

T = 1.0
dt = 1/1000
τ(T) = (x) -> x*(2-x/T)
tt = τ(T).(0.:dt:T)

L = SMatrix{3,3}(1.0I)
Σdiagel = 10^(-3)
Σ = SMatrix{3,3}(Σdiagel*I)

extractcomp(v, i) = map(x->x[i], v)

"""
    The object SphereDiffusion(σ, 𝕊) can be used to generate a diffusion
    on the sphere 𝕊. We will focus on the diffusion equation
        `` dX_t = σ P(X_t)∘dW_t ``
    where σ ∈ ℝ
"""

struct SphereDiffusion{T} <: ContinuousTimeProcess{ℝ{3}}
    Σ::T
    𝕊::Sphere

    function SphereDiffusion(σ::T, 𝕊::Sphere) where {T<:Real}
        if σ == 0
            error("σ cannot be 0")
        end
        new{T}(σ, 𝕊)
    end
end

Bridge.b(t, x, ℙ::SphereDiffusion{T}) where {T} = zeros(3)
Bridge.σ(t, x, ℙ::SphereDiffusion{T}) where {T} = ℙ.Σ*P(x, 𝕊)
Bridge.constdiff(::SphereDiffusion{T}) where {T} = false

"""
    Example: Constructing a Brownian motion on a sphere of radius 1
"""

𝕊 = Sphere(1.0)
ℙ = SphereDiffusion(1.0, 𝕊)

x₀ = [0.,0.,1.]
W = sample(0:dt:T, Wiener{ℝ{3}}())
X = solve(StratonovichEuler(), x₀, W, ℙ)

plotly()
SpherePlot(X, 𝕊)

"""
    Insert the settings for the auxiliary process tildeX
        and set partial bridges for each data point
"""
struct SphereDiffusionAux <: ContinuousTimeProcess{ℝ{3}}
    ξ
    σ
    B
end

Bridge.B(t, ℙt::SphereDiffusionAux) = ℙt.B
Bridge.β(t, ℙt::SphereDiffusionAux) = zeros(3)
Bridge.σ(t, ℙt::SphereDiffusionAux) = ℙt.σ
Bridge.b(t, x, ℙt::SphereDiffusionAux) = Bridge.B(t, ℙt)*x + Bridge.β(t,ℙt)
Bridge.a(t, ℙt::SphereDiffusionAux) = Bridge.σ(t, ℙt)*Bridge.σ(t, ℙt)'
Bridge.constdiff(::SphereDiffusionAux) = true

"""
    Now let us create a proposal diffusion bridge that hits ξ at time T
    we use the transition density of tildeX in the guided proposal

"""
ξ = [0.,1.,0.]
f(ξ, 𝕊) # This should be zero

ℙt = SphereDiffusionAux(ξ, P(ξ, 𝕋), [rand() rand() rand() ; rand() rand() rand() ; rand() rand() rand()])

"""
    Settings for the Guided proposal
"""
Φ(t, ℙt::SphereDiffusionAux) = exp(ℙt.B*t)
Φ(t, s, ℙt::SphereDiffusionAux) = exp(ℙt.B*(t-s)) # = Φ(t)Φ(s)⁻¹
Υ = Σ

Lt(t, ℙt::SphereDiffusionAux) = L*Φ(T, t, ℙt)
μt(t, ℙt::SphereDiffusionAux) = zeros(3)


M⁺ = zeros(typeof(Σ), length(tt))
M = copy(M⁺)
M⁺[end] = Υ
M[end] = inv(Υ)
for i in length(tt)-1:-1:1
    dt = tt[i+1] - tt[i]
    M⁺[i] = M⁺[i+1] + Lt(tt[i+1], ℙt)*Bridge.a(tt[i+1], ℙt)*Lt(tt[i+1], ℙt)'*dt + Υ
    M[i] = inv(M⁺[i])
end

H((i, t)::IndexedTime, x, ℙt::SphereDiffusionAux) = Lt(t, ℙt)'*M[i]*Lt(t, ℙt)
r((i, t)::IndexedTime, x, ℙt::SphereDiffusionAux) = Lt(t, ℙt)'*M[i]*(ℙt.ξ .- μt(t, ℙt) .- Lt(t, ℙt)*x)

struct GuidedProposalSphere <: ContinuousTimeProcess{ℝ{3}}
    ξ
    Target::SphereDiffusion
    Auxiliary::SphereDiffusionAux
end

function Bridge.b(t, x, ℙᵒ::GuidedProposalSphere)
    k = findmin(abs.(tt.-t))[2]
    ℙ = ℙᵒ.Target
    ℙt = ℙᵒ.Auxiliary
    a = Bridge.σ(t, x, ℙ)*Bridge.σ(t, x, ℙ)'
    return Bridge.b(t, x, ℙ) + a*r((k, tt[k]), x, ℙt)
end

Bridge.σ(t, x, ℙᵒ::GuidedProposalSphere) = Bridge.σ(t, x, ℙᵒ.Target)
Bridge.constdiff(::GuidedProposalSphere) = false

ℙᵒ = GuidedProposalSphere(ξ, ℙ, ℙt)
W = sample(0:dt:T, Wiener{ℝ{3}}())
Xᵒ = solve(StratonovichEuler(), [0.,0.,1.], W, ℙᵒ)

plotly()
plot([extractcomp(Xᵒ.yy, 1), extractcomp(Xᵒ.yy, 2), extractcomp(Xᵒ.yy, 3)])
SpherePlot(Xᵒ, 𝕊)
plot!([0.], [0.], [1.],
        legend = true,
        color = :red,
        seriestype = :scatter,
        markersize = 1.5,
        label = "start")
plot!([ξ[1]],  [ξ[2]],  [ξ[3]],
        color = :yellow,
        seriestype = :scatter,
        markersize = 1.5,
        label = "end")
