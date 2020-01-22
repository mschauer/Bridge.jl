using ForwardDiff
using LinearAlgebra
using StaticArrays, Distributions
using Plots
using Bridge

include("Definitions.jl")
include("TorusPlots.jl")
include("GuidedProposals.jl")

T = 1.0
dt = 1/1000
τ(T) = (x) -> x*(2-x/T)
tt = τ(T).(0.:dt:T)

L = SMatrix{3,3}(1.0I)
Σdiagel = 10^(-3)
Σ = SMatrix{3,3}(Σdiagel*I)

extractcomp(v, i) = map(x->x[i], v)

"""
    The object TorusDiffusion(σ, 𝕋) can be used to generate a diffusion
    on the Torus 𝕋. We will focus on the diffusion equation
        `` dX_t = Σ P(X_t)∘dW_t ``
    where Σ ∈ ℝ
"""

struct TorusDiffusion{T} <: ContinuousTimeProcess{ℝ{3}}
    Σ::T
    𝕋::Torus

    function TorusDiffusion(σ::T, 𝕋::Torus) where {T<:Real}
        if σ == 0
            error("σ cannot be 0")
        end
        new{T}(σ, 𝕋)
    end
end

Bridge.b(t, x, ℙ::TorusDiffusion{T}) where {T} = zeros(3)
Bridge.σ(t, x, ℙ::TorusDiffusion{T}) where {T} = ℙ.Σ*P(x, 𝕋)
Bridge.constdiff(::TorusDiffusion{T}) where {T} = false

"""
    Example: Constructing a Brownian motion on a Torus with
    inner radius r = ½ and outer radius R = 2
"""

𝕋 = Torus(2.0, 0.5)
ℙ = TorusDiffusion(1.0, 𝕋)

x₀ = [2.,0.,0.5]
W = sample(0:dt:T, Wiener{ℝ{3}}())
# X = solve(StratonovichEuler(), x₀, W, ℙ)
#
# plotly()
# TorusPlot(X, 𝕋)

"""
    Insert the settings for the auxiliary process tildeX
        and set partial bridges for each data point

    Now let us create a proposal diffusion bridge that hits ξ at time T
    we use the transition density of tildeX in the guided proposal

"""
ξ = [0.,2.,-.5]
f(ξ, 𝕋)

bT = zeros(eltype(ξ),3) # = b(t, X_T), i.e. the drift in the Ito form of the equation dX_t = P(X_t)∘dW_t
for i = 1:3
    for k = 1:3
        Pr = (z) -> P(z, 𝕋)[i, k]
        grad = ForwardDiff.gradient(Pr, ξ)
        for j = 1:3
            bT[i] += 0.5 * P(ξ, 𝕋)[j, k] * grad[j]
        end
    end
end



struct TorusDiffusionAux <: ContinuousTimeProcess{ℝ{3}}
    ξ
    σ
    B
end

Bridge.B(t, ℙt::TorusDiffusionAux) = ℙt.B
Bridge.β(t, ℙt::TorusDiffusionAux) = bT .- ℙt.B*ℙt.ξ
Bridge.σ(t, ℙt::TorusDiffusionAux) = ℙt.σ
Bridge.b(t, x, ℙt::TorusDiffusionAux) = Bridge.B(t, ℙt)*x + Bridge.β(t,ℙt)
Bridge.a(t, ℙt::TorusDiffusionAux) = Bridge.σ(t, ℙt)*Bridge.σ(t, ℙt)'
Bridge.constdiff(::TorusDiffusionAux) = true # This should be zero

ℙt = TorusDiffusionAux(ξ, P(ξ, 𝕋), [rand() rand() rand() ; rand() rand() rand() ; rand() rand() rand()])

"""
    Settings for the Guided proposal
"""
# Φ(t, ℙt::TorusDiffusionAux) = exp(ℙt.B*t)
# Φ(t, s, ℙt::TorusDiffusionAux) = exp(ℙt.B*(t-s)) # = Φ(t)Φ(s)⁻¹
Υ = Σ

# Lt(t, ℙt::TorusDiffusionAux) = L*Φ(T, t, ℙt)


function kernelr3(f, t, y, dt, P)
    k1 = f(t, y, P)
    k2 = f(t + 1/2*dt, y + 1/2*dt*k1, P)
    k3 = f(t + 3/4*dt, y + 3/4*dt*k2, P)
    y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
end



ℙᵒ = GuidedProposal(ξ, ℙ, ℙt)
W = sample(0:dt:T, Wiener{ℝ{3}}())
Xᵒ = solve(StratonovichEuler(), x₀, W, ℙᵒ)

plotly()
plot([extractcomp(Xᵒ.yy, 1), extractcomp(Xᵒ.yy, 2), extractcomp(Xᵒ.yy, 3)])
TorusPlot(Xᵒ, 𝕋)
plot!([2.], [0.], [.5],
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
