using ForwardDiff
using LinearAlgebra
using StaticArrays, Distributions
using Plots
using Bridge

include("TorusPlots.jl")
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
    The object TorusDiffusion(σ, 𝕋) can be used to generate a diffusion
    on the Torus 𝕋. We will focus on the diffusion equation
        `` dX_t = σ P(X_t)∘dW_t ``
    where σ ∈ ℝ
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
X = solve(StratonovichEuler(), x₀, W, ℙ)

plotly()
TorusPlot(X, 𝕋)

"""
    Insert the settings for the auxiliary process tildeX
        and set partial bridges for each data point
"""

struct TorusDiffusionAux <: ContinuousTimeProcess{ℝ{3}}
    xT
    σ
    B
end

Bridge.B(t, ℙt::TorusDiffusionAux) = ℙt.B
Bridge.β(t, ℙt::TorusDiffusionAux) = zeros(3)
Bridge.σ(t, ℙt::TorusDiffusionAux) = ℙt.σ
Bridge.b(t, x, ℙt::TorusDiffusionAux) = Bridge.B(t, ℙt)*x + Bridge.β(t,ℙt)
Bridge.a(t, ℙt::TorusDiffusionAux) = Bridge.σ(t, ℙt)*Bridge.σ(t, ℙt)'
Bridge.constdiff(::TorusDiffusionAux) = true

"""
    Now let us create a proposal diffusion bridge that hits ξ at time T
    we use the transition density of tildeX in the guided proposal

"""
ξ = [0., 2., 0.5]
f(ξ, 𝕋) # This should be zero

ℙt = TorusDiffusionAux(ξ, P(ξ, 𝕋), [rand() rand() rand() ; rand() rand() rand() ; rand() rand() rand()])

"""
    Settings for the Guided proposal
"""
Φ(t, ℙt::TorusDiffusionAux) = exp(ℙt.B*t)
Φ(t, s, ℙt::TorusDiffusionAux) = exp(ℙt.B*(t-s)) # = Φ(t)Φ(s)⁻¹
Υ = Σ

Lt(t, ℙt::TorusDiffusionAux) = L*Φ(T, t, ℙt)
μt(t, ℙt::TorusDiffusionAux) = 0.


M⁺ = zeros(typeof(Σ), length(tt))
M = copy(M⁺)
M⁺[end] = Υ
M[end] = inv(Υ)
for i in length(tt)-1:-1:1
    dt = tt[i+1] - tt[i]
    M⁺[i] = M⁺[i+1] + Lt(tt[i+1], ℙt)*Bridge.a(tt[i+1], ℙt)*Lt(tt[i+1], ℙt)'*dt + Υ
    M[i] = inv(M⁺[i])
end

const IndexedTime = Tuple{Int64,Float64}
H((i, t)::IndexedTime, x, ℙt::TorusDiffusionAux) = Lt(t, ℙt)'*M[i]*Lt(t, ℙt)
r((i, t)::IndexedTime, x, ℙt::TorusDiffusionAux) = Lt(t, ℙt)'*M[i]*(ℙt.ξ .-μt(t, ℙt).-Lt(t, ℙt)*x)

struct GuidedProposal <: ContinuousTimeProcess{ℝ{3}}
    ξ
    Target::TorusDiffusion
    Auxiliary::TorusDiffusionAux
end

function Bridge.b(t, x, ℙᵒ::GuidedProposal)
    k = findmin(abs.(tt.-t))[2]
    ℙ = ℙᵒ.Target
    ℙt = ℙᵒ.Auxiliary
    a = Bridge.σ(t, x, ℙ)*Bridge.σ(t, x, ℙ)'
    return Bridge.b(t, x, ℙ) + a*r((k, tt[k]), x, ℙt)
end

Bridge.σ(t, x, ℙᵒ::GuidedProposal) = Bridge.σ(t, x, ℙᵒ.Target)
Bridge.constdiff(::GuidedProposal) = false

ℙᵒ = GuidedProposal(ξ, ℙ, ℙt)
r
W = sample(0:dt:T, Wiener{ℝ{3}}())
Xᵒ = solve(StratonovichEuler(), x₀, W, ℙᵒ)

plot([extractcomp(Xᵒ.yy[1:1000], 1), extractcomp(Xᵒ.yy[1:1000], 2), extractcomp(Xᵒ.yy[1:1000], 3)])

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
