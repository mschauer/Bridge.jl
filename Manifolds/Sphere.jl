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
    xT
    σ
    B
end

Bridge.B(T, ℙ::SphereDiffusionAux) = ℙ.B
Bridge.β(t, ℙ::SphereDiffusionAux) = zeros(3)
Bridge.σ(t, ℙ::SphereDiffusionAux) = ℙ.σ
