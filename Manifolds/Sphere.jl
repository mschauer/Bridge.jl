using ForwardDiff
using LinearAlgebra
using StaticArrays, Distributions
using Plots

include(SpherePlots.jl)
include(Definitions.jl)

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
    σ::T
    𝕊::Sphere

    function SphereDiffusion(σ::T, 𝕊::Sphere) where {T<:Real}
        if σ == 0
            error("σ cannot be 0")
        end
        new{T}(σ, 𝕊)
    end
end

Bridge.b(t, x, ℙ::SphereDiffusion) = zeros(3)
Bridge.σ(t, x, ℙ::SphereDiffusion) = ℙ.σ*P(x, 𝕊)
Bridge.constdiff(::SphereDiffusion) = false
