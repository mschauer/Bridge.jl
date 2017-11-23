module Models
using Bridge, StaticArrays, Distributions
export Lorenz, ℝ, foci

const ℝ{N} = SVector{N, Float64}

using Bridge.outer

struct FitzHughNagumo  <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ1::Float64
    σ2::Float64
end

Bridge.b(t, x, P::FitzHughNagumo) = ℝ{2}(P.ϵ\(x[1] - x[1]^3 - x[2] + s), P.γ*x[1] - x[2] + P.β)
Bridge.σ(t, x, P::FitzHughNagumo) = ℝ{2}(P.σ1, P.σ2)
Bridge.constdiff(::FitzHughNagumo) = true


struct Linear2  <: ContinuousTimeProcess{ℝ{2}}
    b11::Float64
    b21::Float64
    b12::Float64
    b22::Float64
    β1::Float64
    β2::Float64
    σ1::Float64
    σ2::Float64 
end

Bridge.B(t, P::Linear2) = SMatrix{2,2,Float}(P.b11, P.b21, P.b12, P.b22)
Bridge.β(t, P::Linear2) = ℝ{2}(P.β1, P.β2)
Bridge.σ(t, P::Linear2) =  ℝ{2}(P.σ1, P.σ2)
Bridge.constdiff(::Linear2) = true



struct Lorenz <: ContinuousTimeProcess{ℝ{3}}
    θ::ℝ{3} # σ ρ β
    σ::SDiagonal{3,Float64}
    Lorenz(θ=ℝ{3}(10,28,8/3),σ=ℝ{3}(1,1,1)) = new(θ,σ)
end

Bridge.b(t, x, P::Lorenz) = ℝ{3}(P.θ[1]*(x[2] - x[1]), x[1]*(P.θ[2]-x[3]) - x[2], x[1]*x[2] - P.θ[3]*x[3])

Bridge.bderiv(t, x, P::Lorenz) = @SMatrix Float64[
    -P.θ[1]         P.θ[1]  0
    (P.θ[2]-x[3])   -1      -x[1]
    x[2]            x[1]    -P.θ[3]
]
    


Bridge.σ(t, x, P::Lorenz) = SDiagonal(P.σ)
Bridge.constdiff(::Lorenz) = true

x0(P::Lorenz) = ℝ{3}(1.508870, -1.531271, 25.46091)

critlorenz(θ1, θ3) = θ1*(θ1 + θ3 + 3)/(θ1 - θ3 - 1)

function foci(P::Lorenz)
    σ, ρ, β = P.θ
    ℝ{3}(-√β*√(ρ-1), -√β*√(ρ-1), ρ-1), ℝ{3}(√β*√(ρ-1), √β*√(ρ-1), ρ-1)
end

struct Pendulum <: ContinuousTimeProcess{ℝ{2}}
    θ²::Float64
    γ::Float64
end

Bridge.Btilde(t, x, P::Pendulum) = SMatrix{2,2}(0.0,0.0,1.0,0.0)
Bridge.βtilde(t, x, P::Pendulum) = ℝ{2}(0.0, 0.0)

Bridge.b(t, x, P::Pendulum) = ℝ{2}(x[2], -P.θ²*sin(x[1]))
Bridge.σ(t, x, P::Pendulum) = ℝ{2}(0.0, P.γ)
Bridge.constdiff(::Pendulum) = true

end # Module
