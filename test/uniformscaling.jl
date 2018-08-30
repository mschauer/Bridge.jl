# Define a diffusion process
using Bridge
struct OU2696197128  <: ContinuousTimeProcess{Float64}
    β::Float64 # drift parameter (also known as inverse relaxation time)
    σ::UniformScaling{Float64} # diffusion parameter
    function OU2696197128(β::Float64, σ::UniformScaling{Float64})
        isnan(β) || β > 0. || error("Parameter λ must be positive.")
        isnan(σ.λ) || σ.λ > 0. || error("Parameter σ must be positive.")
        new(β, σ)
    end
end

# define drift and diffusion coefficient of OU2696197128

Bridge.b(t,x, P::OU2696197128) = -P.β*x
Bridge.σ(t, x, P::OU2696197128) = P.σ

# simulate OU2696197128 using Euler scheme
W = sample(0:0.01:10, Wiener{Float64}()) 
X = solve(Euler(), 0.1, W, OU2696197128(20., 1.0*I))

#Check @code_warntype euler!(copy(W),0.0, W, OU2696197128(20., 1.0*I))
