using Bridge
using Bridge: TimeChange, outer, timechange, timeunchange, scale, unscale
using Distributions, StaticArrays
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models

using Makie, Colors
Makie.convert_arguments(P::Type{<:Union{Lines,Scatter}}, X::SamplePath{<:Real}) = Makie.convert_arguments(P, X.tt, X.yy)
Makie.convert_arguments(P::Type{<:Union{Lines,Scatter}}, X::SamplePath{ℝ{2}}) = Makie.convert_arguments(P, first.(X.yy), last.(X.yy))
viridis(X, alpha = 0.9f0, maxviri = 200) = map(x->RGBA(Float32.(x)..., alpha), Bridge._viridis[round.(Int, range(1, stop=maxviri, length=length(X)))])


T1, T2 = 1.0, 2.0
dt = 1/5000


t = T1:dt:T2
struct IntegratedDiffusion <: ContinuousTimeProcess{ℝ{2}}
    γ::Float64
end
struct IntegratedDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    γ::Float64
end

PorPtilde = Union{IntegratedDiffusion, IntegratedDiffusionAux}


βu(t, x::Float64, P::IntegratedDiffusion) = - (x + 1sin(x)) + 1/2
βu(t, x::Float64, P::IntegratedDiffusionAux) = -x + 1/2
# not really a 'beta'

Bridge.b(t::Float64, x, P::PorPtilde) = ℝ{2}(x[2], βu(t, x[2], P))
Bridge.σ(t, x, P::PorPtilde) = ℝ{2}(0.0, P.γ)

Bridge.a(t, P::PorPtilde) = @SArray [0.0 0.0; 0.0 P.γ^2]
Bridge.a(t, x, P::PorPtilde) = Bridge.a(t, P::PorPtilde)

Bridge.constdiff(::PorPtilde) = true

Bridge.B(t, P::IntegratedDiffusionAux) = @SMatrix [0.0 1.0; 0.0 -1.0]
Bridge.β(t, P::IntegratedDiffusionAux) = ℝ{2}(0, 1/2)

# Generate Data
Random.seed!(1)

P = IntegratedDiffusion(0.7)
Pt = IntegratedDiffusionAux(0.7)

W = sample(t, Wiener())
xstart = ℝ{2}(2.0, 1.0)

X = solve(EulerMaruyama(), xstart, W, P)


νend = xend = X.yy[end]
Hend = outer(ℝ{2}(0,0))
Po, _ = Bridge.partialbridgeνH(t, P, Pt, νend, Hend)

W = sample(t, Wiener())
Xo = solve(EulerMaruyama(), xstart, W, Po)

scene = lines(X, color=viridis(t))
lines!(scene, Xo, color=viridis(t))
display(scene)

PU = TimeChange(Po, T1, T2)
@test norm(timeunchange.(timechange.(t, PU), PU) - t) < 1e-13



s = T1:dt:T2
t = timeunchange.(s, PU)
Po, _ = Bridge.partialbridgeνH(t, P, Pt, νT, HT)

W = sample(t, Wiener())
Xo = solve(EulerMaruyama(), xstart, W, Po)
scene = lines(Xo, color=viridis(t))
scatter!(scene, [xstart, xend], markersize=0.001)

U = SamplePath(s[1:end-1], ((Po.ν .- Xo.yy)./(T2 .- s))[1:end-1])

scene = lines(U, color=viridis(U.tt))

scene = lines(U.tt, last.(U.yy))

PU = TimeChange(Po, T1, T2)
@test norm(U.yy - [Bridge.scale((i,s), Xo.yy[i], PU) for (i,s) in enumerate(s)][1:end-1]) < 1e-12
