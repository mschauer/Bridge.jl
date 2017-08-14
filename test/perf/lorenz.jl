using StochasticDiffEq
using Bridge, StaticArrays
using BenchmarkTools
#####

function lorenz!(t,u,du)
 du[1] = 10.0(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end

function σ_lorenz!(t,u,du)
 du[1] = 3.0
 du[2] = 3.0
 du[3] = 3.0
end

n = 5000
T = (0.0, 10.0)
dt = (T[end]-T[1])/n
tt = collect(linspace(T[1], T[2], n+1))
vx = [1.0,0.0,0.0]

prob_sde_lorenz = SDEProblem(lorenz!, σ_lorenz!, vx, T)
sol = StochasticDiffEq.solve(prob_sde_lorenz, StochasticDiffEq.EM(), dt=dt)

######

const SV = SVector{3, Float64}

struct VSamplePath{T} <: Bridge.AbstractPath{T}
    tt::Vector{Float64}
    yy::Matrix{T}
    VSamplePath(tt, yy::Matrix{T}) where {T} = new{T}(tt, yy)
end

import Base: length, start, next, done
start(X::VSamplePath) = 1
next(X::VSamplePath, i) = (i, X.tt[i], X.tt[i+1]-X.tt[i], X.yy[i]), i + 1
done(X::VSamplePath, i) = i + 1 > length(X.tt)
length(X::VSamplePath) = length(X.tt)

type VLorenz <: ContinuousTimeProcess{Vector{Float64}}
end

Bridge.b(s, u, ::VLorenz) = [10.0(u[2]-u[1]), u[1]*(28.0-u[3]) - u[2], u[1]*u[2] - (8/3)*u[3]]
function Bridge.b!(s, u, du, ::VLorenz)
    @inbounds du[1] = 10.0(u[2]-u[1])
    @inbounds du[2] = u[1]*(28.0-u[3]) - u[2]
    @inbounds du[3] = u[1]*u[2] - (8/3)*u[3]
end
Bridge.σ(t, u, ::VLorenz) = 3.0I
function Bridge.σ!(t, u, dw, dm, ::VLorenz) 
    @inbounds @. dm = 3*dw
end



W = sample(tt, Wiener{SV}())
VW = VSamplePath(tt, Bridge.mat(W.yy))
VX = VSamplePath(tt, copy(VW.yy))
Bridge.solve!(Bridge.Euler(), VX, vx, VW, VLorenz())

######

type Lorenz <: ContinuousTimeProcess{SV}
end

Bridge.b(s, u, ::Lorenz) = @SVector [10.0(u[2]-u[1]), u[1]*(28.0-u[3]) - u[2], u[1]*u[2] - (8/3)*u[3]           ]
Bridge.σ(t, u, ::Lorenz) = 3.0I


u = SVector{3}(vx)
X = Bridge.euler(u, sample(tt, Wiener{SV}()), Lorenz())


######



@btime StochasticDiffEq.solve(prob_sde_lorenz, StochasticDiffEq.EM(), dt=dt)
@btime W = sample(tt, Wiener{SV}())
@btime X = copy(W)
@btime Bridge.euler!(X, u, W, Lorenz())
@btime Bridge.solve!(Bridge.EulerMaruyama(), X, u, W, Lorenz())
@btime Bridge.solve!(Bridge.EulerMaruyama!(), VX, vx, VW, VLorenz())
;