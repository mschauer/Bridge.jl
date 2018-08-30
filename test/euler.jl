using Bridge, StaticArrays
using Test


n = 5000
T = (0.0, 10.0)
dt = (T[end]-T[1])/n
tt = collect(range(T[1], stop=T[2], length=n+1))
vx = [1.0,0.0,0.0]


const SV3 = SVector{3, Float64}




mutable struct VLorenz <: ContinuousTimeProcess{Vector{Float64}}
end

Bridge.b(t, u, ::VLorenz) = [10.0(u[2]-u[1]), u[1]*(28.0-u[3]) - u[2], u[1]*u[2] - (8/3)*u[3]]
function Bridge.b!(t, u, du, ::VLorenz)
    @inbounds du[1] = 10.0(u[2]-u[1])
    @inbounds du[2] = u[1]*(28.0-u[3]) - u[2]
    @inbounds du[3] = u[1]*u[2] - (8/3)*u[3]
end
Bridge.σ(t, u, ::VLorenz) = 3.0I
function Bridge.σ!(t, u, dw, dm, ::VLorenz) 
    @inbounds dm[1] = 3*dw[1]
    @inbounds dm[2] = 3*dw[2]
    @inbounds dm[3] = 3*dw[3]
end



W = sample(tt, Wiener{SV3}())
VW = VSamplePath(tt, collect(Bridge.mat(W.yy)))
VX = VSamplePath(tt, copy(VW.yy))
VX2 = VSamplePath(tt, copy(VW.yy))
solve!(EulerMaruyama!(), VX, vx, VW, VLorenz())
solve!(EulerMaruyama(), VX2, vx, VW, VLorenz())

solve(EulerMaruyama!(), zeros(3), VW, Wiener())

######

mutable struct Lorenz <: ContinuousTimeProcess{SV3}
end

Bridge.b(s, u, ::Lorenz) = @SVector [10.0(u[2]-u[1]), u[1]*(28.0-u[3]) - u[2], u[1]*u[2] - (8/3)*u[3]           ]
Bridge.σ(t, u, ::Lorenz) = 3.0I


u = SVector{3}(vx)
X = solve(EulerMaruyama(), u, W, Lorenz())
X2 = copy(X)
X3 = copy(X)

solve!(Euler(), X, u, W, Lorenz())
solve!(EulerMaruyama(), X2, u, W, Lorenz())
invoke(solve!, Tuple{EulerMaruyama,Any,SVector{3,Float64},Bridge.AbstractPath,Lorenz},
EulerMaruyama(), X3, u, W, Lorenz())

@test maximum(abs.(VW.yy - Bridge.mat(W.yy))) < eps()

@test maximum(abs.(VX.yy - Bridge.mat(X.yy))) < eps()
@test maximum(abs.(VX.yy - VX2.yy)) < eps()
@test maximum(norm.(X.yy - X2.yy)) < eps()
@test maximum(norm.(X.yy - X3.yy)) < eps()

