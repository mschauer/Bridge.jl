using Bridge, StaticArrays
using Base.Test


n = 5000
T = (0.0, 10.0)
dt = (T[end]-T[1])/n
tt = collect(linspace(T[1], T[2], n+1))
vx = [1.0,0.0,0.0]


const SV = SVector{3, Float64}




type VLorenz <: ContinuousTimeProcess{Vector{Float64}}
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



W = sample(tt, Wiener{SV}())
VW = VSamplePath(tt, Bridge.mat(W.yy))
VX = VSamplePath(tt, copy(VW.yy))
VX2 = VSamplePath(tt, copy(VW.yy))
Bridge.solve!(Bridge.EulerMaruyama!(), VX, vx, VW, VLorenz())
Bridge.solve!(Bridge.EulerMaruyama(), VX2, vx, VW, VLorenz())


######

type Lorenz <: ContinuousTimeProcess{SV}
end

Bridge.b(s, u, ::Lorenz) = @SVector [10.0(u[2]-u[1]), u[1]*(28.0-u[3]) - u[2], u[1]*u[2] - (8/3)*u[3]           ]
Bridge.σ(t, u, ::Lorenz) = 3.0I


u = SVector{3}(vx)
X = Bridge.euler(u, W, Lorenz())
X2 = copy(X)
X3 = copy(X)

Bridge.solve!(Euler(), X, u, W, Lorenz())
Bridge.euler!(X2, u, W, Lorenz())
invoke(Bridge.solve!, Tuple{Bridge.EulerMaruyama,Any,SVector{3,Float64},Bridge.AbstractPath,Lorenz},
Bridge.EulerMaruyama(), X3, u, W, Lorenz())

@test maximum(abs.(VW.yy - Bridge.mat(W.yy))) < eps()

@test maximum(abs.(VX.yy - Bridge.mat(X.yy))) < eps()
@test maximum(abs.(VX.yy - VX2.yy)) < eps()
@test maximum(norm.(X.yy - X2.yy)) < eps()
@test maximum(norm.(X.yy - X3.yy)) < eps()

