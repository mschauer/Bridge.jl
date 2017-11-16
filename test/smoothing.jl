using Bridge, StaticArrays, Base.Test, Bridge.Models

iterations = 5000
rho = 0.05 # 
independent = false # true independent proposals
t = 1.0
T = 1.75
n = 5001 # total imputed length
m = 10 # number of segments
M = div(n-1,m)
skiplast = 0
skippoints = 2
dt = (T-t)/(n-1)
tt = t:dt:T

si = 3.
P = Bridge.Models.Lorenz(ℝ{3}(10, 20, 8/3), ℝ{3}(si,si,si))

x0 = Models.x0(P)
    

W = sample(tt, Wiener{ℝ{3}}())
X = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X, x0, W, P)

Y = Bridge.solve!(Bridge.R3(), copy(X), x0, P)


# Observation scheme and subsample
_pairs(collection) = Base.Generator(=>, keys(collection), values(collection))

L = I
Σ = SDiagonal(1., 1., 1.)
lΣ = chol(Σ)'
RV = ℝ{3}

V = SamplePath(collect(_pairs(X))[1:M:end])
#Vo = copy(V)
map!(y -> L*y + lΣ*randn(RV), V.yy, V.yy)

XX = Vector{typeof(X)}(m)
XXmean = Vector{typeof(X)}(m)
XXᵒ = Vector{typeof(X)}(m)
WW = Vector{typeof(W)}(m)
WWᵒ = Vector{typeof(W)}(m)



Xtrue = copy(X)

 


# Create linear noise approximations

forward = false # direction of the linear noise approximation

TPt = Any
TPᵒ = Any


Pt = Vector{TPt}(m)
Pᵒ = Vector{TPᵒ}(m)
H♢ = Bridge.outer(zero(x0))
#v = Xtrue.yy[end]
v = ( L' * inv(Σ) * L)\(L' * inv(Σ) *  V.yy[end])
H♢ = one(Bridge.outer(zero(x0)))*inv( L' * inv(Σ) * L)

for i in m:-1:1
    XX[i] = SamplePath(X.tt[1 + (i-1)*M:1 + i*M], X.yy[1 + (i-1)*M:1 + i*M])
    WW[i] = SamplePath(W.tt[1 + (i-1)*M:1 + i*M], W.yy[1 + (i-1)*M:1 + i*M])
    # short-cut, take v later
    Pt[i] =  Bridge.LinearAppr(Y[1 + (i-1)*M:1 + i*M], P)
    #Bridge.LinearNoiseAppr(XX[i].tt, P, XX[i].yy[end], Bridge.a(XX[i].tt[end], XX[i].yy[end], P), forward)
    #Pt[i] = Bridge.LinearNoiseAppr(XX[i].tt, P, v, Bridge.a(XX[i].tt[end], v, P), forward)
    Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pt[i], v, H♢)
    H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
end
 
v0 = V.yy[1]
#v0 = v

y = x0
for i in 1:m
    sample!(WW[i], Wiener{ℝ{3}}())
    y = Bridge.bridge!(XX[i], y, WW[i], Pᵒ[i])
end
XXmean = [zero(XX[i]) for i in 1:m]

μ0 = ℝ{3}(0,0,0)
Σ0 = SDiagonal(40., 40., 40.)
π0 = Bridge.Gaussian(μ0 + Σ0*L'*inv(L*Σ0*L' + Σ)*(v0 - L*μ0), Σ0 - Σ0*L'*inv(L*Σ0*L' + Σ)*L*Σ0)

H♢, v = Bridge.gpupdate(Pᵒ[1], L, Σ0, V.yy[1])
π0 = Bridge.Gaussian(v, H♢)

if true
    using GLVisualize, Colors, GeometryTypes
    window = glscreen()
    sca = 1/15
    phi = Float32(0.2*2pi)

    rotation = @SMatrix Float32[ sca*cos(phi)  -sca*sin(phi)    0.0       0.0
    sca*sin(phi)   sca*cos(phi)    0.0       0.0
    0.0            0.0             sca*1.0  -1.25
    0.0            0.0             0.0       1.0]

    X3d = visualize(
        Point3f0.(X.yy), :lines,
        color = RGBA{Float32}(0.04, 0.15, 0.44, 0.6), 
        model = rotation,
    )

    XX3d = visualize(
        Point3f0.(vcat(XX...).yy), :lines,
        color = RGBA{Float32}(0.04, 0.44, 0.12, 0.6), 
        model = rotation,
    )

    Y3d = visualize(
        Point3f0.(Y.yy), :lines,
        color = RGBA{Float32}(0.54, 0.15, 0.14, 0.6), 
        model = rotation,
    )
    circle = Sphere(Point2f0(0), 1f0)
    V3d = visualize(
        (circle, Point3f0.(V.yy)), 
        scale = fill(sca*0.5Point2f0(1,1), length(V)),
        color = RGBA{Float32}(0.7, 0.3, 0., 0.8),
        model = rotation,
    )


    _view(X3d, window, camera = :perspective)
    _view(XX3d, window, camera = :perspective)
    _view(Y3d, window, camera = :perspective)
    _view(V3d, window, camera = :perspective)
    renderloop(window)
end
h = sqrt(eps())
@test norm(SMatrix{3,3}(Iterators.flatten(h\(Bridge.b(0, x0 + h*eye(SMatrix{3,3,Float64})[i,:], P) - Bridge.b(0, x0, P)) for i in 1:3)...) - Bridge.bderiv(0, x0, P)) < 100h


