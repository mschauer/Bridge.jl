
#module Visualize
#using Bridge, PyPlot, StaticVector
#end

using Bridge, StaticArrays, Bridge.Models
const R = ℝ
srand(1)

iterations = 5000
rho = 0.99

t = 1.0
T = 10.
n = 100001
m = 100
M = div(n-1,m)
skiplast = 10
skippoints = 10
dt = (T-t)/(n-1)
tt = t:dt:T
npath = 1

P = Lorenz(ℝ{3}(10, 20, 8/3), ℝ{3}(5,5,5))


x0 = Models.x0(P)
    
crit(θ1, θ3) = θ1*(θ1 + θ3 + 3)/(θ1 - θ3 - 1)

W = sample(tt, Wiener{ℝ{3}}())
X = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X, x0, W, P)
W = sample(tt, Wiener{ℝ{3}}())
X2 = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X2, x0, W, P)


# Observation scheme and subsample
_pairs(collection) = Base.Generator(=>, keys(collection), values(collection))

L = I
Σ = I
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


#=
GP = Bridge.GuidedBridge(tt, X.yy[1], X.yy[end], P, Pt)
V = SamplePath(tt[9500:end], zeros(R{3}, n))
solvebackward!(Bridge.R3(), Bridge.b, V, X.yy[end], P)
V = SamplePath(tt, zeros(R{3}, n))
solve!(Bridge.R3(), Bridge.b, V, X.yy[1], P)
=#




# Create linear noise approximations

forward = true # direction of the linear noise approximation

TPt = Bridge.LinearNoiseAppr{Bridge.Models.Lorenz,StaticArrays.SDiagonal{3,Float64},SVector{3,Float64}}
TPᵒ = Bridge.GuidedBridge{SVector{3,Float64},StaticArrays.SArray{Tuple{3,3},Float64,2,9},Bridge.LinearNoiseAppr{Bridge.Models.Lorenz,StaticArrays.SDiagonal{3,Float64},SVector{3,Float64}},Bridge.Models.Lorenz}


Pt = Vector{TPt}(m)
Pᵒ = Vector{TPᵒ}(m)
H♢ = Bridge.outer(zero(x0))
v = Xtrue.yy[end]
#zero(ℝ{3})

for i in m:-1:1
    XX[i] = SamplePath(X.tt[1 + (i-1)*M:1 + i*M], X.yy[1 + (i-1)*M:1 + i*M])
    WW[i] = SamplePath(W.tt[1 + (i-1)*M:1 + i*M], W.yy[1 + (i-1)*M:1 + i*M])
    Pt[i] = Bridge.LinearNoiseAppr(XX[i].tt, P, XX[i].yy[forward ? 1 : end], Bridge.a(XX[i].tt[1], XX[i].yy[end], P), forward)
    Pᵒ[i] = Bridge.GuidedBridge(XX[i].tt, P, Pt[i], v, H♢)
    H♢, v = Bridge.gpupdate(Pᵒ[i], L, Σ, V.yy[i])
end

y = x0
for i in 1:m
    sample!(WW[i], Wiener{ℝ{3}}())
    y = Bridge.bridge!(XX[i], y, WW[i], Pᵒ[i])
    XXᵒ[i] = copy(XX[i])
    WWᵒ[i] = copy(WW[i])
    XXmean[i] = SamplePath(XX[i].tt, zeros(XX[i].yy))
end



acc = 0
for it in 1:iterations
    y = x0
    for i in 1:m
        sample!(WWᵒ[i], Wiener{ℝ{3}}())
        WWᵒ[i].yy[:] = rho*WW[i].yy + sqrt(1-rho^2)*WWᵒ[i].yy
        y = Bridge.bridge!(XXᵒ[i], y, WWᵒ[i], Pᵒ[i])
    end

    ll = 0.0
    for i in 1:m
        ll += llikelihood(LeftRule(), XXᵒ[i],  Pᵒ[i], skip=skiplast) - llikelihood(LeftRule(), XX[i],  Pᵒ[i], skip=skiplast)
    end

    print("$it ll $ll ")

    if exp(ll) < rand()
        acc += 1
        println("*")
        for i in 1:m
            XX[i], XXᵒ[i] = XXᵒ[i], XX[i]
            WW[i], WWᵒ[i] = WWᵒ[i], WW[i]
        end
    end
    println()
    for i in 1:m
        XXmean[i].yy[:] += XX[i].yy
    end
end
for i in 1:m
    XXmean[i].yy[:] /= iterations
end
@show acc/iterations    

# Plot result

using GLAbstraction, Colors, GeometryTypes, GLVisualize, Reactive
window = glscreen()
timesignal = bounce(linspace(0.0, 1.0, 360))



shi = ℝ{3}(0,0,1)
sca = 1/15

Y = vcat([Pt[i].Y[1:end-1] for i in 1:m]...)
Yxyz = Point3f0[]
Yintensities = Float32[]
append!(Yxyz, map(Point3f0, Y.yy*sca .- Scalar(shi)))
append!(Yintensities, fill(10, length(Y.tt)))

Xxyz = Point3f0[]
Xintensities = Float32[]
append!(Xxyz, map(Point3f0,Xtrue.yy*sca .- Scalar(shi)))
append!(Xintensities, fill(10, length(Xtrue.tt)))


XXall = vcat([XX[i][1:end-1] for i in 1:m]...)
XXxyz = Point3f0[]
XXintensities = Float32[]
append!(XXxyz, map(Point3f0, XXall.yy*sca .- Scalar(shi)))
append!(XXintensities, fill(10, length(XXall.tt)))

XXmeanall = vcat([XXmean[i][1:end-1] for i in 1:m]...)
XXmeanxyz = Point3f0[]
append!(XXmeanxyz, map(Point3f0, XXmeanall.yy*sca .- Scalar(shi)))


Vxyz = Point3f0[]
Vintensities = Float32[]
append!(Vxyz, map(Point3f0,V.yy*sca .- Scalar(shi)))
append!(Vintensities, fill(10, length(V.tt)))


# map comes from Reactive.jl and allows you to map any Signal to another.
# In this case we create a rotation matrix from the timesignal signal.

rotation = map(timesignal) do t
    rotationmatrix_z(Float32(t*2pi)) # -> 4x4 Float32 rotation matrix
end
rotation = rotationmatrix_z(Float32(0.2*2pi)) 
# creates a color map from which we can sample for each line
# and add some transparency
if npath ==1
    cmap = [RGBA{Float64}(0.04, 0.15,0.44, 0.4)]
else
    cmap = map(x-> RGBA{Float32}(x, 0.4), colormap("Blues", npath))
end

X3d = visualize(
    Xxyz[1:skippoints:end], :lines,
    intensity = Xintensities[1:skippoints:end],
    color_map = cmap,
    color_norm = Vec2f0(0, npath), # normalize intensities. Lookup in cmap will be between 0-1
    model = rotation
)
_view(X3d, window, camera=:perspective)

Y3d = visualize(
    Yxyz[1:skippoints:end], :lines,
    intensity = Yintensities[1:skippoints:end],
    color_map = [RGBA{Float64}(0.34, 0.05,0.14, 0.4)],
    color_norm = Vec2f0(0, 1), # normalize intensities. Lookup in cmap will be between 0-1
    model = rotation
)
_view(Y3d, window, camera=:perspective)

XX3d = visualize(
    XXxyz[1:skippoints:end], :lines,
    intensity = XXintensities[1:skippoints:end],
    color_map = [RGBA{Float64}(0.04, 0.35,0.14, 0.4)],
    color_norm = Vec2f0(0, 1), # normalize intensities. Lookup in cmap will be between 0-1
    model = rotation
)
_view(XX3d, window, camera=:perspective)

XXmean3d = visualize(
    XXmeanxyz, :lines,
    intensity = XXintensities,
    color_map = [RGBA{Float64}(0.04, 0.35,0.14, 0.4)],
    color_norm = Vec2f0(0, 1), # normalize intensities. Lookup in cmap will be between 0-1
    model = rotation
)
_view(XXmean3d, window, camera=:perspective)


circle = Sphere(Point2f0(0), 0.02f0)
V3d = visualize(
    (circle, Vxyz), 
    intensity = Vintensities,
    color_map = [RGBA{Float32}(0.2,0.0,0.6, 0.8)],
    color_norm = Vec2f0(0, 1), # normalize intensities. Lookup in cmap will be between 0-1
    model = rotation
)
_view(V3d, window, camera=:perspective)




renderloop(window)
