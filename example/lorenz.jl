
#module Visualize
#using Bridge, PyPlot, StaticVector
#end

using Bridge, StaticArrays, Examples
const R = ℝ

t = 1.0
T = 10.
n = 10001
skipl = 3
dt = (T-t)/(n-1)
tt = t:dt:T
m = 50_000
npath = 2

P = Lorenz(ℝ{3}(10, 20, 8/3), ℝ{3}(2,2,2))
x0 = Examples.x0(P)
    
crit(θ1, θ3) = θ1*(θ1 + θ3 + 3)/(θ1 - θ3 - 1)

W = sample(tt, Wiener{ℝ{3}}())
X = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X, x0, W, P)
W = sample(tt, Wiener{ℝ{3}}())
X2 = SamplePath(tt, zeros(ℝ{3}, length(tt)))
Bridge.solve!(Euler(), X2, x0, W, P)
  
using GLAbstraction, Colors, GeometryTypes, GLVisualize, Reactive
window = glscreen()
timesignal = bounce(linspace(0.0, 1.0, 360))


# Create points

xyz = Point3f0[]
intensities = Float32[]
 
d = ℝ{3}(0,0,1)

append!(xyz, map(Point3f0,X.yy/15 .- Scalar(d)))
append!(intensities, fill(10, length(X.tt)))
append!(xyz, map(Point3f0,X2.yy/15 .- Scalar(d)))
append!(intensities, fill(10, length(X.tt)))


# map comes from Reactive.jl and allows you to map any Signal to another.
# In this case we create a rotation matrix from the timesignal signal.

rotation = map(timesignal) do t
    rotationmatrix_z(Float32(t*2pi)) # -> 4x4 Float32 rotation matrix
end

# creates a color map from which we can sample for each line
# and add some transparency
cmap = map(x-> RGBA{Float32}(x, 0.4), colormap("Blues", npath))

lines3d = visualize(
    xyz, :lines,
    intensity = intensities,
    color_map = cmap,
    color_norm = Vec2f0(0, npath), # normalize intensities. Lookup in cmap will be between 0-1
    model = rotation
)

_view(lines3d, window, camera=:perspective)

renderloop(window)
