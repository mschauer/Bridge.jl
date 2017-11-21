using Makie, GLVisualize
using GeometryTypes, Colors
using Bridge: _viridis
skippoints = 1
extra = 10
d2 = length(valtype(V))
scene = Scene(resolution = (400, 400))

lines((@view Xtrue.yy[1:skippoints:end]), linewidth = 0.2, color = :darkblue)
if d2 == 3
    scatter(V.yy, marker=Sphere(Point2f0(0), 1.0f0), markersize = 0.2,  strokewidth = 0.02, strokecolor = :white, color = map(x->RGBA(Float32.(x)..., 0.9f0), _viridis[1:2:2length(V.yy)])) 
elseif d2 == 2
    linesegment(map(v->(Point3f0(-10, v...), Point3f0(10, v...)), V.yy), color = map(x->RGBA(Float32.(x)..., 0.9f0), _viridis[1:2length(V.yy)]))
end

    #scatter(V.yy, marker= (Any['0':'9'...])[(1:length(V)) .% 10 + 1], color=:blue, markersize = 0.2,  strokewidth = 0.02, strokecolor = :white, color = map(x->RGBA(Float32.(x)..., 0.9f0), _viridis[1:2:2length(V.yy)])) 
for X in XXmean
    lines((@view X.yy[1:skippoints:end]), linewidth = 0.2, color = :darkgreen)
end
for P in Pt
    lines((@view P.Y.yy[1:skippoints:end]), linewidth = 0.2, color = :darkred)
end

XXmeanxyz = collect(Point3f0, Iterators.flatten(XXmean[i].yy[1:end-1] for i in 1:m))
XXscale = vcat([Point3f0.(XXscal[i][1:end-1]) for i in 1:m]...)
XXrotation = vcat([Vec4f0.(XXrot[i][1:end-1]) for i in 1:m]...)

sphere = Sphere(Point3f0(0,0,0), 1.0f0)
viz = visualize(
    (sphere, XXmeanxyz[1:5extra*skippoints:end]),
    scale = Float32(1)*XXscale[1:5extra*skippoints:end],      
    rotation = XXrotation[1:5extra*skippoints:end],          
    color = RGBA(0.2f0, 0.4f0, 0.9f0, 0.1f0),
).children[]
Makie.insert_scene!(scene, :mesh, viz, Dict(:show=>true, :camera=>:auto))

if d2 == 3
    axis(map(x -> UnitRange(round.(Int,x)...), extrema(Bridge.mat(V.yy), 2))...)
else 
    axis(map(x -> UnitRange(round.(Int,x)...), extrema(Bridge.mat(Xtrue.yy), 2))...)
end

center!(scene)


