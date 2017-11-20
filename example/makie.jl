using Makie, GLVisualize
using GeometryTypes, Colors
using Bridge: _viridis
skippoints = 1
extra = 10

scene = Scene(resolution = (800, 800))

lines((@view Xtrue.yy[1:skippoints:end]), linewidth = 0.2, color = :darkblue)
scatter(V.yy, marker=Sphere(Point2f0(0), 1.0f0), color=:blue, markersize = 0.2,  strokewidth = 0.02, strokecolor = :white, color = map(x->RGBA(Float32.(x)..., 0.9f0), _viridis[1:2:2length(V.yy)])) 
#scatter(V.yy, marker= (Any['0':'9'...])[(1:length(V)) .% 10 + 1], color=:blue, markersize = 0.2,  strokewidth = 0.02, strokecolor = :white, color = map(x->RGBA(Float32.(x)..., 0.9f0), _viridis[1:2:2length(V.yy)])) 
for X in XXmean
    lines((@view X.yy[1:skippoints:end]), linewidth = 0.2, color = :darkgreen)
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

axis(map(x -> UnitRange(round.(Int,x)...), extrema(Bridge.mat(V.yy), 2))...)

center!(scene)


