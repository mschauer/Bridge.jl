
partial = false
include("lorenz.jl")
include("../extra/makie.jl")
using Makie, GLVisualize
using GeometryTypes, Colors
using Bridge: _viridis
skippoints = 1;
scene = Scene(resolution = (800, 800))
eyeposition = Float32[-37.2262, 65.8173, 37.4411]
sphere = Sphere(Point3f0(0,0,0), 1.0f0)
circle = Sphere(Point2f0(0,0), 1.0f0)
#RGB(Bridge._viridis[20]...)
maxviri = 200



for i in indices(Paths)[1]
    viri = _viridis[round.(Int,range(1, stop=maxviri, length=length(Paths[i])))]
    lines(Paths[i], linewidth = 0.1, color = map(x->RGBA(Float32.(x)..., 0.5f0), viri) )
end
viri = _viridis[round.(Int,range(1, stop=maxviri, length=length(Xtrue)))]
lines(( Xtrue.yy), linewidth = 0.4, color = map(x->RGBA(Float32.(x)..., 1.0f0), viri) )
lines(Xtrue.yy, linewidth = 0.2, color = :black )


#scatter([X2.yy[1].+ SVector(-0.2,-0.2,1)], marker = '↘',  markersize = 0.8, color=:black)

viri = _viridis[round.(Int,range(1, stop=maxviri, length=length(V)))]
scatter(V.yy, marker=circle, markersize = 0.6,  strokewidth = 0.02, strokecolor = :white, color = map(x->RGBA(Float32.(x)..., 0.9f0), viri)) 

ma = Float32[10,10,10] 
axis(map(x -> UnitRange(0, round.(Int,x)), ma)..., showaxis = (true,true,true), showgrid = (false,false,false))
ma2 = ma .+ 2
for (x, c) in zip(ℝ{3}[(ma2[1], 0, 0), (0, ma2[2], 0),(0, 0, ma2[3]), (0,0,0)], ['x', 'y', 'z', '•'])
    scatter([x], marker = c, markersize = 1., color = :black)
end





center!(scene)
println(Makie.getscreen(scene).cameras[:perspective].view.value)

push!(Makie.getscreen(scene).cameras[:perspective].view, 
    @SArray Float32[0.433338 0.901231 -3.7998f-7 1.95197; -0.237208 0.114057 0.96474 -20.4296; 0.869454 -0.418059 0.263205 -90.2712; 0.0 0.0 0.0 1.0]
)

Makie.save(joinpath("output", simname, "lorenz3n$simid.png"), scene);

